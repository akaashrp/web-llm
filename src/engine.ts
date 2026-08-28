import * as tvmjs from "@mlc-ai/web-runtime";
import log from "loglevel";
import {
  ChatConfig,
  ChatOptions,
  AppConfig,
  prebuiltAppConfig,
  GenerationConfig,
  postInitAndCheckGenerationConfigValues,
  Role,
  MLCEngineConfig,
  DefaultLogLevel,
  ModelType,
} from "./config";
import {
  CommittedGenerationStep,
  KVCheckpointData,
  LLMChatPipeline,
  ReplayedGenerationToken,
  SampleDecodeOptions,
  SamplePrefillOptions,
  SampledGenerationStep,
} from "./llm_chat";
import {
  // ChatCompletion
  ChatCompletionRequest,
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionMessageParam,
  ChatCompletionRequestNonStreaming,
  ChatCompletionRequestStreaming,
  ChatCompletionRequestBase,
  CompletionUsage,
  ChatCompletionMessageToolCall,
  // Completion
  CompletionCreateParamsNonStreaming,
  CompletionCreateParamsStreaming,
  CompletionCreateParamsBase,
  CompletionCreateParams,
  Completion,
  CompletionChoice,
  EmbeddingCreateParams,
  CreateEmbeddingResponse,
  Embedding,
} from "./openai_api_protocols/index";
import * as API from "./openai_api_protocols/index";
import {
  InitProgressCallback,
  MLCEngineInterface,
  LogitProcessor,
  LogLevel,
  LatencyBreakdown,
  ResumeProbeResult,
  ResumeResult,
  ResumeChatCompletionOptions,
} from "./types";
import {
  compareConversationObject,
  getConversation,
  getConversationFromChatCompletionRequest,
} from "./conversation";
import {
  cleanModelUrl,
  CustomLock,
  findModelRecord,
  getModelIdToUse,
  getToolCallFromOutputMessage,
} from "./support";
import {
  ConfigurationNotInitializedError,
  DeviceLostError,
  EmbeddingUnsupportedModelError,
  FeatureSupportError,
  MissingModelWasmError,
  ShaderF16SupportError,
  WebGPUNotAvailableError,
  ReloadArgumentSizeUnmatchedError,
  IncorrectPipelineLoadedError,
  ReloadModelIdNotUniqueError,
  SpecifiedModelNotFoundError,
  ModelNotLoadedError,
} from "./error";
import {
  asyncLoadTokenizer,
  getCacheOptions,
  getTensorCacheAccessOptions,
} from "./cache_util";
import { EmbeddingPipeline } from "./embedding";
import { verifyIntegrity } from "./integrity";
import {
  NormalizedResumableGenerationConfig,
  ResumableGenerationJournal,
  normalizeResumableGenerationConfig,
} from "./resumable/generation";
import {
  OPFSFileStore,
  createOPFSFileStore,
} from "./resumable/opfs_file_store";
import { ResumableSessionStore } from "./resumable/session_store";
import { probeResumableSession } from "./resumable/session_probe";
import {
  ResumableReplayState,
  hasUnsupportedGrammarReplay,
  readResumableReplayState,
} from "./resumable/replay";
import {
  ResumableCheckpointPayload,
  ResumableCheckpointWriter,
  readResumableCheckpointPayload,
} from "./resumable/checkpoint_writer";
import { isResumableInjectedFault } from "./resumable/fault_injection";
import { ResumableSessionHandle } from "./resumable/types";

function isOPFSUnavailableError(err: unknown): boolean {
  return (
    err instanceof Error &&
    err.message === "OPFS is unavailable in this environment"
  );
}

function getUnixTimestampSeconds(): number {
  return Math.floor(Date.now() / 1000);
}

const MIN_FREE_SPACE_BEFORE_KV_CHECKPOINT_BYTES = 512 * 1024 * 1024;
const checkpointMetadataEncoder = new TextEncoder();

function promptCheckpointId(processedSeqLen: number): string {
  return `checkpoint_00000000_${processedSeqLen.toString().padStart(8, "0")}`;
}

function requestSeed(request: unknown): number | undefined {
  const seed = (request as { seed?: unknown } | undefined)?.seed;
  return typeof seed === "number" ? seed : undefined;
}

function isChatCompletionReplayRequest(
  request: unknown,
): request is ChatCompletionRequest {
  const messages = (request as { messages?: unknown } | undefined)?.messages;
  return (
    typeof request === "object" &&
    request !== null &&
    Array.isArray(messages) &&
    messages.length > 0
  );
}

function countTrailingReplacementChar(value: string): number {
  let count = 0;
  for (let i = value.length - 1; i >= 0; i--) {
    if (value.charAt(i) !== "�") {
      return count;
    }
    count++;
  }
  return count;
}

async function estimateFreeStorageBytes(): Promise<number | undefined> {
  const storage = (
    globalThis.navigator as
      | {
          storage?: {
            estimate?: () => Promise<{ quota?: number; usage?: number }>;
          };
        }
      | undefined
  )?.storage;
  if (storage?.estimate === undefined) {
    return undefined;
  }
  try {
    const estimate = await storage.estimate();
    if (
      typeof estimate.quota !== "number" ||
      typeof estimate.usage !== "number"
    ) {
      return undefined;
    }
    return Math.max(0, estimate.quota - estimate.usage);
  } catch {
    return undefined;
  }
}

async function hasKVCheckpointQuota(
  estimatedCheckpointBytes = 0,
): Promise<boolean> {
  const freeBytes = await estimateFreeStorageBytes();
  if (freeBytes === undefined) {
    return true;
  }
  return (
    freeBytes >=
    Math.max(
      MIN_FREE_SPACE_BEFORE_KV_CHECKPOINT_BYTES,
      2 * estimatedCheckpointBytes,
    )
  );
}

interface DecodeCheckpointScheduler {
  intervalTokens: number;
  lastCheckpointSeqLen: number;
  pageSize?: number;
}

interface KVResumeCheckpoint {
  checkpoint: KVCheckpointData;
  coveredGeneratedTokens: ReplayedGenerationToken[];
  tailGeneratedTokens: ReplayedGenerationToken[];
}

interface ResumableEngineMetrics {
  journalAppendMs: number;
  checkpointWriteMs: number;
  kvRestoreMs: number;
  tokenReplayMs: number;
  resumeFirstTokenMs?: number;
}

interface StreamGenerationState {
  id: string;
  created: number;
  prevMessageLength: number;
}

interface StreamContinuationOptions {
  emitCurrent?: boolean;
  skipEmptyDelta?: boolean;
  completionTokenOffset?: number;
  beforeFinalChunk?: () => Promise<void>;
}

interface ResumeContinuationState {
  recoveryMode: "kv" | "token_replay";
  replayedTokens: number;
  extraEmittedTokens: number;
  firstTokenRecorded: boolean;
  lastCheckpointSeqLen: number;
  checkpointPageSize?: number;
  pendingJournaledToken?: JournaledGenerationStep;
}

interface JournaledGenerationStep {
  sampled: SampledGenerationStep;
  committed: CommittedGenerationStep;
}

function metadataInteger(
  metadata: Record<string, unknown>,
  field: string,
): number | undefined {
  const value = metadata[field];
  return typeof value === "number" && Number.isInteger(value) && value > 0
    ? value
    : undefined;
}

function checkpointPageSize(checkpoint: KVCheckpointData): number | undefined {
  return (
    metadataInteger(checkpoint.metadata, "pageSize") ??
    metadataInteger(checkpoint.metadata, "page_size")
  );
}

function checkpointByteLength(checkpoint: KVCheckpointData): number {
  const pageGroupBytes = checkpoint.pageGroups.reduce(
    (sum, group) => sum + group.data.byteLength,
    0,
  );
  const logitsBytes = checkpoint.nextLogits?.data.byteLength ?? 0;
  const metadataBytes = checkpointMetadataEncoder.encode(
    JSON.stringify(checkpoint.metadata),
  ).byteLength;
  return pageGroupBytes + logitsBytes + metadataBytes;
}

function normalizeCheckpointInterval(
  intervalTokens: number,
  pageSize?: number,
): number {
  if (pageSize === undefined) {
    return intervalTokens;
  }
  return Math.ceil(intervalTokens / pageSize) * pageSize;
}

function shouldCaptureDecodeCheckpoint(
  scheduler: DecodeCheckpointScheduler,
  nextProcessedSeqLen: number,
): boolean {
  const intervalTokens = normalizeCheckpointInterval(
    scheduler.intervalTokens,
    scheduler.pageSize,
  );
  if (nextProcessedSeqLen - scheduler.lastCheckpointSeqLen < intervalTokens) {
    return false;
  }
  return (
    scheduler.pageSize === undefined ||
    nextProcessedSeqLen % scheduler.pageSize === 0
  );
}

function noteCommittedCheckpoint(
  scheduler: DecodeCheckpointScheduler,
  checkpoint: KVCheckpointData,
): void {
  scheduler.lastCheckpointSeqLen = checkpoint.processedSeqLen;
  scheduler.pageSize = checkpointPageSize(checkpoint) ?? scheduler.pageSize;
}

function replayPromptSeqLen(state: ResumableReplayState): number {
  if (state.emittedTokens === 0) {
    return state.promptTokenIds.length;
  }
  return Math.max(0, state.processedSeqLen - state.emittedTokens);
}

/**
 * Creates `MLCEngine`, and loads `modelId` onto WebGPU.
 *
 * Equivalent to `new webllm.MLCEngine().reload(...)`.
 *
 * @param modelId model_id of the model to load, either string or string[]. When multiple models
 *   are provided, we load all models sequentially. Each modelId needs to either be in
 *   `webllm.prebuiltAppConfig`, or in `engineCOnfig.appConfig`.
 * @param engineConfig Optionally configures the engine, see `webllm.MLCEngineConfig`.
 * @param chatOpts Extra options to optionally override the `mlc-chat-config.json` of `modelId`.
 *   The size of which needs to match that of `modelId`; chatOpts[i] will be used for modelId[i].
 * @returns An initialized `WebLLM.MLCEngine` with `modelId` loaded.
 * @throws Throws error when device lost (mostly due to OOM); users should re-call `CreateMLCEngine()`,
 *   potentially with a smaller model or smaller context window size.
 */
export async function CreateMLCEngine(
  modelId: string | string[],
  engineConfig?: MLCEngineConfig,
  chatOpts?: ChatOptions | ChatOptions[],
): Promise<MLCEngine> {
  const engine = new MLCEngine(engineConfig);
  await engine.reload(modelId, chatOpts);
  return engine;
}

/**
 * The main interface of MLCEngine, which loads a model and performs tasks.
 *
 * You can either initialize one with `webllm.CreateMLCEngine(modelId)`, or
 * `webllm.MLCEngine().reload(modelId)`.
 */
export class MLCEngine implements MLCEngineInterface {
  // APIs
  /** For chat.completions.create() */
  public chat: API.Chat;
  /** For completions.create() */
  public completions: API.Completions;
  /** For embeddings.create() */
  public embeddings: API.Embeddings;

  // Maps to maintain states of loaded model(s)
  /** Maps each loaded model's modelId to its pipeline */
  private loadedModelIdToPipeline: Map<
    string,
    LLMChatPipeline | EmbeddingPipeline
  >;
  /** Maps each loaded model's modelId to its chatConfig */
  private loadedModelIdToChatConfig: Map<string, ChatConfig>;
  /** Maps each loaded model's modelId to its modelType */
  private loadedModelIdToModelType: Map<string, ModelType>;
  /** Maps each loaded model's modelId to a lock. Ensures
   * each model only processes one request at at time.
   */
  private loadedModelIdToLock: Map<string, CustomLock>;

  // Others
  private logger: (msg: string) => void = log.info;
  private logitProcessorRegistry?: Map<string, LogitProcessor>;
  private initProgressCallback?: InitProgressCallback;
  private appConfig: AppConfig;
  private resumableFileStore?: OPFSFileStore;
  private resumableSessionStore?: ResumableSessionStore;
  private lastResumableMetrics?: ResumableEngineMetrics;

  // Signals and flags
  private interruptSignal = false;
  private deviceLostIsError = true; // whether device.lost is due to actual error or model reload
  private reloadController: AbortController | undefined;

  constructor(engineConfig?: MLCEngineConfig) {
    this.loadedModelIdToPipeline = new Map<
      string,
      LLMChatPipeline | EmbeddingPipeline
    >();
    this.loadedModelIdToChatConfig = new Map<string, ChatConfig>();
    this.loadedModelIdToModelType = new Map<string, ModelType>();
    this.loadedModelIdToLock = new Map<string, CustomLock>();
    this.appConfig = engineConfig?.appConfig || prebuiltAppConfig;
    this.setLogLevel(engineConfig?.logLevel || DefaultLogLevel);
    this.setInitProgressCallback(engineConfig?.initProgressCallback);
    this.setLogitProcessorRegistry(engineConfig?.logitProcessorRegistry);

    this.chat = new API.Chat(this);
    this.completions = new API.Completions(this);
    this.embeddings = new API.Embeddings(this);
  }

  //-----------------------
  // 0. Setters and getters
  //-----------------------

  setAppConfig(appConfig: AppConfig) {
    this.appConfig = appConfig;
  }

  setInitProgressCallback(initProgressCallback?: InitProgressCallback) {
    this.initProgressCallback = initProgressCallback;
  }

  getInitProgressCallback() {
    return this.initProgressCallback;
  }

  setLogitProcessorRegistry(
    logitProcessorRegistry?: Map<string, LogitProcessor>,
  ) {
    this.logitProcessorRegistry = logitProcessorRegistry;
  }

  private getResumableFileStore(): OPFSFileStore {
    if (this.resumableFileStore === undefined) {
      this.resumableFileStore = createOPFSFileStore();
    }
    return this.resumableFileStore;
  }

  private getResumableSessionStore(): ResumableSessionStore {
    if (this.resumableSessionStore === undefined) {
      this.resumableSessionStore = new ResumableSessionStore(
        this.getResumableFileStore(),
      );
    }
    return this.resumableSessionStore;
  }

  private resetResumableMetrics(): ResumableEngineMetrics {
    const metrics = {
      journalAppendMs: 0,
      checkpointWriteMs: 0,
      kvRestoreMs: 0,
      tokenReplayMs: 0,
    };
    this.lastResumableMetrics = metrics;
    return metrics;
  }

  private resumableMetrics(): ResumableEngineMetrics {
    return this.lastResumableMetrics ?? this.resetResumableMetrics();
  }

  private normalizeResumableForRequest(
    request: ChatCompletionRequest,
  ): NormalizedResumableGenerationConfig | undefined {
    const config = normalizeResumableGenerationConfig(
      request.extra_body?.resumable,
    );
    if (config === undefined) {
      return undefined;
    }
    if ((request.n ?? 1) > 1) {
      throw new Error("Resumable generation currently requires n <= 1.");
    }
    return config;
  }

  private tryCreateResumableJournal(
    config: NormalizedResumableGenerationConfig | undefined,
  ): ResumableGenerationJournal | undefined {
    if (config === undefined) {
      return undefined;
    }
    try {
      const files = this.getResumableFileStore();
      return new ResumableGenerationJournal(
        files,
        this.getResumableSessionStore(),
        config,
      );
    } catch (err) {
      if (config.strictPersistence) {
        throw err;
      }
      log.warn("Resumable journal disabled:", err);
      return undefined;
    }
  }

  private async recordGeneratedToken(
    journal: ResumableGenerationJournal | undefined,
    pipeline: LLMChatPipeline,
    sampled: SampledGenerationStep,
    committed: CommittedGenerationStep,
  ): Promise<void> {
    if (journal === undefined) {
      return;
    }
    const start = performance.now();
    try {
      await journal.recordGeneratedToken({
        globalTokenPos: sampled.globalTokenPos,
        tokenId: sampled.tokenId,
        textDelta: committed.textDelta,
        textPrefixLength: committed.textPrefixLength,
        rngState: pipeline.getRNGState(),
        logprob: sampled.logprob,
      });
    } finally {
      this.resumableMetrics().journalAppendMs += performance.now() - start;
    }
  }

  private async writeKVCheckpoint(
    journal: ResumableGenerationJournal,
    checkpoint: KVCheckpointData | undefined,
  ): Promise<boolean> {
    if (checkpoint === undefined) {
      return false;
    }
    if (!(await hasKVCheckpointQuota(checkpointByteLength(checkpoint)))) {
      return false;
    }
    try {
      const writer = new ResumableCheckpointWriter(
        this.getResumableFileStore(),
        this.getResumableSessionStore(),
      );
      const start = performance.now();
      try {
        await writer.writeCheckpoint({
          sessionId: journal.sessionId,
          checkpointId: promptCheckpointId(checkpoint.processedSeqLen),
          processedSeqLen: checkpoint.processedSeqLen,
          layoutHash: checkpoint.layoutHash,
          metadata: {
            ...checkpoint.metadata,
            nextLogitsShape: checkpoint.nextLogits?.shape,
            nextLogitsDtype: checkpoint.nextLogits?.dtype,
          },
          pageGroups: checkpoint.pageGroups,
          nextLogits: checkpoint.nextLogits?.data,
        });
      } finally {
        this.resumableMetrics().checkpointWriteMs += performance.now() - start;
      }
      await journal.refreshSeqNo();
      return true;
    } catch (err) {
      if (journal.strictPersistence) {
        throw err;
      }
      log.warn("KV checkpoint disabled:", err);
      return false;
    }
  }

  private async cleanupKVAfterFinish(
    journal: ResumableGenerationJournal,
  ): Promise<void> {
    try {
      await this.getResumableSessionStore().deleteKV(journal.sessionId);
    } catch (err) {
      if (journal.strictPersistence) {
        throw err;
      }
      log.warn("KV checkpoint cleanup failed:", err);
    }
  }

  private async writePromptCheckpoint(
    journal: ResumableGenerationJournal,
    sampled: SampledGenerationStep,
  ): Promise<boolean> {
    return this.writeKVCheckpoint(journal, sampled.promptCheckpoint);
  }

  private async writeDecodeCheckpoint(
    journal: ResumableGenerationJournal,
    scheduler: DecodeCheckpointScheduler,
    sampled: SampledGenerationStep,
  ): Promise<void> {
    if (await this.writeKVCheckpoint(journal, sampled.decodeCheckpoint)) {
      noteCommittedCheckpoint(scheduler, sampled.decodeCheckpoint!);
    }
  }

  private async createResumeContinuationJournal(
    files: OPFSFileStore,
    sessions: ResumableSessionStore,
    session: ResumableSessionHandle,
    state: ResumableReplayState,
  ): Promise<ResumableGenerationJournal> {
    if (state.resumableConfig === undefined) {
      throw new Error(
        `Resumable session ${state.sessionId} is missing journal configuration.`,
      );
    }
    const journal = new ResumableGenerationJournal(
      files,
      sessions,
      state.resumableConfig,
    );
    await journal.attachExistingSession(session, state.emittedTokens);
    return journal;
  }

  private createResumeCheckpointScheduler(
    journal: ResumableGenerationJournal,
    restored: ResumeContinuationState,
  ): DecodeCheckpointScheduler {
    return {
      intervalTokens: journal.checkpointIntervalTokens,
      lastCheckpointSeqLen: restored.lastCheckpointSeqLen,
      pageSize: restored.checkpointPageSize,
    };
  }

  private async recordPendingResumeToken(
    journal: ResumableGenerationJournal,
    pipeline: LLMChatPipeline,
    restored: ResumeContinuationState,
  ): Promise<void> {
    if (restored.pendingJournaledToken === undefined) {
      return;
    }
    await this.recordGeneratedToken(
      journal,
      pipeline,
      restored.pendingJournaledToken.sampled,
      restored.pendingJournaledToken.committed,
    );
    restored.pendingJournaledToken = undefined;
  }

  private async decodeResumedGenerationStep(
    journal: ResumableGenerationJournal,
    scheduler: DecodeCheckpointScheduler,
    promptSeqLen: number,
    pipeline: LLMChatPipeline,
    genConfig: GenerationConfig,
  ): Promise<void> {
    const captureDecodeCheckpoint =
      shouldCaptureDecodeCheckpoint(
        scheduler,
        promptSeqLen + journal.emittedTokenCount,
      ) && (await hasKVCheckpointQuota());
    const decodeStep = await this.sampleDecode(pipeline, genConfig, {
      captureCheckpoint: captureDecodeCheckpoint,
      storeCheckpointLogits: journal.storeCheckpointLogits,
    });
    const committedDecode = pipeline.commitSampledStep(decodeStep, genConfig);
    await this.recordGeneratedToken(
      journal,
      pipeline,
      decodeStep,
      committedDecode,
    );
    await this.writeDecodeCheckpoint(journal, scheduler, decodeStep);
  }

  private checkpointPayloadToKVData(
    payload: ResumableCheckpointPayload,
  ): KVCheckpointData {
    const metadata = payload.meta.metadata;
    if (metadata === undefined) {
      throw new Error("KV checkpoint metadata is missing runtime metadata.");
    }
    const nextLogitsShape = metadata.nextLogitsShape;
    const nextLogitsDtype = metadata.nextLogitsDtype;
    return {
      processedSeqLen: payload.meta.processedSeqLen,
      layoutHash: payload.meta.layoutHash,
      metadata,
      pageGroups: payload.pageGroups.map((group) => ({
        groupId: group.groupId,
        layerStart: group.layerStart,
        layerEnd: group.layerEnd,
        data: group.data,
      })),
      nextLogits:
        payload.nextLogits === undefined
          ? undefined
          : {
              shape: Array.isArray(nextLogitsShape)
                ? (nextLogitsShape as number[])
                : [],
              dtype:
                typeof nextLogitsDtype === "string"
                  ? nextLogitsDtype
                  : "float32",
              data: payload.nextLogits.data,
            },
    };
  }

  private async readBestKVCheckpoint(
    files: OPFSFileStore,
    sessions: ResumableSessionStore,
    session: ResumableSessionHandle,
    state: ResumableReplayState,
  ): Promise<KVResumeCheckpoint | undefined> {
    const promptSeqLen = state.promptTokenIds.length;
    const refs = await sessions.listCommittedCheckpoints(session.sessionId);
    let best: KVCheckpointData | undefined;
    for (const ref of refs) {
      const payload = await readResumableCheckpointPayload(files, ref);
      if (
        payload !== undefined &&
        payload.meta.processedSeqLen >= promptSeqLen &&
        payload.meta.processedSeqLen <= state.processedSeqLen &&
        (best === undefined ||
          payload.meta.processedSeqLen > best.processedSeqLen)
      ) {
        best = this.checkpointPayloadToKVData(payload);
      }
    }
    if (best === undefined) {
      return undefined;
    }
    return {
      checkpoint: best,
      coveredGeneratedTokens: state.generatedTokens.filter(
        (token) => token.globalTokenPos < best.processedSeqLen,
      ),
      tailGeneratedTokens: state.generatedTokens.filter(
        (token) => token.globalTokenPos >= best.processedSeqLen,
      ),
    };
  }

  private async tryRestoreKVResumeState(
    files: OPFSFileStore,
    sessions: ResumableSessionStore,
    session: ResumableSessionHandle,
    state: ResumableReplayState,
    pipeline: LLMChatPipeline,
    genConfig: GenerationConfig,
  ): Promise<ResumeContinuationState | undefined> {
    if (
      state.finished ||
      state.scanStoppedReason !== undefined ||
      state.promptTokenIds.length === 0 ||
      state.generationConfig === undefined ||
      hasUnsupportedGrammarReplay(state.generationConfig) ||
      this.logitProcessorRegistry?.has(state.modelId)
    ) {
      return undefined;
    }
    if (
      state.generatedTokens.length > 0 &&
      state.generatedTokens[state.generatedTokens.length - 1].rngState ===
        undefined
    ) {
      return undefined;
    }

    const resume = await this.readBestKVCheckpoint(
      files,
      sessions,
      session,
      state,
    );
    if (resume === undefined) {
      return undefined;
    }
    try {
      if (resume.tailGeneratedTokens.length === 0) {
        const seed = requestSeed(state.request);
        const rngState = resume.coveredGeneratedTokens.at(-1)?.rngState;
        if (rngState !== undefined) {
          if (!pipeline.setRNGState(rngState)) {
            return undefined;
          }
        } else if (seed !== undefined) {
          pipeline.setSeed(seed);
        } else if (state.generatedTokens.length > 0) {
          return undefined;
        }
      }
      const restoreStart = performance.now();
      const replay = await pipeline.replayFromPromptCheckpoint(
        resume.checkpoint,
        state.assistantPrefixTokenIds,
        resume.coveredGeneratedTokens,
        resume.tailGeneratedTokens,
        genConfig,
      );
      const restoreElapsed = performance.now() - restoreStart;
      this.resumableMetrics().kvRestoreMs += restoreElapsed;
      if (replay.sampledFromCheckpointLogits) {
        this.resumableMetrics().resumeFirstTokenMs = restoreElapsed;
      }
      const rngState = resume.tailGeneratedTokens.at(-1)?.rngState;
      if (rngState !== undefined && !pipeline.setRNGState(rngState)) {
        pipeline.resetChat();
        return undefined;
      }

      return {
        recoveryMode: "kv",
        replayedTokens: replay.replayedTokens,
        extraEmittedTokens: replay.sampledFromCheckpointLogits ? 1 : 0,
        firstTokenRecorded: replay.sampledFromCheckpointLogits,
        lastCheckpointSeqLen: resume.checkpoint.processedSeqLen,
        checkpointPageSize: checkpointPageSize(resume.checkpoint),
        pendingJournaledToken:
          replay.sampledToken !== undefined &&
          replay.committedToken !== undefined
            ? {
                sampled: replay.sampledToken,
                committed: replay.committedToken,
              }
            : undefined,
      };
    } catch (err) {
      log.warn("KV checkpoint restore failed; falling back:", err);
      pipeline.resetChat();
      return undefined;
    }
  }

  private finalizeResumedConversation(
    state: ResumableReplayState,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
  ): void {
    if (!isChatCompletionReplayRequest(state.request)) {
      return;
    }
    const conversation = getConversationFromChatCompletionRequest(
      state.request,
      chatConfig,
      true,
    );
    conversation.appendReplyHeader(Role.assistant);
    conversation.finishReply(pipeline.getMessage());
    pipeline.setConversation(conversation);
  }

  private async finishResumeContinuation(
    state: ResumableReplayState,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
    genConfig: GenerationConfig,
    restored: ResumeContinuationState,
    journal: ResumableGenerationJournal,
    checkpointScheduler: DecodeCheckpointScheduler,
    promptSeqLen: number,
  ): Promise<ResumeResult> {
    this.interruptSignal = false;
    const firstTokenStart = performance.now();
    let recordedFirstToken = restored.firstTokenRecorded;
    try {
      await this.recordPendingResumeToken(journal, pipeline, restored);
      while (!pipeline.stopped()) {
        if (this.interruptSignal) {
          pipeline.triggerStop();
          break;
        }
        await this.decodeResumedGenerationStep(
          journal,
          checkpointScheduler,
          promptSeqLen,
          pipeline,
          genConfig,
        );
        if (!recordedFirstToken) {
          this.resumableMetrics().resumeFirstTokenMs =
            performance.now() - firstTokenStart;
          recordedFirstToken = true;
        }
      }
      await journal.recordGenerationEnd({
        finishReason: pipeline.getFinishReason(),
        emittedTokens: journal.emittedTokenCount,
      });
      if (pipeline.getFinishReason() !== "abort") {
        await this.cleanupKVAfterFinish(journal);
      }
      this.finalizeResumedConversation(state, pipeline, chatConfig);
      return {
        sessionId: state.sessionId,
        recoveredText: pipeline.getMessage(),
        emittedTokens: journal.emittedTokenCount,
        processedSeqLen: promptSeqLen + journal.emittedTokenCount,
        replayedTokens: restored.replayedTokens,
        recoveryMode: restored.recoveryMode,
      };
    } catch (err) {
      if (!isResumableInjectedFault(err)) {
        await journal.recordEngineError({ err }).catch(() => undefined);
      }
      throw err;
    } finally {
      await journal.close();
    }
  }

  private async *streamResumeContinuation(
    request: ChatCompletionRequestStreaming,
    model: string,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
    genConfig: GenerationConfig,
    state: ResumableReplayState,
    restored: ResumeContinuationState,
    journal: ResumableGenerationJournal,
    checkpointScheduler: DecodeCheckpointScheduler,
    promptSeqLen: number,
    release: () => Promise<void>,
  ): AsyncGenerator<ChatCompletionChunk, void, void> {
    this.interruptSignal = false;
    const firstTokenStart = performance.now();
    let recordedFirstToken = restored.firstTokenRecorded;
    const streamState: StreamGenerationState = {
      id: crypto.randomUUID(),
      created: getUnixTimestampSeconds(),
      prevMessageLength: pipeline.getMessage().length,
    };

    try {
      await this.recordPendingResumeToken(journal, pipeline, restored);
      const chunks = this.streamCurrentGeneration(
        request,
        model,
        pipeline,
        genConfig,
        Date.now(),
        streamState,
        async () => {
          await this.decodeResumedGenerationStep(
            journal,
            checkpointScheduler,
            promptSeqLen,
            pipeline,
            genConfig,
          );
          if (!recordedFirstToken) {
            this.resumableMetrics().resumeFirstTokenMs =
              performance.now() - firstTokenStart;
            recordedFirstToken = true;
          }
        },
        {
          emitCurrent: restored.extraEmittedTokens > 0,
          skipEmptyDelta: true,
          completionTokenOffset: restored.extraEmittedTokens,
          beforeFinalChunk: async () => {
            await journal.recordGenerationEnd({
              finishReason: pipeline.getFinishReason(),
              emittedTokens: journal.emittedTokenCount,
            });
            if (pipeline.getFinishReason() !== "abort") {
              await this.cleanupKVAfterFinish(journal);
            }
            this.finalizeResumedConversation(state, pipeline, chatConfig);
          },
        },
      );
      for await (const chunk of chunks) {
        yield chunk as ChatCompletionChunk;
      }
    } catch (err) {
      if (!isResumableInjectedFault(err)) {
        await journal.recordEngineError({ err }).catch(() => undefined);
      }
      throw err;
    } finally {
      try {
        await journal.close();
      } finally {
        await release();
      }
    }
  }

  private makeTextOnlyResumeResult(state: ResumableReplayState): ResumeResult {
    return {
      sessionId: state.sessionId,
      recoveredText: state.recoveredText,
      emittedTokens: state.emittedTokens,
      processedSeqLen: state.processedSeqLen,
      replayedTokens: 0,
      recoveryMode: "text_only",
    };
  }

  private getTokenReplayBlockReason(
    state: ResumableReplayState,
  ): string | undefined {
    if (state.finished) {
      return "generation already finished";
    }
    if (state.scanStoppedReason !== undefined) {
      return `journal scan stopped at ${state.scanStoppedReason}`;
    }
    if (state.promptTokenIds.length === 0) {
      return "missing prompt token record";
    }
    if (state.generatedTokens.length === 0) {
      return "missing generated token records";
    }
    if (state.generationConfig === undefined) {
      return "missing generation config record";
    }
    if (hasUnsupportedGrammarReplay(state.generationConfig)) {
      return "unsupported grammar replay";
    }
    if (
      state.generatedTokens[state.generatedTokens.length - 1].rngState ===
      undefined
    ) {
      return "missing RNG state";
    }
    if (state.modelId === "") {
      return "missing model id";
    }
    if (this.logitProcessorRegistry?.has(state.modelId)) {
      return "custom LogitProcessor replay unsupported";
    }
    return undefined;
  }

  /**
   * Set MLCEngine logging output level
   *
   * @param logLevel The new log level
   */
  setLogLevel(logLevel: LogLevel) {
    log.setLevel(logLevel);
  }

  //----------------------------------------
  // 1. Model/pipeline loading and unloading
  //----------------------------------------

  async reload(
    modelId: string | string[],
    chatOpts?: ChatOptions | ChatOptions[],
  ): Promise<void> {
    // 0. Unload all loaded models
    await this.unload();
    // 1. Convert inputs to arrays
    if (!Array.isArray(modelId)) {
      modelId = [modelId];
    }
    if (chatOpts !== undefined && !Array.isArray(chatOpts)) {
      chatOpts = [chatOpts];
    }
    // 2. Check whether size matches
    if (chatOpts !== undefined && modelId.length !== chatOpts.length) {
      throw new ReloadArgumentSizeUnmatchedError(
        modelId.length,
        chatOpts.length,
      );
    }
    // 3. Make sure each model in modelId is unique
    if (new Set(modelId).size < modelId.length) {
      throw new ReloadModelIdNotUniqueError(modelId);
    }
    // 4. Sequentially load each model
    // Single abort should stop all to-be-loaded models
    this.reloadController = new AbortController();
    try {
      for (let i = 0; i < modelId.length; i++) {
        await this.reloadInternal(
          modelId[i],
          chatOpts ? chatOpts[i] : undefined,
        );
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        log.warn("Reload() is aborted.", error.message);
        return;
      }
      throw error;
    } finally {
      this.reloadController = undefined;
    }
  }

  private async reloadInternal(
    modelId: string,
    chatOpts?: ChatOptions,
  ): Promise<void> {
    const logitProcessor = this.logitProcessorRegistry?.get(modelId);
    const tstart = performance.now();

    // look up and parse model record, record model type
    const modelRecord = findModelRecord(modelId, this.appConfig);
    const baseUrl =
      typeof document !== "undefined"
        ? document.URL
        : globalThis.location.origin;
    let modelUrl = cleanModelUrl(modelRecord.model);
    if (!modelUrl.startsWith("http")) {
      modelUrl = new URL(modelUrl, baseUrl).href;
    }
    const modelType =
      modelRecord.model_type === undefined || modelRecord.model_type === null
        ? ModelType.LLM
        : modelRecord.model_type;
    this.loadedModelIdToModelType.set(modelId, modelType);

    // instantiate cache
    const configCache = tvmjs.createArtifactCache(
      "webllm/config",
      getCacheOptions(this.appConfig),
    );

    // load config
    const configUrl = new URL("mlc-chat-config.json", modelUrl).href;
    const configData = (await configCache.fetchWithCache(
      configUrl,
      "arraybuffer",
      this.reloadController?.signal,
    )) as ArrayBuffer;
    if (modelRecord.integrity?.config) {
      await verifyIntegrity(
        configData,
        modelRecord.integrity.config,
        configUrl,
        modelRecord.integrity.onFailure,
      );
    }
    const curModelConfig: ChatConfig = {
      ...JSON.parse(new TextDecoder().decode(configData)),
      ...modelRecord.overrides,
      ...chatOpts,
    } as ChatConfig;
    this.loadedModelIdToChatConfig.set(modelId, curModelConfig);

    // load tvm wasm
    const wasmCache = tvmjs.createArtifactCache(
      "webllm/wasm",
      getCacheOptions(this.appConfig),
    );

    const wasmUrl = modelRecord.model_lib;
    if (wasmUrl === undefined) {
      throw new MissingModelWasmError(modelRecord.model_id);
    }
    const fetchWasmSource = async () => {
      if (wasmUrl.includes("localhost")) {
        // do not cache wasm on local host as we might update code frequently
        return (await fetch(wasmUrl)).arrayBuffer();
      } else if (!wasmUrl.startsWith("http")) {
        // do not cache wasm on the same server as it can also refresh
        // rely on the normal caching strategy
        return (await fetch(new URL(wasmUrl, baseUrl).href)).arrayBuffer();
      } else {
        return await wasmCache.fetchWithCache(
          wasmUrl,
          "arraybuffer",
          this.reloadController?.signal,
        );
      }
    };
    const wasmSource = await fetchWasmSource();

    if (modelRecord.integrity?.model_lib) {
      await verifyIntegrity(
        wasmSource,
        modelRecord.integrity.model_lib,
        wasmUrl,
        modelRecord.integrity.onFailure,
      );
    }

    const wasm = new Uint8Array(wasmSource);
    const tvm = await tvmjs.instantiate(
      wasm.buffer,
      tvmjs.createPolyfillWASI(),
      this.logger,
    );

    if (this.initProgressCallback !== undefined) {
      tvm.registerInitProgressCallback(this.initProgressCallback);
    }

    // detect GPU
    const gpuDetectOutput = await tvmjs.detectGPUDevice();
    if (gpuDetectOutput == undefined) {
      throw new WebGPUNotAvailableError();
    }
    let gpuLabel = "WebGPU";
    if (gpuDetectOutput.adapterInfo.description.length != 0) {
      gpuLabel += " - " + gpuDetectOutput.adapterInfo.description;
    } else {
      gpuLabel += " - " + gpuDetectOutput.adapterInfo.vendor;
    }
    if (modelRecord.required_features !== undefined) {
      for (const feature of modelRecord.required_features) {
        if (!gpuDetectOutput.device.features.has(feature)) {
          if (feature == "shader-f16") {
            throw new ShaderF16SupportError();
          }
          throw new FeatureSupportError(feature);
        }
      }
    }

    // Most device lost happens in `reload()` since we allocate memory ahead of time. So we can
    // use this flag at the end of `reload()` to make the error handling synchronous.
    // This `.then()` exists throughout the lifetime of the device. Though we have not
    // experienced device error outside of `reload()`, it is still possible this `.then()` is
    // triggered outside of `reload()`. TODO: does this cause unexpected behavior?
    let deviceLostInReload = false;
    gpuDetectOutput.device.lost.then((info: any) => {
      if (this.deviceLostIsError) {
        log.error(
          `Device was lost. This can happen due to insufficient memory or other GPU constraints. ` +
            `Detailed error: ${info}. Please try to reload WebLLM with a less resource-intensive model.`,
        );
        this.unload();
        deviceLostInReload = true;
      }
    });
    tvm.initWebGPU(gpuDetectOutput.device);

    const tokenizer = await asyncLoadTokenizer(
      modelUrl,
      curModelConfig,
      this.appConfig,
      this.logger,
      modelRecord.integrity,
    );
    await tvm.fetchTensorCache(modelUrl, tvm.webgpu(), {
      ...getTensorCacheAccessOptions("webllm/model", this.appConfig),
      signal: this.reloadController?.signal,
    });

    // Instantiate pipeline
    // TODO: would be good to somehow check for error when LLMChatPipeline is loaded for an
    // embedding model, and prompt user to use ModelRecord.model_type
    let newPipeline: LLMChatPipeline | EmbeddingPipeline;
    if (modelRecord.model_type === ModelType.embedding) {
      newPipeline = new EmbeddingPipeline(tvm, tokenizer, curModelConfig);
    } else {
      newPipeline = new LLMChatPipeline(
        tvm,
        tokenizer,
        curModelConfig,
        logitProcessor,
      );
    }
    await newPipeline.asyncLoadWebGPUPipelines();
    this.loadedModelIdToPipeline.set(modelId, newPipeline);
    this.loadedModelIdToLock.set(modelId, new CustomLock());

    // Clean up
    const tend = performance.now();
    if (this.initProgressCallback !== undefined) {
      const text = "Finish loading on " + gpuLabel;
      this.initProgressCallback({
        progress: 1,
        timeElapsed: (tend - tstart) / 1e3,
        text: text,
      });
    }
    if (deviceLostInReload) {
      throw new DeviceLostError();
    }
  }

  async unload() {
    this.deviceLostIsError = false; // so that unload() does not trigger device.lost error
    // TODO: can optimize by calling dispose() to all pipelines in parallel. However, need to wait
    // for all sync() to finish before proceeding (e.g. naive forEach does not work)
    for (const entry of Array.from(this.loadedModelIdToPipeline.entries())) {
      const pipeline = entry[1];
      pipeline.dispose();
      // Wait until device is actually destroyed so we can safely set deviceLostIsError back to true
      await pipeline.sync();
    }
    this.loadedModelIdToPipeline.clear();
    this.loadedModelIdToChatConfig.clear();
    this.loadedModelIdToModelType.clear();
    this.loadedModelIdToLock.clear();
    this.deviceLostIsError = true;
    if (this.reloadController) {
      this.reloadController.abort("Engine.unload() is called.");
      this.reloadController = undefined;
    }
  }

  //---------------------------------------------------
  // 2. Underlying auto-regressive generation functions
  //---------------------------------------------------

  private async _generate(
    input:
      | ChatCompletionRequestNonStreaming
      | CompletionCreateParamsNonStreaming,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
    genConfig: GenerationConfig,
  ): Promise<string> {
    this.interruptSignal = false;
    if (genConfig !== undefined) {
      postInitAndCheckGenerationConfigValues(genConfig);
    }
    await this.prefill(input, pipeline, chatConfig, genConfig);

    while (!pipeline.stopped()) {
      if (this.interruptSignal) {
        pipeline.triggerStop();
        break;
      }
      await this.decode(pipeline, genConfig);
    }
    return pipeline.getMessage();
  }

  private async _generateResumable(
    input:
      | ChatCompletionRequestNonStreaming
      | CompletionCreateParamsNonStreaming,
    modelId: string,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
    genConfig: GenerationConfig,
    journal: ResumableGenerationJournal,
  ): Promise<string> {
    this.resetResumableMetrics();
    this.interruptSignal = false;
    if (genConfig !== undefined) {
      postInitAndCheckGenerationConfigValues(genConfig);
    }

    let journalStarted = false;
    try {
      const prefillStep = await this.samplePrefill(
        input,
        pipeline,
        chatConfig,
        genConfig,
        {
          capturePromptCheckpoint:
            journal.checkpointPrompt && (await hasKVCheckpointQuota()),
          storeCheckpointLogits: journal.storeCheckpointLogits,
        },
      );
      if (prefillStep.promptTokenIds === undefined) {
        throw new Error(
          "Resumable generation currently supports text-only prompts.",
        );
      }
      await journal.begin({
        modelId,
        request: input,
        promptTokenIds: prefillStep.promptTokenIds,
        assistantPrefixTokenIds: prefillStep.assistantPrefixTokenIds ?? [],
        generationConfig: genConfig,
      });
      journalStarted = true;
      const checkpointScheduler: DecodeCheckpointScheduler = {
        intervalTokens: journal.checkpointIntervalTokens,
        lastCheckpointSeqLen: prefillStep.globalTokenPos,
      };
      if (await this.writePromptCheckpoint(journal, prefillStep)) {
        noteCommittedCheckpoint(
          checkpointScheduler,
          prefillStep.promptCheckpoint!,
        );
      }
      const committedPrefill = pipeline.commitSampledStep(
        prefillStep,
        genConfig,
      );
      await this.recordGeneratedToken(
        journal,
        pipeline,
        prefillStep,
        committedPrefill,
      );

      while (!pipeline.stopped()) {
        if (this.interruptSignal) {
          pipeline.triggerStop();
          break;
        }
        const captureDecodeCheckpoint =
          shouldCaptureDecodeCheckpoint(
            checkpointScheduler,
            prefillStep.globalTokenPos + journal.emittedTokenCount,
          ) && (await hasKVCheckpointQuota());
        const decodeStep = await this.sampleDecode(pipeline, genConfig, {
          captureCheckpoint: captureDecodeCheckpoint,
          storeCheckpointLogits: journal.storeCheckpointLogits,
        });
        const committedDecode = pipeline.commitSampledStep(
          decodeStep,
          genConfig,
        );
        await this.recordGeneratedToken(
          journal,
          pipeline,
          decodeStep,
          committedDecode,
        );
        await this.writeDecodeCheckpoint(
          journal,
          checkpointScheduler,
          decodeStep,
        );
      }
      await journal.recordGenerationEnd({
        finishReason: pipeline.getFinishReason(),
        emittedTokens: journal.emittedTokenCount,
      });
      if (pipeline.getFinishReason() !== "abort") {
        await this.cleanupKVAfterFinish(journal);
      }
      return pipeline.getMessage();
    } catch (err) {
      if (journalStarted && !isResumableInjectedFault(err)) {
        await journal.recordEngineError({ err }).catch(() => undefined);
      }
      throw err;
    } finally {
      await journal.close();
    }
  }

  private makeStreamDeltaChunk(
    request: ChatCompletionRequestStreaming | CompletionCreateParamsStreaming,
    model: string,
    pipeline: LLMChatPipeline,
    state: StreamGenerationState,
    skipEmptyDelta = false,
  ): ChatCompletionChunk | Completion | undefined {
    const curMessage = pipeline.getMessage();
    if (countTrailingReplacementChar(curMessage) % 4 !== 0) {
      return undefined;
    }

    const deltaMessage = curMessage.slice(state.prevMessageLength);
    state.prevMessageLength = curMessage.length;
    if (skipEmptyDelta && deltaMessage === "") {
      return undefined;
    }

    const logprobs = request.logprobs
      ? ({
          content: pipeline.getTokenLogprobArray().slice(-1),
        } as ChatCompletionChunk.Choice.Logprobs)
      : null;
    if ("messages" in request) {
      return {
        id: state.id,
        choices: [
          {
            delta: { content: deltaMessage, role: "assistant" },
            finish_reason: null,
            index: 0,
            logprobs,
          },
        ],
        model,
        object: "chat.completion.chunk",
        created: state.created,
      };
    }
    return {
      id: state.id,
      choices: [
        {
          text: deltaMessage,
          finish_reason: null,
          index: 0,
          logprobs,
        },
      ],
      model,
      object: "text_completion",
      created: state.created,
    };
  }

  private makeStreamFinalChunk(
    request: ChatCompletionRequestStreaming | CompletionCreateParamsStreaming,
    model: string,
    pipeline: LLMChatPipeline,
    state: StreamGenerationState,
  ): ChatCompletionChunk | Completion {
    let finishReason = pipeline.getFinishReason()!;
    const isChatCompletion = "messages" in request;
    const isFunctionCalling =
      isChatCompletion && request.tools !== undefined && request.tools !== null;
    let toolCalls: Array<ChatCompletionChunk.Choice.Delta.ToolCall> | undefined;
    if (pipeline.getFinishReason() === "stop" && isFunctionCalling) {
      finishReason = "tool_calls";
      toolCalls = getToolCallFromOutputMessage(
        pipeline.getMessage(),
        /*isStreaming=*/ true,
      ) as Array<ChatCompletionChunk.Choice.Delta.ToolCall>;
    }

    if (isChatCompletion) {
      return {
        id: state.id,
        choices: [
          {
            delta: isFunctionCalling
              ? {
                  role: "assistant",
                  tool_calls: toolCalls,
                }
              : {},
            finish_reason: finishReason,
            index: 0,
          },
        ],
        model,
        object: "chat.completion.chunk",
        created: state.created,
      };
    }
    return {
      id: state.id,
      choices: [
        {
          text: "",
          finish_reason: finishReason,
          index: 0,
        },
      ],
      model,
      object: "text_completion",
      created: state.created,
    };
  }

  private makeStreamUsageChunk(
    request: ChatCompletionRequestStreaming | CompletionCreateParamsStreaming,
    model: string,
    pipeline: LLMChatPipeline,
    state: StreamGenerationState,
    completionTokenOffset: number,
    timeReceived: number,
  ): ChatCompletionChunk | Completion | undefined {
    if (request.stream_options?.include_usage !== true) {
      return undefined;
    }
    const usedGrammar =
      "response_format" in request &&
      (request.response_format?.type === "grammar" ||
        request.response_format?.type === "json_object");
    const completionTokens =
      pipeline.getCurRoundDecodingTotalTokens() + completionTokenOffset;
    const promptTokens = pipeline.getCurRoundPrefillTotalTokens();
    const prefillTime = pipeline.getCurRoundPrefillTotalTime();
    const decodeTime = pipeline.getCurRoundDecodingTotalTime();
    const defaultExtra = {
      e2e_latency_s: (Date.now() - timeReceived) / 1000,
      prefill_tokens_per_s: pipeline.getCurRoundPrefillTokensPerSec(),
      decode_tokens_per_s: pipeline.getCurRoundDecodingTokensPerSec(),
      time_to_first_token_s: prefillTime,
      time_per_output_token_s: decodeTime / completionTokens,
      latencyBreakdown: request.extra_body?.enable_latency_breakdown
        ? pipeline.getCurRoundLatencyBreakdown()
        : undefined,
    };
    const usage: CompletionUsage = {
      completion_tokens: completionTokens,
      prompt_tokens: promptTokens,
      total_tokens: completionTokens + promptTokens,
      extra: usedGrammar
        ? {
            ...defaultExtra,
            grammar_init_s: pipeline.getCurRoundGrammarInitTotalTime(),
            grammar_per_token_s:
              pipeline.getCurRoundGrammarPerTokenTotalTime() / completionTokens,
          }
        : defaultExtra,
    };

    if ("messages" in request) {
      return {
        id: state.id,
        choices: [],
        usage,
        model,
        object: "chat.completion.chunk",
        created: state.created,
      };
    }
    return {
      id: state.id,
      choices: [],
      usage,
      model,
      object: "text_completion",
      created: state.created,
    };
  }

  private async *streamCurrentGeneration(
    request: ChatCompletionRequestStreaming | CompletionCreateParamsStreaming,
    model: string,
    pipeline: LLMChatPipeline,
    genConfig: GenerationConfig,
    timeReceived: number,
    state: StreamGenerationState,
    decodeStep: () => Promise<void>,
    opts: StreamContinuationOptions = {},
  ): AsyncGenerator<ChatCompletionChunk | Completion, void, void> {
    if (opts.emitCurrent === true) {
      const chunk = this.makeStreamDeltaChunk(
        request,
        model,
        pipeline,
        state,
        opts.skipEmptyDelta,
      );
      if (chunk !== undefined) {
        yield chunk;
      }
    }

    while (!pipeline.stopped()) {
      if (this.interruptSignal) {
        pipeline.triggerStop();
        break;
      }
      await decodeStep();
      const chunk = this.makeStreamDeltaChunk(
        request,
        model,
        pipeline,
        state,
        opts.skipEmptyDelta,
      );
      if (chunk !== undefined) {
        yield chunk;
      }
    }

    if (request.seed !== null && request.seed !== undefined) {
      pipeline.setSeed(Date.now());
    }
    await opts.beforeFinalChunk?.();
    yield this.makeStreamFinalChunk(request, model, pipeline, state);

    const usageChunk = this.makeStreamUsageChunk(
      request,
      model,
      pipeline,
      state,
      opts.completionTokenOffset ?? 0,
      timeReceived,
    );
    if (usageChunk !== undefined) {
      yield usageChunk;
    }
  }

  /**
   * Similar to `_generate()`; but instead of using callback, we use an async iterable.
   */
  asyncGenerate(
    request: ChatCompletionRequestStreaming,
    model: string,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
    genConfig: GenerationConfig,
    timeReceived: number,
    journal?: ResumableGenerationJournal,
  ): AsyncGenerator<ChatCompletionChunk, void, void>;
  asyncGenerate(
    request: CompletionCreateParamsStreaming,
    model: string,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
    genConfig: GenerationConfig,
    timeReceived: number,
    journal?: ResumableGenerationJournal,
  ): AsyncGenerator<Completion, void, void>;
  async *asyncGenerate(
    request: ChatCompletionRequestStreaming | CompletionCreateParamsStreaming,
    model: string,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
    genConfig: GenerationConfig,
    timeReceived: number,
    journal?: ResumableGenerationJournal,
  ): AsyncGenerator<ChatCompletionChunk | Completion, void, void> {
    // Since it is an async generator, we need to do fine-grained try-catch to ensure lock is
    // released only when errors occur. Then release at the very end when no error occurs.
    // TODO: This makes code less readable, is there a better way to do this?
    const lock = this.loadedModelIdToLock.get(model)!;
    if (journal !== undefined) {
      this.resetResumableMetrics();
    }

    const isChatCompletion = "messages" in request;
    const isFunctionCalling =
      "tools" in request &&
      request.tools !== undefined &&
      request.tools !== null;
    let journalStarted = false;
    let journalClosed = false;

    const recordErrorCloseAndRelease = async (err: unknown) => {
      if (journalStarted && !isResumableInjectedFault(err)) {
        await journal?.recordEngineError({ err }).catch(() => undefined);
      }
      await journal?.close().catch(() => undefined);
      await lock.release();
    };
    try {
      if (isFunctionCalling && !isChatCompletion) {
        throw new Error(
          "Expect `chat.completions` with tools, not `completions`.",
        );
      }
      postInitAndCheckGenerationConfigValues(genConfig);
      if (request.seed !== null && request.seed !== undefined) {
        pipeline.setSeed(request.seed);
      }
    } catch (err) {
      await recordErrorCloseAndRelease(err);
      throw err;
    }

    const streamState: StreamGenerationState = {
      id: crypto.randomUUID(),
      created: getUnixTimestampSeconds(),
      prevMessageLength: 0,
    };
    this.interruptSignal = false;

    let checkpointScheduler: DecodeCheckpointScheduler | undefined;
    let resumablePromptSeqLen = 0;
    try {
      if (journal === undefined) {
        await this.prefill(request, pipeline, chatConfig, genConfig);
      } else {
        const prefillStep = await this.samplePrefill(
          request,
          pipeline,
          chatConfig,
          genConfig,
          {
            capturePromptCheckpoint:
              journal.checkpointPrompt && (await hasKVCheckpointQuota()),
            storeCheckpointLogits: journal.storeCheckpointLogits,
          },
        );
        if (prefillStep.promptTokenIds === undefined) {
          throw new Error(
            "Resumable generation currently supports text-only prompts.",
          );
        }
        await journal.begin({
          modelId: model,
          request,
          promptTokenIds: prefillStep.promptTokenIds,
          assistantPrefixTokenIds: prefillStep.assistantPrefixTokenIds ?? [],
          generationConfig: genConfig,
        });
        journalStarted = true;
        resumablePromptSeqLen = prefillStep.globalTokenPos;
        checkpointScheduler = {
          intervalTokens: journal.checkpointIntervalTokens,
          lastCheckpointSeqLen: prefillStep.globalTokenPos,
        };
        if (await this.writePromptCheckpoint(journal, prefillStep)) {
          noteCommittedCheckpoint(
            checkpointScheduler,
            prefillStep.promptCheckpoint!,
          );
        }
        const committedPrefill = pipeline.commitSampledStep(
          prefillStep,
          genConfig,
        );
        await this.recordGeneratedToken(
          journal,
          pipeline,
          prefillStep,
          committedPrefill,
        );
      }
    } catch (err) {
      await recordErrorCloseAndRelease(err);
      throw err;
    }
    try {
      yield* this.streamCurrentGeneration(
        request,
        model,
        pipeline,
        genConfig,
        timeReceived,
        streamState,
        async () => {
          if (journal === undefined) {
            await this.decode(pipeline, genConfig);
            return;
          }
          const captureDecodeCheckpoint =
            shouldCaptureDecodeCheckpoint(
              checkpointScheduler!,
              resumablePromptSeqLen + journal.emittedTokenCount,
            ) && (await hasKVCheckpointQuota());
          const decodeStep = await this.sampleDecode(pipeline, genConfig, {
            captureCheckpoint: captureDecodeCheckpoint,
            storeCheckpointLogits: journal.storeCheckpointLogits,
          });
          const committedDecode = pipeline.commitSampledStep(
            decodeStep,
            genConfig,
          );
          await this.recordGeneratedToken(
            journal,
            pipeline,
            decodeStep,
            committedDecode,
          );
          await this.writeDecodeCheckpoint(
            journal,
            checkpointScheduler!,
            decodeStep,
          );
        },
        {
          emitCurrent: true,
          beforeFinalChunk: async () => {
            if (journal === undefined) {
              return;
            }
            await journal.recordGenerationEnd({
              finishReason: pipeline.getFinishReason(),
              emittedTokens: journal.emittedTokenCount,
            });
            if (pipeline.getFinishReason() !== "abort") {
              await this.cleanupKVAfterFinish(journal);
            }
            await journal.close();
            journalClosed = true;
          },
        },
      );
    } catch (err) {
      await recordErrorCloseAndRelease(err);
      throw err;
    }

    if (!journalClosed) {
      await journal?.close().catch(() => undefined);
    }
    await lock.release();
  }

  async interruptGenerate() {
    this.interruptSignal = true;
  }

  //------------------------------
  // 3. High-level generation APIs
  //------------------------------

  /**
   * Completes a single ChatCompletionRequest.
   *
   * @param request A OpenAI-style ChatCompletion request.
   *
   * @note For each choice (i.e. `n`), a request is defined by a single `prefill()` and multiple
   * `decode()`. This is important as it determines the behavior of various fields including `seed`.
   */
  async chatCompletion(
    request: ChatCompletionRequestNonStreaming,
  ): Promise<ChatCompletion>;
  async chatCompletion(
    request: ChatCompletionRequestStreaming,
  ): Promise<AsyncIterable<ChatCompletionChunk>>;
  async chatCompletion(
    request: ChatCompletionRequestBase,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion>;
  async chatCompletion(
    request: ChatCompletionRequest,
  ): Promise<AsyncIterable<ChatCompletionChunk> | ChatCompletion> {
    const timeReceived = Date.now();
    // 0. Check model loaded and preprocess inputs
    const [selectedModelId, selectedPipeline, selectedChatConfig] =
      this.getLLMStates("ChatCompletionRequest", request.model);
    const selectedModelType =
      this.loadedModelIdToModelType.get(selectedModelId);
    API.postInitAndCheckFieldsChatCompletion(
      request,
      selectedModelId,
      selectedModelType!,
    );
    const genConfig: GenerationConfig = {
      frequency_penalty: request.frequency_penalty,
      presence_penalty: request.presence_penalty,
      repetition_penalty: request.repetition_penalty,
      max_tokens: request.max_tokens,
      stop: request.stop,
      top_p: request.top_p,
      temperature: request.temperature,
      logit_bias: request.logit_bias,
      logprobs: request.logprobs,
      top_logprobs: request.top_logprobs,
      response_format: request.response_format,
      ignore_eos: request.ignore_eos,
      enable_thinking: request.extra_body?.enable_thinking,
      enable_latency_breakdown: request.extra_body?.enable_latency_breakdown,
    };
    const resumableConfig = this.normalizeResumableForRequest(request);
    const resumableJournal = this.tryCreateResumableJournal(resumableConfig);

    // 0.5 Block wait until this pipeline finishes all previous requests
    const lock = this.loadedModelIdToLock.get(selectedModelId)!;
    await lock.acquire();

    // 1. If request is streaming, return an AsyncIterable (an iterable version of `_generate()`)
    if (request.stream) {
      return this.asyncGenerate(
        request,
        selectedModelId,
        selectedPipeline,
        selectedChatConfig,
        genConfig,
        timeReceived,
        resumableJournal,
      );
    }

    // Big try-finally to release lock in case of errors
    try {
      if (request.seed !== null && request.seed !== undefined) {
        selectedPipeline.setSeed(request.seed);
      }

      // 2. If request is non-streaming, directly reuse `_generate()`
      const n = request.n ? request.n : 1;
      const choices: Array<ChatCompletion.Choice> = [];
      let completion_tokens = 0;
      let prompt_tokens = 0;
      let prefill_time = 0;
      let decode_time = 0;
      let grammar_init_s = 0;
      let grammar_per_token_s = 0;
      for (let i = 0; i < n; i++) {
        let outputMessage: string;
        if (this.interruptSignal) {
          // A single interrupt signal should stop all choices' generations
          selectedPipeline.triggerStop();
          outputMessage = "";
        } else {
          outputMessage =
            resumableJournal === undefined
              ? await this._generate(
                  request,
                  selectedPipeline,
                  selectedChatConfig,
                  genConfig,
                )
              : await this._generateResumable(
                  request,
                  selectedModelId,
                  selectedPipeline,
                  selectedChatConfig,
                  genConfig,
                  resumableJournal,
                );
        }
        let finish_reason = selectedPipeline.getFinishReason()!;

        // 3. Post processing for function calling
        const isFunctionCalling =
          request.tools !== undefined && request.tools !== null;
        let tool_calls: Array<ChatCompletionMessageToolCall> | undefined;
        if (
          selectedPipeline.getFinishReason() === "stop" &&
          isFunctionCalling
        ) {
          // If stopped due to length or abort, cannot output return tool_calls field
          finish_reason = "tool_calls";
          tool_calls = getToolCallFromOutputMessage(
            outputMessage,
            /*isStreaming=*/ false,
          );
        }

        choices.push({
          finish_reason: finish_reason,
          index: i,
          logprobs: request.logprobs
            ? ({
                content: selectedPipeline.getTokenLogprobArray(),
              } as ChatCompletion.Choice.Logprobs)
            : null,
          message: isFunctionCalling
            ? {
                content: null,
                tool_calls: tool_calls,
                role: "assistant",
              }
            : {
                content: outputMessage,
                role: "assistant",
              },
        });
        completion_tokens += selectedPipeline.getCurRoundDecodingTotalTokens();
        prompt_tokens += selectedPipeline.getCurRoundPrefillTotalTokens();
        prefill_time += selectedPipeline.getCurRoundPrefillTotalTime();
        decode_time += selectedPipeline.getCurRoundDecodingTotalTime();
        grammar_init_s += selectedPipeline.getCurRoundGrammarInitTotalTime();
        grammar_per_token_s +=
          selectedPipeline.getCurRoundGrammarPerTokenTotalTime();
      }
      const usedGrammar =
        "response_format" in request &&
        (request.response_format?.type === "grammar" ||
          request.response_format?.type === "json_object");

      const latencyBreakdown: LatencyBreakdown =
        selectedPipeline.getCurRoundLatencyBreakdown();

      const defaultExtra = {
        e2e_latency_s: (Date.now() - timeReceived) / 1000,
        prefill_tokens_per_s: prompt_tokens / prefill_time,
        decode_tokens_per_s: completion_tokens / decode_time,
        time_to_first_token_s: prefill_time,
        time_per_output_token_s: decode_time / completion_tokens,
        latencyBreakdown: request.extra_body?.enable_latency_breakdown
          ? latencyBreakdown
          : undefined,
      };
      const response: ChatCompletion = {
        id: crypto.randomUUID(),
        choices: choices,
        model: selectedModelId,
        object: "chat.completion",
        created: getUnixTimestampSeconds(),
        usage: {
          completion_tokens: completion_tokens,
          prompt_tokens: prompt_tokens,
          total_tokens: completion_tokens + prompt_tokens,
          extra: usedGrammar
            ? {
                ...defaultExtra,
                ...{
                  grammar_init_s: grammar_init_s,
                  grammar_per_token_s: grammar_per_token_s / completion_tokens,
                },
              }
            : defaultExtra,
        } as CompletionUsage,
      };

      // Reset seed -- we do not want this seed to affect future requests
      if (request.seed !== null && request.seed !== undefined) {
        selectedPipeline.setSeed(Date.now());
      }
      return response;
    } finally {
      await lock.release();
    }
  }

  /**
   * Completes a single CompletionCreateParams, a text completion with no chat template.
   *
   * @param request A OpenAI-style Completion request.
   *
   * @note For each choice (i.e. `n`), a request is defined by a single `prefill()` and multiple
   * `decode()`. This is important as it determines the behavior of various fields including `seed`.
   */
  async completion(
    request: CompletionCreateParamsNonStreaming,
  ): Promise<Completion>;
  async completion(
    request: CompletionCreateParamsStreaming,
  ): Promise<AsyncIterable<Completion>>;
  async completion(
    request: CompletionCreateParamsBase,
  ): Promise<AsyncIterable<Completion> | Completion>;
  async completion(
    request: CompletionCreateParams,
  ): Promise<AsyncIterable<Completion> | Completion> {
    const timeReceived = Date.now();
    API.rejectCompletionResumable(request);

    // 0. Check model loaded and preprocess inputs
    const [selectedModelId, selectedPipeline, selectedChatConfig] =
      this.getLLMStates("CompletionCreateParams", request.model);
    API.postInitAndCheckFieldsCompletion(request, selectedModelId);
    const genConfig: GenerationConfig = {
      frequency_penalty: request.frequency_penalty,
      presence_penalty: request.presence_penalty,
      repetition_penalty: request.repetition_penalty,
      max_tokens: request.max_tokens,
      stop: request.stop,
      top_p: request.top_p,
      temperature: request.temperature,
      logit_bias: request.logit_bias,
      logprobs: request.logprobs,
      top_logprobs: request.top_logprobs,
      ignore_eos: request.ignore_eos,
      enable_latency_breakdown: request.extra_body?.enable_latency_breakdown,
    };

    // 0.5 Block wait until this pipeline finishes all previous requests
    const lock = this.loadedModelIdToLock.get(selectedModelId)!;
    await lock.acquire();

    // 1. If request is streaming, return an AsyncIterable (an iterable version of `_generate()`)
    if (request.stream) {
      return this.asyncGenerate(
        request,
        selectedModelId,
        selectedPipeline,
        selectedChatConfig,
        genConfig,
        timeReceived,
      );
    }

    // Big try-finally to release lock in case of errors
    try {
      if (request.seed !== null && request.seed !== undefined) {
        selectedPipeline.setSeed(request.seed);
      }

      // 2. If request is non-streaming, directly reuse `_generate()`
      const n = request.n ? request.n : 1;
      const choices: Array<CompletionChoice> = [];
      let completion_tokens = 0;
      let prompt_tokens = 0;
      let prefill_time = 0;
      let decode_time = 0;
      for (let i = 0; i < n; i++) {
        let outputMessage: string;
        if (this.interruptSignal) {
          // A single interrupt signal should stop all choices' generations
          selectedPipeline.triggerStop();
          outputMessage = "";
        } else {
          outputMessage = await this._generate(
            request,
            selectedPipeline,
            selectedChatConfig,
            genConfig,
          );
        }
        const finish_reason = selectedPipeline.getFinishReason()!;

        choices.push({
          finish_reason: finish_reason,
          index: i,
          logprobs: request.logprobs
            ? ({
                content: selectedPipeline.getTokenLogprobArray(),
              } as ChatCompletion.Choice.Logprobs)
            : null,
          text: request.echo ? request.prompt + outputMessage : outputMessage,
        });
        completion_tokens += selectedPipeline.getCurRoundDecodingTotalTokens();
        prompt_tokens += selectedPipeline.getCurRoundPrefillTotalTokens();
        prefill_time += selectedPipeline.getCurRoundPrefillTotalTime();
        decode_time += selectedPipeline.getCurRoundDecodingTotalTime();
      }

      const latencyBreakdown: LatencyBreakdown =
        selectedPipeline.getCurRoundLatencyBreakdown();

      const response: Completion = {
        id: crypto.randomUUID(),
        choices: choices,
        model: selectedModelId,
        object: "text_completion",
        created: getUnixTimestampSeconds(),
        usage: {
          completion_tokens: completion_tokens,
          prompt_tokens: prompt_tokens,
          total_tokens: completion_tokens + prompt_tokens,
          extra: {
            e2e_latency_s: (Date.now() - timeReceived) / 1000,
            prefill_tokens_per_s: prompt_tokens / prefill_time,
            decode_tokens_per_s: completion_tokens / decode_time,
            time_to_first_token_s: prefill_time,
            time_per_output_token_s: decode_time / completion_tokens,
            latencyBreakdown: request.extra_body?.enable_latency_breakdown
              ? latencyBreakdown
              : undefined,
          },
        } as CompletionUsage,
      };

      // Reset seed -- we do not want this seed to affect future requests
      if (request.seed !== null && request.seed !== undefined) {
        selectedPipeline.setSeed(Date.now());
      }
      return response;
    } finally {
      await lock.release();
    }
  }

  async embedding(
    request: EmbeddingCreateParams,
  ): Promise<CreateEmbeddingResponse> {
    // 0. Preprocess inputs
    const [selectedModelId, selectedPipeline] = this.getEmbeddingStates(
      "EmbeddingCreateParams",
      request.model,
    );
    API.postInitAndCheckFieldsEmbedding(request, selectedModelId);

    // 0.5 Block wait until this pipeline finishes all previous requests
    const lock = this.loadedModelIdToLock.get(selectedModelId)!;
    await lock.acquire();

    try {
      // 1. Call EmbeddingPipeline to get embeddings
      const embedResult: Array<Array<number>> =
        await selectedPipeline.embedStep(request.input);

      // 2. Prepare response
      const batchSize = embedResult.length;
      const data: Array<Embedding> = [];
      for (let i = 0; i < batchSize; i++) {
        const curEmbedding: Embedding = {
          embedding: embedResult[i],
          index: i,
          object: "embedding",
        };
        data.push(curEmbedding);
      }
      return {
        data: data,
        model: selectedModelId,
        object: "list",
        usage: {
          prompt_tokens: selectedPipeline.getCurRoundEmbedTotalTokens(),
          total_tokens: selectedPipeline.getCurRoundEmbedTotalTokens(),
          extra: {
            prefill_tokens_per_s:
              selectedPipeline.getCurRoundEmbedTokensPerSec(),
          },
        },
      };
    } finally {
      await lock.release();
    }
  }

  async listResumableSessions(): Promise<ResumeProbeResult[]> {
    try {
      const files = this.getResumableFileStore();
      const sessions = this.getResumableSessionStore();
      const results: ResumeProbeResult[] = [];
      for (const session of await sessions.listSessions()) {
        let result = await probeResumableSession(files, session);
        if (
          result.resumable &&
          result.recoveryMode === "token_replay" &&
          (await sessions.listCommittedCheckpoints(session.sessionId)).length >
            0
        ) {
          result = { ...result, recoveryMode: "kv" };
        }
        if (
          result.resumable &&
          (result.recoveryMode === "token_replay" ||
            result.recoveryMode === "kv") &&
          this.logitProcessorRegistry?.has(result.modelId)
        ) {
          results.push({
            ...result,
            resumable: false,
            reason: "custom LogitProcessor replay unsupported",
            recoveryMode: result.emittedTokens > 0 ? "text_only" : "none",
          });
        } else {
          results.push(result);
        }
      }
      return results;
    } catch {
      return [];
    }
  }

  async resumeChatCompletion(
    sessionId: string,
    options?: ResumeChatCompletionOptions,
  ): Promise<ResumeResult | AsyncIterable<ChatCompletionChunk>> {
    const files = this.getResumableFileStore();
    const sessions = this.getResumableSessionStore();
    const session = await sessions.openSession(sessionId);
    if (session === undefined) {
      throw new Error(`Resumable session not found: ${sessionId}`);
    }
    if (options?.continueGeneration !== true) {
      const replayState = await readResumableReplayState(files, session);
      return this.makeTextOnlyResumeResult(replayState);
    }

    const releaseSessionLock = await files.tryLock(session.paths.lockPath);
    if (releaseSessionLock === undefined) {
      throw new Error(`Resumable session is already active: ${sessionId}`);
    }
    let releaseSessionLockOnExit = true;
    try {
      this.resetResumableMetrics();
      const replayState = await readResumableReplayState(files, session);
      if (replayState.resumableConfig === undefined) {
        throw new Error(
          `Resumable session ${sessionId} is missing or has malformed resumable generation config.`,
        );
      }

      let selectedModelId: string;
      let selectedPipeline: LLMChatPipeline;
      let selectedChatConfig: ChatConfig;
      try {
        [selectedModelId, selectedPipeline, selectedChatConfig] =
          this.getLLMStates("resumeChatCompletion", replayState.modelId);
      } catch (err) {
        if (
          err instanceof ModelNotLoadedError ||
          err instanceof SpecifiedModelNotFoundError
        ) {
          return this.makeTextOnlyResumeResult(replayState);
        }
        throw err;
      }
      if (selectedModelId !== replayState.modelId) {
        return this.makeTextOnlyResumeResult(replayState);
      }

      const lock = this.loadedModelIdToLock.get(selectedModelId)!;
      await lock.acquire();
      let releaseModelLock = true;
      try {
        if (replayState.generationConfig === undefined) {
          return this.makeTextOnlyResumeResult(replayState);
        }
        const genConfig = replayState.generationConfig;
        postInitAndCheckGenerationConfigValues(genConfig);
        let restored = await this.tryRestoreKVResumeState(
          files,
          sessions,
          session,
          replayState,
          selectedPipeline,
          genConfig,
        );
        if (restored === undefined) {
          const blockReason = this.getTokenReplayBlockReason(replayState);
          if (blockReason !== undefined) {
            log.warn(
              `Resumable session ${sessionId} recovered as text-only: ${blockReason}.`,
            );
            return this.makeTextOnlyResumeResult(replayState);
          }
          const replayStart = performance.now();
          try {
            await selectedPipeline.replayGenerationTokens(
              replayState.promptTokenIds,
              replayState.assistantPrefixTokenIds,
              replayState.generatedTokens,
              genConfig,
            );
          } finally {
            this.resumableMetrics().tokenReplayMs +=
              performance.now() - replayStart;
          }
          const rngState =
            replayState.generatedTokens[replayState.generatedTokens.length - 1]
              .rngState;
          if (!selectedPipeline.setRNGState(rngState)) {
            selectedPipeline.resetChat();
            return this.makeTextOnlyResumeResult(replayState);
          }
          restored = {
            recoveryMode: "token_replay",
            replayedTokens: replayState.emittedTokens,
            extraEmittedTokens: 0,
            firstTokenRecorded: false,
            lastCheckpointSeqLen: replayPromptSeqLen(replayState),
          };
        }

        const resumeJournal = await this.createResumeContinuationJournal(
          files,
          sessions,
          session,
          replayState,
        );
        const checkpointScheduler = this.createResumeCheckpointScheduler(
          resumeJournal,
          restored,
        );
        const promptSeqLen = replayPromptSeqLen(replayState);

        if (
          options?.stream === true &&
          isChatCompletionReplayRequest(replayState.request) &&
          replayState.request.stream === true
        ) {
          releaseModelLock = false;
          releaseSessionLockOnExit = false;
          return this.streamResumeContinuation(
            replayState.request,
            selectedModelId,
            selectedPipeline,
            selectedChatConfig,
            genConfig,
            replayState,
            restored,
            resumeJournal,
            checkpointScheduler,
            promptSeqLen,
            async () => {
              try {
                await lock.release();
              } finally {
                releaseSessionLock();
              }
            },
          );
        }

        return await this.finishResumeContinuation(
          replayState,
          selectedPipeline,
          selectedChatConfig,
          genConfig,
          restored,
          resumeJournal,
          checkpointScheduler,
          promptSeqLen,
        );
      } finally {
        if (releaseModelLock) {
          await lock.release();
        }
      }
    } finally {
      if (releaseSessionLockOnExit) {
        releaseSessionLock();
      }
    }
  }

  async deleteResumableSession(sessionId: string): Promise<void> {
    try {
      await this.getResumableSessionStore().deleteSession(sessionId);
    } catch (err) {
      if (!isOPFSUnavailableError(err)) {
        throw err;
      }
    }
  }

  //-----------------------------
  // 4. WebGPU info-querying helpers
  //-----------------------------

  async getMaxStorageBufferBindingSize(): Promise<number> {
    // First detect GPU
    const gpuDetectOutput = await tvmjs.detectGPUDevice();
    if (gpuDetectOutput == undefined) {
      throw new WebGPUNotAvailableError();
    }

    const computeMB = (value: number) => {
      return Math.ceil(value / (1 << 20)) + "MB";
    };
    const maxStorageBufferBindingSize =
      gpuDetectOutput.device.limits.maxStorageBufferBindingSize;
    const defaultMaxStorageBufferBindingSize = 1 << 30; // 1GB
    if (maxStorageBufferBindingSize < defaultMaxStorageBufferBindingSize) {
      log.warn(
        `WARNING: the current maxStorageBufferBindingSize ` +
          `(${computeMB(maxStorageBufferBindingSize)}) ` +
          `may only work for a limited number of models, e.g.: \n` +
          `- Llama-3.1-8B-Instruct-q4f16_1-MLC-1k \n` +
          `- Llama-2-7b-chat-hf-q4f16_1-MLC-1k \n` +
          `- RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC-1k \n` +
          `- RedPajama-INCITE-Chat-3B-v1-q4f32_1-MLC-1k \n` +
          `- TinyLlama-1.1B-Chat-v0.4-q4f16_1-MLC-1k \n` +
          `- TinyLlama-1.1B-Chat-v0.4-q4f32_1-MLC-1k`,
      );
    }
    return maxStorageBufferBindingSize;
  }

  async getGPUVendor(): Promise<string> {
    // First detect GPU
    const gpuDetectOutput = await tvmjs.detectGPUDevice();
    if (gpuDetectOutput == undefined) {
      throw new WebGPUNotAvailableError();
    }
    return gpuDetectOutput.adapterInfo.vendor;
  }

  //---------------------------------------------------------------
  // 5. Helper for querying currently loaded model/pipeline/config.
  // Needed due to possibly multiple loaded models.
  //---------------------------------------------------------------

  private getLLMStates(
    requestName: string,
    modelId?: string | null,
  ): [string, LLMChatPipeline, ChatConfig] {
    return this.getModelStates(requestName, ModelType.LLM, modelId) as [
      string,
      LLMChatPipeline,
      ChatConfig,
    ];
  }

  private getEmbeddingStates(
    requestName: string,
    modelId?: string | null,
  ): [string, EmbeddingPipeline, ChatConfig] {
    return this.getModelStates(requestName, ModelType.embedding, modelId) as [
      string,
      EmbeddingPipeline,
      ChatConfig,
    ];
  }

  /**
   * Return the model, its LLMChatPipeline, and ChatConfig to use. Throws error when unclear which
   * model to load. Ensure all loadedModelIdToXXX maps contain entry for the selected modelId.
   * @param requestName The type of request or API to load the model for. Needed for error throwing.
   * @param modelType The typ of model, determining what type of pipeline to expect.
   * @param modelId Model the user specified to load via the request. Required when multiple
   *   models are loaded
   */
  private getModelStates(
    requestName: string,
    modelType: ModelType,
    modelId?: string | null,
  ): [string, LLMChatPipeline | EmbeddingPipeline, ChatConfig] {
    // 0. Select model based on request.model and loadedModelIds
    const loadedModelIds: string[] = Array.from(
      this.loadedModelIdToPipeline.keys(),
    );
    const selectedModelId: string = getModelIdToUse(
      loadedModelIds,
      modelId,
      requestName,
    );

    // 1. Retrieve pipeline
    const selectedPipeline = this.loadedModelIdToPipeline.get(selectedModelId);
    if (modelType === ModelType.LLM) {
      if (!(selectedPipeline instanceof LLMChatPipeline)) {
        throw new IncorrectPipelineLoadedError(
          selectedModelId,
          "LLMChatPipeline",
          requestName,
        );
      }
    } else {
      // ModelType.Embedding
      if (!(selectedPipeline instanceof EmbeddingPipeline)) {
        throw new IncorrectPipelineLoadedError(
          selectedModelId,
          "EmbeddingPipeline",
          requestName,
        );
      }
      if (
        findModelRecord(selectedModelId, this.appConfig).model_type !==
        ModelType.embedding
      ) {
        throw new EmbeddingUnsupportedModelError(selectedModelId);
      }
    }

    // 2. Retrieve chat config
    const selectedChatConfig =
      this.loadedModelIdToChatConfig.get(selectedModelId);
    if (selectedChatConfig === undefined) {
      throw new Error(
        `InternalError: chat config not registered for ${selectedModelId}.`,
      );
    }

    // 3. Make sure lock is initialized
    if (!this.loadedModelIdToLock.has(selectedModelId)) {
      throw new Error(
        `InternalError: loadedModelIdToLock does not contain ${selectedModelId}`,
      );
    }
    return [selectedModelId, selectedPipeline, selectedChatConfig];
  }

  //--------------------------------------------------------------------
  // 6. External low-level APIs that directly interacts with a pipeline.
  //--------------------------------------------------------------------

  async forwardTokensAndSample(
    inputIds: Array<number>,
    isPrefill: boolean,
    modelId?: string,
  ): Promise<number> {
    const [, selectedPipeline] = this.getLLMStates(
      "forwardTokensAndSample",
      modelId,
    );
    return selectedPipeline.forwardTokensAndSample(inputIds, isPrefill);
  }

  /**
   * Get the current generated response.
   *
   * @returns The current output message.
   */
  async getMessage(modelId?: string): Promise<string> {
    const [, selectedPipeline] = this.getLLMStates("getMessage", modelId);
    return selectedPipeline.getMessage();
  }

  async runtimeStatsText(modelId?: string): Promise<string> {
    log.warn(
      "WARNING: `runtimeStatsText()` will soon be deprecated. " +
        "Please use `ChatCompletion.usage` for non-streaming requests, or " +
        "`ChatCompletionChunk.usage` for streaming requests, enabled by `stream_options`. " +
        "The only flow that expects to use `runtimeStatsText()` as of now is `forwardTokensAndSample()`.",
    );
    const [, selectedPipeline] = this.getLLMStates("runtimeStatsText", modelId);
    return selectedPipeline.runtimeStatsText();
  }

  async resetChat(keepStats = false, modelId?: string) {
    try {
      const [, selectedPipeline] = this.getLLMStates("resetChat", modelId);
      selectedPipeline.resetChat(keepStats);
    } catch (error) {
      if (
        error instanceof ModelNotLoadedError ||
        error instanceof SpecifiedModelNotFoundError
      ) {
        // Only allow calling resetChat before pipeline instantiated.
        log.debug(
          "Caught an expected error in resetChat, treating it as no-op. Error: ",
          error,
        );
      } else {
        throw error;
      }
    }
  }

  //-----------------------------------------------
  // 7. Prefill and decode given an LLMChatPipeline
  //-----------------------------------------------

  /**
   * Run a prefill step with a given input.
   *
   * If `input` is a chatCompletionRequest, we treat `input.messages[-1]` as the usual user input.
   * We then convert `input.messages[:-1]` to a `Conversation` object, representing a conversation
   * history.
   *
   * If the new `Conversation` object matches the current one loaded, it means we are
   * performing multi-round chatting, so we do not reset, hence reusing KV cache. Otherwise, we
   * reset every thing, treating the request as something completely new.
   *
   * @param input The OpenAI-style prompt to prefill.
   * @param pipeline The loaded pipeline, hence model, to carry out this prefill.
   * @param chatConfig The chat config to use for this model.
   * @param genConfig Generation config.
   */
  private preparePrefillInput(
    input: ChatCompletionRequest | CompletionCreateParams,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
  ): {
    inputStr: string;
    lastMsgRole: Role;
    inputRoleStr?: string;
  } {
    // TODO: SPECIFY MODEL TO PERFORM PREFILL, HENCE RETRIEVE CONFIG
    if (chatConfig === undefined) {
      throw new ConfigurationNotInitializedError();
    }
    let inputStr: string;
    let inputRoleStr: string | undefined;
    let lastMsgRole = Role.user;
    if ("messages" in input) {
      // For ChatCompletionRequest, we prepare input using `messages`
      // 1. Get new conversation based on request, determine if we are in multiround chatting
      const oldConv = pipeline.getConversationObject();
      const newConv = getConversationFromChatCompletionRequest(
        input,
        chatConfig,
      );
      if (!compareConversationObject(oldConv, newConv)) {
        // Not the same conversation, so not multiround chatting, reset everything (KV cache, etc.)
        pipeline.resetChat();
        pipeline.setConversation(newConv);
      } else if (newConv.messages.length === 0) {
        // Empty oldConv, and no chat history in newConv, so reset and setConversation
        pipeline.resetChat();
        pipeline.setConversation(newConv);
      } else {
        log.info("Multiround chatting, reuse KVCache.");
      }

      // 2. Treat the last message as the usual input
      const last_msg = input.messages[
        input.messages.length - 1
      ] as ChatCompletionMessageParam;
      inputStr = last_msg.content as string;
      inputRoleStr =
        last_msg.role === "user" && last_msg.name ? last_msg.name : undefined;
      lastMsgRole = last_msg.role === "tool" ? Role.tool : Role.user;
    } else {
      // For CompletionCreateParams, the input is just the prompt
      inputStr = input.prompt;
      pipeline.resetChat();
      const newConv = getConversation(
        chatConfig.conv_template,
        chatConfig.conv_config,
        true,
      );
      pipeline.setConversation(newConv);
    }
    return { inputStr, lastMsgRole, inputRoleStr };
  }

  async prefill(
    input: ChatCompletionRequest | CompletionCreateParams,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
    genConfig: GenerationConfig,
  ) {
    const { inputStr, lastMsgRole, inputRoleStr } = this.preparePrefillInput(
      input,
      pipeline,
      chatConfig,
    );
    return pipeline.prefillStep(inputStr, lastMsgRole, inputRoleStr, genConfig);
  }

  async samplePrefill(
    input: ChatCompletionRequest | CompletionCreateParams,
    pipeline: LLMChatPipeline,
    chatConfig: ChatConfig,
    genConfig: GenerationConfig,
    opts?: SamplePrefillOptions,
  ): Promise<SampledGenerationStep> {
    const { inputStr, lastMsgRole, inputRoleStr } = this.preparePrefillInput(
      input,
      pipeline,
      chatConfig,
    );
    return pipeline.samplePrefillStep(
      inputStr,
      lastMsgRole,
      inputRoleStr,
      genConfig,
      opts,
    );
  }

  /**
   * Run a decode step to decode the next token.
   */
  async decode(pipeline: LLMChatPipeline, genConfig?: GenerationConfig) {
    return pipeline.decodeStep(genConfig);
  }

  async sampleDecode(
    pipeline: LLMChatPipeline,
    genConfig?: GenerationConfig,
    opts?: SampleDecodeOptions,
  ): Promise<SampledGenerationStep> {
    return pipeline.sampleDecodeStep(genConfig, opts);
  }
}
