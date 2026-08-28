/**
 * Deterministic MLCEngine tests that run without WebGPU by mocking LLMChatPipeline.
 */
import {
  ChatCompletion,
  ChatCompletionRequest,
  Completion,
  CompletionCreateParams,
  EmbeddingCreateParams,
  ChatCompletionChunk,
} from "../src/openai_api_protocols";
import { MLCEngine } from "../src/engine";
import { ModelType } from "../src/config";
import { LLMChatPipeline } from "../src/llm_chat";
import { EmbeddingPipeline } from "../src/embedding";
import { CustomLock } from "../src/support";
import { UnclearModelToUseError } from "../src/error";
import { OPFSFileStore } from "../src/resumable/opfs_file_store";
import {
  readJournalRecords,
  appendJournalRecord,
  decodeJournalRecordAt,
  JournalRecordType,
} from "../src/resumable/journal";
import {
  ResumableInjectedFault,
  setResumableFaultHook,
} from "../src/resumable/fault_injection";
import type {
  ResumableFaultContext,
  ResumableFaultPoint,
} from "../src/resumable/fault_injection";
import { ResumableSessionStore } from "../src/resumable/session_store";
import { jest, test, expect, describe, afterEach } from "@jest/globals";
import log from "loglevel";

type ChatConfig = import("../src/config").ChatConfig;
type Conversation = import("../src/conversation").Conversation;
type TVMInstance = import("@mlc-ai/web-runtime").Instance;
type Tokenizer = import("@mlc-ai/web-tokenizers").Tokenizer;

jest.mock("../src/llm_chat", () => {
  const { getConversation } = jest.requireActual(
    "../src/conversation",
  ) as typeof import("../src/conversation");

  class MockLLMChatPipeline {
    public decodeLimit = 2;
    public prefillCallCount = 0;
    public decodeCallCount = 0;
    public resetCount = 0;
    public enablePromptCheckpoint = false;
    public enableDecodeCheckpoint = false;
    public promptCheckpointRestoreCount = 0;
    public restoredCheckpointSeqLen = 0;
    public checkpointPageSize = 1;
    private conversation: Conversation = getConversation(
      {
        system_template: "{system_message}",
        system_message: "",
        roles: { user: "user", assistant: "assistant" },
        seps: ["\n"],
        stop_token_ids: [0],
        stop_str: [],
      } as any,
      undefined,
    );
    private stopFlag = true;
    private message = "";
    private finishReason: string | undefined = undefined;
    private curRoundPrefillTotalTokens = 0;
    private curRoundDecodingTotalTokens = 0;
    private curRoundPrefillTotalTime = 0.001;
    private curRoundDecodingTotalTime = 0.001;
    private curRoundGrammarPerTokenTotalTime = 0;
    private pendingPrefillMessage = "";
    private pendingDecodeMessage = "";
    private pendingDecodeStop = false;
    private rngState: unknown;

    constructor(_tvm: TVMInstance, _tokenizer: Tokenizer, config: ChatConfig) {
      this.conversation = getConversation(
        config.conv_template,
        config.conv_config,
      );
    }

    async asyncLoadWebGPUPipelines() {}
    dispose() {}
    async sync() {}
    setSeed(seed: number) {
      this.rngState = seed;
    }

    getConversationObject() {
      return this.conversation;
    }

    setConversation(newConv: Conversation) {
      this.conversation = newConv;
    }

    resetChat() {
      this.resetCount++;
      this.stopFlag = true;
      this.decodeCallCount = 0;
      this.message = "";
      this.finishReason = undefined;
      this.curRoundPrefillTotalTokens = 0;
      this.curRoundDecodingTotalTokens = 0;
    }

    async prefillStep(
      inp: string,
      msgRole: string,
      roleName?: string,
    ): Promise<void> {
      const step = await this.samplePrefillStep(inp, msgRole, roleName);
      this.commitSampledStep(step);
    }

    async samplePrefillStep(
      inp: string,
      msgRole: string,
      roleName?: string,
      _genConfig?: unknown,
      opts?: {
        capturePromptCheckpoint?: boolean;
        storeCheckpointLogits?: boolean;
      },
    ): Promise<any> {
      this.prefillCallCount++;
      const roleSuffix = roleName ? `(${roleName})` : "";
      this.pendingPrefillMessage = `${msgRole}${roleSuffix}:${inp}`;
      this.stopFlag = false;
      this.decodeCallCount = 0;
      this.curRoundPrefillTotalTokens = Math.max(1, inp.length);
      this.curRoundPrefillTotalTime = 0.01 * this.curRoundPrefillTotalTokens;
      this.curRoundDecodingTotalTokens = 0;
      this.curRoundDecodingTotalTime = 0.001;
      this.curRoundGrammarPerTokenTotalTime = 0;
      this.finishReason = "length";
      const promptCheckpoint =
        this.enablePromptCheckpoint && opts?.capturePromptCheckpoint === true
          ? {
              processedSeqLen: this.curRoundPrefillTotalTokens,
              layoutHash: "mock-layout",
              metadata: {
                seqLength: this.curRoundPrefillTotalTokens,
                layoutHash: "mock-layout",
                pageSize: this.checkpointPageSize,
                groups: [
                  {
                    groupIndex: 0,
                    layerBegin: 0,
                    layerEnd: 1,
                    shape: [1, 1],
                    dtype: "uint8",
                  },
                ],
              },
              pageGroups: [
                {
                  groupId: 0,
                  layerStart: 0,
                  layerEnd: 1,
                  data: new Uint8Array([7]),
                },
              ],
              nextLogits:
                opts?.storeCheckpointLogits === false
                  ? undefined
                  : {
                      shape: [1],
                      dtype: "uint8",
                      data: new Uint8Array([8]),
                    },
            }
          : undefined;
      return {
        source: "prefill",
        tokenId: 100,
        globalTokenPos: this.curRoundPrefillTotalTokens,
        promptLen: this.curRoundPrefillTotalTokens,
        promptTokenIds: Array.from(inp).map((char) => char.charCodeAt(0)),
        assistantPrefixTokenIds: [],
        promptCheckpoint,
      };
    }

    async decodeStep(genConfig?: { max_tokens?: number | null }) {
      const step = await this.sampleDecodeStep(genConfig);
      this.commitSampledStep(step);
    }

    async sampleDecodeStep(
      genConfig?: { max_tokens?: number | null },
      opts?: {
        captureCheckpoint?: boolean;
        storeCheckpointLogits?: boolean;
      },
    ) {
      if (this.stopFlag) return;
      this.decodeCallCount++;
      const globalTokenPos =
        this.curRoundPrefillTotalTokens + this.decodeCallCount;
      this.pendingDecodeMessage = `|token${this.decodeCallCount}|`;
      this.curRoundDecodingTotalTokens = this.decodeCallCount;
      this.curRoundDecodingTotalTime = this.curRoundDecodingTotalTokens * 0.02;
      this.curRoundGrammarPerTokenTotalTime =
        this.curRoundDecodingTotalTokens * 0.001;
      this.pendingDecodeStop =
        this.decodeCallCount >= this.decodeLimit ||
        (genConfig?.max_tokens !== null &&
          genConfig?.max_tokens !== undefined &&
          this.decodeCallCount >= genConfig.max_tokens);
      return {
        source: "decode",
        tokenId: 100 + this.decodeCallCount,
        globalTokenPos,
        decodeCheckpoint:
          this.enableDecodeCheckpoint && opts?.captureCheckpoint === true
            ? {
                processedSeqLen: globalTokenPos,
                layoutHash: "mock-layout",
                metadata: {
                  seqLength: globalTokenPos,
                  layoutHash: "mock-layout",
                  pageSize: this.checkpointPageSize,
                  groups: [
                    {
                      groupIndex: 0,
                      layerBegin: 0,
                      layerEnd: 1,
                      shape: [1, 1],
                      dtype: "uint8",
                    },
                  ],
                },
                pageGroups: [
                  {
                    groupId: 0,
                    layerStart: 0,
                    layerEnd: 1,
                    data: new Uint8Array([7 + this.decodeCallCount]),
                  },
                ],
                nextLogits:
                  opts?.storeCheckpointLogits === false
                    ? undefined
                    : {
                        shape: [1],
                        dtype: "uint8",
                        data: new Uint8Array([8 + this.decodeCallCount]),
                      },
              }
            : undefined,
      };
    }

    commitSampledStep(step: any) {
      const prevMessage = this.message;
      if (step.source === "prefill") {
        this.message = this.pendingPrefillMessage;
      } else {
        this.message += this.pendingDecodeMessage;
        if (this.pendingDecodeStop) {
          this.stopFlag = true;
          this.finishReason = "stop";
        }
      }
      return {
        source: step.source,
        tokenId: step.tokenId,
        globalTokenPos: step.globalTokenPos,
        textDelta: this.message.slice(prevMessage.length),
        textPrefixLength: prevMessage.length,
        outputMessage: this.message,
        stopped: this.stopFlag,
        finishReason: this.finishReason,
      };
    }

    getRNGState() {
      return this.prefillCallCount * 1000 + this.decodeCallCount;
    }

    setRNGState(state: unknown) {
      this.rngState = state;
      return state !== undefined;
    }

    async replayGenerationTokens(
      promptTokenIds: number[],
      _assistantPrefixTokenIds: number[],
      generatedTokens: Array<{
        tokenId: number;
        textDelta: string;
        textPrefixLength?: number;
      }>,
    ) {
      this.resetChat();
      this.stopFlag = false;
      this.finishReason = "length";
      this.curRoundPrefillTotalTokens = promptTokenIds.length;
      this.curRoundPrefillTotalTime = Math.max(
        0.001,
        promptTokenIds.length * 0.01,
      );
      this.decodeCallCount = Math.max(0, generatedTokens.length - 1);
      this.curRoundDecodingTotalTokens = 0;
      this.curRoundDecodingTotalTime = 0.001;
      this.message = generatedTokens.reduce(
        (message, token) =>
          message.slice(0, token.textPrefixLength ?? message.length) +
          token.textDelta,
        "",
      );
    }

    async replayFromPromptCheckpoint(
      checkpoint: { processedSeqLen?: number },
      _assistantPrefixTokenIds: number[],
      coveredTokens: Array<{
        tokenId: number;
        textDelta: string;
        textPrefixLength?: number;
      }>,
      tailTokens: Array<{
        tokenId: number;
        textDelta: string;
        textPrefixLength?: number;
      }>,
    ) {
      this.promptCheckpointRestoreCount++;
      this.restoredCheckpointSeqLen = checkpoint.processedSeqLen ?? 0;
      this.resetChat();
      this.stopFlag = false;
      this.finishReason = "length";
      const generatedTokens = [...coveredTokens, ...tailTokens];
      this.decodeCallCount = Math.max(0, generatedTokens.length - 1);
      this.curRoundDecodingTotalTokens = 0;
      this.curRoundDecodingTotalTime = 0.001;
      this.message = generatedTokens.reduce(
        (message, token) =>
          message.slice(0, token.textPrefixLength ?? message.length) +
          token.textDelta,
        "",
      );
      return {
        replayedTokens: tailTokens.length,
        sampledFromCheckpointLogits: false,
      };
    }

    stopped() {
      return this.stopFlag;
    }

    triggerStop() {
      this.stopFlag = true;
      this.finishReason = "abort";
    }

    getMessage() {
      return this.message;
    }

    getFinishReason() {
      return this.finishReason ?? "stop";
    }

    getCurRoundDecodingTotalTokens() {
      return this.curRoundDecodingTotalTokens;
    }

    getCurRoundPrefillTotalTokens() {
      return this.curRoundPrefillTotalTokens;
    }

    getCurRoundPrefillTokensPerSec() {
      return this.curRoundPrefillTotalTokens / this.curRoundPrefillTotalTime;
    }

    getCurRoundDecodingTokensPerSec() {
      return this.curRoundDecodingTotalTokens / this.curRoundDecodingTotalTime;
    }

    getCurRoundGrammarInitTotalTime() {
      return 0.001;
    }

    getCurRoundPrefillTotalTime() {
      return this.curRoundPrefillTotalTime;
    }

    getCurRoundDecodingTotalTime() {
      return this.curRoundDecodingTotalTime;
    }

    getCurRoundGrammarPerTokenTotalTime() {
      return this.curRoundGrammarPerTokenTotalTime;
    }

    getCurRoundLatencyBreakdown() {
      return {
        logitProcessorTime: [0.001],
        logitBiasTime: [0.001],
        penaltyTime: [0.001],
        sampleTime: [0.001],
        totalTime: [0.001],
        grammarBitmaskTime: [0.001],
      };
    }

    getTokenLogprobArray() {
      return [];
    }

    async forwardTokensAndSample(inputIds: Array<number>): Promise<number> {
      return inputIds[0] ?? 0;
    }

    async runtimeStatsText() {
      return `prefills=${this.prefillCallCount}`;
    }
  }

  return { LLMChatPipeline: MockLLMChatPipeline };
});

jest.mock("../src/embedding", () => {
  class MockEmbeddingPipeline {
    public inputs: any;
    public embedResult: Array<Array<number>> = [[0.1, 0.2, 0.3]];
    dispose() {}
    async sync() {}
    async embedStep(
      input: string | Array<string> | Array<number> | Array<Array<number>>,
    ): Promise<Array<Array<number>>> {
      this.inputs = input;
      return this.embedResult;
    }
    getCurRoundEmbedTotalTokens(): number {
      if (typeof this.inputs === "string") {
        return this.inputs.length;
      } else if (Array.isArray(this.inputs)) {
        return this.inputs.length;
      }
      return 0;
    }
    getCurRoundEmbedTokensPerSec(): number {
      const tokens = this.getCurRoundEmbedTotalTokens();
      return tokens === 0 ? 0 : tokens / 0.01;
    }
  }
  return { EmbeddingPipeline: MockEmbeddingPipeline };
});

test("MLCEngine resumable helpers tolerate unavailable OPFS for list/delete", async () => {
  const engine = new MLCEngine();
  await expect(engine.listResumableSessions()).resolves.toEqual([]);
  await expect(
    engine.deleteResumableSession("session-a"),
  ).resolves.toBeUndefined();
  await expect(engine.resumeChatCompletion("session-a")).rejects.toThrow(
    "OPFS is unavailable in this environment",
  );
});

const MODEL_ID = "mock-model";
const SECOND_MODEL_ID = "mock-model-2";
const EMBED_MODEL_ID = "mock-embed";
const FIXED_CREATED_DATE = new Date("2024-04-05T06:34:56.789Z");
const FIXED_CREATED_SECONDS = 1712298896;

function normalize(path: string): string {
  return path
    .split("/")
    .filter((part) => part !== "")
    .join("/");
}

function parentDirs(path: string): string[] {
  const parts = normalize(path).split("/");
  parts.pop();
  const dirs: string[] = [];
  for (let i = 1; i <= parts.length; i++) {
    dirs.push(parts.slice(0, i).join("/"));
  }
  return dirs;
}

function bytes(data: BufferSource): Uint8Array<ArrayBuffer> {
  const view = ArrayBuffer.isView(data)
    ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
    : new Uint8Array(data);
  return new Uint8Array(view);
}

class MemoryFileStore implements OPFSFileStore {
  private readonly files = new Map<string, Uint8Array<ArrayBuffer>>();
  private readonly dirs = new Set<string>([""]);
  private readonly locks = new Set<string>();
  public pauseAppends = false;
  public pauseAppendWhen?: (
    path: string,
    data: Uint8Array<ArrayBuffer>,
  ) => boolean;
  private readonly appendResolvers: Array<() => void> = [];

  async read(path: string): Promise<ArrayBuffer | undefined> {
    const data = this.files.get(normalize(path));
    return data === undefined ? undefined : new Uint8Array(data).buffer;
  }

  async write(path: string, data: BufferSource): Promise<void> {
    const normalized = normalize(path);
    for (const dir of parentDirs(normalized)) {
      this.dirs.add(dir);
    }
    this.files.set(normalized, bytes(data));
  }

  async append(path: string, data: BufferSource): Promise<void> {
    const normalized = normalize(path);
    const chunk = bytes(data);
    if (
      this.pauseAppends ||
      (this.pauseAppendWhen?.(normalized, chunk) ?? false)
    ) {
      await new Promise<void>((resolve) => {
        this.appendResolvers.push(resolve);
      });
    }
    const prev = this.files.get(normalized) ?? new Uint8Array();
    const next = new Uint8Array(prev.byteLength + chunk.byteLength);
    next.set(prev);
    next.set(chunk, prev.byteLength);
    await this.write(normalized, next);
  }

  async remove(path: string, opts?: { recursive?: boolean }): Promise<void> {
    const normalized = normalize(path);
    if (this.files.delete(normalized)) {
      return;
    }
    const prefix = `${normalized}/`;
    const childFiles = [...this.files.keys()].filter((key) =>
      key.startsWith(prefix),
    );
    const childDirs = [...this.dirs].filter((dir) => dir.startsWith(prefix));
    if (
      opts?.recursive !== true &&
      (childFiles.length > 0 || childDirs.length > 0)
    ) {
      throw new Error("Directory is not empty");
    }
    this.dirs.delete(normalized);
    childFiles.forEach((key) => this.files.delete(key));
    childDirs.forEach((dir) => this.dirs.delete(dir));
  }

  async list(path: string): Promise<string[]> {
    const normalized = normalize(path);
    const prefix = normalized === "" ? "" : `${normalized}/`;
    if (normalized !== "" && !this.dirs.has(normalized)) {
      return [];
    }
    const names = new Set<string>();
    for (const key of [...this.dirs, ...this.files.keys()]) {
      if (key === normalized || !key.startsWith(prefix)) {
        continue;
      }
      names.add(key.slice(prefix.length).split("/")[0]);
    }
    return [...names].sort();
  }

  async mkdir(path: string): Promise<void> {
    const normalized = normalize(path);
    for (const dir of [...parentDirs(normalized), normalized]) {
      this.dirs.add(dir);
    }
  }

  async lock(path: string): Promise<() => void> {
    const release = await this.tryLock(path);
    if (release === undefined) {
      throw new Error(`Unable to acquire lock: ${path}`);
    }
    return release;
  }

  async tryLock(path: string): Promise<(() => void) | undefined> {
    const normalized = normalize(path);
    if (this.locks.has(normalized)) {
      return undefined;
    }
    this.locks.add(normalized);
    return () => {
      this.locks.delete(normalized);
    };
  }

  releaseOneAppend(): void {
    this.appendResolvers.shift()?.();
  }

  releaseAllAppends(): void {
    while (this.appendResolvers.length > 0) {
      this.releaseOneAppend();
    }
  }
}

function attachResumableStore(engine: MLCEngine, files: MemoryFileStore): void {
  const internal = engine as any;
  internal.resumableFileStore = files;
  internal.resumableSessionStore = new ResumableSessionStore(files, {
    rootPath: "resume-root",
  });
}

async function readSessionJournal(files: MemoryFileStore, sessionId: string) {
  return readJournalRecords(
    files,
    `resume-root/sessions/${sessionId}/journal.bin`,
  );
}

let restoreResumableFaultHook: (() => void) | undefined;

function clearResumableFaultHook(): void {
  restoreResumableFaultHook?.();
  restoreResumableFaultHook = undefined;
}

function injectResumableFaultOnce(
  point: ResumableFaultPoint,
  matches: (context: ResumableFaultContext) => boolean = () => true,
): void {
  clearResumableFaultHook();
  let fired = false;
  restoreResumableFaultHook = setResumableFaultHook((actual, context) => {
    if (!fired && actual === point && matches(context)) {
      fired = true;
      throw new ResumableInjectedFault(actual, context);
    }
  });
}

function setNavigatorStorageEstimate(estimate: {
  quota?: number;
  usage?: number;
}): () => void {
  const previous = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: {
      storage: {
        estimate: jest.fn(async () => estimate),
      },
    },
  });
  return () => {
    if (previous === undefined) {
      delete (globalThis as any).navigator;
    } else {
      Object.defineProperty(globalThis, "navigator", previous);
    }
  };
}

const mockChatConfig: ChatConfig = {
  tokenizer_files: ["tokenizer.json"],
  vocab_size: 10,
  conv_template: {
    system_template: "{system_message}",
    system_message: "You are a helpful assistant.",
    system_prefix_token_ids: [1],
    add_role_after_system_message: false,
    roles: {
      user: "User",
      assistant: "Assistant",
      tool: "Tool",
    },
    role_templates: {
      user: "{user_message}",
      assistant: "{assistant_message}",
      tool: "{tool_message}",
    },
    seps: ["\n"],
    role_content_sep: ": ",
    role_empty_sep: ": ",
    stop_str: [],
    stop_token_ids: [0],
  },
  conv_config: undefined,
  context_window_size: 8,
  sliding_window_size: -1,
  attention_sink_size: -1,
  temperature: 0.8,
  presence_penalty: 0,
  frequency_penalty: 0,
  repetition_penalty: 1,
  top_p: 1,
};

function createEngineWithPipeline(decodeLimit = 2, modelId = MODEL_ID) {
  const engine = new MLCEngine({
    appConfig: {
      model_list: [
        {
          model: "https://example.com/model",
          model_id: modelId,
          model_lib: "https://example.com/model.wasm",
        },
      ],
      cacheBackend: "cache",
    },
  });
  const pipeline = new LLMChatPipeline(
    null as unknown as TVMInstance,
    null as unknown as Tokenizer,
    mockChatConfig,
  ) as any;
  pipeline.decodeLimit = decodeLimit;
  const internal = engine as any;
  internal.loadedModelIdToPipeline.set(modelId, pipeline);
  internal.loadedModelIdToChatConfig.set(modelId, mockChatConfig);
  internal.loadedModelIdToModelType.set(modelId, ModelType.LLM);
  internal.loadedModelIdToLock.set(modelId, new CustomLock());
  return { engine, pipeline };
}

function createEngineWithMultiplePipelines() {
  const engine = new MLCEngine({
    appConfig: {
      model_list: [
        {
          model: "https://example.com/model",
          model_id: MODEL_ID,
          model_lib: "https://example.com/model.wasm",
        },
        {
          model: "https://example.com/model2",
          model_id: SECOND_MODEL_ID,
          model_lib: "https://example.com/model2.wasm",
        },
      ],
      cacheBackend: "cache",
    },
  });
  const pipeline1 = new LLMChatPipeline(
    null as unknown as TVMInstance,
    null as unknown as Tokenizer,
    mockChatConfig,
  ) as any;
  const pipeline2 = new LLMChatPipeline(
    null as unknown as TVMInstance,
    null as unknown as Tokenizer,
    mockChatConfig,
  ) as any;
  const internal = engine as any;
  internal.loadedModelIdToPipeline.set(MODEL_ID, pipeline1);
  internal.loadedModelIdToPipeline.set(SECOND_MODEL_ID, pipeline2);
  internal.loadedModelIdToChatConfig.set(MODEL_ID, mockChatConfig);
  internal.loadedModelIdToChatConfig.set(SECOND_MODEL_ID, mockChatConfig);
  internal.loadedModelIdToModelType.set(MODEL_ID, ModelType.LLM);
  internal.loadedModelIdToModelType.set(SECOND_MODEL_ID, ModelType.LLM);
  internal.loadedModelIdToLock.set(MODEL_ID, new CustomLock());
  internal.loadedModelIdToLock.set(SECOND_MODEL_ID, new CustomLock());
  return engine;
}

const mockEmbeddingConfig: ChatConfig = {
  ...mockChatConfig,
};

function createEngineWithEmbeddingPipeline() {
  const engine = new MLCEngine({
    appConfig: {
      model_list: [
        {
          model: "https://example.com/embed",
          model_id: EMBED_MODEL_ID,
          model_lib: "https://example.com/embed.wasm",
          model_type: ModelType.embedding,
        },
      ],
      cacheBackend: "cache",
    },
  });
  const pipeline = new EmbeddingPipeline(
    null as unknown as TVMInstance,
    null as unknown as Tokenizer,
    mockEmbeddingConfig,
  ) as any;
  const internal = engine as any;
  internal.loadedModelIdToPipeline.set(EMBED_MODEL_ID, pipeline);
  internal.loadedModelIdToChatConfig.set(EMBED_MODEL_ID, mockEmbeddingConfig);
  internal.loadedModelIdToModelType.set(EMBED_MODEL_ID, ModelType.embedding);
  internal.loadedModelIdToLock.set(EMBED_MODEL_ID, new CustomLock());
  return { engine, pipeline };
}

afterEach(() => {
  clearResumableFaultHook();
  jest.useRealTimers();
});

describe("MLCEngine deterministic integration", () => {
  test("chatCompletion aggregates usage without WebGPU", async () => {
    jest.useFakeTimers().setSystemTime(FIXED_CREATED_DATE);
    const { engine, pipeline } = createEngineWithPipeline(3);
    const request: ChatCompletionRequest = {
      model: MODEL_ID,
      messages: [
        { role: "system", content: "Stay concise." },
        { role: "user", content: "What is new?" },
      ],
      n: 2,
    };
    const response = (await engine.chatCompletion(request)) as ChatCompletion;

    expect(response.choices).toHaveLength(2);
    response.choices.forEach((choice) => {
      expect(choice.message?.content).toContain("What is new?");
    });
    expect(response.created).toBe(FIXED_CREATED_SECONDS);
    expect(response.usage?.completion_tokens).toBe(6);
    expect(response.usage?.prompt_tokens).toBeGreaterThan(0);
    expect((pipeline as any).prefillCallCount).toBe(2);
  });

  test("completion echoes prompt when requested", async () => {
    jest.useFakeTimers().setSystemTime(FIXED_CREATED_DATE);
    const { engine } = createEngineWithPipeline(1);
    const request: CompletionCreateParams = {
      model: MODEL_ID,
      prompt: "Alpha ",
      n: 1,
      echo: true,
    };
    const response = (await engine.completion(request)) as Completion;

    expect(response.choices).toHaveLength(1);
    expect(response.choices[0].text.startsWith("Alpha ")).toBe(true);
    expect(response.created).toBe(FIXED_CREATED_SECONDS);
    expect(response.usage?.completion_tokens).toBe(1);
    expect(response.usage?.prompt_tokens).toBeGreaterThan(0);
  });

  test("completion rejects resumable extra_body without creating a session", async () => {
    const { engine } = createEngineWithPipeline(1);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);

    await expect(
      engine.completion({
        model: MODEL_ID,
        prompt: "Do not persist this",
        extra_body: {
          resumable: {
            enabled: true,
            sessionId: "completion-session",
          },
        },
      } as unknown as CompletionCreateParams),
    ).rejects.toThrow("extra_body.resumable");
    await expect(files.list("resume-root/sessions")).resolves.toEqual([]);
  });

  test("forwardTokensAndSample and runtimeStatsText use mock pipeline", async () => {
    const { engine } = createEngineWithPipeline();
    await expect(
      engine.forwardTokensAndSample([9, 4, 2], true, MODEL_ID),
    ).resolves.toBe(9);
    await expect(engine.runtimeStatsText(MODEL_ID)).resolves.toContain(
      "prefills=",
    );
  });

  test("chatCompletion streaming yields chunks, final delta, and usage data", async () => {
    jest.useFakeTimers().setSystemTime(FIXED_CREATED_DATE);
    const { engine } = createEngineWithPipeline(2);
    const request: ChatCompletionRequest = {
      model: MODEL_ID,
      messages: [
        { role: "system", content: "rules" },
        { role: "user", content: "Stream please" },
      ],
      stream: true,
      stream_options: { include_usage: true },
    };
    const iterable = (await engine.chatCompletion(
      request,
    )) as AsyncIterable<ChatCompletionChunk>;
    const chunks: ChatCompletionChunk[] = [];
    for await (const chunk of iterable) {
      chunks.push(chunk);
    }
    expect(chunks.length).toBeGreaterThanOrEqual(3);
    expect(chunks[0].choices[0].delta?.content).toContain("Stream please");
    expect(
      chunks.every((chunk) => chunk.created === FIXED_CREATED_SECONDS),
    ).toBe(true);
    const finalChunk = chunks[chunks.length - 2];
    expect(finalChunk.choices[0].finish_reason).toEqual("stop");
    const usageChunk = chunks[chunks.length - 1];
    expect(usageChunk.usage?.completion_tokens).toBeGreaterThan(0);
    expect(usageChunk.usage?.prompt_tokens).toBeGreaterThan(0);
  });

  test("resumable chatCompletion journals visible text and token ids", async () => {
    const { engine } = createEngineWithPipeline(2);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const response = (await engine.chatCompletion({
      model: MODEL_ID,
      messages: [{ role: "user", content: "Persist this" }],
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-a",
        },
      },
    })) as ChatCompletion;

    const journal = await readSessionJournal(files, "session-a");
    const generated = journal.records.filter(
      (record) => record.type === JournalRecordType.GeneratedToken,
    );
    expect(generated.map((record) => record.payload.tokenId)).toEqual([
      100, 101, 102,
    ]);
    expect(generated.map((record) => record.payload.textDelta).join("")).toBe(
      response.choices[0].message.content,
    );
    expect(
      journal.records.some(
        (record) => record.type === JournalRecordType.GenerationFinished,
      ),
    ).toBe(true);
    await expect(engine.listResumableSessions()).resolves.toEqual([
      expect.objectContaining({
        sessionId: "session-a",
        resumable: false,
        reason: "generation finished",
        modelId: MODEL_ID,
        emittedTokens: 3,
        recoveryMode: "none",
      }),
    ]);
  });

  test("resumable prefill does not reuse a compatible conversation KV cache", () => {
    const { engine, pipeline } = createEngineWithPipeline(1);
    const request: ChatCompletionRequest = {
      model: MODEL_ID,
      messages: [
        { role: "user", content: "First" },
        { role: "assistant", content: "Answer" },
        { role: "user", content: "Second" },
      ],
    };
    const internal = engine as any;

    internal.preparePrefillInput(request, pipeline, mockChatConfig, true);
    pipeline.resetCount = 0;
    internal.preparePrefillInput(request, pipeline, mockChatConfig, true);
    expect(pipeline.resetCount).toBe(0);

    internal.preparePrefillInput(request, pipeline, mockChatConfig, false);
    expect(pipeline.resetCount).toBe(1);
  });

  test("exact-mode resumable streaming waits for generated token journal append", async () => {
    const { engine } = createEngineWithPipeline(1);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    let generatedTokenAppendCount = 0;
    files.pauseAppendWhen = (_path, data) => {
      const record = decodeJournalRecordAt(data.buffer).record;
      if (record.type !== JournalRecordType.GeneratedToken) {
        return false;
      }
      generatedTokenAppendCount++;
      return true;
    };
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      messages: [{ role: "user", content: "Stream persist" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-stream",
          durabilityMode: "exact",
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();
    let resolved = false;
    const firstChunk = iterator.next().then((value) => {
      resolved = true;
      return value;
    });

    for (let i = 0; i < 10 && generatedTokenAppendCount === 0; i++) {
      await new Promise<void>((resolve) => {
        setTimeout(resolve, 0);
      });
    }
    expect(generatedTokenAppendCount).toBe(1);
    expect(resolved).toBe(false);
    files.pauseAppendWhen = undefined;
    files.releaseAllAppends();
    const result = await firstChunk;
    expect(result.done).toBe(false);
    expect(result.value.choices[0].delta?.content).toContain("Stream persist");
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate releases the model lock
    }
  });

  test("listResumableSessions reports interrupted token replay candidates", async () => {
    const { engine } = createEngineWithPipeline(5);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      messages: [{ role: "user", content: "Interrupt persist" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-interrupted",
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    const first = await iterator.next();
    expect(first.done).toBe(false);
    await engine.interruptGenerate();
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate records the abort and releases the lock
    }

    const sessions = await engine.listResumableSessions();
    expect(sessions).toEqual([
      expect.objectContaining({
        sessionId: "session-interrupted",
        resumable: true,
        reason: "generation aborted",
        modelId: MODEL_ID,
        emittedTokens: 1,
        recoveryMode: "token_replay",
      }),
    ]);
    expect(sessions[0].processedSeqLen).toBeGreaterThan(0);

    await engine.deleteResumableSession("session-interrupted");
    await expect(engine.listResumableSessions()).resolves.toEqual([]);
    await expect(files.list("resume-root/sessions")).resolves.toEqual([]);
  });

  test("resumeChatCompletion continues interrupted generation by token replay", async () => {
    const { engine, pipeline } = createEngineWithPipeline(3);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      seed: 1234,
      messages: [{ role: "user", content: "Resume persist" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-resume",
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    const first = await iterator.next();
    expect(first.value.choices[0].delta?.content).toBe("user:Resume persist");
    await engine.interruptGenerate();
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate records the abort and releases the lock
    }

    const result = (await engine.resumeChatCompletion("session-resume", {
      continueGeneration: true,
    })) as import("../src/types").ResumeResult;
    expect(result).toMatchObject({
      sessionId: "session-resume",
      recoveryMode: "token_replay",
      replayedTokens: 1,
      emittedTokens: 4,
      recoveredText: "user:Resume persist|token1||token2||token3|",
    });
    const journal = await readSessionJournal(files, "session-resume");
    const generated = journal.records.filter(
      (record) => record.type === JournalRecordType.GeneratedToken,
    );
    expect(generated.map((record) => record.payload.tokenId)).toEqual([
      100, 101, 102, 103,
    ]);
    expect(
      journal.records.some(
        (record) => record.type === JournalRecordType.GenerationFinished,
      ),
    ).toBe(true);

    const resetCountAfterResume = pipeline.resetCount;
    await engine.chatCompletion({
      model: MODEL_ID,
      messages: [
        { role: "user", content: "Resume persist" },
        { role: "assistant", content: result.recoveredText },
        { role: "user", content: "Follow up" },
      ],
    });
    expect(pipeline.resetCount).toBe(resetCountAfterResume);
  });

  test("resumeChatCompletion can stream continuation after token replay", async () => {
    const { engine } = createEngineWithPipeline(3);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      seed: 1234,
      messages: [{ role: "user", content: "Resume stream" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-resume-stream",
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    await iterator.next();
    await engine.interruptGenerate();
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate records the abort and releases the lock
    }

    const result = await engine.resumeChatCompletion("session-resume-stream", {
      continueGeneration: true,
      stream: true,
    });
    expect(Symbol.asyncIterator in (result as object)).toBe(true);
    const chunks: ChatCompletionChunk[] = [];
    for await (const chunk of result as AsyncIterable<ChatCompletionChunk>) {
      chunks.push(chunk);
    }
    expect(
      chunks.map((chunk) => chunk.choices[0]?.delta?.content ?? "").join(""),
    ).toBe("|token1||token2||token3|");
    expect(chunks[chunks.length - 1].choices[0].finish_reason).toBe("stop");
    const journal = await readSessionJournal(files, "session-resume-stream");
    const generated = journal.records.filter(
      (record) => record.type === JournalRecordType.GeneratedToken,
    );
    expect(generated.map((record) => record.payload.tokenId)).toEqual([
      100, 101, 102, 103,
    ]);
    expect(
      journal.records.some(
        (record) => record.type === JournalRecordType.GenerationFinished,
      ),
    ).toBe(true);
  });

  test("resumeChatCompletion can resume again after resumed journal append crash", async () => {
    const { engine } = createEngineWithPipeline(3);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      seed: 1234,
      messages: [{ role: "user", content: "Resume twice" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-resume-twice",
          strictPersistence: true,
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    await iterator.next();
    await engine.interruptGenerate();
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate records the abort and releases the lock
    }

    injectResumableFaultOnce(
      "journal.after_append",
      (context) => context.recordType === JournalRecordType.GeneratedToken,
    );
    await expect(
      engine.resumeChatCompletion("session-resume-twice", {
        continueGeneration: true,
      }),
    ).rejects.toThrow("Injected resumable fault");
    clearResumableFaultHook();

    const crashedJournal = await readSessionJournal(
      files,
      "session-resume-twice",
    );
    expect(
      crashedJournal.records.filter(
        (record) => record.type === JournalRecordType.GeneratedToken,
      ),
    ).toHaveLength(2);
    expect(
      crashedJournal.records.some(
        (record) => record.type === JournalRecordType.EngineError,
      ),
    ).toBe(false);

    const result = (await engine.resumeChatCompletion("session-resume-twice", {
      continueGeneration: true,
    })) as import("../src/types").ResumeResult;
    expect(result).toMatchObject({
      sessionId: "session-resume-twice",
      recoveryMode: "token_replay",
      replayedTokens: 2,
      emittedTokens: 4,
      recoveredText: "user:Resume twice|token1||token2||token3|",
    });
  });

  test("resumeChatCompletion continueGeneration rejects missing resumable config", async () => {
    const { engine } = createEngineWithPipeline(3);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const sessions = new ResumableSessionStore(files, {
      rootPath: "resume-root",
    });
    const session = await sessions.createSession("session-missing-resumable", {
      modelId: MODEL_ID,
    });
    await appendJournalRecord(files, session.paths.journalPath, {
      type: JournalRecordType.SessionBegin,
      seqNo: 1,
      createdAtMs: Date.now(),
      payload: {
        sessionId: "session-missing-resumable",
        modelId: MODEL_ID,
        request: {
          model: MODEL_ID,
          messages: [{ role: "user", content: "Missing config" }],
        },
      },
    });
    await appendJournalRecord(files, session.paths.journalPath, {
      type: JournalRecordType.PromptTokens,
      seqNo: 2,
      createdAtMs: Date.now(),
      payload: { tokenIds: [1, 2, 3] },
    });
    await appendJournalRecord(files, session.paths.journalPath, {
      type: JournalRecordType.GenerationConfig,
      seqNo: 3,
      createdAtMs: Date.now(),
      payload: { config: { max_tokens: 3 } },
    });
    await appendJournalRecord(files, session.paths.journalPath, {
      type: JournalRecordType.GeneratedToken,
      seqNo: 4,
      createdAtMs: Date.now(),
      payload: {
        globalTokenPos: 3,
        tokenId: 100,
        textDelta: "partial",
        rngState: 1,
      },
    });
    await appendJournalRecord(files, session.paths.journalPath, {
      type: JournalRecordType.GenerationAborted,
      seqNo: 5,
      createdAtMs: Date.now(),
      payload: { reason: "abort" },
    });

    await expect(
      engine.resumeChatCompletion("session-missing-resumable"),
    ).resolves.toMatchObject({
      sessionId: "session-missing-resumable",
      recoveredText: "partial",
      recoveryMode: "text_only",
    });
    await expect(engine.listResumableSessions()).resolves.toEqual([
      expect.objectContaining({
        sessionId: "session-missing-resumable",
        resumable: false,
        reason: "missing resumable generation config; continuation unavailable",
        recoveryMode: "text_only",
      }),
    ]);
    await expect(
      engine.resumeChatCompletion("session-missing-resumable", {
        continueGeneration: true,
      }),
    ).rejects.toThrow(
      "Resumable session session-missing-resumable is missing or has malformed resumable generation config.",
    );
  });

  test("resumeChatCompletion continueGeneration fails while session is active", async () => {
    const { engine } = createEngineWithPipeline(3);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      seed: 1234,
      messages: [{ role: "user", content: "Resume lock" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-resume-lock",
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    await iterator.next();
    await engine.interruptGenerate();
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate records the abort and releases the lock
    }

    const release = await files.tryLock(
      "resume-root/sessions/session-resume-lock/lock",
    );
    expect(release).toBeDefined();
    await expect(
      engine.resumeChatCompletion("session-resume-lock", {
        continueGeneration: true,
      }),
    ).rejects.toThrow(
      "Resumable session is already active: session-resume-lock",
    );
    await expect(
      engine.resumeChatCompletion("session-resume-lock"),
    ).resolves.toMatchObject({
      sessionId: "session-resume-lock",
      recoveryMode: "text_only",
      recoveredText: "user:Resume lock",
    });

    release!();
    await expect(
      engine.resumeChatCompletion("session-resume-lock", {
        continueGeneration: true,
      }),
    ).resolves.toMatchObject({
      sessionId: "session-resume-lock",
      recoveryMode: "token_replay",
    });
  });

  test("resumeChatCompletion restores from prompt checkpoint when available", async () => {
    const { engine, pipeline } = createEngineWithPipeline(3);
    pipeline.enablePromptCheckpoint = true;
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      seed: 1234,
      messages: [{ role: "user", content: "KV resume" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-kv-resume",
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    await iterator.next();
    await engine.interruptGenerate();
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate records the abort and releases the lock
    }

    const checkpoints = await files.list(
      "resume-root/sessions/session-kv-resume/kv",
    );
    expect(checkpoints).toEqual(["checkpoint_00000000_00000009"]);
    await expect(engine.listResumableSessions()).resolves.toEqual([
      expect.objectContaining({
        sessionId: "session-kv-resume",
        recoveryMode: "kv",
      }),
    ]);

    const result = (await engine.resumeChatCompletion("session-kv-resume", {
      continueGeneration: true,
    })) as import("../src/types").ResumeResult;
    expect(result).toMatchObject({
      sessionId: "session-kv-resume",
      recoveryMode: "kv",
      replayedTokens: 1,
      recoveredText: "user:KV resume|token1||token2||token3|",
    });
    expect(pipeline.promptCheckpointRestoreCount).toBe(1);

    const resetCountAfterResume = pipeline.resetCount;
    await engine.chatCompletion({
      model: MODEL_ID,
      messages: [
        { role: "user", content: "KV resume" },
        { role: "assistant", content: result.recoveredText },
        { role: "user", content: "Follow up" },
      ],
    });
    expect(pipeline.resetCount).toBe(resetCountAfterResume);
  });

  test("resumeChatCompletion falls back to token replay when its only checkpoint is corrupt", async () => {
    const { engine, pipeline } = createEngineWithPipeline(3);
    pipeline.enablePromptCheckpoint = true;
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const sessionId = "session-corrupt-prompt-checkpoint";
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      seed: 1234,
      messages: [{ role: "user", content: "Corrupt KV" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId,
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    await iterator.next();
    await engine.interruptGenerate();
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate records the abort and releases the lock
    }

    const kvRoot = `resume-root/sessions/${sessionId}/kv`;
    const checkpoints = await files.list(kvRoot);
    expect(checkpoints).toHaveLength(1);
    await files.write(
      `${kvRoot}/${checkpoints[0]}/meta.json`,
      new TextEncoder().encode("{"),
    );
    const warn = jest.spyOn(log, "warn").mockImplementation(() => undefined);

    const result = (await engine.resumeChatCompletion(sessionId, {
      continueGeneration: true,
    })) as import("../src/types").ResumeResult;
    expect(result).toMatchObject({
      sessionId,
      recoveryMode: "token_replay",
      replayedTokens: 1,
      recoveredText: "user:Corrupt KV|token1||token2||token3|",
    });
    expect(pipeline.promptCheckpointRestoreCount).toBe(0);
    expect(warn).toHaveBeenCalledWith(
      `Ignoring invalid KV checkpoint ${checkpoints[0]}:`,
      expect.any(SyntaxError),
    );
    warn.mockRestore();
  });

  test("resumable crash after GENERATED_TOKEN append resumes by token replay", async () => {
    const { engine } = createEngineWithPipeline(3);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    injectResumableFaultOnce(
      "journal.after_append",
      (context) => context.recordType === JournalRecordType.GeneratedToken,
    );
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      seed: 1234,
      messages: [{ role: "user", content: "Crash token" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-crash-token",
          strictPersistence: true,
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    await expect(iterator.next()).rejects.toThrow("Injected resumable fault");
    clearResumableFaultHook();

    const journal = await readSessionJournal(files, "session-crash-token");
    expect(
      journal.records.filter(
        (record) => record.type === JournalRecordType.GeneratedToken,
      ),
    ).toHaveLength(1);
    expect(
      journal.records.some(
        (record) => record.type === JournalRecordType.EngineError,
      ),
    ).toBe(false);

    const result = (await engine.resumeChatCompletion("session-crash-token", {
      continueGeneration: true,
    })) as import("../src/types").ResumeResult;
    expect(result).toMatchObject({
      sessionId: "session-crash-token",
      recoveryMode: "token_replay",
      replayedTokens: 1,
      recoveredText: "user:Crash token|token1||token2||token3|",
    });
  });

  test.each([
    ["checkpoint.after_page_group", "token_replay"],
    ["checkpoint.after_next_logits", "token_replay"],
    ["checkpoint.after_meta", "token_replay"],
    ["checkpoint.after_complete", "token_replay"],
    ["checkpoint.before_commit", "token_replay"],
    ["checkpoint.after_commit", "kv"],
  ] as Array<[ResumableFaultPoint, "token_replay" | "kv"]>)(
    "decode checkpoint crash at %s resumes by %s",
    async (faultPoint, recoveryMode) => {
      const sessionId = `session-${faultPoint.replaceAll(".", "-")}`;
      const { engine, pipeline } = createEngineWithPipeline(4);
      pipeline.enableDecodeCheckpoint = true;
      const files = new MemoryFileStore();
      attachResumableStore(engine, files);
      injectResumableFaultOnce(faultPoint);
      const iterable = (await engine.chatCompletion({
        model: MODEL_ID,
        seed: 1234,
        messages: [{ role: "user", content: "KV crash" }],
        stream: true,
        extra_body: {
          resumable: {
            enabled: true,
            sessionId,
            checkpointPrompt: false,
            checkpointIntervalTokens: 1,
            strictPersistence: true,
          },
        },
      })) as AsyncIterable<ChatCompletionChunk>;
      const iterator = iterable[Symbol.asyncIterator]();

      await expect(iterator.next()).resolves.toMatchObject({
        done: false,
      });
      await expect(iterator.next()).rejects.toThrow("Injected resumable fault");
      clearResumableFaultHook();

      const journal = await readSessionJournal(files, sessionId);
      expect(
        journal.records.some(
          (record) => record.type === JournalRecordType.EngineError,
        ),
      ).toBe(false);
      await expect(engine.listResumableSessions()).resolves.toEqual([
        expect.objectContaining({
          sessionId,
          recoveryMode,
        }),
      ]);

      const result = (await engine.resumeChatCompletion(sessionId, {
        continueGeneration: true,
      })) as import("../src/types").ResumeResult;
      expect(result).toMatchObject({
        sessionId,
        recoveryMode,
      });
      expect(result.recoveredText).toBe(
        "user:KV crash|token1||token2||token3||token4|",
      );
    },
  );

  test("resumeChatCompletion writes decode checkpoints during resumed continuation", async () => {
    const { engine, pipeline } = createEngineWithPipeline(4);
    pipeline.enableDecodeCheckpoint = true;
    pipeline.checkpointPageSize = 1;
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      seed: 1234,
      messages: [{ role: "user", content: "Resume checkpoint" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-resume-checkpoint",
          checkpointPrompt: false,
          checkpointIntervalTokens: 1,
          strictPersistence: true,
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    await iterator.next();
    await engine.interruptGenerate();
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate records the abort and releases the lock
    }

    injectResumableFaultOnce("checkpoint.after_commit");
    await expect(
      engine.resumeChatCompletion("session-resume-checkpoint", {
        continueGeneration: true,
      }),
    ).rejects.toThrow("Injected resumable fault");
    clearResumableFaultHook();

    const crashedJournal = await readSessionJournal(
      files,
      "session-resume-checkpoint",
    );
    expect(
      crashedJournal.records.some(
        (record) => record.type === JournalRecordType.CheckpointCommit,
      ),
    ).toBe(true);
    await expect(engine.listResumableSessions()).resolves.toEqual([
      expect.objectContaining({
        sessionId: "session-resume-checkpoint",
        recoveryMode: "kv",
      }),
    ]);

    const result = (await engine.resumeChatCompletion(
      "session-resume-checkpoint",
      { continueGeneration: true },
    )) as import("../src/types").ResumeResult;
    expect(result).toMatchObject({
      sessionId: "session-resume-checkpoint",
      recoveryMode: "kv",
      recoveredText: "user:Resume checkpoint|token1||token2||token3||token4|",
    });
  });

  test("resumable metrics report journal, checkpoint, and restore timing", async () => {
    const { engine, pipeline } = createEngineWithPipeline(3);
    pipeline.enablePromptCheckpoint = true;
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      seed: 1234,
      messages: [{ role: "user", content: "Metrics" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-metrics",
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    await iterator.next();
    await engine.interruptGenerate();
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate records the abort and releases the lock
    }

    const writeMetrics = (engine as any).lastResumableMetrics;
    expect(writeMetrics.journalAppendMs).toBeGreaterThanOrEqual(0);
    expect(writeMetrics.checkpointWriteMs).toBeGreaterThanOrEqual(0);

    const result = (await engine.resumeChatCompletion("session-metrics", {
      continueGeneration: true,
    })) as import("../src/types").ResumeResult;
    expect(result.recoveryMode).toBe("kv");
    const resumeMetrics = (engine as any).lastResumableMetrics;
    expect(resumeMetrics.kvRestoreMs).toBeGreaterThanOrEqual(0);
    expect(resumeMetrics.resumeFirstTokenMs).toBeGreaterThanOrEqual(0);
  });

  test("low storage quota skips KV checkpoints but keeps token journal", async () => {
    const restoreNavigator = setNavigatorStorageEstimate({
      quota: 128 * 1024 * 1024,
      usage: 120 * 1024 * 1024,
    });
    try {
      const { engine, pipeline } = createEngineWithPipeline(3);
      pipeline.enablePromptCheckpoint = true;
      const files = new MemoryFileStore();
      attachResumableStore(engine, files);
      const iterable = (await engine.chatCompletion({
        model: MODEL_ID,
        seed: 1234,
        messages: [{ role: "user", content: "Low quota" }],
        stream: true,
        extra_body: {
          resumable: {
            enabled: true,
            sessionId: "session-low-quota",
          },
        },
      })) as AsyncIterable<ChatCompletionChunk>;
      const iterator = iterable[Symbol.asyncIterator]();

      await iterator.next();
      await engine.interruptGenerate();
      while (!(await iterator.next()).done) {
        // drain the generator so asyncGenerate records the abort and releases the lock
      }

      const journal = await readSessionJournal(files, "session-low-quota");
      expect(
        journal.records.some(
          (record) => record.type === JournalRecordType.GeneratedToken,
        ),
      ).toBe(true);
      await expect(
        files.list("resume-root/sessions/session-low-quota/kv"),
      ).resolves.toEqual([]);
      await expect(engine.listResumableSessions()).resolves.toEqual([
        expect.objectContaining({
          sessionId: "session-low-quota",
          recoveryMode: "token_replay",
        }),
      ]);
    } finally {
      restoreNavigator();
    }
  });

  test("finished resumable generation removes persisted KV", async () => {
    const { engine, pipeline } = createEngineWithPipeline(1);
    pipeline.enablePromptCheckpoint = true;
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);

    await engine.chatCompletion({
      model: MODEL_ID,
      messages: [{ role: "user", content: "Finish cleanup" }],
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-finish-cleanup",
        },
      },
    });

    await expect(
      files.list("resume-root/sessions/session-finish-cleanup/kv"),
    ).resolves.toEqual([]);
    await expect(
      files.list("resume-root/sessions/session-finish-cleanup"),
    ).resolves.not.toContain("kv");
  });

  test("resumeChatCompletion restores from latest complete decode checkpoint", async () => {
    const { engine, pipeline } = createEngineWithPipeline(5);
    pipeline.enableDecodeCheckpoint = true;
    pipeline.checkpointPageSize = 2;
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      seed: 1234,
      messages: [{ role: "user", content: "KVdecode" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-decode-kv-resume",
          checkpointPrompt: false,
          checkpointIntervalTokens: 2,
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    await iterator.next();
    await iterator.next();
    await iterator.next();
    await engine.interruptGenerate();
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate records the abort and releases the lock
    }

    const incompleteCheckpoint = "checkpoint_00000000_00000012";
    const incompletePath = `resume-root/sessions/session-decode-kv-resume/kv/${incompleteCheckpoint}`;
    await files.mkdir(incompletePath);
    await appendJournalRecord(
      files,
      "resume-root/sessions/session-decode-kv-resume/journal.bin",
      {
        type: JournalRecordType.CheckpointCommit,
        seqNo: 999,
        createdAtMs: Date.now(),
        payload: {
          checkpointId: incompleteCheckpoint,
          processedSeqLen: 12,
          path: incompletePath,
          layoutHash: "mock-layout",
        },
      },
    );
    await expect(
      files.list("resume-root/sessions/session-decode-kv-resume/kv"),
    ).resolves.toEqual(["checkpoint_00000000_00000010", incompleteCheckpoint]);

    const result = (await engine.resumeChatCompletion(
      "session-decode-kv-resume",
      { continueGeneration: true },
    )) as import("../src/types").ResumeResult;
    expect(result).toMatchObject({
      sessionId: "session-decode-kv-resume",
      recoveryMode: "kv",
      replayedTokens: 1,
      recoveredText: "user:KVdecode|token1||token2||token3||token4||token5|",
    });
    expect(pipeline.restoredCheckpointSeqLen).toBe(10);
  });

  test("resumeChatCompletion skips a corrupt newer checkpoint and restores an older one", async () => {
    const { engine, pipeline } = createEngineWithPipeline(7);
    pipeline.enableDecodeCheckpoint = true;
    pipeline.checkpointPageSize = 2;
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const sessionId = "session-corrupt-newest-checkpoint";
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      seed: 1234,
      messages: [{ role: "user", content: "KVdecode" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId,
          checkpointPrompt: false,
          checkpointIntervalTokens: 2,
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    for (let i = 0; i < 5; i++) {
      await iterator.next();
    }
    await engine.interruptGenerate();
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate records the abort and releases the lock
    }

    const kvRoot = `resume-root/sessions/${sessionId}/kv`;
    const checkpoints = await files.list(kvRoot);
    expect(checkpoints).toEqual([
      "checkpoint_00000000_00000010",
      "checkpoint_00000000_00000012",
    ]);
    const newestCheckpoint = checkpoints[1];
    const newestPath = `${kvRoot}/${newestCheckpoint}`;
    const pageGroupFile = (await files.list(newestPath)).find((name) =>
      name.endsWith(".wkv"),
    );
    expect(pageGroupFile).toBeDefined();
    const pageGroupPath = `${newestPath}/${pageGroupFile!}`;
    const pageGroupData = await files.read(pageGroupPath);
    expect(pageGroupData).toBeDefined();
    const corruptPageGroup = new Uint8Array(pageGroupData!);
    corruptPageGroup[0] ^= 1;
    await files.write(pageGroupPath, corruptPageGroup);
    const warn = jest.spyOn(log, "warn").mockImplementation(() => undefined);

    const result = (await engine.resumeChatCompletion(sessionId, {
      continueGeneration: true,
    })) as import("../src/types").ResumeResult;
    expect(result).toMatchObject({
      sessionId,
      recoveryMode: "kv",
      replayedTokens: 3,
      recoveredText:
        "user:KVdecode|token1||token2||token3||token4||token5||token6||token7|",
    });
    expect(pipeline.restoredCheckpointSeqLen).toBe(10);
    expect(warn).toHaveBeenCalledWith(
      `Ignoring invalid KV checkpoint ${newestCheckpoint}:`,
      expect.objectContaining({
        message: expect.stringContaining("CRC mismatch"),
      }),
    );
    warn.mockRestore();
  });

  test("resumeChatCompletion falls back to text-only when model is unavailable", async () => {
    const { engine } = createEngineWithPipeline(1);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    await engine.chatCompletion({
      model: MODEL_ID,
      messages: [{ role: "user", content: "Model mismatch" }],
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-model-mismatch",
        },
      },
    });

    const secondEngine = new MLCEngine();
    attachResumableStore(secondEngine, files);
    const result = (await secondEngine.resumeChatCompletion(
      "session-model-mismatch",
      { continueGeneration: true },
    )) as import("../src/types").ResumeResult;
    expect(result).toMatchObject({
      sessionId: "session-model-mismatch",
      recoveryMode: "text_only",
      recoveredText: "user:Model mismatch|token1|",
      replayedTokens: 0,
    });
  });

  test("listResumableSessions reports unsupported grammar replay as text-only", async () => {
    const { engine } = createEngineWithPipeline(5);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      messages: [{ role: "user", content: "Grammar persist" }],
      response_format: { type: "json_object" },
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-grammar",
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    await iterator.next();
    await engine.interruptGenerate();
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate records the abort and releases the lock
    }

    await expect(engine.listResumableSessions()).resolves.toEqual([
      expect.objectContaining({
        sessionId: "session-grammar",
        resumable: false,
        recoveryMode: "text_only",
        reason: "unsupported grammar replay; token replay unavailable",
      }),
    ]);
  });

  test("listResumableSessions reports custom logit processor replay as text-only", async () => {
    const { engine } = createEngineWithPipeline(5);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    (engine as any).logitProcessorRegistry = new Map([
      [
        MODEL_ID,
        {
          processLogits: (logits: Float32Array) => logits,
          processSampledToken: () => undefined,
          resetState: () => undefined,
        },
      ],
    ]);
    const iterable = (await engine.chatCompletion({
      model: MODEL_ID,
      messages: [{ role: "user", content: "Processor persist" }],
      stream: true,
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-logit-processor",
        },
      },
    })) as AsyncIterable<ChatCompletionChunk>;
    const iterator = iterable[Symbol.asyncIterator]();

    await iterator.next();
    await engine.interruptGenerate();
    while (!(await iterator.next()).done) {
      // drain the generator so asyncGenerate records the abort and releases the lock
    }

    await expect(engine.listResumableSessions()).resolves.toEqual([
      expect.objectContaining({
        sessionId: "session-logit-processor",
        resumable: false,
        recoveryMode: "text_only",
        reason: "custom LogitProcessor replay unsupported",
      }),
    ]);
  });

  test("listResumableSessions reports corrupted journals as text-only", async () => {
    const { engine } = createEngineWithPipeline(1);
    const files = new MemoryFileStore();
    attachResumableStore(engine, files);
    await engine.chatCompletion({
      model: MODEL_ID,
      messages: [{ role: "user", content: "Corrupt persist" }],
      extra_body: {
        resumable: {
          enabled: true,
          sessionId: "session-corrupt",
        },
      },
    });
    await files.append(
      "resume-root/sessions/session-corrupt/journal.bin",
      new Uint8Array([1, 2, 3]),
    );

    await expect(engine.listResumableSessions()).resolves.toEqual([
      expect.objectContaining({
        sessionId: "session-corrupt",
        resumable: false,
        recoveryMode: "text_only",
        reason:
          "journal scan stopped at partial_record; token replay unavailable",
      }),
    ]);
  });

  test("chatCompletion without specifying model when multiple loaded throws error", async () => {
    const engine = createEngineWithMultiplePipelines();
    await expect(
      engine.chatCompletion({
        // purposely omit model to trigger ambiguity
        model: undefined as any,
        messages: [{ role: "user", content: "Hello" }],
      }),
    ).rejects.toBeInstanceOf(UnclearModelToUseError);
  });

  test("embedding API uses mock pipeline and returns usage", async () => {
    const { engine } = createEngineWithEmbeddingPipeline();
    const request: EmbeddingCreateParams = {
      model: EMBED_MODEL_ID,
      input: "abc",
    };
    const response = await engine.embedding(request);
    expect(response.data).toHaveLength(1);
    expect(response.data[0].embedding).toEqual([0.1, 0.2, 0.3]);
    expect(response.usage?.prompt_tokens).toBeGreaterThan(0);
    expect(response.usage?.extra?.prefill_tokens_per_s).toBeGreaterThan(0);
  });
});
