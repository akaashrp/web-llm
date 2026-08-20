import { GenerationConfig } from "../config";
import { ReplayedGenerationToken } from "../llm_chat";
import { NormalizedResumableGenerationConfig } from "./generation";
import { OPFSFileStore } from "./opfs_file_store";
import {
  JournalRecord,
  JournalRecordType,
  JournalScanResult,
  readJournalRecords,
} from "./journal";
import { ResumableSessionHandle } from "./types";

export interface ResumableReplayState {
  sessionId: string;
  modelId: string;
  request?: unknown;
  promptTokenIds: number[];
  assistantPrefixTokenIds: number[];
  generatedTokens: ReplayedGenerationToken[];
  recoveredText: string;
  emittedTokens: number;
  processedSeqLen: number;
  generationConfig?: GenerationConfig;
  resumableConfig?: NormalizedResumableGenerationConfig;
  scanStoppedReason?: JournalScanResult["stoppedReason"];
  finished: boolean;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function booleanField(
  value: Record<string, unknown>,
  field: keyof NormalizedResumableGenerationConfig,
): boolean | undefined {
  const fieldValue = value[field];
  return typeof fieldValue === "boolean" ? fieldValue : undefined;
}

function positiveIntegerField(
  value: Record<string, unknown>,
  field: keyof NormalizedResumableGenerationConfig,
): number | undefined {
  const fieldValue = value[field];
  return Number.isInteger(fieldValue) && (fieldValue as number) > 0
    ? (fieldValue as number)
    : undefined;
}

function parseResumableConfig(
  value: unknown,
): NormalizedResumableGenerationConfig | undefined {
  if (!isObject(value) || value.enabled !== true) {
    return undefined;
  }
  const sessionId = value.sessionId;
  const checkpointIntervalTokens = positiveIntegerField(
    value,
    "checkpointIntervalTokens",
  );
  const checkpointPrompt = booleanField(value, "checkpointPrompt");
  const durabilityMode = value.durabilityMode;
  const strictPersistence = booleanField(value, "strictPersistence");
  const storeCheckpointLogits = booleanField(value, "storeCheckpointLogits");
  if (
    typeof sessionId !== "string" ||
    checkpointIntervalTokens === undefined ||
    checkpointPrompt === undefined ||
    (durabilityMode !== "exact" && durabilityMode !== "relaxed") ||
    strictPersistence === undefined ||
    storeCheckpointLogits === undefined
  ) {
    return undefined;
  }
  return {
    enabled: true,
    sessionId,
    checkpointIntervalTokens,
    checkpointPrompt,
    durabilityMode,
    strictPersistence,
    storeCheckpointLogits,
  };
}

export function getGenerationConfig(
  record: JournalRecord,
): GenerationConfig | undefined {
  if (record.type !== JournalRecordType.GenerationConfig) {
    return undefined;
  }
  const config = record.payload.config;
  if (!isObject(config)) {
    return undefined;
  }
  const generationConfig = config.generationConfig;
  return isObject(generationConfig)
    ? (generationConfig as GenerationConfig)
    : (config as GenerationConfig);
}

export function getResumableGenerationConfig(
  record: JournalRecord,
): NormalizedResumableGenerationConfig | undefined {
  if (record.type !== JournalRecordType.GenerationConfig) {
    return undefined;
  }
  const config = record.payload.config;
  if (!isObject(config)) {
    return undefined;
  }
  return parseResumableConfig(config.resumable);
}

export function hasUnsupportedGrammarReplay(
  genConfig?: GenerationConfig,
): boolean {
  const type = genConfig?.response_format?.type;
  return (
    type === "json_object" || type === "grammar" || type === "structural_tag"
  );
}

function summarizeReplayState(
  session: ResumableSessionHandle,
  scan: JournalScanResult,
): ResumableReplayState {
  let modelId = session.manifest?.modelId ?? "";
  let request: unknown;
  let promptTokenIds: number[] = [];
  let assistantPrefixTokenIds: number[] = [];
  let generationConfig: GenerationConfig | undefined;
  let resumableConfig: NormalizedResumableGenerationConfig | undefined;
  let processedSeqLen = 0;
  let finished = false;
  const generatedTokens: ReplayedGenerationToken[] = [];

  for (const record of scan.records) {
    switch (record.type) {
      case JournalRecordType.SessionBegin:
        if (modelId === "") {
          modelId = record.payload.modelId ?? "";
        }
        request = record.payload.request;
        break;
      case JournalRecordType.PromptTokens:
        promptTokenIds = [...record.payload.tokenIds];
        processedSeqLen = Math.max(processedSeqLen, promptTokenIds.length);
        break;
      case JournalRecordType.AssistantPrefixTokens:
        assistantPrefixTokenIds = [...record.payload.tokenIds];
        break;
      case JournalRecordType.GenerationConfig:
        generationConfig = getGenerationConfig(record);
        resumableConfig = getResumableGenerationConfig(record);
        break;
      case JournalRecordType.GeneratedToken:
        generatedTokens.push({
          globalTokenPos: record.payload.globalTokenPos,
          tokenId: record.payload.tokenId,
          textDelta: record.payload.textDelta,
          rngState: record.payload.rngState,
        });
        processedSeqLen = Math.max(
          processedSeqLen,
          record.payload.globalTokenPos + 1,
        );
        break;
      case JournalRecordType.GenerationFinished:
        finished = true;
        break;
    }
  }

  return {
    sessionId: session.sessionId,
    modelId,
    request,
    promptTokenIds,
    assistantPrefixTokenIds,
    generatedTokens,
    recoveredText: generatedTokens.map((token) => token.textDelta).join(""),
    emittedTokens: generatedTokens.length,
    processedSeqLen,
    generationConfig,
    resumableConfig,
    scanStoppedReason: scan.stoppedReason,
    finished,
  };
}

export async function readResumableReplayState(
  files: OPFSFileStore,
  session: ResumableSessionHandle,
): Promise<ResumableReplayState> {
  const scan = await readJournalRecords(files, session.paths.journalPath);
  return summarizeReplayState(session, scan);
}
