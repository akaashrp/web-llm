import { ResumeProbeResult } from "../types";
import { OPFSFileStore } from "./opfs_file_store";
import {
  JournalRecord,
  JournalRecordType,
  JournalScanResult,
  readJournalRecords,
} from "./journal";
import {
  getGenerationConfig,
  getResumableGenerationConfig,
  hasUnsupportedGrammarReplay,
} from "./replay";
import { ResumableSessionHandle } from "./types";

interface JournalSummary {
  modelId: string;
  hasPromptTokens: boolean;
  hasGenerationConfig: boolean;
  hasResumableGenerationConfig: boolean;
  emittedTokens: number;
  processedSeqLen: number;
  finished: boolean;
  aborted: boolean;
  engineErrorMessage?: string;
  hasUnsupportedGrammarReplay: boolean;
}

function none(
  sessionId: string,
  modelId: string,
  emittedTokens: number,
  processedSeqLen: number,
  reason: string,
): ResumeProbeResult {
  return {
    sessionId,
    resumable: false,
    reason,
    modelId,
    emittedTokens,
    processedSeqLen,
    recoveryMode: "none",
  };
}

function textOnly(
  sessionId: string,
  modelId: string,
  emittedTokens: number,
  processedSeqLen: number,
  reason: string,
): ResumeProbeResult {
  return {
    sessionId,
    resumable: false,
    reason,
    modelId,
    emittedTokens,
    processedSeqLen,
    recoveryMode: emittedTokens > 0 ? "text_only" : "none",
  };
}

function tokenReplay(
  sessionId: string,
  summary: JournalSummary,
  reason?: string,
): ResumeProbeResult {
  return {
    sessionId,
    resumable: true,
    reason,
    modelId: summary.modelId,
    emittedTokens: summary.emittedTokens,
    processedSeqLen: summary.processedSeqLen,
    recoveryMode: "token_replay",
  };
}

function summarizeJournal(
  session: ResumableSessionHandle,
  records: JournalRecord[],
): JournalSummary {
  let modelId = session.manifest?.modelId ?? "";
  let promptTokenCount = 0;
  let hasPromptTokens = false;
  let hasGenerationConfig = false;
  let hasResumableGenerationConfig = false;
  let emittedTokens = 0;
  let processedSeqLen = 0;
  let finished = false;
  let aborted = false;
  let engineErrorMessage: string | undefined;
  let unsupportedGrammarReplay = false;

  for (const record of records) {
    switch (record.type) {
      case JournalRecordType.SessionBegin:
        if (modelId === "") {
          modelId = record.payload.modelId ?? "";
        }
        break;
      case JournalRecordType.PromptTokens:
        hasPromptTokens = true;
        promptTokenCount = record.payload.tokenIds.length;
        processedSeqLen = Math.max(processedSeqLen, promptTokenCount);
        break;
      case JournalRecordType.GenerationConfig:
        hasGenerationConfig = true;
        hasResumableGenerationConfig ||=
          getResumableGenerationConfig(record) !== undefined;
        if (hasUnsupportedGrammarReplay(getGenerationConfig(record))) {
          unsupportedGrammarReplay = true;
        }
        break;
      case JournalRecordType.GeneratedToken:
        emittedTokens++;
        processedSeqLen = Math.max(
          processedSeqLen,
          record.payload.globalTokenPos + 1,
        );
        break;
      case JournalRecordType.GenerationFinished:
        finished = true;
        break;
      case JournalRecordType.GenerationAborted:
        aborted = true;
        break;
      case JournalRecordType.EngineError:
        engineErrorMessage = record.payload.message;
        break;
    }
  }

  return {
    modelId,
    hasPromptTokens,
    hasGenerationConfig,
    hasResumableGenerationConfig,
    emittedTokens,
    processedSeqLen,
    finished,
    aborted,
    engineErrorMessage,
    hasUnsupportedGrammarReplay: unsupportedGrammarReplay,
  };
}

function probeFromScan(
  session: ResumableSessionHandle,
  scan: JournalScanResult,
): ResumeProbeResult {
  const summary = summarizeJournal(session, scan.records);
  if (scan.records.length === 0) {
    return none(
      session.sessionId,
      summary.modelId,
      0,
      0,
      scan.stoppedReason === undefined
        ? "missing journal records"
        : `journal scan stopped at ${scan.stoppedReason}`,
    );
  }
  if (scan.stoppedReason !== undefined) {
    return textOnly(
      session.sessionId,
      summary.modelId,
      summary.emittedTokens,
      summary.processedSeqLen,
      `journal scan stopped at ${scan.stoppedReason}; token replay unavailable`,
    );
  }
  if (summary.finished) {
    return none(
      session.sessionId,
      summary.modelId,
      summary.emittedTokens,
      summary.processedSeqLen,
      "generation finished",
    );
  }
  if (!summary.hasPromptTokens) {
    return textOnly(
      session.sessionId,
      summary.modelId,
      summary.emittedTokens,
      summary.processedSeqLen,
      "missing prompt token record; token replay unavailable",
    );
  }
  if (!summary.hasGenerationConfig) {
    return textOnly(
      session.sessionId,
      summary.modelId,
      summary.emittedTokens,
      summary.processedSeqLen,
      "missing generation config record; token replay unavailable",
    );
  }
  if (!summary.hasResumableGenerationConfig) {
    return textOnly(
      session.sessionId,
      summary.modelId,
      summary.emittedTokens,
      summary.processedSeqLen,
      "missing resumable generation config; continuation unavailable",
    );
  }
  if (summary.hasUnsupportedGrammarReplay) {
    return textOnly(
      session.sessionId,
      summary.modelId,
      summary.emittedTokens,
      summary.processedSeqLen,
      "unsupported grammar replay; token replay unavailable",
    );
  }
  if (summary.aborted) {
    return tokenReplay(session.sessionId, summary, "generation aborted");
  }
  if (summary.engineErrorMessage !== undefined) {
    return tokenReplay(
      session.sessionId,
      summary,
      `engine error: ${summary.engineErrorMessage}`,
    );
  }
  return tokenReplay(session.sessionId, summary, "generation incomplete");
}

export async function probeResumableSession(
  files: OPFSFileStore,
  session: ResumableSessionHandle,
): Promise<ResumeProbeResult> {
  const scan = await readJournalRecords(files, session.paths.journalPath);
  return probeFromScan(session, scan);
}
