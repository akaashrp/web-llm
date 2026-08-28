import { OPFSFileStore } from "../src/resumable/opfs_file_store";
import {
  JournalRecord,
  JournalRecordType,
  appendJournalRecord,
  decodeJournalRecordAt,
  encodeJournalRecord,
  readJournalRecords,
  scanJournalRecords,
} from "../src/resumable/journal";
import {
  ResumableGenerationJournal,
  normalizeResumableGenerationConfig,
} from "../src/resumable/generation";
import { readResumableReplayState } from "../src/resumable/replay";
import { ResumableSessionStore } from "../src/resumable/session_store";
import { test, expect } from "@jest/globals";

const HEADER_SIZE = 30;

function bytes(data: BufferSource): Uint8Array<ArrayBuffer> {
  const view = ArrayBuffer.isView(data)
    ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
    : new Uint8Array(data);
  return new Uint8Array(view);
}

function concat(...parts: ArrayBuffer[]): ArrayBuffer {
  const total = parts.reduce((sum, part) => sum + part.byteLength, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    out.set(new Uint8Array(part), offset);
    offset += part.byteLength;
  }
  return out.buffer;
}

class MemoryFileStore implements OPFSFileStore {
  private readonly files = new Map<string, Uint8Array<ArrayBuffer>>();
  private readonly locks = new Set<string>();

  async read(path: string): Promise<ArrayBuffer | undefined> {
    const data = this.files.get(path);
    return data === undefined ? undefined : new Uint8Array(data).buffer;
  }

  async write(path: string, data: BufferSource): Promise<void> {
    this.files.set(path, bytes(data));
  }

  async append(path: string, data: BufferSource): Promise<void> {
    const prev = this.files.get(path) ?? new Uint8Array();
    const chunk = bytes(data);
    const next = new Uint8Array(prev.byteLength + chunk.byteLength);
    next.set(prev);
    next.set(chunk, prev.byteLength);
    this.files.set(path, next);
  }

  async remove(path: string): Promise<void> {
    this.files.delete(path);
  }

  async list(): Promise<string[]> {
    return [];
  }

  async mkdir(): Promise<void> {}

  async lock(path: string): Promise<() => void> {
    const release = await this.tryLock(path);
    if (release === undefined) {
      throw new Error(`Unable to acquire lock: ${path}`);
    }
    return release;
  }

  async tryLock(path: string): Promise<(() => void) | undefined> {
    if (this.locks.has(path)) {
      return undefined;
    }
    this.locks.add(path);
    return () => {
      this.locks.delete(path);
    };
  }
}

function record<T extends JournalRecord>(record: T): T {
  return record;
}

const records: JournalRecord[] = [
  record({
    type: JournalRecordType.SessionBegin,
    seqNo: 1,
    createdAtMs: 100,
    payload: { sessionId: "s1", modelId: "m1", request: { temperature: 0.7 } },
  }),
  record({
    type: JournalRecordType.PromptTokens,
    seqNo: 2,
    createdAtMs: 101,
    payload: { tokenIds: [1, 2, 3], text: "prompt" },
  }),
  record({
    type: JournalRecordType.AssistantPrefixTokens,
    seqNo: 3,
    createdAtMs: 102,
    payload: { tokenIds: [4, 5], text: "<think></think>" },
  }),
  record({
    type: JournalRecordType.GenerationConfig,
    seqNo: 4,
    createdAtMs: 103,
    payload: { config: { max_tokens: 8, temperature: 0 } },
  }),
  record({
    type: JournalRecordType.GeneratedToken,
    seqNo: 5,
    createdAtMs: 104,
    payload: {
      globalTokenPos: 9,
      tokenId: 42,
      textDelta: "hi",
      rngState: 7,
    },
  }),
  record({
    type: JournalRecordType.CheckpointCommit,
    seqNo: 6,
    createdAtMs: 105,
    payload: {
      checkpointId: "checkpoint_000",
      processedSeqLen: 512,
      layoutHash: "abc",
    },
  }),
  record({
    type: JournalRecordType.GenerationFinished,
    seqNo: 7,
    createdAtMs: 106,
    payload: { finishReason: "stop", emittedTokens: 2 },
  }),
  record({
    type: JournalRecordType.GenerationAborted,
    seqNo: 8,
    createdAtMs: 107,
    payload: { reason: "user" },
  }),
  record({
    type: JournalRecordType.EngineError,
    seqNo: 9,
    createdAtMs: 108,
    payload: { message: "boom", name: "Error" },
  }),
];

test("journal records round-trip through binary codec", () => {
  for (const original of records) {
    const encoded = encodeJournalRecord(original);
    const decoded = decodeJournalRecordAt(encoded);
    expect(decoded.record).toEqual(original);
    expect(decoded.nextOffset).toBe(encoded.byteLength);
  }
});

test("journal scanner returns all valid records", () => {
  const data = concat(...records.map(encodeJournalRecord));
  const result = scanJournalRecords(data);

  expect(result.stoppedReason).toBeUndefined();
  expect(result.validBytes).toBe(data.byteLength);
  expect(result.records).toEqual(records);
});

test("journal scanner ignores partial trailing record", () => {
  const first = encodeJournalRecord(records[0]);
  const partialSecond = encodeJournalRecord(records[1]).slice(
    0,
    HEADER_SIZE + 3,
  );
  const result = scanJournalRecords(concat(first, partialSecond));

  expect(result.records).toEqual([records[0]]);
  expect(result.validBytes).toBe(first.byteLength);
  expect(result.stoppedReason).toBe("partial_record");
});

test("journal scanner stops at CRC mismatch", () => {
  const first = encodeJournalRecord(records[0]);
  const second = encodeJournalRecord(records[1]);
  const data = new Uint8Array(concat(first, second));
  data[first.byteLength + HEADER_SIZE] ^= 1;

  const result = scanJournalRecords(data.buffer);

  expect(result.records).toEqual([records[0]]);
  expect(result.validBytes).toBe(first.byteLength);
  expect(result.stoppedReason).toBe("crc_mismatch");
});

test("journal append and read helpers use OPFS file store", async () => {
  const store = new MemoryFileStore();

  await appendJournalRecord(store, "journal.bin", records[0]);
  await appendJournalRecord(store, "journal.bin", records[1]);

  const result = await readJournalRecords(store, "journal.bin");
  expect(result.records).toEqual([records[0], records[1]]);
  expect(await readJournalRecords(store, "missing.bin")).toEqual({
    records: [],
    validBytes: 0,
  });
});

test("resumable generation journal rejects active writers and session reuse", async () => {
  const store = new MemoryFileStore();
  const sessions = new ResumableSessionStore(store, {
    rootPath: "resume-root",
  });
  const config = normalizeResumableGenerationConfig({
    enabled: true,
    sessionId: "session-a",
  })!;
  const init = {
    modelId: "model-a",
    request: { messages: [] },
    promptTokenIds: [1, 2, 3],
    assistantPrefixTokenIds: [],
    generationConfig: {},
  };
  const first = new ResumableGenerationJournal(store, sessions, config);
  const second = new ResumableGenerationJournal(store, sessions, config);

  await first.begin(init);
  await expect(second.begin(init)).rejects.toThrow(
    "Resumable session is already active: session-a",
  );

  await first.close();
  await expect(second.begin(init)).rejects.toThrow(
    "Resumable session already exists: session-a",
  );
});

test("replay rejects journals containing multiple session beginnings", async () => {
  const store = new MemoryFileStore();
  const sessions = new ResumableSessionStore(store, {
    rootPath: "resume-root",
  });
  const session = await sessions.createSession("session-a");
  const sessionBegin = record({
    type: JournalRecordType.SessionBegin,
    seqNo: 1,
    createdAtMs: 100,
    payload: { sessionId: "session-a", modelId: "model-a" },
  });
  await appendJournalRecord(store, session.paths.journalPath, sessionBegin);
  await appendJournalRecord(store, session.paths.journalPath, {
    ...sessionBegin,
    seqNo: 2,
  });

  await expect(readResumableReplayState(store, session)).rejects.toThrow(
    "Resumable session session-a contains multiple session-begin records.",
  );
});
