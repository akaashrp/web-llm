import { OPFSFileStore } from "../src/resumable/opfs_file_store";
import {
  ResumableCheckpointWriter,
  readResumableCheckpointPayload,
} from "../src/resumable/checkpoint_writer";
import {
  JournalRecordType,
  appendJournalRecord,
  readJournalRecords,
} from "../src/resumable/journal";
import { ResumableSessionStore } from "../src/resumable/session_store";
import { test, expect } from "@jest/globals";

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
    const prev = this.files.get(normalized) ?? new Uint8Array();
    const chunk = bytes(data);
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
    if (!this.dirs.has(normalized) && childFiles.length === 0) {
      return;
    }
    if (
      opts?.recursive !== true &&
      (childFiles.length > 0 || childDirs.length > 0)
    ) {
      throw new Error("Directory is not empty");
    }
    this.dirs.delete(normalized);
    for (const key of childFiles) {
      this.files.delete(key);
    }
    for (const dir of childDirs) {
      this.dirs.delete(dir);
    }
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

  async lock(): Promise<() => void> {
    return () => undefined;
  }

  async tryLock(): Promise<() => void> {
    return () => undefined;
  }
}

const decoder = new TextDecoder();
const encoder = new TextEncoder();

function makeStore(): {
  files: MemoryFileStore;
  sessions: ResumableSessionStore;
  writer: ResumableCheckpointWriter;
} {
  const files = new MemoryFileStore();
  const sessions = new ResumableSessionStore(files, {
    rootPath: "resume-root",
    now: () => 1000,
  });
  return {
    files,
    sessions,
    writer: new ResumableCheckpointWriter(files, sessions, () => 1200),
  };
}

test("checkpoint writer commits files, metadata, and journal record", async () => {
  const { files, sessions, writer } = makeStore();

  const ref = await writer.writeCheckpoint({
    sessionId: "session-a",
    checkpointId: "checkpoint_00000000_00000004",
    processedSeqLen: 4,
    layoutHash: "layout-a",
    metadata: { source: "prompt" },
    pageGroups: [
      {
        groupId: 0,
        layerStart: 0,
        layerEnd: 1,
        data: new Uint8Array([1, 2, 3, 4]),
      },
    ],
    nextLogits: new Uint8Array([5, 6]),
  });

  expect(await files.list(ref.path)).toEqual([
    "complete",
    "group_000_layers_000_001_pages.wkv",
    "meta.json",
    "next_logits.f16",
  ]);
  const meta = JSON.parse(
    decoder.decode((await files.read(`${ref.path}/meta.json`))!),
  );
  expect(meta).toMatchObject({
    checkpointId: "checkpoint_00000000_00000004",
    processedSeqLen: 4,
    layoutHash: "layout-a",
    createdAtMs: 1200,
    metadata: { source: "prompt" },
    pageGroups: [
      {
        path: "group_000_layers_000_001_pages.wkv",
        bytes: 4,
        crc32c: expect.stringMatching(/^[0-9a-f]{8}$/),
      },
    ],
    nextLogits: {
      path: "next_logits.f16",
      bytes: 2,
      crc32c: expect.stringMatching(/^[0-9a-f]{8}$/),
    },
  });

  expect((await sessions.listCommittedCheckpoints("session-a"))[0]).toEqual(
    ref,
  );
  const payload = await readResumableCheckpointPayload(files, ref);
  expect(payload?.pageGroups[0].data).toEqual(new Uint8Array([1, 2, 3, 4]));
  expect(payload?.nextLogits?.data).toEqual(new Uint8Array([5, 6]));
  expect(
    (await sessions.getManifestRebuildInputs("session-a"))?.checkpoints,
  ).toEqual([ref]);

  const journal = await readJournalRecords(
    files,
    "resume-root/sessions/session-a/journal.bin",
  );
  expect(journal.records).toEqual([
    expect.objectContaining({
      type: JournalRecordType.CheckpointCommit,
      seqNo: 1,
      payload: expect.objectContaining({
        checkpointId: "checkpoint_00000000_00000004",
        processedSeqLen: 4,
        layoutHash: "layout-a",
      }),
    }),
  ]);
});

test("checkpoint without complete marker is ignored", async () => {
  const { files, sessions } = makeStore();
  const session = await sessions.createSession("session-a");
  const ref = sessions.getCheckpointRef("session-a", "checkpoint_incomplete");
  await files.write(`${ref.path}/meta.json`, encoder.encode("{}"));
  await appendJournalRecord(files, session.paths.journalPath, {
    type: JournalRecordType.CheckpointCommit,
    seqNo: 1,
    createdAtMs: 1000,
    payload: {
      checkpointId: ref.checkpointId,
      processedSeqLen: 4,
      path: ref.path,
    },
  });

  expect(await sessions.listCommittedCheckpoints("session-a")).toEqual([]);
  expect(await sessions.cleanupIncompleteCheckpoints("session-a")).toEqual([
    ref,
  ]);
});

test("checkpoint with complete marker but no journal commit is ignored", async () => {
  const { files, sessions } = makeStore();
  await sessions.createSession("session-a");
  const ref = sessions.getCheckpointRef("session-a", "checkpoint_complete");
  await files.write(`${ref.path}/complete`, encoder.encode(""));

  expect(await sessions.listCommittedCheckpoints("session-a")).toEqual([]);
  expect(
    (await sessions.getManifestRebuildInputs("session-a"))?.checkpoints,
  ).toEqual([]);
});
