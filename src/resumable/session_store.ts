import { OPFSFileStore } from "./opfs_file_store";
import { JournalRecordType, readJournalRecords } from "./journal";
import {
  RESUMABLE_STORE_ROOT,
  ResumableCheckpointRef,
  ResumableSessionHandle,
  ResumableSessionManifest,
  ResumableSessionManifestInit,
  ResumableSessionPaths,
  SessionManifestRebuildInputs,
} from "./types";

const MANIFEST_FILE = "manifest.json";
const JOURNAL_FILE = "journal.bin";
const LOCK_FILE = "lock";
const KV_DIR = "kv";
const COMPLETE_FILE = "complete";

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

export interface ResumableSessionStoreOptions {
  rootPath?: string;
  now?: () => number;
}

function joinPath(...parts: string[]): string {
  return parts
    .flatMap((part) => part.split("/"))
    .filter((part) => part !== "")
    .join("/");
}

function encodeText(data: string): ArrayBuffer {
  const bytes = textEncoder.encode(data);
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy.buffer;
}

function isSafeSegment(value: string): boolean {
  return (
    value !== "" &&
    value !== "." &&
    value !== ".." &&
    !value.includes("/") &&
    !value.includes("\0")
  );
}

function assertSafeSegment(value: string, label: string): void {
  if (!isSafeSegment(value)) {
    throw new Error(`Invalid resumable ${label}: ${value}`);
  }
}

function parseManifest(
  data: ArrayBuffer | undefined,
  sessionId: string,
): ResumableSessionManifest | undefined {
  if (data === undefined) {
    return undefined;
  }
  try {
    const parsed = JSON.parse(
      textDecoder.decode(data),
    ) as Partial<ResumableSessionManifest>;
    if (
      parsed.sessionId === sessionId &&
      typeof parsed.createdAtMs === "number" &&
      typeof parsed.updatedAtMs === "number"
    ) {
      return parsed as ResumableSessionManifest;
    }
    return undefined;
  } catch {
    return undefined;
  }
}

export class ResumableSessionStore {
  private readonly rootPath: string;
  private readonly now: () => number;

  constructor(
    private readonly files: OPFSFileStore,
    opts: ResumableSessionStoreOptions = {},
  ) {
    this.rootPath = opts.rootPath ?? RESUMABLE_STORE_ROOT;
    this.now = opts.now ?? Date.now;
  }

  getSessionPaths(sessionId: string): ResumableSessionPaths {
    assertSafeSegment(sessionId, "session id");
    const sessionDir = joinPath(this.rootPath, "sessions", sessionId);
    return {
      sessionDir,
      manifestPath: joinPath(sessionDir, MANIFEST_FILE),
      journalPath: joinPath(sessionDir, JOURNAL_FILE),
      lockPath: joinPath(sessionDir, LOCK_FILE),
      kvDir: joinPath(sessionDir, KV_DIR),
    };
  }

  getCheckpointRef(
    sessionId: string,
    checkpointId: string,
  ): ResumableCheckpointRef {
    assertSafeSegment(checkpointId, "checkpoint id");
    const path = joinPath(this.getSessionPaths(sessionId).kvDir, checkpointId);
    return {
      checkpointId,
      path,
      completePath: joinPath(path, COMPLETE_FILE),
    };
  }

  async createSession(
    sessionId: string,
    init: ResumableSessionManifestInit = {},
  ): Promise<ResumableSessionHandle> {
    const paths = this.getSessionPaths(sessionId);
    await this.files.mkdir(paths.sessionDir);
    const existing = await this.readManifest(sessionId);
    const now = this.now();
    const manifest: ResumableSessionManifest = {
      ...existing,
      ...init,
      sessionId,
      createdAtMs: existing?.createdAtMs ?? init.createdAtMs ?? now,
      updatedAtMs: init.updatedAtMs ?? now,
    };
    await this.tryWriteManifest(manifest);
    return { sessionId, paths, manifest };
  }

  async createNewSession(
    sessionId: string,
    init: ResumableSessionManifestInit = {},
  ): Promise<ResumableSessionHandle> {
    const paths = this.getSessionPaths(sessionId);
    await this.files.mkdir(paths.sessionDir);
    const [manifestData, journalData, entries] = await Promise.all([
      this.files.read(paths.manifestPath),
      this.files.read(paths.journalPath),
      this.files.list(paths.sessionDir),
    ]);
    const persistedEntries = entries.filter((entry) => entry !== LOCK_FILE);
    if (
      manifestData !== undefined ||
      journalData !== undefined ||
      persistedEntries.length > 0
    ) {
      throw new Error(`Resumable session already exists: ${sessionId}`);
    }

    const now = this.now();
    const manifest: ResumableSessionManifest = {
      ...init,
      sessionId,
      createdAtMs: init.createdAtMs ?? now,
      updatedAtMs: init.updatedAtMs ?? now,
    };
    await this.tryWriteManifest(manifest);
    return { sessionId, paths, manifest };
  }

  async openSession(
    sessionId: string,
  ): Promise<ResumableSessionHandle | undefined> {
    const paths = this.getSessionPaths(sessionId);
    const manifest = await this.readManifest(sessionId);
    const entries = await this.files.list(paths.sessionDir);
    if (manifest === undefined && entries.length === 0) {
      return undefined;
    }
    return { sessionId, paths, manifest };
  }

  async listSessions(): Promise<ResumableSessionHandle[]> {
    const sessionRoot = joinPath(this.rootPath, "sessions");
    const sessionIds = (await this.files.list(sessionRoot))
      .filter(isSafeSegment)
      .sort();
    const sessions: ResumableSessionHandle[] = [];
    for (const sessionId of sessionIds) {
      const session = await this.openSession(sessionId);
      if (session !== undefined) {
        sessions.push(session);
      }
    }
    return sessions;
  }

  async deleteSession(sessionId: string): Promise<void> {
    await this.files.remove(this.getSessionPaths(sessionId).sessionDir, {
      recursive: true,
    });
  }

  async deleteKV(sessionId: string): Promise<void> {
    await this.files.remove(this.getSessionPaths(sessionId).kvDir, {
      recursive: true,
    });
  }

  async readManifest(
    sessionId: string,
  ): Promise<ResumableSessionManifest | undefined> {
    const data = await this.files.read(
      this.getSessionPaths(sessionId).manifestPath,
    );
    return parseManifest(data, sessionId);
  }

  async tryWriteManifest(manifest: ResumableSessionManifest): Promise<boolean> {
    try {
      await this.files.write(
        this.getSessionPaths(manifest.sessionId).manifestPath,
        encodeText(`${JSON.stringify(manifest)}\n`),
      );
      return true;
    } catch {
      return false;
    }
  }

  async listCommittedCheckpoints(
    sessionId: string,
  ): Promise<ResumableCheckpointRef[]> {
    const paths = this.getSessionPaths(sessionId);
    const journal = await readJournalRecords(this.files, paths.journalPath);
    const committedIds = new Set<string>();
    for (const record of journal.records) {
      if (record.type === JournalRecordType.CheckpointCommit) {
        committedIds.add(record.payload.checkpointId);
      }
    }
    const checkpointIds = (await this.files.list(paths.kvDir))
      .filter(isSafeSegment)
      .sort();
    const refs: ResumableCheckpointRef[] = [];
    for (const checkpointId of checkpointIds) {
      const ref = this.getCheckpointRef(sessionId, checkpointId);
      if (
        committedIds.has(checkpointId) &&
        (await this.files.read(ref.completePath)) !== undefined
      ) {
        refs.push(ref);
      }
    }
    return refs;
  }

  async cleanupIncompleteCheckpoints(
    sessionId?: string,
  ): Promise<ResumableCheckpointRef[]> {
    const sessionIds =
      sessionId === undefined
        ? (await this.listSessions()).map((session) => session.sessionId)
        : [sessionId];
    const removed: ResumableCheckpointRef[] = [];
    for (const id of sessionIds) {
      const checkpointIds = (
        await this.files.list(this.getSessionPaths(id).kvDir)
      )
        .filter(isSafeSegment)
        .sort();
      for (const checkpointId of checkpointIds) {
        const ref = this.getCheckpointRef(id, checkpointId);
        if ((await this.files.read(ref.completePath)) === undefined) {
          await this.files.remove(ref.path, { recursive: true });
          removed.push(ref);
        }
      }
    }
    return removed;
  }

  async getManifestRebuildInputs(
    sessionId: string,
  ): Promise<SessionManifestRebuildInputs | undefined> {
    const session = await this.openSession(sessionId);
    if (session === undefined) {
      return undefined;
    }
    const hasJournal =
      (await this.files.read(session.paths.journalPath)) !== undefined;
    return {
      sessionId,
      manifest: session.manifest,
      journalPath: session.paths.journalPath,
      hasJournal,
      checkpoints: await this.listCommittedCheckpoints(sessionId),
    };
  }
}
