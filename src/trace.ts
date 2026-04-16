export type TraceContext =
  | "main"
  | "decode_worker"
  | "io_worker"
  | "ui_worker"
  | string;

export type TraceLevel = "major" | "verbose";
export type TraceDevtoolsMode = "off" | "major" | "all";
export type TraceLane = "cpu" | "gpu";

export type TraceValue = string | number | boolean | null;
export type TraceMeta = Record<string, TraceValue>;

export interface TraceEvent {
  abs_ts_ms: number;
  ctx: TraceContext;
  phase: string;
  step?: number | string;
  request_id?: string;
  session_id?: string;
  meta?: TraceMeta;
  lane?: TraceLane;
  seq: number;
}

export interface TraceDrainOptions {
  clear?: boolean;
  max_events?: number;
}

export interface TraceEventOptions {
  level?: TraceLevel;
  lane?: TraceLane;
  step?: number | string;
  request_id?: string;
  session_id?: string;
  meta?: TraceMeta;
  ctx?: TraceContext;
}

export interface TraceRequestConfig {
  enabled: boolean;
  level: TraceLevel;
  devtools: TraceDevtoolsMode;
  request_id?: string;
  session_id?: string;
  enable_gpu_timestamps?: boolean;
}

interface RuntimeTracePayload {
  phase: string;
  level?: TraceLevel;
  lane?: TraceLane;
  step?: number | string;
  request_id?: string;
  session_id?: string;
  meta?: TraceMeta;
  abs_ts_ms?: number;
  ctx?: TraceContext;
}

interface RuntimeTraceState {
  enabled: boolean;
  level: TraceLevel;
  devtools: TraceDevtoolsMode;
  ctx: TraceContext;
  step?: number | string;
  request_id?: string;
  session_id?: string;
  enable_gpu_timestamps?: boolean;
}

interface TraceSpan {
  seq: number;
  phase: string;
  level: TraceLevel;
  lane: TraceLane;
  ctx: TraceContext;
  step?: number | string;
  request_id?: string;
  session_id?: string;
  meta?: TraceMeta;
  start_abs_ts_ms: number;
  start_mark?: string;
  end_mark?: string;
}

declare global {
  interface Window {
    __WEBLLM_TRACE_COLLECTOR__?: TraceCollector;
    __WEBLLM_TRACE_RUNTIME_PUSH__?: (payload: RuntimeTracePayload) => void;
    __WEBLLM_TRACE_RUNTIME_STATE__?: RuntimeTraceState;
  }

  interface WorkerGlobalScope {
    __WEBLLM_TRACE_COLLECTOR__?: TraceCollector;
    __WEBLLM_TRACE_RUNTIME_PUSH__?: (payload: RuntimeTracePayload) => void;
    __WEBLLM_TRACE_RUNTIME_STATE__?: RuntimeTraceState;
  }
}

function nowAbsMs(): number {
  return performance.timeOrigin + performance.now();
}

function makeLabel(prefix: string, phase: string, seq: number): string {
  return `${prefix}:${phase}:${seq}`;
}

function shouldEmitTimeStamp(
  devtools: TraceDevtoolsMode,
  level: TraceLevel,
): boolean {
  if (devtools === "off") {
    return false;
  }
  if (devtools === "all") {
    return true;
  }
  return level === "major";
}

function emitTimeStamp(label: string): void {
  const ts = (console as any).timeStamp;
  if (typeof ts === "function") {
    ts.call(console, label);
  }
}

export class TraceCollector {
  private events: TraceEvent[] = [];
  private capacity = 50000;
  private seq = 0;
  private context: TraceContext = "main";
  private runtimeStepScopes: Array<{ id: number; step?: number | string }> = [];
  private nextRuntimeStepScopeId = 1;
  private activeRequest: TraceRequestConfig = {
    enabled: false,
    level: "major",
    devtools: "off",
  };

  constructor() {
    this.installRuntimeBridge();
    this.updateRuntimeState();
  }

  setContext(ctx: TraceContext) {
    this.context = ctx;
    this.updateRuntimeState();
  }

  getContext(): TraceContext {
    return this.context;
  }

  withRuntimeStep<T>(step: number | string | undefined, fn: () => T): T;
  withRuntimeStep<T>(
    step: number | string | undefined,
    fn: () => Promise<T>,
  ): Promise<T>;
  withRuntimeStep<T>(
    step: number | string | undefined,
    fn: () => T | Promise<T>,
  ): T | Promise<T> {
    const scopeId = this.enterRuntimeStepScope(step);
    try {
      const result = fn();
      if (
        result !== null &&
        typeof result === "object" &&
        typeof (result as Promise<T>).then === "function"
      ) {
        return (result as Promise<T>).finally(() => {
          this.exitRuntimeStepScope(scopeId);
        });
      }
      this.exitRuntimeStepScope(scopeId);
      return result as T;
    } catch (err) {
      this.exitRuntimeStepScope(scopeId);
      throw err;
    }
  }

  beginRequest(config: Partial<TraceRequestConfig>): TraceRequestConfig {
    const prev = { ...this.activeRequest };
    this.activeRequest = {
      enabled: config.enabled ?? false,
      level: config.level ?? "major",
      devtools: config.devtools ?? "off",
      request_id: config.request_id,
      session_id: config.session_id,
      enable_gpu_timestamps: config.enable_gpu_timestamps ?? false,
    };
    this.updateRuntimeState();
    return prev;
  }

  endRequest(prev: TraceRequestConfig) {
    this.activeRequest = prev;
    this.updateRuntimeState();
  }

  setCapacity(capacity: number) {
    this.capacity = Math.max(1024, capacity);
    if (this.events.length > this.capacity) {
      this.events.splice(0, this.events.length - this.capacity);
    }
  }

  getRequestConfig(): TraceRequestConfig {
    return { ...this.activeRequest };
  }

  isEnabled(level: TraceLevel = "major"): boolean {
    if (!this.activeRequest.enabled) {
      return false;
    }
    if (this.activeRequest.level === "major" && level === "verbose") {
      return false;
    }
    return true;
  }

  instant(phase: string, options: TraceEventOptions = {}): void {
    const level = options.level ?? "major";
    if (!this.isEnabled(level)) {
      return;
    }
    const request_id = options.request_id ?? this.activeRequest.request_id;
    const session_id = options.session_id ?? this.activeRequest.session_id;
    const ctx = options.ctx ?? this.context;
    const seq = this.nextSeq();
    const abs_ts_ms = nowAbsMs();
    const event: TraceEvent = {
      abs_ts_ms,
      ctx,
      phase,
      step: options.step,
      request_id,
      session_id,
      meta: options.meta,
      lane: options.lane ?? "cpu",
      seq,
    };
    this.pushEvent(event);

    if (shouldEmitTimeStamp(this.activeRequest.devtools, level)) {
      emitTimeStamp(`${phase}#${seq}`);
    }
  }

  beginSpan(phase: string, options: TraceEventOptions = {}): TraceSpan | null {
    const level = options.level ?? "major";
    if (!this.isEnabled(level)) {
      return null;
    }

    const request_id = options.request_id ?? this.activeRequest.request_id;
    const session_id = options.session_id ?? this.activeRequest.session_id;
    const lane = options.lane ?? "cpu";
    const ctx = options.ctx ?? this.context;
    const seq = this.nextSeq();
    const start_abs_ts_ms = nowAbsMs();

    const start_mark = makeLabel("webllm:start", phase, seq);
    const end_mark = makeLabel("webllm:end", phase, seq);
    performance.mark(start_mark);

    if (shouldEmitTimeStamp(this.activeRequest.devtools, level)) {
      emitTimeStamp(`${phase}:start#${seq}`);
    }

    return {
      seq,
      phase,
      level,
      lane,
      step: options.step,
      request_id,
      session_id,
      ctx,
      meta: options.meta,
      start_abs_ts_ms,
      start_mark,
      end_mark,
    };
  }

  endSpan(
    span: TraceSpan | null,
    extra: Partial<TraceEventOptions> = {},
  ): void {
    if (span === null) {
      return;
    }

    const abs_ts_ms = nowAbsMs();
    if (span.end_mark !== undefined) {
      performance.mark(span.end_mark);
      performance.measure(makeLabel("webllm:measure", span.phase, span.seq), {
        start: span.start_mark,
        end: span.end_mark,
      });
      performance.clearMarks(span.start_mark);
      performance.clearMarks(span.end_mark);
      performance.clearMeasures(
        makeLabel("webllm:measure", span.phase, span.seq),
      );
    }

    if (shouldEmitTimeStamp(this.activeRequest.devtools, span.level)) {
      emitTimeStamp(`${span.phase}:end#${span.seq}`);
    }

    const mergedMeta: TraceMeta = {
      ...(span.meta ?? {}),
      ...(extra.meta ?? {}),
      duration_ms: abs_ts_ms - span.start_abs_ts_ms,
    };

    const event: TraceEvent = {
      abs_ts_ms,
      ctx: extra.ctx ?? span.ctx,
      phase: span.phase,
      step: extra.step ?? span.step,
      request_id: extra.request_id ?? span.request_id,
      session_id: extra.session_id ?? span.session_id,
      meta: mergedMeta,
      lane: extra.lane ?? span.lane,
      seq: span.seq,
    };
    this.pushEvent(event);
  }

  ingestRuntimeEvent(payload: RuntimeTracePayload): void {
    const level = payload.level ?? "verbose";
    if (!this.isEnabled(level)) {
      return;
    }

    const event: TraceEvent = {
      abs_ts_ms: payload.abs_ts_ms ?? nowAbsMs(),
      ctx: payload.ctx ?? this.context,
      phase: payload.phase,
      step: payload.step,
      request_id: payload.request_id ?? this.activeRequest.request_id,
      session_id: payload.session_id ?? this.activeRequest.session_id,
      meta: payload.meta,
      lane: payload.lane ?? "cpu",
      seq: this.nextSeq(),
    };
    this.pushEvent(event);

    if (shouldEmitTimeStamp(this.activeRequest.devtools, level)) {
      emitTimeStamp(`${event.phase}#${event.seq}`);
    }
  }

  drain(options: TraceDrainOptions = {}): TraceEvent[] {
    const clear = options.clear ?? true;
    const max_events = options.max_events;

    const copy = [...this.events];
    copy.sort((a, b) => {
      if (a.abs_ts_ms !== b.abs_ts_ms) {
        return a.abs_ts_ms - b.abs_ts_ms;
      }
      if (a.ctx !== b.ctx) {
        return String(a.ctx).localeCompare(String(b.ctx));
      }
      return a.seq - b.seq;
    });

    const sliced =
      max_events !== undefined && max_events > 0 && copy.length > max_events
        ? copy.slice(copy.length - max_events)
        : copy;

    if (clear) {
      this.events = [];
    }
    return sliced;
  }

  private installRuntimeBridge(): void {
    (globalThis as any).__WEBLLM_TRACE_RUNTIME_PUSH__ = (
      payload: RuntimeTracePayload,
    ) => {
      this.ingestRuntimeEvent(payload);
    };
  }

  private updateRuntimeState(): void {
    (globalThis as any).__WEBLLM_TRACE_RUNTIME_STATE__ = {
      enabled: this.activeRequest.enabled,
      level: this.activeRequest.level,
      devtools: this.activeRequest.devtools,
      ctx: this.context,
      step: this.getCurrentRuntimeStep(),
      request_id: this.activeRequest.request_id,
      session_id: this.activeRequest.session_id,
      enable_gpu_timestamps: this.activeRequest.enable_gpu_timestamps ?? false,
    } as RuntimeTraceState;
  }

  private enterRuntimeStepScope(step: number | string | undefined): number {
    const scopeId = this.nextRuntimeStepScopeId++;
    this.runtimeStepScopes.push({
      id: scopeId,
      step,
    });
    this.updateRuntimeState();
    return scopeId;
  }

  private exitRuntimeStepScope(scopeId: number): void {
    const idx = this.runtimeStepScopes.findIndex(
      (scope) => scope.id === scopeId,
    );
    if (idx === -1) {
      return;
    }
    this.runtimeStepScopes.splice(idx, 1);
    this.updateRuntimeState();
  }

  private getCurrentRuntimeStep(): number | string | undefined {
    if (this.runtimeStepScopes.length === 0) {
      return undefined;
    }
    return this.runtimeStepScopes[this.runtimeStepScopes.length - 1].step;
  }

  private nextSeq(): number {
    this.seq += 1;
    return this.seq;
  }

  private pushEvent(event: TraceEvent): void {
    this.events.push(event);
    if (this.events.length > this.capacity) {
      this.events.splice(0, this.events.length - this.capacity);
    }
  }
}

export function getTraceCollector(context?: TraceContext): TraceCollector {
  const existing = (globalThis as any).__WEBLLM_TRACE_COLLECTOR__ as
    | TraceCollector
    | undefined;
  if (existing !== undefined) {
    if (context !== undefined) {
      existing.setContext(context);
    }
    return existing;
  }
  const collector = new TraceCollector();
  if (context !== undefined) {
    collector.setContext(context);
  }
  (globalThis as any).__WEBLLM_TRACE_COLLECTOR__ = collector;
  return collector;
}

export function mergeTraceEvents(
  lists: Array<Array<TraceEvent>>,
): TraceEvent[] {
  const merged = lists.flat();
  merged.sort((a, b) => {
    if (a.abs_ts_ms !== b.abs_ts_ms) {
      return a.abs_ts_ms - b.abs_ts_ms;
    }
    if (a.ctx !== b.ctx) {
      return String(a.ctx).localeCompare(String(b.ctx));
    }
    return a.seq - b.seq;
  });
  return merged;
}

export function traceInstant(
  phase: string,
  options: TraceEventOptions = {},
): void {
  getTraceCollector(options.ctx).instant(phase, options);
}
