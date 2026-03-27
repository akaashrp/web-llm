import { beforeEach, expect, test } from "@jest/globals";
import { TraceCollector, mergeTraceEvents } from "../src/trace";

beforeEach(() => {
  delete (globalThis as any).__WEBLLM_TRACE_RUNTIME_PUSH__;
  delete (globalThis as any).__WEBLLM_TRACE_RUNTIME_STATE__;
});

test("TraceCollector enforces ring buffer capacity and preserves recent events", () => {
  const collector = new TraceCollector();
  collector.setContext("main");
  collector.setCapacity(1024);
  const prev = collector.beginRequest({
    enabled: true,
    level: "verbose",
    devtools: "off",
    request_id: "req-1",
    session_id: "sess-1",
  });

  for (let i = 0; i < 1026; i++) {
    collector.instant(`phase.${i}`);
  }

  const events = collector.drain({ clear: false });
  expect(events).toHaveLength(1024);
  expect(events[0].phase).toBe("phase.2");
  expect(events[events.length - 1].phase).toBe("phase.1025");
  collector.endRequest(prev);
});

test("TraceCollector filters verbose events when level is major", () => {
  const collector = new TraceCollector();
  collector.setContext("main");
  const prev = collector.beginRequest({
    enabled: true,
    level: "major",
    devtools: "off",
    request_id: "req-2",
    session_id: "sess-2",
  });

  collector.instant("major.visible", { level: "major" });
  collector.instant("verbose.hidden", { level: "verbose" });
  const events = collector.drain();
  expect(events).toHaveLength(1);
  expect(events[0].phase).toBe("major.visible");
  collector.endRequest(prev);
});

test("TraceCollector runtime bridge ingests events from runtime hooks", () => {
  const collector = new TraceCollector();
  collector.setContext("decode_worker");
  const prev = collector.beginRequest({
    enabled: true,
    level: "verbose",
    devtools: "off",
    request_id: "req-3",
    session_id: "sess-3",
  });

  (globalThis as any).__WEBLLM_TRACE_RUNTIME_PUSH__?.({
    phase: "webgpu.queue.submit",
    level: "verbose",
    lane: "cpu",
    meta: { bytes: 1024 },
  });

  const events = collector.drain();
  expect(events).toHaveLength(1);
  expect(events[0].phase).toBe("webgpu.queue.submit");
  expect(events[0].ctx).toBe("decode_worker");
  expect(events[0].meta?.bytes).toBe(1024);
  collector.endRequest(prev);
});

test("mergeTraceEvents orders deterministically by timestamp, ctx, and seq", () => {
  const merged = mergeTraceEvents([
    [
      {
        abs_ts_ms: 100,
        ctx: "main",
        phase: "a",
        seq: 2,
      },
      {
        abs_ts_ms: 101,
        ctx: "main",
        phase: "b",
        seq: 1,
      },
    ],
    [
      {
        abs_ts_ms: 100,
        ctx: "decode_worker",
        phase: "c",
        seq: 1,
      },
    ],
  ]);
  expect(merged.map((e) => e.phase)).toEqual(["c", "a", "b"]);
});
