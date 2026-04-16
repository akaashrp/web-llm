#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";

function usage() {
  console.log(
    [
      "Usage:",
      "  node scripts/trace_breakdown.mjs <events.json> [options]",
      "",
      "Options:",
      "  --request-id <id>   Filter to a specific request_id",
      "  --session-id <id>   Filter to a specific session_id",
      "  --no-prefill        Exclude prefill.step rows from output",
      "  --json              Print JSON instead of tables",
      "  --out <file>        Write JSON output to a file",
      "  --help              Show this help",
      "",
      "Input format:",
      "  A JSON array returned by engine.drainTraceEvents(...).",
    ].join("\n"),
  );
}

function toFiniteNumber(v) {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function round3(v) {
  return Number(v.toFixed(3));
}

function mean(arr) {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function sum(arr) {
  return arr.reduce((a, b) => a + b, 0);
}

function minFinite(values) {
  let found = false;
  let best = Infinity;
  for (const v of values) {
    if (typeof v !== "number" || !Number.isFinite(v)) continue;
    if (v < best) best = v;
    found = true;
  }
  return found ? best : null;
}

function toStepKey(step) {
  if (step === undefined || step === null) {
    return null;
  }
  return String(step);
}

function makeScopeStepKey(requestId, sessionId, step) {
  const stepKey = toStepKey(step);
  if (stepKey === null) {
    return null;
  }
  return `${String(requestId ?? "")}|${String(sessionId ?? "")}|${stepKey}`;
}

function parseArgs(argv) {
  const opts = {
    inputPath: null,
    requestId: null,
    sessionId: null,
    includePrefill: true,
    json: false,
    outPath: null,
  };

  for (let i = 0; i < argv.length; ++i) {
    const arg = argv[i];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--request-id") {
      opts.requestId = argv[++i] ?? null;
    } else if (arg === "--session-id") {
      opts.sessionId = argv[++i] ?? null;
    } else if (arg === "--no-prefill") {
      opts.includePrefill = false;
    } else if (arg === "--json") {
      opts.json = true;
    } else if (arg === "--out") {
      opts.outPath = argv[++i] ?? null;
    } else if (arg.startsWith("-")) {
      throw new Error(`Unknown option: ${arg}`);
    } else if (opts.inputPath === null) {
      opts.inputPath = arg;
    } else {
      throw new Error(`Unexpected positional argument: ${arg}`);
    }
  }

  if (opts.inputPath === null) {
    throw new Error("Missing input file path");
  }
  return opts;
}

function chooseMostRecentRequestId(events) {
  const byRequest = new Map();
  for (const e of events) {
    if (!e || e.request_id == null) continue;
    const cur = byRequest.get(e.request_id) ?? -Infinity;
    byRequest.set(e.request_id, Math.max(cur, e.abs_ts_ms ?? -Infinity));
  }
  if (byRequest.size <= 1) {
    return null;
  }
  let bestId = null;
  let bestTs = -Infinity;
  for (const [id, ts] of byRequest.entries()) {
    if (ts > bestTs) {
      bestTs = ts;
      bestId = id;
    }
  }
  return bestId;
}

function sortEvents(events) {
  events.sort((a, b) => {
    const ta = toFiniteNumber(a?.abs_ts_ms) ?? 0;
    const tb = toFiniteNumber(b?.abs_ts_ms) ?? 0;
    if (ta !== tb) return ta - tb;
    const ca = String(a?.ctx ?? "");
    const cb = String(b?.ctx ?? "");
    if (ca !== cb) return ca.localeCompare(cb);
    const sa = toFiniteNumber(a?.seq) ?? 0;
    const sb = toFiniteNumber(b?.seq) ?? 0;
    return sa - sb;
  });
  return events;
}

function getDurationMs(event) {
  return toFiniteNumber(event?.meta?.duration_ms);
}

function extractSpans(events, includePrefill) {
  const spans = [];
  for (const e of events) {
    if (
      e?.phase !== "decode.step" &&
      (!includePrefill || e?.phase !== "prefill.step")
    ) {
      continue;
    }
    const duration = getDurationMs(e);
    const end = toFiniteNumber(e?.abs_ts_ms);
    if (duration === null || end === null) continue;
    spans.push({
      kind: e.phase === "decode.step" ? "decode" : "prefill",
      step: e.step,
      ctx: e.ctx,
      request_id: e.request_id,
      session_id: e.session_id,
      start_ms: end - duration,
      end_ms: end,
      iter_ms: duration,
      seq: toFiniteNumber(e?.seq),
    });
  }
  spans.sort((a, b) => a.start_ms - b.start_ms);
  return spans;
}

function isReadbackMapAsyncStart(event) {
  return (
    event?.phase === "webgpu.map_async.start" ||
    event?.phase === "webgpu.readback_ring.map_async.start"
  );
}

function isReadbackMapAsyncEnd(event) {
  return (
    event?.phase === "webgpu.map_async.end" ||
    event?.phase === "webgpu.readback_ring.map_async.end"
  );
}

function mapAsyncKey(event) {
  const phase = String(event?.phase ?? "");
  const common = [
    String(event?.ctx ?? ""),
    String(event?.request_id ?? ""),
    String(event?.session_id ?? ""),
  ];

  if (phase.startsWith("webgpu.readback_ring.map_async.")) {
    const ringId = event?.meta?.ring_id ?? "unknown";
    const batchSeq = event?.meta?.batch_seq ?? "unknown";
    const slotIdx = event?.meta?.slot_idx ?? "unknown";
    return [
      ...common,
      "ring",
      String(ringId),
      String(batchSeq),
      String(slotIdx),
    ].join("|");
  }

  const submitSeq = event?.meta?.submit_seq ?? "unknown";
  const bytes = event?.meta?.bytes ?? "unknown";
  return [...common, "baseline", String(submitSeq), String(bytes)].join("|");
}

function buildReadbackIntervals(events) {
  const starts = new Map();
  const intervals = [];
  for (const e of events) {
    if (isReadbackMapAsyncStart(e)) {
      const key = mapAsyncKey(e);
      if (!starts.has(key)) starts.set(key, []);
      starts.get(key).push(e);
    } else if (isReadbackMapAsyncEnd(e)) {
      const key = mapAsyncKey(e);
      const queue = starts.get(key);
      if (!queue || queue.length === 0) continue;
      const s = queue.shift();
      const sTs = toFiniteNumber(s?.abs_ts_ms);
      const eTs = toFiniteNumber(e?.abs_ts_ms);
      if (sTs === null || eTs === null || eTs < sTs) continue;
      intervals.push({
        start_ms: sTs,
        end_ms: eTs,
        duration_ms: eTs - sTs,
        type: String(e?.phase ?? "").includes("readback_ring")
          ? "ring"
          : "baseline",
        start_seq: toFiniteNumber(s?.seq),
        end_seq: toFiniteNumber(e?.seq),
        step: e?.step !== undefined ? e.step : s?.step,
        ctx: e.ctx,
        request_id: e.request_id,
        session_id: e.session_id,
      });
    }
  }
  return intervals;
}

function overlapMs(aStart, aEnd, bStart, bEnd) {
  const lo = Math.max(aStart, bStart);
  const hi = Math.min(aEnd, bEnd);
  return Math.max(0, hi - lo);
}

function mergeIntervals(intervals) {
  if (intervals.length === 0) return [];
  const sorted = [...intervals].sort((a, b) => a.start_ms - b.start_ms);
  const merged = [sorted[0]];
  for (let i = 1; i < sorted.length; ++i) {
    const cur = sorted[i];
    const last = merged[merged.length - 1];
    if (cur.start_ms <= last.end_ms) {
      last.end_ms = Math.max(last.end_ms, cur.end_ms);
    } else {
      merged.push({ start_ms: cur.start_ms, end_ms: cur.end_ms });
    }
  }
  return merged;
}

function intervalOverlapTotal(aIntervals, bIntervals, winStart, winEnd) {
  const overlaps = [];
  for (const a of aIntervals) {
    if (a.end_ms <= winStart || a.start_ms >= winEnd) continue;
    for (const b of bIntervals) {
      if (b.end_ms <= winStart || b.start_ms >= winEnd) continue;
      const s = Math.max(a.start_ms, b.start_ms, winStart);
      const e = Math.min(a.end_ms, b.end_ms, winEnd);
      if (e > s) overlaps.push({ start_ms: s, end_ms: e });
    }
  }
  const merged = mergeIntervals(overlaps);
  return sum(merged.map((x) => x.end_ms - x.start_ms));
}

function buildGpuDispatchIntervals(events) {
  const intervals = [];
  for (const e of events) {
    if (e?.phase !== "gpu.compute.dispatch" || e?.lane !== "gpu") continue;
    const start = toFiniteNumber(e?.abs_ts_ms);
    const dur = toFiniteNumber(e?.meta?.gpu_duration_ms);
    if (start === null || dur === null || dur < 0) continue;
    intervals.push({
      start_ms: start,
      end_ms: start + dur,
      ctx: e.ctx,
      request_id: e.request_id,
      session_id: e.session_id,
    });
  }
  return intervals;
}

function buildGpuDurationBySubmitSeq(events) {
  const bySubmitSeq = new Map();
  for (const e of events) {
    if (e?.phase !== "gpu.compute.dispatch" || e?.lane !== "gpu") continue;
    const submitSeq = toFiniteNumber(e?.meta?.submit_seq);
    const dur = toFiniteNumber(e?.meta?.gpu_duration_ms);
    if (submitSeq === null || dur === null || dur < 0) continue;
    bySubmitSeq.set(submitSeq, (bySubmitSeq.get(submitSeq) ?? 0) + dur);
  }
  return bySubmitSeq;
}

function buildGpuDurationByStep(events) {
  const byStep = new Map();
  for (const e of events) {
    if (e?.phase !== "gpu.compute.dispatch" || e?.lane !== "gpu") continue;
    const dur = toFiniteNumber(e?.meta?.gpu_duration_ms);
    if (dur === null || dur < 0) continue;
    const key = makeScopeStepKey(e?.request_id, e?.session_id, e?.step);
    if (key === null) continue;
    byStep.set(key, (byStep.get(key) ?? 0) + dur);
  }
  return byStep;
}

function buildReadbackDurationByStep(readbackIntervals) {
  const byStep = new Map();
  for (const interval of readbackIntervals) {
    const key = makeScopeStepKey(
      interval.request_id,
      interval.session_id,
      interval.step,
    );
    if (key === null) continue;
    byStep.set(key, (byStep.get(key) ?? 0) + interval.duration_ms);
  }
  return byStep;
}

function computeBreakdown(events, spans, readbackIntervals) {
  const hasGpuLane = events.some(
    (e) => e?.phase === "gpu.compute.dispatch" && e?.lane === "gpu",
  );
  const hasRingReadback = readbackIntervals.some((i) => i.type === "ring");
  const gpuIntervals = buildGpuDispatchIntervals(events);
  const gpuBySubmitSeq = buildGpuDurationBySubmitSeq(events);
  const gpuByStep = buildGpuDurationByStep(events);
  const readbackByStep = buildReadbackDurationByStep(readbackIntervals);
  const gpuSeqEvents = events
    .filter((e) => e?.phase === "gpu.compute.dispatch" && e?.lane === "gpu")
    .map((e) => ({
      seq: toFiniteNumber(e?.seq),
      request_id: e?.request_id,
      duration_ms: toFiniteNumber(e?.meta?.gpu_duration_ms) ?? 0,
    }))
    .filter((x) => x.seq !== null && x.duration_ms >= 0);

  const spansBySeq = [...spans].sort((a, b) => {
    const sa = toFiniteNumber(a?.seq) ?? Number.MAX_SAFE_INTEGER;
    const sb = toFiniteNumber(b?.seq) ?? Number.MAX_SAFE_INTEGER;
    return sa - sb;
  });
  const nextSeqBySpan = new Map();
  for (let i = 0; i < spansBySeq.length; ++i) {
    const cur = spansBySeq[i];
    const next = spansBySeq[i + 1];
    const nextSeq = toFiniteNumber(next?.seq);
    nextSeqBySpan.set(cur, nextSeq ?? Number.POSITIVE_INFINITY);
  }

  const ringReadbackIntervalsWithSeq = readbackIntervals.filter(
    (i) => i.type === "ring" && toFiniteNumber(i.end_seq) !== null,
  );
  const rows = [];

  for (const span of spans) {
    const windowEvents = events.filter(
      (e) =>
        toFiniteNumber(e?.abs_ts_ms) !== null &&
        e.abs_ts_ms >= span.start_ms &&
        e.abs_ts_ms <= span.end_ms,
    );

    const tencode = sum(
      windowEvents
        .filter((e) => e.phase === "webgpu.command_encode.end")
        .map((e) => getDurationMs(e) ?? 0),
    );

    const spanGpuIntervals = gpuIntervals.filter(
      (i) =>
        i.start_ms <= span.end_ms &&
        i.end_ms >= span.start_ms &&
        (span.request_id == null || i.request_id === span.request_id),
    );
    const tgpuAbs = sum(
      spanGpuIntervals.map((i) =>
        overlapMs(i.start_ms, i.end_ms, span.start_ms, span.end_ms),
      ),
    );
    let tgpu = tgpuAbs;
    let tgpuSource = "abs_ts";
    const spanStepKey = makeScopeStepKey(
      span.request_id,
      span.session_id,
      span.step,
    );
    if (hasGpuLane && spanStepKey !== null) {
      const tgpuByStep = gpuByStep.get(spanStepKey) ?? 0;
      if (tgpuByStep > 0) {
        tgpu = tgpuByStep;
        tgpuSource = "step";
      }
    }
    if (hasGpuLane && tgpu === 0) {
      const spanSeq = toFiniteNumber(span?.seq);
      const spanNextSeq = nextSeqBySpan.get(span) ?? Number.POSITIVE_INFINITY;
      if (spanSeq !== null) {
        const tgpuByTraceSeq = sum(
          gpuSeqEvents
            .filter(
              (e) =>
                e.seq >= spanSeq &&
                e.seq < spanNextSeq &&
                (span.request_id == null || e.request_id === span.request_id),
            )
            .map((e) => e.duration_ms),
        );
        if (tgpuByTraceSeq > 0) {
          tgpu = tgpuByTraceSeq;
          tgpuSource = "trace_seq";
        }
      }
    }
    if (hasGpuLane && tgpu === 0) {
      const submitSeqs = new Set(
        windowEvents
          .filter((e) => e.phase === "webgpu.queue.submit")
          .map((e) => toFiniteNumber(e?.meta?.submit_seq))
          .filter((x) => x !== null),
      );
      let tgpuBySubmit = 0;
      for (const submitSeq of submitSeqs) {
        tgpuBySubmit += gpuBySubmitSeq.get(submitSeq) ?? 0;
      }
      if (tgpuBySubmit > 0) {
        tgpu = tgpuBySubmit;
        tgpuSource = "submit_seq";
      }
    }

    const tpostproc = sum(
      windowEvents
        .filter((e) => e.phase === "token.detokenize")
        .map((e) => getDurationMs(e) ?? 0),
    );

    const spanReadbackIntervals = readbackIntervals.filter(
      (i) =>
        i.start_ms <= span.end_ms &&
        i.end_ms >= span.start_ms &&
        (span.request_id == null || i.request_id === span.request_id),
    );
    const treadbackRaw = sum(
      spanReadbackIntervals.map((i) =>
        overlapMs(i.start_ms, i.end_ms, span.start_ms, span.end_ms),
      ),
    );
    // mapAsync includes waiting for GPU work. Avoid double-counting with Tgpu by
    // subtracting the overlap between readback and GPU dispatch intervals.
    const treadbackGpuOverlap = intervalOverlapTotal(
      spanReadbackIntervals,
      spanGpuIntervals,
      span.start_ms,
      span.end_ms,
    );
    let treadbackRawChosen = treadbackRaw;
    let treadbackSource = "abs_overlap";
    if (spanStepKey !== null) {
      const readbackRawByStep = readbackByStep.get(spanStepKey) ?? 0;
      if (readbackRawByStep > 0) {
        treadbackRawChosen = readbackRawByStep;
        treadbackSource = "step";
      }
    }
    const spanSeq = toFiniteNumber(span?.seq);
    const spanNextSeq = nextSeqBySpan.get(span) ?? Number.POSITIVE_INFINITY;
    if (hasRingReadback && spanSeq !== null && treadbackSource !== "step") {
      const ringSeqRaw = sum(
        ringReadbackIntervalsWithSeq
          .filter(
            (i) =>
              i.end_seq >= spanSeq &&
              i.end_seq < spanNextSeq &&
              (span.request_id == null || i.request_id === span.request_id),
          )
          .map((i) => i.duration_ms),
      );
      if (ringSeqRaw > 0) {
        treadbackRawChosen = ringSeqRaw;
        treadbackSource = "trace_seq_end";
      }
    }
    const treadbackUnbounded = Math.max(
      0,
      treadbackRawChosen -
        (treadbackSource === "abs_overlap" ? treadbackGpuOverlap : 0),
    );
    // In pipelined ring mode, raw async readback duration can exceed critical-path
    // decode latency of a single step. Clamp to per-step wall time budget.
    const treadbackBudget = Math.max(0, span.iter_ms - (tencode + tgpu + tpostproc));
    const treadback = Math.min(treadbackUnbounded, treadbackBudget);

    const tprep = span.iter_ms - (tencode + tgpu + treadback + tpostproc);

    rows.push({
      kind: span.kind,
      step: span.step,
      iter_ms: round3(span.iter_ms),
      Tprep_ms: round3(tprep),
      Tencode_ms: round3(tencode),
      Tgpu_ms: round3(tgpu),
      Treadback_raw_ms: round3(treadbackRawChosen),
      Treadback_ms: round3(treadback),
      Tpostproc_ms: round3(tpostproc),
      Tgpu_source: tgpuSource,
      Treadback_source: treadbackSource,
      coverage_pct: round3(
        span.iter_ms > 0
          ? ((tencode + tgpu + treadback + tpostproc) / span.iter_ms) * 100
          : 0,
      ),
      ctx: span.ctx,
      request_id: span.request_id,
      session_id: span.session_id,
      window_start_ms: round3(span.start_ms),
      window_end_ms: round3(span.end_ms),
    });
  }

  const decodeRows = rows.filter((r) => r.kind === "decode");
  const summarySource = decodeRows.length > 0 ? decodeRows : rows;
  const stepAttributedRows = summarySource.filter(
    (r) => r.Tgpu_source === "step" || r.Treadback_source === "step",
  ).length;
  const summary = {
    rows_used: summarySource.length,
    has_gpu_lane: hasGpuLane,
    has_ring_readback: hasRingReadback,
    step_attributed_rows: stepAttributedRows,
    mean_iter_ms: round3(mean(summarySource.map((r) => r.iter_ms))),
    mean_Tprep_ms: round3(mean(summarySource.map((r) => r.Tprep_ms))),
    mean_Tencode_ms: round3(mean(summarySource.map((r) => r.Tencode_ms))),
    mean_Tgpu_ms: round3(mean(summarySource.map((r) => r.Tgpu_ms))),
    mean_Treadback_raw_ms: round3(
      mean(summarySource.map((r) => r.Treadback_raw_ms)),
    ),
    mean_Treadback_ms: round3(mean(summarySource.map((r) => r.Treadback_ms))),
    mean_Tpostproc_ms: round3(mean(summarySource.map((r) => r.Tpostproc_ms))),
  };

  return {
    rows,
    summary,
    hasGpuLane,
    hasRingReadback,
    hasStepAttribution: stepAttributedRows > 0,
  };
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const raw = await fs.readFile(opts.inputPath, "utf-8");
  let events = JSON.parse(raw);
  if (!Array.isArray(events)) {
    throw new Error("Input JSON must be an array of trace events");
  }

  events = sortEvents(
    events.filter(
      (e) => e && typeof e === "object" && toFiniteNumber(e.abs_ts_ms) !== null,
    ),
  );

  let appliedRequestId = opts.requestId;
  if (appliedRequestId == null) {
    const autoRequestId = chooseMostRecentRequestId(events);
    if (autoRequestId != null) {
      appliedRequestId = autoRequestId;
      console.error(
        `[trace_breakdown] Multiple request_id values found; auto-selected latest request_id=${appliedRequestId}`,
      );
    }
  }

  if (appliedRequestId != null) {
    events = events.filter((e) => e.request_id === appliedRequestId);
  }
  if (opts.sessionId != null) {
    events = events.filter((e) => e.session_id === opts.sessionId);
  }
  if (events.length === 0) {
    throw new Error("No events remain after filtering");
  }

  const minSeq = minFinite(events.map((e) => toFiniteNumber(e?.seq)));
  const hasRequestStart = events.some((e) => e.phase === "request.start");
  if (!hasRequestStart || (minSeq !== null && minSeq > 1)) {
    console.error(
      "[trace_breakdown] Trace likely truncated by ring buffer (missing early events). " +
        "Decode-step coverage may be partial.",
    );
  }

  const spans = extractSpans(events, opts.includePrefill);
  if (spans.length === 0) {
    throw new Error(
      "No prefill.step/decode.step span events found (with meta.duration_ms)",
    );
  }

  const readbackIntervals = buildReadbackIntervals(events);
  const result = computeBreakdown(events, spans, readbackIntervals);
  const output = {
    input: path.resolve(opts.inputPath),
    filters: {
      request_id: appliedRequestId,
      session_id: opts.sessionId,
      include_prefill: opts.includePrefill,
    },
    summary: result.summary,
    rows: result.rows,
  };

  if (opts.outPath != null) {
    await fs.writeFile(opts.outPath, JSON.stringify(output, null, 2), "utf-8");
    console.error(`[trace_breakdown] wrote ${path.resolve(opts.outPath)}`);
  }

  if (opts.json) {
    console.log(JSON.stringify(output, null, 2));
    return;
  }

  console.log("\nSummary");
  console.table([output.summary]);
  console.log("\nPer-step breakdown");
  console.table(
    output.rows.map((r) => ({
      kind: r.kind,
      step: r.step,
      iter_ms: r.iter_ms,
      Tprep_ms: r.Tprep_ms,
      Tencode_ms: r.Tencode_ms,
      Tgpu_ms: r.Tgpu_ms,
      Treadback_raw_ms: r.Treadback_raw_ms,
      Treadback_ms: r.Treadback_ms,
      Tpostproc_ms: r.Tpostproc_ms,
      coverage_pct: r.coverage_pct,
    })),
  );

  if (!result.hasGpuLane) {
    console.error(
      "[trace_breakdown] No lane=gpu dispatch events found; Tgpu_ms is 0 and may be folded into Tprep_ms.",
    );
  }
  if (result.hasRingReadback) {
    if (result.hasStepAttribution) {
      console.error(
        "[trace_breakdown] Ring readback events detected: using direct step-based attribution where available.",
      );
    } else {
      console.error(
        "[trace_breakdown] Ring readback events detected: per-step decomposition is overlap-based and may under/over-attribute due pipelining.",
      );
    }
  }
}

main().catch((err) => {
  console.error(`[trace_breakdown] ${String(err?.message ?? err)}`);
  process.exit(1);
});
