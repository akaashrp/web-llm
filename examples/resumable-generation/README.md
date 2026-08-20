# WebLLM Resumable Generation Example

This example is a browser harness for testing crash-resumable generation with
real WebGPU.

It intentionally targets only:

```text
Qwen3-0.6B-q4f16_1-MLC
```

The model artifact comes from WebLLM's prebuilt model config, but `model_lib`
is overridden to:

```text
https://raw.githubusercontent.com/akaashrp/mlc-binaries/main/resumable/Qwen3-0.6B-q4f16_1-webgpu-mlc-new-runtime.wasm
```

## Run

```bash
cd examples/resumable-generation
npm install
npm run dev
```

Open the printed localhost URL in Chrome or Edge with WebGPU enabled.

The example depends on the repo root through `file:../..`. Its `dev` and
`build` scripts rebuild the root package first, then Vite serves the generated
`lib/index.js`. This still exercises the current checkout, while avoiding raw
TypeScript runtime export issues in Vite.

## Manual Crash Test

1. Keep `Dedicated Worker` selected.
2. Click `Load`.
3. Click `Start`.
4. After several chunks appear, click `Reload Page` or close and reopen the tab.
5. Click `Load`.
6. Click `List Sessions`.
7. Click `Resume Continue`.

Expected results:

- If no committed KV checkpoint existed yet, recovery should use
  `token_replay`.
- If a committed checkpoint existed, recovery should use `kv`.
- The restored output should start with the same prefix that was visible before
  reload.

## Automated Reload Test

Enable `Reload after chunks`, set `Chunks`, then click `Start`. The page reloads
itself after that many streamed chunks. After reload, click `Load`, then
`Resume Continue`.

## Validation Runner

Click `Run Validation` to exercise real WebGPU resumability checks in worker
mode:

- Worker-mode streaming resume after an interrupted generation.
- Same-session repeated resume by interrupting a resumed continuation and
  resuming it again.
- Session lock exclusion from a second worker client against the same OPFS
  session.
- Low-quota behavior when the browser reports less than 512 MiB free storage.

The low-quota case reports `skip` on normal profiles with enough free quota.
For a hard low-quota check, run the page in a constrained browser profile or
manually fill origin storage, then rerun `Run Validation`. The expected behavior
is that KV checkpointing is skipped while token journaling remains recoverable.

For an exact cross-tab lock check, start a resumable generation in one tab,
open this same page in a second tab, enter the same session ID, and click
`Resume Continue`. The second tab should fail with
`Resumable session is already active`.

## Metrics

`Dedicated Worker` mode matches the intended runtime path, but internal engine
metrics live inside the worker. Switch to `Main Thread` mode if you need to
inspect `lastResumableMetrics` directly from the page.
