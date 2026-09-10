import { expect, test } from "./fixtures.mjs";

// Explicitly opt in: downloads model weights and needs a checkpoint-capable
// model library, a WebGPU adapter, and shader-f16 support. No mocked inference.
const modelLib =
  globalThis.process.env.WEBLLM_TEST_MODEL_LIB ??
  (globalThis.process.env.WEBLLM_TEST_MODEL_LIB_PATH
    ? "http://127.0.0.1:4178/model.wasm"
    : undefined);
const modelId = "Qwen3-0.6B-q4f16_1-MLC";
const modelSource = globalThis.process.env.WEBLLM_TEST_MODEL_PATH
  ? "http://127.0.0.1:4178/model/"
  : undefined;

if (
  globalThis.process.env.WEBLLM_TEST_MODEL_LIB &&
  globalThis.process.env.WEBLLM_TEST_MODEL_LIB_PATH
) {
  throw new Error("Set only one model library URL or local path.");
}

async function loadModel(page) {
  await page.goto("/");
  await page.waitForFunction(
    () => globalThis.webllmBrowserHarness !== undefined,
  );
  await page.evaluate(
    async ({ modelId, modelLib, modelSource }) => {
      const { MLCEngine, prebuiltAppConfig } = globalThis.webllmBrowserHarness;
      const model = prebuiltAppConfig.model_list.find(
        (item) => item.model_id === modelId,
      );
      const engine = new MLCEngine({
        appConfig: {
          model_list: [
            {
              ...model,
              model: modelSource ?? model.model,
              model_lib: modelLib,
            },
          ],
        },
      });
      await Promise.race([
        engine.reload(modelId, {
          context_window_size: 512,
          prefill_chunk_size: 128,
        }),
        globalThis.gpuFailure,
      ]);
      globalThis.gpuEngine = engine;
    },
    { modelId, modelLib, modelSource },
  );
}

for (const durabilityMode of ["exact", "relaxed"]) {
  for (const checkpointPrompt of [false, true]) {
    test(`real WebGPU survives repeated reloads with ${checkpointPrompt ? "KV" : "token"} recovery (${durabilityMode})`, async ({
      page,
    }) => {
      test.skip(
        !modelLib,
        "Set WEBLLM_TEST_MODEL_LIB or WEBLLM_TEST_MODEL_LIB_PATH to a checkpoint-capable Qwen3-0.6B WASM",
      );
      test.setTimeout(300_000);
      await page.addInitScript(() => {
        // Surface the first validation error instead of letting invalid GPU
        // command buffers accumulate until the browser process crashes.
        let fail;
        globalThis.gpuFailure = new Promise((_, reject) => {
          fail = reject;
        });
        void globalThis.gpuFailure.catch(() => undefined);
        if (globalThis.GPUAdapter === undefined) return;
        const requestDevice = globalThis.GPUAdapter.prototype.requestDevice;
        globalThis.GPUAdapter.prototype.requestDevice = async function (
          ...args
        ) {
          const device = await requestDevice.apply(this, args);
          device.addEventListener(
            "uncapturederror",
            (event) => fail(new Error(event.error.message)),
            { once: true },
          );
          return device;
        };
      });
      const pageErrors = [];
      page.on("pageerror", (err) => pageErrors.push(err.message));
      page.on("requestfailed", (request) => {
        const url = new globalThis.URL(request.url());
        globalThis.console.error(
          url.origin + url.pathname,
          request.failure()?.errorText,
        );
      });
      let gpuErrors = 0;
      page.on("console", (message) => {
        if (message.type() === "error" && gpuErrors++ < 3) {
          globalThis.console.error(message.text());
        }
      });
      await loadModel(page);
      const sessionId = `browser-gpu-${checkpointPrompt}-${durabilityMode}`;
      const baseline = await page.evaluate(
        async ({ modelId, checkpointPrompt, sessionId, durabilityMode }) => {
          const request = {
            model: modelId,
            messages: [
              {
                role: "user",
                content:
                  "What is 2 + 2? " +
                  "Remember this conversation for later. ".repeat(24),
              },
              { role: "assistant", content: "4." },
              { role: "user", content: "What is 3 + 3?" },
              { role: "assistant", content: "6." },
              { role: "user", content: "List the numbers from 1 through 20." },
            ],
            seed: 17,
            temperature: 0.7,
            max_tokens: 24,
            ignore_eos: true,
          };
          const response = await Promise.race([
            globalThis.gpuEngine.chatCompletion(request),
            globalThis.gpuFailure,
          ]);
          const stream = await globalThis.gpuEngine.chatCompletion({
            ...request,
            stream: true,
            extra_body: {
              resumable: {
                enabled: true,
                sessionId,
                checkpointPrompt,
                checkpointIntervalTokens: 512,
                durabilityMode,
                strictPersistence: durabilityMode === "exact",
              },
            },
          });
          globalThis.gpuStream = stream[Symbol.asyncIterator]();
          await globalThis.gpuStream.next();
          await globalThis.gpuStream.next();
          return {
            request,
            text: response.choices[0].message.content,
            promptTokens: response.usage.prompt_tokens,
          };
        },
        { modelId, checkpointPrompt, sessionId, durabilityMode },
      );
      expect(baseline.promptTokens).toBeGreaterThan(128);

      // Navigate without return()/interruptGenerate(): this destroys the engine
      // while generation is unfinished, releasing browser-owned Web Locks.
      await loadModel(page);
      const mode = await page.evaluate(async (sessionId) => {
        const sessions = await globalThis.gpuEngine.listResumableSessions();
        return sessions.find((session) => session.sessionId === sessionId)
          ?.recoveryMode;
      }, sessionId);
      expect(mode).toBe(checkpointPrompt ? "kv" : "token_replay");
      await page.evaluate(async (sessionId) => {
        const stream = await globalThis.gpuEngine.resumeChatCompletion(
          sessionId,
          { continueGeneration: true, stream: true },
        );
        globalThis.gpuStream = stream[Symbol.asyncIterator]();
        await globalThis.gpuStream.next();
        await globalThis.gpuStream.next();
      }, sessionId);

      await loadModel(page);
      const result = await page.evaluate(
        async ({ sessionId, request }) => {
          const resumed = await globalThis.gpuEngine.resumeChatCompletion(
            sessionId,
            { continueGeneration: true },
          );
          const metrics = globalThis.gpuEngine.lastResumableMetrics;
          const finished = (
            await globalThis.gpuEngine.listResumableSessions()
          ).find((session) => session.sessionId === sessionId);
          const saved =
            await globalThis.gpuEngine.resumeChatCompletion(sessionId);
          await globalThis.gpuEngine.deleteResumableSession(sessionId);
          const followup = {
            ...request,
            max_tokens: 4,
            messages: [
              ...request.messages,
              { role: "assistant", content: resumed.recoveredText },
              { role: "user", content: "Continue." },
            ],
          };
          const warm = await globalThis.gpuEngine.chatCompletion(followup);
          const newSessionId = `${sessionId}-next`;
          const fresh = await globalThis.gpuEngine.chatCompletion({
            ...followup,
            extra_body: {
              resumable: {
                enabled: true,
                sessionId: newSessionId,
                checkpointPrompt: false,
                strictPersistence: true,
              },
            },
          });
          await globalThis.gpuEngine.deleteResumableSession(newSessionId);
          await globalThis.gpuEngine.unload();
          return {
            resumed,
            metrics,
            finished,
            saved,
            warmPromptTokens: warm.usage.prompt_tokens,
            freshPromptTokens: fresh.usage.prompt_tokens,
          };
        },
        { sessionId, request: baseline.request },
      );
      expect(result.resumed.recoveredText).toBe(baseline.text);
      expect(result.resumed.recoveryMode).toBe(
        checkpointPrompt ? "kv" : "token_replay",
      );
      expect(result.finished).toMatchObject({
        resumable: false,
        recoveryMode: "none",
      });
      expect(result.saved).toMatchObject({
        recoveredText: baseline.text,
        recoveryMode: "text_only",
      });
      expect(result.warmPromptTokens).toBeLessThan(result.freshPromptTokens);
      expect(result.freshPromptTokens).toBeGreaterThan(baseline.promptTokens);
      expect(pageErrors).toEqual([]);
      expect(gpuErrors).toBe(0);
    });
  }
}
