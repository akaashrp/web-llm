import { expect, test } from "@playwright/test";

// Explicitly opt in: downloads model weights and needs a checkpoint-capable
// model library, a WebGPU adapter, and shader-f16 support. No mocked inference.
const modelLib = globalThis.process.env.WEBLLM_TEST_MODEL_LIB;
const modelId = "Qwen3-0.6B-q4f16_1-MLC";

async function loadModel(page) {
  await page.goto("/");
  await page.waitForFunction(
    () => globalThis.webllmBrowserHarness !== undefined,
  );
  await page.evaluate(
    async ({ modelId, modelLib }) => {
      const { MLCEngine, prebuiltAppConfig } = globalThis.webllmBrowserHarness;
      const model = prebuiltAppConfig.model_list.find(
        (item) => item.model_id === modelId,
      );
      const engine = new MLCEngine({
        appConfig: { model_list: [{ ...model, model_lib: modelLib }] },
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
    { modelId, modelLib },
  );
}

for (const checkpointPrompt of [false, true]) {
  test(`real WebGPU survives repeated reloads with ${checkpointPrompt ? "KV" : "token"} recovery`, async ({
    page,
  }) => {
    test.skip(
      !modelLib,
      "Set WEBLLM_TEST_MODEL_LIB to a checkpoint-capable Qwen3-0.6B WASM URL",
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
      globalThis.GPUAdapter.prototype.requestDevice = async function (...args) {
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
    let gpuErrors = 0;
    page.on("console", (message) => {
      if (message.type() === "error" && gpuErrors++ < 3) {
        globalThis.console.error(message.text());
      }
    });
    await loadModel(page);
    const sessionId = `browser-gpu-${checkpointPrompt}`;
    const baseline = await page.evaluate(
      async ({ modelId, checkpointPrompt, sessionId }) => {
        const request = {
          model: modelId,
          messages: [
            { role: "user", content: "What is 2 + 2?" },
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
              durabilityMode: "exact",
              strictPersistence: true,
            },
          },
        });
        globalThis.gpuStream = stream[Symbol.asyncIterator]();
        await globalThis.gpuStream.next();
        await globalThis.gpuStream.next();
        return response.choices[0].message.content;
      },
      { modelId, checkpointPrompt, sessionId },
    );

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
    const result = await page.evaluate(async (sessionId) => {
      const resumed = await globalThis.gpuEngine.resumeChatCompletion(
        sessionId,
        { continueGeneration: true },
      );
      const metrics = globalThis.gpuEngine.lastResumableMetrics;
      await globalThis.gpuEngine.deleteResumableSession(sessionId);
      await globalThis.gpuEngine.unload();
      return { resumed, metrics };
    }, sessionId);
    expect(result.resumed.recoveredText).toBe(baseline);
    expect(result.resumed.recoveryMode).toBe(
      checkpointPrompt ? "kv" : "token_replay",
    );
    expect(pageErrors).toEqual([]);
  });
}
