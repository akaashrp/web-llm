import { test, expect, loadModel, makeRequest } from "./webgpu.mjs";

for (const durabilityMode of ["exact", "relaxed"]) {
  test(`real WebGPU active-session reads stay available (${durabilityMode})`, async ({
    page,
  }) => {
    await loadModel(page);
    const result = await page.evaluate(
      async ({ request, durabilityMode }) => {
        let reads = 0;
        let errorCount = 0;
        const firstErrors = [];
        for (let run = 0; run < 5; run++) {
          const sessionId = `live-read-${run}`;
          const stream = await globalThis.gpuEngine.chatCompletion({
            ...request,
            stream: true,
            extra_body: {
              resumable: {
                enabled: true,
                sessionId,
                durabilityMode,
                checkpointPrompt: false,
                strictPersistence: true,
              },
            },
          });
          const iterator = stream[Symbol.asyncIterator]();
          await iterator.next();
          let done = false;
          const writing = (async () => {
            try {
              while (!(await iterator.next()).done) {
                /* continue inference */
              }
            } finally {
              done = true;
            }
          })();
          void writing.catch(() => undefined);
          while (!done) {
            try {
              // No fault injection: exercise the public read-only API concurrently
              // with the production journal writer and actual GPU generation.
              await globalThis.gpuEngine.resumeChatCompletion(sessionId);
            } catch (err) {
              errorCount++;
              if (firstErrors.length < 2) {
                firstErrors.push({ name: err.name, message: err.message });
              }
            }
            reads++;
          }
          await writing;
        }
        return { reads, errorCount, firstErrors };
      },
      { request: makeRequest({ max_tokens: 64 }), durabilityMode },
    );
    expect(result.reads).toBeGreaterThan(20);
    expect(result.errorCount, JSON.stringify(result.firstErrors)).toBe(0);
  });
}
