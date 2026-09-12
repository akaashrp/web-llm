import { test, expect, loadModel } from "./webgpu.mjs";

test("real WebGPU ordinary legacy completions stream and cancel without locking out later requests", async ({
  page,
}) => {
  await loadModel(page);
  const result = await page.evaluate(async () => {
    const request = {
      prompt: "List the numbers from 1 through 10.",
      max_tokens: 8,
      ignore_eos: true,
      seed: 17,
      temperature: 0.7,
    };
    const engine = globalThis.gpuEngine;
    const baseline = (await engine.completion(request)).choices[0].text;
    let streamed = "";
    for await (const chunk of await engine.completion({
      ...request,
      stream: true,
    })) {
      streamed += chunk.choices[0]?.text ?? "";
    }
    const cancelled = (await engine.completion({ ...request, stream: true }))[
      Symbol.asyncIterator
    ]();
    await cancelled.next();
    await cancelled.return();
    const unused = (await engine.completion({ ...request, stream: true }))[
      Symbol.asyncIterator
    ]();
    await unused.return();
    const afterCancel = (await engine.completion(request)).choices[0].text;
    return { baseline, streamed, afterCancel };
  });
  expect(result.baseline.length).toBeGreaterThan(0);
  expect(result.streamed).toBe(result.baseline);
  expect(result.afterCancel).toBe(result.baseline);
});
