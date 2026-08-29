import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(
    () => globalThis.webllmBrowserHarness !== undefined,
  );
});

test("OPFS journal repair and cross-context locking use browser primitives", async ({
  page,
}) => {
  const result = await page.evaluate(() =>
    globalThis.webllmBrowserHarness.runOPFSRegression(),
  );

  expect(result).toEqual({
    text: "one-two",
    webLocksAvailable: true,
    acquiredWhileHeld: false,
    acquiredAfterRelease: true,
    stoppedBeforeRepair: "partial_record",
    stoppedAfterRepair: undefined,
    repairedRecordCount: 1,
  });
});

test("LLMChatPipeline samples the first token from a zero-token replay", async ({
  page,
}) => {
  const result = await page.evaluate(() =>
    globalThis.webllmBrowserHarness.runFirstTokenReplayRegression(),
  );

  expect(result).toEqual({
    forwardedPrompt: [1, 2, 3],
    outputIds: [9],
    promptLogitsDisposed: true,
    replayedTokens: 0,
    sampledTokenId: 17,
    sampledTokenPosition: 3,
    committedText: "first",
  });
});

test("LLMChatPipeline retains only final logits while forwarding replay tokens", async ({
  page,
}) => {
  const result = await page.evaluate(() =>
    globalThis.webllmBrowserHarness.runKnownTokenForwardingRegression(),
  );

  expect(result).toEqual({
    forwardedTokens: 10,
    chunkCount: 3,
    detached: [3],
    finalLogitsId: 3,
    endedScopes: 1,
  });
});
