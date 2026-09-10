import { test as base, expect } from "@playwright/test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

// A fresh profile gives every test cold storage while preserving it across page
// crashes. Closing the browser also avoids macOS Chromium's incognito-context
// teardown crash, which reproduces with blank pages and no WebLLM loaded.
const test = base.extend({
  context: async (
    { playwright, browserName, launchOptions, headless, baseURL },
    use,
  ) => {
    const profile = await mkdtemp(
      join(
        globalThis.process.env.WEBLLM_TEST_PROFILE_ROOT ?? tmpdir(),
        "webllm-browser-",
      ),
    );
    try {
      const context = await playwright[browserName].launchPersistentContext(
        profile,
        {
          ...launchOptions,
          headless,
          baseURL,
          ignoreDefaultArgs: ["--no-startup-window"],
        },
      );
      try {
        await use(context);
      } finally {
        await context.close();
      }
    } finally {
      await rm(profile, { recursive: true, force: true });
    }
  },
});

export { test, expect };
