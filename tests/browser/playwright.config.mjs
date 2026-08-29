import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: ".",
  testMatch: "resumable.spec.js",
  fullyParallel: false,
  workers: 1,
  reporter: "line",
  webServer: {
    command: "node server.mjs",
    port: 4178,
    reuseExistingServer: true,
  },
  use: {
    baseURL: "http://127.0.0.1:4178",
    browserName: "chromium",
  },
});
