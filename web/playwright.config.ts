import { defineConfig, devices } from "@playwright/test";

// The suite drives the REAL stack: FastAPI serving the built frontend, with a
// real detector loaded. Mocking the API here would test the mock, and the
// failures worth catching (SSE dropping, a result shape changing, the verdict
// rendering as a bare binary) only appear end to end.
//
// DDETECT_RUN_DIR must point at a trained run. `make e2e` sets it.
export default defineConfig({
  testDir: "./tests",
  timeout: 180_000,            // an analysis takes ~10s on CPU, plus model load
  expect: { timeout: 15_000 },
  fullyParallel: false,        // one CPU-bound worker pool behind all of this
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["github"], ["list"]] : [["list"]],
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://127.0.0.1:8000",
    trace: "retain-on-failure",
    video: "retain-on-failure",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile", use: { ...devices["Pixel 7"] } },
  ],
  webServer: process.env.E2E_NO_SERVER
    ? undefined
    : {
        command:
          // The venv on a dev machine; the runner's Python in CI.
          "cd .. && $(test -x .venv/bin/python && echo .venv/bin/python || echo python) -m uvicorn api.main:app --host 127.0.0.1 --port 8000",
        url: "http://127.0.0.1:8000/healthz",
        reuseExistingServer: true,
        timeout: 120_000,
        stdout: "pipe",
        stderr: "pipe",
      },
});
