import { expect, test } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

// package.json sets "type": "module", so __dirname does not exist here.
const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.resolve(HERE, "../../tests/fixtures/videos");
const WITH_AUDIO = path.join(FIXTURES, "real_sync_audio_v0.mp4");
const SILENT = path.join(FIXTURES, "real_silent_v0.mp4");
const NO_FACE = path.join(FIXTURES, "edge_no_face.mp4");

async function analyse(page: import("@playwright/test").Page, file: string) {
  await page.goto("/");
  await page.setInputFiles('input[type="file"]', file);
  // The verdict card is the completion signal; the progress tracker is
  // transient and racing it makes the test flaky. Match on the element id,
  // not the accessible name -- the heading's text is the verdict label
  // ("Likely authentic"), while the word "Verdict" is a sibling <p>.
  await expect(page.locator("#verdict-heading")).toBeVisible({ timeout: 170_000 });
}

test.describe("upload and analysis", () => {
  test("renders a three-state verdict, never a bare binary", async ({ page }) => {
    await analyse(page, WITH_AUDIO);

    const verdict = page.locator("#verdict-heading");
    // Section 13: the UI must never render an unqualified FAKE/REAL.
    await expect(verdict).toHaveText(
      /Likely authentic|Likely manipulated|Inconclusive/,
    );
    await expect(page.getByText(/^FAKE$|^REAL$/)).toHaveCount(0);
  });

  test("always shows the limitations disclaimer", async ({ page }) => {
    await analyse(page, WITH_AUDIO);
    await expect(page.getByText(/not evidence/i)).toBeVisible();
    await expect(page.getByRole("heading", { name: /limitations/i })).toBeVisible();
  });

  test("shows the evidence: timeline, streams and attention overlays", async ({ page }) => {
    await analyse(page, WITH_AUDIO);
    await expect(page.getByRole("heading", { name: /per-frame score/i })).toBeVisible();
    await expect(page.getByRole("heading", { name: /what each check found/i })).toBeVisible();
    await expect(page.getByRole("heading", { name: /where the model looked/i })).toBeVisible();
    await expect(page.locator('img[alt*="attention overlay"]').first()).toBeVisible();
  });

  test("explains the verdict in plain language", async ({ page }) => {
    await analyse(page, WITH_AUDIO);
    const box = page.getByRole("heading", { name: /in plain language/i });
    await expect(box).toBeVisible();
    // Labelled honestly as generated or rule-based -- never silently swapped.
    await expect(page.getByText(/generated summary|rule-based summary/i)).toBeVisible();
  });
});

test.describe("degraded inputs", () => {
  test("a silent clip reports its audio checks as not run", async ({ page }) => {
    await analyse(page, SILENT);
    await expect(page.getByText("No audio — visual only", { exact: true })).toBeVisible();
    await expect(page.getByText(/no audio track/i).first()).toBeVisible();
  });

  test("a clip with no face warns rather than failing", async ({ page }) => {
    await analyse(page, NO_FACE);
    // exact: the badge says "No face detected" and a caveat says "no face
    // detected in any sampled frame", so a loose regex matches both and
    // Playwright's strict mode rejects it.
    await expect(page.getByText("No face detected", { exact: true })).toBeVisible();
    await expect(page.getByRole("heading", { name: /caveats/i })).toBeVisible();
    await expect(page.getByText(/treat this result as unreliable/i)).toBeVisible();
  });
});

test.describe("history", () => {
  test("lists a completed analysis and can delete it", async ({ page }) => {
    await analyse(page, WITH_AUDIO);
    await page.getByRole("button", { name: /analyse another clip/i }).click();

    const history = page.getByTestId("history");
    await expect(history).toBeVisible();
    const before = await history.locator("li").count();
    expect(before).toBeGreaterThan(0);

    // The button carries an aria-label, so its accessible name is
    // "Delete analysis of <file>" -- matching on the visible text alone
    // finds nothing.
    await history.getByRole("button", { name: /^Delete analysis of/ }).first().click();
    await expect(history.locator("li")).toHaveCount(before - 1);
  });
});

test.describe("reports and reopening", () => {
  test("downloads a self-contained HTML report", async ({ page }) => {
    await analyse(page, WITH_AUDIO);
    await expect(page.getByRole("heading", { name: /save a report/i })).toBeVisible();
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("button", { name: /download html/i }).click(),
    ]);
    expect(download.suggestedFilename()).toMatch(/^avforge-report-.*\.html$/);
    const fs = await import("node:fs/promises");
    const html = await fs.readFile((await download.path())!, "utf8");
    // The report travels without this page, so it must carry the caveat.
    expect(html).toContain("not evidence");
    expect(html).toMatch(/data-verdict='(likely_authentic|likely_manipulated|inconclusive)'/);
  });

  test("reopens a past analysis from the history list", async ({ page }) => {
    await analyse(page, SILENT);
    await page.getByRole("button", { name: /analyse another clip/i }).click();

    const history = page.getByTestId("history");
    await history.getByRole("button", { name: /real_silent_v0\.mp4/ }).first().click();
    await expect(page.locator("#verdict-heading")).toBeVisible();
    await expect(page.getByText(/results for/i)).toContainText("real_silent_v0.mp4");
  });

  test("explains the model's threshold and abstention band", async ({ page }) => {
    await page.goto("/");
    await page.getByText("About this model").click();
    await expect(page.getByText(/is reported as inconclusive/i)).toBeVisible();
  });
});

test.describe("batch queue", () => {
  test("analyses several clips at once and opens each result", async ({ page }) => {
    await page.goto("/");
    await page.setInputFiles('input[type="file"]', [WITH_AUDIO, SILENT]);

    const queue = page.getByTestId("queue");
    await expect(queue).toBeVisible();
    await expect(queue.getByRole("heading")).toHaveText(/2 of 2 finished/, { timeout: 170_000 });
    await expect(queue.getByText(/Likely authentic|Likely manipulated|Inconclusive/)).toHaveCount(2);

    await queue.getByRole("button", { name: /open result for real_silent_v0\.mp4/i }).click();
    await expect(page.locator("#verdict-heading")).toBeVisible();
    await expect(page.getByText(/results for/i)).toContainText("real_silent_v0.mp4");

    // Coming back keeps the batch, so the other result is still reachable.
    await page.getByRole("button", { name: /analyse another clip/i }).click();
    await expect(page.getByTestId("queue")).toBeVisible();
  });

  test("skips non-video files with a reason instead of dropping them", async ({ page }) => {
    await page.goto("/");
    // Playwright cannot mix paths and buffers in one call, so both are buffers.
    const fs = await import("node:fs/promises");
    await page.setInputFiles('input[type="file"]', [
      { name: "real_silent_v0.mp4", mimeType: "video/mp4", buffer: await fs.readFile(SILENT) },
      { name: "notes.txt", mimeType: "text/plain", buffer: Buffer.from("x") },
    ]);
    await expect(page.getByRole("alert")).toContainText(/notes\.txt/);
    await expect(page.getByTestId("queue")).toBeVisible();
  });
});

test.describe("accessibility and layout", () => {
  test("the dropzone is keyboard reachable", async ({ page }) => {
    await page.goto("/");
    const zone = page.getByRole("button", { name: /choose or drop a video/i });
    await zone.focus();
    await expect(zone).toBeFocused();
  });

  test("no horizontal overflow at phone width", async ({ page }) => {
    await page.goto("/");
    await page.setViewportSize({ width: 375, height: 812 });
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth + 1,
    );
    expect(overflow).toBe(false);
  });

  test("surfaces a clear error for a non-video file", async ({ page }) => {
    await page.goto("/");
    await page.setInputFiles('input[type="file"]', {
      name: "notes.txt",
      mimeType: "text/plain",
      buffer: Buffer.from("this is not a video"),
    });
    await expect(page.getByRole("alert")).toContainText(/does not look like a video/i);
  });
});
