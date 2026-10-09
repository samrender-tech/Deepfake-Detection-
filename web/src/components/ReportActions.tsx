import { useState } from "react";
import { downloadReport } from "../api/client";
import type { ReportFormat } from "../types/api";

/**
 * Save a copy of this analysis.
 *
 * The HTML report is self-contained (heat maps inlined, no network requests)
 * and carries the same disclaimer as this page, because a downloaded report
 * is precisely the thing that gets forwarded without its context.
 */
export function ReportActions({ jobId }: { jobId: string }) {
  const [busy, setBusy] = useState<ReportFormat | null>(null);
  const [error, setError] = useState<string | null>(null);

  const get = async (format: ReportFormat) => {
    setBusy(format);
    setError(null);
    try {
      await downloadReport(jobId, format);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const btn =
    "rounded-md px-3 py-1.5 text-xs text-neutral-200 ring-1 ring-neutral-700 hover:bg-neutral-800 disabled:opacity-50";

  return (
    <section
      aria-labelledby="report-heading"
      className="rounded-xl bg-neutral-900 p-4 ring-1 ring-neutral-800"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 id="report-heading" className="text-sm font-medium text-neutral-200">
            Save a report
          </h3>
          <p className="mt-0.5 text-xs text-neutral-400">
            A single file you can keep after the result expires. It includes the
            limitations below.
          </p>
        </div>
        <div className="flex gap-2">
          <button onClick={() => void get("html")} disabled={busy !== null} className={btn}>
            {busy === "html" ? "Preparing…" : "Download HTML"}
          </button>
          <button onClick={() => void get("json")} disabled={busy !== null} className={btn}>
            {busy === "json" ? "Preparing…" : "Download JSON"}
          </button>
        </div>
      </div>
      {error && (
        <p role="alert" className="mt-2 text-xs text-amber-200">
          Could not build the report: {error}
        </p>
      )}
    </section>
  );
}
