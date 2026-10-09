import { useCallback, useEffect, useState } from "react";
import { deleteAnalysis, listAnalyses } from "../api/client";
import type { AnalysisStatus, Verdict } from "../types/api";
import { VERDICT_LABEL } from "../types/api";

const DOT: Record<Verdict, string> = {
  likely_authentic: "bg-authentic-fg",
  likely_manipulated: "bg-manipulated-fg",
  inconclusive: "bg-inconclusive-fg",
};

/**
 * Recent analyses, with a working delete.
 *
 * The delete button is not a convenience: the service stores verdicts about
 * whether a real person's video is fake, and the retention promise in the
 * model card has to be actionable by the person who uploaded it, not only by
 * a timer.
 */
export function History({
  onOpen,
  refreshKey,
}: {
  onOpen?: (jobId: string) => void;
  refreshKey?: number;
}) {
  const [rows, setRows] = useState<AnalysisStatus[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      // Through the client, so the API key is sent when one is configured.
      setRows(await listAnalyses(20));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load, refreshKey]);

  const remove = async (jobId: string) => {
    setBusy(jobId);
    try {
      await deleteAnalysis(jobId);
      setRows((rs) => rs.filter((r) => r.job_id !== jobId));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  if (!rows.length && !error) return null;

  return (
    <section
      aria-labelledby="history-heading"
      className="rounded-xl bg-neutral-900 p-4 ring-1 ring-neutral-800"
      data-testid="history"
    >
      <div className="flex items-center justify-between">
        <h3 id="history-heading" className="text-sm font-medium text-neutral-200">
          Recent analyses
        </h3>
        <button
          onClick={() => void load()}
          className="rounded-md px-2 py-1 text-xs text-neutral-400 ring-1 ring-neutral-700 hover:text-neutral-200"
        >
          Refresh
        </button>
      </div>

      {error && (
        <p role="alert" className="mt-2 text-xs text-amber-200">
          Could not load history: {error}
        </p>
      )}

      <ul className="mt-3 divide-y divide-neutral-800">
        {rows.map((r) => (
          <li key={r.job_id} className="flex items-center gap-3 py-2">
            <span
              aria-hidden
              className={`h-2 w-2 shrink-0 rounded-full ${
                r.result ? DOT[r.result.verdict] : "bg-neutral-600"
              }`}
            />
            <button
              onClick={() => r.result && onOpen?.(r.job_id)}
              disabled={!r.result || !onOpen}
              title={r.result && onOpen ? "Open this analysis" : undefined}
              className="min-w-0 flex-1 rounded text-left hover:bg-neutral-800/60 disabled:cursor-default disabled:hover:bg-transparent"
            >
              <span className="block truncate text-xs text-neutral-200">
                {r.filename ?? r.job_id.slice(0, 8)}
              </span>
              <span className="block text-[11px] text-neutral-500">
                {r.result
                  ? `${VERDICT_LABEL[r.result.verdict]} · ${Math.round(
                      r.result.calibrated_probability * 100,
                    )}%`
                  : r.state === "failed"
                    ? "failed"
                    : "in progress"}
              </span>
            </button>
            <time
              dateTime={r.created_at}
              className="shrink-0 text-[11px] tabular-nums text-neutral-600"
            >
              {new Date(r.created_at).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </time>
            <button
              onClick={() => void remove(r.job_id)}
              disabled={busy === r.job_id}
              aria-label={`Delete analysis of ${r.filename ?? r.job_id}`}
              className="shrink-0 rounded px-1.5 py-0.5 text-[11px] text-neutral-500 hover:bg-neutral-800 hover:text-neutral-200 disabled:opacity-50"
            >
              {busy === r.job_id ? "…" : "Delete"}
            </button>
          </li>
        ))}
      </ul>

      <p className="mt-3 text-[11px] leading-relaxed text-neutral-600">
        Deleting removes the stored result and its images immediately. Anything
        not deleted expires on the service's retention timer.
      </p>
    </section>
  );
}
