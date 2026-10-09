import type { QueueItem } from "../hooks/useQueue";
import { VERDICT_LABEL } from "../types/api";

const DOT = {
  likely_authentic: "bg-authentic-fg",
  likely_manipulated: "bg-manipulated-fg",
  inconclusive: "bg-inconclusive-fg",
} as const;

const STATE_LABEL = {
  waiting: "waiting",
  uploading: "uploading…",
  analysing: "analysing…",
  done: "",
  failed: "failed",
} as const;

/** The multi-file queue: live status per clip, and a way into each result. */
export function Queue({
  items,
  onOpen,
  onClear,
}: {
  items: QueueItem[];
  onOpen: (jobId: string) => void;
  onClear: () => void;
}) {
  if (!items.length) return null;
  const finished = items.filter((x) => x.state === "done" || x.state === "failed").length;

  return (
    <section
      aria-labelledby="queue-heading"
      className="rounded-xl bg-neutral-900 p-4 ring-1 ring-neutral-800"
      data-testid="queue"
    >
      <div className="flex items-center justify-between">
        <h3 id="queue-heading" className="text-sm font-medium text-neutral-200">
          Batch · {finished} of {items.length} finished
        </h3>
        {finished === items.length && (
          <button
            onClick={onClear}
            className="rounded-md px-2 py-1 text-xs text-neutral-400 ring-1 ring-neutral-700 hover:text-neutral-200"
          >
            Clear
          </button>
        )}
      </div>
      <div
        className="mt-3 h-1 overflow-hidden rounded-full bg-neutral-800"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={items.length}
        aria-valuenow={finished}
      >
        <div
          className="h-full bg-violet-400 transition-all"
          style={{ width: `${(100 * finished) / items.length}%` }}
        />
      </div>
      <ul className="mt-3 divide-y divide-neutral-800">
        {items.map((x) => (
          <li key={x.key} className="flex items-center gap-3 py-2">
            <span
              aria-hidden
              className={`h-2 w-2 shrink-0 rounded-full ${
                x.verdict ? DOT[x.verdict] : x.state === "failed" ? "bg-rose-500" : "bg-neutral-600"
              }`}
            />
            <span className="min-w-0 flex-1">
              <span className="block truncate text-xs text-neutral-200">{x.name}</span>
              <span className="block text-[11px] text-neutral-500">
                {x.verdict
                  ? `${VERDICT_LABEL[x.verdict]} · ${Math.round((x.probability ?? 0) * 100)}%`
                  : x.state === "failed"
                    ? `failed — ${x.error ?? "unknown error"}`
                    : STATE_LABEL[x.state]}
              </span>
            </span>
            {x.state === "done" && x.jobId && (
              <button
                onClick={() => onOpen(x.jobId!)}
                aria-label={`Open result for ${x.name}`}
                className="shrink-0 rounded px-2 py-0.5 text-[11px] text-neutral-300 ring-1 ring-neutral-700 hover:bg-neutral-800"
              >
                Open
              </button>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
