import { STAGE_LABEL, type Stage } from "../types/api";

const ORDER: Stage[] = [
  "queued", "probe", "preprocess", "infer", "calibrate", "explain", "persist", "done",
];

/** Stage tracker driven by the SSE stream. */
export function Progress({
  stage,
  progress,
  uploadFraction,
  uploading,
  filename,
}: {
  stage: Stage;
  progress: number;
  uploadFraction: number;
  uploading: boolean;
  filename?: string;
}) {
  const pct = uploading ? uploadFraction * 100 : Math.max(progress * 100, 4);
  const current = ORDER.indexOf(stage);

  return (
    <section aria-live="polite" className="rounded-xl bg-neutral-900 p-4 ring-1 ring-neutral-800">
      <div className="flex items-baseline justify-between gap-2">
        <p className="text-sm text-neutral-200">
          {uploading ? "Uploading" : STAGE_LABEL[stage]}
          {filename && <span className="text-neutral-500"> · {filename}</span>}
        </p>
        <p className="text-xs tabular-nums text-neutral-500">{pct.toFixed(0)}%</p>
      </div>
      <div className="mt-2.5 h-2 w-full overflow-hidden rounded-full bg-neutral-800">
        <div
          className="h-full rounded-full bg-violet-400 transition-[width] duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
      {!uploading && (
        <ol className="mt-3 flex flex-wrap gap-x-3 gap-y-1 text-[11px]">
          {ORDER.slice(1, -1).map((s, i) => (
            <li
              key={s}
              className={
                i + 1 < current
                  ? "text-emerald-400/70"
                  : i + 1 === current
                    ? "text-neutral-100"
                    : "text-neutral-600"
              }
            >
              {i + 1 < current ? "✓ " : ""}
              {STAGE_LABEL[s]}
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
