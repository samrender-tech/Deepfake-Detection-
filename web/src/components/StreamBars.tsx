import type { AnalysisResult } from "../types/api";

const LABEL = {
  visual: "Picture",
  audio: "Voice",
  sync: "Lip-sync",
} as const;

const DESC = {
  visual: "Blending artefacts and texture inconsistency in the face region",
  audio: "Signs of synthetic or cloned speech",
  sync: "Whether the lips match the speech",
} as const;

/**
 * Per-stream scores and the reliability gate's weighting.
 *
 * Showing the gate weight is what makes a visual-only verdict legible: the
 * user can see that the audio stream contributed nothing and why, rather than
 * wondering whether it was consulted.
 */
export function StreamBars({ result }: { result: AnalysisResult }) {
  if (!result.streams.length) return null;

  return (
    <section aria-labelledby="streams-heading" className="rounded-xl bg-neutral-900 p-4 ring-1 ring-neutral-800">
      <h3 id="streams-heading" className="text-sm font-medium text-neutral-200">
        What each check found
      </h3>
      <ul className="mt-3 space-y-3">
        {result.streams.map((s) => {
          const weight = result.gate_weights[s.name];
          return (
            <li key={s.name}>
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-sm text-neutral-200">{LABEL[s.name]}</span>
                <span className="text-xs tabular-nums text-neutral-400">
                  {s.available && s.score !== null ? s.score.toFixed(3) : "not run"}
                </span>
              </div>
              <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-neutral-800">
                {s.available && s.score !== null ? (
                  <div
                    className="h-full rounded-full bg-violet-400"
                    style={{ width: `${Math.max(s.score * 100, 1.5)}%` }}
                  />
                ) : (
                  <div className="h-full w-full bg-[repeating-linear-gradient(45deg,#27272a_0,#27272a_4px,#1c1c1f_4px,#1c1c1f_8px)]" />
                )}
              </div>
              <p className="mt-1 text-[11px] leading-snug text-neutral-500">
                {s.note || DESC[s.name]}
                {typeof weight === "number" && (
                  <>
                    {" · "}
                    weight {weight.toFixed(2)}
                    {weight === 0 && " (ignored for this clip)"}
                  </>
                )}
              </p>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
