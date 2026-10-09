import type { AnalysisResult } from "../types/api";
import { VERDICT_LABEL } from "../types/api";

const STYLE = {
  likely_authentic: "bg-authentic-bg text-authentic-fg ring-authentic-ring",
  likely_manipulated: "bg-manipulated-bg text-manipulated-fg ring-manipulated-ring",
  inconclusive: "bg-inconclusive-bg text-inconclusive-fg ring-inconclusive-ring",
} as const;

/**
 * The verdict, its calibrated probability and its conformal band.
 *
 * Three states, never two. "Inconclusive" is styled as a
 * first-class outcome, not an error: at cross-dataset accuracy it is often
 * the correct answer, and making it look like a failure would push users to
 * read the raw percentage instead.
 */
export function VerdictCard({ result }: { result: AnalysisResult }) {
  const pct = Math.round(result.calibrated_probability * 100);
  const [lo, hi] = result.confidence_interval;
  const thresholdPct = Math.round(result.decision_threshold * 100);

  return (
    <section
      aria-labelledby="verdict-heading"
      className={`rounded-xl p-5 ring-1 ${STYLE[result.verdict]}`}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-[11px] uppercase tracking-widest opacity-70">Verdict</p>
          <h2 id="verdict-heading" className="mt-0.5 text-2xl font-semibold">
            {VERDICT_LABEL[result.verdict]}
          </h2>
        </div>
        <div className="text-right">
          <p className="text-[11px] uppercase tracking-widest opacity-70">
            Estimated probability of manipulation
          </p>
          <p className="mt-0.5 text-3xl font-semibold tabular-nums">{pct}%</p>
        </div>
      </div>

      {/* The probability bar, with the abstention band and threshold marked.
          Showing the band is the point: it is where the model declines to
          call it, and a bare percentage hides that. */}
      <div className="mt-5">
        <div className="relative h-7 w-full overflow-hidden rounded-md bg-black/40">
          <div
            className="absolute inset-y-0 bg-white/10"
            style={{ left: `${lo * 100}%`, width: `${Math.max((hi - lo) * 100, 0.5)}%` }}
            title={`Inconclusive band: ${Math.round(lo * 100)}%–${Math.round(hi * 100)}%`}
          />
          <div
            className="absolute inset-y-0 w-0.5 bg-current opacity-60"
            style={{ left: `${thresholdPct}%` }}
            title={`Decision threshold: ${thresholdPct}%`}
          />
          <div
            className="absolute inset-y-1 w-1.5 rounded-full bg-current"
            style={{ left: `calc(${pct}% - 3px)` }}
            aria-hidden
          />
        </div>
        <div className="mt-1.5 flex justify-between text-[11px] opacity-60">
          <span>0% authentic</span>
          <span>
            shaded = inconclusive band ({Math.round(lo * 100)}–{Math.round(hi * 100)}%)
            · line = threshold {thresholdPct}%
          </span>
          <span>100% manipulated</span>
        </div>
      </div>

      {!result.calibration_reliable && (
        <p className="mt-4 rounded-md bg-black/30 p-2.5 text-xs leading-relaxed">
          <strong>This percentage is not trustworthy.</strong> The loaded model's
          confidence calibration was fitted on too little validation data. Use the
          verdict as a rough indication only.
        </p>
      )}
    </section>
  );
}
