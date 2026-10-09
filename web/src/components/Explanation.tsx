import type { AnalysisResult } from "../types/api";

/**
 * A7's plain-language explanation.
 *
 * The source is labelled honestly. When the LLM is unavailable or its output
 * failed the grounding guard, a deterministic template is shown instead, and
 * the user is told which they are reading — silently swapping them would
 * misrepresent how the text was produced.
 */
export function Explanation({ result }: { result: AnalysisResult }) {
  if (!result.explanation) return null;

  return (
    <section aria-labelledby="explanation-heading" className="rounded-xl bg-neutral-900 p-4 ring-1 ring-neutral-800">
      <div className="flex items-center justify-between gap-2">
        <h3 id="explanation-heading" className="text-sm font-medium text-neutral-200">
          In plain language
        </h3>
        <span className="text-[10px] uppercase tracking-wider text-neutral-500">
          {result.explanation_source === "agent" ? "generated summary" : "rule-based summary"}
        </span>
      </div>
      <p className="mt-2 text-sm leading-relaxed text-neutral-300">{result.explanation}</p>
    </section>
  );
}
