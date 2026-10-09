import { useState } from "react";
import type { AnalysisResult } from "../types/api";

/**
 * Grad-CAM overlays and the audio spectrum: the raw evidence behind the
 * verdict.
 *
 * Presented with an explicit caveat. A heat map shows where the model's
 * gradient was largest, which is informative but is not proof of anything --
 * saying so is part of not overclaiming.
 */
export function Evidence({
  result,
  selectedFrame,
}: {
  result: AnalysisResult;
  selectedFrame?: number | null;
}) {
  const keys = Object.keys(result.gradcam_urls);
  const [open, setOpen] = useState(false);
  if (!keys.length && !result.spectrum_url) return null;

  const preferred =
    selectedFrame != null && result.gradcam_urls[`frame_${selectedFrame}`]
      ? [`frame_${selectedFrame}`, ...keys.filter((k) => k !== `frame_${selectedFrame}`)]
      : keys;

  return (
    <section aria-labelledby="evidence-heading" className="rounded-xl bg-neutral-900 p-4 ring-1 ring-neutral-800">
      <div className="flex items-center justify-between gap-2">
        <h3 id="evidence-heading" className="text-sm font-medium text-neutral-200">
          Where the model looked
        </h3>
        <button
          onClick={() => setOpen((v) => !v)}
          className="rounded-md px-2 py-1 text-xs text-neutral-400 ring-1 ring-neutral-700 hover:text-neutral-200"
          aria-expanded={open}
        >
          {open ? "Hide detail" : "What is this?"}
        </button>
      </div>

      {open && (
        <p className="mt-2 rounded-md bg-neutral-950/60 p-2.5 text-[11px] leading-relaxed text-neutral-400">
          Warm regions are where the model's score was most sensitive to the
          image — typically blend boundaries around the jaw, hairline or mouth.
          It indicates what the model reacted to, not proof that the region was
          edited. A heat map on an authentic video is normal.
        </p>
      )}

      {preferred.length > 0 && (
        <div className="mt-3 grid grid-cols-3 gap-2">
          {preferred.slice(0, 3).map((k) => (
            <figure key={k} className="overflow-hidden rounded-lg ring-1 ring-neutral-800">
              <img
                src={result.gradcam_urls[k]}
                alt={`Model attention overlay for ${k.replace("frame_", "frame ")}`}
                className="aspect-square w-full object-cover"
                loading="lazy"
              />
              <figcaption className="bg-neutral-950/80 px-2 py-1 text-[10px] text-neutral-400">
                {k.replace("frame_", "frame ")}
                {selectedFrame != null && k === `frame_${selectedFrame}` && " · selected"}
              </figcaption>
            </figure>
          ))}
        </div>
      )}

      {result.spectrum_url && (
        <figure className="mt-3">
          <img
            src={result.spectrum_url}
            alt="Audio frequency spectrum of the clip"
            className="w-full rounded-lg bg-neutral-950 ring-1 ring-neutral-800"
            loading="lazy"
          />
          <figcaption className="mt-1 text-[11px] text-neutral-500">
            Audio spectrum. Synthetic speech and heavy re-encoding often show a
            sharp high-frequency cutoff.
          </figcaption>
        </figure>
      )}
    </section>
  );
}
