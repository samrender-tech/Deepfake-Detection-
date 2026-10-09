import { useEffect, useState } from "react";
import { getModelInfo } from "../api/client";
import type { ModelInfo } from "../types/api";

const pct = (p: number) => `${Math.round(p * 100)}%`;

/**
 * What model is answering, and how it turns a score into a verdict.
 *
 * Collapsed by default: most users want the verdict, but anyone asking "why
 * is 55% inconclusive?" should be able to see the band without reading code.
 */
export function ModelPanel() {
  const [info, setInfo] = useState<ModelInfo | null>(null);

  useEffect(() => {
    getModelInfo()
      .then(setInfo)
      .catch(() => setInfo(null)); // no model loaded: the upload path reports it
  }, []);

  if (!info) return null;
  const [lo, hi] = info.conformal_band;

  return (
    <details className="group rounded-xl bg-neutral-900/60 p-4 ring-1 ring-neutral-800">
      <summary className="cursor-pointer list-none text-sm font-medium text-neutral-300">
        <span className="mr-1.5 inline-block transition-transform group-open:rotate-90">›</span>
        About this model
      </summary>
      <p className="mt-2 text-xs leading-relaxed text-neutral-400">
        A probability at or above {pct(info.decision_threshold)} reads as likely manipulated.
        Anything between {pct(lo)} and {pct(hi)} is reported as inconclusive rather than
        forced into a call.
        {!info.calibration_reliable &&
          " This model's calibration was fitted on too little data, so treat its percentages as indicative."}
      </p>
      <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-[11px] text-neutral-500">
        <dt>Model</dt>
        <dd className="truncate font-mono">{info.model_version}</dd>
        <dt>Architecture</dt>
        <dd className="font-mono">{info.architecture}</dd>
        <dt>Trained on</dt>
        <dd className="truncate font-mono">{info.trained_on}</dd>
        <dt>Input</dt>
        <dd className="tabular-nums">
          {info.n_frames} frames · {info.image_size}px · {info.audio_seconds}s audio
        </dd>
        <dt>Calibration</dt>
        <dd>
          {info.calibration_method} (T={info.temperature.toFixed(2)}, n={info.calibration_n_val})
        </dd>
        <dt>Threshold rule</dt>
        <dd>{info.threshold_criterion}</dd>
        <dt>Abstention level</dt>
        <dd className="tabular-nums">α = {info.conformal_alpha}</dd>
        <dt>Unfamiliar-input check</dt>
        <dd>{info.ood_detection ? "on" : "not fitted"}</dd>
        <dt>Device</dt>
        <dd className="font-mono">{info.device}</dd>
      </dl>
    </details>
  );
}
