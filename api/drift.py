"""F28 - input and score drift monitoring.

The question this answers: *is live traffic still the kind of video this model
was calibrated on?* A detector whose cross-dataset accuracy is already its
headline weakness degrades silently when the input distribution moves, and the
first sign is usually not an error but a shift in the shape of the scores.

Two monitors:

* **Score drift (PSI).** Population Stability Index between the live
  calibrated-probability histogram and the validation baseline. The
  conventional reading -- <0.1 stable, 0.1-0.25 moderate, >0.25 significant --
  comes from credit scoring and is a rule of thumb, not a test, so it is
  reported with the raw number rather than as a verdict.

* **Input drift.** Resolution, duration, codec, face count and audio presence
  against what the training manifests contained. These are the dimensions the
  preprocessing cache is sensitive to.

The baseline is built from a run's own ``preds_val.csv``, so it is the same
distribution the threshold and the conformal band were fitted on.
"""

from __future__ import annotations

import os
import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from api.observability import get_log

log = get_log(__name__)

N_BINS = 10
#: Conventional PSI thresholds. Reported alongside the raw value so a reader
#: is not forced to accept the convention.
PSI_MODERATE, PSI_SIGNIFICANT = 0.10, 0.25


def population_stability_index(
    baseline: np.ndarray, live: np.ndarray, n_bins: int = N_BINS
) -> float:
    """PSI between two samples over [0, 1].

    Both histograms are smoothed by a small epsilon: an empty bin would
    otherwise make the log term infinite, and early live traffic has many
    empty bins purely because it is small.
    """
    if len(baseline) < 10 or len(live) < 10:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.histogram(np.clip(baseline, 0, 1), bins=edges)[0].astype(float)
    liv = np.histogram(np.clip(live, 0, 1), bins=edges)[0].astype(float)
    b = (b + 1e-6) / (b.sum() + 1e-6 * n_bins)
    liv = (liv + 1e-6) / (liv.sum() + 1e-6 * n_bins)
    return float(np.sum((liv - b) * np.log(liv / b)))


@dataclass
class Baseline:
    """The validation distribution a run was calibrated against."""

    scores: list[float] = field(default_factory=list)
    has_audio_rate: float = 0.0
    n: int = 0
    source: str = "unset"

    @classmethod
    def from_run(cls, run_dir: str | Path) -> Baseline:
        p = Path(run_dir) / "preds_val.csv"
        if not p.exists():
            return cls(source=f"MISSING ({p})")
        import pandas as pd

        df = pd.read_csv(p)
        col = "calibrated_prob" if "calibrated_prob" in df else "video_score"
        return cls(
            scores=df[col].astype(float).tolist(),
            has_audio_rate=float(df.has_audio.astype(bool).mean()) if "has_audio" in df else 0.0,
            n=len(df),
            source=str(p),
        )


@dataclass
class DriftMonitor:
    """Rolling window of live observations, compared against the baseline."""

    baseline: Baseline
    window: int = 500
    min_samples: int = 50

    def __post_init__(self) -> None:
        self._scores: deque[float] = deque(maxlen=self.window)
        self._audio: deque[bool] = deque(maxlen=self.window)
        self._faces: deque[int] = deque(maxlen=self.window)
        self._duration: deque[float] = deque(maxlen=self.window)
        self._resolution: deque[tuple[int, int]] = deque(maxlen=self.window)
        self._codec: Counter[str] = Counter()
        self._ood = 0
        self._abstain = 0
        self._total = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def observe_upload(self, width: int, height: int, duration_s: float, codec: str) -> None:
        with self._lock:
            self._resolution.append((width, height))
            self._duration.append(duration_s)
            self._codec[codec] += 1

    def observe_result(self, result: dict[str, Any]) -> None:
        with self._lock:
            self._total += 1
            p = result.get("calibrated_prob")
            if isinstance(p, (int, float)):
                self._scores.append(float(p))
            self._audio.append(bool(result.get("has_audio")))
            self._faces.append(int(result.get("n_faces_found", 0)))
            if result.get("ood_flag"):
                self._ood += 1
            if result.get("verdict") == "inconclusive":
                self._abstain += 1

    # ------------------------------------------------------------------
    def report(self) -> dict[str, Any]:
        with self._lock:
            n = len(self._scores)
            scores = np.array(self._scores, dtype=float)
            audio_rate = float(np.mean(self._audio)) if self._audio else 0.0
            faces = np.array(self._faces, dtype=float) if self._faces else np.array([])
            durations = np.array(self._duration, dtype=float) if self._duration else np.array([])
            res = list(self._resolution)
            codecs = dict(self._codec.most_common(5))
            ood_rate = self._ood / max(self._total, 1)
            abstain_rate = self._abstain / max(self._total, 1)

        out: dict[str, Any] = {
            "n_observed": n,
            "baseline_n": self.baseline.n,
            "baseline_source": self.baseline.source,
            "ready": n >= self.min_samples,
            "ood_rate": round(ood_rate, 4),
            "abstain_rate": round(abstain_rate, 4),
            "codecs": codecs,
        }
        if not out["ready"]:
            out["note"] = f"need {self.min_samples} analyses before drift is meaningful; have {n}"
            return out

        psi = population_stability_index(np.array(self.baseline.scores), scores)
        out["score_psi"] = round(psi, 4) if psi == psi else None
        out["score_drift"] = (
            (
                "significant"
                if psi > PSI_SIGNIFICANT
                else "moderate"
                if psi > PSI_MODERATE
                else "stable"
            )
            if psi == psi
            else "unknown"
        )
        out["score_mean_live"] = round(float(scores.mean()), 4)
        out["score_mean_baseline"] = (
            round(float(np.mean(self.baseline.scores)), 4) if self.baseline.scores else None
        )
        out["has_audio_rate_live"] = round(audio_rate, 4)
        out["has_audio_rate_baseline"] = round(self.baseline.has_audio_rate, 4)
        if faces.size:
            out["no_face_rate"] = round(float((faces == 0).mean()), 4)
            out["median_faces"] = float(np.median(faces))
        if durations.size:
            out["median_duration_s"] = round(float(np.median(durations)), 2)
        if res:
            heights = np.array([h for _, h in res], dtype=float)
            out["median_height_px"] = float(np.median(heights))
            out["small_video_rate"] = round(float((heights < 360).mean()), 4)

        alerts = []
        if psi == psi and psi > PSI_SIGNIFICANT:
            alerts.append(
                f"score distribution has shifted (PSI {psi:.3f}); the calibration "
                f"and abstention band were fitted on a different distribution"
            )
        if ood_rate > 0.30:
            alerts.append(
                f"{ood_rate:.0%} of inputs flagged unlike the training data; the "
                f"model is being used outside its domain"
            )
        if out.get("no_face_rate", 0) > 0.25:
            alerts.append(
                f"{out['no_face_rate']:.0%} of uploads have no detectable face; "
                f"verdicts on those are unreliable"
            )
        if abs(audio_rate - self.baseline.has_audio_rate) > 0.5:
            alerts.append(
                "audio presence differs sharply from the calibration set, so the "
                "fusion gate is operating in an untested regime"
            )
        out["alerts"] = alerts
        return out


_MONITOR: DriftMonitor | None = None


def get_monitor() -> DriftMonitor:
    global _MONITOR
    if _MONITOR is None:
        run_dir = os.environ.get("DDETECT_RUN_DIR", "")
        _MONITOR = DriftMonitor(baseline=Baseline.from_run(run_dir) if run_dir else Baseline())
        log.info("drift monitor ready", baseline=_MONITOR.baseline.source, n=_MONITOR.baseline.n)
    return _MONITOR


def reset_monitor() -> None:
    """Test hook."""
    global _MONITOR
    _MONITOR = None
