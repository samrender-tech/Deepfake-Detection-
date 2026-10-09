"""F11 - calibration, and F12 - conformal abstention.

Both are fitted on the SOURCE validation split only, and both are then applied
unchanged to every target test set. That asymmetry is the point: measuring how
badly a source-fitted calibrator transfers to an unseen forgery dataset is a
finding, not a bug (F11: "cross-domain miscalibration is itself a finding
worth a paper paragraph").

Why this is in the critical path rather than a nicety: at cross-dataset
accuracy the raw sigmoid of a deep classifier is wildly overconfident --
routinely 0.99 on videos it gets wrong. Shipping that number to a user as
"99% confident this person's video is fake" is the single most harmful thing
this system could do.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ddetect.utils.log import get_logger

log = get_logger(__name__)


# ==========================================================================
# F11: calibration
# ==========================================================================
class TemperatureScaler(nn.Module):
    """Single-parameter logit scaling (Guo et al., 2017).

    One parameter means it cannot change the ranking, so AUC is untouched and
    only the probabilities move. That is exactly what is wanted: the
    discrimination result and the calibration result stay separable.
    """

    def __init__(self, init_t: float = 1.0) -> None:
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(float(np.log(init_t))))

    @property
    def temperature(self) -> float:
        return float(self.log_t.exp())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_t.exp().clamp(min=1e-3)

    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> TemperatureScaler:
        x = torch.as_tensor(np.asarray(logits, dtype=np.float32))
        y = torch.as_tensor(np.asarray(labels, dtype=np.float32))
        opt = torch.optim.LBFGS([self.log_t], lr=0.1, max_iter=max_iter)

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(self(x), y)
            loss.backward()
            return loss

        opt.step(closure)  # type: ignore[arg-type]
        t = self.temperature
        # A huge T flattens every probability to 0.5: the calibrator has
        # concluded the logits carry no information. That is the correct answer
        # on an uninformative model, but shipping it silently would turn every
        # verdict into a coin flip that LOOKS calibrated. Clamp and shout.
        if not (0.05 <= t <= 20.0):
            log.warning(
                "temperature scaling produced T=%.3f, outside [0.05, 20]. This "
                "means the validation logits are near-uninformative (too few "
                "val videos, or the model has not learned). Clamping to %.2f -- "
                "do NOT report calibrated probabilities from this run.",
                t,
                min(max(t, 0.05), 20.0),
            )
            with torch.no_grad():
                self.log_t.fill_(float(np.log(min(max(t, 0.05), 20.0))))
        else:
            log.info("temperature scaling fitted: T=%.4f", t)
        return self


class VectorScaler(nn.Module):
    """Affine logit scaling (a*z + b). Two parameters; can shift the operating
    point as well as sharpen, so it is reported alongside temperature scaling
    rather than instead of it."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.a * logits + self.b

    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> VectorScaler:
        x = torch.as_tensor(np.asarray(logits, dtype=np.float32))
        y = torch.as_tensor(np.asarray(labels, dtype=np.float32))
        opt = torch.optim.LBFGS([self.a, self.b], lr=0.1, max_iter=max_iter)

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(self(x), y)
            loss.backward()
            return loss

        opt.step(closure)  # type: ignore[arg-type]
        return self


@dataclass
class Calibration:
    """Serialisable calibration + abstention state for one run.

    Written to ``runs/{exp}/{seed}/calibration.json`` and loaded by
    ``Detector``, so the demo applies the identical transform the paper reports.
    """

    method: str = "temperature"
    temperature: float = 1.0
    a: float = 1.0
    b: float = 0.0
    threshold: float = 0.5
    threshold_criterion: str = "eer"
    #: F12 conformal band on the calibrated probability.
    conformal_lo: float = 0.5
    conformal_hi: float = 0.5
    conformal_alpha: float = 0.1
    #: F13 OOD threshold on the energy score.
    ood_threshold: float = float("inf")
    #: Provenance -- what this was fitted on. Printed in the paper's table.
    fitted_on: str = "UNSET"
    n_val: int = 0
    #: True when the fit was degenerate or the val split was too small to
    #: calibrate against. Surfaced by the API as a warning on every verdict.
    degenerate: bool = False

    # -- apply ----------------------------------------------------------
    def probability(self, logit: np.ndarray | float) -> np.ndarray:
        """logit -> calibrated probability."""
        z = np.asarray(logit, dtype=float)
        if self.method == "temperature":
            z = z / max(self.temperature, 1e-3)
        elif self.method == "vector":
            z = self.a * z + self.b
        elif self.method == "none":
            pass
        else:
            raise ValueError(f"unknown calibration method {self.method!r}")
        return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=float)

    def abstains(self, prob: np.ndarray | float) -> np.ndarray:
        """True where the calibrated probability falls inside the band."""
        p = np.asarray(prob, dtype=float)
        return np.asarray((p >= self.conformal_lo) & (p <= self.conformal_hi))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        p = Path(path)
        if not p.exists():
            log.warning(
                "no calibration.json at %s -- falling back to an UNCALIBRATED "
                "identity transform at threshold 0.5. Probabilities will be "
                "overconfident; do not report them.",
                p,
            )
            return cls(method="none", fitted_on="MISSING (uncalibrated)")
        return cls(**json.loads(p.read_text()))


# ==========================================================================
# F12: split conformal abstention
# ==========================================================================
def conformal_band(
    probs: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.1,
    threshold: float = 0.5,
) -> tuple[float, float]:
    """Derive an abstention band from source-val nonconformity scores.

    Split conformal prediction: the nonconformity of a validation point is
    ``1 - p_assigned_to_its_true_class``. Taking the (1-alpha) quantile gives a
    score level at which coverage is guaranteed (marginally, exchangeably) at
    1-alpha. Translating that back to probability space gives the band around
    the threshold inside which we refuse to call it.

    The guarantee is only valid under exchangeability, which cross-dataset
    evaluation violates by construction. We say so in the paper: the band is
    calibrated on the source domain and the *measured* coverage on each target
    set is reported, which is the honest version of the claim.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if len(p) == 0:
        return threshold, threshold

    true_class_p = np.where(y == 1, p, 1 - p)
    nonconf = 1.0 - true_class_p

    n = len(nonconf)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q = float(np.quantile(nonconf, q_level, method="higher"))

    # Standard conformal prediction SETS, then read off the abstention band.
    # Class k enters the set when its nonconformity 1 - p_k <= q, so:
    #     class 1 in set  <=>  p >= 1 - q
    #     class 0 in set  <=>  p <= q
    # The verdict is ambiguous exactly when the set is not a singleton:
    #     both classes  (q >= 0.5):  1 - q <= p <= q
    #     neither class (q <  0.5):      q <  p <  1 - q
    # Both cases are the interval between q and 1 - q, so one expression
    # covers them. (An earlier version derived the band from the threshold and
    # collapsed to zero width whenever q > 0.5 -- i.e. it silently stopped
    # abstaining on precisely the models that most needed to.)
    lo, hi = float(min(q, 1.0 - q)), float(max(q, 1.0 - q))
    log.info(
        "conformal band at alpha=%.2f: abstain when calibrated p in [%.3f, %.3f] "
        "(q=%.3f, n_val=%d). A wide band means the model is frequently wrong "
        "while confident -- that is information, not a defect.",
        alpha,
        lo,
        hi,
        q,
        n,
    )
    return lo, hi


def fit_calibration(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    method: str = "temperature",
    threshold_criterion: str = "eer",
    conformal_alpha: float = 0.1,
    fitted_on: str = "source val",
    ood_scores: np.ndarray | None = None,
    ood_quantile: float = 0.95,
) -> Calibration:
    """Fit calibration + threshold + conformal band + OOD cut, all on val.

    One function so there is exactly one place where "this was fitted on
    validation data" is true, and nothing downstream can accidentally refit on
    a test set.
    """
    from ddetect.metrics import threshold_from_val

    val_logits = np.asarray(val_logits, dtype=float)
    val_labels = np.asarray(val_labels, dtype=float)

    cal = Calibration(
        method=method,
        threshold_criterion=threshold_criterion,
        conformal_alpha=conformal_alpha,
        fitted_on=fitted_on,
        n_val=len(val_labels),
    )

    if method == "temperature":
        cal.temperature = TemperatureScaler().fit(val_logits, val_labels).temperature
    elif method == "vector":
        vs = VectorScaler().fit(val_logits, val_labels)
        cal.a, cal.b = float(vs.a), float(vs.b)
    elif method != "none":
        raise ValueError(f"unknown calibration method {method!r}")

    MIN_VAL_FOR_CALIBRATION = 100
    if cal.n_val < MIN_VAL_FOR_CALIBRATION:
        cal.degenerate = True
        log.warning(
            "calibrating on only %d validation videos (want >= %d). The "
            "temperature, threshold and conformal band are all unreliable; "
            "treat this run's probabilities as uncalibrated.",
            cal.n_val,
            MIN_VAL_FOR_CALIBRATION,
        )
    if method == "temperature" and not (0.05 <= cal.temperature <= 20.0):
        cal.degenerate = True

    probs = cal.probability(val_logits)
    # The threshold is derived on the CALIBRATED probabilities, so the
    # threshold and the band live in the same space.
    cal.threshold = float(threshold_from_val(val_labels, probs, threshold_criterion))
    cal.conformal_lo, cal.conformal_hi = conformal_band(
        probs, val_labels, conformal_alpha, cal.threshold
    )

    if ood_scores is not None and len(ood_scores):
        cal.ood_threshold = float(np.quantile(np.asarray(ood_scores, float), ood_quantile))
        log.info("OOD threshold at val q%.0f: %.4f", ood_quantile * 100, cal.ood_threshold)

    return cal
