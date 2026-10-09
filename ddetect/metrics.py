"""F16 - metrics.

Reads only ``preds_*.csv`` (contract 5.4), never a model. That is what lets the
whole evaluation stack be written and unit-tested before any model trains.

Two rules encoded here rather than left to discipline:

1. **The threshold comes from the source-domain validation split.** Picking a
   threshold on the test set inflates accuracy and F1 by several points and is
   the most common way a cross-dataset number gets quietly faked.
   ``threshold_from_val`` is the only supported way to obtain one, and
   ``classification_metrics`` demands one be passed in.

2. **Every headline number carries an interval.** A single AUC from a single
   seed on a few hundred test videos is not evidence. DeLong gives the
   analytic CI; the bootstrap covers everything else.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

EPS = 1e-12


# ==========================================================================
# core discrimination metrics
# ==========================================================================
def auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC AUC. Returns NaN for a single-class input rather than raising.

    A single-class test set happens legitimately (e.g. a per-method breakdown
    on fakes only), and an exception there would kill a whole results table.
    """
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def eer(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Equal error rate and the threshold achieving it."""
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def partial_auc(y: np.ndarray, s: np.ndarray, max_fpr: float = 0.1) -> float:
    """Standardised pAUC over FPR <= max_fpr.

    The operating region that matters: a detector that only separates well at
    30% false-positive rate is useless for flagging content, because the
    false accusations swamp the true ones.
    """
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s, max_fpr=max_fpr))


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, target_fpr: float) -> float:
    """Detection rate at a fixed false-positive budget."""
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(target_fpr, fpr, tpr))


def delong_auc_variance(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """AUC and its DeLong variance.

    DeLong et al. (1988) via the Sun & Xu (2014) fast midrank formulation.
    Gives an analytic CI and, more usefully, supports the *paired* test for
    "is Proposed significantly better than Baseline on the same videos",
    which a bootstrap over independent samples does not.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan"), float("nan")

    def midrank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x)
        xs = x[order]
        r = np.zeros(len(x), dtype=float)
        i = 0
        while i < len(xs):
            j = i
            while j < len(xs) - 1 and xs[j + 1] == xs[i]:
                j += 1
            r[i : j + 1] = 0.5 * (i + j) + 1
            i = j + 1
        out = np.empty(len(x), dtype=float)
        out[order] = r
        return out

    tx, ty, tz = midrank(pos), midrank(neg), midrank(s)
    tz_pos, tz_neg = tz[y == 1], tz[y == 0]

    a = (tz_pos.sum() - m * (m + 1) / 2) / (m * n)
    v01 = (tz_pos - tx) / n
    v10 = 1 - (tz_neg - ty) / m
    var = np.var(v01, ddof=1) / m + np.var(v10, ddof=1) / n
    return float(a), float(var)


def delong_ci(y: np.ndarray, s: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """(auc, lo, hi) via DeLong, logit-transformed to stay inside [0,1].

    Falls back to ``bootstrap_ci`` when the DeLong variance degenerates (a
    perfectly separable test set), so a near-ceiling AUC still reports an
    interval instead of NaN.
    """
    from scipy.stats import norm

    a, var = delong_auc_variance(y, s)
    if not np.isfinite(a):
        return a, float("nan"), float("nan")
    # Degenerate cases: a perfectly separable test set gives zero variance, and
    # the logit transform below is undefined at exactly 0 or 1. Baseline A is
    # expected to sit near the ceiling, so this path is routine, not exotic --
    # fall back to the bootstrap, which stays finite there.
    if not np.isfinite(var) or var <= 0 or a >= 1 - 1e-9 or a <= 1e-9:
        return bootstrap_ci(y, s, auc, n_boot=1000, alpha=alpha)
    z = norm.ppf(1 - alpha / 2)
    # Logit scale: a symmetric interval on the raw AUC can exceed 1.0 when
    # the AUC is near the ceiling, which is exactly where Baseline A sits.
    la = np.log(a / (1 - a + EPS) + EPS)
    se = np.sqrt(var) / (a * (1 - a) + EPS)
    lo, hi = la - z * se, la + z * se
    return a, float(1 / (1 + np.exp(-lo))), float(1 / (1 + np.exp(-hi)))


def delong_paired_test(y: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> tuple[float, float]:
    """Paired AUC comparison on the same videos -> (auc_diff, two-sided p).

    Correlation between the two models' scores is estimated by bootstrap, which
    is a documented simplification of the full DeLong covariance; it is
    conservative in practice and avoids a fragile hand-rolled covariance.
    """
    from scipy.stats import norm

    a1, v1 = delong_auc_variance(y, s1)
    a2, v2 = delong_auc_variance(y, s2)
    if not all(np.isfinite([a1, a2, v1, v2])):
        return float("nan"), float("nan")

    rng = np.random.default_rng(0)
    n = len(y)
    d1, d2 = [], []
    for _ in range(200):
        idx = rng.integers(0, n, n)
        if len(np.unique(np.asarray(y)[idx])) < 2:
            continue
        d1.append(auc(np.asarray(y)[idx], np.asarray(s1)[idx]))
        d2.append(auc(np.asarray(y)[idx], np.asarray(s2)[idx]))
    cov = float(np.cov(d1, d2)[0, 1]) if len(d1) > 10 else 0.0

    var = max(v1 + v2 - 2 * cov, EPS)
    z = (a1 - a2) / np.sqrt(var)
    return float(a1 - a2), float(2 * (1 - norm.cdf(abs(z))))


def mcnemar_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> tuple[int, int, float]:
    """Paired accuracy comparison on binary predictions -> (b, c, p).

    b = model1 right / model2 wrong, c = the reverse. Exact binomial, so it is
    valid on the small discordant counts a few-hundred-video test set produces
    (where the chi-square approximation is not).
    """
    from scipy.stats import binomtest

    c1 = np.asarray(p1) == np.asarray(y)
    c2 = np.asarray(p2) == np.asarray(y)
    b = int(np.sum(c1 & ~c2))
    c = int(np.sum(~c1 & c2))
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(b, b + c, 0.5).pvalue)


def bootstrap_ci(
    y: np.ndarray,
    s: np.ndarray,
    fn: Any = auc,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for any metric -> (point, lo, hi)."""
    y, s = np.asarray(y), np.asarray(s)
    point = fn(y, s)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        v = fn(y[idx], s[idx])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 50:
        return point, float("nan"), float("nan")
    return (
        point,
        float(np.percentile(vals, 100 * alpha / 2)),
        float(np.percentile(vals, 100 * (1 - alpha / 2))),
    )


# ==========================================================================
# thresholds -- the provenance rule
# ==========================================================================
def threshold_from_val(y_val: np.ndarray, s_val: np.ndarray, criterion: str = "eer") -> float:
    """Choose a decision threshold on the SOURCE validation split.

    The only sanctioned way to get a threshold. ``criterion``:
      eer  equal error rate (default; balanced and dataset-size independent)
      f1   maximises F1 on val
      youden  maximises tpr - fpr
    """
    y_val, s_val = np.asarray(y_val), np.asarray(s_val)
    if len(np.unique(y_val)) < 2:
        return 0.5
    if criterion == "eer":
        return eer(y_val, s_val)[1]
    fpr, tpr, thr = roc_curve(y_val, s_val)
    if criterion == "youden":
        return float(thr[int(np.argmax(tpr - fpr))])
    if criterion == "f1":
        best, best_t = -1.0, 0.5
        for t in np.unique(np.round(s_val, 4)):
            f = f1_score(y_val, (s_val >= t).astype(int), zero_division=0)
            if f > best:
                best, best_t = f, float(t)
        return best_t
    raise ValueError(f"criterion must be eer|f1|youden, got {criterion!r}")


# ==========================================================================
# calibration
# ==========================================================================
def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """ECE with equal-width bins."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= edges[i]) & (p <= edges[i + 1])
        if not m.any():
            continue
        ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def max_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    worst = 0.0
    for i in range(n_bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum() >= 10:  # ignore near-empty bins; they are noise
            worst = max(worst, abs(y[m].mean() - p[m].mean()))
    return float(worst)


def reliability_curve(
    y: np.ndarray, p: np.ndarray, n_bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bin_centres, observed_freq, bin_counts) for the reliability diagram."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    centres, obs, cnt = [], [], []
    for i in range(n_bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= edges[i]) & (p <= edges[i + 1])
        centres.append((edges[i] + edges[i + 1]) / 2)
        obs.append(y[m].mean() if m.any() else np.nan)
        cnt.append(int(m.sum()))
    return np.array(centres), np.array(obs), np.array(cnt)


# ==========================================================================
# selective prediction (F12)
# ==========================================================================
def risk_coverage_curve(
    y: np.ndarray, p: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """(coverage, selective_risk) as the abstention band widens.

    Confidence is distance from the decision threshold. A useful abstention
    policy produces a monotonically falling risk as coverage drops; a flat
    curve means the model's confidence carries no information, which is itself
    a publishable finding.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    conf = np.abs(p - threshold)
    order = np.argsort(-conf)
    correct = ((p >= threshold).astype(int) == y.astype(int))[order]
    n = len(y)
    cov = np.arange(1, n + 1) / n
    risk = 1 - np.cumsum(correct) / np.arange(1, n + 1)
    return cov, risk


def selective_risk_at_coverage(
    y: np.ndarray, p: np.ndarray, threshold: float, coverage: float
) -> float:
    cov, risk = risk_coverage_curve(y, p, threshold)
    return float(np.interp(coverage, cov, risk))


# ==========================================================================
# the report object
# ==========================================================================
@dataclass
class MetricReport:
    """Everything reported for one (model, test set) pair."""

    name: str = ""
    n: int = 0
    n_pos: int = 0
    n_neg: int = 0
    threshold: float = 0.5
    threshold_source: str = "UNSET"

    auc: float = float("nan")
    auc_lo: float = float("nan")
    auc_hi: float = float("nan")
    pauc10: float = float("nan")
    ap: float = float("nan")
    eer: float = float("nan")
    tpr_at_1fpr: float = float("nan")
    tpr_at_5fpr: float = float("nan")
    tpr_at_10fpr: float = float("nan")

    accuracy: float = float("nan")
    precision: float = float("nan")
    recall: float = float("nan")
    f1: float = float("nan")
    confusion: list[list[int]] = field(default_factory=list)

    ece: float = float("nan")
    mce: float = float("nan")
    brier: float = float("nan")

    abstain_rate: float = float("nan")
    risk_at_90cov: float = float("nan")
    risk_at_80cov: float = float("nan")
    risk_at_70cov: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"{self.name:34s} n={self.n:5d}  "
            f"AUC {self.auc:.4f} [{self.auc_lo:.4f},{self.auc_hi:.4f}]  "
            f"pAUC10 {self.pauc10:.4f}  EER {self.eer:.4f}  "
            f"Acc {self.accuracy:.4f}  F1 {self.f1:.4f}  ECE {self.ece:.4f}"
        )


def evaluate_predictions(
    y: Sequence[float] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    threshold: float,
    threshold_source: str,
    name: str = "",
    calibrated: Sequence[float] | np.ndarray | None = None,
    abstain: Sequence[bool] | np.ndarray | None = None,
    n_boot: int = 0,
) -> MetricReport:
    """Full metric suite for one prediction set.

    ``threshold_source`` is mandatory and free text: it goes into the results
    table so a reader can see where the operating point came from. Passing
    something that does not mention a validation split is a smell the
    aggregator flags.
    """
    y = np.asarray(y, dtype=float)
    s = np.asarray(scores, dtype=float)
    p = np.asarray(calibrated, dtype=float) if calibrated is not None else s

    r = MetricReport(
        name=name,
        n=len(y),
        n_pos=int((y == 1).sum()),
        n_neg=int((y == 0).sum()),
        threshold=float(threshold),
        threshold_source=threshold_source,
    )
    if r.n == 0:
        return r

    a, lo, hi = delong_ci(y, s)
    if n_boot:
        a, lo, hi = bootstrap_ci(y, s, auc, n_boot=n_boot)
    r.auc, r.auc_lo, r.auc_hi = a, lo, hi
    r.pauc10 = partial_auc(y, s, 0.1)
    r.ap = float(average_precision_score(y, s)) if r.n_pos and r.n_neg else float("nan")
    r.eer = eer(y, s)[0]
    r.tpr_at_1fpr = tpr_at_fpr(y, s, 0.01)
    r.tpr_at_5fpr = tpr_at_fpr(y, s, 0.05)
    r.tpr_at_10fpr = tpr_at_fpr(y, s, 0.10)

    pred = (s >= threshold).astype(int)
    r.accuracy = float(accuracy_score(y, pred))
    r.precision = float(precision_score(y, pred, zero_division=0))
    r.recall = float(recall_score(y, pred, zero_division=0))
    r.f1 = float(f1_score(y, pred, zero_division=0))
    r.confusion = confusion_matrix(y, pred, labels=[0, 1]).tolist()

    r.ece = expected_calibration_error(y, p)
    r.mce = max_calibration_error(y, p)
    r.brier = float(brier_score_loss(y, np.clip(p, 0, 1)))

    if abstain is not None:
        r.abstain_rate = float(np.mean(np.asarray(abstain, dtype=bool)))
    for cov, attr in ((0.9, "risk_at_90cov"), (0.8, "risk_at_80cov"), (0.7, "risk_at_70cov")):
        setattr(r, attr, selective_risk_at_coverage(y, p, threshold, cov))
    return r


# ==========================================================================
# preds.csv entry point
# ==========================================================================
def load_preds(path: str | Path) -> pd.DataFrame:
    """Read a predictions CSV and validate it against contract 5.4."""
    from ddetect.contracts import PREDS_COLUMNS

    df = pd.read_csv(path)
    missing = set(PREDS_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: preds CSV missing columns {sorted(missing)}")
    return df


def evaluate_preds_file(
    test_csv: str | Path,
    val_csv: str | Path | None = None,
    threshold: float | None = None,
    name: str = "",
    group_by: str | None = None,
    n_boot: int = 0,
) -> dict[str, MetricReport]:
    """Evaluate a preds file, optionally broken down by a column.

    Exactly one of ``val_csv`` / ``threshold`` must be given. Requiring it is
    the enforcement point for rule 1: there is no code path that evaluates a
    test set with a threshold derived from that same test set.
    """
    test = load_preds(test_csv)

    if (val_csv is None) == (threshold is None):
        raise ValueError(
            "pass exactly one of val_csv (to derive the threshold on the source "
            "validation split) or threshold (a value already derived that way). "
            "A threshold must never come from the test set."
        )
    if val_csv is not None:
        val = load_preds(val_csv)
        thr = threshold_from_val(val.label.to_numpy(), val.video_score.to_numpy())
        src = f"EER on source val ({Path(val_csv).name}, n={len(val)})"
    else:
        assert threshold is not None  # guaranteed by the exactly-one check above
        thr = float(threshold)
        src = "supplied by caller (verify it came from source val)"

    out: dict[str, MetricReport] = {}
    groups: list[tuple[str, pd.DataFrame]] = [(name or "all", test)]
    if group_by:
        if group_by not in test.columns:
            raise ValueError(f"group_by={group_by!r} not in {sorted(test.columns)}")
        groups += [(f"{name}/{k}", g) for k, g in test.groupby(group_by)]

    for gname, g in groups:
        out[gname] = evaluate_predictions(
            g.label.to_numpy(),
            g.video_score.to_numpy(),
            threshold=thr,
            threshold_source=src,
            name=gname,
            calibrated=g.calibrated_prob.to_numpy() if "calibrated_prob" in g else None,
            abstain=g.abstain.to_numpy() if "abstain" in g else None,
            n_boot=n_boot,
        )
    return out


def reports_to_frame(reports: dict[str, MetricReport]) -> pd.DataFrame:
    df: pd.DataFrame = pd.DataFrame([r.to_dict() for r in reports.values()])
    return df


def save_reports(reports: dict[str, MetricReport], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: v.to_dict() for k, v in reports.items()}, indent=2))
    return path
