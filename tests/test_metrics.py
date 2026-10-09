"""F16 - metrics, validated against sklearn and closed-form cases.

Written and verified BEFORE any model trains: these tests use
hand-made score arrays, so the whole evaluation and tables stack can be
finished while the data pipeline is still being built.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from ddetect.metrics import (
    auc,
    bootstrap_ci,
    delong_ci,
    delong_paired_test,
    eer,
    evaluate_predictions,
    expected_calibration_error,
    mcnemar_test,
    partial_auc,
    reliability_curve,
    risk_coverage_curve,
    selective_risk_at_coverage,
    threshold_from_val,
    tpr_at_fpr,
)


@pytest.fixture
def separable():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200), np.ones(200)]
    s = np.r_[rng.uniform(0.0, 0.4, 200), rng.uniform(0.6, 1.0, 200)]
    return y, s


@pytest.fixture
def noisy():
    rng = np.random.default_rng(1)
    y = (rng.uniform(0, 1, 1000) < 0.5).astype(float)
    s = np.clip(np.where(y == 1, rng.normal(0.65, 0.2, 1000), rng.normal(0.35, 0.2, 1000)), 0, 1)
    return y, s


# ==========================================================================
def test_auc_matches_sklearn(noisy):
    y, s = noisy
    assert auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_ap_matches_sklearn(noisy):
    y, s = noisy
    r = evaluate_predictions(y, s, 0.5, "test")
    assert r.ap == pytest.approx(average_precision_score(y, s), abs=1e-12)


def test_auc_closed_form_perfect_and_inverted(separable):
    y, s = separable
    assert auc(y, s) == pytest.approx(1.0)
    assert auc(y, -s) == pytest.approx(0.0)


def test_auc_single_class_is_nan_not_an_exception():
    # A per-method breakdown legitimately yields a fakes-only subset; raising
    # there would kill an entire results table.
    assert np.isnan(auc(np.ones(10), np.random.default_rng(0).random(10)))


def test_eer_zero_when_separable(separable):
    y, s = separable
    assert eer(y, s)[0] == pytest.approx(0.0, abs=1e-9)


def test_eer_half_when_uninformative():
    y = np.r_[np.zeros(500), np.ones(500)]
    s = np.full(1000, 0.5)
    assert eer(y, s)[0] == pytest.approx(0.5, abs=0.05)


def test_partial_auc_is_standardised(noisy):
    y, s = noisy
    p = partial_auc(y, s, 0.1)
    assert 0.5 <= p <= 1.0, "standardised pAUC must lie in [0.5, 1]"
    assert p == pytest.approx(roc_auc_score(y, s, max_fpr=0.1), abs=1e-12)


def test_tpr_at_fpr_is_monotone(noisy):
    y, s = noisy
    a, b, c = (tpr_at_fpr(y, s, f) for f in (0.01, 0.05, 0.10))
    assert a <= b <= c


# ==========================================================================
# confidence intervals
# ==========================================================================
def test_delong_ci_brackets_the_point_estimate(noisy):
    y, s = noisy
    a, lo, hi = delong_ci(y, s)
    assert lo < a < hi
    assert lo >= 0.0 and hi <= 1.0


def test_delong_ci_stays_finite_at_the_ceiling(separable):
    # Baseline A is expected to sit near AUC 1.0, where the DeLong variance
    # degenerates. It must still report an interval.
    y, s = separable
    a, lo, hi = delong_ci(y, s)
    assert a == pytest.approx(1.0)
    assert np.isfinite([lo, hi]).all(), "no interval reported at AUC = 1.0"
    assert hi <= 1.0


def test_delong_and_bootstrap_agree(noisy):
    y, s = noisy
    _, dlo, dhi = delong_ci(y, s)
    _, blo, bhi = bootstrap_ci(y, s, auc, n_boot=1000)
    assert not (bhi < dlo or blo > dhi), "the two interval methods do not overlap"


def test_narrower_ci_with_more_data():
    rng = np.random.default_rng(3)

    def width(n: int) -> float:
        y = (rng.uniform(0, 1, n) < 0.5).astype(float)
        s = np.where(y == 1, rng.normal(0.7, 0.3, n), rng.normal(0.3, 0.3, n))
        _, lo, hi = delong_ci(y, s)
        return hi - lo

    assert width(2000) < width(200)


# ==========================================================================
# significance
# ==========================================================================
def test_paired_test_null_when_scores_identical(noisy):
    y, s = noisy
    d, p = delong_paired_test(y, s, s)
    assert d == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0, abs=1e-6)


def test_paired_test_detects_a_real_difference(separable):
    y, s = separable
    rng = np.random.default_rng(4)
    d, p = delong_paired_test(y, s, rng.uniform(0, 1, len(y)))
    assert d > 0.3 and p < 0.01


def test_mcnemar_counts_discordant_pairs():
    y = np.array([1, 1, 0, 0])
    p1 = np.array([1, 1, 0, 1])  # 3 right
    p2 = np.array([0, 1, 0, 0])  # 3 right, different ones
    b, c, p = mcnemar_test(y, p1, p2)
    assert (b, c) == (1, 1)
    assert p == pytest.approx(1.0)


def test_mcnemar_identical_predictions_is_p1():
    y = np.array([1, 0, 1, 0])
    assert mcnemar_test(y, y, y) == (0, 0, 1.0)


# ==========================================================================
# the threshold-provenance rule
# ==========================================================================
def test_threshold_from_val_criteria_all_valid(noisy):
    y, s = noisy
    for crit in ("eer", "f1", "youden"):
        t = threshold_from_val(y, s, crit)
        assert 0.0 <= t <= 1.0, f"{crit} produced an out-of-range threshold {t}"


def test_evaluate_preds_file_demands_a_threshold_source(tmp_path):
    import pandas as pd

    from ddetect.contracts import PREDS_COLUMNS
    from ddetect.metrics import evaluate_preds_file

    df = pd.DataFrame({c: [0] * 10 for c in PREDS_COLUMNS})
    df["label"] = [0] * 5 + [1] * 5
    df["video_score"] = np.linspace(0, 1, 10)
    df["calibrated_prob"] = df["video_score"]
    p = tmp_path / "preds_test.csv"
    df.to_csv(p, index=False)

    # Neither argument -> refused. Both -> refused. This is the enforcement
    # point for "a threshold must never come from the test set".
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_preds_file(p)
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_preds_file(p, val_csv=p, threshold=0.5)

    out = evaluate_preds_file(p, threshold=0.5, name="t")
    assert "t" in out and out["t"].threshold_source


# ==========================================================================
# calibration + selective prediction
# ==========================================================================
def test_ece_low_for_calibrated_high_for_overconfident():
    rng = np.random.default_rng(5)
    p_true = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p_true).astype(float)
    good = expected_calibration_error(y, p_true)
    bad = expected_calibration_error(y, np.clip(p_true * 2 - 0.5, 0, 1))
    assert good < 0.05
    assert bad > good * 2


def test_reliability_curve_shapes():
    rng = np.random.default_rng(6)
    y = (rng.uniform(0, 1, 500) < 0.5).astype(float)
    c, o, n = reliability_curve(y, rng.uniform(0, 1, 500), n_bins=10)
    assert len(c) == len(o) == len(n) == 10
    assert n.sum() == 500


def test_risk_falls_as_coverage_falls(noisy):
    y, s = noisy
    cov, risk = risk_coverage_curve(y, s, 0.5)
    assert cov[0] < cov[-1] and cov[-1] == pytest.approx(1.0)
    # Abstaining on the least confident cases must not make things worse.
    assert selective_risk_at_coverage(y, s, 0.5, 0.7) <= risk[-1] + 1e-9


def test_report_summary_renders(noisy):
    y, s = noisy
    r = evaluate_predictions(y, s, 0.5, "EER on source val", name="x")
    assert "AUC" in r.summary() and "ECE" in r.summary()
    assert r.n == len(y) and r.n_pos + r.n_neg == r.n
    assert len(r.confusion) == 2
