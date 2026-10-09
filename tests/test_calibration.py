"""F11 calibration, F12 conformal abstention, F13 OOD.

These three are what make the system safe to show a user at cross-dataset
accuracy. Each has a failure mode that looks fine in a log:

* a degenerate temperature flattens every probability to 0.5, so every verdict
  becomes a coin flip that LOOKS calibrated;
* a collapsed conformal band stops abstaining entirely, on exactly the models
  that most need to abstain;
* an asymmetric energy score reports confidently-authentic videos as novel.

All three were real bugs. All three are pinned below.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddetect.calibrate import Calibration, TemperatureScaler, conformal_band, fit_calibration
from ddetect.metrics import expected_calibration_error
from ddetect.ood import MahalanobisOOD, energy_score, novelty_auroc


@pytest.fixture
def val_logits():
    """Overconfident logits from a model with real but imperfect skill."""
    rng = np.random.default_rng(0)
    n = 4000
    y = (rng.uniform(0, 1, n) < 0.5).astype(float)
    z = np.where(y == 1, rng.normal(1.2, 1.2, n), rng.normal(-1.2, 1.2, n))
    return y, z * 3.0


# ==========================================================================
# F11 calibration
# ==========================================================================
def test_temperature_scaling_improves_ece(val_logits):
    y, z = val_logits
    before = expected_calibration_error(y, 1 / (1 + np.exp(-z)))
    cal = fit_calibration(z, y, method="temperature")
    after = expected_calibration_error(y, cal.probability(z))
    assert after < before, f"calibration made ECE worse: {before:.4f} -> {after:.4f}"


def test_temperature_scaling_preserves_ranking(val_logits):
    from ddetect.metrics import auc

    y, z = val_logits
    cal = fit_calibration(z, y, method="temperature")
    # One monotone parameter: AUC must be untouched, so the discrimination
    # result and the calibration result stay independently reportable.
    assert auc(y, z) == pytest.approx(auc(y, cal.probability(z)), abs=1e-9)


def test_degenerate_temperature_is_clamped_and_flagged():
    # Uninformative logits: the optimiser wants T -> infinity.
    rng = np.random.default_rng(1)
    y = (rng.uniform(0, 1, 200) < 0.5).astype(float)
    z = rng.normal(0, 0.001, 200)
    ts = TemperatureScaler().fit(z, y)
    assert 0.05 <= ts.temperature <= 20.0, "an unbounded temperature was accepted"


def test_small_val_split_is_marked_degenerate():
    rng = np.random.default_rng(2)
    y = (rng.uniform(0, 1, 8) < 0.5).astype(float)
    cal = fit_calibration(rng.normal(0, 1, 8), y)
    assert cal.degenerate, "calibrating on 8 videos was not flagged as unreliable"


def test_vector_scaling_also_works(val_logits):
    y, z = val_logits
    cal = fit_calibration(z, y, method="vector")
    p = cal.probability(z)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_calibration_round_trips_through_json(tmp_path, val_logits):
    y, z = val_logits
    cal = fit_calibration(z, y)
    p = tmp_path / "calibration.json"
    cal.save(p)
    back = Calibration.load(p)
    np.testing.assert_allclose(cal.probability(z), back.probability(z))


def test_missing_calibration_falls_back_to_identity(tmp_path):
    cal = Calibration.load(tmp_path / "nope.json")
    assert cal.method == "none"
    assert "MISSING" in cal.fitted_on
    # An identity transform, not a crash: the API must still serve a verdict,
    # with the warning carried in `fitted_on`.
    assert cal.probability(np.array([0.0]))[0] == pytest.approx(0.5)


# ==========================================================================
# F12 conformal abstention
# ==========================================================================
def test_conformal_band_is_never_degenerate(val_logits):
    y, z = val_logits
    cal = fit_calibration(z, y, conformal_alpha=0.1)
    assert cal.conformal_hi > cal.conformal_lo, "the abstention band collapsed to zero width"


def test_abstention_improves_accuracy(val_logits):
    y, z = val_logits
    cal = fit_calibration(z, y, conformal_alpha=0.1)
    p = cal.probability(z)
    ab = cal.abstains(p)

    assert ab.any(), "nothing was abstained on"
    acc_all = ((p >= cal.threshold).astype(float) == y).mean()
    acc_kept = ((p[~ab] >= cal.threshold).astype(float) == y[~ab]).mean()
    assert acc_kept > acc_all, (
        f"abstaining did not help: {acc_all:.4f} -> {acc_kept:.4f}. The "
        "confidence signal carries no information."
    )


def test_smaller_alpha_abstains_more(val_logits):
    y, z = val_logits
    widths = []
    for alpha in (0.01, 0.05, 0.1, 0.2):
        cal = fit_calibration(z, y, conformal_alpha=alpha)
        widths.append(cal.conformal_hi - cal.conformal_lo)
    assert widths == sorted(widths, reverse=True), (
        f"band width must shrink as alpha grows, got {widths}"
    )


def test_conformal_band_handles_empty_input():
    lo, hi = conformal_band(np.array([]), np.array([]), 0.1, 0.5)
    assert lo == hi == 0.5


# ==========================================================================
# F13 OOD
# ==========================================================================
def test_energy_score_is_symmetric_in_the_logit():
    # The asymmetric (0, z) completion reported confidently-authentic videos
    # (z very negative) as maximally novel. The symmetric form must not.
    assert energy_score(np.array([5.0]))[0] == pytest.approx(
        energy_score(np.array([-5.0]))[0], abs=1e-9
    )


def test_energy_score_peaks_at_ambivalence():
    e = energy_score(np.array([-8.0, -2.0, 0.0, 2.0, 8.0]))
    assert e[2] == e.max(), "the ambivalent logit was not the most novel"
    assert e[0] == pytest.approx(e[4], abs=1e-9)


def test_energy_separates_ambivalent_from_confident():
    rng = np.random.default_rng(3)
    confident = energy_score(np.concatenate([rng.normal(4, 1, 500), rng.normal(-4, 1, 500)]))
    ambivalent = energy_score(rng.normal(0, 0.3, 500))
    assert novelty_auroc(confident, ambivalent) > 0.8


def test_mahalanobis_detects_a_feature_shift():
    rng = np.random.default_rng(4)
    det = MahalanobisOOD().fit(rng.normal(0, 1, (2000, 32)))
    in_d = det.score(rng.normal(0, 1, (500, 32)))
    out_d = det.score(rng.normal(2.5, 1, (500, 32)))
    assert novelty_auroc(in_d, out_d) > 0.9
    assert det.dim == 32 and det.n_fit == 2000


def test_mahalanobis_round_trips(tmp_path):
    rng = np.random.default_rng(5)
    det = MahalanobisOOD().fit(rng.normal(0, 1, (500, 16)))
    p = tmp_path / "ood.json"
    det.save(p)
    back = MahalanobisOOD.load(p)
    x = rng.normal(0, 1, (10, 16))
    np.testing.assert_allclose(det.score(x), back.score(x), rtol=1e-6)


def test_mahalanobis_load_missing_returns_none(tmp_path):
    assert MahalanobisOOD.load(tmp_path / "nope.json") is None


def test_unfitted_mahalanobis_scores_zero():
    assert MahalanobisOOD().score(np.zeros((3, 8))).tolist() == [0.0, 0.0, 0.0]
