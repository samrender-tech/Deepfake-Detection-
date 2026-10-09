"""F18 figures and the offline degradation script.

Both exist so the paper's figures and robustness rows are generated rather
than made by hand. The failure mode is quiet in both cases: a figure that
silently renders empty, or a "degraded" copy that was never actually
re-encoded.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ddetect.contracts import PREDS_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
VIDEOS = ROOT / "tests" / "fixtures" / "videos"


def _write_preds(path: Path, n: int = 60, sep: float = 0.35, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)
    s = np.clip(
        np.where(y == 1, rng.normal(0.5 + sep, 0.18, n), rng.normal(0.5 - sep, 0.18, n)), 0, 1
    )
    df = pd.DataFrame({c: [""] * n for c in PREDS_COLUMNS})
    df["video_id"] = [f"v{i}" for i in range(n)]
    df["label"] = y
    df["dataset"] = "ffpp"
    df["forgery_method"] = np.where(y == 1, "Deepfakes", "real")
    df["compression"] = "c23"
    df["video_score"] = s
    df["calibrated_prob"] = s
    df["has_audio"] = False
    df["abstain"] = False
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# ==========================================================================
# figures
# ==========================================================================
@pytest.fixture
def two_runs(tmp_path):
    """One strong in-dataset run and one weak cross-dataset run."""
    _write_preds(tmp_path / "baseline_a" / "seed0" / "preds_val.csv", sep=0.40, seed=1)
    _write_preds(tmp_path / "baseline_a" / "seed0" / "preds_celebdf.csv", sep=0.08, seed=2)
    return tmp_path


def test_discovers_prediction_files(two_runs):
    from experiments.figures import discover_preds

    found = discover_preds(two_runs)
    assert set(found) == {"baseline_a/val", "baseline_a/celebdf"}


@pytest.mark.parametrize("fn_name", ["roc_figure", "reliability_figure", "risk_coverage_figure"])
def test_each_figure_renders_a_non_empty_pdf(two_runs, tmp_path, fn_name):
    import experiments.figures as F

    preds = F.discover_preds(two_runs)
    out = tmp_path / f"{fn_name}.pdf"
    getattr(F, fn_name)(preds, out)
    assert out.exists()
    # A blank matplotlib page is still a valid PDF, so check it carries ink.
    assert out.stat().st_size > 5000, "figure looks empty"
    assert out.read_bytes()[:4] == b"%PDF"


def test_generate_all_writes_a_manifest(two_runs, tmp_path):
    from experiments.figures import generate_all

    made = generate_all(two_runs, tmp_path / "noresults", tmp_path / "figs")
    assert {"roc", "reliability", "risk_coverage"} <= set(made)
    assert (tmp_path / "figs" / "manifest.json").exists()


def test_gap_figure_renders_from_a_gap_table(tmp_path):
    from experiments.figures import gap_figure

    (tmp_path / "cross_dataset_gap.csv").write_text(
        "exp,test_set,in_dataset_auc,cross_dataset_auc,gap,n_seeds\n"
        "baseline_b,celebdf,0.9612,0.6703,0.2909,3\n"
        "baseline_b,dfdc,0.9612,0.7101,0.2511,3\n"
    )
    out = gap_figure(tmp_path / "cross_dataset_gap.csv", tmp_path / "gap.pdf")
    assert out and out.exists() and out.stat().st_size > 5000


def test_missing_inputs_return_none_rather_than_raising(tmp_path):
    from experiments.figures import gap_figure, per_method_figure

    assert gap_figure(tmp_path / "nope.csv", tmp_path / "a.pdf") is None
    assert per_method_figure(tmp_path / "nope.csv", tmp_path / "b.pdf") is None


# ==========================================================================
# offline degradation
# ==========================================================================
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_crf40_actually_re_encodes(tmp_path, a_video):
    """Online JPEG augmentation is not H.264; this must really re-encode."""
    from agents.a8_redteam import RECIPES, apply_recipe

    recipe = next(r for r in RECIPES if r.name == "crf40")
    dst = tmp_path / "out.mp4"
    assert apply_recipe(a_video, dst, recipe)
    assert dst.exists() and dst.stat().st_size > 0
    assert dst.stat().st_size < a_video.stat().st_size, "the file was not compressed"

    from ddetect.data.manifest_io import probe_video

    assert probe_video(dst)["vcodec"] == "h264"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_audio_strip_removes_the_track(tmp_path, a_video):
    from agents.a8_redteam import RECIPES, apply_recipe
    from ddetect.data.manifest_io import probe_video

    assert probe_video(a_video)["has_audio"] is True
    recipe = next(r for r in RECIPES if r.name == "audio_strip")
    dst = tmp_path / "silent.mp4"
    assert apply_recipe(a_video, dst, recipe)
    assert probe_video(dst)["has_audio"] is False


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_audio_swap_needs_a_partner_clip(tmp_path, a_video, a_silent_video):
    from agents.a8_redteam import RECIPES, apply_recipe

    recipe = next(r for r in RECIPES if r.name == "audio_swap")
    # Without a partner it must refuse rather than silently produce a copy.
    assert not apply_recipe(a_video, tmp_path / "x.mp4", recipe, second=None)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_degrade_script_writes_a_test_only_manifest(tmp_path, fixture_manifest):
    """A degraded copy must never be able to join its own original in training."""
    import scripts.degrade_dataset as deg
    from ddetect.data.manifest_io import read_manifest

    out_man = tmp_path / "degraded.parquet"
    rc = deg.main(
        [
            "--manifest",
            str(fixture_manifest),
            "--recipe",
            "crf40",
            "--out-root",
            str(tmp_path / "vids"),
            "--out-manifest",
            str(out_man),
            "--limit",
            "3",
            "--workers",
            "2",
        ]
    )
    assert rc == 0
    df = read_manifest(out_man)
    assert len(df) == 3
    assert set(df.split) == {"test"}, "a degraded copy was not marked test-only"
    assert set(df.compression) == {"crf40"}
    assert all(v.endswith("__crf40") for v in df.video_id)
    for p in df.path:
        assert Path(p).exists()
