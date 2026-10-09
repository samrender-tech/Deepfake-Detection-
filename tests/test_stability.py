"""Re-encode stability (ddetect/stability.py).

The verdict logic is tested against a fake detector, so these run without a
model; the ffmpeg path is tested on a real fixture clip.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ddetect.contracts import Result, Verdict
from ddetect.report import render_batch_index, render_html
from ddetect.stability import StabilityReport, VariantScore, reencode, stability_check

VIDEOS = Path(__file__).resolve().parent / "fixtures" / "videos"
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


class FakeDetector:
    """Returns scripted results by call order; records what it was asked."""

    def __init__(self, probs: list[float]) -> None:
        self.probs = list(probs)
        self.seen: list[Path] = []

    def predict(self, path, explain: bool = True) -> Result:
        self.seen.append(Path(path))
        p = self.probs.pop(0)
        v = Verdict.MANIPULATED if p >= 0.5 else Verdict.AUTHENTIC
        return Result(verdict=v, calibrated_prob=p)


def _report(orig: float, *probs: float) -> StabilityReport:
    rep = StabilityReport("likely_manipulated" if orig >= 0.5 else "likely_authentic", orig)
    for i, p in enumerate(probs):
        v = "likely_manipulated" if p >= 0.5 else "likely_authentic"
        rep.variants.append(VariantScore(f"v{i}", v, p))
    return rep


def test_agreeing_variants_are_stable():
    rep = _report(0.8, 0.78, 0.74)
    assert rep.verdict_agreement == 1.0
    assert rep.stable


def test_a_flipped_verdict_is_unstable():
    rep = _report(0.8, 0.78, 0.4)
    assert rep.verdict_agreement == pytest.approx(0.5)
    assert not rep.stable
    assert "UNSTABLE" in rep.summary()


def test_a_large_probability_swing_is_unstable_even_without_a_flip():
    rep = _report(0.95, 0.6)
    assert rep.verdict_agreement == 1.0
    assert not rep.stable


def test_a_check_that_could_not_run_is_not_a_pass():
    rep = StabilityReport("likely_authentic", 0.2, [VariantScore("crf32", "error", 0.0, "boom")])
    assert not rep.stable
    assert "could not run" in rep.summary()


@needs_ffmpeg
def test_reencode_writes_a_smaller_playable_file(tmp_path):
    src = VIDEOS / "real_sync_audio_v0.mp4"
    if not src.exists():
        pytest.skip("fixtures not generated")
    dst = tmp_path / "out.mp4"
    reencode(src, dst, ["-crf", "40"])
    assert dst.exists() and dst.stat().st_size < src.stat().st_size


@needs_ffmpeg
def test_stability_check_scores_every_variant(tmp_path):
    src = VIDEOS / "real_sync_audio_v0.mp4"
    if not src.exists():
        pytest.skip("fixtures not generated")
    det = FakeDetector([0.9, 0.85, 0.2, 0.88])
    rep = stability_check(det, src)
    assert [v.name for v in rep.variants] == ["crf32", "crf40", "half_res"]
    # Variants keep the source stem, so they get the same video_id.
    assert all(p.stem == src.stem for p in det.seen)
    assert not rep.stable and rep.verdict_agreement == pytest.approx(2 / 3)


def test_a_failed_variant_is_recorded(tmp_path):
    bogus = tmp_path / "not_a_video.mp4"
    bogus.write_bytes(b"nope")
    det = FakeDetector([0.3])
    rep = stability_check(det, bogus, variants={"crf32": ["-crf", "32"]})
    assert rep.variants[0].verdict == "error" and rep.variants[0].error
    assert not rep.stable


# ==========================================================================
# rendering
# ==========================================================================
def test_report_shows_the_stability_section():
    res = Result(verdict=Verdict.MANIPULATED, calibrated_prob=0.8).to_dict()
    page = render_html(res, stability=_report(0.8, 0.3).to_dict())
    assert "Re-encode stability" in page
    assert "data-stable='false'" in page
    assert "not evidence" in page


def test_batch_index_counts_and_links_and_escapes():
    rows = [
        {
            "file": "a/x.mp4",
            "verdict": "likely_authentic",
            "calibrated_prob": 0.1,
            "report": "x.html",
        },
        {"file": "a/y.mp4", "verdict": "inconclusive", "calibrated_prob": 0.5, "stable": False},
        {"file": "a/<z>.mp4", "verdict": "error", "error": "<boom>"},
    ]
    page = render_batch_index(rows)
    assert "3 clips analysed" in page
    assert "href='x.html'" in page
    assert "unstable" in page
    assert "&lt;boom&gt;" in page and "<boom>" not in page
    assert "not evidence" in page
