"""Suspicious segments and the shareable report (ddetect/report.py).

The report is the artefact most likely to be forwarded without its context,
so the tests pin down the two things it must never lose: the three-state
verdict and the disclaimer. Segment extraction is tested on hand-made scores,
in the same spirit as the metrics stack -- no model needed.
"""

from __future__ import annotations

import pytest

from ddetect.contracts import Result, StreamScore, Verdict
from ddetect.report import (
    REPORT_DISCLAIMER,
    render_html,
    segments_for,
    summary_row,
    suspicious_segments,
)

TS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]


# ==========================================================================
# segments
# ==========================================================================
def test_no_frames_no_segments():
    assert suspicious_segments([], [], 0.5) == []


def test_nothing_above_threshold():
    assert suspicious_segments([0.1, 0.2, 0.3], TS[:3], 0.5) == []


def test_contiguous_runs_become_segments():
    segs = suspicious_segments([0.1, 0.8, 0.9, 0.2, 0.7, 0.1], TS, 0.5)
    assert len(segs) == 2
    a, b = segs
    assert (a["start_s"], a["end_s"], a["n_frames"]) == (0.5, 1.5, 2)
    assert (a["peak_score"], a["peak_time_s"]) == (0.9, 1.0)
    assert (b["start_s"], b["end_s"], b["n_frames"]) == (2.0, 2.5, 1)


def test_threshold_is_inclusive():
    assert len(suspicious_segments([0.5], [0.0], 0.5)) == 1


def test_last_frame_segment_has_width():
    segs = suspicious_segments([0.1, 0.1, 0.9], [0.0, 0.4, 0.8], 0.5)
    assert segs[0]["start_s"] == 0.8
    assert segs[0]["end_s"] == pytest.approx(1.2)


def test_merge_gap_joins_nearby_runs():
    scores = [0.9, 0.1, 0.9, 0.1, 0.1, 0.9]
    assert len(suspicious_segments(scores, TS, 0.5, merge_gap_s=0.0)) == 3
    merged = suspicious_segments(scores, TS, 0.5, merge_gap_s=0.5)
    assert [s["n_frames"] for s in merged] == [3, 1]


def test_missing_timestamps_fall_back_to_indices():
    segs = suspicious_segments([0.9, 0.9], [], 0.5)
    assert segs[0]["start_s"] == 0.0 and segs[0]["end_s"] == 2.0


def test_decreasing_timestamps_are_rejected():
    with pytest.raises(ValueError, match="non-decreasing"):
        suspicious_segments([0.9, 0.9], [1.0, 0.5], 0.5)


# ==========================================================================
# the report
# ==========================================================================
def _result(**kw) -> dict:
    r = Result(
        video_id="clip",
        model_version="smoke/seed0@abc",
        verdict=kw.pop("verdict", Verdict.MANIPULATED),
        calibrated_prob=0.83,
        conformal_lo=0.4,
        conformal_hi=0.6,
        threshold=0.5,
        frame_scores=[0.1, 0.8, 0.9, 0.2, 0.7, 0.1],
        frame_timestamps=TS,
        has_audio=True,
        n_faces_found=6,
        streams=[
            StreamScore(name="visual", score=0.8, available=True),
            StreamScore(name="audio", score=None, available=False, note="masked"),
        ],
        warnings=kw.pop("warnings", []),
        **kw,
    )
    return r.to_dict()


def test_disclaimer_matches_the_api():
    pytest.importorskip("fastapi")
    from api.schemas import DISCLAIMER

    assert REPORT_DISCLAIMER == DISCLAIMER


@pytest.mark.parametrize("verdict", list(Verdict))
def test_report_renders_every_verdict_with_the_disclaimer(verdict):
    page = render_html(_result(verdict=verdict))
    assert page.startswith("<!doctype html>")
    assert f"data-verdict='{verdict.value}'" in page
    assert "not evidence" in page
    assert "83.0%" in page


def test_report_never_renders_a_bare_binary():
    page = render_html(_result())
    assert ">FAKE<" not in page and ">REAL<" not in page


def test_report_escapes_user_controlled_strings():
    page = render_html(
        _result(warnings=["<script>alert(1)</script>"]),
        filename="<img src=x onerror=alert(1)>.mp4",
    )
    assert "<script>alert(1)" not in page
    assert "<img src=x" not in page
    assert "&lt;script&gt;" in page


def test_report_is_self_contained():
    png = b"\x89PNG\r\n\x1a\nfake"
    page = render_html(_result(), images={"frame_2": png, "spectrum": png})
    assert "data:image/png;base64," in page
    assert "attention overlay frame_2" in page
    # No external fetches: a forwarded report must open offline.
    assert "http://" not in page and "https://" not in page


def test_report_lists_flagged_segments_and_caveats():
    page = render_html(_result(ood_flag=True, calibration_degenerate=True))
    assert "Flagged segment" in page
    assert "0.50s – 1.50s" in page
    assert "unlike the videos" in page
    assert "calibration is not reliable" in page


def test_report_rejects_an_unknown_verdict():
    d = _result()
    d["verdict"] = "fake"
    with pytest.raises(ValueError, match="unknown verdict"):
        render_html(d)


def test_summary_row_is_flat():
    row = summary_row(_result())
    assert row["verdict"] == "likely_manipulated"
    assert row["n_suspicious_segments"] == len(segments_for(_result())) == 2
    assert row["peak_frame_score"] == 0.9
    assert all(not isinstance(v, (list, dict)) for v in row.values())
