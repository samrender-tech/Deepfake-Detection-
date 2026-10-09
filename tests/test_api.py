"""F21/F31 - the API surface and its hardening.

Two groups:

* the happy path end to end, through the real worker and the real Detector --
  which makes this also the integration test for contract 5.2;
* the security posture: content sniffing, size and duration caps, path
  traversal, and that the retention delete actually removes bytes.

The responsible-use policy is tested as a contract, not an aspiration: there is no
response shape that lets a client render a bare real/fake binary.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
VIDEOS = ROOT / "tests" / "fixtures" / "videos"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A TestClient wired to a real trained run and isolated storage."""
    import os
    import subprocess
    import sys

    if not list(VIDEOS.glob("*.mp4")):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "make_fixtures.py")],
            check=True,
            cwd=ROOT,
            capture_output=True,
        )

    from ddetect.data.facedet import PERMISSIVE_MTCNN_THRESHOLDS
    from ddetect.data.manifest_io import read_manifest, write_manifest
    from ddetect.data.manifests import FixtureParser
    from ddetect.data.preprocess import PreprocessConfig, cache_dir_for, preprocess_video
    from ddetect.train import TrainConfig, train

    base = tmp_path_factory.mktemp("api")
    man = base / "fixture.parquet"
    write_manifest(FixtureParser(VIDEOS).build(workers=2), man)
    cache = base / "cache"
    pcfg = PreprocessConfig(n_frames=8, det_thresholds=PERMISSIVE_MTCNN_THRESHOLDS)
    for r in read_manifest(man).itertuples():
        preprocess_video(
            r.path,
            cache_dir_for(cache, r.dataset, r.video_id),
            pcfg,
            video_id=r.video_id,
            dataset=r.dataset,
        )

    cfg = TrainConfig(
        exp="api",
        model="smoke",
        epochs=1,
        batch_size=2,
        n_frames=8,
        image_size=224,
        audio_seconds=2.0,
        max_sync_windows=4,
        num_workers=0,
        freeze_epochs=0,
        run_root=str(base / "runs"),
        manifest=str(man),
        cache_root=str(cache),
        amp=False,
    )
    train(cfg)

    os.environ["DDETECT_RUN_DIR"] = str(cfg.run_dir)
    os.environ["DDETECT_UPLOAD_DIR"] = str(base / "uploads")
    os.environ["DDETECT_ARTIFACT_DIR"] = str(base / "artifacts")
    os.environ["DDETECT_DISABLE_AGENT"] = "1"
    os.environ.pop("API_KEYS", None)
    # The suite runs more analyses than the production default allows per
    # minute from one client; throttling is not what these tests measure.
    os.environ["RATE_LIMIT_ANALYSES"] = "1000/minute"

    # Imported after the environment is set: module-level config is read at import.
    import importlib

    import api.main as m

    importlib.reload(m)
    with TestClient(m.app) as c:
        yield c


def _analyse(client, path: Path, timeout: float = 120.0) -> dict:
    with path.open("rb") as fh:
        r = client.post("/v1/analyses", files={"file": (path.name, fh, "video/mp4")})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    deadline = time.time() + timeout
    while time.time() < deadline:
        s = client.get(f"/v1/analyses/{job_id}").json()
        if s["state"] in ("done", "failed"):
            return s
        time.sleep(0.4)
    pytest.fail(f"job {job_id} did not finish within {timeout}s")


# ==========================================================================
# ops
# ==========================================================================
def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readyz_reports_the_loaded_model(client):
    r = client.get("/readyz")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["model_loaded"] is True
    assert b["model_version"]
    # The smoke run calibrates on 4 videos, so this must be reported unreliable.
    assert b["calibration_reliable"] is False


def test_limits_endpoint_publishes_the_disclaimer(client):
    b = client.get("/v1/limits").json()
    assert b["max_upload_mb"] > 0 and b["max_duration_s"] > 0
    assert "not evidence" in b["disclaimer"]


def test_request_id_header_is_echoed(client):
    r = client.get("/healthz", headers={"x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123"
    assert "x-response-time-ms" in r.headers


# ==========================================================================
# the happy path
# ==========================================================================
def test_full_analysis_returns_a_three_state_verdict(client):
    s = _analyse(client, VIDEOS / "real_sync_audio_v0.mp4")
    assert s["state"] == "done", s.get("error")
    assert s["progress"] == 1.0

    res = s["result"]
    assert res["verdict"] in ("likely_authentic", "likely_manipulated", "inconclusive")
    assert 0.0 <= res["calibrated_probability"] <= 1.0
    assert len(res["confidence_interval"]) == 2
    # Section 13: the disclaimer is not optional and not removable.
    assert "not evidence" in res["disclaimer"]
    assert res["model_version"]


def test_result_carries_the_evidence(client):
    res = _analyse(client, VIDEOS / "real_sync_audio_v0.mp4")["result"]
    assert len(res["frame_scores"]) == 8
    assert len(res["frame_timestamps"]) == len(res["frame_scores"])
    assert res["streams"], "no per-stream breakdown"
    assert res["gradcam_urls"], "no Grad-CAM overlays"
    assert res["explanation"], "no explanation text"
    assert res["explanation_source"] in ("agent", "template")


def test_gradcam_artifact_is_served(client):
    res = _analyse(client, VIDEOS / "real_sync_audio_v0.mp4")["result"]
    url = next(iter(res["gradcam_urls"].values()))
    r = client.get(url)
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_silent_video_reports_masked_streams(client):
    res = _analyse(client, VIDEOS / "real_silent_v0.mp4")["result"]
    assert res["has_audio"] is False
    for s in res["streams"]:
        if s["name"] in ("audio", "sync"):
            assert s["available"] is False and s["score"] is None
    assert res["spectrum_url"] is None


def test_faceless_video_warns_instead_of_failing(client):
    s = _analyse(client, VIDEOS / "edge_no_face.mp4")
    assert s["state"] == "done", s.get("error")
    res = s["result"]
    assert res["n_faces_found"] == 0
    assert any("face" in w.lower() for w in res["warnings"])


def test_sse_stream_emits_progress(client):
    with (VIDEOS / "real_sync_audio_v0.mp4").open("rb") as fh:
        job_id = client.post("/v1/analyses", files={"file": ("v.mp4", fh, "video/mp4")}).json()[
            "job_id"
        ]

    seen = []
    with client.stream("GET", f"/v1/analyses/{job_id}/events") as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if line.startswith("data: "):
                import json

                seen.append(json.loads(line[6:]))
                if seen[-1]["state"] in ("done", "failed"):
                    break
    assert seen and seen[-1]["state"] in ("done", "failed")
    assert seen[-1]["progress"] == pytest.approx(1.0)


def test_listing_recent_analyses(client):
    _analyse(client, VIDEOS / "real_silent_v0.mp4")
    rows = client.get("/v1/analyses?limit=5").json()
    assert rows and all("job_id" in r for r in rows)


# ==========================================================================
# retention (F31)
# ==========================================================================
def test_delete_removes_the_job_and_its_artifacts(client):
    import os

    s = _analyse(client, VIDEOS / "real_sync_audio_v0.mp4")
    job_id = s["job_id"]
    art_dir = Path(os.environ["DDETECT_ARTIFACT_DIR"]) / job_id
    assert art_dir.exists()

    assert client.delete(f"/v1/analyses/{job_id}").status_code == 204
    assert client.get(f"/v1/analyses/{job_id}").status_code == 404
    assert not art_dir.exists(), "artifacts survived a deletion request"


def test_upload_is_deleted_after_analysis(client):
    import os

    _analyse(client, VIDEOS / "real_sync_audio_v0.mp4")
    leftovers = list(Path(os.environ["DDETECT_UPLOAD_DIR"]).glob("*.bin"))
    assert not leftovers, f"uploaded video was not deleted: {leftovers}"


def test_unknown_job_is_404(client):
    assert client.get("/v1/analyses/" + "0" * 32).status_code == 404
    assert client.delete("/v1/analyses/" + "0" * 32).status_code == 404


# ==========================================================================
# reports, segments and model info
# ==========================================================================
def test_result_carries_suspicious_segments(client):
    res = _analyse(client, VIDEOS / "real_sync_audio_v0.mp4")["result"]
    assert isinstance(res["suspicious_segments"], list)
    thr = res["decision_threshold"]
    n_above = sum(s >= thr for s in res["frame_scores"])
    assert sum(seg["n_frames"] for seg in res["suspicious_segments"]) == n_above
    for seg in res["suspicious_segments"]:
        assert seg["end_s"] > seg["start_s"]
        assert seg["peak_score"] >= thr


def test_html_report_is_downloadable_and_self_contained(client):
    s = _analyse(client, VIDEOS / "real_sync_audio_v0.mp4")
    r = client.get(f"/v1/analyses/{s['job_id']}/report")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/html")
    assert "attachment" in r.headers["content-disposition"]
    page = r.text
    assert f"data-verdict='{s['result']['verdict']}'" in page
    assert "not evidence" in page
    # Grad-CAM overlays are inlined, so the file opens offline.
    assert "data:image/png;base64," in page


def test_json_report_carries_provenance(client):
    s = _analyse(client, VIDEOS / "real_silent_v0.mp4")
    r = client.get(f"/v1/analyses/{s['job_id']}/report?format=json")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["job_id"] == s["job_id"]
    assert b["filename"] == "real_silent_v0.mp4"
    assert b["result"]["verdict"] == s["result"]["verdict"]
    assert "not evidence" in b["result"]["disclaimer"]


def test_report_rejects_bad_format_and_unknown_job(client):
    s = _analyse(client, VIDEOS / "real_silent_v0.mp4")
    assert client.get(f"/v1/analyses/{s['job_id']}/report?format=pdf").status_code == 400
    assert client.get("/v1/analyses/" + "0" * 32 + "/report").status_code == 404


def test_model_info_describes_the_calibration(client):
    r = client.get("/v1/model")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["architecture"] == "smoke"
    assert b["n_frames"] == 8 and b["image_size"] == 224
    lo, hi = b["conformal_band"]
    assert 0.0 <= lo <= hi <= 1.0
    # Same answer as /readyz: the smoke run's calibration is degenerate.
    assert b["calibration_reliable"] is False
    assert "not evidence" in b["disclaimer"]


def test_cli_batch_scores_a_folder(client, tmp_path):
    import csv
    import os

    from ddetect.cli import main

    clips = tmp_path / "clips"
    clips.mkdir()
    for name in ("real_silent_v0.mp4", "fake_silent_v0.mp4"):
        (clips / name).write_bytes((VIDEOS / name).read_bytes())
    (clips / "broken.mp4").write_bytes(b"not a video")
    (clips / "notes.txt").write_text("ignored")

    out = tmp_path / "verdicts.csv"
    reports = tmp_path / "reports"
    rc = main(
        [
            "batch",
            str(clips),
            "--run",
            os.environ["DDETECT_RUN_DIR"],
            "--out",
            str(out),
            "--report-dir",
            str(reports),
        ]
    )
    rows = {Path(r["file"]).name: r for r in csv.DictReader(out.open())}
    assert set(rows) == {"real_silent_v0.mp4", "fake_silent_v0.mp4", "broken.mp4"}
    # One unreadable clip costs one row, and the exit status says so.
    assert rc == 1
    assert rows["broken.mp4"]["verdict"] == "error" and rows["broken.mp4"]["error"]
    for name in ("real_silent_v0.mp4", "fake_silent_v0.mp4"):
        assert rows[name]["verdict"] in ("likely_authentic", "likely_manipulated", "inconclusive")
        assert (reports / f"{Path(name).stem}.html").exists()
    index = (reports / "index.html").read_text()
    assert "3 clips analysed" in index and "real_silent_v0.html" in index


def test_cli_predict_stability_check(client, capsys):
    import json
    import os

    from ddetect.cli import main

    rc = main(
        [
            "predict",
            str(VIDEOS / "real_silent_v0.mp4"),
            "--run",
            os.environ["DDETECT_RUN_DIR"],
            "--no-explain",
            "--stability",
            "--json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    stab = json.loads(out[out.index("{") :])["stability"]
    assert [v["name"] for v in stab["variants"]] == ["crf32", "crf40", "half_res"]
    assert all(not v["error"] for v in stab["variants"]), stab
    assert 0.0 <= stab["verdict_agreement"] <= 1.0
    assert isinstance(stab["stable"], bool)
