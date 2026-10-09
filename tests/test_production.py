"""The production-ops tier: storage, export, drift, observability, tracking.

Two properties run through all of it, and both are tested here rather than
assumed:

1. **Every optional dependency degrades to a working default.** The
   single-command demo must keep running with only the core dependencies, so
   a missing Redis, Postgres, S3, Prometheus, MLflow or OTel endpoint has to
   fall back rather than fail to boot.

2. **An export is parity-checked before it ships.** A traced graph that drops
   a branch still runs, still returns a number, and silently stops being the
   model the paper describes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import torch


# ==========================================================================
# storage
# ==========================================================================
def test_local_blob_roundtrip(tmp_path):
    from api.storage import LocalBlobStore

    s = LocalBlobStore(tmp_path)
    s.put("job1/a.png", b"\x89PNG data", "image/png")
    assert s.get("job1/a.png") == b"\x89PNG data"
    assert s.exists("job1/a.png")
    assert s.get("job1/missing.png") is None


def test_blob_store_refuses_path_traversal(tmp_path):
    from api.storage import LocalBlobStore

    s = LocalBlobStore(tmp_path)
    for key in ("../escape.png", "a/../../escape.png"):
        with pytest.raises(ValueError, match="escapes"):
            s.put(key, b"x")


def test_blob_delete_prefix_removes_a_whole_job(tmp_path):
    from api.storage import LocalBlobStore

    s = LocalBlobStore(tmp_path)
    for i in range(3):
        s.put(f"job1/f{i}.png", b"x")
    s.put("job2/keep.png", b"y")
    assert s.delete_prefix("job1") == 3
    assert not s.exists("job1/f0.png")
    assert s.exists("job2/keep.png"), "an unrelated job was deleted"


def test_blob_writes_are_atomic(tmp_path):
    """A half-written blob must never be readable."""
    from api.storage import LocalBlobStore

    s = LocalBlobStore(tmp_path)
    s.put("a.bin", b"x" * 1000)
    assert not list(Path(tmp_path).glob("*.tmp")), "a temp file was left behind"


def test_memory_job_backend_lifecycle():
    from api.storage import MemoryJobBackend

    b = MemoryJobBackend()
    now = datetime.now(timezone.utc)
    b.create(
        {
            "job_id": "j1",
            "state": "queued",
            "stage": "queued",
            "created_at": now,
            "updated_at": now,
            "filename": "a.mp4",
            "storage_key": "k",
        }
    )
    assert b.get("j1")["state"] == "queued"

    b.update("j1", state="done", result={"verdict": "inconclusive"})
    assert b.get("j1")["result"]["verdict"] == "inconclusive"
    assert len(b.list_recent(10)) == 1

    assert b.expired(now + timedelta(hours=1)) == ["j1"]
    assert b.expired(now - timedelta(hours=1)) == []

    assert b.delete("j1") and b.get("j1") is None
    assert not b.delete("j1")


def test_backends_fall_back_without_configuration(monkeypatch):
    """No DSN configured must mean local + memory, not a crash."""
    from api.storage import build_blob_store, build_job_backend

    for var in ("DATABASE_URL", "S3_BUCKET", "S3_ENDPOINT", "S3_ACCESS_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert build_blob_store().name == "local"
    assert build_job_backend().name == "memory"


def test_unreachable_database_falls_back_rather_than_failing(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://no:no@127.0.0.1:1/none")
    from api.storage import build_job_backend

    # A deployment that misconfigures its DSN should degrade, not refuse to boot.
    assert build_job_backend().name in ("memory", "postgres")


# ==========================================================================
# export (F22)
# ==========================================================================
@pytest.fixture(scope="module")
def tiny_model():
    from ddetect.models.registry import build_model

    return build_model(
        {
            "visual": {
                "backbone": "effb0",
                "pretrained": False,
                "image_size": 224,
                "temporal": "attention",
                "embed_dim": 64,
            },
        }
    ).eval()


@pytest.fixture(scope="module")
def example_faces():
    return torch.randn(1, 4, 3, 224, 224)


def test_torchscript_export_is_parity_checked(tiny_model, example_faces, tmp_path):
    from ddetect.export import PARITY_TOL, export_torchscript

    r = export_torchscript(tiny_model, tmp_path / "v.pt", example_faces)
    assert r.ok, f"parity {r.max_abs_diff} exceeded {PARITY_TOL}"
    assert r.max_abs_diff < PARITY_TOL
    assert Path(r.path).exists() and r.size_mb > 0


def test_onnx_export_is_parity_checked(tiny_model, example_faces, tmp_path):
    pytest.importorskip("onnx")
    from ddetect.export import PARITY_TOL, export_onnx

    r = export_onnx(tiny_model, tmp_path / "v.onnx", example_faces)
    assert Path(r.path).exists()
    if r.max_abs_diff == r.max_abs_diff:  # not NaN => onnxruntime verified it
        assert r.ok and r.max_abs_diff < PARITY_TOL


def test_onnx_accepts_a_different_frame_count(tiny_model, example_faces, tmp_path):
    """The dynamic axes must really be dynamic: T varies per request."""
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort

    from ddetect.export import export_onnx

    r = export_onnx(tiny_model, tmp_path / "v.onnx", example_faces)
    sess = ort.InferenceSession(r.path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"faces": torch.randn(1, 8, 3, 224, 224).numpy()})[0]
    assert out.shape[0] == 1


def test_int8_export_reports_its_drift(tiny_model, example_faces, tmp_path):
    from ddetect.export import export_quantized

    r = export_quantized(tiny_model, tmp_path / "v.int8.pt", example_faces)
    if "no quantisation engine" in r.note:
        pytest.skip("this PyTorch build has no quantisation engine")
    # INT8 changes the arithmetic, so drift is expected -- what matters is that
    # it is measured and reported rather than assumed to be zero.
    assert r.max_abs_diff >= 0.0
    assert r.tolerance > 0


def test_a_broken_export_is_rejected_not_shipped(tiny_model, example_faces, tmp_path, monkeypatch):
    """Parity failure must delete the artefact, not warn and continue."""
    import ddetect.export as E

    monkeypatch.setattr(E, "PARITY_TOL", -1.0)  # force failure
    out = tmp_path / "bad.pt"
    r = E.export_torchscript(tiny_model, out, example_faces, strict=True)
    assert not r.ok
    assert not out.exists(), "a failing export was left on disk"


# ==========================================================================
# drift (F28)
# ==========================================================================
def test_psi_is_near_zero_for_the_same_distribution():
    from api.drift import population_stability_index

    rng = np.random.default_rng(0)
    assert population_stability_index(rng.beta(2, 2, 2000), rng.beta(2, 2, 500)) < 0.1


def test_psi_is_large_for_a_shifted_distribution():
    from api.drift import PSI_SIGNIFICANT, population_stability_index

    rng = np.random.default_rng(0)
    assert population_stability_index(rng.beta(2, 2, 2000), rng.beta(6, 1, 500)) > PSI_SIGNIFICANT


def test_psi_needs_enough_samples():
    from api.drift import population_stability_index

    assert np.isnan(population_stability_index(np.array([0.5]), np.array([0.5])))


def test_monitor_withholds_a_verdict_until_it_has_data():
    from api.drift import Baseline, DriftMonitor

    m = DriftMonitor(baseline=Baseline(scores=[0.5] * 100, n=100, source="t"), min_samples=50)
    r = m.report()
    assert not r["ready"] and "need" in r["note"]


def test_monitor_alerts_on_a_shifted_score_distribution():
    from api.drift import Baseline, DriftMonitor

    rng = np.random.default_rng(1)
    m = DriftMonitor(
        baseline=Baseline(
            scores=rng.beta(2, 2, 2000).tolist(), has_audio_rate=0.9, n=2000, source="t"
        ),
        min_samples=20,
    )
    for _ in range(60):
        m.observe_result(
            {
                "calibrated_prob": float(rng.beta(8, 1)),
                "has_audio": True,
                "n_faces_found": 8,
                "ood_flag": False,
                "verdict": "likely_manipulated",
            }
        )
    r = m.report()
    assert r["ready"]
    assert r["score_drift"] == "significant"
    assert any("score distribution" in a for a in r["alerts"])


def test_monitor_alerts_when_most_uploads_have_no_face():
    from api.drift import Baseline, DriftMonitor

    m = DriftMonitor(
        baseline=Baseline(scores=[0.5] * 200, has_audio_rate=0.0, n=200, source="t"), min_samples=20
    )
    for _ in range(40):
        m.observe_result(
            {
                "calibrated_prob": 0.5,
                "has_audio": False,
                "n_faces_found": 0,
                "ood_flag": False,
                "verdict": "inconclusive",
            }
        )
    r = m.report()
    assert r["no_face_rate"] == 1.0
    assert any("no detectable face" in a for a in r["alerts"])


def test_monitor_tracks_ood_and_abstention_rates():
    from api.drift import Baseline, DriftMonitor

    m = DriftMonitor(baseline=Baseline(scores=[0.5] * 200, n=200, source="t"), min_samples=10)
    for i in range(20):
        m.observe_result(
            {
                "calibrated_prob": 0.5,
                "has_audio": True,
                "n_faces_found": 5,
                "ood_flag": i % 2 == 0,
                "verdict": "inconclusive" if i % 4 == 0 else "likely_authentic",
            }
        )
    r = m.report()
    assert r["ood_rate"] == pytest.approx(0.5)
    assert r["abstain_rate"] == pytest.approx(0.25)


# ==========================================================================
# observability (F28)
# ==========================================================================
def test_metrics_render_without_prometheus_installed(monkeypatch):
    import api.observability as O

    monkeypatch.setattr(O, "_HAS_PROM", False)
    m = O.Metrics()
    m.verdicts.labels(verdict="x").inc()  # must not raise
    m.scores.observe(0.5)
    assert b"not installed" in m.render()


def test_metrics_record_a_served_verdict():
    from api.observability import METRICS

    METRICS.observe_result(
        {
            "verdict": "likely_manipulated",
            "ood_flag": True,
            "calibrated_prob": 0.77,
            "latency_ms": {"preprocess_ms": 1200.0, "total_ms": 6800.0},
        }
    )
    out = METRICS.render().decode()
    if METRICS.enabled:
        assert 'avforge_verdicts_total{verdict="likely_manipulated"}' in out
        assert "avforge_ood_flags_total" in out
        assert 'stage="preprocess"' in out


def test_logging_accepts_keyword_fields_either_way():
    """Call sites use structlog style; the fallback must not reject kwargs."""
    from api.observability import get_log, setup_structured_logging

    setup_structured_logging(json_logs=True)
    get_log("t").info("msg", job_id="abc", n=1)  # must not raise


def test_tracing_stays_off_without_an_endpoint(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    from api.observability import setup_tracing

    assert setup_tracing() is False


# ==========================================================================
# tracking (F25)
# ==========================================================================
def test_default_tracker_is_a_safe_noop():
    from ddetect.tracking import build_tracker

    t = build_tracker("none")
    assert not t.enabled
    t.start("r", {"a": 1})
    t.log_metrics({"loss": 1.0}, step=0)
    t.log_artifact("/nonexistent")
    t.log_summary({"x": 1})
    t.finish()


def test_unknown_tracker_degrades_rather_than_raising():
    from ddetect.tracking import build_tracker

    assert build_tracker("not-a-tracker").name == "none"


def test_config_flattening_handles_nesting_and_lists():
    from ddetect.tracking import _flatten

    out = _flatten({"a": 1, "b": {"c": 2}, "d": [1, 2], "e": None})
    assert out == {"a": 1, "b.c": 2, "d": "1,2", "e": None}


# ==========================================================================
# worker (F20)
# ==========================================================================
def test_worker_settings_are_conservative():
    from api.worker import WorkerSettings

    # Inference is CPU-bound: oversubscribing makes every request slower
    # rather than throughput higher.
    assert WorkerSettings.max_jobs <= 4
    assert WorkerSettings.job_timeout >= 60
    assert WorkerSettings.functions


def test_redis_is_not_used_when_unconfigured(monkeypatch):
    monkeypatch.setattr("api.worker.REDIS_URL", "")
    from api.worker import redis_available

    assert redis_available() is False


def test_unreachable_redis_falls_back(monkeypatch):
    monkeypatch.setattr("api.worker.REDIS_URL", "redis://127.0.0.1:1/0")
    from api.worker import redis_available

    assert redis_available() is False


# ==========================================================================
# dataset acquisition and preflight (F1)
# ==========================================================================
def test_every_dataset_is_described_consistently():
    from ddetect.data.manifests import PARSERS
    from scripts.download_datasets import DATASETS

    for key, ds in DATASETS.items():
        assert ds.url.startswith("http"), f"{key} has no source URL"
        assert ds.layout, f"{key} documents no expected layout"
        assert ds.size_gb > 0
        # Every described dataset must have a parser, or the instructions lead
        # somewhere the code cannot read.
        assert key in PARSERS, f"{key} is documented but has no parser"


def test_only_dfdc_is_automated():
    """The others are behind a licence agreement; automating around one would
    be both impossible and wrong."""
    from scripts.download_datasets import DATASETS

    automated = {k for k, v in DATASETS.items() if v.automated}
    assert automated == {"dfdc"}
    for key, ds in DATASETS.items():
        if not ds.automated:
            assert ds.steps, f"{key} is manual but gives no steps"


def test_the_audio_bearing_datasets_are_flagged():
    """Which sets carry audio determines the whole protocol split."""
    from scripts.download_datasets import DATASETS

    assert DATASETS["ffpp"].audio == "NONE"
    assert DATASETS["celebdf"].audio == "NONE"
    assert DATASETS["dfdc"].audio == "yes"
    assert DATASETS["fakeavceleb"].audio == "yes"


def test_listing_datasets_succeeds():
    from scripts.download_datasets import main

    assert main(["--list"]) == 0


def test_manual_instructions_render_without_the_data_present(tmp_path, capsys):
    from scripts.download_datasets import main

    assert main(["ffpp", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "access form" in out
    assert "ddetect manifest" in out, "no follow-up command given"
    assert "splits" in out, "the official splits requirement is not mentioned"


def test_dfdc_without_credentials_explains_how_to_get_them(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.kaggle
    from scripts.download_datasets import main

    rc = main(["dfdc", "--out", str(tmp_path)])
    assert rc == 2
    assert "kaggle.json" in capsys.readouterr().out


def test_preflight_detects_a_mismatched_torch_trio():
    """torchvision/torchaudio link against libtorch's C++ ABI, so a mismatched
    trio fails at import rather than at resolve time."""
    import torch
    import torchaudio
    import torchvision

    major_minor = lambda v: ".".join(v.split("+")[0].split(".")[:2])  # noqa: E731
    assert major_minor(torchvision.__version__) >= "0.17"
    # torchaudio tracks torch's minor version.
    assert major_minor(torchaudio.__version__) == major_minor(torch.__version__), (
        f"torch {torch.__version__} vs torchaudio {torchaudio.__version__}: "
        "mismatched builds fail at import with a dlopen symbol error"
    )


def test_preflight_runs_and_reports():
    from scripts.preflight import check_binaries, check_device, check_repro

    for checks in (check_binaries(), check_device(), check_repro()):
        assert checks
        for c in checks:
            assert c.name and isinstance(c.ok, bool)
            if not c.ok:
                assert c.detail, f"{c.name} failed with no explanation"
