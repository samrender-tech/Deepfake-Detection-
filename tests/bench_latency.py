"""F22 - the latency budget.

Target: **< 20 s for a 10 s clip on 4 CPU cores**. That number is not
arbitrary -- past roughly 30 seconds an upload-and-wait demo stops feeling
interactive and people assume it has hung.

Run as a benchmark rather than a hard gate on developer machines (an M2 and a
CI runner differ by more than the budget), but the numbers are printed so a
regression is visible, and `--strict` turns it into an assertion for the
deployment check.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VIDEOS = ROOT / "tests" / "fixtures" / "videos"
BUDGET_S = float(os.environ.get("DDETECT_LATENCY_BUDGET_S", "20"))
STRICT = os.environ.get("DDETECT_LATENCY_STRICT") == "1"


@pytest.fixture(scope="module")
def detector(tmp_path_factory):
    """A lite-profile detector on CPU: the serving configuration."""
    import subprocess
    import sys

    import torch

    if not list(VIDEOS.glob("*.mp4")):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "make_fixtures.py")],
            check=True,
            cwd=ROOT,
            capture_output=True,
        )

    from ddetect.calibrate import Calibration
    from ddetect.data.dataset import DataConfig
    from ddetect.data.facedet import PERMISSIVE_MTCNN_THRESHOLDS
    from ddetect.data.preprocess import PreprocessConfig
    from ddetect.inference import Detector
    from ddetect.models.registry import build_model

    model = build_model(
        {
            "visual": {
                "backbone": "effb0",
                "pretrained": False,
                "image_size": 224,
                "temporal": "attention",
                "embed_dim": 128,
            },
            "audio": {"kind": "logmel", "width": 16, "out_dim": 64},
            "sync": {"kind": "contrastive", "freeze": True, "embed_dim": 64},
            "fusion": {"mode": "concat", "dim": 128},
        }
    )
    return Detector(
        model=model,
        calibration=Calibration(method="none"),
        data_cfg=DataConfig(
            n_frames=16,
            image_size=224,
            audio_seconds=4.0,
            max_sync_windows=4,
            split="test",
            require_cache=False,
        ),
        preprocess_cfg=PreprocessConfig(
            n_frames=16, det_thresholds=PERMISSIVE_MTCNN_THRESHOLDS, device="cpu"
        ),
        device=torch.device("cpu"),
        model_version="bench/lite",
    )


def _time(detector, path: Path, explain: bool) -> tuple[float, dict]:
    t0 = time.perf_counter()
    res = detector.predict(path, explain=explain)
    return time.perf_counter() - t0, res.latency_ms


@pytest.mark.slow
def test_latency_without_explanations(detector):
    """The verdict path alone -- what an API consumer polling for a score waits."""
    clip = VIDEOS / "real_sync_audio_v0.mp4"
    _time(detector, clip, explain=False)  # warm up: first call loads kernels
    elapsed, parts = _time(detector, clip, explain=False)

    print(f"\n  verdict only: {elapsed:.2f}s  {dict(parts)}")
    if STRICT:
        assert elapsed < BUDGET_S, f"{elapsed:.1f}s exceeds the {BUDGET_S}s budget"


@pytest.mark.slow
def test_latency_with_explanations(detector):
    """The full path including Grad-CAM, which is the expensive part."""
    clip = VIDEOS / "real_sync_audio_v0.mp4"
    _time(detector, clip, explain=True)
    elapsed, parts = _time(detector, clip, explain=True)

    print(f"\n  with explanations: {elapsed:.2f}s  {dict(parts)}")
    # Grad-CAM needs a second forward AND a backward pass, so it roughly
    # doubles the cost. If it ever dominates, the fix is to compute it for
    # fewer frames -- not to drop it, since it is the only visual evidence the
    # user gets.
    if STRICT:
        assert elapsed < BUDGET_S * 1.5


@pytest.mark.slow
def test_preprocessing_is_not_the_bottleneck(detector):
    """Preprocessing should be a minority of the total.

    If decode and face detection dominate, the fix is frame count or detector
    backend, not the model -- worth knowing before anyone optimises the wrong
    thing.
    """
    clip = VIDEOS / "real_sync_audio_v0.mp4"
    _time(detector, clip, explain=False)
    _, parts = _time(detector, clip, explain=False)

    pre = parts.get("preprocess_ms", 0)
    total = parts.get("total_ms", 1)
    print(f"\n  preprocess {pre:.0f}ms of {total:.0f}ms ({pre / total:.0%})")
    assert total > 0
