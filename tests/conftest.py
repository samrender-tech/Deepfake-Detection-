"""Shared fixtures.

The synthetic clips come from ``scripts/make_fixtures.py`` and are regenerated
on demand, so a fresh clone needs no dataset access to run the suite.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_VIDEOS = ROOT / "tests" / "fixtures" / "videos"


@pytest.fixture(scope="session")
def fixture_videos() -> Path:
    if not list(FIXTURE_VIDEOS.glob("*.mp4")):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "make_fixtures.py")],
            check=True,
            cwd=ROOT,
            capture_output=True,
        )
    return FIXTURE_VIDEOS


@pytest.fixture(scope="session")
def a_video(fixture_videos: Path) -> Path:
    return fixture_videos / "real_sync_audio_v0.mp4"


@pytest.fixture(scope="session")
def a_silent_video(fixture_videos: Path) -> Path:
    return fixture_videos / "real_silent_v0.mp4"


@pytest.fixture(scope="session")
def a_faceless_video(fixture_videos: Path) -> Path:
    return fixture_videos / "edge_no_face.mp4"


@pytest.fixture(scope="session")
def permissive_cfg():
    """Preprocess config tuned for the synthetic fixtures."""
    from ddetect.data.facedet import PERMISSIVE_MTCNN_THRESHOLDS
    from ddetect.data.preprocess import PreprocessConfig

    return PreprocessConfig(n_frames=8, det_thresholds=PERMISSIVE_MTCNN_THRESHOLDS, device="cpu")


@pytest.fixture(scope="session")
def fixture_manifest(tmp_path_factory, fixture_videos) -> Path:
    from ddetect.data.manifest_io import write_manifest
    from ddetect.data.manifests import FixtureParser

    out = tmp_path_factory.mktemp("manifests") / "fixture.parquet"
    write_manifest(FixtureParser(fixture_videos).build(workers=2), out)
    return out


@pytest.fixture(scope="session")
def fixture_cache(tmp_path_factory, fixture_manifest, permissive_cfg) -> Path:
    """A populated cache for all fixture clips, built once per session.

    Built FROM the manifest, not from the directory listing: the parser
    prefixes video ids (``fx_<stem>``), and keying the cache on the bare file
    stem silently produces a cache the Dataset cannot find -- which yields
    all-zero clips and tests that pass for the wrong reason.
    """
    from ddetect.data.manifest_io import read_manifest
    from ddetect.data.preprocess import cache_dir_for, preprocess_video

    root = tmp_path_factory.mktemp("cache")
    for r in read_manifest(fixture_manifest).itertuples():
        preprocess_video(
            r.path,
            cache_dir_for(root, r.dataset, r.video_id),
            permissive_cfg,
            video_id=r.video_id,
            dataset=r.dataset,
        )
    return root


@pytest.fixture
def smoke_model():
    """Tiny randomly-initialised model: no weight download, runs on CPU."""
    from ddetect.models.registry import build_model

    return build_model(
        {
            "visual": {
                "backbone": "effb0",
                "pretrained": False,
                "image_size": 224,
                "temporal": "attention",
                "embed_dim": 128,
                "use_blend_head": True,
            },
            "audio": {"kind": "logmel", "width": 8, "out_dim": 32},
            "sync": {"kind": "contrastive", "freeze": False, "embed_dim": 32},
            "fusion": {"mode": "coattn", "dim": 64, "depth": 1, "heads": 2},
        }
    )


@pytest.fixture
def a_batch():
    """A contract-5.3 batch of random tensors.

    Deliberately synthetic: the model streams are developed against this
    before the cache exists.
    """
    import torch

    B, T, W = 2, 4, 4
    return {
        "faces": torch.randn(B, T, 3, 224, 224),
        "mouth": torch.randn(B, W, 16, 1, 96, 96),
        "mel": torch.randn(B, 1, 80, 201),
        "wav": torch.randn(B, 32000),
        "has_audio": torch.tensor([True, False]),
        "reliab": torch.rand(B, 4),
        "label": torch.tensor([0.0, 1.0]),
        "mask": torch.rand(B, T, 1, 28, 28),
        "video_id": ["a", "b"],
        "dataset": ["custom", "custom"],
        "forgery_method": ["real", "fixture_synthetic"],
    }
