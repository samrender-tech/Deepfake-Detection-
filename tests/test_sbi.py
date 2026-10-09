"""F4 - Self-Blended Images.

SBI is the strongest generalisation lever in the project, and it is also the one
most able to fail silently: a blend that produces no visible seam trains the
model on nothing, and a mask that covers the whole face teaches it to segment
faces rather than find boundaries. Both failures still produce a falling loss
curve, so they are pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddetect.data.sbi import (
    SBIConfig,
    boundary_target,
    self_blend,
    self_blend_clip,
)


@pytest.fixture
def face_img():
    """A textured, face-like image: flat colour has no blendable detail."""
    import cv2

    rng = np.random.default_rng(0)
    img = np.full((256, 256, 3), 170, dtype=np.uint8)
    cv2.ellipse(img, (128, 130), (80, 100), 0, 0, 360, (195, 165, 145), -1)
    cv2.circle(img, (102, 110), 12, (60, 50, 45), -1)
    cv2.circle(img, (154, 110), 12, (60, 50, 45), -1)
    cv2.ellipse(img, (128, 180), (28, 12), 0, 0, 360, (110, 60, 65), -1)
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def test_blend_changes_the_image(face_img):
    out, _ = self_blend(face_img, seed=0)
    assert out.shape == face_img.shape
    assert out.dtype == np.uint8
    diff = np.abs(out.astype(float) - face_img.astype(float)).mean()
    assert diff > 1.0, f"self-blend produced a near-identical image (mean diff {diff:.3f})"


def test_alpha_is_a_valid_mask(face_img):
    _, alpha = self_blend(face_img, seed=1)
    assert alpha.shape == face_img.shape[:2]
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0


def test_boundary_target_peaks_only_on_the_seam(face_img):
    _, alpha = self_blend(face_img, seed=2)
    t = boundary_target(alpha)

    assert t.min() >= 0.0 and t.max() <= 1.0
    # 4a(1-a) must be ~0 deep inside and deep outside the blended region, and
    # ~1 only where alpha is near 0.5. A target that fired everywhere would
    # train a face segmenter instead of a boundary detector.
    inside = alpha > 0.98
    outside = alpha < 0.02
    if inside.any():
        assert t[inside].max() < 0.1, "target is non-zero inside the blended region"
    if outside.any():
        assert t[outside].max() < 0.1, "target is non-zero outside the blended region"

    seam_fraction = float((t > 0.5).mean())
    assert 0.0005 < seam_fraction < 0.30, (
        f"seam covers {seam_fraction:.1%} of the image; expected a thin band"
    )


def test_boundary_target_closed_form():
    a = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    np.testing.assert_allclose(boundary_target(a), [0.0, 0.75, 1.0, 0.75, 0.0], atol=1e-6)


def test_same_seed_reproduces_the_blend(face_img):
    a, ma = self_blend(face_img, seed=7)
    b, mb = self_blend(face_img, seed=7)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(ma, mb)


def test_different_seeds_differ(face_img):
    a, _ = self_blend(face_img, seed=1)
    b, _ = self_blend(face_img, seed=2)
    assert not np.array_equal(a, b)


def test_clip_blend_is_temporally_coherent(face_img):
    """A real face swap is applied by one pipeline on every frame.

    If the mask were resampled per frame, the seam would flicker -- a temporal
    artefact of our own making that the temporal head would detect trivially
    and that would transfer to nothing.
    """
    frames = [face_img] * 12
    out, masks = self_blend_clip(frames, SBIConfig(), seed=3)
    assert len(out) == len(masks) == 12

    centroids = []
    for m in masks:
        ys, xs = np.nonzero(m > 0.3)
        centroids.append((xs.mean(), ys.mean()) if len(xs) else (np.nan, np.nan))
    c = np.array([p for p in centroids if not np.isnan(p[0])])
    assert len(c) >= 8, "most frames produced no seam"
    # Frame-to-frame drift must be small relative to the image.
    drift = np.abs(np.diff(c, axis=0)).max()
    assert drift < face_img.shape[0] * 0.25, f"seam jumped {drift:.1f}px between frames"


def test_dataset_converts_a_real_clip_into_a_supervised_fake(fixture_manifest, fixture_cache):
    """The end-to-end SBI path: label flips to 1 and a finite mask appears."""
    import torch

    from ddetect.data.dataset import DataConfig, DeepfakeClipDataset
    from ddetect.data.manifest_io import read_manifest

    df = read_manifest(fixture_manifest, split="train")
    reals = df[df.label == 0]
    assert len(reals), "no real videos in the fixture train split"

    cfg = DataConfig(
        manifest=str(fixture_manifest),
        cache_root=str(fixture_cache),
        split="train",
        n_frames=4,
        image_size=224,
        audio_seconds=1.0,
        max_sync_windows=2,
        sbi=SBIConfig(enabled=True, p=1.0),
        require_cache=False,
    )
    ds = DeepfakeClipDataset(cfg, df=reals.reset_index(drop=True))
    item = ds[0]

    assert float(item["label"]) == 1.0, "SBI did not relabel the real clip as fake"
    assert torch.isfinite(item["mask"]).all(), "SBI produced no boundary supervision"
    assert float(item["mask"].max()) > 0.1, "the boundary target is empty"


def test_dataset_leaves_real_clips_alone_when_sbi_disabled(fixture_manifest, fixture_cache):
    import torch

    from ddetect.data.dataset import DataConfig, DeepfakeClipDataset
    from ddetect.data.manifest_io import read_manifest

    df = read_manifest(fixture_manifest, split="train")
    reals = df[df.label == 0].reset_index(drop=True)
    cfg = DataConfig(
        manifest=str(fixture_manifest),
        cache_root=str(fixture_cache),
        split="train",
        n_frames=4,
        image_size=224,
        audio_seconds=1.0,
        max_sync_windows=2,
        sbi=SBIConfig(enabled=False),
        require_cache=False,
    )
    item = DeepfakeClipDataset(cfg, df=reals)[0]
    assert float(item["label"]) == 0.0
    # NaN, per contract 5.3: the loss masks these out rather than training the
    # head toward an all-zero map.
    assert torch.isnan(item["mask"]).all()
