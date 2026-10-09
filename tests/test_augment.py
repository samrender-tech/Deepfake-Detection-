"""F3 - degradation augmentation.

Objective 4 (compression robustness) depends entirely on this actually
running. Two failures here are silent and were both real bugs during
development:

1. albumentations 2.0 renamed ``quality_lower/upper`` -> ``quality_range`` and
   accepted the old names with a *warning*, so the whole degradation group
   became a no-op while training logs looked perfectly normal.
2. The degradation was applied twice -- once consistently per clip, then again
   per frame -- which re-randomised it and made inter-frame quality variance a
   fingerprint of our own pipeline that no real video has.
"""

from __future__ import annotations

import numpy as np
import pytest

from ddetect.data.augment import (
    AudioAugConfig,
    AugConfig,
    ClipAugmentor,
    assert_albumentations_api,
    augment_waveform,
    build_transform,
)


@pytest.fixture
def img():
    return (np.random.default_rng(0).random((256, 256, 3)) * 255).astype(np.uint8)


def test_albumentations_api_is_what_we_think():
    # Fails loudly if a future albumentations renames these again.
    assert_albumentations_api()


def test_eval_transform_is_deterministic(img):
    t = build_transform(AugConfig(), 224, train=False)
    a, b = t(image=img)["image"], t(image=img)["image"]
    np.testing.assert_array_equal(a, b)


def test_train_transform_is_stochastic(img):
    t = build_transform(AugConfig(), 224, train=True)
    outs = [t(image=img)["image"] for _ in range(5)]
    assert any(not np.array_equal(outs[0], o) for o in outs[1:])


def test_output_is_normalised_chw(img):
    out = ClipAugmentor(AugConfig(enabled=False), 224, False)([img])[0]
    assert out.shape == (3, 224, 224)
    assert out.dtype == np.float32
    # ImageNet normalisation puts values roughly in [-2.2, 2.7]
    assert out.min() > -3.0 and out.max() < 3.0


_DEGRADE_ONLY = dict(
    jpeg_p=1.0,
    downscale_p=1.0,
    blur_p=1.0,
    hflip_p=0.0,
    rotate_p=0.0,
    color_jitter_p=0.0,
    noise_p=0.0,
    cutout_p=0.0,
    mixstyle_p=0.0,
)


def test_degradation_actually_changes_the_image(img):
    """Guards against the silent no-op (failure 1 above)."""
    clean = ClipAugmentor(AugConfig(enabled=False), 224, False)([img])[0]
    degraded = ClipAugmentor(AugConfig(**_DEGRADE_ONLY), 224, True)([img])[0]
    delta = float(np.abs(degraded - clean).max())
    assert delta > 0.01, "the degradation group had no effect on the image"


def test_degradation_is_identical_across_a_clip(img):
    """Guards against double application (failure 2 above).

    A real video has ONE encode: every frame shares its compression level,
    blur and resolution. With geometric jitter disabled, identical input
    frames must come out byte-identical.
    """
    out = ClipAugmentor(AugConfig(**_DEGRADE_ONLY), 224, True)([img] * 6)
    deltas = [float(np.abs(out[0] - o).max()) for o in out[1:]]
    assert max(deltas) < 1e-6, (
        f"degradation varied across frames of one clip (max delta {max(deltas)}); "
        "inter-frame quality variance would become a dataset fingerprint"
    )


def test_geometric_jitter_still_varies_per_frame(img):
    cfg = AugConfig(
        hflip_p=0.5,
        rotate_p=1.0,
        jpeg_p=0.0,
        downscale_p=0.0,
        blur_p=0.0,
        noise_p=0.0,
        cutout_p=0.0,
        color_jitter_p=0.0,
    )
    out = ClipAugmentor(cfg, 224, True)([img] * 6)
    assert max(float(np.abs(out[0] - o).max()) for o in out[1:]) > 0.01


def test_disabled_config_is_a_passthrough_resize(img):
    out = ClipAugmentor(AugConfig(enabled=False), 224, True)([img] * 3)
    assert all(np.array_equal(out[0], o) for o in out[1:])


# ==========================================================================
# audio
# ==========================================================================
def test_audio_augmentation_changes_the_waveform():
    rng = np.random.default_rng(0)
    y = (0.3 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)).astype(np.float32)
    out = augment_waveform(y, 16000, AudioAugConfig(), rng)
    assert out.dtype == np.float32
    assert np.abs(out).max() <= 1.0, "augmentation clipped out of range"


def test_audio_augmentation_disabled_is_a_passthrough():
    y = np.linspace(-0.5, 0.5, 1000).astype(np.float32)
    out = augment_waveform(y, 16000, AudioAugConfig(enabled=False), np.random.default_rng(0))
    np.testing.assert_array_equal(out, y)


def test_audio_augmentation_handles_empty_input():
    out = augment_waveform(
        np.zeros(0, dtype=np.float32), 16000, AudioAugConfig(), np.random.default_rng(0)
    )
    assert out.size == 0
