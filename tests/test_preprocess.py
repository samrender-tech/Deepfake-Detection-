"""F2 - the preprocessing cache (contract 5.2).

The cache is the one thing every workstream depends on, and the same
``preprocess_video`` runs in training and in the API. These tests pin the
behaviour that both rely on, including the two degraded paths the design names
explicitly: no audio, and no face.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ddetect.contracts import CACHE_SCHEMA_VERSION, RELIABILITY_FIELDS
from ddetect.data.manifest_io import probe_video
from ddetect.data.preprocess import (
    AUDIO_SR,
    MOUTH_SIZE,
    crop_face,
    is_cached,
    preprocess_video,
    sample_indices,
)


# ==========================================================================
# frame sampling
# ==========================================================================
def test_uniform_sampling_spans_the_clip():
    idx = sample_indices(100, 8, "uniform")
    assert len(idx) == 8
    assert idx[0] == 0 and idx[-1] == 99
    assert idx == sorted(idx)


def test_sampling_handles_short_clips():
    assert sample_indices(5, 32, "uniform") == list(range(5))


def test_random_sampling_is_seed_deterministic():
    # Training, evaluation and serving must land on the SAME frames for the
    # same video, or the parity test is impossible.
    a = sample_indices(100, 8, "random", seed=42)
    b = sample_indices(100, 8, "random", seed=42)
    c = sample_indices(100, 8, "random", seed=43)
    assert a == b
    assert a != c


def test_dense_sampling_is_contiguous():
    idx = sample_indices(100, 16, "dense")
    assert idx == list(range(idx[0], idx[0] + 16))


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown sampling strategy"):
        sample_indices(100, 8, "spiral")


# ==========================================================================
# cropping
# ==========================================================================
def test_crop_face_reflects_rather_than_zero_pads():
    from ddetect.data.facedet import Detection

    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    # a box running off the left/top edge
    crop = crop_face(frame, Detection((-20.0, -20.0, 30.0, 30.0), 0.9), size=64)
    assert crop.shape == (64, 64, 3)
    # A zero-padded crop would introduce a black bar, which is itself a strong
    # artefact the model would learn as a dataset fingerprint.
    assert crop.min() > 0, "padding introduced black pixels"


def test_crop_face_centre_crops_when_no_detection():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    frame[:, 80:120] = 255
    crop = crop_face(frame, None, size=64)
    assert crop.shape == (64, 64, 3)
    assert crop.max() == 255, "the centre crop missed the centre"


# ==========================================================================
# probing -- the no-audio finding evidence
# ==========================================================================
def test_probe_detects_audio_presence(a_video, a_silent_video):
    assert probe_video(a_video)["has_audio"] is True
    assert probe_video(a_silent_video)["has_audio"] is False


def test_probe_reports_geometry(a_video):
    p = probe_video(a_video)
    assert p["fps"] > 0 and p["n_frames"] > 0 and p["duration_s"] > 0
    assert p["width"] > 0 and p["height"] > 0


# ==========================================================================
# the cache
# ==========================================================================
def test_preprocess_writes_the_full_cache(tmp_path, a_video, permissive_cfg):
    cdir = tmp_path / "custom" / a_video.stem
    meta = preprocess_video(a_video, cdir, permissive_cfg, video_id=a_video.stem)

    assert (cdir / "meta.json").exists()
    assert len(list((cdir / "faces").glob("*.jpg"))) == permissive_cfg.n_frames
    assert (cdir / "audio.wav").exists()
    assert meta.cache_schema_version == CACHE_SCHEMA_VERSION
    assert meta.n_faces_found > 0, "no face found in the face fixture"
    assert set(meta.reliability) == set(RELIABILITY_FIELDS)
    assert len(meta.frame_indices) == len(meta.frame_timestamps) == permissive_cfg.n_frames
    assert len(meta.phash) == permissive_cfg.n_frames


def test_audio_is_16k_mono(tmp_path, a_video, permissive_cfg):
    import soundfile as sf

    cdir = tmp_path / "custom" / a_video.stem
    preprocess_video(a_video, cdir, permissive_cfg, video_id=a_video.stem)
    info = sf.info(str(cdir / "audio.wav"))
    assert info.samplerate == AUDIO_SR
    assert info.channels == 1


def test_mouth_crops_are_the_syncnet_size(tmp_path, a_video, permissive_cfg):
    import cv2

    cdir = tmp_path / "custom" / a_video.stem
    preprocess_video(a_video, cdir, permissive_cfg, video_id=a_video.stem)
    mouths = sorted((cdir / "mouth").glob("*.jpg"))
    assert mouths, "no mouth crops produced"
    m = cv2.imread(str(mouths[0]), cv2.IMREAD_GRAYSCALE)
    assert m.shape == (MOUTH_SIZE, MOUTH_SIZE)


def test_preprocess_is_idempotent_and_resumable(tmp_path, a_video, permissive_cfg):
    cdir = tmp_path / "custom" / a_video.stem
    m1 = preprocess_video(a_video, cdir, permissive_cfg, video_id=a_video.stem)
    assert is_cached(cdir)
    mtime = (cdir / "meta.json").stat().st_mtime_ns

    m2 = preprocess_video(a_video, cdir, permissive_cfg, video_id=a_video.stem)
    assert (cdir / "meta.json").stat().st_mtime_ns == mtime, "re-ran a cached video"
    assert m1.phash == m2.phash


def test_stale_cache_version_is_invalidated(tmp_path, a_video, permissive_cfg):
    cdir = tmp_path / "custom" / a_video.stem
    preprocess_video(a_video, cdir, permissive_cfg, video_id=a_video.stem)

    meta = json.loads((cdir / "meta.json").read_text())
    meta["cache_schema_version"] = CACHE_SCHEMA_VERSION - 1
    (cdir / "meta.json").write_text(json.dumps(meta))

    assert not is_cached(cdir), "a stale cache version was treated as current"


# ==========================================================================
# the degraded paths named in the edge-case list
# ==========================================================================
def test_silent_video_masks_audio_and_warns(tmp_path, a_silent_video, permissive_cfg):
    cdir = tmp_path / "custom" / a_silent_video.stem
    meta = preprocess_video(a_silent_video, cdir, permissive_cfg, video_id=a_silent_video.stem)

    assert meta.has_audio is False
    assert not (cdir / "audio.wav").exists(), "an empty wav was left behind"
    assert meta.reliability["audio_snr"] == 0.0
    assert any("audio" in w for w in meta.warnings)
    # The visual path must still be complete: this is the FF++/Celeb-DF case.
    assert meta.n_faces_found > 0


def test_faceless_video_still_produces_a_cache_and_warns(
    tmp_path, a_faceless_video, permissive_cfg
):
    cdir = tmp_path / "custom" / a_faceless_video.stem
    meta = preprocess_video(a_faceless_video, cdir, permissive_cfg, video_id=a_faceless_video.stem)

    assert meta.n_faces_found == 0
    assert meta.reliability["face_conf"] == 0.0
    assert any("face" in w for w in meta.warnings)
    # Still cached: the API must be able to return "no face found" as a result
    # rather than a 500.
    assert len(list((cdir / "faces").glob("*.jpg"))) > 0


def test_corrupt_video_raises_clearly(tmp_path, permissive_cfg):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not a video at all")
    with pytest.raises(RuntimeError):
        preprocess_video(bad, tmp_path / "custom" / "broken", permissive_cfg, video_id="broken")
