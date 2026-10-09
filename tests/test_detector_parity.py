"""F15 - ``Detector`` parity with training-time evaluation.

Contract 5.2's whole purpose: the demo calls the SAME ``preprocess_video`` and
the SAME Dataset loading as training, so a verdict shown to a user is the
verdict the paper's numbers describe.

This test caught the single worst bug in development: the Detector resolved its
cache root one directory too high, found no frames, and scored an all-zero clip
-- returning an identical confident verdict for every video, with no error.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ddetect.contracts import Verdict


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    """Train the smoke model for 1 epoch on the fixtures -> a run directory."""
    # fixtures are session-scoped; re-derive them here without the fixture
    # machinery so this module-scoped fixture can depend on them.
    import subprocess
    import sys
    from pathlib import Path

    from ddetect.train import TrainConfig, train

    root = Path(__file__).resolve().parent.parent
    vids = root / "tests" / "fixtures" / "videos"
    if not list(vids.glob("*.mp4")):
        subprocess.run(
            [sys.executable, str(root / "scripts" / "make_fixtures.py")],
            check=True,
            cwd=root,
            capture_output=True,
        )

    from ddetect.data.facedet import PERMISSIVE_MTCNN_THRESHOLDS
    from ddetect.data.manifest_io import read_manifest, write_manifest
    from ddetect.data.manifests import FixtureParser
    from ddetect.data.preprocess import PreprocessConfig, cache_dir_for, preprocess_video

    base = tmp_path_factory.mktemp("parity")
    man = base / "fixture.parquet"
    write_manifest(FixtureParser(vids).build(workers=2), man)

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
        exp="parity",
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
    return cfg.run_dir, man, cache


def test_detector_loads_from_a_run(trained_run):
    from ddetect.inference import Detector

    run_dir, _, _ = trained_run
    d = Detector.from_run(run_dir)
    assert d.model_version
    # Inference geometry must come from the checkpoint, never from a caller.
    assert d.data_cfg.n_frames == 8
    assert d.data_cfg.image_size == 224


def test_detector_refuses_an_all_zero_clip(trained_run, tmp_path):
    """The blank-input bug must now fail loudly rather than silently."""
    from ddetect.inference import Detector

    run_dir, _, _ = trained_run
    d = Detector.from_run(run_dir)

    empty = tmp_path / "custom" / "ghost"
    empty.mkdir(parents=True)
    (empty / "faces").mkdir()
    (empty / "meta.json").write_text('{"dataset":"custom","has_audio":false,"fps":25.0}')

    with pytest.raises(RuntimeError, match=r"all-zero clip|cache"):
        d._build_batch(empty, "ghost")


def test_detector_produces_distinct_scores_for_distinct_videos(trained_run):
    """Guards the exact failure the cache-path bug caused."""
    from pathlib import Path

    from ddetect.inference import Detector

    run_dir, _man, _ = trained_run
    d = Detector.from_run(run_dir)
    vids = Path(__file__).resolve().parent / "fixtures" / "videos"

    embs = []
    for name in ("real_sync_audio_v0", "fake_desync_audio_v0", "edge_no_face"):
        res = d.predict(vids / f"{name}.mp4", explain=False)
        embs.append(res.raw_score)
        assert 0.0 <= res.raw_score <= 1.0

    # The three clips differ substantially in content, so an untrained model
    # may still score them similarly -- but the EMBEDDINGS must differ, which
    # proves distinct input actually reached the network.
    import tempfile

    from ddetect.data.preprocess import preprocess_video

    feats = []
    for name in ("real_sync_audio_v0", "edge_no_face"):
        with tempfile.TemporaryDirectory() as td:
            cdir = Path(td) / "custom" / name
            preprocess_video(vids / f"{name}.mp4", cdir, d.pre_cfg, video_id=name, dataset="custom")
            batch, _ = d._build_batch(cdir, name)
            feats.append(d._forward(batch)["emb"])
    assert float((feats[0] - feats[1]).abs().max()) > 1e-4, (
        "two very different videos produced the same embedding -- the "
        "Detector is probably not reading the cache it wrote"
    )


def test_detector_matches_training_time_evaluation(trained_run):
    """The parity claim: same cache -> same logit, via both code paths."""

    from torch.utils.data import DataLoader

    from ddetect.data.dataset import DataConfig, DeepfakeClipDataset, collate
    from ddetect.inference import Detector
    from ddetect.train import predict_split

    run_dir, man, cache = trained_run
    d = Detector.from_run(run_dir)

    from ddetect.data.manifest_io import read_manifest

    df = read_manifest(man, split="val").head(2).reset_index(drop=True)
    dcfg = DataConfig(
        manifest=str(man),
        cache_root=str(cache),
        split="val",
        n_frames=8,
        image_size=224,
        audio_seconds=2.0,
        max_sync_windows=4,
    )
    loader = DataLoader(DeepfakeClipDataset(dcfg, df=df), batch_size=2, collate_fn=collate)
    train_side = predict_split(d.model, loader, d.device, desc="parity")
    train_logits = dict(zip(train_side["video_id"], np.asarray(train_side["logit"]), strict=True))

    # Detector side: predict from the SAME cache directory, so only the code
    # path differs, not the pixels.
    for r in df.itertuples():
        from ddetect.data.preprocess import cache_dir_for

        batch, _ = d._build_batch(cache_dir_for(cache, r.dataset, r.video_id), r.video_id)
        with torch.no_grad():
            logit = float(d._forward(batch)["logit"][0])
        assert logit == pytest.approx(train_logits[r.video_id], abs=1e-4), (
            f"{r.video_id}: Detector {logit:.6f} != training-time {train_logits[r.video_id]:.6f}"
        )


def test_verdict_respects_the_abstention_band(trained_run):
    from pathlib import Path

    from ddetect.inference import Detector

    run_dir, _, _ = trained_run
    d = Detector.from_run(run_dir)
    vids = Path(__file__).resolve().parent / "fixtures" / "videos"
    res = d.predict(vids / "real_sync_audio_v0.mp4", explain=False)

    in_band = d.cal.conformal_lo <= res.calibrated_prob <= d.cal.conformal_hi
    if in_band or res.ood_flag:
        assert res.verdict is Verdict.INCONCLUSIVE
    else:
        assert res.verdict in (Verdict.AUTHENTIC, Verdict.MANIPULATED)
        expected = (
            Verdict.MANIPULATED if res.calibrated_prob >= d.cal.threshold else Verdict.AUTHENTIC
        )
        assert res.verdict is expected


def test_silent_video_masks_audio_streams(trained_run):
    from pathlib import Path

    from ddetect.inference import Detector

    run_dir, _, _ = trained_run
    d = Detector.from_run(run_dir)
    vids = Path(__file__).resolve().parent / "fixtures" / "videos"
    res = d.predict(vids / "real_silent_v0.mp4", explain=False)

    assert res.has_audio is False
    for s in res.streams:
        if s.name in ("audio", "sync"):
            assert not s.available
            assert s.score is None
            assert "no audio" in s.note
    if res.gate_weights:
        assert res.gate_weights["audio"] == 0.0
        assert res.gate_weights["sync"] == 0.0


def test_faceless_video_warns_but_still_returns(trained_run):
    from pathlib import Path

    from ddetect.inference import Detector

    run_dir, _, _ = trained_run
    d = Detector.from_run(run_dir)
    vids = Path(__file__).resolve().parent / "fixtures" / "videos"
    res = d.predict(vids / "edge_no_face.mp4", explain=False)

    assert res.n_faces_found == 0
    assert any("face" in w.lower() for w in res.warnings)


def test_explain_payload_contains_only_result_fields(trained_run):
    """A7's grounding depends on this payload being closed over the Result."""
    from pathlib import Path

    from ddetect.inference import Detector

    run_dir, _, _ = trained_run
    d = Detector.from_run(run_dir)
    vids = Path(__file__).resolve().parent / "fixtures" / "videos"
    res = d.predict(vids / "real_sync_audio_v0.mp4", explain=False)
    payload = d.explain_payload(res)

    assert payload["verdict"] == res.verdict.value
    assert payload["calibrated_probability"] == pytest.approx(res.calibrated_prob, abs=1e-4)
    assert payload["has_audio"] == res.has_audio
    assert payload["n_faces_found"] == res.n_faces_found
    assert payload["ood_flag"] == res.ood_flag
    # Must be JSON-serialisable: it is sent to an LLM and persisted in the audit log.
    import json

    json.dumps(payload)


def test_explain_artifacts_are_produced(trained_run):
    from pathlib import Path

    from ddetect.inference import Detector

    run_dir, _, _ = trained_run
    d = Detector.from_run(run_dir)
    vids = Path(__file__).resolve().parent / "fixtures" / "videos"
    res = d.predict(vids / "real_sync_audio_v0.mp4", explain=True)

    assert res.frame_scores and len(res.frame_scores) == 8
    assert res.gradcam_keys, "no Grad-CAM overlays produced"
    assert res.latency_ms.get("total_ms", 0) > 0


def test_concurrent_predictions_match_sequential_ones(trained_run):
    """The API shares one Detector across worker threads.

    Grad-CAM hooks on the shared model used to capture another thread's
    forward pass, which either crashed the explanation (silently caught) or
    attached a heat map computed from a different clip. Concurrent results
    must be identical to sequential ones, heat maps included.
    """
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    from ddetect.inference import Detector

    run_dir, _, _ = trained_run
    d = Detector.from_run(run_dir)
    vids = Path(__file__).resolve().parent / "fixtures" / "videos"
    clips = [vids / "real_sync_audio_v0.mp4", vids / "fake_desync_audio_v0.mp4"] * 2

    seq = [d.predict(c, explain=True) for c in clips]
    with ThreadPoolExecutor(max_workers=4) as pool:
        par = list(pool.map(lambda c: d.predict(c, explain=True), clips))

    for a, b in zip(seq, par, strict=True):
        assert b.gradcam_keys, "explanation failed under concurrency"
        assert a.calibrated_prob == pytest.approx(b.calibrated_prob, abs=1e-6)
        assert a.gradcam_keys == b.gradcam_keys
        assert a._gradcam_png == b._gradcam_png, "heat map differs from the sequential run"
