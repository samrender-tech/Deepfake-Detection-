"""Contract 5.x enforcement.

These tests are the mechanism that lets each part be built separately: if a
boundary changes without agreement, this file fails rather than some
downstream component silently misaligning.
"""

from __future__ import annotations

import pytest
import torch

from ddetect.contracts import (
    BATCH_META_KEYS,
    BATCH_SPEC,
    MANIFEST_COLUMNS,
    PREDS_COLUMNS,
    RELIABILITY_FIELDS,
    REQUIRED_MODEL_OUTPUT_KEYS,
    ManifestRow,
    Result,
    Verdict,
)
from ddetect.models.registry import PRESETS, make_config


def _build_preset(preset: str):
    """Instantiate a preset with weight downloads disabled.

    Presets whose optional extra is not installed are skipped rather than
    failed: the log-mel audio tier is the one that must always work, and the
    WavLM tier is explicitly an upgrade path.
    """
    from ddetect.models.avforge import AVForge

    spec = {k: dict(v) if isinstance(v, dict) else v for k, v in PRESETS[preset].items()}
    spec["visual"] = {**spec["visual"], "pretrained": False, "image_size": 224}
    if spec.get("audio", {}).get("kind") == "wavlm":
        pytest.importorskip("transformers", reason="the 'audio' extra is not installed")
    try:
        return AVForge(make_config(spec))
    except OSError as e:  # no network for the pretrained SSL weights
        pytest.skip(f"{preset}: could not fetch pretrained weights ({e})")


def test_batch_spec_covers_dataset_output(a_batch):
    for key, spec in BATCH_SPEC.items():
        if spec.required:
            assert key in a_batch, f"required batch key {key!r} missing"
        if key in a_batch:
            assert a_batch[key].ndim == spec.ndim, (
                f"{key}: expected {spec.ndim} dims {spec.dims}, got {a_batch[key].shape}"
            )


def test_meta_keys_are_not_tensors(a_batch):
    for k in BATCH_META_KEYS:
        assert k in a_batch
        assert not torch.is_tensor(a_batch[k]), f"{k} must stay a python list"


def test_reliability_field_order_is_stable():
    # The fusion gate and the UI index this positionally; reordering it would
    # silently swap "audio quality" for "blurriness".
    assert RELIABILITY_FIELDS == ("audio_snr", "face_conf", "mouth_vis", "blur")


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_every_preset_builds_and_honours_output_contract(preset, a_batch):
    model = _build_preset(preset).eval()

    with torch.no_grad():
        out = model(a_batch)

    for k in REQUIRED_MODEL_OUTPUT_KEYS:
        assert k in out, f"{preset}: model output missing required key {k!r}"
    assert out["logit"].shape == (a_batch["faces"].shape[0],)
    assert torch.isfinite(out["logit"]).all(), f"{preset}: non-finite logit"


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_every_preset_backprops(preset, a_batch):
    from ddetect.losses import CompositeLoss

    model = _build_preset(preset)

    out = model(a_batch)
    loss, parts = CompositeLoss(lambda_mask=0.3, lambda_supcon=0.1, lambda_sync=0.2)(out, a_batch)
    assert torch.isfinite(loss), f"{preset}: non-finite loss {parts}"
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, f"{preset}: no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads), f"{preset}: non-finite gradient"


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_output_does_not_depend_on_batch_composition(preset, a_batch):
    """A sample's score must not change because of what else is in the batch.

    This is a fundamental inference invariant, and breaking it is almost
    invisible: training and validation both batch, so curves look healthy
    while the served verdict for a single upload silently disagrees with the
    evaluated one.

    It was broken for real. ``SyncStream.sync_features`` used ``torch.roll``,
    which is cyclic: with W windows, offsets k and k-W give a bit-identical
    distance, and those exact ties were broken by ~1e-7 of batching noise.
    The argmin flipped, ``best_offset`` moved a whole step (0.33 normalised),
    and the final logit moved 0.15 between batch sizes 1 and 2.
    """
    model = _build_preset(preset).eval()
    # Slicing works for tensors and for the meta-key lists alike.
    single = {k: v[:1] for k, v in a_batch.items()}

    with torch.no_grad():
        alone = model(single)["logit"][0]
        # Same sample duplicated: any batch-composition dependence shows up
        # here even though the input is literally identical.
        doubled = model(
            {k: (torch.cat([v, v], 0) if torch.is_tensor(v) else v + v) for k, v in single.items()}
        )["logit"][0]
        in_mixed = model(a_batch)["logit"][0]

    assert float((alone - doubled).abs()) < 1e-5, (
        f"{preset}: logit changed when the same sample was duplicated "
        f"({float(alone):.6f} vs {float(doubled):.6f})"
    )
    assert float((alone - in_mixed).abs()) < 1e-5, (
        f"{preset}: logit changed when batched with a different sample "
        f"({float(alone):.6f} vs {float(in_mixed):.6f})"
    )


def test_manifest_row_rejects_label_method_disagreement():
    base = dict(
        video_id="v",
        path="p",
        split="train",
        has_audio=False,
        source_identity="i",
        fps=25.0,
        n_frames=10,
        duration_s=0.4,
        sha256="0" * 64,
        dataset="ffpp",
    )
    # authentic row must say 'real'
    with pytest.raises(ValueError):
        ManifestRow(**base, label=0, forgery_method="Deepfakes")
    # manipulated row must name a generator
    with pytest.raises(ValueError):
        ManifestRow(**base, label=1, forgery_method="real")
    # both valid forms accepted
    ManifestRow(**base, label=0, forgery_method="real")
    ManifestRow(**base, label=1, forgery_method="Deepfakes")


def test_manifest_columns_match_model_fields():
    assert tuple(ManifestRow.model_fields.keys()) == MANIFEST_COLUMNS


def test_preds_columns_are_what_metrics_reads():
    for essential in ("video_id", "label", "video_score", "calibrated_prob", "abstain"):
        assert essential in PREDS_COLUMNS


def test_result_defaults_to_inconclusive():
    # Section 13: the system must never default to asserting a verdict.
    assert Result().verdict is Verdict.INCONCLUSIVE
    assert not Result().is_confident


def test_result_is_json_serialisable():
    import json

    r = Result(video_id="x", verdict=Verdict.MANIPULATED, calibrated_prob=0.7)
    d = r.to_dict()
    assert d["verdict"] == "likely_manipulated"
    json.dumps(d)


def test_ood_flag_forces_not_confident():
    r = Result(verdict=Verdict.MANIPULATED, ood_flag=True)
    assert not r.is_confident, "an OOD-flagged verdict must not read as confident"
