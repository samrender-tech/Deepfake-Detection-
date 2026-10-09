"""Frozen interface contracts.

These definitions are the boundary between the project's three parts. They
are versioned, test-enforced (``tests/test_contracts.py``) and must not be
changed casually: the metrics code reads ``preds_*.csv`` without ever importing
a model, the model streams are built against ``BATCH_SPEC`` without ever
touching the cache, and the serving path crosses into ``Result``.

Bump the relevant ``*_VERSION`` when a shape or column changes, so stale caches
and stale prediction files fail loudly instead of silently misaligning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# --------------------------------------------------------------------------
# Versions. A bump invalidates on-disk artefacts of that kind.
# --------------------------------------------------------------------------
MANIFEST_SCHEMA_VERSION = 1
CACHE_SCHEMA_VERSION = 1
PREDS_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1

# --------------------------------------------------------------------------
# 5.1  manifest.parquet
# --------------------------------------------------------------------------
DatasetName = Literal["ffpp", "celebdf", "dfdc", "fakeavceleb", "lavdf", "custom"]
SplitName = Literal["train", "val", "test"]

#: Compression / quality tier. FF++ ships c0 (raw), c23, c40; our offline
#: degradation grid writes crf23/crf32/crf40 manifests.
CompressionTag = Literal["raw", "c0", "c23", "c40", "crf23", "crf32", "crf40", "unknown"]


class ManifestRow(BaseModel):
    """One video. The only thing any downstream component is handed.

    ``source_identity`` is what the leakage guard (F5) groups on: it must be
    stable for a person across real *and* fake videos derived from them, or an
    identity-disjoint split is impossible to verify. Each parser is responsible
    for deriving it (FF++ youtube id, Celeb-DF ``idN``, DFDC the real source
    video, FakeAVCeleb the speaker directory).
    """

    model_config = {"frozen": True, "extra": "forbid"}

    video_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    label: int = Field(ge=0, le=1, description="0 = authentic, 1 = manipulated")
    dataset: DatasetName
    forgery_method: str = Field(
        default="real",
        description="'real' for authentic; else the generator (Deepfakes, FaceSwap, "
        "Face2Face, NeuralTextures, rtvc, wav2lip, ...). Used for per-method "
        "breakdowns and the method-balanced sampler.",
    )
    split: SplitName
    has_audio: bool
    source_identity: str = Field(
        min_length=1,
        description="Identity/provenance group key. Splits must be disjoint on this.",
    )
    compression: CompressionTag = "unknown"
    fps: float = Field(gt=0)
    n_frames: int = Field(gt=0)
    duration_s: float = Field(gt=0)
    sha256: str = Field(min_length=64, max_length=64)

    @field_validator("forgery_method")
    @classmethod
    def _real_iff_label_zero(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def _check_label_method_agreement(self) -> ManifestRow:
        if self.label == 0 and self.forgery_method != "real":
            raise ValueError(
                f"{self.video_id}: label=0 (authentic) but forgery_method="
                f"{self.forgery_method!r}; authentic rows must use 'real'"
            )
        if self.label == 1 and self.forgery_method == "real":
            raise ValueError(
                f"{self.video_id}: label=1 (manipulated) but forgery_method='real'; "
                "name the generator so per-method breakdowns work"
            )
        return self


MANIFEST_COLUMNS: tuple[str, ...] = tuple(ManifestRow.model_fields.keys())

# --------------------------------------------------------------------------
# 5.3  batch dict
#
# Declared as data rather than code so test_contracts.py can check every
# registered model against it without instantiating a dataset.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorSpec:
    """Expected shape of one batch key.

    ``dims`` mixes ints (a fixed extent, e.g. the 3 RGB channels or SyncNet's
    16-frame window) with named strings for extents that vary by config
    (``"B"``, ``"T"``, ``"H"``). ``None`` means unconstrained.
    """

    dims: tuple[str | int | None, ...]
    dtype: str
    required: bool
    doc: str

    @property
    def ndim(self) -> int:
        return len(self.dims)


#: T = frames per clip (default 32), W_ = 1-second sync windows,
#: L = mel frames, N = raw audio samples, B = batch.
BATCH_SPEC: dict[str, TensorSpec] = {
    "faces": TensorSpec(("B", "T", 3, "H", "W"), "float32", True, "normalised face crops"),
    "mouth": TensorSpec(
        ("B", "W_", 16, 1, 96, 96), "float32", False, "grey mouth crops for SyncNet"
    ),
    "mel": TensorSpec(("B", 1, 80, "L"), "float32", False, "log-mel spectrogram"),
    "wav": TensorSpec(("B", "N"), "float32", False, "16 kHz mono waveform for WavLM"),
    "has_audio": TensorSpec(("B",), "bool", True, "false => audio/sync masked off"),
    "reliab": TensorSpec(
        ("B", 4), "float32", True, "[audio_snr, face_conf, mouth_vis, blur] in [0,1]"
    ),
    "label": TensorSpec(("B",), "float32", True, "0.0 authentic / 1.0 manipulated"),
    "mask": TensorSpec(
        ("B", "T", 1, "h", "w"), "float32", False, "SBI blend-boundary target; NaN if n/a"
    ),
}

#: Non-tensor batch keys.
BATCH_META_KEYS: tuple[str, ...] = ("video_id", "dataset", "forgery_method")

#: Order of the reliability vector. Referenced by the reliability gate (F10)
#: and surfaced in the explainability payload (F23), so it is fixed here.
RELIABILITY_FIELDS: tuple[str, ...] = ("audio_snr", "face_conf", "mouth_vis", "blur")

#: Keys a model's ``forward`` may return. ``logit`` is the only mandatory one.
MODEL_OUTPUT_KEYS: tuple[str, ...] = (
    "logit",  # [B]      video-level logit                     (required)
    "frame_logits",  # [B, T]   per-frame logits, for the UI timeline
    "emb",  # [B, D]   fusion-space embedding, for OOD (F13)
    "stream_logits",  # dict    {visual, audio, sync} -> [B]
    "sync_seq",  # [B, W_]  per-window AV distance, for the UI chart
    "pred_mask",  # [B,T,1,h,w] predicted blending boundary
    "gate_weights",  # [B, 3]   reliability gate output
)
REQUIRED_MODEL_OUTPUT_KEYS: tuple[str, ...] = ("logit",)

# --------------------------------------------------------------------------
# 5.4  runs/{exp}/{seed}/preds_{testset}.csv
# --------------------------------------------------------------------------

#: The ONLY interface between training and evaluation. F16-F18 read this and
#: nothing else, which is what lets the metrics stack be written and tested
#: before any model exists.
PREDS_COLUMNS: tuple[str, ...] = (
    "video_id",
    "label",
    "dataset",
    "forgery_method",
    "compression",
    "video_score",  # raw model score (sigmoid of logit), [0,1]
    "calibrated_prob",  # after temperature scaling (F11); == video_score if uncalibrated
    "frame_scores",  # json list[float], may be "[]"
    "sync_feats",  # json dict, may be "{}"
    "stream_scores",  # json dict {visual,audio,sync}, may be "{}"
    "ood_score",  # F13 energy score; higher = more novel
    "has_audio",
    "abstain",  # bool, from the conformal band (F12)
)

#: Files a completed run directory must contain (checked by
#: ``experiments/aggregate.py`` before it will read a run).
RUN_ARTEFACTS: tuple[str, ...] = (
    "config.yaml",
    "git_sha",
    "env.lock",
    "metrics.jsonl",
)

# --------------------------------------------------------------------------
# 5.5  Result -- the offline/online boundary
# --------------------------------------------------------------------------


class Verdict(str, Enum):
    """Three states, never two.

    A binary real/fake call at cross-dataset accuracy is misinformation, so
    ``INCONCLUSIVE`` is a first-class outcome produced by the conformal band
    (F12) and is rendered as such by the API and the UI. See the responsible-use policy.
    """

    AUTHENTIC = "likely_authentic"
    MANIPULATED = "likely_manipulated"
    INCONCLUSIVE = "inconclusive"


class StreamScore(BaseModel):
    """One stream's contribution, for the UI's per-stream bars."""

    model_config = {"extra": "forbid"}

    name: Literal["visual", "audio", "sync"]
    score: float | None = Field(None, ge=0.0, le=1.0)
    available: bool
    reliability: float = Field(1.0, ge=0.0, le=1.0)
    note: str = ""


@dataclass
class Result:
    """What ``Detector.predict`` returns; serialised identically by the CLI,
    the API and A7's input payload.

    Every number A7 is allowed to mention in a user-facing explanation comes
    from this object: the grounding eval replays these fields with
    values removed or contradicted and requires the agent to refuse.
    """

    # --- identity / provenance -------------------------------------------
    schema_version: int = RESULT_SCHEMA_VERSION
    video_id: str = ""
    model_version: str = ""
    config_hash: str = ""
    run_dir: str = ""

    # --- the verdict ------------------------------------------------------
    verdict: Verdict = Verdict.INCONCLUSIVE
    calibrated_prob: float = 0.5
    conformal_lo: float = 0.0
    conformal_hi: float = 1.0
    raw_score: float = 0.5
    threshold: float = 0.5

    # --- evidence (F23) ---------------------------------------------------
    streams: list[StreamScore] = field(default_factory=list)
    frame_scores: list[float] = field(default_factory=list)
    frame_timestamps: list[float] = field(default_factory=list)
    sync_curve: list[float] = field(default_factory=list)
    sync_offset_ms: float | None = None
    sync_confidence: float | None = None
    gate_weights: dict[str, float] = field(default_factory=dict)
    gradcam_keys: list[str] = field(default_factory=list)
    spectrum_key: str | None = None

    # --- honesty flags ----------------------------------------------------
    ood_score: float = 0.0
    ood_flag: bool = False
    has_audio: bool = False
    n_faces_found: int = 0
    reliability: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # --- narrative (A7, optional; never blocks) ---------------------------
    explanation: str | None = None
    explanation_source: Literal["agent", "template", "none"] = "none"

    # --- timing -----------------------------------------------------------
    latency_ms: dict[str, float] = field(default_factory=dict)

    # --- inline artefacts -------------------------------------------------
    #: Base64 PNGs attached by the explainability pass and stripped by the API
    #: worker once written to object storage. Declared here (rather than set
    #: with setattr) so the names are checkable and a typo cannot silently
    #: drop a heat map.
    _gradcam_png: dict[str, str] = field(default_factory=dict, repr=False)
    _spectrum_png: str | None = field(default=None, repr=False)

    #: True when this run's calibration was fitted degenerately (too small a
    #: validation split, or a clamped temperature). Surfaced by the API so a
    #: client can tell the user the probability is not trustworthy.
    calibration_degenerate: bool = False

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict. Used by the API response and the audit log.

        ``streams`` holds pydantic models, which ``dataclasses.asdict`` does
        NOT convert -- it leaves them as objects, and the API then fails on
        subscripting them. They are serialised explicitly.
        """
        from dataclasses import fields

        d = {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith("_")}
        d["verdict"] = self.verdict.value
        d["streams"] = [s.model_dump() for s in self.streams]
        # Everything else is a primitive, list or dict of primitives.
        for k, v in list(d.items()):
            if hasattr(v, "model_dump"):
                d[k] = v.model_dump()
        return d

    @property
    def is_confident(self) -> bool:
        return self.verdict is not Verdict.INCONCLUSIVE and not self.ood_flag
