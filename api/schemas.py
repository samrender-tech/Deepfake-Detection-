"""API request/response models (pydantic v2).

The response is deliberately shaped so a client CANNOT render a bare
real/fake binary: ``verdict`` is a three-state enum and ``disclaimer`` is a
required field (responsible-use policy).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

JobState = Literal["queued", "running", "done", "failed", "cancelled"]
Stage = Literal["queued", "probe", "preprocess", "infer", "calibrate", "explain", "persist", "done"]

#: Shown with every verdict. Not optional, and not configurable away.
DISCLAIMER = (
    "This is an automated, probabilistic estimate from a research prototype. "
    "Accuracy drops substantially on manipulation methods the model has not "
    "seen. It is not evidence, and must not be used for legal, forensic, "
    "journalistic, employment or immigration decisions."
)


class AnalysisCreated(BaseModel):
    job_id: str
    state: JobState
    poll: str = Field(description="GET this for the result")
    events: str = Field(description="SSE stream of stage progress")


class StreamScoreOut(BaseModel):
    name: Literal["visual", "audio", "sync"]
    score: float | None = None
    available: bool
    reliability: float
    note: str = ""


class SuspiciousSegment(BaseModel):
    """A contiguous run of frames scoring at or above the decision threshold."""

    start_s: float
    end_s: float
    peak_score: float = Field(ge=0.0, le=1.0)
    peak_time_s: float
    n_frames: int = Field(ge=1)


class AnalysisResult(BaseModel):
    # --- verdict -----------------------------------------------------
    verdict: Literal["likely_authentic", "likely_manipulated", "inconclusive"]
    calibrated_probability: float = Field(ge=0.0, le=1.0)
    confidence_interval: list[float] = Field(
        description="[lo, hi] of the conformal abstention band on the probability"
    )
    decision_threshold: float
    abstained: bool

    # --- honesty flags -----------------------------------------------
    ood_flag: bool = Field(
        description="True when the input is unlike the model's training distribution"
    )
    calibration_reliable: bool = Field(
        description="False when this model's calibration was fitted degenerately"
    )
    warnings: list[str] = []
    disclaimer: str = DISCLAIMER

    # --- evidence -----------------------------------------------------
    streams: list[StreamScoreOut] = []
    frame_scores: list[float] = []
    frame_timestamps: list[float] = []
    sync_curve: list[float] = []
    suspicious_segments: list[SuspiciousSegment] = Field(
        default=[],
        description="Where to look: time ranges whose per-frame score crosses the "
        "decision threshold. Frame scores are uncalibrated; this is not a second verdict.",
    )
    gate_weights: dict[str, float] = {}
    reliability: dict[str, float] = {}
    has_audio: bool
    n_faces_found: int
    gradcam_urls: dict[str, str] = {}
    spectrum_url: str | None = None

    # --- narrative (A7; may be absent, never blocking) ---------------
    explanation: str | None = None
    explanation_source: Literal["agent", "template", "none"] = "none"

    # --- provenance ----------------------------------------------------
    model_version: str
    latency_ms: dict[str, float] = {}


class AnalysisStatus(BaseModel):
    job_id: str
    state: JobState
    stage: Stage
    progress: float = Field(ge=0.0, le=1.0)
    created_at: datetime
    updated_at: datetime
    filename: str | None = None
    error: str | None = None
    result: AnalysisResult | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    model_loaded: bool
    model_version: str | None = None
    device: str | None = None
    queue: str
    calibration_reliable: bool | None = None


class ModelInfo(BaseModel):
    """What the loaded model is and how its verdicts are produced.

    Published so a client can show *why* a probability lands in the
    inconclusive band, and so nobody has to guess which calibration a
    deployment is running.
    """

    model_version: str
    architecture: str
    device: str
    trained_on: str = Field(description="manifest the checkpoint was trained on")
    n_frames: int
    image_size: int
    audio_seconds: float
    calibration_method: str
    temperature: float
    decision_threshold: float
    threshold_criterion: str
    conformal_band: list[float]
    conformal_alpha: float
    calibration_fitted_on: str
    calibration_n_val: int
    calibration_reliable: bool
    ood_detection: bool = Field(description="True when an OOD threshold was fitted")
    disclaimer: str = DISCLAIMER


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
