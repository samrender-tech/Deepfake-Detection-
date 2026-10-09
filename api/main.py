"""F21 - the FastAPI application.

Runs standalone with no external services:

    DDETECT_RUN_DIR=runs/smoke_test/seed0 uvicorn api.main:app --reload

Design notes that are policy, not taste:

* Every result carries a ``disclaimer`` and a three-state ``verdict``. There is
  no response shape that lets a client render a bare real/fake binary.
* The uploaded file is deleted as soon as it has been analysed, and jobs expire
  on a retention timer. ``DELETE /v1/analyses/{id}`` works and actually removes
  the stored bytes.
* Errors never leak a traceback.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from api.drift import get_monitor
from api.jobs import JobStore, LocalWorker
from api.observability import (
    CONTENT_TYPE_LATEST,
    METRICS,
    setup_structured_logging,
    setup_tracing,
)
from api.schemas import (
    DISCLAIMER,
    AnalysisCreated,
    AnalysisResult,
    AnalysisStatus,
    HealthResponse,
    JobState,
    ModelInfo,
    StreamScoreOut,
    SuspiciousSegment,
)
from api.security import (
    MAX_DURATION_S,
    MAX_UPLOAD_MB,
    UploadRejected,
    new_storage_key,
    probe_and_validate,
    safe_display_name,
    storage_path,
    stream_to_disk,
    verify_ffmpeg_available,
)
from api.storage import build_blob_store
from ddetect import __version__
from ddetect.report import render_html, segments_for
from ddetect.utils.log import get_logger, setup_logging

log = get_logger(__name__)

UPLOAD_ROOT = Path(os.environ.get("DDETECT_UPLOAD_DIR", "/tmp/ddetect_uploads"))
ARTIFACT_ROOT = Path(os.environ.get("DDETECT_ARTIFACT_DIR", "/tmp/ddetect_artifacts"))
API_KEYS = {k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip()}
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")
    if o.strip()
]

store = JobStore()
worker = LocalWorker(store, max_workers=int(os.environ.get("WORKER_THREADS", "2")))
blobs = build_blob_store()

# Rate limiting. Analysis is CPU-bound and takes seconds, so an unthrottled
# client can trivially exhaust the worker pool for everyone else.
RATE_LIMIT_ANALYSES = os.environ.get("RATE_LIMIT_ANALYSES", "10/minute")
RATE_LIMIT_DEFAULT = os.environ.get("RATE_LIMIT_DEFAULT", "120/minute")
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    limiter: Any = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT_DEFAULT])
    _HAS_LIMITER = True
except ImportError:  # pragma: no cover - optional
    limiter = None
    _HAS_LIMITER = False


# ==========================================================================
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    setup_structured_logging(os.environ.get("LOG_LEVEL", "INFO"))
    if setup_tracing(_app):
        log.info("OpenTelemetry tracing enabled")
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    ok, ver = verify_ffmpeg_available()
    log.info("ffmpeg: %s", ver if ok else f"UNAVAILABLE ({ver})")
    log.info("storage: jobs=%s blobs=%s queue=%s", store.backend.name, blobs.name, _queue_kind())
    get_monitor()  # build the drift baseline at boot, not on first request

    # Warm up the model so the first request is not slow. Non-fatal on failure:
    # /readyz reports it instead of the process refusing to boot.
    await asyncio.get_running_loop().run_in_executor(None, worker.warmup)
    METRICS.model_loaded.set(1 if worker.ready else 0)

    async def purge_loop() -> None:
        while True:
            await asyncio.sleep(900)
            try:
                store.purge_expired(UPLOAD_ROOT)
            except Exception:
                log.exception("retention purge failed")

    task = asyncio.create_task(purge_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(
    title="AVFORGE deepfake detection",
    version=__version__,
    description=(
        "Research prototype. Reports a three-state verdict with a calibrated "
        "probability and abstains when it is not reliable enough to call. " + DISCLAIMER
    ),
    lifespan=lifespan,
)

if _HAS_LIMITER:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    # slowapi reads the client address off the request, so it needs the
    # decorated endpoint to accept `request` -- create_analysis does.
    log.info("rate limiting: %s on analyses, %s default", RATE_LIMIT_ANALYSES, RATE_LIMIT_DEFAULT)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next: Any) -> Response:
    """Request id + timing, and a guarantee that no traceback escapes."""
    import uuid

    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    t0 = time.time()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled error on %s %s [rid=%s]", request.method, request.url.path, rid)
        return JSONResponse(
            status_code=500,
            content={"error": "internal error", "detail": None, "request_id": rid},
        )
    response.headers["x-request-id"] = rid
    response.headers["x-response-time-ms"] = f"{(time.time() - t0) * 1000:.0f}"
    return response


def _queue_kind() -> str:
    from api.worker import redis_available

    return "redis+arq" if redis_available() else "local-threadpool"


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    """API-key auth. Open when ``API_KEYS`` is unset, for local development."""
    if not API_KEYS:
        return
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


# ==========================================================================
# health
# ==========================================================================
@app.get("/healthz", response_model=HealthResponse, tags=["ops"])
def healthz() -> HealthResponse:
    """Liveness. Always 200 if the process is up."""
    return HealthResponse(
        status="ok",
        version=__version__,
        model_loaded=worker.ready,
        queue="local-threadpool",
    )


@app.get("/readyz", response_model=HealthResponse, tags=["ops"])
def readyz(response: Response) -> HealthResponse:
    """Readiness: can we actually serve an analysis?"""
    det = worker._detector
    ready = worker.ready
    if not ready:
        response.status_code = 503
    return HealthResponse(
        status="ok" if ready else "degraded",
        version=__version__,
        model_loaded=ready,
        model_version=getattr(det, "model_version", None),
        device=str(getattr(det, "device", "")) or None,
        queue=_queue_kind(),
        calibration_reliable=(not det.cal.degenerate) if det is not None else None,
    )


@app.get("/metrics", include_in_schema=False, tags=["ops"])
def metrics() -> Response:
    """Prometheus scrape endpoint.

    Deliberately not behind the API key: a scraper is infrastructure, and the
    metrics carry no per-user data -- only aggregate counters and histograms.
    """
    METRICS.queue_depth.set(
        sum(1 for j in store.list_recent(200) if j.state in ("queued", "running"))
    )
    METRICS.model_loaded.set(1 if worker.ready else 0)
    return Response(content=METRICS.render(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/drift", tags=["ops"], dependencies=[Depends(require_key)])
def drift() -> dict[str, Any]:
    """Is live traffic still the kind of video this model was calibrated on?

    A detector whose cross-dataset accuracy is already its headline weakness
    degrades silently when the input distribution moves.
    """
    return get_monitor().report()


@app.get("/v1/model", response_model=ModelInfo, tags=["ops"])
def model_info() -> ModelInfo:
    """The loaded checkpoint, its input geometry, and its calibration."""
    det = worker._detector if worker.ready else None
    if det is None:
        raise HTTPException(status_code=503, detail="no detector is loaded")
    cal, tcfg = det.cal, det.train_cfg
    return ModelInfo(
        model_version=det.model_version,
        architecture=str(tcfg.get("model", "unknown")),
        device=str(det.device),
        trained_on=Path(str(tcfg.get("manifest", "unknown"))).name,
        n_frames=det.data_cfg.n_frames,
        image_size=det.data_cfg.image_size,
        audio_seconds=det.data_cfg.audio_seconds,
        calibration_method=cal.method,
        temperature=float(cal.temperature),
        decision_threshold=float(cal.threshold),
        threshold_criterion=cal.threshold_criterion,
        conformal_band=[float(cal.conformal_lo), float(cal.conformal_hi)],
        conformal_alpha=float(cal.conformal_alpha),
        calibration_fitted_on=cal.fitted_on,
        calibration_n_val=int(cal.n_val),
        calibration_reliable=not cal.degenerate,
        ood_detection=math.isfinite(cal.ood_threshold),
    )


@app.get("/v1/limits", tags=["ops"])
def limits() -> dict[str, Any]:
    return {
        "max_upload_mb": MAX_UPLOAD_MB,
        "max_duration_s": MAX_DURATION_S,
        "accepted_containers": ["mp4", "mov", "webm", "mkv", "avi", "flv"],
        "retention_hours": float(os.environ.get("RETENTION_HOURS", "24")),
        "disclaimer": DISCLAIMER,
    }


# ==========================================================================
# analyses
# ==========================================================================
def _rate_limited(spec: str) -> Any:
    """Apply a slowapi limit when available, else a no-op decorator."""

    def deco(fn: Any) -> Any:
        return limiter.limit(spec)(fn) if _HAS_LIMITER else fn

    return deco


@app.post(
    "/v1/analyses",
    response_model=AnalysisCreated,
    status_code=202,
    tags=["analysis"],
    dependencies=[Depends(require_key)],
)
@_rate_limited(RATE_LIMIT_ANALYSES)
async def create_analysis(
    request: Request,  # noqa: ARG001 - slowapi reads the client address off it
    file: UploadFile,
) -> AnalysisCreated:
    """Accept a video and queue it. Returns immediately with a job id."""
    if not worker.ready and not worker.warmup():
        raise HTTPException(
            status_code=503,
            detail=(
                "No detector is loaded. Set DDETECT_RUN_DIR to a trained run "
                f"directory and restart. ({worker.load_error})"
            ),
        )

    key = new_storage_key()
    dest = UPLOAD_ROOT / f"{key}.bin"
    try:
        size = await stream_to_disk(file, dest, MAX_UPLOAD_MB)
        info = probe_and_validate(dest, size)
    except UploadRejected as e:
        dest.unlink(missing_ok=True)
        # Categorise the rejection so the metric says WHY uploads are failing.
        msg = str(e).lower()
        reason = (
            "too_large"
            if "exceeds" in msg
            else "too_long"
            if "limit is" in msg
            else "bad_container"
            if "container" in msg
            else "undecodable"
            if "decode" in msg
            else "other"
        )
        METRICS.upload_rejects.labels(reason=reason).inc()
        raise HTTPException(status_code=400, detail=str(e)) from e

    METRICS.upload_bytes.observe(size)
    get_monitor().observe_upload(info.width, info.height, info.duration_s, info.vcodec)

    job = store.create(filename=safe_display_name(file.filename), storage_key=key)
    log.info(
        "job %s queued: %s %.1fs %dx%d %s audio=%s",
        job.job_id[:8],
        job.filename,
        info.duration_s,
        info.width,
        info.height,
        info.vcodec,
        info.has_audio,
    )
    worker.submit(job, dest, ARTIFACT_ROOT)
    return AnalysisCreated(
        job_id=job.job_id,
        state=cast("JobState", job.state),
        poll=f"/v1/analyses/{job.job_id}",
        events=f"/v1/analyses/{job.job_id}/events",
    )


def _to_status(job: Any) -> AnalysisStatus:
    result = None
    if job.result:
        result = _to_result(job.result, job.artifacts)
    return AnalysisStatus(
        job_id=job.job_id,
        state=job.state,
        stage=job.stage,
        progress=job.progress,
        created_at=job.created_at,
        updated_at=job.updated_at,
        filename=job.filename,
        error=job.error,
        result=result,
    )


def _to_result(d: dict[str, Any], artifacts: dict[str, str]) -> AnalysisResult:
    """Map the internal ``Result`` onto the public response.

    Field names are deliberately different from the internal ones: the public
    contract should not shift because an internal dataclass was refactored.
    """
    return AnalysisResult(
        verdict=d["verdict"],
        calibrated_probability=round(float(d["calibrated_prob"]), 4),
        confidence_interval=[float(d["conformal_lo"]), float(d["conformal_hi"])],
        decision_threshold=float(d["threshold"]),
        abstained=d["verdict"] == "inconclusive",
        ood_flag=bool(d["ood_flag"]),
        calibration_reliable=not bool(d.get("calibration_degenerate", False)),
        warnings=list(d.get("warnings", [])),
        # Construct the response model explicitly rather than passing dicts:
        # pydantic would coerce them anyway, but an extra or misspelt key would
        # be silently dropped instead of raising.
        streams=[StreamScoreOut(**s) for s in d.get("streams", [])],
        frame_scores=list(d.get("frame_scores", [])),
        frame_timestamps=list(d.get("frame_timestamps", [])),
        sync_curve=list(d.get("sync_curve", [])),
        suspicious_segments=[SuspiciousSegment(**seg) for seg in segments_for(d)],
        gate_weights=dict(d.get("gate_weights", {})),
        reliability=dict(d.get("reliability", {})),
        has_audio=bool(d["has_audio"]),
        n_faces_found=int(d["n_faces_found"]),
        gradcam_urls={k: v for k, v in artifacts.items() if k != "spectrum"},
        spectrum_url=artifacts.get("spectrum"),
        explanation=d.get("explanation"),
        explanation_source=d.get("explanation_source", "none"),
        model_version=d.get("model_version", ""),
        latency_ms=dict(d.get("latency_ms", {})),
    )


@app.get(
    "/v1/analyses/{job_id}",
    response_model=AnalysisStatus,
    tags=["analysis"],
    dependencies=[Depends(require_key)],
)
def get_analysis(job_id: str) -> AnalysisStatus:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown or expired job")
    return _to_status(job)


@app.get("/v1/analyses/{job_id}/events", tags=["analysis"])
async def analysis_events(job_id: str, request: Request) -> StreamingResponse:
    """SSE stage progress.

    Not key-protected: the job id is the capability, and EventSource in the
    browser cannot set custom headers.
    """
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown or expired job")

    async def gen() -> AsyncIterator[str]:
        q = store.subscribe(job_id)
        try:
            cur = store.get(job_id)
            if cur:
                yield f"data: {json.dumps({'state': cur.state, 'stage': cur.stage, 'progress': cur.progress})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"  # keeps proxies from closing the stream
                    continue
                yield f"data: {json.dumps({'state': payload['state'], 'stage': payload['stage'], 'progress': payload['progress']}, default=str)}\n\n"
                if payload["state"] in ("done", "failed", "cancelled"):
                    break
        finally:
            store.unsubscribe(job_id, q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/v1/analyses/{job_id}/artifacts/{name}", tags=["analysis"])
def get_artifact(job_id: str, name: str) -> FileResponse:
    """Serve a Grad-CAM overlay or the spectrum plot."""
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown or expired job")
    try:
        d = storage_path(ARTIFACT_ROOT, job_id)
    except UploadRejected as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    import re

    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}\.png", name):
        raise HTTPException(status_code=400, detail="invalid artifact name")
    p = (d / name).resolve()
    if not str(p).startswith(str(d.resolve())) or not p.exists():
        raise HTTPException(status_code=404, detail="no such artifact")
    return FileResponse(p, media_type="image/png")


@app.get(
    "/v1/analyses/{job_id}/report",
    tags=["analysis"],
    dependencies=[Depends(require_key)],
    response_model=None,
)
def get_report(job_id: str, format: str = "html") -> Response:
    """A downloadable copy of one finished analysis.

    ``format=html`` is a single self-contained file (heat maps inlined, no
    external requests) that opens offline and prints; ``format=json`` is the
    public result plus provenance. Both carry the disclaimer.
    """
    if format not in ("html", "json"):
        raise HTTPException(status_code=400, detail="format must be 'html' or 'json'")
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown or expired job")
    if job.state != "done" or not job.result:
        raise HTTPException(status_code=409, detail=f"analysis is {job.state}, not done")

    stem = Path(job.filename or job_id).stem or job_id
    safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:64] or "analysis"
    disposition = f'attachment; filename="avforge-report-{safe_stem}.{format}"'

    if format == "json":
        body = {
            "job_id": job.job_id,
            "filename": job.filename,
            "created_at": job.created_at.isoformat(),
            "service_version": __version__,
            "result": _to_result(job.result, job.artifacts).model_dump(),
        }
        return JSONResponse(body, headers={"Content-Disposition": disposition})

    images: dict[str, bytes] = {}
    art_dir = ARTIFACT_ROOT / job_id
    for key in job.artifacts:
        p = art_dir / f"{key}.png"
        if p.is_file():
            images[key] = p.read_bytes()
    page = render_html(job.result, filename=job.filename, images=images)
    return HTMLResponse(page, headers={"Content-Disposition": disposition})


@app.delete(
    "/v1/analyses/{job_id}", status_code=204, tags=["analysis"], dependencies=[Depends(require_key)]
)
def delete_analysis(job_id: str) -> Response:
    """Delete a job and everything stored for it.

    The retention promise must be backed by a working delete, not a
    documented intention.
    """
    import shutil

    if not store.delete(job_id, UPLOAD_ROOT):
        raise HTTPException(status_code=404, detail="unknown or expired job")
    shutil.rmtree(ARTIFACT_ROOT / job_id, ignore_errors=True)
    return Response(status_code=204)


@app.get("/v1/analyses", tags=["analysis"], dependencies=[Depends(require_key)])
def list_analyses(limit: int = 20) -> list[AnalysisStatus]:
    return [_to_status(j) for j in store.list_recent(min(limit, 100))]


# ==========================================================================
# the built frontend, when present
# ==========================================================================
_WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
if _WEB_DIST.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_WEB_DIST, html=True), name="web")
    log.info("serving the built frontend from %s", _WEB_DIST)
else:

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "service": "avforge",
            "version": __version__,
            "docs": "/docs",
            "note": "frontend not built; run `make web` or use /docs",
        }
