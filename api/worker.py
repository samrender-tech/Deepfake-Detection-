"""F20 - the out-of-process worker.

Runs analyses in a separate process from the API, so a 7-second inference
cannot block the event loop and a worker crash cannot take the API down with
it.

    arq api.worker.WorkerSettings

The API enqueues to Redis when ``REDIS_URL`` is set, and falls back to its
in-process thread pool otherwise. Both paths call the same ``run_analysis``,
so there is one implementation of what an analysis *is* -- the split is about
where it runs, never about what it does.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

from api.observability import METRICS, get_log, setup_structured_logging, span

log = get_log(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "")
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT_S", "300"))
MAX_TRIES = int(os.environ.get("JOB_MAX_TRIES", "2"))


# ==========================================================================
def run_analysis(job_id: str, video_path: str) -> dict[str, Any]:
    """Analyse one video. The single implementation, shared by both paths.

    Returns a dict rather than a ``Result`` so it can cross a process boundary
    without the consumer needing to import the model package.
    """
    from api.explain_agent import explain_result
    from ddetect.inference import load_detector

    path = Path(video_path)
    try:
        with span("analysis", job_id=job_id):
            det = load_detector()
            with span("predict"):
                result = det.predict(path, explain=True)
            with span("explain"):
                text, source = explain_result(det.explain_payload(result))
            result.explanation, result.explanation_source = text, source

        payload = result.to_dict()
        artifacts = persist_artifacts(result, job_id)
        METRICS.observe_result(payload)
        METRICS.jobs.labels(state="done").inc()
        log.info(
            "analysis complete",
            job_id=job_id[:8],
            verdict=payload["verdict"],
            prob=round(payload["calibrated_prob"], 3),
        )
        return {"ok": True, "result": payload, "artifacts": artifacts}
    except Exception as e:
        METRICS.jobs.labels(state="failed").inc()
        log.exception("analysis failed", job_id=job_id[:8])
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        # The upload is deleted as soon as it has been analysed,
        # whether or not the analysis succeeded.
        path.unlink(missing_ok=True)


def persist_artifacts(result: Any, job_id: str) -> dict[str, str]:
    """Move the inline base64 images onto the blob store.

    Leaving them in the response makes a single result several MB and breaks
    any client that caches responses.
    """
    import base64

    from api.storage import build_blob_store

    store = build_blob_store()
    out: dict[str, str] = {}

    for key in list(result.gradcam_keys):
        b64 = result._gradcam_png.get(key)
        if not b64:
            continue
        store.put(f"{job_id}/{key}.png", base64.b64decode(b64), "image/png")
        out[key] = f"/v1/analyses/{job_id}/artifacts/{key}.png"
    result._gradcam_png.clear()

    if result._spectrum_png:
        store.put(f"{job_id}/spectrum.png", base64.b64decode(result._spectrum_png), "image/png")
        out["spectrum"] = f"/v1/analyses/{job_id}/artifacts/spectrum.png"
        result._spectrum_png = None
    return out


# ==========================================================================
# arq entry points
# ==========================================================================
async def analyse_job(ctx: dict[str, Any], job_id: str, video_path: str) -> dict[str, Any]:
    """arq task. Runs the blocking analysis in a thread so the worker's event
    loop stays responsive for health checks and cancellation."""
    import asyncio

    from api.storage import build_job_backend

    backend = ctx.get("jobs") or build_job_backend()
    backend.update(job_id, state="running", stage="infer")

    out = await asyncio.get_running_loop().run_in_executor(None, run_analysis, job_id, video_path)

    if out["ok"]:
        backend.update(
            job_id, state="done", stage="done", result=out["result"], artifacts=out["artifacts"]
        )
    else:
        backend.update(job_id, state="failed", stage="done", error=out["error"])
    return out


async def purge_expired(ctx: dict[str, Any]) -> int:
    """Cron task: honour the retention window."""
    from api.storage import build_blob_store, build_job_backend, retention_cutoff

    jobs = ctx.get("jobs") or build_job_backend()
    blobs = ctx.get("blobs") or build_blob_store()
    stale = jobs.expired(retention_cutoff())
    for job_id in stale:
        blobs.delete_prefix(job_id)
        jobs.delete(job_id)
    if stale:
        log.info("retention purge", removed=len(stale))
    return len(stale)


async def startup(ctx: dict[str, Any]) -> None:
    setup_structured_logging()
    from api.storage import build_blob_store, build_job_backend

    ctx["jobs"] = build_job_backend()
    ctx["blobs"] = build_blob_store()

    # Load the model once per worker process, not once per job: a cold load is
    # several seconds and would dominate every request.
    try:
        from ddetect.inference import load_detector

        det = load_detector()
        METRICS.model_loaded.set(1)
        log.info(
            "worker ready", model=det.model_version, jobs=ctx["jobs"].name, blobs=ctx["blobs"].name
        )
    except Exception as e:
        METRICS.model_loaded.set(0)
        log.error("worker started with NO detector loaded", error=str(e))


async def shutdown(_ctx: dict[str, Any]) -> None:
    log.info("worker shutting down")


class WorkerSettings:
    """arq configuration. ``arq api.worker.WorkerSettings``"""

    # arq reads these as class attributes; the list is never mutated.
    functions: ClassVar[list[Any]] = [analyse_job]
    on_startup = startup
    on_shutdown = shutdown
    job_timeout = JOB_TIMEOUT
    max_tries = MAX_TRIES
    # Inference is CPU-bound; more concurrency per worker makes every request
    # slower rather than throughput higher. Scale by adding worker processes.
    max_jobs = int(os.environ.get("WORKER_MAX_JOBS", "2"))
    keep_result = int(os.environ.get("WORKER_KEEP_RESULT_S", "3600"))

    @staticmethod
    def cron_jobs() -> list[Any]:
        try:
            from arq import cron
        except ImportError:
            return []
        return [cron(purge_expired, minute={0, 15, 30, 45}, run_at_startup=False)]

    @staticmethod
    def redis_settings() -> Any:
        from arq.connections import RedisSettings

        return RedisSettings.from_dsn(REDIS_URL or "redis://localhost:6379/0")


def redis_available() -> bool:
    """Whether the API should enqueue rather than use its thread pool."""
    if not REDIS_URL:
        return False
    try:
        import redis

        redis.Redis.from_url(REDIS_URL, socket_connect_timeout=2).ping()
        return True
    except Exception as e:
        log.warning("REDIS_URL set but unreachable; using the in-process pool", error=str(e))
        return False
