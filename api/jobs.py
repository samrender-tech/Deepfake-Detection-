"""F19/F20 - the job store and worker.

Two backends behind one interface:

  memory  an in-process thread pool and dict. Default. Lets the demo run from
          a single ``uvicorn`` command with no Redis, no Postgres, no MinIO --
          which is what makes it usable as a project demo and testable in CI.
  redis   arq queue + a separate worker process, for the containerised deploy.

The interface is identical, so ``main.py`` never branches on which is active.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ddetect.utils.log import get_logger

log = get_logger(__name__)

RETENTION_HOURS = float(os.environ.get("RETENTION_HOURS", "24"))
STAGE_ORDER = ("queued", "probe", "preprocess", "infer", "calibrate", "explain", "persist", "done")


@dataclass
class Job:
    job_id: str
    filename: str
    storage_key: str
    state: str = "queued"
    stage: str = "queued"
    error: str | None = None
    result: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def progress(self) -> float:
        try:
            return STAGE_ORDER.index(self.stage) / (len(STAGE_ORDER) - 1)
        except ValueError:
            return 0.0

    @property
    def expires_at(self) -> datetime:
        return self.created_at + timedelta(hours=RETENTION_HOURS)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["progress"] = self.progress
        return d


class JobStore:
    """Thread-safe in-memory job store with TTL expiry.

    Deliberately not a database in the default configuration: uploads are
    deleted on a retention timer and keeping verdicts about real
    people's faces in a durable store by default would be the wrong default.
    The containerised deploy swaps in Postgres where an audit trail is wanted.
    """

    def __init__(self, backend: Any = None) -> None:
        from api.storage import build_job_backend

        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        # The durable backend, when one is configured. The in-memory dict stays
        # as the hot path and SSE source; the backend is written through so a
        # restart does not lose history.
        self.backend = backend or build_job_backend()

    def create(self, filename: str, storage_key: str) -> Job:
        job = Job(job_id=uuid.uuid4().hex, filename=filename, storage_key=storage_key)
        with self._lock:
            self._jobs[job.job_id] = job
        try:
            self.backend.create(job.to_dict())
        except Exception as e:
            log.warning("job backend write failed: %s", e)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields: Any) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for k, v in fields.items():
                setattr(job, k, v)
            job.updated_at = datetime.now(timezone.utc)
            snapshot = job.to_dict()
        try:
            self.backend.update(job_id, **fields)
        except Exception as e:
            log.warning("job backend update failed: %s", e)

        if fields.get("state") == "done" and job.result:
            from api.drift import get_monitor
            from api.observability import METRICS

            METRICS.observe_result(job.result)
            METRICS.jobs.labels(state="done").inc()
            get_monitor().observe_result(job.result)
            if "total_ms" in (job.result.get("latency_ms") or {}):
                METRICS.job_seconds.observe(job.result["latency_ms"]["total_ms"] / 1000.0)
        elif fields.get("state") == "failed":
            from api.observability import METRICS

            METRICS.jobs.labels(state="failed").inc()

        self._publish(job_id, snapshot)
        return job

    def list_recent(self, limit: int = 50) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)[:limit]

    def delete(self, job_id: str, upload_root: Path | None = None) -> bool:
        """Honour a deletion request: the row AND the stored upload."""
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        if upload_root is not None:
            for p in upload_root.glob(f"{job.storage_key}*"):
                p.unlink(missing_ok=True)
        try:
            self.backend.delete(job_id)
        except Exception as e:
            log.warning("job backend delete failed: %s", e)
        return True

    def purge_expired(self, upload_root: Path | None = None) -> int:
        """Delete jobs past the retention window. Called by a background task."""
        now = datetime.now(timezone.utc)
        with self._lock:
            stale = [j.job_id for j in self._jobs.values() if j.expires_at < now]
        for jid in stale:
            self.delete(jid, upload_root)
        if stale:
            log.info(
                "purged %d expired job(s) past the %gh retention window",
                len(stale),
                RETENTION_HOURS,
            )
        return len(stale)

    # -- SSE ------------------------------------------------------------
    def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        with self._lock:
            self._subscribers.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        with self._lock:
            subs = self._subscribers.get(job_id, [])
            if q in subs:
                subs.remove(q)

    def _publish(self, job_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._subscribers.get(job_id, []))
        for q in subs:
            # A slow client must never block the worker, so a full queue is
            # simply dropped: the polling fallback in the UI catches up.
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(payload)


# ==========================================================================
class LocalWorker:
    """Runs analyses on a bounded thread pool.

    ``max_workers`` is small on purpose: inference is CPU-bound and
    oversubscribing turns a 15-second analysis into a 90-second one for
    everybody. Requests queue instead, and the client sees the queue position
    through the progress stream.
    """

    def __init__(self, store: JobStore, max_workers: int = 2) -> None:
        self.store = store
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ddetect")
        self._detector: Any = None
        self._detector_lock = threading.Lock()
        self._detector_error: str | None = None

    # -- model lifecycle ------------------------------------------------
    def detector(self) -> Any:
        """Lazily load and cache the Detector (one per process)."""
        if self._detector is not None:
            return self._detector
        with self._detector_lock:
            if self._detector is None:
                from ddetect.inference import load_detector

                t0 = time.time()
                self._detector = load_detector()
                log.info(
                    "detector loaded in %.1fs: %s", time.time() - t0, self._detector.model_version
                )
        return self._detector

    def warmup(self) -> bool:
        """Load the model at boot so the first request is not slow.

        Failure is non-fatal: ``/readyz`` reports it and uploads are rejected
        with a clear message, rather than the process refusing to start.
        """
        try:
            self.detector()
            return True
        except Exception as e:
            self._detector_error = f"{type(e).__name__}: {e}"
            log.error("detector failed to load: %s", self._detector_error)
            return False

    @property
    def ready(self) -> bool:
        return self._detector is not None

    @property
    def load_error(self) -> str | None:
        return self._detector_error

    # -- execution ------------------------------------------------------
    def submit(self, job: Job, video_path: Path, artifact_root: Path) -> None:
        self.pool.submit(self._run, job.job_id, video_path, artifact_root)

    def _run(self, job_id: str, video_path: Path, artifact_root: Path) -> None:
        from api.explain_agent import explain_result

        store = self.store
        try:
            store.update(job_id, state="running", stage="probe")
            det = self.detector()

            store.update(job_id, stage="preprocess")
            # Detector.predict covers preprocess -> infer -> calibrate; the
            # stage labels below are coarse because splitting them would mean
            # duplicating its internals here, which contract 5.2 forbids.
            store.update(job_id, stage="infer")
            result = det.predict(video_path, explain=True)

            store.update(job_id, stage="calibrate")

            store.update(job_id, stage="explain")
            payload = det.explain_payload(result)
            text, source = explain_result(payload)
            result.explanation = text
            result.explanation_source = source

            store.update(job_id, stage="persist")
            artifacts = _persist_artifacts(result, job_id, artifact_root)

            store.update(
                job_id,
                state="done",
                stage="done",
                result=result.to_dict(),
                artifacts=artifacts,
            )
            log.info(
                "job %s done: %s p=%.3f", job_id[:8], result.verdict.value, result.calibrated_prob
            )
        except Exception as e:
            log.exception("job %s failed", job_id[:8])
            # The client gets a type and message, never a traceback.
            store.update(job_id, state="failed", stage="done", error=f"{type(e).__name__}: {e}")
        finally:
            # The upload is deleted as soon as it has been analysed; we keep
            # only the derived artefacts (retention policy).
            video_path.unlink(missing_ok=True)


def _persist_artifacts(result: Any, job_id: str, root: Path) -> dict[str, str]:
    """Move the inline base64 images out of the Result onto disk.

    Keeping them in the JSON response would make a single result several MB
    and blow up any client that caches responses.
    """
    import base64

    out: dict[str, str] = {}
    d = root / job_id
    d.mkdir(parents=True, exist_ok=True)

    for key in list(result.gradcam_keys):
        b64 = result._gradcam_png.get(key)
        if not b64:
            continue
        p = d / f"{key}.png"
        p.write_bytes(base64.b64decode(b64))
        out[key] = f"/v1/analyses/{job_id}/artifacts/{key}.png"
    result._gradcam_png.clear()

    if result._spectrum_png:
        p = d / "spectrum.png"
        p.write_bytes(base64.b64decode(result._spectrum_png))
        out["spectrum"] = f"/v1/analyses/{job_id}/artifacts/spectrum.png"
        result._spectrum_png = None
    return out
