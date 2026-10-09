"""F19 - pluggable job and blob storage.

Three backends behind two interfaces, selected by environment:

    memory + local disk   the default. No external service, so the demo runs
                          from one `uvicorn` command.
    postgres + s3         the containerised deploy.

The in-memory default is a deliberate choice, not a stub. This service stores
verdicts about whether a real person's video is fake, and uploads of their
face. Persisting that by default would be the wrong default; the operator
opts in by configuring a database, and the retention policy applies either way.

Selection is by DSN presence (`DATABASE_URL`, `S3_ENDPOINT`), so a deployment
that forgets to configure one degrades to local rather than failing to boot --
and `/readyz` reports which backend is actually live.
"""

from __future__ import annotations

import os
import shutil
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from api.observability import get_log

log = get_log(__name__)

RETENTION_HOURS = float(os.environ.get("RETENTION_HOURS", "24"))


# ==========================================================================
# blobs
# ==========================================================================
class BlobStore(ABC):
    name = "abstract"

    @abstractmethod
    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str: ...

    @abstractmethod
    def get(self, key: str) -> bytes | None: ...

    @abstractmethod
    def delete_prefix(self, prefix: str) -> int: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...


class LocalBlobStore(BlobStore):
    """Files under a root directory. Keys are validated against traversal."""

    name = "local"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if not str(p).startswith(str(self.root.resolve())):
            raise ValueError(f"blob key escapes the store root: {key!r}")
        return p

    def put(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",  # noqa: ARG002 - S3 uses it
    ) -> str:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)  # atomic; a half-written blob must never be readable
        return key

    def get(self, key: str) -> bytes | None:
        p = self._path(key)
        return p.read_bytes() if p.exists() else None

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete_prefix(self, prefix: str) -> int:
        d = self._path(prefix)
        if d.is_dir():
            n = sum(1 for _ in d.rglob("*") if _.is_file())
            shutil.rmtree(d, ignore_errors=True)
            return n
        if d.exists():
            d.unlink()
            return 1
        return 0


class S3BlobStore(BlobStore):
    """S3 / MinIO. Used when ``S3_ENDPOINT`` or ``S3_BUCKET`` is configured."""

    name = "s3"

    def __init__(self, bucket: str, endpoint: str | None = None) -> None:
        import boto3

        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint or None,
            aws_access_key_id=os.environ.get("S3_ACCESS_KEY") or None,
            aws_secret_access_key=os.environ.get("S3_SECRET_KEY") or None,
            region_name=os.environ.get("S3_REGION", "us-east-1"),
        )
        try:
            self.client.head_bucket(Bucket=bucket)
        except Exception:
            try:
                self.client.create_bucket(Bucket=bucket)
                log.info("created bucket", bucket=bucket)
            except Exception as e:
                log.warning("could not create bucket", bucket=bucket, error=str(e))

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        return key

    def get(self, key: str) -> bytes | None:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()  # type: ignore[no-any-return]
        except Exception:
            return None

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def delete_prefix(self, prefix: str) -> int:
        paginator = self.client.get_paginator("list_objects_v2")
        n = 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if objs:
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": objs})
                n += len(objs)
        return n


# ==========================================================================
# jobs
# ==========================================================================
class JobBackend(ABC):
    name = "abstract"

    @abstractmethod
    def create(self, job: dict[str, Any]) -> None: ...

    @abstractmethod
    def get(self, job_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def update(self, job_id: str, **fields: Any) -> dict[str, Any] | None: ...

    @abstractmethod
    def list_recent(self, limit: int) -> list[dict[str, Any]]: ...

    @abstractmethod
    def delete(self, job_id: str) -> bool: ...

    @abstractmethod
    def expired(self, before: datetime) -> list[str]: ...


class MemoryJobBackend(JobBackend):
    name = "memory"

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def create(self, job: dict[str, Any]) -> None:
        with self._lock:
            self._rows[job["job_id"]] = dict(job)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._rows.get(job_id)
            return dict(row) if row else None

    def update(self, job_id: str, **fields: Any) -> dict[str, Any] | None:
        with self._lock:
            row = self._rows.get(job_id)
            if row is None:
                return None
            row.update(fields)
            row["updated_at"] = datetime.now(timezone.utc)
            return dict(row)

    def list_recent(self, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = sorted(self._rows.values(), key=lambda r: r["created_at"], reverse=True)
            return [dict(r) for r in rows[:limit]]

    def delete(self, job_id: str) -> bool:
        with self._lock:
            return self._rows.pop(job_id, None) is not None

    def expired(self, before: datetime) -> list[str]:
        with self._lock:
            return [j for j, r in self._rows.items() if r["created_at"] < before]


class PostgresJobBackend(JobBackend):
    """SQLAlchemy-backed. Used when ``DATABASE_URL`` is set.

    Persisting verdicts is an operator decision with privacy consequences, so
    the schema keeps exactly what an audit needs -- the verdict, the model
    version and the timing -- and nothing that identifies a person.
    """

    name = "postgres"

    def __init__(self, dsn: str) -> None:
        from sqlalchemy import (
            JSON,
            Column,
            DateTime,
            Float,
            String,
            Text,
            create_engine,
        )
        from sqlalchemy.orm import declarative_base, sessionmaker

        self.Base = declarative_base()

        class Job(self.Base):  # type: ignore[misc, valid-type]
            __tablename__ = "analyses"
            job_id = Column(String(64), primary_key=True)
            state = Column(String(16), index=True)
            stage = Column(String(24))
            filename = Column(String(256))
            storage_key = Column(String(128))
            error = Column(Text, nullable=True)
            result = Column(JSON, nullable=True)
            artifacts = Column(JSON, nullable=True)
            created_at = Column(DateTime(timezone=True), index=True)
            updated_at = Column(DateTime(timezone=True))
            # Denormalised for dashboards and the drift monitor, so a query
            # does not have to parse the JSON blob.
            verdict = Column(String(24), nullable=True, index=True)
            calibrated_prob = Column(Float, nullable=True)
            model_version = Column(String(128), nullable=True)

        self.Job = Job
        self.engine = create_engine(dsn, pool_pre_ping=True, future=True)
        self.Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)
        log.info("postgres job backend ready", dsn=dsn.split("@")[-1])

    @staticmethod
    def _to_dict(row: Any) -> dict[str, Any]:
        return {
            "job_id": row.job_id,
            "state": row.state,
            "stage": row.stage,
            "filename": row.filename,
            "storage_key": row.storage_key,
            "error": row.error,
            "result": row.result,
            "artifacts": row.artifacts or {},
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }

    def create(self, job: dict[str, Any]) -> None:
        with self.Session() as s:
            s.add(self.Job(**{k: v for k, v in job.items() if hasattr(self.Job, k)}))
            s.commit()

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.Session() as s:
            row = s.get(self.Job, job_id)
            return self._to_dict(row) if row else None

    def update(self, job_id: str, **fields: Any) -> dict[str, Any] | None:
        with self.Session() as s:
            row = s.get(self.Job, job_id)
            if row is None:
                return None
            for k, v in fields.items():
                if hasattr(row, k):
                    setattr(row, k, v)
            if "result" in fields and isinstance(fields["result"], dict):
                r = fields["result"]
                row.verdict = r.get("verdict")
                row.calibrated_prob = r.get("calibrated_prob")
                row.model_version = r.get("model_version")
            row.updated_at = datetime.now(timezone.utc)
            s.commit()
            return self._to_dict(row)

    def list_recent(self, limit: int) -> list[dict[str, Any]]:
        with self.Session() as s:
            rows = s.query(self.Job).order_by(self.Job.created_at.desc()).limit(limit).all()
            return [self._to_dict(r) for r in rows]

    def delete(self, job_id: str) -> bool:
        with self.Session() as s:
            row = s.get(self.Job, job_id)
            if row is None:
                return False
            s.delete(row)
            s.commit()
            return True

    def expired(self, before: datetime) -> list[str]:
        with self.Session() as s:
            return [r.job_id for r in s.query(self.Job).filter(self.Job.created_at < before).all()]


# ==========================================================================
def build_blob_store() -> BlobStore:
    bucket = os.environ.get("S3_BUCKET")
    endpoint = os.environ.get("S3_ENDPOINT")
    if bucket and (endpoint or os.environ.get("S3_ACCESS_KEY")):
        try:
            return S3BlobStore(bucket, endpoint)
        except Exception as e:
            log.warning("S3 unavailable, falling back to local disk", error=str(e))
    return LocalBlobStore(os.environ.get("DDETECT_ARTIFACT_DIR", "/tmp/ddetect_artifacts"))


def build_job_backend() -> JobBackend:
    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        try:
            return PostgresJobBackend(dsn)
        except Exception as e:
            log.warning("database unavailable, falling back to memory", error=str(e))
    return MemoryJobBackend()


def retention_cutoff() -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=RETENTION_HOURS)
