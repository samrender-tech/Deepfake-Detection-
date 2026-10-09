"""F28 - structured logging, metrics and tracing.

Everything here degrades to a no-op when its library is absent, because the
single-command demo (`uvicorn api.main:app`) must keep working with only the
core dependencies installed. Install the ``serve`` extra to light it up.

The metrics chosen are the ones that would actually tell you something is
wrong with THIS service rather than generic HTTP counters:

* ``abstention_rate`` and ``ood_rate`` -- if these move, the input distribution
  has shifted away from what the model was calibrated on.
* ``score_histogram`` -- the shape of the live score distribution, which is
  what the PSI drift monitor compares against the validation baseline.
* ``stage_seconds`` -- preprocess vs inference vs explain, so a regression is
  attributable rather than just "slower".
* ``verdict_total`` by verdict -- a sudden swing toward "manipulated" is
  either an attack or a broken checkpoint, and both need to be visible.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

# ==========================================================================
# structured logging
# ==========================================================================
try:
    import structlog

    _HAS_STRUCTLOG = True
except ImportError:  # pragma: no cover - optional
    structlog = None  # type: ignore[assignment]
    _HAS_STRUCTLOG = False


def setup_structured_logging(level: str = "INFO", json_logs: bool | None = None) -> None:
    """Configure structlog with a JSON renderer in production.

    JSON by default when not attached to a TTY: a log aggregator cannot parse
    the pretty console format, and a human reading a terminal cannot parse JSON.
    """
    if not _HAS_STRUCTLOG:
        logging.basicConfig(level=level.upper())
        return

    use_json = json_logs if json_logs is not None else not os.isatty(2)
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer()
        if use_json
        else structlog.dev.ConsoleRenderer(colors=True)
    )
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=__import__("sys").stderr),
        cache_logger_on_first_use=True,
    )


class _KwLogger:
    """stdlib logging adapter that accepts structlog-style keyword fields.

    Without this, every ``log.info("x", job_id=...)`` call site would have to
    branch on whether structlog is installed.
    """

    def __init__(self, name: str) -> None:
        self._log = logging.getLogger(name)

    def _fmt(self, msg: str, kw: dict[str, Any]) -> str:
        return f"{msg} " + " ".join(f"{k}={v!r}" for k, v in kw.items()) if kw else msg

    def debug(self, msg: str, **kw: Any) -> None:
        self._log.debug(self._fmt(msg, kw))

    def info(self, msg: str, **kw: Any) -> None:
        self._log.info(self._fmt(msg, kw))

    def warning(self, msg: str, **kw: Any) -> None:
        self._log.warning(self._fmt(msg, kw))

    def error(self, msg: str, **kw: Any) -> None:
        self._log.error(self._fmt(msg, kw))

    def exception(self, msg: str, **kw: Any) -> None:
        self._log.exception(self._fmt(msg, kw))


def get_log(name: str) -> Any:
    if _HAS_STRUCTLOG:
        return structlog.get_logger(name)
    return _KwLogger(name)


def bind_request(**kw: Any) -> None:
    """Attach fields to every log line for the rest of this request."""
    if _HAS_STRUCTLOG:
        structlog.contextvars.bind_contextvars(**kw)


def clear_request() -> None:
    if _HAS_STRUCTLOG:
        structlog.contextvars.clear_contextvars()


# ==========================================================================
# metrics
# ==========================================================================
try:
    from prometheus_client import CONTENT_TYPE_LATEST as _CTL
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _HAS_PROM = True
    CONTENT_TYPE_LATEST = _CTL
except ImportError:  # pragma: no cover - optional
    _HAS_PROM = False
    CONTENT_TYPE_LATEST = "text/plain"


class _NoopMetric:
    """Stand-in so call sites need no branching."""

    def labels(self, *_a: Any, **_k: Any) -> _NoopMetric:
        return self

    def inc(self, *a: Any, **k: Any) -> None: ...
    def observe(self, *a: Any, **k: Any) -> None: ...
    def set(self, *a: Any, **k: Any) -> None: ...


class Metrics:
    """The service's metric surface. Safe to use whether or not Prometheus
    is installed."""

    def __init__(self) -> None:
        self.enabled = _HAS_PROM
        if not _HAS_PROM:
            noop = _NoopMetric()
            for n in (
                "requests",
                "request_seconds",
                "jobs",
                "job_seconds",
                "stage_seconds",
                "verdicts",
                "abstentions",
                "ood_flags",
                "scores",
                "queue_depth",
                "model_loaded",
                "upload_bytes",
                "upload_rejects",
            ):
                setattr(self, n, noop)
            self.registry = None
            return

        self.registry = CollectorRegistry()
        r = self.registry

        self.requests = Counter(
            "avforge_requests_total", "HTTP requests", ["method", "path", "status"], registry=r
        )
        self.request_seconds = Histogram(
            "avforge_request_seconds", "HTTP request latency", ["method", "path"], registry=r
        )
        self.jobs = Counter(
            "avforge_jobs_total", "Analysis jobs by terminal state", ["state"], registry=r
        )
        self.job_seconds = Histogram(
            "avforge_job_seconds",
            "End-to-end analysis latency",
            buckets=(1, 2, 5, 10, 15, 20, 30, 45, 60, 120),
            registry=r,
        )
        self.stage_seconds = Histogram(
            "avforge_stage_seconds",
            "Per-stage latency",
            ["stage"],
            buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 40),
            registry=r,
        )

        # The model-behaviour metrics. These are the ones worth alerting on.
        self.verdicts = Counter(
            "avforge_verdicts_total", "Verdicts served", ["verdict"], registry=r
        )
        self.abstentions = Counter("avforge_abstentions_total", "Inconclusive verdicts", registry=r)
        self.ood_flags = Counter(
            "avforge_ood_flags_total", "Inputs flagged unlike training data", registry=r
        )
        self.scores = Histogram(
            "avforge_calibrated_probability",
            "Calibrated probability served",
            buckets=[i / 10 for i in range(11)],
            registry=r,
        )

        self.queue_depth = Gauge("avforge_queue_depth", "Jobs queued or running", registry=r)
        self.model_loaded = Gauge("avforge_model_loaded", "1 when a detector is loaded", registry=r)
        self.upload_bytes = Histogram(
            "avforge_upload_bytes",
            "Accepted upload size",
            buckets=(1e5, 1e6, 5e6, 1e7, 2.5e7, 5e7, 1e8),
            registry=r,
        )
        self.upload_rejects = Counter(
            "avforge_upload_rejects_total", "Rejected uploads", ["reason"], registry=r
        )

    def render(self) -> bytes:
        if not self.enabled or self.registry is None:
            return b"# prometheus_client not installed; install the 'serve' extra\n"
        return generate_latest(self.registry)

    def observe_result(self, result: dict[str, Any]) -> None:
        """Record everything a served verdict tells us about model behaviour."""
        verdict = str(result.get("verdict", "unknown"))
        self.verdicts.labels(verdict=verdict).inc()
        if verdict == "inconclusive":
            self.abstentions.inc()
        if result.get("ood_flag"):
            self.ood_flags.inc()
        p = result.get("calibrated_prob")
        if isinstance(p, (int, float)):
            self.scores.observe(float(p))
        for stage, ms in (result.get("latency_ms") or {}).items():
            if stage.endswith("_ms") and isinstance(ms, (int, float)):
                self.stage_seconds.labels(stage=stage[:-3]).observe(ms / 1000.0)


METRICS = Metrics()


@contextmanager
def timed(stage: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        METRICS.stage_seconds.labels(stage=stage).observe(time.perf_counter() - t0)


# ==========================================================================
# tracing
# ==========================================================================
def setup_tracing(app: Any = None, service_name: str = "avforge") -> bool:
    """Enable OpenTelemetry when configured. Returns whether it was set up.

    Requires ``OTEL_EXPORTER_OTLP_ENDPOINT``; without it tracing stays off
    rather than exporting to a default that does not exist.
    """
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        get_log(__name__).warning(
            "OTEL_EXPORTER_OTLP_ENDPOINT is set but opentelemetry is not "
            "installed; tracing disabled"
        )
        return False

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)

    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
        except ImportError:
            pass
    return True


def tracer() -> Any:
    try:
        from opentelemetry import trace

        return trace.get_tracer("avforge")
    except ImportError:
        return None


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[None]:
    t = tracer()
    if t is None:
        yield
        return
    with t.start_as_current_span(name) as s:
        for k, v in attrs.items():
            s.set_attribute(k, v)
        yield
