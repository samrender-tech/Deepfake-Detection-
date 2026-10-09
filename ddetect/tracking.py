"""F25 - experiment tracking.

Optional by design. The run directory (``runs/<exp>/seed<N>/``) is already the
source of truth: it holds the config, git sha, environment lock, metrics and
predictions, and ``experiments/aggregate.py`` reads only that. A tracking
server is a nicer way to *look* at runs, never where a paper number comes
from.

That ordering matters. If MLflow were authoritative, a result would depend on
a server being up, and "reproduce this from a clean clone" would stop being
true. So tracking failures are logged and swallowed -- a tracking outage must
never kill a training run that has been going for six hours.

Backends: MLflow (local file store by default, so it needs no server), or
Weights & Biases. Neither is required.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ddetect.utils.log import get_logger

log = get_logger(__name__)


class Tracker:
    """No-op base. Every method must be safe to call unconditionally."""

    name = "none"
    enabled = False

    def start(
        self, run_name: str, config: dict[str, Any], tags: dict[str, str] | None = None
    ) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...
    def log_artifact(self, path: str | Path, kind: str = "") -> None: ...
    def log_summary(self, summary: dict[str, Any]) -> None: ...
    def finish(self, status: str = "FINISHED") -> None: ...


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Tracking backends reject nested params, so flatten with dotted keys."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        elif isinstance(v, (list, tuple)):
            out[key] = ",".join(map(str, v))
        elif v is None or isinstance(v, (str, int, float, bool)):
            out[key] = v
        else:
            out[key] = str(v)
    return out


class MLflowTracker(Tracker):
    name = "mlflow"

    def __init__(self, tracking_uri: str | None = None, experiment: str = "avforge") -> None:
        import mlflow

        self.mlflow = mlflow
        # Default to a local file store: a tracking server is an operational
        # dependency we do not want training to require.
        mlflow.set_tracking_uri(
            tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
        )
        mlflow.set_experiment(experiment)
        self.enabled = True

    def start(
        self, run_name: str, config: dict[str, Any], tags: dict[str, str] | None = None
    ) -> None:
        try:
            self.mlflow.start_run(run_name=run_name, tags=tags or {})
            params = _flatten(config)
            # MLflow caps params per call; chunk rather than lose the tail.
            items = list(params.items())
            for i in range(0, len(items), 90):
                self.mlflow.log_params(dict(items[i : i + 90]))
        except Exception as e:
            log.warning("mlflow start failed (%s); continuing untracked", e)
            self.enabled = False

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self.enabled:
            return
        try:
            clean = {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and v == v  # drop NaN
            }
            if clean:
                self.mlflow.log_metrics(clean, step=step)
        except Exception as e:
            log.debug("mlflow log_metrics failed: %s", e)

    def log_artifact(self, path: str | Path, kind: str = "") -> None:
        if not self.enabled or not Path(path).exists():
            return
        try:
            self.mlflow.log_artifact(str(path), artifact_path=kind or None)
        except Exception as e:
            log.debug("mlflow log_artifact failed: %s", e)

    def log_summary(self, summary: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self.mlflow.log_params(
                {f"summary.{k}": str(v)[:250] for k, v in _flatten(summary).items()}
            )
        except Exception as e:
            log.debug("mlflow log_summary failed: %s", e)

    def finish(self, status: str = "FINISHED") -> None:
        if not self.enabled:
            return
        try:
            self.mlflow.end_run(status=status)
        except Exception as e:
            log.debug("mlflow end_run failed: %s", e)


class WandbTracker(Tracker):
    name = "wandb"

    def __init__(self, project: str = "avforge") -> None:
        import wandb

        self.wandb = wandb
        self.project = project
        self.enabled = True
        self.run: Any = None

    def start(
        self, run_name: str, config: dict[str, Any], tags: dict[str, str] | None = None
    ) -> None:
        try:
            self.run = self.wandb.init(
                project=self.project,
                name=run_name,
                config=_flatten(config),
                tags=list((tags or {}).values()),
                reinit=True,
            )
        except Exception as e:
            log.warning("wandb init failed (%s); continuing untracked", e)
            self.enabled = False

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self.enabled and self.run:
            try:
                self.wandb.log(metrics, step=step)
            except Exception as e:
                log.debug("wandb log failed: %s", e)

    def log_artifact(
        self,
        path: str | Path,
        kind: str = "",  # noqa: ARG002 - mlflow uses it
    ) -> None:
        if self.enabled and self.run and Path(path).exists():
            try:
                self.wandb.save(str(path))
            except Exception as e:
                log.debug("wandb save failed: %s", e)

    def log_summary(self, summary: dict[str, Any]) -> None:
        if self.enabled and self.run:
            try:
                self.run.summary.update(_flatten(summary))
            except Exception as e:
                log.debug("wandb summary failed: %s", e)

    def finish(self, status: str = "FINISHED") -> None:
        if self.enabled and self.run:
            try:
                self.wandb.finish(exit_code=0 if status == "FINISHED" else 1)
            except Exception as e:
                log.debug("wandb finish failed: %s", e)


def build_tracker(backend: str | None = None, experiment: str = "avforge") -> Tracker:
    """Select a tracker. ``none`` (the default) is a working configuration."""
    backend = (backend or os.environ.get("DDETECT_TRACKER", "none")).lower()
    if backend in ("", "none", "off"):
        return Tracker()
    try:
        if backend == "mlflow":
            return MLflowTracker(experiment=experiment)
        if backend == "wandb":
            return WandbTracker(project=experiment)
    except ImportError:
        log.warning(
            "tracker %r requested but not installed; continuing untracked "
            "(pip install -e '.[track]')",
            backend,
        )
        return Tracker()
    log.warning("unknown tracker %r; continuing untracked", backend)
    return Tracker()
