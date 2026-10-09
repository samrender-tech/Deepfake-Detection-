"""Logging + run provenance.

Two jobs: readable console output, and the provenance stamp that makes a run
reproducible (git sha, env lock, config hash). Section 11.3's acceptance test is
"a clean clone reproduces a headline table", which only works if every run
records exactly what produced it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

console = Console(stderr=True)
_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def git_sha(short: bool = False) -> str:
    """Current commit, with ``-dirty`` appended when the tree is modified.

    ``make eval-final`` refuses to run on a dirty tree: a test-set
    number stamped with a dirty sha is not reproducible, so it is not a result.
    """
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD"]
        sha = subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "--ignore-submodules"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def is_tree_dirty() -> bool:
    return git_sha().endswith("-dirty")


def config_hash(cfg: Any) -> str:
    """Stable 12-char hash of a config, for joining runs in the aggregator."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            cfg = OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    blob = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def env_lock() -> str:
    """``pip freeze`` equivalent, written to every run dir."""
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze", "--exclude-editable"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "# pip freeze unavailable\n"


def stamp_run(run_dir: Path, cfg: Any) -> dict[str, str]:
    """Write the RUN_ARTEFACTS provenance files. Returns the stamp."""
    from omegaconf import OmegaConf

    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = {
        "git_sha": git_sha(),
        "config_hash": config_hash(cfg),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
    }
    (run_dir / "git_sha").write_text(stamp["git_sha"] + "\n")
    (run_dir / "env.lock").write_text(env_lock())
    if OmegaConf.is_config(cfg):
        OmegaConf.save(cfg, run_dir / "config.yaml")
    else:
        (run_dir / "config.yaml").write_text(json.dumps(cfg, indent=2, default=str))
    (run_dir / "stamp.json").write_text(json.dumps(stamp, indent=2))
    return stamp


class JsonlWriter:
    """Append-only metrics log. One json object per line, per step/epoch."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **record: Any) -> None:
        record.setdefault("ts", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        with self.path.open("a") as fh:
            fh.write(json.dumps(record, default=float) + "\n")
