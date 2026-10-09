"""Deterministic response cache for agent tool calls and model replies.

An agent-assisted result must be reproducible. Keying on
``(prompt_version, tool, args_hash)`` means re-running an agent over unchanged
inputs replays the recorded answer instead of asking the model again, so:

* a paper number that an agent helped produce can be regenerated exactly;
* re-running a failed agent costs nothing for the steps that already succeeded;
* an agent's behaviour can be reviewed from the cache without an API key.

The cache is content-addressed on disk, so it survives across sessions and can
be committed alongside a result when a run needs to be auditable.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ddetect.utils.log import get_logger

log = get_logger(__name__)

CACHE_ROOT = Path(os.environ.get("DDETECT_AGENT_CACHE", "audit/agent_cache"))


def _key(prompt_version: str, tool: str, args: Any) -> str:
    blob = json.dumps(
        {"v": prompt_version, "tool": tool, "args": args}, sort_keys=True, default=str
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


class ResponseCache:
    """File-backed cache. ``enabled=False`` turns it into a pass-through."""

    def __init__(self, root: Path | None = None, enabled: bool = True) -> None:
        self.root = Path(root or CACHE_ROOT)
        self.enabled = enabled and os.environ.get("DDETECT_AGENT_CACHE_OFF") != "1"
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        # Two-level fan-out: a flat directory with thousands of entries is slow
        # to list and unpleasant to inspect.
        return self.root / key[:2] / f"{key}.json"

    def get(self, prompt_version: str, tool: str, args: Any) -> Any | None:
        if not self.enabled:
            return None
        p = self._path(_key(prompt_version, tool, args))
        if not p.exists():
            self.misses += 1
            return None
        try:
            self.hits += 1
            return json.loads(p.read_text())["result"]
        except (OSError, json.JSONDecodeError, KeyError):
            # A corrupt entry is a miss, not a crash.
            self.misses += 1
            return None

    def put(self, prompt_version: str, tool: str, args: Any, result: Any) -> None:
        if not self.enabled:
            return
        p = self._path(_key(prompt_version, tool, args))
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "prompt_version": prompt_version,
            "tool": tool,
            "args": args,
            "result": result,
            "cached_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        tmp.replace(p)  # atomic: a half-written entry must never be readable

    def memoize(self, prompt_version: str, tool: str, args: Any, fn: Callable[[], Any]) -> Any:
        """Return the cached value, else call ``fn`` and cache its result."""
        hit = self.get(prompt_version, tool, args)
        if hit is not None:
            return hit
        result = fn()
        if result is not None:
            self.put(prompt_version, tool, args, result)
        return result

    def stats(self) -> str:
        total = self.hits + self.misses
        rate = (self.hits / total * 100) if total else 0.0
        return f"cache {self.hits} hit / {self.misses} miss ({rate:.0f}%)"
