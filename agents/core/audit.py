"""Append-only audit log for every agent action.

Two reasons this is not optional:

1. A7's output is user-facing prose about whether a real person's video is
   fake. Every such statement must be reconstructable from its input payload,
   so a disputed verdict can be examined (the responsible-use policy, "Auditability").
2. If an agent ever influences a paper number, the log is the evidence of what
   it saw and when -- which is what makes the disclosure in section 9.10
   checkable rather than a promise.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
AUDIT_PATH = Path(os.environ.get("DDETECT_AUDIT_LOG", "audit/agent_actions.jsonl"))


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def record(
    agent: str,
    action: str,
    *,
    prompt_version: str = "",
    inputs: Any = None,
    output: Any = None,
    tokens: int | None = None,
    ok: bool = True,
    note: str = "",
    path: Path | None = None,
) -> dict[str, Any]:
    """Append one action. Returns the record written.

    Inputs and outputs are stored as hashes plus a truncated preview rather
    than in full: the payloads contain per-frame scores about an identifiable
    person's video, and the audit log is not the right place to accumulate
    that. The hash is enough to prove what was sent.
    """
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agent": agent,
        "action": action,
        "prompt_version": prompt_version,
        "inputs_hash": _hash(inputs) if inputs is not None else None,
        "output_hash": _hash(output) if output is not None else None,
        "output_preview": (str(output)[:280] if output is not None else None),
        "tokens": tokens,
        "ok": ok,
        "note": note,
    }
    p = path or AUDIT_PATH
    with _LOCK:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
    return rec


def read_all(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or AUDIT_PATH
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
