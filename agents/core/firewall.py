"""The test-set firewall. The agentic layer's hard constraint.

The project's entire claim is an honest cross-dataset number. An agent that
reads target-domain test results and then proposes a model change has performed
manual test-set optimisation by proxy, and the headline number becomes
worthless -- while every log still looks like diligent research.

So reads are mediated. A3 (experiment orchestration) and A4 (failure
diagnosis) may see ONLY train/val splits of the source dataset. A5 (paper) may
read test results, because by then the configuration is frozen and A5 has no
training tools at all.

This is enforced in code and asserted by ``tests/test_agents.py``. A policy
that lives only in a document is not a control.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

from agents.core.audit import record
from ddetect.utils.log import get_logger

log = get_logger(__name__)

#: Agents permitted to read test-split predictions and target-domain results.
TEST_READERS = frozenset({"a5_paper", "a6_defence", "human"})

#: Splits every other agent is confined to.
ALLOWED_SPLITS = frozenset({"train", "val"})


class FirewallViolation(PermissionError):
    """An agent attempted to read data its role forbids."""


@dataclass(frozen=True)
class ReadRequest:
    agent: str
    path: str | Path
    split: str | None = None
    dataset: str | None = None


def _looks_like_test_artifact(path: str | Path) -> bool:
    name = Path(path).name.lower()
    return "test" in name or name.startswith("preds_test")


def check_read(req: ReadRequest, source_dataset: str | None = None) -> None:
    """Raise ``FirewallViolation`` unless this read is permitted.

    Every denial is written to the audit log, so an attempt is visible even
    though it was blocked.
    """
    if req.agent in TEST_READERS:
        return

    reasons: list[str] = []
    if req.split is not None and req.split not in ALLOWED_SPLITS:
        reasons.append(f"split={req.split!r} is not in {sorted(ALLOWED_SPLITS)}")
    if _looks_like_test_artifact(req.path):
        reasons.append(f"path {Path(req.path).name!r} looks like a test artefact")
    if source_dataset is not None and req.dataset is not None and req.dataset != source_dataset:
        reasons.append(f"dataset={req.dataset!r} is a target domain (source is {source_dataset!r})")

    if reasons:
        msg = (
            f"firewall: agent {req.agent!r} may not read {req.path}: "
            + "; ".join(reasons)
            + ". Target test sets are evaluated once per frozen config by a "
            "human via `make eval-final`."
        )
        record("firewall", "deny", inputs=req.__dict__, ok=False, note=msg)
        log.error(msg)
        raise FirewallViolation(msg)


def guarded_read_manifest(agent: str, manifest: str | Path, split: str, **kw: Any) -> pd.DataFrame:
    """``read_manifest`` with the firewall applied."""
    from ddetect.data.manifest_io import read_manifest

    check_read(ReadRequest(agent=agent, path=manifest, split=split))
    return read_manifest(manifest, split=split, **kw)


def guarded_read_preds(agent: str, path: str | Path) -> pd.DataFrame:
    """``load_preds`` with the firewall applied."""
    from ddetect.metrics import load_preds

    check_read(ReadRequest(agent=agent, path=path))
    return load_preds(path)


def eval_final_allowed() -> tuple[bool, str]:
    """Gate for the one sanctioned target-test evaluation.

    Refuses on a dirty working tree: a test-set number stamped with a
    ``-dirty`` sha cannot be reproduced, so it is not a result.
    """
    from ddetect.utils.log import git_sha, is_tree_dirty

    if os.environ.get("DDETECT_ALLOW_DIRTY_EVAL") == "1":
        return True, f"override accepted (git {git_sha()})"
    if is_tree_dirty():
        return False, (
            "the working tree has uncommitted changes. A target-test number "
            "must be reproducible from a commit. Commit first, or set "
            "DDETECT_ALLOW_DIRTY_EVAL=1 for a throwaway run that must NOT be "
            "reported."
        )
    return True, f"clean tree at {git_sha()}"
