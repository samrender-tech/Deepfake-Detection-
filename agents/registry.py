"""The agent roster, and what each is permitted to see.

Importing this is the cheapest way to answer "which agents exist and what can
they read", which is the question the agent policy document and the paper's
disclosure subsection both have to answer.
"""

from __future__ import annotations

from typing import Any

from agents.a1_literature import LiteratureAgent
from agents.a2_data import DataCurationAgent
from agents.a3_experiment import ExperimentAgent
from agents.a4_failure import FailureDiagnosisAgent
from agents.a5_paper import PaperAgent
from agents.a6_defence import DefenceAgent
from agents.a8_redteam import RedTeamAgent

AGENTS: dict[str, type] = {
    "a1_literature": LiteratureAgent,
    "a2_data": DataCurationAgent,
    "a3_experiment": ExperimentAgent,
    "a4_failure": FailureDiagnosisAgent,
    "a5_paper": PaperAgent,
    "a6_defence": DefenceAgent,
    "a8_redteam": RedTeamAgent,
}

#: A7 ships inside the serving path rather than as a batch agent, so it lives
#: in api/explain_agent.py and is not constructed here.
SHIPS_IN_PRODUCT = ("a7_explainer",)

#: Documented reading permissions, mirroring agents/core/firewall.py.
#: A5 and A6 may read test results because the config is frozen by then AND
#: neither has a training tool -- they can describe, not act.
PERMISSIONS: dict[str, dict[str, Any]] = {
    "a1_literature": {"reads": ["paper/refs.bib", "arxiv (network)"], "test_results": False},
    "a2_data": {"reads": ["manifests", "video files"], "test_results": False},
    "a3_experiment": {"reads": ["source train/val only"], "test_results": False, "gpu_hours": 40.0},
    "a4_failure": {"reads": ["source val predictions only"], "test_results": False},
    "a5_paper": {
        "reads": ["results/", "paper/"],
        "test_results": True,
        "note": "no training tool, so it cannot act on what it reads",
    },
    "a6_defence": {
        "reads": ["results/", "docs/"],
        "test_results": True,
        "note": "no training tool",
    },
    "a7_explainer": {"reads": ["one job's explainability payload"], "test_results": False},
    "a8_redteam": {
        "reads": ["source val only", "real media"],
        "test_results": False,
        "note": "no generative tool, no network",
    },
}


def build(name: str, **kw: Any) -> Any:
    if name not in AGENTS:
        raise KeyError(f"unknown agent {name!r}; known: {sorted(AGENTS)}")
    return AGENTS[name](**kw)


def describe() -> str:
    lines = ["agent          test results  notes"]
    for name in sorted({*AGENTS, *SHIPS_IN_PRODUCT}):
        p = PERMISSIONS.get(name, {})
        lines.append(
            f"  {name:14s} {'yes' if p.get('test_results') else 'NO':>12s}  "
            f"{p.get('note', '') or ', '.join(p.get('reads', []))}"
        )
    return "\n".join(lines)
