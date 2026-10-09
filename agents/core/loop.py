"""Shared agent scaffolding.

Every agent here is built the same way, and the shape matters:

    deterministic analysis (Python)  ->  optional LLM synthesis  ->  proposal

The analysis is the valuable part and it runs with no API key: A2 really does
probe every file, A4 really does cluster the failures, A5 really does check
each claim against the results. The model is a narrative layer on top. That
split means the agents are useful offline, testable without mocking a model,
and -- most importantly -- their *conclusions* are reproducible even when their
prose is not.

An agent never writes outside ``proposals/`` without explicit approval, and
``AgentResult`` is deliberately plain data so a human can read what was
concluded before deciding to act on it.
"""

from __future__ import annotations

import json
import os
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.core.audit import record
from agents.core.budget import Budget, budget_for
from agents.core.cache import ResponseCache
from agents.core.tools import ToolRegistry
from ddetect.utils.log import get_logger

log = get_logger(__name__)

PROPOSALS = Path(os.environ.get("DDETECT_PROPOSALS", "proposals"))
PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


@dataclass
class Finding:
    """One concrete, checkable conclusion."""

    severity: str  # info | warn | blocker
    title: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)
    suggested_action: str = ""


@dataclass
class AgentResult:
    agent: str
    ok: bool = True
    summary: str = ""
    findings: list[Finding] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    narrative: str | None = None
    narrative_source: str = "none"  # agent | template | none
    budget: str = ""

    @property
    def blockers(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "blocker"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        lines = [f"# {self.agent}", "", self.summary, ""]
        if self.narrative:
            lines += [self.narrative, ""]
        for sev in ("blocker", "warn", "info"):
            group = [f for f in self.findings if f.severity == sev]
            if not group:
                continue
            lines.append(f"## {sev}s ({len(group)})")
            for f in group:
                lines.append(f"\n### {f.title}")
                lines.append(textwrap.fill(f.detail, 78))
                if f.suggested_action:
                    lines.append(f"\n**Action.** {f.suggested_action}")
                if f.evidence:
                    lines.append(
                        "\n```json\n"
                        + json.dumps(f.evidence, indent=2, default=str)[:1500]
                        + "\n```"
                    )
            lines.append("")
        if self.budget:
            lines += ["---", f"_{self.budget}_"]
        return "\n".join(lines)


class Agent:
    """Base class. Subclasses implement :meth:`analyse`."""

    name: str = "agent"
    prompt_version: str = "v1"
    #: Set True for agents whose conclusions must never be acted on
    #: automatically (A8's red-team recipes, A3's launches).
    needs_human_review: bool = True

    def __init__(
        self,
        budget: Budget | None = None,
        cache: ResponseCache | None = None,
        approved_writes: bool = False,
    ) -> None:
        self.budget = budget or budget_for(self.name)
        self.registry = ToolRegistry(
            agent=self.name,
            budget=self.budget,
            cache=cache or ResponseCache(),
            prompt_version=f"{self.name}/{self.prompt_version}",
            approved_writes=approved_writes,
        )
        self.register_tools()

    # ---- subclass contract ---------------------------------------------
    def register_tools(self) -> None:
        """Declare the tools this agent may call. Default: none."""

    def analyse(self, **kwargs: Any) -> AgentResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def narrative_prompt(self, result: AgentResult) -> str | None:  # noqa: ARG002
        """Prompt for the optional LLM synthesis. None skips it.

        Subclasses use ``result``; the base implementation deliberately does
        not, so an agent with no narrative layer needs no override.
        """
        return None

    # ---- shared --------------------------------------------------------
    def run(self, **kwargs: Any) -> AgentResult:
        log.info("%s: starting", self.name)
        try:
            result = self.analyse(**kwargs)
        except Exception as e:
            log.exception("%s failed", self.name)
            record(self.name, "failed", ok=False, note=f"{type(e).__name__}: {e}")
            return AgentResult(
                agent=self.name,
                ok=False,
                summary=f"{self.name} failed: {type(e).__name__}: {e}",
            )

        prompt = self.narrative_prompt(result)
        if prompt:
            text, source = call_model(
                prompt, self.budget, self.name, f"{self.name}/{self.prompt_version}"
            )
            result.narrative, result.narrative_source = text, source

        result.budget = self.budget.summary()
        record(
            self.name,
            "completed",
            prompt_version=self.prompt_version,
            output=result.summary,
            ok=result.ok,
            note=f"{len(result.findings)} findings",
        )
        log.info("%s: %s", self.name, result.summary)
        return result

    def write_proposal(self, result: AgentResult, slug: str | None = None) -> Path:
        """Write the result to ``proposals/`` -- the only place agents may write."""
        PROPOSALS.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        p = PROPOSALS / f"{stamp}-{slug or self.name}.md"
        p.write_text(result.render())
        log.info("%s: wrote %s", self.name, p)
        return p


# ==========================================================================
def load_prompt(agent: str, version: str = "v1") -> str | None:
    p = PROMPTS / agent / f"{version}.md"
    return p.read_text() if p.exists() else None


def call_model(
    prompt: str,
    budget: Budget,
    agent: str,
    prompt_version: str,
    max_tokens: int = 1200,
    timeout: float = 60.0,
) -> tuple[str | None, str]:
    """One model call, budget-charged and audited. Returns (text, source).

    Returns ``(None, "none")`` whenever the model is unavailable, so every
    caller must already work without it. No agent blocks on the LLM.
    """
    if os.environ.get("DDETECT_DISABLE_AGENT") == "1":
        return None, "none"
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None, "none"
    try:
        from anthropic import Anthropic
    except ImportError:
        log.debug("anthropic SDK not installed; %s runs analysis-only", agent)
        return None, "none"

    try:
        budget.guard()
        client = Anthropic(api_key=key, timeout=timeout)
        resp = client.messages.create(
            model=os.environ.get("AGENT_MODEL", "claude-opus-5-5"),
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        used = resp.usage.input_tokens + resp.usage.output_tokens
        budget.spend_call()
        budget.spend_tokens(used)
        record(
            agent,
            "model_call",
            prompt_version=prompt_version,
            inputs={"chars": len(prompt)},
            output=text,
            tokens=used,
            ok=True,
        )
        return text.strip(), "agent"
    except Exception as e:
        log.warning("%s: model call failed (%s); continuing analysis-only", agent, type(e).__name__)
        record(agent, "model_call_failed", ok=False, note=str(e)[:200])
        return None, "none"
