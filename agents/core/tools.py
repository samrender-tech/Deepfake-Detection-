"""Typed tool registry.

Every agent gets an explicit allowlist of tools. The registry is the place
where the per-agent permissions stop being prose:

* A5 (paper) has no training tool, which is *why* it is allowed to read test
  results -- it cannot act on what it sees.
* A3 is the only agent with a launch tool, and the only one with a GPU budget.
* A8 has no generative tool and no network tool, so the no-generation scope
  commitment holds by construction rather than by intention.

Each call is budget-charged, firewall-checked, cached and audited in one place,
so an agent author cannot forget one of those steps.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agents.core.audit import record
from agents.core.budget import Budget
from agents.core.cache import ResponseCache
from agents.core.firewall import ReadRequest, check_read
from ddetect.utils.log import get_logger

log = get_logger(__name__)


class ToolDenied(PermissionError):
    """The agent asked for a tool it is not permitted to use."""


@dataclass
class Tool:
    name: str
    fn: Callable[..., Any]
    doc: str
    #: True when the result depends only on its arguments, so it is cacheable.
    pure: bool = True
    #: True when the tool reads dataset splits or run predictions, in which
    #: case the firewall inspects the request first.
    reads_data: bool = False
    #: True when the tool writes outside proposals/ and needs human approval.
    writes: bool = False


@dataclass
class ToolRegistry:
    """The set of tools one agent may call."""

    agent: str
    budget: Budget
    cache: ResponseCache = field(default_factory=ResponseCache)
    prompt_version: str = "v1"
    tools: dict[str, Tool] = field(default_factory=dict)
    #: Set by the caller when a human has approved repo writes for this run.
    approved_writes: bool = False

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def add(self, name: str, fn: Callable[..., Any], doc: str, **kw: bool) -> None:
        self.register(Tool(name=name, fn=fn, doc=doc, **kw))

    def describe(self) -> str:
        return "\n".join(f"  {t.name:24s} {t.doc}" for t in self.tools.values())

    # ------------------------------------------------------------------
    def call(self, name: str, **kwargs: Any) -> Any:
        """Invoke a tool with every guard applied, in order."""
        tool = self.tools.get(name)
        if tool is None:
            msg = f"{self.agent} may not use {name!r}. Available: {sorted(self.tools)}"
            record(self.agent, "tool_denied", inputs={"tool": name}, ok=False, note=msg)
            raise ToolDenied(msg)

        # 1. budget (before anything expensive happens)
        self.budget.spend_call()

        # 2. firewall for anything that reads data
        if tool.reads_data:
            check_read(
                ReadRequest(
                    agent=self.agent,
                    path=str(kwargs.get("path") or kwargs.get("manifest") or name),
                    split=kwargs.get("split"),
                    dataset=kwargs.get("dataset"),
                )
            )

        # 3. human approval for repo writes
        if tool.writes and not self.approved_writes:
            msg = (
                f"{self.agent} tried to write via {name!r} without approval. "
                f"Agents write freely only to proposals/."
            )
            record(self.agent, "write_denied", inputs={"tool": name}, ok=False, note=msg)
            raise ToolDenied(msg)

        # 4. cache, then call
        def run() -> Any:
            return tool.fn(**kwargs)

        result = self.cache.memoize(self.prompt_version, name, kwargs, run) if tool.pure else run()

        record(
            self.agent,
            f"tool:{name}",
            prompt_version=self.prompt_version,
            inputs=kwargs,
            output=result,
            ok=True,
        )
        return result
