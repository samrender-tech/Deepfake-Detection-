"""Per-agent token, call and compute budgets with hard stops.

An agent that silently burns the project's GPU allocation or runs up an API
bill is a worse failure than one that refuses to finish, so every budget is a
hard stop rather than a warning. ``A3`` in particular schedules training runs,
and section 8.4 estimates the whole grid at ~198 GPU-hours against roughly 90
free Kaggle hours a week -- there is no slack to absorb a runaway loop.

Budgets are process-local and reset per run. They are not a security boundary
(an agent could in principle not call ``spend``); they are a correctness
boundary, enforced by routing every tool call through ``Budget.guard``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from agents.core.audit import record
from ddetect.utils.log import get_logger

log = get_logger(__name__)


class BudgetExceeded(RuntimeError):
    """An agent hit a hard limit and must stop."""


@dataclass
class Budget:
    """Limits for one agent run.

    ``gpu_hours`` is only meaningful for A3; the others leave it at 0 and any
    attempt to spend it raises, which is the intended behaviour -- an agent
    that was never meant to launch training should fail loudly if it tries.
    """

    agent: str
    max_tokens: int = 200_000
    max_calls: int = 50
    max_seconds: float = 900.0
    max_gpu_hours: float = 0.0

    tokens: int = field(default=0, init=False)
    calls: int = field(default=0, init=False)
    gpu_hours: float = field(default=0.0, init=False)
    started: float = field(default_factory=time.monotonic, init=False)

    # ---- queries -------------------------------------------------------
    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def exhausted(self) -> bool:
        return (
            self.tokens >= self.max_tokens
            or self.calls >= self.max_calls
            or self.elapsed >= self.max_seconds
            or self.gpu_hours >= self.max_gpu_hours > 0
        )

    def remaining(self) -> dict[str, float]:
        return {
            "tokens": max(self.max_tokens - self.tokens, 0),
            "calls": max(self.max_calls - self.calls, 0),
            "seconds": max(self.max_seconds - self.elapsed, 0.0),
            "gpu_hours": max(self.max_gpu_hours - self.gpu_hours, 0.0),
        }

    # ---- spending ------------------------------------------------------
    def _fail(self, what: str, used: float, cap: float) -> None:
        msg = (
            f"{self.agent}: {what} budget exhausted ({used:.1f} of {cap:.1f}). "
            f"Stopping. Raise the budget deliberately if this run genuinely "
            f"needs more."
        )
        record(self.agent, "budget_exceeded", ok=False, note=msg)
        log.error(msg)
        raise BudgetExceeded(msg)

    def guard(self) -> None:
        """Call before every tool invocation."""
        if self.elapsed >= self.max_seconds:
            self._fail("wall-clock", self.elapsed, self.max_seconds)
        if self.calls >= self.max_calls:
            self._fail("call", self.calls, self.max_calls)
        if self.tokens > self.max_tokens:
            self._fail("token", self.tokens, self.max_tokens)

    def spend_call(self, n: int = 1) -> None:
        """Charge a tool call. Guards BEFORE incrementing, so an agent gets
        exactly ``max_calls`` usable calls rather than one fewer."""
        self.guard()
        self.calls += n

    def spend_tokens(self, n: int) -> None:
        """Charge tokens. Fails only once the cap is actually exceeded, so a
        run that lands exactly on its allocation still completes."""
        self.tokens += n
        if self.tokens > self.max_tokens:
            self._fail("token", self.tokens, self.max_tokens)

    def spend_gpu_hours(self, hours: float) -> None:
        """Charge compute. Raises if the agent has no GPU allocation at all."""
        if self.max_gpu_hours <= 0:
            msg = (
                f"{self.agent} has no GPU-hour allocation but tried to spend "
                f"{hours:.1f}h. Only the experiment orchestrator may launch "
                f"training."
            )
            record(self.agent, "gpu_denied", ok=False, note=msg)
            raise BudgetExceeded(msg)
        if self.gpu_hours + hours > self.max_gpu_hours:
            self._fail("GPU-hour", self.gpu_hours + hours, self.max_gpu_hours)
        self.gpu_hours += hours

    def summary(self) -> str:
        return (
            f"{self.agent}: {self.calls}/{self.max_calls} calls, "
            f"{self.tokens}/{self.max_tokens} tokens, "
            f"{self.elapsed:.0f}/{self.max_seconds:.0f}s"
            + (
                f", {self.gpu_hours:.1f}/{self.max_gpu_hours:.1f} GPU-h"
                if self.max_gpu_hours
                else ""
            )
        )


#: Default allocations. A3 is the only agent with compute, and A7 is tight
#: because it runs per user request in the serving path.
DEFAULTS: dict[str, dict[str, float]] = {
    "a1_literature": {"max_tokens": 300_000, "max_calls": 80, "max_seconds": 1800},
    "a2_data": {"max_tokens": 150_000, "max_calls": 40, "max_seconds": 1800},
    "a3_experiment": {
        "max_tokens": 200_000,
        "max_calls": 60,
        "max_seconds": 3600,
        "max_gpu_hours": 40.0,
    },
    "a4_failure": {"max_tokens": 200_000, "max_calls": 40, "max_seconds": 1200},
    "a5_paper": {"max_tokens": 400_000, "max_calls": 60, "max_seconds": 1800},
    "a6_defence": {"max_tokens": 150_000, "max_calls": 30, "max_seconds": 900},
    "a7_explainer": {"max_tokens": 4_000, "max_calls": 2, "max_seconds": 30},
    "a8_redteam": {"max_tokens": 150_000, "max_calls": 40, "max_seconds": 1200},
}


def budget_for(agent: str, **overrides: float) -> Budget:
    cfg = {**DEFAULTS.get(agent, {}), **overrides}
    return Budget(agent=agent, **cfg)  # type: ignore[arg-type]
