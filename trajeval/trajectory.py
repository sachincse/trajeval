"""Core data model: a Trajectory is an ordered list of agent Steps.

`trajeval` evaluates *what an agent did* — the sequence of tool calls it made
on the way to an answer — rather than only the final text it produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


def _freeze(value: Any) -> Any:
    """Make nested args hashable so two identical tool calls compare equal."""
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


@dataclass
class Step:
    """A single step in an agent's trajectory.

    A step is usually one tool-call cycle (the agent picks a tool, the tool
    runs, an observation comes back). Pure-reasoning steps leave ``tool`` None.
    """

    tool: Optional[str] = None
    args: dict = field(default_factory=dict)
    output: Any = None
    error: bool = False
    tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    thought: Optional[str] = None

    def signature(self) -> tuple:
        """Identity used for loop detection: same tool + same args."""
        return (self.tool, _freeze(self.args))


@dataclass
class Trajectory:
    """An ordered list of steps plus the goal and final answer."""

    steps: list[Step] = field(default_factory=list)
    goal: Optional[str] = None
    final_output: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def tool_calls(self) -> list[str]:
        return [s.tool for s in self.steps if s.tool is not None]

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens for s in self.steps)

    @property
    def total_cost(self) -> float:
        return round(sum(s.cost_usd for s in self.steps), 6)

    @property
    def total_latency_ms(self) -> float:
        return sum(s.latency_ms for s in self.steps)

    @property
    def num_errors(self) -> int:
        return sum(1 for s in self.steps if s.error)
