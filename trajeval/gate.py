"""Decision gate: turn metric results into SHIP / RETRY / REGENERATE.

This is the "missing layer" most eval tools skip — they tell you a score but
not what to *do* with the output. The gate makes that call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from .metrics import Metric, MetricResult
from .trajectory import Trajectory


class Decision(str, Enum):
    SHIP = "ship"
    RETRY = "retry"
    REGENERATE = "regenerate"


@dataclass
class GateResult:
    decision: Decision
    results: list[MetricResult] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.decision == Decision.SHIP

    def __str__(self) -> str:
        lines = [f"Decision: {self.decision.value.upper()}"]
        lines += [f"  {r}" for r in self.results]
        return "\n".join(lines)


class Gate:
    """Evaluate metrics and decide whether the output ships.

    - All metrics pass        -> SHIP
    - Only soft metrics fail   -> RETRY
    - Any hard metric fails    -> REGENERATE
    """

    def __init__(self, metrics: Sequence[Metric]) -> None:
        self.metrics = list(metrics)

    def evaluate(self, traj: Trajectory) -> list[MetricResult]:
        return [m.evaluate(traj) for m in self.metrics]

    def decide(self, traj: Trajectory) -> GateResult:
        results = self.evaluate(traj)
        failed = [r for r in results if not r.passed]
        if not failed:
            return GateResult(Decision.SHIP, results, [])
        reasons = [str(r) for r in failed]
        if any(r.severity == "hard" for r in failed):
            return GateResult(Decision.REGENERATE, results, reasons)
        return GateResult(Decision.RETRY, results, reasons)
