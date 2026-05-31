"""Trajectory metrics.

Structural metrics (tool selection, efficiency, loops, cost, errors) need no
API key and run instantly. ``GoalCompletion`` uses an LLM judge — see
``trajeval.judges``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from .trajectory import Trajectory


@dataclass
class MetricResult:
    name: str
    score: float
    passed: bool
    threshold: float
    severity: str = "hard"  # "hard" -> regenerate on fail, "soft" -> retry
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}: {self.score:.2f} (threshold {self.threshold:.2f})"


class Metric:
    """Base class. Subclasses implement ``evaluate`` and return a score in [0, 1]."""

    name = "metric"

    def __init__(self, threshold: float = 1.0, severity: str = "hard") -> None:
        self.threshold = threshold
        self.severity = severity

    def evaluate(self, traj: Trajectory) -> MetricResult:  # pragma: no cover
        raise NotImplementedError

    def _result(self, score: float, details: Optional[dict] = None) -> MetricResult:
        return MetricResult(
            name=self.name,
            score=score,
            passed=score >= self.threshold,
            threshold=self.threshold,
            severity=self.severity,
            details=details or {},
        )


class ToolSelectionAccuracy(Metric):
    """Did the agent call the tools it was supposed to?"""

    name = "tool_selection_accuracy"

    def __init__(self, expected_tools: Sequence[str], order_matters: bool = False,
                 threshold: float = 1.0, severity: str = "hard") -> None:
        super().__init__(threshold, severity)
        self.expected = list(expected_tools)
        self.order_matters = order_matters

    def evaluate(self, traj: Trajectory) -> MetricResult:
        called = traj.tool_calls
        if not self.expected:
            return self._result(1.0, {"expected": [], "called": called})
        if self.order_matters:
            matched = sum(1 for i, t in enumerate(self.expected)
                          if i < len(called) and called[i] == t)
        else:
            remaining = list(called)
            matched = 0
            for t in self.expected:
                if t in remaining:
                    remaining.remove(t)
                    matched += 1
        score = matched / len(self.expected)
        return self._result(score, {"expected": self.expected, "called": called})


class StepEfficiency(Metric):
    """Penalise agents that take more steps than necessary."""

    name = "step_efficiency"

    def __init__(self, optimal_steps: int, threshold: float = 0.7, severity: str = "soft") -> None:
        super().__init__(threshold, severity)
        self.optimal_steps = max(1, optimal_steps)

    def evaluate(self, traj: Trajectory) -> MetricResult:
        actual = max(1, traj.num_steps)
        score = min(1.0, self.optimal_steps / actual)
        return self._result(score, {"optimal": self.optimal_steps, "actual": traj.num_steps})


class LoopDetection(Metric):
    """Flag repeated identical tool calls (the classic agent failure mode)."""

    name = "loop_detection"

    def __init__(self, max_repeats: int = 1, threshold: float = 1.0, severity: str = "hard") -> None:
        super().__init__(threshold, severity)
        self.max_repeats = max_repeats

    def evaluate(self, traj: Trajectory) -> MetricResult:
        seen: dict = {}
        worst = 0
        offending: Optional[str] = None
        for s in traj.steps:
            if s.tool is None:
                continue
            sig = s.signature()
            seen[sig] = seen.get(sig, 0) + 1
            if seen[sig] > worst:
                worst = seen[sig]
                offending = s.tool
        score = 1.0 if worst <= self.max_repeats else self.max_repeats / worst
        return self._result(score, {"max_observed_repeats": worst, "tool": offending})


class CostBudget(Metric):
    """Did the trajectory stay within a dollar budget?"""

    name = "cost_budget"

    def __init__(self, max_cost_usd: float, severity: str = "soft") -> None:
        super().__init__(threshold=1.0, severity=severity)
        self.max_cost_usd = max_cost_usd

    def evaluate(self, traj: Trajectory) -> MetricResult:
        cost = traj.total_cost
        if cost <= self.max_cost_usd:
            score = 1.0
        else:
            score = self.max_cost_usd / cost if cost else 1.0
        return self._result(score, {"cost_usd": cost, "budget_usd": self.max_cost_usd})


class ToolErrorRate(Metric):
    """Fraction of steps that errored (failed tool calls / retries)."""

    name = "tool_error_rate"

    def __init__(self, max_rate: float = 0.0, severity: str = "hard") -> None:
        super().__init__(threshold=1.0, severity=severity)
        self.max_rate = max_rate

    def evaluate(self, traj: Trajectory) -> MetricResult:
        n = traj.num_steps or 1
        rate = traj.num_errors / n
        score = 1.0 if rate <= self.max_rate else (1.0 - rate)
        return self._result(score, {"error_rate": round(rate, 3), "max_rate": self.max_rate})


class GoalCompletion(Metric):
    """LLM-judged: did the final output actually satisfy the goal?"""

    name = "goal_completion"

    def __init__(self, judge: Optional[Callable] = None, threshold: float = 0.5,
                 severity: str = "hard") -> None:
        super().__init__(threshold, severity)
        self.judge = judge

    def evaluate(self, traj: Trajectory) -> MetricResult:
        if self.judge is None:
            raise ValueError(
                "GoalCompletion needs a `judge` callable. "
                "Try: from trajeval.judges import anthropic_judge"
            )
        score = float(self.judge(traj.goal, traj.final_output, traj.steps))
        return self._result(score, {"goal": traj.goal})
