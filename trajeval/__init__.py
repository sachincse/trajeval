"""trajeval — trajectory evaluation for LLM agents.

Grade *what your agent did* (its tool-call path), not just what it said.
"""
from . import adapters
from .gate import Decision, Gate, GateResult
from .metrics import (
    CostBudget,
    GoalCompletion,
    LoopDetection,
    Metric,
    MetricResult,
    StepEfficiency,
    ToolErrorRate,
    ToolSelectionAccuracy,
)
from .pytest_plugin import assert_trajectory
from .trajectory import Step, Trajectory

__version__ = "0.1.0"

__all__ = [
    "Step",
    "Trajectory",
    "Metric",
    "MetricResult",
    "ToolSelectionAccuracy",
    "StepEfficiency",
    "LoopDetection",
    "CostBudget",
    "ToolErrorRate",
    "GoalCompletion",
    "Gate",
    "GateResult",
    "Decision",
    "assert_trajectory",
    "adapters",
]
