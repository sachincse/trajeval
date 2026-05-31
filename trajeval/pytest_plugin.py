"""Pytest-friendly assertion helper.

    from trajeval import assert_trajectory, ToolSelectionAccuracy

    def test_my_agent():
        traj = run_my_agent("What is the weather in Berlin?")
        assert_trajectory(traj, ToolSelectionAccuracy(["search_weather"]))
"""
from __future__ import annotations

from .metrics import Metric
from .trajectory import Trajectory


def assert_trajectory(traj: Trajectory, *metrics: Metric) -> None:
    """Assert every metric passes for the trajectory.

    Raises ``AssertionError`` listing each failing metric with its details.
    """
    failures = []
    for metric in metrics:
        result = metric.evaluate(traj)
        if not result.passed:
            failures.append(f"{result}  details={result.details}")
    if failures:
        raise AssertionError("Trajectory failed:\n  " + "\n  ".join(failures))
