import pytest

from trajeval import (
    CostBudget,
    Decision,
    Gate,
    LoopDetection,
    Step,
    StepEfficiency,
    ToolErrorRate,
    ToolSelectionAccuracy,
    Trajectory,
    assert_trajectory,
)
from trajeval import adapters


def make_traj() -> Trajectory:
    return Trajectory(
        goal="Find weather in Berlin and convert to Fahrenheit",
        steps=[
            Step(tool="search_weather", args={"city": "Berlin"}, output="12C",
                 tokens=120, cost_usd=0.001, latency_ms=400),
            Step(tool="convert_temp", args={"c": 12}, output="53.6F",
                 tokens=80, cost_usd=0.0006, latency_ms=200),
        ],
        final_output="It is 53.6F in Berlin.",
    )


def test_tool_selection_perfect():
    r = ToolSelectionAccuracy(["search_weather", "convert_temp"]).evaluate(make_traj())
    assert r.passed and r.score == 1.0


def test_tool_selection_partial():
    r = ToolSelectionAccuracy(["search_weather", "convert_temp", "send_email"]).evaluate(make_traj())
    assert not r.passed
    assert round(r.score, 2) == 0.67


def test_step_efficiency_ok():
    r = StepEfficiency(optimal_steps=2).evaluate(make_traj())
    assert r.passed and r.score == 1.0


def test_loop_detection_flags_repeat():
    traj = Trajectory(steps=[Step(tool="search", args={"q": "x"}) for _ in range(3)])
    r = LoopDetection(max_repeats=1).evaluate(traj)
    assert not r.passed
    assert r.details["max_observed_repeats"] == 3


def test_cost_budget_ok():
    assert CostBudget(max_cost_usd=0.01).evaluate(make_traj()).passed


def test_tool_error_rate_flags_error():
    traj = make_traj()
    traj.steps[0].error = True
    assert not ToolErrorRate(max_rate=0.0).evaluate(traj).passed


def test_gate_ships_clean_run():
    gate = Gate([
        ToolSelectionAccuracy(["search_weather", "convert_temp"]),
        StepEfficiency(optimal_steps=2),
        LoopDetection(max_repeats=1),
        CostBudget(max_cost_usd=0.01),
    ])
    result = gate.decide(make_traj())
    assert result.decision == Decision.SHIP
    assert result.passed


def test_gate_regenerates_on_hard_fail():
    gate = Gate([ToolSelectionAccuracy(["search_weather", "convert_temp", "missing"])])
    assert gate.decide(make_traj()).decision == Decision.REGENERATE


def test_gate_retries_on_soft_fail():
    # optimal 1 vs actual 2 -> score 0.5 < 0.7, soft severity -> retry
    gate = Gate([StepEfficiency(optimal_steps=1, threshold=0.7, severity="soft")])
    assert gate.decide(make_traj()).decision == Decision.RETRY


def test_assert_trajectory_raises_on_failure():
    with pytest.raises(AssertionError):
        assert_trajectory(make_traj(), ToolSelectionAccuracy(["nonexistent"]))


def test_anthropic_adapter():
    messages = [
        {"role": "user", "content": "weather in Berlin?"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "search_weather", "input": {"city": "Berlin"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "12C"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "It is 12C."}]},
    ]
    traj = adapters.from_anthropic(messages, goal="weather")
    assert traj.tool_calls == ["search_weather"]
    assert traj.final_output == "It is 12C."
