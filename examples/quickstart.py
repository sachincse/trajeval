"""Quickstart: evaluate an agent trajectory and gate the output.

Run with:  python examples/quickstart.py
No API key needed — every metric here is structural.
"""
from trajeval import (
    CostBudget,
    Gate,
    LoopDetection,
    Step,
    StepEfficiency,
    ToolErrorRate,
    ToolSelectionAccuracy,
    Trajectory,
)

# A trajectory you captured from your agent (or build via trajeval.adapters).
traj = Trajectory(
    goal="Find the weather in Berlin and convert it to Fahrenheit",
    steps=[
        Step(tool="search_weather", args={"city": "Berlin"}, output="12C",
             tokens=120, cost_usd=0.0010, latency_ms=410),
        Step(tool="search_weather", args={"city": "Berlin"}, output="12C",  # oops, repeated
             tokens=120, cost_usd=0.0010, latency_ms=395),
        Step(tool="convert_temp", args={"c": 12}, output="53.6F",
             tokens=80, cost_usd=0.0006, latency_ms=210),
    ],
    final_output="It is 53.6F in Berlin.",
)

gate = Gate([
    ToolSelectionAccuracy(["search_weather", "convert_temp"]),
    StepEfficiency(optimal_steps=2),         # took 3, so this dips
    LoopDetection(max_repeats=1),            # the duplicate call trips this (hard)
    CostBudget(max_cost_usd=0.005),
    ToolErrorRate(max_rate=0.0),
])

result = gate.decide(traj)
print(result)
print(f"\nTotals: {traj.num_steps} steps, "
      f"{traj.total_tokens} tokens, ${traj.total_cost}, "
      f"{traj.total_latency_ms:.0f}ms")
print(f"\n=> {result.decision.value.upper()}")
