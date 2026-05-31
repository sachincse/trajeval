# trajeval

**Grade what your agent _did_ — not just what it said.**

[![PyPI](https://img.shields.io/badge/pypi-trajeval-blue)](https://pypi.org/project/trajeval/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)

LLM eval tools (DeepEval, Ragas, Langfuse…) score the **final answer**. But agents
fail in the **middle** — they pick the wrong tool, loop, burn budget on a bad path,
or pass a final-output check while the workflow is quietly broken.

`trajeval` scores the **trajectory**: the sequence of tool calls your agent made.
It's lightweight, local-first, framework-agnostic, and needs **no API key** for its
structural metrics.

```bash
pip install trajeval
```

## Why trajeval

> "A single intermediate mistake can pass a final-output check while still corrupting the full workflow."

| | Final-output evals (DeepEval/Ragas) | Observability (Langfuse) | **trajeval** |
|---|:---:|:---:|:---:|
| Scores the final answer | ✅ | ➖ | ✅ |
| Scores the **tool-call path** | ❌ | ➖ | ✅ |
| Tool-selection accuracy | ❌ | ❌ | ✅ |
| Step-efficiency / loop detection | ❌ | ❌ | ✅ |
| Per-path cost & latency budget | ❌ | ➖ | ✅ |
| **Ship / retry / regenerate gate** | ❌ | ❌ | ✅ |
| Zero-dependency, local, CI-ready | ➖ | ❌ | ✅ |

## Quickstart

```python
from trajeval import Trajectory, Step, Gate
from trajeval import ToolSelectionAccuracy, StepEfficiency, LoopDetection, CostBudget

traj = Trajectory(
    goal="Find the weather in Berlin and convert to Fahrenheit",
    steps=[
        Step(tool="search_weather", args={"city": "Berlin"}, output="12C", cost_usd=0.001),
        Step(tool="convert_temp",   args={"c": 12},          output="53.6F", cost_usd=0.0006),
    ],
    final_output="It is 53.6F in Berlin.",
)

gate = Gate([
    ToolSelectionAccuracy(["search_weather", "convert_temp"]),
    StepEfficiency(optimal_steps=2),
    LoopDetection(max_repeats=1),
    CostBudget(max_cost_usd=0.01),
])

result = gate.decide(traj)
print(result.decision)   # Decision.SHIP | RETRY | REGENERATE
print(result)            # per-metric PASS/FAIL breakdown
```

## The decision gate

Most eval tools give you a number and stop. `trajeval` answers the question you
actually have in production — **what do I do with this output?**

- **All metrics pass** → `SHIP`
- **Only _soft_ metrics fail** (e.g. slightly inefficient) → `RETRY`
- **Any _hard_ metric fails** (wrong tool, infinite loop) → `REGENERATE`

## Use it in CI / pytest

```python
from trajeval import assert_trajectory, ToolSelectionAccuracy, LoopDetection

def test_booking_agent_path():
    traj = run_agent("Book a table for 2 at 7pm")
    assert_trajectory(
        traj,
        ToolSelectionAccuracy(["search_restaurants", "create_booking"], order_matters=True),
        LoopDetection(max_repeats=1),
    )
```

## Bring your own framework

Convert any run into a `Trajectory` once, then every metric works:

```python
from trajeval import adapters

traj = adapters.from_anthropic(messages)   # Anthropic Messages API
traj = adapters.from_openai(messages)      # OpenAI tool_calls
traj = adapters.from_steps(my_dicts)       # raw dicts / your own loop
```

## Metrics

| Metric | What it catches | Needs LLM? |
|---|---|:---:|
| `ToolSelectionAccuracy` | Wrong / missing tools | No |
| `StepEfficiency` | Bloated, wandering paths | No |
| `LoopDetection` | Repeated identical calls | No |
| `CostBudget` | Paths that overspend | No |
| `ToolErrorRate` | Flaky / failing tool calls | No |
| `GoalCompletion` | Final state vs the goal | Yes (judge) |

LLM-judged goal completion:

```python
from trajeval import GoalCompletion
from trajeval.judges import anthropic_judge

GoalCompletion(judge=anthropic_judge()).evaluate(traj)
```

## Roadmap

- [ ] LangChain / LlamaIndex / CrewAI adapters
- [ ] HTML trajectory report (`trajeval report run.json`)
- [ ] Regression mode (compare two trajectories on the same task)
- [ ] More judges (OpenAI, local models)

Contributions welcome — open an issue with the failure mode you wish you could catch.

## License

MIT
