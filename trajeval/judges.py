"""Optional LLM-judge factories for ``GoalCompletion``.

These require the relevant SDK and an API key. They are kept out of the core so
structural metrics stay zero-dependency.
"""
from __future__ import annotations

import re
from typing import Callable, Optional


def _parse_score(text: str) -> float:
    match = re.search(r"[01](?:\.\d+)?", text)
    return float(match.group()) if match else 0.0


def anthropic_judge(model: str = "claude-haiku-4-5-20251001",
                    client=None) -> Callable:
    """Return a judge callable that scores goal completion in [0, 1] using Claude."""

    def judge(goal: Optional[str], final_output: Optional[str], steps) -> float:
        nonlocal client
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        prompt = (
            "You are grading whether an AI agent achieved its goal.\n"
            f"GOAL:\n{goal}\n\nFINAL OUTPUT:\n{final_output}\n\n"
            "Reply with ONLY a number from 0.0 to 1.0 for how fully the goal was met."
        )
        msg = client.messages.create(
            model=model,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_score(msg.content[0].text.strip())

    return judge
