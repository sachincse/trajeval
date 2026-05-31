"""Adapters to build a Trajectory from common agent frameworks.

`trajeval` is framework-agnostic: convert your run into a Trajectory once, then
all metrics work. Adapters exist for raw dicts, OpenAI, and Anthropic message
lists. Adding your own is a few lines.
"""
from __future__ import annotations

import json
from typing import Any, Optional, Sequence

from .trajectory import Step, Trajectory


def from_steps(raw_steps: Sequence[dict], goal: Optional[str] = None,
               final_output: Optional[str] = None) -> Trajectory:
    """Build from a list of dicts whose keys match :class:`Step` fields."""
    fields = Step.__dataclass_fields__
    steps = [Step(**{k: v for k, v in s.items() if k in fields}) for s in raw_steps]
    return Trajectory(steps=steps, goal=goal, final_output=final_output)


def from_openai(messages: Sequence[dict], goal: Optional[str] = None) -> Trajectory:
    """Build from an OpenAI chat-completions message list with ``tool_calls``."""
    steps: list[Step] = []
    final: Optional[str] = None
    pending: dict = {}
    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            final = msg.get("content") or final
            for tc in msg.get("tool_calls", []) or []:
                fn = tc.get("function", {})
                pending[tc.get("id")] = Step(tool=fn.get("name"), args=_loads(fn.get("arguments")))
        elif role == "tool":
            step = pending.pop(msg.get("tool_call_id"), Step(tool=msg.get("name")))
            step.output = msg.get("content")
            steps.append(step)
    steps.extend(pending.values())
    return Trajectory(steps=steps, goal=goal, final_output=final)


def from_anthropic(messages: Sequence[dict], goal: Optional[str] = None) -> Trajectory:
    """Build from an Anthropic Messages API list (tool_use / tool_result blocks)."""
    steps: list[Step] = []
    final: Optional[str] = None
    pending: dict = {}
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            if msg.get("role") == "assistant":
                final = content
            continue
        for block in content or []:
            btype = block.get("type")
            if btype == "text" and msg.get("role") == "assistant":
                final = block.get("text")
            elif btype == "tool_use":
                pending[block.get("id")] = Step(tool=block.get("name"), args=block.get("input", {}))
            elif btype == "tool_result":
                step = pending.pop(block.get("tool_use_id"), Step())
                step.output = block.get("content")
                step.error = bool(block.get("is_error"))
                steps.append(step)
    steps.extend(pending.values())
    return Trajectory(steps=steps, goal=goal, final_output=final)


def _loads(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw) if raw else {}
    except (ValueError, TypeError):
        return {"_raw": raw}
