"""Pure helpers for turning the planner LLM's raw output into a plan.

Kept free of LangChain/LangGraph imports so they can be unit-tested
without Ollama or a graph.
"""

import json
import re

NO_TOOL_STRINGS = {"", "null", "none", "no tool", "n/a"}


def fallback_plan() -> list[dict]:
    """A safe single-step plan used when the planner output is unusable."""
    return [
        {
            "description": "Answer the user's request directly without using any tool.",
            "status": "pending",
            "tool_name": None,
        }
    ]


def _extract_json(text: str):
    """Parse JSON, tolerating <think> blocks, code fences and prose."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rstrip()
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as first_error:
        # Last resort: take the outermost {...} or [...] in the text.
        for open_char, close_char in (("{", "}"), ("[", "]")):
            start, end = text.find(open_char), text.rfind(close_char)
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
        raise ValueError(
            f"Planner did not return valid JSON ({first_error.msg})"
        ) from first_error


def parse_plan(content, valid_tools: set[str]) -> list[dict]:
    """Validate raw planner output and return a clean list of steps.

    Raises ValueError with a short, model-readable reason on any problem,
    so the caller can retry or fall back instead of crashing.
    """
    if not isinstance(content, str):
        raise ValueError("Planner returned non-text content")

    data = _extract_json(content)
    plan = data.get("plan") if isinstance(data, dict) else data

    if not isinstance(plan, list) or not plan:
        raise ValueError("Planner returned an empty or invalid plan")

    cleaned = []
    for number, step in enumerate(plan, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Step {number} is not a JSON object")

        description = step.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Step {number} needs a description")

        tool_name = step.get("tool_name")
        if isinstance(tool_name, str) and tool_name.strip().lower() in NO_TOOL_STRINGS:
            tool_name = None  # small models often write "null" as a string

        if tool_name is not None:
            if not isinstance(tool_name, str) or tool_name not in valid_tools:
                raise ValueError(f"Step {number} uses unknown tool {tool_name!r}")

        cleaned.append(
            {
                "description": description.strip(),
                "status": "pending",
                "tool_name": tool_name,
            }
        )

    return cleaned
