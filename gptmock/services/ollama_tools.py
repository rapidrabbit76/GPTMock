"""Translate completed OpenAI tool calls into native Ollama objects."""
from __future__ import annotations

import json
from typing import Any

from gptmock.services.chat import ChatCompletionError


def native_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for call in calls:
        function = call.get("function", {})
        arguments = function.get("arguments", "{}")
        try:
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict) or not isinstance(function.get("name"), str) or not function["name"]:
                raise ValueError("Expected a named function with object arguments")
        except (ValueError, TypeError) as exc:
            raise ChatCompletionError("Invalid upstream tool arguments; no tool was emitted", status_code=502) from exc
        result.append({"function": {"name": function["name"], "arguments": arguments}})
    return result


def accumulate_tool_deltas(pending: dict[int, dict[str, Any]], deltas: list[dict[str, Any]]) -> None:
    for delta in deltas:
        index = delta.get("index", 0)
        call = pending.setdefault(index, {"function": {"name": "", "arguments": ""}})
        function = delta.get("function", {})
        for key in ("name", "arguments"):
            value = function.get(key)
            if isinstance(value, str):
                call["function"][key] += value
