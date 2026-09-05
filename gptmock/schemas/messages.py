from __future__ import annotations

import base64
from typing import Any

_TOOL_NAME_LIMIT = 64


def _base_short_candidate(name: str) -> str:
    if len(name) <= _TOOL_NAME_LIMIT:
        return name
    if name.startswith("mcp__"):
        idx = name.rfind("__")
        if idx > 0:
            candidate = f"mcp__{name[idx + 2:]}"
            return candidate[:_TOOL_NAME_LIMIT] if len(candidate) > _TOOL_NAME_LIMIT else candidate
    return name[:_TOOL_NAME_LIMIT]


def build_short_name_map(names: list[str]) -> dict[str, str]:
    """Map original names to unique shortened names (<=64 chars)."""
    used: set[str] = set()
    mapping: dict[str, str] = {}

    for name in names:
        candidate = _base_short_candidate(name)
        if candidate in used:
            base = candidate
            i = 1
            while True:
                suffix = f"_{i}"
                allowed = max(0, _TOOL_NAME_LIMIT - len(suffix))
                trimmed = base[:allowed] if len(base) > allowed else base
                unique_candidate = trimmed + suffix
                if unique_candidate not in used:
                    candidate = unique_candidate
                    break
                i += 1
        used.add(candidate)
        mapping[name] = candidate
    return mapping


def convert_tools_with_mapping(tools: Any) -> tuple[list[dict[str, Any]], dict[str, str]]:
    out: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return out, {}

    original_names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = fn.get("name") if isinstance(fn, dict) else None
        if isinstance(name, str) and name:
            original_names.append(name)

    short_name_map = build_short_name_map(original_names)

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(name, str) or not name:
            continue
        desc = fn.get("description") if isinstance(fn, dict) else None
        params = fn.get("parameters") if isinstance(fn, dict) else None
        strict = fn.get("strict") if isinstance(fn, dict) else None
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        out.append(
            {
                "type": "function",
                "name": short_name_map.get(name, name),
                "description": desc or "",
                "strict": strict if isinstance(strict, bool) else False,
                "parameters": params,
            },
        )

    return out, short_name_map


def _normalize_image_data_url(url: str) -> str:
    try:
        if not isinstance(url, str):
            return url
        if not url.startswith("data:image/"):
            return url
        if ";base64," not in url:
            return url
        header, data = url.split(",", 1)
        try:
            from urllib.parse import unquote

            data = unquote(data)
        except Exception:
            pass
        data = data.strip().replace("\n", "").replace("\r", "")
        data = data.replace("-", "+").replace("_", "/")
        pad = (-len(data)) % 4
        if pad:
            data = data + ("=" * pad)
        try:
            base64.b64decode(data, validate=True)
        except Exception:
            return url
        return f"{header},{data}"
    except Exception:
        return url


def _convert_tool_message(message: dict[str, Any]) -> dict[str, Any] | None:
    call_id = message.get("tool_call_id") or message.get("id")
    if not isinstance(call_id, str) or not call_id:
        return None
    content = message.get("content", "")
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if isinstance(text, str) and text:
                    texts.append(text)
        content = "\n".join(texts)
    if not isinstance(content, str):
        return None
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": content,
    }


def _convert_assistant_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tc in message.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        tc_type = tc.get("type", "function")
        if tc_type != "function":
            continue
        call_id = tc.get("id") or tc.get("call_id")
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = fn.get("name") if isinstance(fn, dict) else None
        args = fn.get("arguments") if isinstance(fn, dict) else None
        if isinstance(call_id, str) and isinstance(name, str) and isinstance(args, str):
            out.append(
                {
                    "type": "function_call",
                    "name": name,
                    "arguments": args,
                    "call_id": call_id,
                },
            )
    return out


def _convert_content_parts(content: Any, role: Any) -> list[dict[str, Any]]:
    content_items: list[dict[str, Any]] = []
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                text = part.get("text") or part.get("content") or ""
                if isinstance(text, str) and text:
                    kind = "output_text" if role == "assistant" else "input_text"
                    content_items.append({"type": kind, "text": text})
            elif ptype == "image_url":
                image = part.get("image_url")
                url = image.get("url") if isinstance(image, dict) else image
                if isinstance(url, str) and url:
                    content_items.append(
                        {
                            "type": "input_image",
                            "image_url": _normalize_image_data_url(url),
                        },
                    )
    elif isinstance(content, str) and content:
        kind = "output_text" if role == "assistant" else "input_text"
        content_items.append({"type": kind, "text": content})
    return content_items


def convert_chat_messages_to_responses_input(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    input_items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "tool":
            tool_item = _convert_tool_message(message)
            if tool_item:
                input_items.append(tool_item)
            continue

        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            input_items.extend(_convert_assistant_tool_calls(message))

        content_items = _convert_content_parts(message.get("content", ""), role)

        if not content_items:
            continue
        role_out = role if role in {"assistant", "developer", "system", "user"} else "user"
        input_items.append(
            {"type": "message", "role": role_out, "content": content_items},
        )
    return input_items


def convert_tools_chat_to_responses(tools: Any) -> list[dict[str, Any]]:
    out, _ = convert_tools_with_mapping(tools)
    return out
