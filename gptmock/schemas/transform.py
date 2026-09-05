from __future__ import annotations

import json
from typing import Any


def to_data_url(image_str: str) -> str:
    if not isinstance(image_str, str) or not image_str:
        return image_str
    s = image_str.strip()
    if s.startswith("data:image/"):
        return s
    if s.startswith("http://") or s.startswith("https://"):
        return s
    b64 = s.replace("\n", "").replace("\r", "")
    kind = "image/png"
    if b64.startswith("/9j/"):
        kind = "image/jpeg"
    elif b64.startswith("iVBORw0KGgo"):
        kind = "image/png"
    elif b64.startswith("R0lGOD"):
        kind = "image/gif"
    return f"data:{kind};base64,{b64}"


def _build_content_parts(content: Any, images: list[Any]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    if isinstance(content, list):
        for p in content:
            if (
                isinstance(p, dict)
                and p.get("type") == "text"
                and isinstance(p.get("text"), str)
            ):
                parts.append({"type": "text", "text": p.get("text")})
    elif isinstance(content, str):
        parts.append({"type": "text", "text": content})
    for img in images:
        url = to_data_url(img)
        if isinstance(url, str) and url:
            parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def _build_ollama_tool_calls(
    message: dict[str, Any], pending_call_ids: list[str], call_counter: int,
) -> tuple[list[dict[str, Any]], int]:
    tcs: list[dict[str, Any]] = []
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return tcs, call_counter
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn_raw = tc.get("function")
        fn: dict[str, Any] = fn_raw if isinstance(fn_raw, dict) else {}
        fn_name = fn.get("name")
        name = fn_name if isinstance(fn_name, str) else None
        args = fn.get("arguments")
        if name is None:
            continue
        call_id_raw = tc.get("id") or tc.get("call_id")
        if isinstance(call_id_raw, str) and call_id_raw:
            call_id = call_id_raw
        else:
            call_counter += 1
            call_id = f"ollama_call_{call_counter}"
        pending_call_ids.append(call_id)
        tcs.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": (
                        args
                        if isinstance(args, str)
                        else (json.dumps(args) if isinstance(args, dict) else "{}")
                    ),
                },
            },
        )
    return tcs, call_counter


def _attach_top_images(out: list[dict[str, Any]], top_images: list[str] | None) -> None:
    if not isinstance(top_images, list) or not top_images:
        return
    attach_to: dict[str, Any] | None = None
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            attach_to = out[i]
            break
    if attach_to is None:
        attach_to = {"role": "user", "content": []}
        out.append(attach_to)
    content_parts = attach_to.get("content")
    if not isinstance(content_parts, list):
        content_parts = []
        attach_to["content"] = content_parts
    for img in top_images:
        url = to_data_url(img)
        if isinstance(url, str) and url:
            content_parts.append({"type": "image_url", "image_url": {"url": url}})


def convert_ollama_messages(
    messages: list[dict[str, Any]] | None, top_images: list[str] | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    msgs = messages if isinstance(messages, list) else []
    pending_call_ids: list[str] = []
    pending_names: dict[str, list[str]] = {}
    call_counter = 0
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or "user"
        nm: dict[str, Any] = {"role": role}

        content = m.get("content")
        images_raw = m.get("images")
        images: list[Any] = []
        if isinstance(images_raw, list):
            images = images_raw
        parts = _build_content_parts(content, images)
        if parts:
            nm["content"] = parts

        if role == "assistant" and isinstance(m.get("tool_calls"), list):
            tcs, call_counter = _build_ollama_tool_calls(
                m, pending_call_ids, call_counter,
            )
            if tcs:
                nm["tool_calls"] = tcs
                for call in tcs:
                    pending_names.setdefault(call["function"]["name"], []).append(call["id"])

        if role == "tool":
            tci = m.get("tool_call_id") or m.get("id")
            if not isinstance(tci, str) or not tci:
                named_calls = pending_names.get(m.get("tool_name"), [])
                if named_calls:
                    tci = named_calls[0]
                elif pending_call_ids:
                    tci = pending_call_ids[0]
            if isinstance(tci, str) and tci:
                nm["tool_call_id"] = tci
                if tci in pending_call_ids:
                    pending_call_ids.remove(tci)
                for ids in pending_names.values():
                    if tci in ids:
                        ids.remove(tci)

            if not parts and isinstance(content, str):
                nm["content"] = content

        out.append(nm)

    _attach_top_images(out, top_images)
    return out


def _normalize_single_ollama_tool(t: dict[str, Any]) -> dict[str, Any] | None:
    fn_raw = t.get("function")
    if isinstance(fn_raw, dict):
        fn: dict[str, Any] = fn_raw
        fn_name = fn.get("name")
        name = fn_name if isinstance(fn_name, str) else None
        if not name:
            return None
        parameters = fn.get("parameters")
        normalized_function: dict[str, Any] = {
            "name": name,
            "description": fn.get("description") or "",
            "parameters": (
                parameters
                if isinstance(parameters, dict)
                else {"type": "object", "properties": {}}
            ),
        }
        strict = fn.get("strict")
        if not isinstance(strict, bool):
            strict = t.get("strict")
        if isinstance(strict, bool):
            normalized_function["strict"] = strict
        return {
            "type": "function",
            "function": normalized_function,
        }
    name = t.get("name") if isinstance(t.get("name"), str) else None
    if name:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": t.get("description") or "",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    return None


def normalize_ollama_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return out
    for t in tools:
        if not isinstance(t, dict):
            continue
        normalized = _normalize_single_ollama_tool(t)
        if normalized is not None:
            out.append(normalized)
    return out
