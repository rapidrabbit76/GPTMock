from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator, Dict, List

import httpx

from chatmock.core.constants import CHATGPT_RESPONSES_URL
from chatmock.core.logging import log_json
from chatmock.core.settings import Settings
from chatmock.infra.auth import get_effective_chatgpt_auth
from chatmock.infra.session import ensure_session_id
from chatmock.services.chat import ChatCompletionError
from chatmock.services.model_registry import get_instructions_for_model, normalize_model_name
from chatmock.services.reasoning import allowed_efforts_for_model, build_reasoning_param


def _merge_instructions(base_instructions: str | None, requested_instructions: Any) -> str | None:
    base = base_instructions.strip() if isinstance(base_instructions, str) and base_instructions.strip() else ""
    requested = requested_instructions.strip() if isinstance(requested_instructions, str) and requested_instructions.strip() else ""
    if base and requested:
        return f"{base}\n\n{requested}"
    if requested:
        return requested
    return base or None


def _safe_tool_choice(value: Any) -> Any:
    if value in ("auto", "none"):
        return value
    if isinstance(value, dict):
        return value
    return "auto"


def _is_strict_json_text_format(text_obj: Any) -> bool:
    if not isinstance(text_obj, dict):
        return False
    fmt = text_obj.get("format")
    if not isinstance(fmt, dict):
        return False
    t = fmt.get("type")
    return isinstance(t, str) and t in ("json_schema", "json_object")


def _apply_reasoning_text(
    output_text: str,
    reasoning_summary_text: str,
    reasoning_full_text: str,
    reasoning_compat: str,
    *,
    strict_json_text: bool,
) -> str:
    if strict_json_text:
        return output_text
    compat = (reasoning_compat or "think-tags").strip().lower()
    if compat != "think-tags":
        return output_text

    parts: List[str] = []
    if isinstance(reasoning_summary_text, str) and reasoning_summary_text.strip():
        parts.append(reasoning_summary_text)
    if isinstance(reasoning_full_text, str) and reasoning_full_text.strip():
        parts.append(reasoning_full_text)
    if not parts:
        return output_text

    return f"<think>{'\n\n'.join(parts)}</think>{output_text}"


def _extract_usage(response_obj: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(response_obj, dict):
        return None
    usage = response_obj.get("usage")
    if isinstance(usage, dict):
        return usage
    return None


def _extract_output_text_from_response(response_obj: Dict[str, Any] | None) -> str:
    if not isinstance(response_obj, dict):
        return ""
    output = response_obj.get("output")
    if not isinstance(output, list):
        return ""

    chunks: List[str] = []
    for item in output:
        if not (isinstance(item, dict) and item.get("type") == "message"):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                chunks.append(part.get("text") or "")
    return "".join(chunks)


async def _proxy_stream(upstream: httpx.Response) -> AsyncGenerator[str, None]:
    try:
        async for raw_line in upstream.aiter_lines():
            line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="ignore")
            yield f"{line}\n"
    except httpx.HTTPError as exc:
        err_event = {
            "type": "response.failed",
            "response": {
                "status": "failed",
                "error": {"message": f"Upstream stream interrupted: {exc}"},
            },
        }
        yield f"data: {json.dumps(err_event)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        await upstream.aclose()


async def _collect_non_stream_response(
    upstream: httpx.Response,
    requested_model: Any,
    normalized_model: str,
    settings: Settings,
    request_text_obj: Dict[str, Any] | None,
) -> Dict[str, Any]:
    response_id = "resp"
    created_at = float(time.time())
    status = "completed"
    final_response_obj: Dict[str, Any] | None = None
    full_text = ""
    reasoning_summary_text = ""
    reasoning_full_text = ""
    function_calls: List[Dict[str, Any]] = []
    error_message: str | None = None

    try:
        async for raw_line in upstream.aiter_lines():
            if not raw_line:
                continue
            line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="ignore")
            if not line.startswith("data: "):
                continue

            data = line[len("data: ") :].strip()
            if not data:
                continue
            if data == "[DONE]":
                break

            try:
                evt = json.loads(data)
            except Exception:
                continue

            kind = evt.get("type")
            response_obj = evt.get("response") if isinstance(evt.get("response"), dict) else None
            if isinstance(response_obj, dict):
                response_id_value = response_obj.get("id")
                if isinstance(response_id_value, str) and response_id_value:
                    response_id = response_id_value
                created_at_value = response_obj.get("created_at")
                if isinstance(created_at_value, (int, float)):
                    created_at = float(created_at_value)
                status_value = response_obj.get("status")
                if isinstance(status_value, str) and status_value:
                    status = status_value

            if kind == "response.output_text.delta":
                full_text += evt.get("delta") or ""
            elif kind == "response.reasoning_summary_text.delta":
                reasoning_summary_text += evt.get("delta") or ""
            elif kind == "response.reasoning_text.delta":
                reasoning_full_text += evt.get("delta") or ""
            elif kind == "response.output_item.done":
                item = evt.get("item")
                if isinstance(item, dict) and item.get("type") == "function_call":
                    fc: Dict[str, Any] = {
                        "type": "function_call",
                        "status": item.get("status") if isinstance(item.get("status"), str) else "completed",
                    }
                    if isinstance(item.get("id"), str):
                        fc["id"] = item.get("id")
                    if isinstance(item.get("call_id"), str):
                        fc["call_id"] = item.get("call_id")
                    if isinstance(item.get("name"), str):
                        fc["name"] = item.get("name")
                    if isinstance(item.get("arguments"), str):
                        fc["arguments"] = item.get("arguments")
                    function_calls.append(fc)
            elif kind in ("response.completed", "response.failed"):
                if isinstance(response_obj, dict):
                    final_response_obj = response_obj
                if kind == "response.failed":
                    status = "failed"
                    if isinstance(response_obj, dict):
                        err_obj = response_obj.get("error")
                        if isinstance(err_obj, dict):
                            err_message = err_obj.get("message")
                            if isinstance(err_message, str) and err_message:
                                error_message = err_message
                    if not error_message:
                        error_message = "response.failed"
                break
    finally:
        await upstream.aclose()

    if error_message:
        raise ChatCompletionError(
            error_message,
            status_code=502,
            error_data={"error": {"message": error_message}},
        )

    if not full_text:
        full_text = _extract_output_text_from_response(final_response_obj)

    strict_json_text = _is_strict_json_text_format(request_text_obj)
    rendered_text = _apply_reasoning_text(
        full_text,
        reasoning_summary_text,
        reasoning_full_text,
        settings.reasoning_compat,
        strict_json_text=strict_json_text,
    )

    output: List[Dict[str, Any]] = [
        {
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": rendered_text}],
        }
    ]
    output.extend(function_calls)

    response: Dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "model": requested_model if isinstance(requested_model, str) and requested_model else normalized_model,
        "output": output,
    }

    usage = _extract_usage(final_response_obj)
    if usage:
        response["usage"] = usage
    if isinstance(request_text_obj, dict):
        response["text"] = request_text_obj

    if not strict_json_text and settings.reasoning_compat in ("legacy", "current"):
        if reasoning_summary_text:
            response["reasoning_summary"] = reasoning_summary_text
        if reasoning_full_text:
            response["reasoning"] = reasoning_full_text
    elif not strict_json_text and settings.reasoning_compat == "o3":
        reasoning_blocks: List[str] = []
        if reasoning_summary_text:
            reasoning_blocks.append(reasoning_summary_text)
        if reasoning_full_text:
            reasoning_blocks.append(reasoning_full_text)
        if reasoning_blocks:
            response["reasoning"] = {
                "content": [{"type": "text", "text": "\n\n".join(reasoning_blocks)}]
            }

    return response


async def process_responses_api(
    payload: Dict[str, Any],
    settings: Settings,
    http_client: httpx.AsyncClient,
    *,
    client_session_id: str | None = None,
) -> tuple[Any, bool]:
    requested_model = payload.get("model")
    requested_stream = bool(payload.get("stream", False))

    model = normalize_model_name(requested_model, settings.debug_model)
    base_instructions = get_instructions_for_model(
        model,
        settings.base_instructions,
        settings.gpt5_codex_instructions,
    )
    instructions = _merge_instructions(base_instructions, payload.get("instructions"))

    raw_input_items = payload.get("input")
    input_items = raw_input_items if isinstance(raw_input_items, list) else []

    reasoning_overrides = payload.get("reasoning") if isinstance(payload.get("reasoning"), dict) else None
    reasoning_param = build_reasoning_param(
        settings.reasoning_effort,
        settings.reasoning_summary,
        reasoning_overrides,
        allowed_efforts=allowed_efforts_for_model(model),
    )

    tools = payload.get("tools") if isinstance(payload.get("tools"), list) else []
    tool_choice = _safe_tool_choice(payload.get("tool_choice", "auto"))
    parallel_tool_calls = bool(payload.get("parallel_tool_calls", False))
    text_obj = payload.get("text") if isinstance(payload.get("text"), dict) else None

    access_token, account_id = await get_effective_chatgpt_auth()
    if not access_token or not account_id:
        raise ChatCompletionError(
            "Missing ChatGPT credentials. Run 'python3 chatmock.py login' first.",
            status_code=401,
            error_data={"error": {"message": "Missing ChatGPT credentials. Run 'python3 chatmock.py login' first."}},
        )

    session_id = ensure_session_id(instructions, input_items, client_session_id)

    include: List[str] = []
    if isinstance(reasoning_param, dict):
        include.append("reasoning.encrypted_content")

    upstream_payload: Dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_items,
        "tools": tools,
        "tool_choice": tool_choice,
        "parallel_tool_calls": parallel_tool_calls,
        "reasoning": reasoning_param,
        "store": False,
        "stream": True,
        "prompt_cache_key": session_id,
    }
    if isinstance(text_obj, dict):
        upstream_payload["text"] = text_obj
    if include:
        upstream_payload["include"] = include

    if settings.verbose:
        log_json("OUTBOUND >> ChatGPT Responses API payload", upstream_payload, logger=print)

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "session_id": session_id,
    }

    try:
        upstream = await http_client.post(
            CHATGPT_RESPONSES_URL,
            headers=headers,
            json=upstream_payload,
            timeout=600.0,
        )
    except httpx.RequestError as exc:
        raise ChatCompletionError(
            f"Upstream ChatGPT request failed: {exc}",
            status_code=502,
        ) from exc

    if upstream.status_code >= 400:
        try:
            err_body = upstream.json() if upstream.content else {"raw": upstream.text}
        except Exception:
            err_body = {"raw": upstream.text}
        await upstream.aclose()
        err_message = "Upstream error"
        if isinstance(err_body, dict):
            err_obj = err_body.get("error")
            if isinstance(err_obj, dict):
                msg = err_obj.get("message")
                if isinstance(msg, str) and msg:
                    err_message = msg
        raise ChatCompletionError(
            err_message,
            status_code=upstream.status_code,
            error_data={"error": {"message": err_message}},
        )

    if requested_stream:
        if settings.verbose:
            print(f"OUT responses API (streaming response, model={requested_model or model})")
        return _proxy_stream(upstream), True

    response_obj = await _collect_non_stream_response(
        upstream,
        requested_model,
        model,
        settings,
        text_obj,
    )
    if settings.verbose:
        log_json("OUT responses API", response_obj, logger=print)
    return response_obj, False
