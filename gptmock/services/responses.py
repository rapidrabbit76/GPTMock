from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from gptmock.core.constants import (
    SSE_CONTENT_PART_DONE,
    SSE_OUTPUT_ITEM_DONE,
    SSE_OUTPUT_TEXT_DELTA,
    SSE_REASONING_SUMMARY_TEXT_DELTA,
    SSE_REASONING_TEXT_DELTA,
    SSE_RESPONSE_COMPLETED,
    SSE_RESPONSE_FAILED,
)
from gptmock.core.logging import log_json
from gptmock.core.settings import Settings
from gptmock.infra.auth import get_effective_chatgpt_auth
from gptmock.infra.session import ensure_session_id
from gptmock.services.chat import ChatCompletionError
from gptmock.services.model_registry import (
    get_instructions_for_model,
    normalize_model_name,
    resolve_upstream_model,
)
from gptmock.services.reasoning import allowed_efforts_for_model, build_reasoning_param
from gptmock.services.upstream import UpstreamError, send_upstream_request
from gptmock.services.view_image import (
    execute_view_image,
    is_view_image_tool_call,
    normalize_view_image_tools,
)

logger = logging.getLogger(__name__)
_MAX_VIEW_IMAGE_FOLLOWUPS = 4


def _merge_instructions(
    base_instructions: str | None, requested_instructions: Any,
) -> str | None:
    base = (
        base_instructions.strip()
        if isinstance(base_instructions, str) and base_instructions.strip()
        else ""
    )
    requested = (
        requested_instructions.strip()
        if isinstance(requested_instructions, str) and requested_instructions.strip()
        else ""
    )
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

    parts: list[str] = []
    if isinstance(reasoning_summary_text, str) and reasoning_summary_text.strip():
        parts.append(reasoning_summary_text)
    if isinstance(reasoning_full_text, str) and reasoning_full_text.strip():
        parts.append(reasoning_full_text)
    if not parts:
        return output_text

    return f"<think>{'\n\n'.join(parts)}</think>{output_text}"


def _extract_usage(response_obj: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(response_obj, dict):
        return None
    usage = response_obj.get("usage")
    if isinstance(usage, dict):
        return usage
    return None


def _extract_output_text_from_response(response_obj: dict[str, Any] | None) -> str:
    if not isinstance(response_obj, dict):
        return ""
    output = response_obj.get("output")
    if not isinstance(output, list):
        return ""

    chunks: list[str] = []
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


async def _proxy_stream(upstream: httpx.Response) -> AsyncGenerator[str]:
    try:
        async for raw_line in upstream.aiter_lines():
            line = (
                raw_line
                if isinstance(raw_line, str)
                else raw_line.decode("utf-8", errors="ignore")
            )
            yield f"{line}\n"
    except httpx.HTTPError as exc:
        err_event = {
            "type": SSE_RESPONSE_FAILED,
            "response": {
                "status": "failed",
                "error": {"message": f"Upstream stream interrupted: {exc}"},
            },
        }
        yield f"data: {json.dumps(err_event)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        await upstream.aclose()


@dataclass
class CollectorState:
    """Mutable state for non-stream response collection."""

    response_id: str = "resp"
    created_at: float = 0.0
    status: str = "completed"
    final_response_obj: dict[str, Any] | None = None
    full_text: str = ""
    reasoning_summary_text: str = ""
    reasoning_full_text: str = ""
    function_calls: list[dict[str, Any]] = field(default_factory=list)
    image_generations: list[dict[str, Any]] = field(default_factory=list)
    annotations: list[dict[str, Any]] = field(default_factory=list)
    error_message: str | None = None


def _parse_sse_data(raw_line: str | bytes) -> str | None:
    """Parse a raw SSE line, returning the data string or None."""

    line = (
        raw_line
        if isinstance(raw_line, str)
        else raw_line.decode("utf-8", errors="ignore")
    )
    if not line.startswith("data: "):
        return None
    data = line[len("data: ") :].strip()
    if not data:
        return None
    return data


def _update_response_metadata(
    state: CollectorState, response_obj: dict[str, Any] | None,
) -> None:
    """Update response_id, created_at, status from event's response object."""

    if not isinstance(response_obj, dict):
        return
    rid = response_obj.get("id")
    if isinstance(rid, str) and rid:
        state.response_id = rid
    cat = response_obj.get("created_at")
    if isinstance(cat, (int, float)):
        state.created_at = float(cat)
    st = response_obj.get("status")
    if isinstance(st, str) and st:
        state.status = st


def _handle_text_delta(state: CollectorState, evt: dict[str, Any]) -> None:
    state.full_text += evt.get("delta") or ""


def _handle_reasoning_summary_delta(state: CollectorState, evt: dict[str, Any]) -> None:
    state.reasoning_summary_text += evt.get("delta") or ""


def _handle_reasoning_text_delta(state: CollectorState, evt: dict[str, Any]) -> None:
    state.reasoning_full_text += evt.get("delta") or ""


def _handle_output_item_done(state: CollectorState, evt: dict[str, Any]) -> None:
    item = evt.get("item")
    if not isinstance(item, dict):
        return

    if item.get("type") == "image_generation_call":
        state.image_generations.append(dict(item))
        return

    if item.get("type") != "function_call":
        return

    fc: dict[str, Any] = {
        "type": "function_call",
        "status": item.get("status")
        if isinstance(item.get("status"), str)
        else "completed",
    }
    for key in ("id", "call_id", "name", "arguments"):
        if isinstance(item.get(key), str):
            fc[key] = item.get(key)
    state.function_calls.append(fc)


def _handle_content_part_done(state: CollectorState, evt: dict[str, Any]) -> None:
    part = evt.get("part")
    if isinstance(part, dict) and part.get("type") == "output_text":
        part_annotations = part.get("annotations")
        if isinstance(part_annotations, list):
            state.annotations.extend(part_annotations)


def _handle_response_terminal(
    state: CollectorState, evt: dict[str, Any], kind: str,
) -> bool:
    """Handle response.completed or response.failed. Returns True if loop should break."""

    response_obj = (
        evt.get("response") if isinstance(evt.get("response"), dict) else None
    )
    if isinstance(response_obj, dict):
        state.final_response_obj = response_obj
    if kind == SSE_RESPONSE_FAILED:
        state.status = "failed"
        if isinstance(response_obj, dict):
            err_obj = response_obj.get("error")
            if isinstance(err_obj, dict):
                err_message = err_obj.get("message")
                if isinstance(err_message, str) and err_message:
                    state.error_message = err_message
        if not state.error_message:
            state.error_message = "response.failed"
    return True


_RESPONSES_EVENT_HANDLERS: dict[
    str, Callable[[CollectorState, dict[str, Any]], None],
] = {
    SSE_OUTPUT_TEXT_DELTA: _handle_text_delta,
    SSE_REASONING_SUMMARY_TEXT_DELTA: _handle_reasoning_summary_delta,
    SSE_REASONING_TEXT_DELTA: _handle_reasoning_text_delta,
    SSE_OUTPUT_ITEM_DONE: _handle_output_item_done,
    SSE_CONTENT_PART_DONE: _handle_content_part_done,
}


def _build_responses_api_result(
    state: CollectorState,
    requested_model: Any,
    normalized_model: str,
    settings: Settings,
    request_text_obj: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the final Responses API response dict from collector state."""

    full_text = state.full_text
    if not full_text:
        full_text = _extract_output_text_from_response(state.final_response_obj)

    strict_json_text = _is_strict_json_text_format(request_text_obj)
    rendered_text = _apply_reasoning_text(
        full_text,
        state.reasoning_summary_text,
        state.reasoning_full_text,
        settings.reasoning_compat,
        strict_json_text=strict_json_text,
    )

    annotations = state.annotations
    if not annotations and isinstance(state.final_response_obj, dict):
        for _item in state.final_response_obj.get("output") or []:
            if isinstance(_item, dict) and _item.get("type") == "message":
                for _part in _item.get("content") or []:
                    if isinstance(_part, dict) and _part.get("type") == "output_text":
                        _ann = _part.get("annotations")
                        if isinstance(_ann, list):
                            annotations.extend(_ann)

    content_item: dict[str, Any] = {"type": "output_text", "text": rendered_text}
    if annotations:
        content_item["annotations"] = annotations

    output: list[dict[str, Any]] = [
        {
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [content_item],
        },
    ]
    output.extend(state.function_calls)
    output.extend(state.image_generations)

    response: dict[str, Any] = {
        "id": state.response_id,
        "object": "response",
        "created_at": state.created_at,
        "status": state.status,
        "model": requested_model
        if isinstance(requested_model, str) and requested_model
        else normalized_model,
        "output": output,
    }

    usage = _extract_usage(state.final_response_obj)
    if usage:
        response["usage"] = usage
    if isinstance(request_text_obj, dict):
        response["text"] = request_text_obj

    if not strict_json_text and settings.reasoning_compat in ("legacy", "current"):
        if state.reasoning_summary_text:
            response["reasoning_summary"] = state.reasoning_summary_text
        if state.reasoning_full_text:
            response["reasoning"] = state.reasoning_full_text
    elif not strict_json_text and settings.reasoning_compat == "o3":
        reasoning_blocks: list[str] = []
        if state.reasoning_summary_text:
            reasoning_blocks.append(state.reasoning_summary_text)
        if state.reasoning_full_text:
            reasoning_blocks.append(state.reasoning_full_text)
        if reasoning_blocks:
            response["reasoning"] = {
                "content": [{"type": "text", "text": "\n\n".join(reasoning_blocks)}],
            }

    return response


async def _collect_non_stream_response(
    upstream: httpx.Response,
    requested_model: Any,
    normalized_model: str,
    settings: Settings,
    request_text_obj: dict[str, Any] | None,
) -> dict[str, Any]:
    return _build_responses_api_result_from_state(
        await _collect_non_stream_state(upstream),
        requested_model,
        normalized_model,
        settings,
        request_text_obj,
    )


def _build_responses_api_result_from_state(
    state: CollectorState,
    requested_model: Any,
    normalized_model: str,
    settings: Settings,
    request_text_obj: dict[str, Any] | None,
) -> dict[str, Any]:
    if state.error_message:
        raise ChatCompletionError(
            state.error_message,
            status_code=502,
            error_data={"error": {"message": state.error_message}},
        )
    return _build_responses_api_result(
        state,
        requested_model,
        normalized_model,
        settings,
        request_text_obj,
    )


async def _collect_non_stream_state(upstream: httpx.Response) -> CollectorState:
    state = CollectorState(created_at=float(time.time()))

    try:
        async for raw_line in upstream.aiter_lines():
            if not raw_line:
                continue
            data = _parse_sse_data(raw_line)
            if data is None:
                continue
            if data == "[DONE]":
                break

            try:
                evt = json.loads(data)
            except Exception:
                logger.debug("Failed to parse SSE event JSON", exc_info=True)
                continue

            kind = evt.get("type")
            response_obj = (
                evt.get("response") if isinstance(evt.get("response"), dict) else None
            )
            _update_response_metadata(state, response_obj)

            handler = _RESPONSES_EVENT_HANDLERS.get(kind)
            if handler is not None:
                handler(state, evt)
            elif kind in (SSE_RESPONSE_COMPLETED, SSE_RESPONSE_FAILED):
                if _handle_response_terminal(state, evt, kind):
                    break
    finally:
        await upstream.aclose()

    return state


def _view_image_followup_input(
    input_items: list[dict[str, Any]],
    function_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    followup = list(input_items)
    for call in function_calls:
        followup.append(call)
        call_id = call.get("call_id") or call.get("id")
        output = execute_view_image(call.get("arguments"))
        if isinstance(call_id, str) and call_id:
            followup.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                },
            )
    return followup


def _extract_upstream_error_message(err_body: Any) -> str:
    if isinstance(err_body, dict):
        err_obj = err_body.get("error")
        if isinstance(err_obj, dict):
            msg = err_obj.get("message")
            if isinstance(msg, str) and msg:
                return msg
    return "Upstream error"


async def _raise_for_upstream_error(upstream: httpx.Response) -> None:
    if upstream.status_code < 400:
        return
    try:
        await upstream.aread()
        err_body = upstream.json() if upstream.content else {"raw": upstream.text}
    except Exception:
        logger.debug("Failed to read upstream error response", exc_info=True)
        err_body = {"raw": upstream.text}
    await upstream.aclose()
    err_message = _extract_upstream_error_message(err_body)
    raise ChatCompletionError(
        err_message,
        status_code=upstream.status_code,
        error_data={"error": {"message": err_message}},
    )


async def process_responses_api(
    payload: dict[str, Any],
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

    reasoning_overrides = (
        payload.get("reasoning") if isinstance(payload.get("reasoning"), dict) else None
    )
    reasoning_param = build_reasoning_param(
        settings.reasoning_effort,
        settings.reasoning_summary,
        reasoning_overrides,
        allowed_efforts=allowed_efforts_for_model(model),
    )

    tools = payload.get("tools") if isinstance(payload.get("tools"), list) else []
    tools, view_image_enabled = normalize_view_image_tools(tools)
    tool_choice = _safe_tool_choice(payload.get("tool_choice", "auto"))
    parallel_tool_calls = bool(payload.get("parallel_tool_calls", False))
    text_obj = payload.get("text") if isinstance(payload.get("text"), dict) else None

    access_token, account_id = await get_effective_chatgpt_auth()
    if not access_token or not account_id:
        raise ChatCompletionError(
            "Missing ChatGPT credentials. Run 'python3 gptmock.py login' first.",
            status_code=401,
            error_data={
                "error": {
                    "message": "Missing ChatGPT credentials. Run 'python3 gptmock.py login' first.",
                },
            },
        )

    session_id = ensure_session_id(instructions, input_items, client_session_id)

    include: list[str] = []
    if isinstance(reasoning_param, dict):
        include.append("reasoning.encrypted_content")

    upstream_model, service_tier = resolve_upstream_model(model)

    upstream_payload: dict[str, Any] = {
        "model": upstream_model,
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
    if service_tier is not None:
        upstream_payload["service_tier"] = service_tier
    if isinstance(text_obj, dict):
        upstream_payload["text"] = text_obj
    if include:
        upstream_payload["include"] = include

    try:
        upstream = await send_upstream_request(
            upstream_payload,
            access_token,
            account_id,
            session_id,
            http_client,
            verbose=settings.verbose,
        )
    except UpstreamError as exc:
        raise ChatCompletionError(
            exc.message,
            status_code=exc.status_code,
        ) from exc

    await _raise_for_upstream_error(upstream)

    if requested_stream:
        if settings.verbose:
            logger.debug(
                "OUT responses API (streaming response, model=%s)",
                requested_model or model,
            )
        return _proxy_stream(upstream), True

    state = await _collect_non_stream_state(upstream)
    for _ in range(_MAX_VIEW_IMAGE_FOLLOWUPS):
        if state.error_message or not view_image_enabled:
            break
        if not state.function_calls or not all(is_view_image_tool_call(call) for call in state.function_calls):
            break
        upstream_payload["input"] = _view_image_followup_input(
            upstream_payload["input"],
            state.function_calls,
        )
        try:
            upstream = await send_upstream_request(
                upstream_payload,
                access_token,
                account_id,
                session_id,
                http_client,
                verbose=settings.verbose,
            )
        except UpstreamError as exc:
            raise ChatCompletionError(
                exc.message,
                status_code=exc.status_code,
            ) from exc
        await _raise_for_upstream_error(upstream)
        state = await _collect_non_stream_state(upstream)

    response_obj = _build_responses_api_result_from_state(
        state,
        requested_model,
        model,
        settings,
        text_obj,
    )
    if settings.verbose:
        log_json("OUT responses API", response_obj, logger=logger.debug)
    return response_obj, False
