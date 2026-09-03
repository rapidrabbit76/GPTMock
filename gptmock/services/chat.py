from __future__ import annotations

import json
import logging
import time
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
    SSE_RESPONSE_INCOMPLETE,
)
from gptmock.core.logging import log_json
from gptmock.core.settings import Settings
from gptmock.core.utils import extract_usage
from gptmock.infra.auth import get_effective_chatgpt_auth
from gptmock.infra.session import ensure_session_id
from gptmock.infra.sse import sse_translate_chat, sse_translate_text
from gptmock.schemas.messages import (
    convert_chat_messages_to_responses_input,
    convert_tools_with_mapping,
)
from gptmock.services.model_registry import (
    apply_model_overrides,
    get_instructions_for_model,
    normalize_model_name,
    resolve_upstream_model,
)
from gptmock.services.reasoning import (
    allowed_efforts_for_model,
    apply_reasoning_to_message,
    build_reasoning_param,
    extract_reasoning_from_model_name,
)
from gptmock.services.upstream import UpstreamError, send_upstream_request
from gptmock.services.upstream_errors import extract_upstream_error_message

logger = logging.getLogger(__name__)


@dataclass
class ChatCompletionContext:
    """Mutable context threaded through the chat completion pipeline."""

    payload: dict[str, Any]
    settings: Settings
    http_client: httpx.AsyncClient
    client_session_id: str | None = None
    is_stream_override: bool | None = None
    requested_model: str | None = None
    model: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    is_stream: bool = False
    include_usage: bool = False
    reasoning_param: dict[str, Any] | None = None
    instructions: str | None = None
    tools_responses: list[dict[str, Any]] | None = None
    tool_choice: Any = "auto"
    parallel_tool_calls: bool = False
    text_format: dict[str, Any] | None = None
    upstream_options: dict[str, Any] = field(default_factory=dict)
    input_items: list[dict[str, Any]] = field(default_factory=list)
    tool_name_reverse_map: dict[str, str] = field(default_factory=dict)
    access_token: str | None = None
    account_id: str | None = None
    session_id: str = ""


class ChatCompletionError(Exception):
    """Exception raised during chat completion processing."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_data: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_data = error_data or {}


def supplied_parameters(
    payload: dict[str, Any],
    parameter_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Return explicitly supplied, non-null parameter names in stable order."""
    return tuple(
        parameter
        for parameter in parameter_names
        if parameter in payload and payload[parameter] is not None
    )


def apply_output_token_policy(
    payload: dict[str, Any],
    settings: Settings,
    parameter_names: tuple[str, ...],
    *,
    event_logger: logging.Logger | None = None,
) -> tuple[str, ...]:
    """Apply the configured policy for limits unsupported by ChatGPT upstream."""
    supplied = supplied_parameters(payload, parameter_names)
    if not supplied:
        return ()

    if settings.output_token_policy == "reject":
        parameter = supplied[0]
        message = f"Unsupported parameter: {parameter}"
        raise ChatCompletionError(
            message,
            status_code=400,
            error_data={
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": parameter,
                    "code": "unsupported_parameter",
                },
            },
        )

    (event_logger or logger).warning(
        "Ignoring output token limit(s) unsupported by ChatGPT upstream: %s",
        ", ".join(supplied),
    )
    return supplied


async def _call_upstream(
    model: str,
    input_items: list[dict[str, Any]],
    access_token: str,
    account_id: str,
    session_id: str,
    http_client: httpx.AsyncClient,
    settings: Settings,
    *,
    instructions: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any | None = None,
    parallel_tool_calls: bool = False,
    reasoning_param: dict[str, Any] | None = None,
    text_format: dict[str, Any] | None = None,
    request_options: dict[str, Any] | None = None,
) -> httpx.Response:
    """Build a Responses-API payload and send it upstream."""
    include: list[str] = []
    if isinstance(reasoning_param, dict):
        include.append("reasoning.encrypted_content")

    upstream_model, model_overrides = resolve_upstream_model(model)

    payload: dict[str, Any] = {
        "model": upstream_model,
        "instructions": instructions
        if isinstance(instructions, str) and instructions.strip()
        else instructions,
        "input": input_items,
        "tools": tools or [],
        "tool_choice": tool_choice
        if tool_choice in ("auto", "none", "required") or isinstance(tool_choice, dict)
        else "auto",
        "parallel_tool_calls": bool(parallel_tool_calls),
        "store": False,
        "stream": True,
        "prompt_cache_key": session_id,
    }
    if request_options:
        payload.update(request_options)
    if include:
        payload["include"] = include
    if reasoning_param is not None:
        payload["reasoning"] = reasoning_param
    apply_model_overrides(payload, model_overrides)
    if isinstance(text_format, dict):
        payload["text"] = {"format": text_format}

    try:
        return await send_upstream_request(
            payload,
            access_token,
            account_id,
            session_id,
            http_client,
            verbose=settings.verbose,
        )
    except UpstreamError as e:
        raise ChatCompletionError(e.message, status_code=e.status_code) from e


def _build_text_format(response_format: Any) -> dict[str, Any] | None:
    if not isinstance(response_format, dict):
        return None

    fmt_type = response_format.get("type")
    if not isinstance(fmt_type, str):
        return None

    if fmt_type == "json_schema":
        json_schema = response_format.get("json_schema")
        source = json_schema if isinstance(json_schema, dict) else response_format

        name = source.get("name") if isinstance(source, dict) else None
        schema = source.get("schema") if isinstance(source, dict) else None
        strict = source.get("strict") if isinstance(source, dict) else None

        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(schema, dict)
        ):
            raise ChatCompletionError(
                "response_format.type=json_schema requires json_schema.name and json_schema.schema",
                status_code=400,
                error_data={
                    "error": {
                        "message": "response_format.type=json_schema requires json_schema.name and json_schema.schema",
                        "code": "INVALID_RESPONSE_FORMAT",
                    },
                },
            )

        out: dict[str, Any] = {
            "type": "json_schema",
            "name": name.strip(),
            "schema": schema,
        }
        if isinstance(strict, bool):
            out["strict"] = strict
        return out

    if fmt_type == "json_object":
        return {"type": "json_object"}

    if fmt_type == "text":
        return {"type": "text"}

    raise ChatCompletionError(
        f"Unsupported response_format.type: {fmt_type}",
        status_code=400,
        error_data={
            "error": {
                "message": f"Unsupported response_format.type: {fmt_type}",
                "code": "INVALID_RESPONSE_FORMAT",
            },
        },
    )


def _is_strict_json_text_format(text_format: dict[str, Any] | None) -> bool:
    if not isinstance(text_format, dict):
        return False
    t = text_format.get("type")
    return isinstance(t, str) and t in ("json_schema", "json_object")


def _extract_and_normalize(ctx: ChatCompletionContext) -> None:
    payload = ctx.payload
    ctx.requested_model = payload.get("model")
    messages = payload.get("messages")

    if messages is None and isinstance(payload.get("prompt"), str):
        messages = [{"role": "user", "content": payload.get("prompt") or ""}]
    if messages is None and isinstance(payload.get("input"), str):
        messages = [{"role": "user", "content": payload.get("input") or ""}]
    if messages is None:
        messages = []

    if not isinstance(messages, list):
        err_data = {"error": {"message": "Request must include messages: []"}}
        raise ChatCompletionError(
            "Request must include messages: []",
            status_code=400,
            error_data=err_data,
        )

    ctx.messages = messages
    ctx.is_stream = (
        bool(payload.get("stream", False))
        if ctx.is_stream_override is None
        else bool(ctx.is_stream_override)
    )
    stream_options_obj = payload.get("stream_options")
    stream_options: dict[str, Any] = (
        stream_options_obj if isinstance(stream_options_obj, dict) else {}
    )
    ctx.include_usage = bool(stream_options.get("include_usage", False))
    ctx.model = normalize_model_name(ctx.requested_model, ctx.settings.debug_model)


def _derive_policies(ctx: ChatCompletionContext) -> None:
    payload = ctx.payload
    settings = ctx.settings

    model_reasoning = extract_reasoning_from_model_name(ctx.requested_model)
    reasoning_overrides = _chat_reasoning_overrides(payload, model_reasoning)
    try:
        ctx.reasoning_param = build_reasoning_param(
            settings.reasoning_effort,
            settings.reasoning_summary,
            reasoning_overrides,
            allowed_efforts=allowed_efforts_for_model(ctx.model),
        )
    except ValueError as exc:
        raise ChatCompletionError(
            str(exc),
            status_code=400,
            error_data={"error": {"message": str(exc), "code": "INVALID_REASONING"}},
        ) from exc

    ctx.instructions = get_instructions_for_model(
        ctx.model,
        settings.base_instructions,
        settings.gpt5_codex_instructions,
    )

    ctx.tools_responses, tool_name_map = convert_tools_with_mapping(payload.get("tools"))
    ctx.tool_name_reverse_map = {short: original for original, short in tool_name_map.items()}

    ctx.tool_choice = payload.get("tool_choice", "auto")
    if isinstance(ctx.tool_choice, dict):
        function_obj: dict[str, Any] | None = (
            ctx.tool_choice.get("function")
            if isinstance(ctx.tool_choice.get("function"), dict)
            else None
        )
        name = function_obj.get("name") if function_obj is not None else None
        if isinstance(name, str) and name in tool_name_map and function_obj is not None:
            ctx.tool_choice = {
                **ctx.tool_choice,
                "function": {**function_obj, "name": tool_name_map[name]},
            }
    ctx.parallel_tool_calls = bool(payload.get("parallel_tool_calls", False))
    ctx.upstream_options = _chat_upstream_options(payload)

    extra_tools: list[dict[str, Any]] = []
    responses_tools_payload = (
        payload.get("responses_tools")
        if isinstance(payload.get("responses_tools"), list)
        else []
    )

    if isinstance(responses_tools_payload, list):
        for _t in responses_tools_payload:
            if not (isinstance(_t, dict) and isinstance(_t.get("type"), str)):
                continue
            if _t.get("type") not in ("web_search", "web_search_preview"):
                raise ChatCompletionError(
                    "Only web_search/web_search_preview are supported in responses_tools",
                    status_code=400,
                    error_data={
                        "error": {
                            "message": "Only web_search/web_search_preview are supported in responses_tools",
                            "code": "RESPONSES_TOOL_UNSUPPORTED",
                        },
                    },
                )
            extra_tools.append(_t)

        if not extra_tools and settings.default_web_search:
            responses_tool_choice = payload.get("responses_tool_choice")
            if not (
                isinstance(responses_tool_choice, str)
                and responses_tool_choice == "none"
            ):
                extra_tools = [{"type": "web_search"}]

        if extra_tools:
            MAX_TOOLS_BYTES = 32768
            try:
                size = len(json.dumps(extra_tools))
            except Exception:
                logger.debug("Failed to calculate tools JSON size", exc_info=True)
                size = 0
            if size > MAX_TOOLS_BYTES:
                raise ChatCompletionError(
                    "responses_tools too large",
                    status_code=400,
                    error_data={
                        "error": {
                            "message": "responses_tools too large",
                            "code": "RESPONSES_TOOLS_TOO_LARGE",
                        },
                    },
                )
            ctx.tools_responses = (ctx.tools_responses or []) + extra_tools

    responses_tool_choice = payload.get("responses_tool_choice")
    if isinstance(responses_tool_choice, str) and responses_tool_choice in (
        "auto",
        "none",
    ):
        ctx.tool_choice = responses_tool_choice


def _build_upstream_request(ctx: ChatCompletionContext) -> None:
    payload = ctx.payload
    ctx.text_format = _build_text_format(payload.get("response_format"))

    ctx.input_items = convert_chat_messages_to_responses_input(ctx.messages)
    prompt = payload.get("prompt")
    if not ctx.input_items and isinstance(prompt, str) and prompt.strip():
        ctx.input_items = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        ]


def _chat_upstream_options(payload: dict[str, Any]) -> dict[str, Any]:
    """Map supported Chat Completions options to the upstream Responses request."""
    options: dict[str, Any] = {}
    for key in (
        "metadata",
        "previous_response_id",
        "prompt_cache_retention",
        "safety_identifier",
        "service_tier",
        "truncation",
    ):
        if key in payload and payload[key] is not None:
            options[key] = payload[key]
    return options


def _chat_reasoning_overrides(
    payload: dict[str, Any],
    model_reasoning: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Normalize Chat Completions reasoning fields without hiding conflicts."""
    reasoning = payload.get("reasoning")
    explicit_reasoning = isinstance(reasoning, dict)
    overrides = dict(reasoning) if explicit_reasoning else {}

    effort = payload.get("reasoning_effort")
    if effort is None:
        if overrides:
            return overrides
        return dict(model_reasoning) if isinstance(model_reasoning, dict) else None
    if not isinstance(effort, str):
        raise ChatCompletionError(
            "reasoning_effort must be a string",
            status_code=400,
            error_data={
                "error": {
                    "message": "reasoning_effort must be a string",
                    "type": "invalid_request_error",
                    "param": "reasoning_effort",
                    "code": "invalid_parameter",
                },
            },
        )

    nested_effort = overrides.get("effort") if explicit_reasoning else None
    if nested_effort is not None and nested_effort != effort:
        message = "Conflicting reasoning effort values"
        raise ChatCompletionError(
            message,
            status_code=400,
            error_data={
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": "reasoning_effort",
                    "code": "conflicting_parameters",
                },
            },
        )
    overrides["effort"] = effort
    return overrides


async def _authenticate(ctx: ChatCompletionContext) -> None:
    ctx.access_token, ctx.account_id = await get_effective_chatgpt_auth()
    if not ctx.access_token or not ctx.account_id:
        raise ChatCompletionError(
            "Missing ChatGPT credentials. Run 'python3 gptmock.py login' first.",
            status_code=401,
            error_data={
                "error": {
                    "message": "Missing ChatGPT credentials. Run 'python3 gptmock.py login' first.",
                },
            },
        )

    ctx.session_id = ensure_session_id(
        ctx.instructions,
        ctx.input_items,
        ctx.client_session_id,
    )


async def _call_upstream_with_context(
    ctx: ChatCompletionContext,
    *,
    instructions: str | None,
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
) -> httpx.Response:
    if not ctx.access_token or not ctx.account_id:
        raise ChatCompletionError(
            "Missing ChatGPT credentials. Run 'python3 gptmock.py login' first.",
            status_code=401,
            error_data={
                "error": {
                    "message": "Missing ChatGPT credentials. Run 'python3 gptmock.py login' first.",
                },
            },
        )

    return await _call_upstream(
        model=ctx.model,
        input_items=ctx.input_items,
        access_token=ctx.access_token,
        account_id=ctx.account_id,
        session_id=ctx.session_id,
        http_client=ctx.http_client,
        settings=ctx.settings,
        instructions=instructions,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=ctx.parallel_tool_calls,
        reasoning_param=ctx.reasoning_param,
        text_format=ctx.text_format,
        request_options=ctx.upstream_options,
    )


async def _read_upstream_error_body(upstream: httpx.Response) -> Any:
    try:
        await upstream.aread()
        return upstream.json() if upstream.content else {"raw": upstream.text}
    except Exception:
        logger.debug("Failed to read upstream error response", exc_info=True)
        return {"raw": getattr(upstream, "text", "unknown error")}
    finally:
        await upstream.aclose()


async def _send_upstream(ctx: ChatCompletionContext) -> httpx.Response:
    upstream = await _call_upstream_with_context(
        ctx,
        instructions=ctx.instructions,
        tools=ctx.tools_responses,
        tool_choice=ctx.tool_choice,
    )
    if upstream.status_code < 400:
        return upstream

    err_body = await _read_upstream_error_body(upstream)
    if ctx.settings.verbose:
        logger.debug("Upstream error status=%s", upstream.status_code)
    message = extract_upstream_error_message(
        err_body,
        status_code=upstream.status_code,
    )
    raise ChatCompletionError(
        message,
        status_code=upstream.status_code,
        error_data={"error": {"message": message}},
    )


def _adapt_streaming_response(
    ctx: ChatCompletionContext,
    upstream: httpx.Response,
    created: int,
) -> tuple[Any, bool]:
    if ctx.settings.verbose:
        logger.debug(
            "OUT chat completion (streaming response, model=%s)",
            ctx.requested_model or ctx.model,
        )

    stream_iter = sse_translate_chat(
        upstream,
        ctx.requested_model or ctx.model,
        created,
        verbose=ctx.settings.verbose_obfuscation,
        vlog=print if ctx.settings.verbose_obfuscation else None,
        reasoning_compat=ctx.settings.reasoning_compat,
        include_usage=ctx.include_usage,
        tool_name_reverse=ctx.tool_name_reverse_map,
    )
    return stream_iter, True


def _decode_chat_sse_data(raw: str | bytes) -> str | None:
    if not raw:
        return None
    line = raw if isinstance(raw, str) else raw.decode("utf-8", errors="ignore")
    if not line.startswith("data: "):
        return None
    data = line[len("data: ") :].strip()
    return data or None


def _update_chat_sse_metadata(
    evt: Any,
    response_id: str,
    usage_obj: dict[str, int] | None,
) -> tuple[str, dict[str, int] | None]:
    mu = extract_usage(evt)
    if mu:
        usage_obj = mu

    response = evt.get("response")
    if isinstance(response, dict) and isinstance(response.get("id"), str):
        response_id = response.get("id") or response_id
    return response_id, usage_obj


def _handle_chat_sse_event(
    evt: Any,
    full_text: str,
    reasoning_summary_text: str,
    reasoning_full_text: str,
    tool_calls: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    tool_name_reverse: dict[str, str],
) -> tuple[str, str, str, str | None, bool]:
    kind = evt.get("type")
    if kind == SSE_OUTPUT_TEXT_DELTA:
        return (
            full_text + (evt.get("delta") or ""),
            reasoning_summary_text,
            reasoning_full_text,
            None,
            False,
        )
    if kind == SSE_REASONING_SUMMARY_TEXT_DELTA:
        return (
            full_text,
            reasoning_summary_text + (evt.get("delta") or ""),
            reasoning_full_text,
            None,
            False,
        )
    if kind == SSE_REASONING_TEXT_DELTA:
        return (
            full_text,
            reasoning_summary_text,
            reasoning_full_text + (evt.get("delta") or ""),
            None,
            False,
        )
    if kind == SSE_OUTPUT_ITEM_DONE:
        item = evt.get("item") or {}
        if isinstance(item, dict) and item.get("type") == "function_call":
            call_id = item.get("call_id") or item.get("id") or ""
            name = item.get("name") or ""
            args = item.get("arguments") or ""
            if (
                isinstance(call_id, str)
                and isinstance(name, str)
                and isinstance(args, str)
            ):
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name_reverse.get(name, name),
                            "arguments": args,
                        },
                    },
                )
        return full_text, reasoning_summary_text, reasoning_full_text, None, False
    if kind == SSE_CONTENT_PART_DONE:
        part = evt.get("part")
        if isinstance(part, dict) and part.get("type") == "output_text":
            part_annotations = part.get("annotations")
            if isinstance(part_annotations, list):
                annotations.extend(part_annotations)
        return full_text, reasoning_summary_text, reasoning_full_text, None, False
    if kind == SSE_RESPONSE_FAILED:
        response = evt.get("response")
        if isinstance(response, dict):
            error = response.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str):
                    return (
                        full_text,
                        reasoning_summary_text,
                        reasoning_full_text,
                        message,
                        True,
                    )
        return (
            full_text,
            reasoning_summary_text,
            reasoning_full_text,
            "response.failed",
            True,
        )
    if kind in (SSE_RESPONSE_COMPLETED, SSE_RESPONSE_INCOMPLETE):
        return full_text, reasoning_summary_text, reasoning_full_text, None, True
    return full_text, reasoning_summary_text, reasoning_full_text, None, False


async def _collect_chat_sse_events(
    upstream: httpx.Response,
    tool_name_reverse: dict[str, str],
) -> tuple[
    str,
    str,
    str,
    str,
    list[dict[str, Any]],
    str | None,
    dict[str, int] | None,
    list[dict[str, Any]],
    dict[str, Any],
]:
    full_text = ""
    reasoning_summary_text = ""
    reasoning_full_text = ""
    response_id = "chatcmpl"
    tool_calls: list[dict[str, Any]] = []
    error_message: str | None = None
    usage_obj: dict[str, int] | None = None
    annotations: list[dict[str, Any]] = []
    response_metadata: dict[str, Any] = {}
    terminal_received = False

    try:
        async for raw in upstream.aiter_lines():
            data = _decode_chat_sse_data(raw)
            if not data:
                continue
            if data == "[DONE]":
                if not terminal_received and not error_message:
                    error_message = "Upstream stream ended before a terminal response event"
                break

            try:
                evt = json.loads(data)
            except Exception:
                logger.debug("Failed to parse SSE event JSON", exc_info=True)
                continue

            response_id, usage_obj = _update_chat_sse_metadata(
                evt, response_id, usage_obj,
            )
            kind = evt.get("type")
            response = evt.get("response")
            if isinstance(response, dict):
                for key in ("model", "service_tier", "status", "incomplete_details"):
                    if response.get(key) is not None:
                        response_metadata[key] = response[key]
            if kind in (
                SSE_RESPONSE_COMPLETED,
                SSE_RESPONSE_FAILED,
                SSE_RESPONSE_INCOMPLETE,
            ):
                terminal_received = True
            (
                full_text,
                reasoning_summary_text,
                reasoning_full_text,
                event_error,
                should_break,
            ) = _handle_chat_sse_event(
                evt,
                full_text,
                reasoning_summary_text,
                reasoning_full_text,
                tool_calls,
                annotations,
                tool_name_reverse,
            )
            if event_error:
                error_message = event_error
            if should_break:
                break
    finally:
        await upstream.aclose()

    if not terminal_received and not error_message:
        error_message = "Upstream stream ended before a terminal response event"

    return (
        full_text,
        reasoning_summary_text,
        reasoning_full_text,
        response_id,
        tool_calls,
        error_message,
        usage_obj,
        annotations,
        response_metadata,
    )


async def _adapt_non_streaming_response(
    ctx: ChatCompletionContext,
    upstream: httpx.Response,
    created: int,
) -> tuple[Any, bool]:
    (
        full_text,
        reasoning_summary_text,
        reasoning_full_text,
        response_id,
        tool_calls,
        error_message,
        usage_obj,
        annotations,
        response_metadata,
    ) = await _collect_chat_sse_events(upstream, ctx.tool_name_reverse_map)

    if error_message:
        raise ChatCompletionError(
            error_message,
            status_code=502,
            error_data={"error": {"message": error_message}},
        )

    message: dict[str, Any] = {
        "role": "assistant",
        "content": full_text if full_text else None,
    }
    if annotations:
        message["annotations"] = annotations
    if tool_calls:
        message["tool_calls"] = tool_calls
    if not _is_strict_json_text_format(ctx.text_format):
        message = apply_reasoning_to_message(
            message,
            reasoning_summary_text,
            reasoning_full_text,
            ctx.settings.reasoning_compat,
        )

    finish_reason = "tool_calls" if tool_calls else _finish_reason(response_metadata)

    completion: dict[str, Any] = {
        "id": response_id or "chatcmpl",
        "object": "chat.completion",
        "created": created,
        "model": response_metadata.get("model") or ctx.requested_model or ctx.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            },
        ],
        **({"usage": usage_obj} if usage_obj else {}),
    }
    if response_metadata.get("service_tier") is not None:
        completion["service_tier"] = response_metadata["service_tier"]

    if ctx.settings.verbose:
        log_json("OUT chat completion", completion, logger=logger.debug)

    return completion, False


def _finish_reason(response_metadata: dict[str, Any]) -> str:
    if response_metadata.get("status") != "incomplete":
        return "stop"
    details = response_metadata.get("incomplete_details")
    reason = details.get("reason") if isinstance(details, dict) else None
    return "content_filter" if reason == "content_filter" else "length"


async def process_chat_completion(
    payload: dict[str, Any],
    settings: Settings,
    http_client: httpx.AsyncClient,
    *,
    client_session_id: str | None = None,
    is_stream: bool | None = None,
) -> tuple[Any, bool]:
    """Process chat completion request."""
    apply_output_token_policy(
        payload,
        settings,
        ("max_completion_tokens", "max_tokens", "max_output_tokens"),
    )
    ctx = ChatCompletionContext(
        payload=payload,
        settings=settings,
        http_client=http_client,
        client_session_id=client_session_id,
        is_stream_override=is_stream,
    )

    _extract_and_normalize(ctx)
    _derive_policies(ctx)
    _build_upstream_request(ctx)
    await _authenticate(ctx)
    upstream = await _send_upstream(ctx)

    created = int(time.time())
    if ctx.is_stream:
        return _adapt_streaming_response(ctx, upstream, created)
    return await _adapt_non_streaming_response(ctx, upstream, created)


async def process_text_completion(
    payload: dict[str, Any],
    settings: Settings,
    http_client: httpx.AsyncClient,
    *,
    client_session_id: str | None = None,
) -> tuple[Any, bool]:
    """Process text completion request (/v1/completions).

    Args:
        payload: Request payload
        settings: Application settings
        http_client: Async HTTP client
        client_session_id: Optional session ID from request headers

    Returns:
        Tuple of (response_generator_or_dict, is_streaming)

    Raises:
        ChatCompletionError: On processing errors
    """
    apply_output_token_policy(
        payload,
        settings,
        ("max_tokens", "max_completion_tokens", "max_output_tokens"),
    )
    # 1. Extract request parameters
    requested_model = payload.get("model")
    prompt = payload.get("prompt")

    if isinstance(prompt, list):
        prompt = "".join([p if isinstance(p, str) else "" for p in prompt])
    if not isinstance(prompt, str):
        prompt = payload.get("suffix") or ""

    stream_req = bool(payload.get("stream", False))
    stream_options_obj = payload.get("stream_options")
    stream_options: dict[str, Any] = (
        stream_options_obj if isinstance(stream_options_obj, dict) else {}
    )
    include_usage = bool(stream_options.get("include_usage", False))

    # 2. Normalize model
    model = normalize_model_name(requested_model, settings.debug_model)

    # 3. Convert to messages format
    messages = [{"role": "user", "content": prompt or ""}]
    input_items = convert_chat_messages_to_responses_input(messages)

    # 4. Build reasoning parameters
    model_reasoning = extract_reasoning_from_model_name(requested_model)
    reasoning_overrides = _chat_reasoning_overrides(payload, model_reasoning)
    try:
        reasoning_param = build_reasoning_param(
            settings.reasoning_effort,
            settings.reasoning_summary,
            reasoning_overrides,
            allowed_efforts=allowed_efforts_for_model(model),
        )
    except ValueError as exc:
        raise ChatCompletionError(
            str(exc),
            status_code=400,
            error_data={"error": {"message": str(exc), "code": "INVALID_REASONING"}},
        ) from exc

    # 5. Get instructions
    instructions = get_instructions_for_model(
        model,
        settings.base_instructions,
        settings.gpt5_codex_instructions,
    )

    # 6. Get auth credentials
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

    # 7. Get session ID
    session_id = ensure_session_id(instructions, input_items, client_session_id)

    # 8. Call upstream
    try:
        upstream = await _call_upstream(
            model=model,
            input_items=input_items,
            access_token=access_token,
            account_id=account_id,
            session_id=session_id,
            http_client=http_client,
            settings=settings,
            instructions=instructions,
            reasoning_param=reasoning_param,
        )
    except ChatCompletionError:
        raise

    # 9. Handle upstream errors
    if upstream.status_code >= 400:
        try:
            await upstream.aread()
            err_body: Any = upstream.json() if upstream.content else {"raw": upstream.text}
        except Exception:
            logger.debug("Failed to read upstream error response", exc_info=True)
            err_body = {"raw": getattr(upstream, "text", "unknown error")}
        message = extract_upstream_error_message(
            err_body,
            status_code=upstream.status_code,
        )
        raise ChatCompletionError(
            message,
            status_code=upstream.status_code,
            error_data={
                "error": {
                    "message": message,
                },
            },
        )

    # 10. Return streaming or non-streaming response
    created = int(time.time())

    if stream_req:
        if settings.verbose:
            logger.debug(
                "OUT text completion (streaming response, model=%s)",
                requested_model or model,
            )

        stream_iter = sse_translate_text(
            upstream,
            requested_model or model,
            created,
            verbose=settings.verbose_obfuscation,
            vlog=print if settings.verbose_obfuscation else None,
            include_usage=include_usage,
        )
        return stream_iter, True
    # Collect full response
    full_text = ""
    response_id = "cmpl"
    usage_obj: dict[str, int] | None = None
    response_model: str | None = None
    service_tier: str | None = None
    finish_reason = "stop"
    terminal_received = False
    error_message: str | None = None

    try:
        async for raw_line in upstream.aiter_lines():
            if not raw_line:
                continue
            line = (
                raw_line
                if isinstance(raw_line, str)
                else raw_line.decode("utf-8", errors="ignore")
            )
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :].strip()
            if not data or data == "[DONE]":
                if data == "[DONE]":
                    if not terminal_received:
                        error_message = "Upstream stream ended before a terminal response event"
                    break
                continue
            try:
                evt = json.loads(data)
            except Exception:
                logger.debug("Failed to parse SSE event JSON", exc_info=True)
                continue

            response = evt.get("response")
            if isinstance(response, dict):
                if isinstance(response.get("id"), str):
                    response_id = response.get("id") or response_id
                if isinstance(response.get("model"), str) and response.get("model"):
                    response_model = response["model"]
                if isinstance(response.get("service_tier"), str):
                    service_tier = response["service_tier"]
            mu = extract_usage(evt)
            if mu:
                usage_obj = mu
            kind = evt.get("type")
            if kind == SSE_OUTPUT_TEXT_DELTA:
                full_text += evt.get("delta") or ""
            elif kind == SSE_RESPONSE_COMPLETED:
                terminal_received = True
                break
            elif kind == SSE_RESPONSE_INCOMPLETE:
                terminal_received = True
                metadata = response if isinstance(response, dict) else {"status": "incomplete"}
                finish_reason = _finish_reason({**metadata, "status": "incomplete"})
                break
            elif kind == SSE_RESPONSE_FAILED:
                terminal_received = True
                response = evt.get("response")
                error = response.get("error") if isinstance(response, dict) else None
                message = error.get("message") if isinstance(error, dict) else None
                error_message = message if isinstance(message, str) and message else "response.failed"
                break
    finally:
        await upstream.aclose()

    if not terminal_received and not error_message:
        error_message = "Upstream stream ended before a terminal response event"
    if error_message:
        raise ChatCompletionError(
            error_message,
            status_code=502,
            error_data={"error": {"message": error_message}},
        )

    completion: dict[str, Any] = {
        "id": response_id or "cmpl",
        "object": "text_completion",
        "created": created,
        "model": response_model or requested_model or model,
        "choices": [
            {
                "index": 0,
                "text": full_text,
                "finish_reason": finish_reason,
                "logprobs": None,
            },
        ],
        **({"usage": usage_obj} if usage_obj else {}),
    }
    if service_tier is not None:
        completion["service_tier"] = service_tier

    if settings.verbose:
        log_json("OUT text completion", completion, logger=logger.debug)

    return completion, False
