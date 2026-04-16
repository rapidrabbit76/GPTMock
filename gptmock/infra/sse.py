from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from gptmock.core.constants import (
    SSE_CONTENT_PART_DONE,
    SSE_FUNCTION_CALL_ARGS_DELTA,
    SSE_FUNCTION_CALL_ARGS_DONE,
    SSE_OUTPUT_ITEM_ADDED,
    SSE_OUTPUT_ITEM_DONE,
    SSE_OUTPUT_TEXT_DELTA,
    SSE_OUTPUT_TEXT_DONE,
    SSE_REASONING_SUMMARY_PART_ADDED,
    SSE_REASONING_SUMMARY_TEXT_DELTA,
    SSE_REASONING_TEXT_DELTA,
    SSE_RESPONSE_COMPLETED,
    SSE_RESPONSE_FAILED,
)
from gptmock.core.utils import extract_usage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SSE Chat Context — replaces ~12 loose local variables
# ---------------------------------------------------------------------------


@dataclass
class SSEChatContext:
    """Accumulated state for translating upstream SSE into OpenAI chat chunks."""

    # Immutable config (set once)
    model: str
    created: int
    compat: str
    verbose: bool
    vlog: Any
    include_usage: bool

    # Mutable stream state
    response_id: str = "chatcmpl-stream"
    think_open: bool = False
    think_closed: bool = False
    sent_stop_chunk: bool = False
    saw_any_summary: bool = False
    pending_summary_paragraph: bool = False
    done: bool = False
    upstream_usage: dict[str, int] | None = None

    # Web-search tool call state
    ws_state: dict[str, Any] = field(default_factory=dict)
    ws_index: dict[str, int] = field(default_factory=dict)
    ws_next_index: int = 0

    # Incremental tool-call streaming state
    tc_streamed: set[str] = field(default_factory=set)
    tool_call_detected: bool = False

    # ---- helpers --------------------------------------------------------

    def chunk(
        self,
        delta: dict[str, Any],
        *,
        finish_reason: str | None = None,
        **extra: Any,
    ) -> bytes:
        """Build and SSE-encode one ``chat.completion.chunk``."""
        obj: dict[str, Any] = {
            "id": self.response_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        obj.update(extra)
        return f"data: {json.dumps(obj)}\n\n".encode()

    def ws_idx(self, call_id: str) -> int:
        """Get or assign a stable tool-call index for *call_id*."""
        if call_id not in self.ws_index:
            self.ws_index[call_id] = self.ws_next_index
            self.ws_next_index += 1
        return self.ws_index.get(call_id, 0)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _serialize_tool_args(eff_args: Any) -> str:
    """Serialize tool-call arguments to a JSON string."""
    if isinstance(eff_args, (dict, list)):
        return json.dumps(eff_args)
    if isinstance(eff_args, str):
        try:
            parsed = json.loads(eff_args)
            if isinstance(parsed, (dict, list)):
                return json.dumps(parsed)
            return json.dumps({"query": eff_args})
        except (json.JSONDecodeError, ValueError):
            return json.dumps({"query": eff_args})
    return "{}"


def _merge_ws_params(src: Any, params: dict[str, Any]) -> None:
    """Merge web-search parameters from *src* into *params* (in-place)."""
    if not isinstance(src, dict):
        return
    for whole in ("parameters", "args", "arguments", "input"):
        if isinstance(src.get(whole), dict):
            params.update(src[whole])
    if isinstance(src.get("query"), str):
        params.setdefault("query", src["query"])
    if isinstance(src.get("q"), str):
        params.setdefault("query", src["q"])
    for rk in ("recency", "time_range", "days"):
        if src.get(rk) is not None and rk not in params:
            params[rk] = src[rk]
    for dk in ("domains", "include_domains", "include"):
        if isinstance(src.get(dk), list) and "domains" not in params:
            params["domains"] = src[dk]
    for mk in ("max_results", "topn", "limit"):
        if src.get(mk) is not None and "max_results" not in params:
            params["max_results"] = src[mk]


# ---------------------------------------------------------------------------
# Event handlers — each returns list[bytes] of SSE frames to yield
# ---------------------------------------------------------------------------


def _handle_web_search(
    ctx: SSEChatContext,
    evt: dict[str, Any],
    kind: str,
) -> list[bytes]:
    try:
        call_id = evt.get("item_id") or "ws_call"
        if ctx.verbose and ctx.vlog:
            try:
                ctx.vlog(f"CM_TOOLS {kind} id={call_id} -> tool_calls(web_search)")
            except Exception:
                logger.debug(
                    "Failed to log verbose message for web_search", exc_info=True,
                )

        item = evt.get("item") if isinstance(evt.get("item"), dict) else {}
        params_dict = (
            ctx.ws_state.setdefault(call_id, {})
            if isinstance(ctx.ws_state.get(call_id), dict)
            else {}
        )
        _merge_ws_params(item, params_dict)
        _merge_ws_params(evt if isinstance(evt, dict) else None, params_dict)

        if isinstance(params_dict, dict):
            try:
                ctx.ws_state.setdefault(call_id, {}).update(params_dict)
            except Exception:
                logger.debug("Failed to update web_search state", exc_info=True)

        return []
    except Exception:
        logger.debug("Failed to handle web_search event", exc_info=True)
        return []


def _handle_output_item_added(
    ctx: SSEChatContext,
    evt: dict[str, Any],
    kind: str,
) -> list[bytes]:
    item = evt.get("item") or {}
    if not isinstance(item, dict):
        return []

    item_type = item.get("type")
    if item_type not in ("function_call", "web_search_call"):
        return []

    item_id = item.get("id") or ""
    call_id = item.get("call_id") or item_id
    if not isinstance(call_id, str) or not call_id:
        return []

    idx = ctx.ws_idx(call_id)
    if isinstance(item_id, str) and item_id and item_id != call_id:
        ctx.ws_index[item_id] = idx
        if item_id in ctx.ws_state and call_id not in ctx.ws_state:
            ctx.ws_state[call_id] = ctx.ws_state[item_id]

    ctx.tool_call_detected = True

    if item_type != "function_call":
        return []

    name = item.get("name") or ""
    if not isinstance(name, str):
        return []

    ctx.tc_streamed.add(call_id)
    if isinstance(item_id, str) and item_id:
        ctx.tc_streamed.add(item_id)

    return [
        ctx.chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": idx,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    },
                ],
            },
        ),
    ]


def _handle_function_call_args_delta(
    ctx: SSEChatContext,
    evt: dict[str, Any],
    kind: str,
) -> list[bytes]:
    raw = evt.get("delta")
    if not isinstance(raw, str) or not raw:
        raw = evt.get("arguments")
    if not isinstance(raw, str) or not raw:
        return []

    call_id = evt.get("item_id") or evt.get("call_id") or ""
    if not isinstance(call_id, str) or not call_id:
        return []

    if call_id not in ctx.ws_index:
        logger.debug(
            "function_call_arguments.%s before output_item.added for %s; dropping",
            kind,
            call_id,
        )
        return []

    idx = ctx.ws_idx(call_id)
    ctx.tc_streamed.add(call_id)

    return [
        ctx.chunk(
            {
                "tool_calls": [
                    {
                        "index": idx,
                        "function": {"arguments": raw},
                    },
                ],
            },
        ),
    ]


def _handle_text_delta(
    ctx: SSEChatContext,
    evt: dict[str, Any],
    kind: str,
) -> list[bytes]:
    if ctx.tool_call_detected:
        return []

    out: list[bytes] = []
    delta = evt.get("delta") or ""
    if ctx.compat == "think-tags" and ctx.think_open and not ctx.think_closed:
        out.append(ctx.chunk({"content": "</think>"}))
        ctx.think_open = False
        ctx.think_closed = True
    out.append(ctx.chunk({"content": delta}))
    return out


def _handle_output_item_done(
    ctx: SSEChatContext,
    evt: dict[str, Any],
    kind: str,
) -> list[bytes]:
    item = evt.get("item") or {}
    if not isinstance(item, dict):
        return []
    if item.get("type") not in ("function_call", "web_search_call"):
        return []

    item_id = item.get("id") or ""
    call_id = item.get("call_id") or item_id
    if not isinstance(call_id, str):
        call_id = ""

    if (
        isinstance(item_id, str) and item_id
        and call_id and item_id != call_id
    ):
        if item_id in ctx.ws_index and call_id not in ctx.ws_index:
            ctx.ws_index[call_id] = ctx.ws_index[item_id]
        elif call_id in ctx.ws_index and item_id not in ctx.ws_index:
            ctx.ws_index[item_id] = ctx.ws_index[call_id]
        if item_id in ctx.tc_streamed:
            ctx.tc_streamed.add(call_id)
        elif call_id in ctx.tc_streamed:
            ctx.tc_streamed.add(item_id)
        if item_id in ctx.ws_state and call_id not in ctx.ws_state:
            ctx.ws_state[call_id] = ctx.ws_state[item_id]

    name = item.get("name") or (
        "web_search" if item.get("type") == "web_search_call" else ""
    )
    raw_args = item.get("arguments") or item.get("parameters")

    if isinstance(raw_args, dict):
        try:
            ctx.ws_state.setdefault(call_id, {}).update(raw_args)
        except Exception:
            logger.debug(
                "Failed to update tool call state with raw arguments", exc_info=True,
            )

    if item.get("type") == "web_search_call" and ctx.verbose and ctx.vlog:
        try:
            ctx.vlog(
                f"CM_TOOLS {SSE_OUTPUT_ITEM_DONE} web_search_call "
                f"id={call_id} has_args={bool(raw_args)}",
            )
        except Exception:
            logger.debug("Failed to log verbose message for tool call", exc_info=True)

    ctx.tool_call_detected = True

    already_streamed = bool(call_id) and call_id in ctx.tc_streamed
    if already_streamed:
        return []

    eff_args = ctx.ws_state.get(
        call_id,
        raw_args if isinstance(raw_args, (dict, list, str)) else {},
    )
    try:
        args = _serialize_tool_args(eff_args)
    except Exception:
        logger.debug("Failed to serialize tool arguments", exc_info=True)
        args = "{}"

    idx = ctx.ws_idx(call_id)

    if not (
        isinstance(call_id, str)
        and isinstance(name, str)
        and isinstance(args, str)
    ):
        return []

    ctx.tc_streamed.add(call_id)
    if isinstance(item_id, str) and item_id:
        ctx.tc_streamed.add(item_id)

    return [
        ctx.chunk(
            {
                "tool_calls": [
                    {
                        "index": idx,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    },
                ],
            },
        ),
    ]


def _handle_summary_part_added(
    ctx: SSEChatContext,
    evt: dict[str, Any],
    kind: str,
) -> list[bytes]:
    if ctx.compat in ("think-tags", "o3"):
        if ctx.saw_any_summary:
            ctx.pending_summary_paragraph = True
        else:
            ctx.saw_any_summary = True
    return []


def _handle_reasoning_delta(
    ctx: SSEChatContext,
    evt: dict[str, Any],
    kind: str,
) -> list[bytes]:
    if ctx.tool_call_detected:
        return []

    delta_txt = evt.get("delta") or ""
    out: list[bytes] = []

    if ctx.compat == "o3":
        if kind == SSE_REASONING_SUMMARY_TEXT_DELTA and ctx.pending_summary_paragraph:
            out.append(
                ctx.chunk({"reasoning": {"content": [{"type": "text", "text": "\n"}]}}),
            )
            ctx.pending_summary_paragraph = False
        out.append(
            ctx.chunk(
                {"reasoning": {"content": [{"type": "text", "text": delta_txt}]}},
            ),
        )

    elif ctx.compat == "think-tags":
        if not ctx.think_open and not ctx.think_closed:
            out.append(ctx.chunk({"content": "<think>"}))
            ctx.think_open = True
        if ctx.think_open and not ctx.think_closed:
            if (
                kind == SSE_REASONING_SUMMARY_TEXT_DELTA
                and ctx.pending_summary_paragraph
            ):
                out.append(ctx.chunk({"content": "\n"}))
                ctx.pending_summary_paragraph = False
            out.append(ctx.chunk({"content": delta_txt}))

    else:  # legacy
        if kind == SSE_REASONING_SUMMARY_TEXT_DELTA:
            out.append(
                ctx.chunk({"reasoning_summary": delta_txt, "reasoning": delta_txt}),
            )
        else:
            out.append(ctx.chunk({"reasoning": delta_txt}))

    return out


def _handle_content_part_done(
    ctx: SSEChatContext,
    evt: dict[str, Any],
    kind: str,
) -> list[bytes]:
    part = evt.get("part")
    if not isinstance(part, dict) or part.get("type") != "output_text":
        return []
    part_annotations = part.get("annotations")
    if not isinstance(part_annotations, list) or not part_annotations:
        return []
    return [ctx.chunk({"annotations": part_annotations})]


def _handle_text_done(
    ctx: SSEChatContext,
    evt: dict[str, Any],
    kind: str,
) -> list[bytes]:
    return []


def _handle_failed(
    ctx: SSEChatContext,
    evt: dict[str, Any],
    kind: str,
) -> list[bytes]:
    err = evt.get("response", {}).get("error", {}).get("message", "response.failed")
    return [f"data: {json.dumps({'error': {'message': err}})}\n\n".encode()]


def _handle_completed(
    ctx: SSEChatContext,
    evt: dict[str, Any],
    kind: str,
) -> list[bytes]:
    out: list[bytes] = []

    m = extract_usage(evt)
    if m:
        ctx.upstream_usage = m

    if ctx.compat == "think-tags" and ctx.think_open and not ctx.think_closed:
        out.append(ctx.chunk({"content": "</think>"}))
        ctx.think_open = False
        ctx.think_closed = True

    if not ctx.sent_stop_chunk:
        terminal = "tool_calls" if ctx.tool_call_detected else "stop"
        out.append(ctx.chunk({}, finish_reason=terminal))
        ctx.sent_stop_chunk = True

    # Usage chunk
    if ctx.include_usage and ctx.upstream_usage:
        try:
            out.append(ctx.chunk({}, usage=ctx.upstream_usage))
        except Exception:
            logger.debug("Failed to emit usage chunk", exc_info=True)

    out.append(b"data: [DONE]\n\n")
    ctx.done = True
    return out


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_CHAT_DISPATCH: dict[
    str, Callable[[SSEChatContext, dict[str, Any], str], list[bytes]],
] = {
    SSE_OUTPUT_ITEM_ADDED: _handle_output_item_added,
    SSE_OUTPUT_TEXT_DELTA: _handle_text_delta,
    SSE_FUNCTION_CALL_ARGS_DELTA: _handle_function_call_args_delta,
    SSE_FUNCTION_CALL_ARGS_DONE: _handle_function_call_args_delta,
    SSE_OUTPUT_ITEM_DONE: _handle_output_item_done,
    SSE_REASONING_SUMMARY_PART_ADDED: _handle_summary_part_added,
    SSE_REASONING_SUMMARY_TEXT_DELTA: _handle_reasoning_delta,
    SSE_REASONING_TEXT_DELTA: _handle_reasoning_delta,
    SSE_CONTENT_PART_DONE: _handle_content_part_done,
    SSE_OUTPUT_TEXT_DONE: _handle_text_done,
    SSE_RESPONSE_FAILED: _handle_failed,
    SSE_RESPONSE_COMPLETED: _handle_completed,
}


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


async def sse_translate_chat(
    upstream: httpx.Response,
    model: str,
    created: int,
    verbose: bool = False,
    vlog: Any = None,
    reasoning_compat: str = "think-tags",
    *,
    include_usage: bool = False,
) -> AsyncGenerator[bytes]:
    ctx = SSEChatContext(
        model=model,
        created=created,
        compat=(reasoning_compat or "think-tags").strip().lower(),
        verbose=verbose,
        vlog=vlog,
        include_usage=include_usage,
    )

    try:
        try:
            line_iterator = upstream.aiter_lines()
        except httpx.HTTPError as e:
            if verbose and vlog:
                vlog(f"Failed to start stream: {e}")
            yield b"data: [DONE]\n\n"
            return

        async for raw in line_iterator:
            # ---- parse SSE line ----
            try:
                if not raw:
                    continue
                line = (
                    raw
                    if isinstance(raw, str)
                    else raw.decode("utf-8", errors="ignore")
                )
                if verbose and vlog:
                    vlog(line)
                if not line.startswith("data: "):
                    continue
                data = line[len("data: ") :].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break
                try:
                    evt = json.loads(data)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
            except (
                httpx.ReadError,
                httpx.RemoteProtocolError,
                ConnectionError,
                BrokenPipeError,
            ) as e:
                if verbose and vlog:
                    vlog(f"Stream interrupted: {e}")
                yield b"data: [DONE]\n\n"
                return

            # ---- dispatch ----
            kind = evt.get("type")
            if isinstance(evt.get("response"), dict) and isinstance(
                evt["response"].get("id"),
                str,
            ):
                ctx.response_id = evt["response"].get("id") or ctx.response_id

            if isinstance(kind, str) and "web_search_call" in kind:
                for frame in _handle_web_search(ctx, evt, kind):
                    yield frame

            if kind in _CHAT_DISPATCH:
                for frame in _CHAT_DISPATCH[kind](ctx, evt, kind):
                    yield frame

            if ctx.done:
                break

    finally:
        await upstream.aclose()


async def sse_translate_text(
    upstream: httpx.Response,
    model: str,
    created: int,
    verbose: bool = False,
    vlog: Any = None,
    *,
    include_usage: bool = False,
) -> AsyncGenerator[bytes]:
    response_id = "cmpl-stream"
    upstream_usage: dict[str, int] | None = None

    def _chunk(
        delta: dict[str, Any],
        *,
        finish_reason: str | None = None,
        **extra: Any,
    ) -> bytes:
        obj: dict[str, Any] = {
            "id": response_id,
            "object": "text_completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, **delta, "finish_reason": finish_reason}],
        }
        obj.update(extra)
        return f"data: {json.dumps(obj)}\n\n".encode()

    try:
        async for raw_line in upstream.aiter_lines():
            if not raw_line:
                continue
            line = (
                raw_line
                if isinstance(raw_line, str)
                else raw_line.decode("utf-8", errors="ignore")
            )
            if verbose and vlog:
                vlog(line)
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :].strip()
            if not data or data == "[DONE]":
                if data == "[DONE]":
                    yield _chunk({"text": ""}, finish_reason="stop")
                continue
            try:
                evt = json.loads(data)
            except Exception:
                logger.debug("Failed to parse SSE event JSON", exc_info=True)
                continue

            kind = evt.get("type")
            if isinstance(evt.get("response"), dict) and isinstance(
                evt["response"].get("id"),
                str,
            ):
                response_id = evt["response"].get("id") or response_id

            if kind == SSE_OUTPUT_TEXT_DELTA:
                yield _chunk({"text": evt.get("delta") or ""})
            elif kind == SSE_OUTPUT_TEXT_DONE:
                yield _chunk({"text": ""}, finish_reason="stop")
            elif kind == SSE_RESPONSE_COMPLETED:
                m = extract_usage(evt)
                if m:
                    upstream_usage = m
                if include_usage and upstream_usage:
                    try:
                        yield _chunk({"text": ""}, usage=upstream_usage)
                    except Exception:
                        logger.debug("Failed to emit final usage chunk", exc_info=True)
                yield b"data: [DONE]\n\n"
                break
    finally:
        await upstream.aclose()
