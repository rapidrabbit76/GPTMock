from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict

import httpx


def _extract_usage(evt: Dict[str, Any]) -> Dict[str, int] | None:
    """Extract usage statistics from upstream event."""
    try:
        usage = (evt.get("response") or {}).get("usage")
        if not isinstance(usage, dict):
            return None
        pt = int(usage.get("input_tokens") or 0)
        ct = int(usage.get("output_tokens") or 0)
        tt = int(usage.get("total_tokens") or (pt + ct))
        return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}
    except Exception:
        return None


async def sse_translate_chat(
    upstream,
    model: str,
    created: int,
    verbose: bool = False,
    vlog=None,
    reasoning_compat: str = "think-tags",
    *,
    include_usage: bool = False,
) -> AsyncGenerator[bytes, None]:
    response_id = "chatcmpl-stream"
    compat = (reasoning_compat or "think-tags").strip().lower()
    think_open = False
    think_closed = False
    saw_output = False
    sent_stop_chunk = False
    saw_any_summary = False
    pending_summary_paragraph = False
    upstream_usage = None
    ws_state: dict[str, Any] = {}
    ws_index: dict[str, int] = {}
    ws_next_index: int = 0
    
    def _serialize_tool_args(eff_args: Any) -> str:
        """
        Serialize tool call arguments with proper JSON handling.
        
        Args:
            eff_args: Arguments to serialize (dict, list, str, or other)
            
        Returns:
            JSON string representation of the arguments
        """
        if isinstance(eff_args, (dict, list)):
            return json.dumps(eff_args)
        elif isinstance(eff_args, str):
            try:
                parsed = json.loads(eff_args)
                if isinstance(parsed, (dict, list)):
                    return json.dumps(parsed) 
                else:
                    return json.dumps({"query": eff_args})  
            except (json.JSONDecodeError, ValueError):
                return json.dumps({"query": eff_args})
        else:
            return "{}"
    
    try:
        try:
            line_iterator = upstream.aiter_lines()
        except httpx.HTTPError as e:
            if verbose and vlog:
                vlog(f"Failed to start stream: {e}")
            yield b"data: [DONE]\n\n"
            return

        async for raw in line_iterator:
            try:
                if not raw:
                    continue
                line = raw if isinstance(raw, str) else raw.decode("utf-8", errors="ignore")
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
                # Connection interrupted mid-stream - end gracefully
                if verbose and vlog:
                    vlog(f"Stream interrupted: {e}")
                yield b"data: [DONE]\n\n"
                return
            kind = evt.get("type")
            if isinstance(evt.get("response"), dict) and isinstance(evt["response"].get("id"), str):
                response_id = evt["response"].get("id") or response_id

            if isinstance(kind, str) and ("web_search_call" in kind):
                try:
                    call_id = evt.get("item_id") or "ws_call"
                    if verbose and vlog:
                        try:
                            vlog(f"CM_TOOLS {kind} id={call_id} -> tool_calls(web_search)")
                        except Exception:
                            pass
                    item = evt.get('item') if isinstance(evt.get('item'), dict) else {}
                    params_dict = ws_state.setdefault(call_id, {}) if isinstance(ws_state.get(call_id), dict) else {}
                    def _merge_from(src):
                        if not isinstance(src, dict):
                            return
                        for whole in ('parameters','args','arguments','input'):
                            if isinstance(src.get(whole), dict):
                                params_dict.update(src.get(whole))
                        if isinstance(src.get('query'), str): params_dict.setdefault('query', src.get('query'))
                        if isinstance(src.get('q'), str): params_dict.setdefault('query', src.get('q'))
                        for rk in ('recency','time_range','days'):
                            if src.get(rk) is not None and rk not in params_dict: params_dict[rk] = src.get(rk)
                        for dk in ('domains','include_domains','include'):
                            if isinstance(src.get(dk), list) and 'domains' not in params_dict: params_dict['domains'] = src.get(dk)
                        for mk in ('max_results','topn','limit'):
                            if src.get(mk) is not None and 'max_results' not in params_dict: params_dict['max_results'] = src.get(mk)
                    _merge_from(item)
                    _merge_from(evt if isinstance(evt, dict) else None)
                    params = params_dict if params_dict else None
                    if isinstance(params, dict):
                        try:
                            ws_state.setdefault(call_id, {}).update(params)
                        except Exception:
                            pass
                    eff_params = ws_state.get(call_id, params if isinstance(params, (dict, list, str)) else {})
                    args_str = _serialize_tool_args(eff_params)
                    if call_id not in ws_index:
                        ws_index[call_id] = ws_next_index
                        ws_next_index += 1
                    _idx = ws_index.get(call_id, 0)
                    delta_chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": _idx,
                                            "id": call_id,
                                            "type": "function",
                                            "function": {"name": "web_search", "arguments": args_str},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(delta_chunk)}\n\n".encode("utf-8")
                    if kind.endswith(".completed") or kind.endswith(".done"):
                        finish_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                            ],
                        }
                        yield f"data: {json.dumps(finish_chunk)}\n\n".encode("utf-8")
                except Exception:
                    pass

            if kind == "response.output_text.delta":
                delta = evt.get("delta") or ""
                if compat == "think-tags" and think_open and not think_closed:
                    close_chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": "</think>"}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(close_chunk)}\n\n".encode("utf-8")
                    think_open = False
                    think_closed = True
                saw_output = True
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
            elif kind == "response.output_item.done":
                item = evt.get("item") or {}
                if isinstance(item, dict) and (item.get("type") == "function_call" or item.get("type") == "web_search_call"):
                    call_id = item.get("call_id") or item.get("id") or ""
                    name = item.get("name") or ("web_search" if item.get("type") == "web_search_call" else "")
                    raw_args = item.get("arguments") or item.get("parameters")
                    if isinstance(raw_args, dict):
                        try:
                            ws_state.setdefault(call_id, {}).update(raw_args)
                        except Exception:
                            pass
                    eff_args = ws_state.get(call_id, raw_args if isinstance(raw_args, (dict, list, str)) else {})
                    try:
                        args = _serialize_tool_args(eff_args)
                    except Exception:
                        args = "{}"
                    if item.get("type") == "web_search_call" and verbose and vlog:
                        try:
                            vlog(f"CM_TOOLS response.output_item.done web_search_call id={call_id} has_args={bool(args)}")
                        except Exception:
                            pass
                    if call_id not in ws_index:
                        ws_index[call_id] = ws_next_index
                        ws_next_index += 1
                    _idx = ws_index.get(call_id, 0)
                    if isinstance(call_id, str) and isinstance(name, str) and isinstance(args, str):
                        delta_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": _idx,
                                                "id": call_id,
                                                "type": "function",
                                                "function": {"name": name, "arguments": args},
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(delta_chunk)}\n\n".encode("utf-8")

                        finish_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                        }
                        yield f"data: {json.dumps(finish_chunk)}\n\n".encode("utf-8")
            elif kind == "response.reasoning_summary_part.added":
                if compat in ("think-tags", "o3"):
                    if saw_any_summary:
                        pending_summary_paragraph = True
                    else:
                        saw_any_summary = True
            elif kind in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
                delta_txt = evt.get("delta") or ""
                if compat == "o3":
                    if kind == "response.reasoning_summary_text.delta" and pending_summary_paragraph:
                        nl_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"reasoning": {"content": [{"type": "text", "text": "\n"}]}},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(nl_chunk)}\n\n".encode("utf-8")
                        pending_summary_paragraph = False
                    chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"reasoning": {"content": [{"type": "text", "text": delta_txt}]}},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                elif compat == "think-tags":
                    if not think_open and not think_closed:
                        open_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": "<think>"}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(open_chunk)}\n\n".encode("utf-8")
                        think_open = True
                    if think_open and not think_closed:
                        if kind == "response.reasoning_summary_text.delta" and pending_summary_paragraph:
                            nl_chunk = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{"index": 0, "delta": {"content": "\n"}, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(nl_chunk)}\n\n".encode("utf-8")
                            pending_summary_paragraph = False
                        content_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": delta_txt}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(content_chunk)}\n\n".encode("utf-8")
                else:
                    if kind == "response.reasoning_summary_text.delta":
                        chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"reasoning_summary": delta_txt, "reasoning": delta_txt},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                    else:
                        chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {"index": 0, "delta": {"reasoning": delta_txt}, "finish_reason": None}
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
            elif kind == "response.content_part.done":
                part = evt.get("part")
                if isinstance(part, dict) and part.get("type") == "output_text":
                    part_annotations = part.get("annotations")
                    if isinstance(part_annotations, list) and part_annotations:
                        ann_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"annotations": part_annotations}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(ann_chunk)}\n\n".encode("utf-8")
            elif kind == "response.output_text.done":
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                sent_stop_chunk = True
            elif isinstance(kind, str) and kind.endswith(".done"):
                pass
            elif kind == "response.failed":
                err = evt.get("response", {}).get("error", {}).get("message", "response.failed")
                chunk = {"error": {"message": err}}
                yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
            elif kind == "response.completed":
                m = _extract_usage(evt)
                if m:
                    upstream_usage = m
                if compat == "think-tags" and think_open and not think_closed:
                    close_chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": "</think>"}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(close_chunk)}\n\n".encode("utf-8")
                    think_open = False
                    think_closed = True
                if not sent_stop_chunk:
                    chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                    sent_stop_chunk = True

                if include_usage and upstream_usage:
                    try:
                        usage_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                            "usage": upstream_usage,
                        }
                        yield f"data: {json.dumps(usage_chunk)}\n\n".encode("utf-8")
                    except Exception:
                        pass
                yield b"data: [DONE]\n\n"
                break
    finally:
        await upstream.aclose()


async def sse_translate_text(
    upstream, 
    model: str, 
    created: int, 
    verbose: bool = False, 
    vlog=None, 
    *, 
    include_usage: bool = False
) -> AsyncGenerator[bytes, None]:
    response_id = "cmpl-stream"
    upstream_usage = None
    
    try:
        async for raw_line in upstream.aiter_lines():
            if not raw_line:
                continue
            line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="ignore")
            if verbose and vlog:
                vlog(line)
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):].strip()
            if not data or data == "[DONE]":
                if data == "[DONE]":
                    chunk = {
                        "id": response_id,
                        "object": "text_completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                continue
            try:
                evt = json.loads(data)
            except Exception:
                continue
            kind = evt.get("type")
            if isinstance(evt.get("response"), dict) and isinstance(evt["response"].get("id"), str):
                response_id = evt["response"].get("id") or response_id
            if kind == "response.output_text.delta":
                delta_text = evt.get("delta") or ""
                chunk = {
                    "id": response_id,
                    "object": "text_completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "text": delta_text, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
            elif kind == "response.output_text.done":
                chunk = {
                    "id": response_id,
                    "object": "text_completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
            elif kind == "response.completed":
                m = _extract_usage(evt)
                if m:
                    upstream_usage = m
                if include_usage and upstream_usage:
                    try:
                        usage_chunk = {
                            "id": response_id,
                            "object": "text_completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "text": "", "finish_reason": None}],
                            "usage": upstream_usage,
                        }
                        yield f"data: {json.dumps(usage_chunk)}\n\n".encode("utf-8")
                    except Exception:
                        pass
                yield b"data: [DONE]\n\n"
                break
    finally:
        await upstream.aclose()
