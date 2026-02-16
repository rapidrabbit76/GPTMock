from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import httpx

from chatmock.core.constants import CHATGPT_RESPONSES_URL
from chatmock.core.logging import log_json
from chatmock.core.settings import Settings
from chatmock.infra.auth import get_effective_chatgpt_auth
from chatmock.infra.sse import sse_translate_chat, sse_translate_text
from chatmock.infra.session import ensure_session_id
from chatmock.schemas.messages import convert_chat_messages_to_responses_input, convert_tools_chat_to_responses
from chatmock.services.model_registry import get_instructions_for_model, normalize_model_name
from chatmock.services.reasoning import (
    allowed_efforts_for_model,
    apply_reasoning_to_message,
    build_reasoning_param,
    extract_reasoning_from_model_name,
)


class ChatCompletionError(Exception):
    """Exception raised during chat completion processing."""
    def __init__(self, message: str, status_code: int = 500, error_data: Dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_data = error_data or {}


async def _call_upstream(
    model: str,
    input_items: List[Dict[str, Any]],
    access_token: str,
    account_id: str,
    session_id: str,
    http_client: httpx.AsyncClient,
    settings: Settings,
    *,
    instructions: str | None = None,
    tools: List[Dict[str, Any]] | None = None,
    tool_choice: Any | None = None,
    parallel_tool_calls: bool = False,
    reasoning_param: Dict[str, Any] | None = None,
) -> httpx.Response:
    """Call ChatGPT Responses API with upstream request."""
    include: List[str] = []
    if isinstance(reasoning_param, dict):
        include.append("reasoning.encrypted_content")

    responses_payload = {
        "model": model,
        "instructions": instructions if isinstance(instructions, str) and instructions.strip() else instructions,
        "input": input_items,
        "tools": tools or [],
        "tool_choice": tool_choice if tool_choice in ("auto", "none") or isinstance(tool_choice, dict) else "auto",
        "parallel_tool_calls": bool(parallel_tool_calls),
        "store": False,
        "stream": True,
        "prompt_cache_key": session_id,
    }
    if include:
        responses_payload["include"] = include

    if reasoning_param is not None:
        responses_payload["reasoning"] = reasoning_param

    if settings.verbose:
        log_json("OUTBOUND >> ChatGPT Responses API payload", responses_payload, logger=print)

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
            json=responses_payload,
            timeout=600.0,
        )
        return upstream
    except httpx.RequestError as e:
        raise ChatCompletionError(
            f"Upstream ChatGPT request failed: {e}",
            status_code=502,
        ) from e


def _extract_usage(evt: Dict[str, Any]) -> Dict[str, int] | None:
    """Extract usage statistics from upstream event (for non-streaming)."""
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


async def process_chat_completion(
    payload: Dict[str, Any],
    settings: Settings,
    http_client: httpx.AsyncClient,
    *,
    client_session_id: str | None = None,
    is_stream: bool | None = None,
) -> tuple[Any, bool]:
    """
    Process chat completion request.
    
    Args:
        payload: Request payload (OpenAI or Ollama format)
        settings: Application settings
        http_client: Async HTTP client
        client_session_id: Optional session ID from request headers
        is_stream: Override stream setting (None = use payload)
    
    Returns:
        Tuple of (response_generator_or_dict, is_streaming)
    
    Raises:
        ChatCompletionError: On processing errors
    """
    # 1. Extract and validate request parameters
    requested_model = payload.get("model")
    messages = payload.get("messages")
    
    # Handle alternative message fields
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

    # 2. Move system messages to user role (ChatGPT doesn't support system role)
    if isinstance(messages, list):
        sys_idx = next((i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "system"), None)
        if isinstance(sys_idx, int):
            sys_msg = messages.pop(sys_idx)
            content = sys_msg.get("content") if isinstance(sys_msg, dict) else ""
            messages.insert(0, {"role": "user", "content": content})

    # 3. Determine streaming mode
    if is_stream is None:
        is_stream = bool(payload.get("stream", False))
    
    stream_options = payload.get("stream_options") if isinstance(payload.get("stream_options"), dict) else {}
    include_usage = bool(stream_options.get("include_usage", False))

    # 4. Normalize model name
    model = normalize_model_name(requested_model, settings.debug_model)

    # 5. Build reasoning parameters
    model_reasoning = extract_reasoning_from_model_name(requested_model)
    reasoning_overrides = payload.get("reasoning") if isinstance(payload.get("reasoning"), dict) else model_reasoning
    reasoning_param = build_reasoning_param(
        settings.reasoning_effort,
        settings.reasoning_summary,
        reasoning_overrides,
        allowed_efforts=allowed_efforts_for_model(model),
    )

    # 6. Get system instructions
    instructions = get_instructions_for_model(
        model,
        settings.base_instructions,
        settings.gpt5_codex_instructions,
    )

    # 7. Process tools
    tools_responses = convert_tools_chat_to_responses(payload.get("tools"))
    tool_choice = payload.get("tool_choice", "auto")
    parallel_tool_calls = bool(payload.get("parallel_tool_calls", False))
    
    # Handle responses_tools (web_search, etc.)
    extra_tools: List[Dict[str, Any]] = []
    had_responses_tools = False
    responses_tools_payload = payload.get("responses_tools") if isinstance(payload.get("responses_tools"), list) else []
    
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
                        }
                    },
                )
            extra_tools.append(_t)

        if not extra_tools and settings.default_web_search:
            responses_tool_choice = payload.get("responses_tool_choice")
            if not (isinstance(responses_tool_choice, str) and responses_tool_choice == "none"):
                extra_tools = [{"type": "web_search"}]

        if extra_tools:
            MAX_TOOLS_BYTES = 32768
            try:
                size = len(json.dumps(extra_tools))
            except Exception:
                size = 0
            if size > MAX_TOOLS_BYTES:
                raise ChatCompletionError(
                    "responses_tools too large",
                    status_code=400,
                    error_data={"error": {"message": "responses_tools too large", "code": "RESPONSES_TOOLS_TOO_LARGE"}},
                )
            had_responses_tools = True
            tools_responses = (tools_responses or []) + extra_tools

    responses_tool_choice = payload.get("responses_tool_choice")
    if isinstance(responses_tool_choice, str) and responses_tool_choice in ("auto", "none"):
        tool_choice = responses_tool_choice

    # 8. Convert messages to upstream format
    input_items = convert_chat_messages_to_responses_input(messages)
    if not input_items and isinstance(payload.get("prompt"), str) and payload.get("prompt").strip():
        input_items = [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": payload.get("prompt")}]}
        ]

    # 9. Get auth credentials
    access_token, account_id = await get_effective_chatgpt_auth()
    if not access_token or not account_id:
        raise ChatCompletionError(
            "Missing ChatGPT credentials. Run 'python3 chatmock.py login' first.",
            status_code=401,
            error_data={"error": {"message": "Missing ChatGPT credentials. Run 'python3 chatmock.py login' first."}},
        )

    # 10. Get or create session ID
    session_id = ensure_session_id(instructions, input_items, client_session_id)

    # 11. Call upstream
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
            tools=tools_responses,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            reasoning_param=reasoning_param,
        )
    except ChatCompletionError:
        raise

    # 12. Handle upstream errors
    if upstream.status_code >= 400:
        try:
            err_body = upstream.json() if upstream.content else {"raw": upstream.text}
        except Exception:
            err_body = {"raw": upstream.text}

        # Retry without extra tools if they were rejected
        if had_responses_tools:
            if settings.verbose:
                print("[Passthrough] Upstream rejected tools; retrying without extra tools (args redacted)")
            base_tools_only = convert_tools_chat_to_responses(payload.get("tools"))
            safe_choice = payload.get("tool_choice", "auto")
            
            try:
                upstream2 = await _call_upstream(
                    model=model,
                    input_items=input_items,
                    access_token=access_token,
                    account_id=account_id,
                    session_id=session_id,
                    http_client=http_client,
                    settings=settings,
                    instructions=settings.base_instructions,
                    tools=base_tools_only,
                    tool_choice=safe_choice,
                    parallel_tool_calls=parallel_tool_calls,
                    reasoning_param=reasoning_param,
                )
                
                if upstream2.status_code < 400:
                    upstream = upstream2
                else:
                    raise ChatCompletionError(
                        (err_body.get("error", {}) or {}).get("message", "Upstream error"),
                        status_code=upstream2.status_code,
                        error_data={
                            "error": {
                                "message": (err_body.get("error", {}) or {}).get("message", "Upstream error"),
                                "code": "RESPONSES_TOOLS_REJECTED",
                            }
                        },
                    )
            except ChatCompletionError:
                raise
        else:
            if settings.verbose:
                print("Upstream error status=", upstream.status_code)
            raise ChatCompletionError(
                (err_body.get("error", {}) or {}).get("message", "Upstream error"),
                status_code=upstream.status_code,
                error_data={"error": {"message": (err_body.get("error", {}) or {}).get("message", "Upstream error")}},
            )

    # 13. Return streaming or non-streaming response
    created = int(time.time())
    
    if is_stream:
        # Return async generator for streaming
        if settings.verbose:
            print(f"OUT chat completion (streaming response, model={requested_model or model})")
        
        stream_iter = sse_translate_chat(
            upstream,
            requested_model or model,
            created,
            verbose=settings.verbose_obfuscation,
            vlog=print if settings.verbose_obfuscation else None,
            reasoning_compat=settings.reasoning_compat,
            include_usage=include_usage,
        )
        return stream_iter, True
    else:
        # Collect full response for non-streaming
        full_text = ""
        reasoning_summary_text = ""
        reasoning_full_text = ""
        response_id = "chatcmpl"
        tool_calls: List[Dict[str, Any]] = []
        error_message: str | None = None
        usage_obj: Dict[str, int] | None = None

        try:
            async for raw in upstream.aiter_lines():
                if not raw:
                    continue
                line = raw if isinstance(raw, str) else raw.decode("utf-8", errors="ignore")
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break
                try:
                    evt = json.loads(data)
                except Exception:
                    continue
                
                kind = evt.get("type")
                mu = _extract_usage(evt)
                if mu:
                    usage_obj = mu
                if isinstance(evt.get("response"), dict) and isinstance(evt["response"].get("id"), str):
                    response_id = evt["response"].get("id") or response_id
                
                if kind == "response.output_text.delta":
                    full_text += evt.get("delta") or ""
                elif kind == "response.reasoning_summary_text.delta":
                    reasoning_summary_text += evt.get("delta") or ""
                elif kind == "response.reasoning_text.delta":
                    reasoning_full_text += evt.get("delta") or ""
                elif kind == "response.output_item.done":
                    item = evt.get("item") or {}
                    if isinstance(item, dict) and item.get("type") == "function_call":
                        call_id = item.get("call_id") or item.get("id") or ""
                        name = item.get("name") or ""
                        args = item.get("arguments") or ""
                        if isinstance(call_id, str) and isinstance(name, str) and isinstance(args, str):
                            tool_calls.append(
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": name, "arguments": args},
                                }
                            )
                elif kind == "response.failed":
                    error_message = evt.get("response", {}).get("error", {}).get("message", "response.failed")
                elif kind == "response.completed":
                    break
        finally:
            await upstream.aclose()

        if error_message:
            raise ChatCompletionError(
                error_message,
                status_code=502,
                error_data={"error": {"message": error_message}},
            )

        message: Dict[str, Any] = {"role": "assistant", "content": full_text if full_text else None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        message = apply_reasoning_to_message(
            message,
            reasoning_summary_text,
            reasoning_full_text,
            settings.reasoning_compat,
        )
        
        completion = {
            "id": response_id or "chatcmpl",
            "object": "chat.completion",
            "created": created,
            "model": requested_model or model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "stop",
                }
            ],
            **({"usage": usage_obj} if usage_obj else {}),
        }
        
        if settings.verbose:
            log_json("OUT chat completion", completion, logger=print)
        
        return completion, False


async def process_text_completion(
    payload: Dict[str, Any],
    settings: Settings,
    http_client: httpx.AsyncClient,
    *,
    client_session_id: str | None = None,
) -> tuple[Any, bool]:
    """
    Process text completion request (/v1/completions).
    
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
    # 1. Extract request parameters
    requested_model = payload.get("model")
    prompt = payload.get("prompt")
    
    if isinstance(prompt, list):
        prompt = "".join([p if isinstance(p, str) else "" for p in prompt])
    if not isinstance(prompt, str):
        prompt = payload.get("suffix") or ""
    
    stream_req = bool(payload.get("stream", False))
    stream_options = payload.get("stream_options") if isinstance(payload.get("stream_options"), dict) else {}
    include_usage = bool(stream_options.get("include_usage", False))

    # 2. Normalize model
    model = normalize_model_name(requested_model, settings.debug_model)

    # 3. Convert to messages format
    messages = [{"role": "user", "content": prompt or ""}]
    input_items = convert_chat_messages_to_responses_input(messages)

    # 4. Build reasoning parameters
    model_reasoning = extract_reasoning_from_model_name(requested_model)
    reasoning_overrides = payload.get("reasoning") if isinstance(payload.get("reasoning"), dict) else model_reasoning
    reasoning_param = build_reasoning_param(
        settings.reasoning_effort,
        settings.reasoning_summary,
        reasoning_overrides,
        allowed_efforts=allowed_efforts_for_model(model),
    )

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
            "Missing ChatGPT credentials. Run 'python3 chatmock.py login' first.",
            status_code=401,
            error_data={"error": {"message": "Missing ChatGPT credentials. Run 'python3 chatmock.py login' first."}},
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
            err_body = upstream.json() if upstream.content else {"raw": upstream.text}
        except Exception:
            err_body = {"raw": upstream.text}
        raise ChatCompletionError(
            (err_body.get("error", {}) or {}).get("message", "Upstream error"),
            status_code=upstream.status_code,
            error_data={"error": {"message": (err_body.get("error", {}) or {}).get("message", "Upstream error")}},
        )

    # 10. Return streaming or non-streaming response
    created = int(time.time())
    
    if stream_req:
        if settings.verbose:
            print(f"OUT text completion (streaming response, model={requested_model or model})")
        
        stream_iter = sse_translate_text(
            upstream,
            requested_model or model,
            created,
            verbose=settings.verbose_obfuscation,
            vlog=print if settings.verbose_obfuscation else None,
            include_usage=include_usage,
        )
        return stream_iter, True
    else:
        # Collect full response
        full_text = ""
        response_id = "cmpl"
        usage_obj: Dict[str, int] | None = None
        
        try:
            async for raw_line in upstream.aiter_lines():
                if not raw_line:
                    continue
                line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="ignore")
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):].strip()
                if not data or data == "[DONE]":
                    if data == "[DONE]":
                        break
                    continue
                try:
                    evt = json.loads(data)
                except Exception:
                    continue
                
                if isinstance(evt.get("response"), dict) and isinstance(evt["response"].get("id"), str):
                    response_id = evt["response"].get("id") or response_id
                mu = _extract_usage(evt)
                if mu:
                    usage_obj = mu
                kind = evt.get("type")
                if kind == "response.output_text.delta":
                    full_text += evt.get("delta") or ""
                elif kind == "response.completed":
                    break
        finally:
            await upstream.aclose()

        completion = {
            "id": response_id or "cmpl",
            "object": "text_completion",
            "created": created,
            "model": requested_model or model,
            "choices": [
                {"index": 0, "text": full_text, "finish_reason": "stop", "logprobs": None}
            ],
            **({"usage": usage_obj} if usage_obj else {}),
        }
        
        if settings.verbose:
            log_json("OUT text completion", completion, logger=print)
        
        return completion, False
