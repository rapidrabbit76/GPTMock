from __future__ import annotations

import datetime
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from gptmock.core.dependencies import get_http_client, get_settings
from gptmock.core.logging import log_json
from gptmock.core.settings import Settings
from gptmock.schemas.requests import OllamaChatRequest, OllamaShowRequest
from gptmock.schemas.transform import convert_ollama_messages, normalize_ollama_tools
from gptmock.services.chat import ChatCompletionError, process_chat_completion
from gptmock.services.model_registry import get_model_list, get_ollama_models, resolve_upstream_model

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_openai_payload(ollama_payload: dict[str, Any], model: str) -> dict[str, Any]:
    raw_messages = ollama_payload.get("messages")
    messages = convert_ollama_messages(
        raw_messages,
        ollama_payload.get("images")
        if isinstance(ollama_payload.get("images"), list)
        else None,
    )

    stream_req = ollama_payload.get("stream")
    if stream_req is None:
        stream_req = True
    stream_req = bool(stream_req)

    openai_payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream_req,
    }

    tools_req = (
        ollama_payload.get("tools")
        if isinstance(ollama_payload.get("tools"), list)
        else []
    )
    if tools_req:
        openai_tools = normalize_ollama_tools(tools_req)
        if openai_tools:
            openai_payload["tools"] = openai_tools

    tool_choice = ollama_payload.get("tool_choice", "auto")
    if tool_choice in ("auto", "none", "required"):
        openai_payload["tool_choice"] = tool_choice

    parallel_tool_calls = bool(ollama_payload.get("parallel_tool_calls", False))
    if parallel_tool_calls:
        openai_payload["parallel_tool_calls"] = parallel_tool_calls

    responses_tools_payload = (
        ollama_payload.get("responses_tools")
        if isinstance(ollama_payload.get("responses_tools"), list)
        else []
    )
    if responses_tools_payload:
        openai_payload["responses_tools"] = responses_tools_payload

    responses_tool_choice = ollama_payload.get("responses_tool_choice")
    if isinstance(responses_tool_choice, str) and responses_tool_choice in (
        "auto",
        "none",
    ):
        openai_payload["responses_tool_choice"] = responses_tool_choice

    return openai_payload


async def _convert_openai_to_ollama_stream(
    response: Any, model: str,
) -> AsyncGenerator[bytes]:
    try:
        async for sse_chunk in response:
            if not sse_chunk.startswith(b"data: "):
                continue

            json_bytes = sse_chunk[6:].strip()

            if json_bytes == b"[DONE]":
                done_chunk = {
                    "model": model,
                    "created_at": datetime.datetime.now(
                        datetime.UTC,
                    ).isoformat().replace("+00:00", "Z"),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                }
                yield (json.dumps(done_chunk) + "\n").encode("utf-8")
                break

            try:
                openai_chunk = json.loads(json_bytes)
                if isinstance(openai_chunk.get("error"), dict):
                    yield (json.dumps({"error": openai_chunk["error"].get("message", "upstream error")}) + "\n").encode("utf-8")
                    break
                choices = openai_chunk.get("choices", [])

                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    tool_calls = delta.get("tool_calls")

                    if content or reasoning or tool_calls:
                        message: dict[str, Any] = {
                            "role": "assistant",
                            "content": content,
                        }
                        if isinstance(reasoning, str) and reasoning:
                            message["thinking"] = reasoning
                        if isinstance(tool_calls, list) and tool_calls:
                            message["tool_calls"] = tool_calls
                        ollama_chunk = {
                            "model": model,
                            "created_at": datetime.datetime.now(
                                datetime.UTC,
                            ).isoformat().replace("+00:00", "Z"),
                            "message": message,
                            "done": False,
                        }
                        yield (json.dumps(ollama_chunk) + "\n").encode("utf-8")
            except Exception:
                logger.debug("Failed to parse OpenAI SSE chunk JSON", exc_info=True)
                continue
    finally:
        if hasattr(response, "aclose"):
            await response.aclose()


def _convert_openai_to_ollama_response(
    response: dict[str, Any], model: str,
) -> dict[str, Any]:
    choice = response.get("choices", [{}])[0]
    message = choice.get("message", {})

    return {
        "model": model,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"),
        "message": message,
        "done": True,
        "done_reason": choice.get("finish_reason", "stop"),
    }


@router.get("/api/version")
async def ollama_version(
    settings: Settings = Depends(get_settings),
):
    """Return Ollama version."""
    if settings.verbose:
        logger.debug("IN GET /api/version")

    version = settings.ollama_version
    payload = {"version": version}

    if settings.verbose:
        log_json("OUT GET /api/version", payload, logger=logger.debug)

    return JSONResponse(payload)


@router.get("/api/tags")
async def ollama_tags(
    settings: Settings = Depends(get_settings),
):
    """List available models in Ollama format."""
    if settings.verbose:
        logger.debug("IN GET /api/tags")

    models = get_ollama_models(expose_reasoning=settings.expose_reasoning_models)

    payload = {"models": models}

    if settings.verbose:
        log_json("OUT GET /api/tags", payload, logger=logger.debug)

    return JSONResponse(payload)


@router.post("/api/show")
async def ollama_show(
    body: OllamaShowRequest,
    settings: Settings = Depends(get_settings),
):
    """Show model details."""
    if settings.verbose:
        log_json("IN POST /api/show", body.model_dump(), logger=logger.debug)

    if not body.model.strip():
        err = {"error": "Model not found"}
        if settings.verbose:
            log_json("OUT POST /api/show", err, logger=logger.debug)
        return JSONResponse(err, status_code=400)

    available_models = set(get_model_list(expose_reasoning=settings.expose_reasoning_models))
    if body.model not in available_models:
        err = {"error": f"Model '{body.model}' not found"}
        if settings.verbose:
            log_json("OUT POST /api/show", err, logger=logger.debug)
        return JSONResponse(err, status_code=404)

    upstream_model, overrides = resolve_upstream_model(body.model)
    response = {
        "details": {
            "parent_model": "",
            "format": "remote",
            "family": "openai",
            "families": ["openai"],
        },
        "model_info": {
            "gptmock.remote": True,
            "gptmock.upstream_model": upstream_model,
            "gptmock.request_overrides": overrides,
        },
        "capabilities": ["completion", "tools", "thinking"],
    }

    if settings.verbose:
        log_json("OUT POST /api/show", response, logger=logger.debug)

    return JSONResponse(response)


@router.post("/api/chat")
async def ollama_chat(
    body: OllamaChatRequest,
    settings: Settings = Depends(get_settings),
    http_client: httpx.AsyncClient = Depends(get_http_client),
):
    """Ollama-compatible chat endpoint.

    Converts Ollama format → OpenAI format → calls service → converts back to Ollama ndjson.
    """
    ollama_payload = body.model_dump()
    model = body.model

    if settings.verbose:
        log_json("IN POST /api/chat", ollama_payload, logger=logger.debug)

    raw_messages = ollama_payload.get("messages")

    if not isinstance(raw_messages, list) or not raw_messages:
        err = {"error": "Invalid request format"}
        if settings.verbose:
            log_json("OUT POST /api/chat", err, logger=logger.debug)
        return JSONResponse(err, status_code=400)

    openai_payload = _build_openai_payload(ollama_payload, model)

    # 3. Call service layer
    try:
        response, is_streaming = await process_chat_completion(
            payload=openai_payload,
            settings=settings,
            http_client=http_client,
        )

        # 4. Convert response to Ollama format
        if is_streaming:
            if settings.verbose:
                logger.debug("OUT POST /api/chat (streaming response)")

            return StreamingResponse(
                _convert_openai_to_ollama_stream(response, model),
                media_type="application/x-ndjson",
            )
        ollama_response = _convert_openai_to_ollama_response(response, model)

        if settings.verbose:
            log_json("OUT POST /api/chat", ollama_response, logger=logger.debug)

        return JSONResponse(ollama_response)

    except ChatCompletionError as e:
        error_response = {"error": e.message}
        if settings.verbose:
            log_json("OUT POST /api/chat ERROR", error_response, logger=logger.debug)
        return JSONResponse(error_response, status_code=e.status_code)
