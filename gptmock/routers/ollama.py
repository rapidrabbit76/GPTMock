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
from gptmock.schemas.requests import OllamaChatRequest, OllamaGenerateRequest, OllamaShowRequest
from gptmock.schemas.transform import convert_ollama_messages, normalize_ollama_tools
from gptmock.services.chat import ChatCompletionError, process_chat_completion
from gptmock.services.model_registry import (
    FAST_MODEL_ALIASES,
    get_model_list,
    get_ollama_models,
    resolve_upstream_model,
)

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

    reasoning_effort = ollama_payload.get("reasoning_effort")
    think = ollama_payload.get("think")
    if think is not None and not isinstance(think, (bool, str)):
        raise ChatCompletionError("Ollama think must be a boolean or string", status_code=400)
    if reasoning_effort is not None:
        openai_payload["reasoning_effort"] = reasoning_effort
    if isinstance(think, str):
        if reasoning_effort is not None:
            openai_payload["reasoning"] = {"effort": think}
        else:
            openai_payload["reasoning_effort"] = think
    elif think is False:
        openai_payload["reasoning"] = {"summary": "none"}

    service_tier = ollama_payload.get("service_tier")
    if service_tier is not None:
        openai_payload["service_tier"] = service_tier

    response_format = ollama_payload.get("format")
    if response_format == "json":
        openai_payload["response_format"] = {"type": "json_object"}
    elif isinstance(response_format, dict):
        openai_payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "ollama_response",
                "strict": True,
                "schema": response_format,
            },
        }
    elif response_format is not None:
        raise ChatCompletionError("Unsupported Ollama format", status_code=400)

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


def _ollama_policy_headers(
    ollama_payload: dict[str, Any], settings: Settings,
) -> dict[str, str]:
    options = ollama_payload.get("options")
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise ChatCompletionError("Ollama options must be an object", status_code=400)

    supplied = [f"options.{key}" for key, value in options.items() if value is not None]
    unsupported = [parameter for parameter in supplied if parameter != "options.num_predict"]
    if unsupported:
        parameters = ", ".join(unsupported)
        raise ChatCompletionError(
            f"Unsupported Ollama option(s): {parameters}",
            status_code=400,
        )

    if "options.num_predict" not in supplied:
        return {}
    if settings.output_token_policy == "reject":
        raise ChatCompletionError(
            "Unsupported parameter: options.num_predict",
            status_code=400,
        )

    logger.warning(
        "Ignoring output token limit unsupported by ChatGPT upstream: options.num_predict",
    )
    return {"X-GPTMock-Omitted-Parameters": "options.num_predict"}


async def _convert_openai_to_ollama_stream(
    response: Any, model: str,
) -> AsyncGenerator[bytes]:
    response_model = model
    service_tier: str | None = None
    done_reason: str | None = None
    terminal_seen = False
    try:
        async for sse_chunk in response:
            if not sse_chunk.startswith(b"data: "):
                continue

            json_bytes = sse_chunk[6:].strip()

            if json_bytes == b"[DONE]":
                terminal_seen = True
                done_chunk = {
                    "model": response_model,
                    "created_at": datetime.datetime.now(
                        datetime.UTC,
                    ).isoformat().replace("+00:00", "Z"),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                }
                if done_reason is not None:
                    done_chunk["done_reason"] = done_reason
                if service_tier is not None:
                    done_chunk["service_tier"] = service_tier
                yield (json.dumps(done_chunk) + "\n").encode("utf-8")
                break

            try:
                openai_chunk = json.loads(json_bytes)
                if isinstance(openai_chunk.get("error"), dict):
                    terminal_seen = True
                    yield (json.dumps({"error": openai_chunk["error"].get("message", "upstream error")}) + "\n").encode("utf-8")
                    break
                if isinstance(openai_chunk.get("model"), str) and openai_chunk["model"]:
                    response_model = openai_chunk["model"]
                if isinstance(openai_chunk.get("service_tier"), str):
                    service_tier = openai_chunk["service_tier"]
                choices = openai_chunk.get("choices", [])

                if choices:
                    finish_reason = choices[0].get("finish_reason")
                    if isinstance(finish_reason, str):
                        done_reason = finish_reason
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
                            "model": response_model,
                            "created_at": datetime.datetime.now(
                                datetime.UTC,
                            ).isoformat().replace("+00:00", "Z"),
                            "message": message,
                            "done": False,
                        }
                        if service_tier is not None:
                            ollama_chunk["service_tier"] = service_tier
                        yield (json.dumps(ollama_chunk) + "\n").encode("utf-8")
            except Exception:
                logger.debug("Failed to parse OpenAI SSE chunk JSON", exc_info=True)
                continue
        if not terminal_seen:
            yield (json.dumps({"error": "Upstream stream ended before a terminal event"}) + "\n").encode("utf-8")
    finally:
        if hasattr(response, "aclose"):
            await response.aclose()


def _convert_openai_to_ollama_response(
    response: dict[str, Any], model: str,
) -> dict[str, Any]:
    choice = response.get("choices", [{}])[0]
    source_message = choice.get("message", {})
    message = dict(source_message) if isinstance(source_message, dict) else {}
    reasoning = message.pop("reasoning_content", None) or message.pop("reasoning", None)
    if isinstance(reasoning, str) and reasoning:
        message["thinking"] = reasoning

    result = {
        "model": response.get("model") or model,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"),
        "message": message,
        "done": True,
        "done_reason": choice.get("finish_reason", "stop"),
    }
    if response.get("service_tier") is not None:
        result["service_tier"] = response["service_tier"]
    return result


async def _convert_openai_to_ollama_generate_stream(
    response: Any, model: str,
) -> AsyncGenerator[bytes]:
    async for chat_frame in _convert_openai_to_ollama_stream(response, model):
        chunk = json.loads(chat_frame)
        if "error" in chunk:
            yield chat_frame
            continue
        message = chunk.pop("message", {})
        chunk["response"] = message.get("content", "")
        thinking = message.get("thinking")
        if isinstance(thinking, str) and thinking:
            chunk["thinking"] = thinking
        yield (json.dumps(chunk) + "\n").encode("utf-8")


def _convert_openai_to_ollama_generate_response(
    response: dict[str, Any], model: str,
) -> dict[str, Any]:
    chat_response = _convert_openai_to_ollama_response(response, model)
    message = chat_response.pop("message", {})
    chat_response["response"] = message.get("content", "")
    thinking = message.get("thinking")
    if isinstance(thinking, str) and thinking:
        chat_response["thinking"] = thinking
    return chat_response


def _generate_chat_payload(ollama_payload: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    system = ollama_payload.get("system")
    if isinstance(system, str) and system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": ollama_payload.get("prompt", "")})
    return {**ollama_payload, "messages": messages}


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

    show_payload = body.model_dump()
    model = body.model.strip()
    if not model and isinstance(show_payload.get("name"), str):
        model = show_payload["name"].strip()

    if not model:
        err = {"error": "Model not found"}
        if settings.verbose:
            log_json("OUT POST /api/show", err, logger=logger.debug)
        return JSONResponse(err, status_code=400)

    available_models = set(get_model_list(expose_reasoning=settings.expose_reasoning_models))
    available_models.update(FAST_MODEL_ALIASES)
    if model not in available_models:
        err = {"error": f"Model '{model}' not found"}
        if settings.verbose:
            log_json("OUT POST /api/show", err, logger=logger.debug)
        return JSONResponse(err, status_code=404)

    upstream_model, overrides = resolve_upstream_model(model)
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
    policy_headers: dict[str, str] = {}

    if settings.verbose:
        log_json("IN POST /api/chat", ollama_payload, logger=logger.debug)

    raw_messages = ollama_payload.get("messages")

    if not isinstance(raw_messages, list) or not raw_messages:
        err = {"error": "Invalid request format"}
        if settings.verbose:
            log_json("OUT POST /api/chat", err, logger=logger.debug)
        return JSONResponse(err, status_code=400)

    # 3. Call service layer
    try:
        policy_headers = _ollama_policy_headers(ollama_payload, settings)
        openai_payload = _build_openai_payload(ollama_payload, model)
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
                headers=policy_headers,
            )
        ollama_response = _convert_openai_to_ollama_response(response, model)

        if settings.verbose:
            log_json("OUT POST /api/chat", ollama_response, logger=logger.debug)

        return JSONResponse(ollama_response, headers=policy_headers)

    except ChatCompletionError as e:
        error_response = {"error": e.message}
        if settings.verbose:
            log_json("OUT POST /api/chat ERROR", error_response, logger=logger.debug)
        return JSONResponse(
            error_response,
            status_code=e.status_code,
            headers=policy_headers,
        )


@router.post("/api/generate")
async def ollama_generate(
    body: OllamaGenerateRequest,
    settings: Settings = Depends(get_settings),
    http_client: httpx.AsyncClient = Depends(get_http_client),
):
    """Ollama-compatible text generation backed by the chat service."""
    ollama_payload = body.model_dump()
    model = body.model
    policy_headers: dict[str, str] = {}

    if settings.verbose:
        log_json("IN POST /api/generate", ollama_payload, logger=logger.debug)

    try:
        if ollama_payload.get("suffix"):
            raise ChatCompletionError("Unsupported Ollama parameter: suffix", status_code=400)
        if ollama_payload.get("raw") is True:
            raise ChatCompletionError("Unsupported Ollama parameter: raw", status_code=400)
        if ollama_payload.get("template"):
            raise ChatCompletionError("Unsupported Ollama parameter: template", status_code=400)
        if ollama_payload.get("context") is not None:
            raise ChatCompletionError("Unsupported Ollama parameter: context", status_code=400)

        policy_headers = _ollama_policy_headers(ollama_payload, settings)
        chat_payload = _generate_chat_payload(ollama_payload)
        openai_payload = _build_openai_payload(chat_payload, model)
        response, is_streaming = await process_chat_completion(
            payload=openai_payload,
            settings=settings,
            http_client=http_client,
        )

        if is_streaming:
            return StreamingResponse(
                _convert_openai_to_ollama_generate_stream(response, model),
                media_type="application/x-ndjson",
                headers=policy_headers,
            )
        ollama_response = _convert_openai_to_ollama_generate_response(response, model)
        if settings.verbose:
            log_json("OUT POST /api/generate", ollama_response, logger=logger.debug)
        return JSONResponse(ollama_response, headers=policy_headers)
    except ChatCompletionError as error:
        error_response = {"error": error.message}
        if settings.verbose:
            log_json("OUT POST /api/generate ERROR", error_response, logger=logger.debug)
        return JSONResponse(
            error_response,
            status_code=error.status_code,
            headers=policy_headers,
        )
