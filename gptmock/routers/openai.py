from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gptmock.core.dependencies import get_http_client, get_settings
from gptmock.core.logging import log_json
from gptmock.core.settings import Settings
from gptmock.schemas.requests import (
    ChatCompletionRequest,
    ResponsesCreateRequest,
    TextCompletionRequest,
)
from gptmock.services.chat import (
    ChatCompletionError,
    process_chat_completion,
    process_text_completion,
    supplied_parameters,
)
from gptmock.services.model_registry import get_openai_models
from gptmock.services.responses import process_responses_api

logger = logging.getLogger(__name__)

router = APIRouter()


def _output_token_headers(
    payload: dict[str, object],
    settings: Settings,
    parameter_names: tuple[str, ...],
) -> dict[str, str]:
    """Expose output limits intentionally omitted from the upstream request."""
    if settings.output_token_policy != "omit":
        return {}
    omitted = supplied_parameters(payload, parameter_names)
    if not omitted:
        return {}
    return {"X-GPTMock-Omitted-Parameters": ",".join(omitted)}


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    settings: Settings = Depends(get_settings),
    http_client: httpx.AsyncClient = Depends(get_http_client),
):
    """OpenAI-compatible chat completions endpoint.

    Handles both streaming and non-streaming requests.
    """
    payload = body.model_dump()
    policy_headers = _output_token_headers(
        payload,
        settings,
        ("max_completion_tokens", "max_tokens", "max_output_tokens"),
    )

    if settings.verbose:
        log_json("IN POST /v1/chat/completions", payload, logger=logger.debug)

    # 2. Call service layer
    try:
        response, is_streaming = await process_chat_completion(
            payload=payload,
            settings=settings,
            http_client=http_client,
        )

        # 3. Return appropriate response type
        if is_streaming:
            # response is an async generator
            return StreamingResponse(
                response,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    **policy_headers,
                },
            )
        # response is a dict
        return JSONResponse(response, headers=policy_headers)

    except ChatCompletionError as e:
        error_response = e.error_data or {"error": {"message": e.message}}
        if settings.verbose:
            log_json(
                "OUT POST /v1/chat/completions ERROR",
                error_response,
                logger=logger.debug,
            )
        return JSONResponse(error_response, status_code=e.status_code, headers=policy_headers)


@router.post("/v1/completions")
async def completions(
    body: TextCompletionRequest,
    settings: Settings = Depends(get_settings),
    http_client: httpx.AsyncClient = Depends(get_http_client),
):
    """OpenAI-compatible text completions endpoint.
    """
    payload = body.model_dump()
    policy_headers = _output_token_headers(
        payload,
        settings,
        ("max_tokens", "max_completion_tokens", "max_output_tokens"),
    )

    if settings.verbose:
        log_json("IN POST /v1/completions", payload, logger=logger.debug)

    # 2. Call service layer
    try:
        response, is_streaming = await process_text_completion(
            payload=payload,
            settings=settings,
            http_client=http_client,
        )

        # 3. Return appropriate response type
        if is_streaming:
            # response is an async generator
            return StreamingResponse(
                response,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    **policy_headers,
                },
            )
        # response is a dict
        return JSONResponse(response, headers=policy_headers)

    except ChatCompletionError as e:
        error_response = e.error_data or {"error": {"message": e.message}}
        if settings.verbose:
            log_json(
                "OUT POST /v1/completions ERROR", error_response, logger=logger.debug,
            )
        return JSONResponse(error_response, status_code=e.status_code, headers=policy_headers)


@router.post("/v1/responses")
async def responses_create(
    request: Request,
    body: ResponsesCreateRequest,
    settings: Settings = Depends(get_settings),
    http_client: httpx.AsyncClient = Depends(get_http_client),
):
    payload = body.model_dump()
    policy_headers = _output_token_headers(payload, settings, ("max_output_tokens",))

    if settings.verbose:
        log_json("IN POST /v1/responses", payload, logger=logger.debug)

    try:
        response, is_streaming = await process_responses_api(
            payload=payload,
            settings=settings,
            http_client=http_client,
            client_session_id=request.headers.get("session_id"),
        )

        if is_streaming:
            return StreamingResponse(
                response,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    **policy_headers,
                },
            )

        return JSONResponse(response, headers=policy_headers)

    except ChatCompletionError as e:
        error_response = e.error_data or {"error": {"message": e.message}}
        if settings.verbose:
            log_json(
                "OUT POST /v1/responses ERROR", error_response, logger=logger.debug,
            )
        return JSONResponse(error_response, status_code=e.status_code, headers=policy_headers)


@router.get("/v1/models")
async def list_models(
    settings: Settings = Depends(get_settings),
):
    """List available models in OpenAI format."""
    models = get_openai_models(
        expose_reasoning=settings.expose_reasoning_models,
        default_effort=settings.reasoning_effort,
    )

    response = {
        "object": "list",
        "data": models,
    }

    if settings.verbose:
        log_json("OUT GET /v1/models", response, logger=logger.debug)

    return JSONResponse(response)
