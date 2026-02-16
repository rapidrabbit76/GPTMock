from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from chatmock.core.dependencies import get_settings
from chatmock.core.logging import log_json
from chatmock.core.settings import Settings
from chatmock.services.chat import ChatCompletionError, process_chat_completion, process_text_completion
from chatmock.services.model_registry import get_openai_models


router = APIRouter()


async def get_http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Provide httpx AsyncClient for making upstream requests."""
    async with httpx.AsyncClient() as client:
        yield client


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    settings: Settings = Depends(get_settings),
    http_client: httpx.AsyncClient = Depends(get_http_client),
):
    """
    OpenAI-compatible chat completions endpoint.
    
    Handles both streaming and non-streaming requests.
    """
    # 1. Parse request body
    try:
        raw_body = await request.body()
        raw_text = raw_body.decode("utf-8")
        
        if settings.verbose:
            print(f"IN POST /v1/chat/completions\n{raw_text}")
        
        payload = json.loads(raw_text) if raw_text else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        error_response = {"error": {"message": "Invalid JSON body"}}
        if settings.verbose:
            log_json("OUT POST /v1/chat/completions ERROR", error_response, logger=print)
        return JSONResponse(error_response, status_code=400)
    
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
                },
            )
        else:
            # response is a dict
            return JSONResponse(response)
            
    except ChatCompletionError as e:
        error_response = e.error_data or {"error": {"message": e.message}}
        if settings.verbose:
            log_json("OUT POST /v1/chat/completions ERROR", error_response, logger=print)
        return JSONResponse(error_response, status_code=e.status_code)


@router.post("/v1/completions")
async def completions(
    request: Request,
    settings: Settings = Depends(get_settings),
    http_client: httpx.AsyncClient = Depends(get_http_client),
):
    """
    OpenAI-compatible text completions endpoint.
    """
    # 1. Parse request body
    try:
        raw_body = await request.body()
        raw_text = raw_body.decode("utf-8")
        
        if settings.verbose:
            print(f"IN POST /v1/completions\n{raw_text}")
        
        payload = json.loads(raw_text) if raw_text else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        error_response = {"error": {"message": "Invalid JSON body"}}
        if settings.verbose:
            log_json("OUT POST /v1/completions ERROR", error_response, logger=print)
        return JSONResponse(error_response, status_code=400)
    
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
                },
            )
        else:
            # response is a dict
            return JSONResponse(response)
            
    except ChatCompletionError as e:
        error_response = e.error_data or {"error": {"message": e.message}}
        if settings.verbose:
            log_json("OUT POST /v1/completions ERROR", error_response, logger=print)
        return JSONResponse(error_response, status_code=e.status_code)


@router.get("/v1/models")
async def list_models(
    settings: Settings = Depends(get_settings),
):
    """
    List available models in OpenAI format.
    """
    models = get_openai_models(
        expose_reasoning=settings.expose_reasoning_models
    )
    
    response = {
        "object": "list",
        "data": models,
    }
    
    if settings.verbose:
        log_json("OUT GET /v1/models", response, logger=print)
    
    return JSONResponse(response)
