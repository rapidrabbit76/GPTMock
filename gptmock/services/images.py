"""Proxy for the ChatGPT Codex backend's direct Images API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from gptmock.core.logging import log_json
from gptmock.core.settings import Settings
from gptmock.infra.auth import get_effective_chatgpt_auth
from gptmock.services.chat import ChatCompletionError
from gptmock.services.upstream import UpstreamError, send_image_generation_request
from gptmock.services.upstream_errors import extract_upstream_error_message

logger = logging.getLogger(__name__)

_MISSING_CREDENTIALS = "Missing ChatGPT credentials. Run 'python3 gptmock.py login' first."


def _invalid_request(message: str, param: str, code: str) -> ChatCompletionError:
    return ChatCompletionError(
        message,
        status_code=400,
        error_data={"error": {"message": message, "param": param, "code": code}},
    )


def _validate_request(payload: dict[str, Any]) -> str:
    """Reject requests that must never reach authentication; return the trimmed model."""
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise _invalid_request("A non-empty model name is required", "model", "invalid_model")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise _invalid_request("A non-empty prompt is required", "prompt", "invalid_prompt")
    if payload.get("stream"):
        raise _invalid_request("streaming image generation is not supported", "stream", "unsupported_parameter")
    return model.strip()


def _upstream_body(upstream: httpx.Response) -> Any:
    try:
        return upstream.json()
    except ValueError:
        if upstream.is_success:
            message = f"Upstream returned a non-JSON image response (HTTP {upstream.status_code})"
            raise ChatCompletionError(message, status_code=502) from None
        return {"error": {"message": extract_upstream_error_message(upstream.text, status_code=upstream.status_code)}}


async def process_image_generation(
    payload: dict[str, Any],
    settings: Settings,
    http_client: httpx.AsyncClient,
) -> tuple[Any, int]:
    """Forward an Images API generation request and return the upstream JSON body and status code.

    The body is sent unchanged apart from the trimmed ``model``; image options are validated upstream.
    """
    model = _validate_request(payload)

    access_token, account_id = await get_effective_chatgpt_auth()
    if not access_token or not account_id:
        raise ChatCompletionError(
            _MISSING_CREDENTIALS,
            status_code=401,
            error_data={"error": {"message": _MISSING_CREDENTIALS}},
        )

    try:
        upstream = await send_image_generation_request(
            {**payload, "model": model},
            access_token,
            account_id,
            http_client,
            verbose=settings.verbose,
        )
    except UpstreamError as exc:
        raise ChatCompletionError(exc.message, status_code=exc.status_code) from exc

    body = _upstream_body(upstream)
    if settings.verbose and not upstream.is_success:
        log_json(f"OUT images API ERROR (HTTP {upstream.status_code})", body, logger=logger.debug)
    return body, upstream.status_code
