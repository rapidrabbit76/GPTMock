"""Shared HTTP mechanics for calling the ChatGPT Responses API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from gptmock.core.constants import CHATGPT_RESPONSES_URL
from gptmock.core.logging import log_json

logger = logging.getLogger(__name__)


def _adapt_astra_system_messages(payload: dict[str, Any]) -> dict[str, Any]:
    """Carry text system instructions into Astra's supported instructions field."""
    if payload.get("model") != "gpt-6-astra":
        return payload
    input_items = payload.get("input")
    if not isinstance(input_items, list):
        return payload
    instructions = payload.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        return payload
    system_texts: list[str] = []
    remaining: list[Any] = []
    for item in input_items:
        if not isinstance(item, dict) or item.get("role") != "system" or item.get("type", "message") != "message":
            remaining.append(item)
            continue
        content = item.get("content")
        if isinstance(content, str):
            system_texts.append(content)
        elif isinstance(content, list) and all(
            isinstance(part, dict)
            and part.get("type") in {"input_text", "text"}
            and isinstance(part.get("text"), str)
            for part in content
        ):
            system_texts.extend(part["text"] for part in content)
        else:
            # Preserve unsupported content for the provider to validate; never drop it.
            return payload
    if not system_texts:
        return payload
    return {
        **payload,
        "instructions": "\n\n".join(text for text in [instructions or "", *system_texts] if text),
        "input": remaining,
    }


class UpstreamError(Exception):
    """Raised when the upstream ChatGPT request fails at the HTTP level."""

    def __init__(
        self,
        message: str,
        status_code: int = 502,
        error_data: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_data = error_data or {}


async def send_upstream_request(
    payload: dict[str, Any],
    access_token: str,
    account_id: str,
    session_id: str,
    http_client: httpx.AsyncClient,
    *,
    verbose: bool = False,
) -> httpx.Response:
    """Build and send a streaming POST to the ChatGPT Responses API.

    Parameters
    ----------
    payload:
        Fully-constructed JSON body (each service builds its own).
    access_token, account_id, session_id:
        Auth / session values for the request headers.
    http_client:
        The lifespan-managed ``httpx.AsyncClient``.
    verbose:
        When *True*, log the outbound payload via :func:`log_json`.

    Returns
    -------
    httpx.Response
        A **streaming** response whose body has not been consumed yet.

    Raises
    ------
    UpstreamError
        If the HTTP request itself fails (network / timeout).
    """
    payload = _adapt_astra_system_messages(payload)
    if verbose:
        log_json("OUTBOUND >> ChatGPT Responses API payload", payload, logger=logger.debug)

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "session_id": session_id,
    }

    try:
        req = http_client.build_request(
            "POST",
            CHATGPT_RESPONSES_URL,
            headers=headers,
            json=payload,
            timeout=600.0,
        )
        return await http_client.send(req, stream=True)
    except httpx.RequestError as e:
        raise UpstreamError(
            f"Upstream ChatGPT request failed: {e}",
            status_code=502,
        ) from e
