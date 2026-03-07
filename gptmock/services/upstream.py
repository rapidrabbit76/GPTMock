"""Shared HTTP mechanics for calling the ChatGPT Responses API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from gptmock.core.constants import CHATGPT_RESPONSES_URL
from gptmock.core.logging import log_json

logger = logging.getLogger(__name__)


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
