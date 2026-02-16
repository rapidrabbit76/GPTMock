from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import httpx

from chatmock.core.constants import CHATGPT_RESPONSES_URL
from chatmock.infra.auth import get_effective_chatgpt_auth
from chatmock.infra.session import ensure_session_id


def _log_json(prefix: str, payload: Any) -> None:
    try:
        print(f"{prefix}\n{json.dumps(payload, indent=2, ensure_ascii=False)}")
    except Exception:
        try:
            print(f"{prefix}\n{payload}")
        except Exception:
            pass


def normalize_model_name(name: str | None, debug_model: str | None = None) -> str:
    if isinstance(debug_model, str) and debug_model.strip():
        return debug_model.strip()
    if not isinstance(name, str) or not name.strip():
        return "gpt-5"
    base = name.split(":", 1)[0].strip()
    for sep in ("-", "_"):
        lowered = base.lower()
        for effort in ("minimal", "low", "medium", "high", "xhigh"):
            suffix = f"{sep}{effort}"
            if lowered.endswith(suffix):
                base = base[: -len(suffix)]
                break
    mapping = {
        "gpt5": "gpt-5",
        "gpt-5-latest": "gpt-5",
        "gpt-5": "gpt-5",
        "gpt-5.1": "gpt-5.1",
        "gpt5.2": "gpt-5.2",
        "gpt-5.2": "gpt-5.2",
        "gpt-5.2-latest": "gpt-5.2",
        "gpt5.2-codex": "gpt-5.2-codex",
        "gpt-5.2-codex": "gpt-5.2-codex",
        "gpt-5.2-codex-latest": "gpt-5.2-codex",
        "gpt5-codex": "gpt-5-codex",
        "gpt-5-codex": "gpt-5-codex",
        "gpt-5-codex-latest": "gpt-5-codex",
        "gpt-5.1-codex": "gpt-5.1-codex",
        "gpt-5.1-codex-max": "gpt-5.1-codex-max",
        "codex": "codex-mini-latest",
        "codex-mini": "codex-mini-latest",
        "codex-mini-latest": "codex-mini-latest",
        "gpt-5.1-codex-mini": "gpt-5.1-codex-mini",
    }
    return mapping.get(base, base)


async def start_upstream_request(
    model: str,
    input_items: List[Dict[str, Any]],
    *,
    instructions: str | None = None,
    tools: List[Dict[str, Any]] | None = None,
    tool_choice: Any | None = None,
    parallel_tool_calls: bool = False,
    reasoning_param: Dict[str, Any] | None = None,
    client_session_id: str | None = None,
    verbose: bool = False,
) -> Tuple[Any, Dict[str, Any] | None]:
    """
    Start an upstream request to ChatGPT Responses API.
    
    Returns:
        Tuple of (upstream_response, error_dict)
        - If successful: (httpx.Response streaming context, None)
        - If error: (None, error_dict with status code and message)
    """
    access_token, account_id = await get_effective_chatgpt_auth()
    if not access_token or not account_id:
        return None, {
            "status": 401,
            "body": {
                "error": {
                    "message": "Missing ChatGPT credentials. Run 'python3 chatmock.py login' first.",
                }
            },
        }

    include: List[str] = []
    if isinstance(reasoning_param, dict):
        include.append("reasoning.encrypted_content")

    session_id = ensure_session_id(instructions, input_items, client_session_id)

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

    if verbose:
        _log_json("OUTBOUND >> ChatGPT Responses API payload", responses_payload)

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "session_id": session_id,
    }

    try:
        client = httpx.AsyncClient(timeout=600.0)
        response = await client.post(
            CHATGPT_RESPONSES_URL,
            headers=headers,
            json=responses_payload,
        )
        # Return the client along with response so it stays open for streaming
        # The caller should close both when done
        response._client = client
        return response, None
    except httpx.RequestError as e:
        return None, {
            "status": 502,
            "body": {"error": {"message": f"Upstream ChatGPT request failed: {e}"}},
        }
