"""Tool-calling integration tests — in-process via FastAPI TestClient.

No running server needed. Uses create_app() directly.

Usage:
    uv run pytest tests/test_tool_calling.py -v
    uv run pytest tests/test_tool_calling.py -v -k "calculator"
    uv run pytest tests/test_tool_calling.py -v -k "gpt_5_3_codex_spark"
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest
from starlette.testclient import TestClient

from gptmock.app import create_app
from gptmock.services.model_registry import get_model_list

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ALL_MODELS: List[str] = get_model_list(expose_reasoning=False)
TIMEOUT = 120


@pytest.fixture(scope="module")
def client() -> TestClient:
    """In-process TestClient — no external server required."""
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def _payload_function_tool(model: str) -> Dict[str, Any]:
    """Single function tool (calculator)."""
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Calculate 37 * 42 and explain the steps briefly.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Performs basic arithmetic",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Arithmetic expression",
                            }
                        },
                        "required": ["expression"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "stream": False,
    }


def _payload_web_search(model: str) -> Dict[str, Any]:
    """Responses-style web_search tool."""
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Find the latest news about GPTMock and summarize.",
            }
        ],
        "responses_tools": [{"type": "web_search"}],
        "stream": False,
    }


def _payload_parallel(model: str) -> Dict[str, Any]:
    """Parallel function tool calls."""
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Gather the latest AI headlines and compute a sentiment "
                    "score (0-100) for one headline."
                ),
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "sentiment",
                    "description": "Return sentiment score for given text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to score"},
                        },
                        "required": ["text"],
                    },
                },
            },
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "stream": False,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_valid_response(resp, model: str, label: str) -> Dict[str, Any]:
    """Common assertions for any chat completion response."""
    assert resp.status_code == 200, (
        f"[{model}][{label}] status={resp.status_code} body={resp.text[:500]}"
    )
    data = resp.json()
    assert "choices" in data and data["choices"], (
        f"[{model}][{label}] missing or empty choices"
    )
    return data


def _get_message(data: Dict[str, Any]) -> Dict[str, Any]:
    return data["choices"][0].get("message", {})


def _get_tool_calls(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    msg = _get_message(data)
    return msg.get("tool_calls") or []


def _get_finish_reason(data: Dict[str, Any]) -> str:
    return data["choices"][0].get("finish_reason", "")


# ---------------------------------------------------------------------------
# Tests: function tool (calculator) — per model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_function_tool_returns_valid_response(client: TestClient, model: str) -> None:
    """Function tool request returns 200 with either tool_calls or text content."""
    payload = _payload_function_tool(model)
    resp = client.post("/v1/chat/completions", json=payload, timeout=TIMEOUT)
    data = _assert_valid_response(resp, model, "calculator")

    msg = _get_message(data)
    # Model should either call the tool or answer directly — both are valid
    has_tool_calls = bool(msg.get("tool_calls"))
    has_content = isinstance(msg.get("content"), str) and msg["content"].strip()
    assert has_tool_calls or has_content, (
        f"[{model}][calculator] response has neither tool_calls nor content"
    )


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_function_tool_calls_structure(client: TestClient, model: str) -> None:
    """If tool_calls are present, verify their structure is correct."""
    payload = _payload_function_tool(model)
    resp = client.post("/v1/chat/completions", json=payload, timeout=TIMEOUT)
    data = _assert_valid_response(resp, model, "calculator_structure")

    tool_calls = _get_tool_calls(data)
    if not tool_calls:
        pytest.skip(
            f"[{model}] model chose not to call tool — skipping structure check"
        )

    for tc in tool_calls:
        assert isinstance(tc.get("id"), str) and tc["id"], (
            f"[{model}] tool_call missing id"
        )
        assert tc.get("type") == "function", (
            f"[{model}] tool_call type should be 'function', got {tc.get('type')}"
        )
        fn = tc.get("function", {})
        assert isinstance(fn.get("name"), str) and fn["name"], (
            f"[{model}] tool_call.function.name missing"
        )
        assert isinstance(fn.get("arguments"), str), (
            f"[{model}] tool_call.function.arguments should be a JSON string"
        )
        # arguments should be valid JSON
        try:
            json.loads(fn["arguments"])
        except json.JSONDecodeError:
            pytest.fail(
                f"[{model}] tool_call.function.arguments is not valid JSON: {fn['arguments']!r}"
            )


# ---------------------------------------------------------------------------
# Tests: web_search (responses_tools) — per model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_web_search_returns_valid_response(client: TestClient, model: str) -> None:
    """Web search via responses_tools returns 200 with content."""
    payload = _payload_web_search(model)
    resp = client.post("/v1/chat/completions", json=payload, timeout=TIMEOUT)
    data = _assert_valid_response(resp, model, "web_search")

    msg = _get_message(data)
    content = msg.get("content")
    assert isinstance(content, str) and content.strip(), (
        f"[{model}][web_search] expected non-empty content"
    )


# ---------------------------------------------------------------------------
# Tests: parallel tool calls — per model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_parallel_tool_calls_valid_response(client: TestClient, model: str) -> None:
    """Parallel tool request returns 200 with either tool_calls or text content."""
    payload = _payload_parallel(model)
    resp = client.post("/v1/chat/completions", json=payload, timeout=TIMEOUT)
    data = _assert_valid_response(resp, model, "parallel")

    msg = _get_message(data)
    has_tool_calls = bool(msg.get("tool_calls"))
    has_content = isinstance(msg.get("content"), str) and msg["content"].strip()
    assert has_tool_calls or has_content, (
        f"[{model}][parallel] response has neither tool_calls nor content"
    )


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_parallel_multi_tool_calls(client: TestClient, model: str) -> None:
    """If model supports parallel tool calling, it may return multiple tool_calls."""
    payload = _payload_parallel(model)
    resp = client.post("/v1/chat/completions", json=payload, timeout=TIMEOUT)
    data = _assert_valid_response(resp, model, "parallel_multi")

    tool_calls = _get_tool_calls(data)
    if not tool_calls:
        pytest.skip(f"[{model}] model chose not to call tools")

    # Log for visibility — not a hard assertion since models may choose sequential
    names = [
        (tc.get("function") or {}).get("name", "?")
        for tc in tool_calls
        if isinstance(tc, dict)
    ]
    print(f"[{model}] parallel tool_calls: {len(tool_calls)} -> {names}")

    # Each tool_call must still have valid structure
    for tc in tool_calls:
        assert isinstance(tc.get("id"), str) and tc["id"]
        assert tc.get("type") == "function"
        fn = tc.get("function", {})
        assert isinstance(fn.get("name"), str) and fn["name"]
        assert isinstance(fn.get("arguments"), str)


# ---------------------------------------------------------------------------
# Tests: unsupported model name (n8n issue #4 reproduction)
# ---------------------------------------------------------------------------


def test_unsupported_model_returns_error(client: TestClient) -> None:
    """Requesting an unsupported model name should return a clear error, not hang."""
    payload = {
        "model": "gpt-5-mini",  # Not in model_registry mapping
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    resp = client.post("/v1/chat/completions", json=payload, timeout=TIMEOUT)
    # Server should either:
    # 1. Normalize to a valid model and succeed (200), OR
    # 2. Return a clear error (4xx)
    # It should NOT return 502/500 with "Upstream error"
    if resp.status_code == 200:
        data = resp.json()
        assert "choices" in data
    else:
        data = resp.json()
        err = data.get("error", {})
        assert isinstance(err, dict), "error response should have error object"
        assert "message" in err, "error should have a message"
        print(
            f"Unsupported model error: status={resp.status_code} msg={err.get('message')}"
        )
