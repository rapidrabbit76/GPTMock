"""Integration tests: Ollama API endpoints.

Validates Ollama-compatible API endpoints via direct HTTP calls.
No running server needed — uses FastAPI TestClient in-process.
"""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from tests.conftest import ALL_MODELS, TEST_PROMPT, TIMEOUT


def test_ollama_version(client: TestClient) -> None:
    """GET /api/version returns version string."""
    resp = client.get("/api/version", timeout=TIMEOUT)
    assert resp.status_code == 200
    data = resp.json()
    assert "version" in data, "missing 'version' in response"
    assert isinstance(data["version"], str), "version is not a string"


def test_ollama_tags(client: TestClient) -> None:
    """GET /api/tags returns model list."""
    resp = client.get("/api/tags", timeout=TIMEOUT)
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data, "missing 'models' in response"
    assert isinstance(data["models"], list), "models is not a list"
    assert len(data["models"]) > 0, "no models returned"


def test_ollama_show(client: TestClient) -> None:
    """POST /api/show returns model details."""
    payload = {"model": "gpt-5"}
    resp = client.post("/api/show", json=payload, timeout=TIMEOUT)
    assert resp.status_code == 200
    data = resp.json()
    assert "details" in data, "missing 'details' in response"
    assert "capabilities" in data, "missing 'capabilities' in response"


def test_ollama_show_invalid_model(client: TestClient) -> None:
    """POST /api/show with missing model returns 400."""
    resp = client.post("/api/show", json={}, timeout=TIMEOUT)
    assert resp.status_code == 400


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_ollama_chat_non_stream(client: TestClient, model: str) -> None:
    """POST /api/chat (non-streaming) returns valid Ollama response for each model."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "stream": False,
    }

    resp = client.post("/api/chat", json=payload, timeout=TIMEOUT)

    assert resp.status_code == 200, (
        f"[{model}] status={resp.status_code} body={resp.text}"
    )

    data = resp.json()
    assert "message" in data, f"[{model}] missing 'message' in response"
    content = data["message"].get("content")
    assert isinstance(content, str) and len(content.strip()) > 0, (
        f"[{model}] ollama response content is empty or missing"
    )
    assert data.get("done") is True, f"[{model}] ollama response 'done' is not True"


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_ollama_chat_stream(client: TestClient, model: str) -> None:
    """POST /api/chat (streaming) returns valid ndjson chunks for each model."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "stream": True,
    }

    with client.stream("POST", "/api/chat", json=payload, timeout=TIMEOUT) as resp:
        assert resp.status_code == 200, f"[{model}] status={resp.status_code}"

        chunks: list[str] = []
        got_done = False

        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if data.get("done") is True:
                got_done = True
                break
            content = data.get("message", {}).get("content", "")
            if content:
                chunks.append(content)

    assert got_done, f"[{model}] stream never received done=True"
    full_text = "".join(chunks)
    assert len(full_text.strip()) > 0, f"[{model}] streamed content is empty"
