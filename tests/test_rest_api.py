"""Integration tests: REST API (httpx) for all supported models.

Validates that every model returns a valid chat completion response
via direct HTTP calls to /v1/chat/completions.
"""

from __future__ import annotations

import json

import httpx
import pytest

from tests.conftest import ALL_MODELS, OPENAI_BASE_URL, TEST_PROMPT, TIMEOUT


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_chat_completions_non_stream(model: str) -> None:
    """POST /v1/chat/completions (non-streaming) returns valid response for each model."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "stream": False,
    }

    with httpx.Client(base_url=OPENAI_BASE_URL, timeout=TIMEOUT) as client:
        resp = client.post("/chat/completions", json=payload)

    assert resp.status_code == 200, f"[{model}] status={resp.status_code} body={resp.text}"

    data = resp.json()
    assert "choices" in data, f"[{model}] missing 'choices' in response"
    assert len(data["choices"]) > 0, f"[{model}] empty choices"

    message = data["choices"][0].get("message", {})
    content = message.get("content")
    assert content is not None, f"[{model}] message.content is None"
    assert isinstance(content, str), f"[{model}] content is not str: {type(content)}"
    assert len(content.strip()) > 0, f"[{model}] content is empty"


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_chat_completions_stream(model: str) -> None:
    """POST /v1/chat/completions (streaming) returns valid SSE chunks for each model."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "stream": True,
    }

    with httpx.Client(base_url=OPENAI_BASE_URL, timeout=TIMEOUT) as client:
        with client.stream("POST", "/chat/completions", json=payload) as resp:
            assert resp.status_code == 200, f"[{model}] status={resp.status_code}"

            chunks: list[str] = []
            got_done = False

            for line in resp.iter_lines():
                if not line:
                    continue
                if line == "data: [DONE]":
                    got_done = True
                    break
                if line.startswith("data: "):
                    chunk = json.loads(line[6:])
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content")
                    if isinstance(text, str):
                        chunks.append(text)

    assert got_done, f"[{model}] stream never received [DONE]"
    full_text = "".join(chunks)
    assert len(full_text.strip()) > 0, f"[{model}] streamed content is empty"


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_ollama_chat(model: str) -> None:
    """POST /api/chat (Ollama format) returns valid response for each model."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "stream": False,
    }

    with httpx.Client(base_url=OPENAI_BASE_URL.rsplit("/v1", 1)[0], timeout=TIMEOUT) as client:
        resp = client.post("/api/chat", json=payload)

    assert resp.status_code == 200, f"[{model}] status={resp.status_code} body={resp.text}"

    data = resp.json()
    assert "message" in data, f"[{model}] missing 'message' in response"
    content = data["message"].get("content")
    assert isinstance(content, str) and len(content.strip()) > 0, (
        f"[{model}] ollama response content is empty or missing"
    )
    assert data.get("done") is True, f"[{model}] ollama response 'done' is not True"


def test_chat_completions_non_stream_web_search_annotations() -> None:
    """POST /v1/chat/completions (non-streaming) surfaces url_citation annotations when web search is enabled."""
    payload = {
        "model": "gpt-5",
        "messages": [
            {
                "role": "user",
                "content": "Find a recent news article about SpaceX launches and include sources.",
            }
        ],
        "stream": False,
        "responses_tools": [{"type": "web_search"}],
        "responses_tool_choice": "auto",
    }

    with httpx.Client(base_url=OPENAI_BASE_URL, timeout=TIMEOUT) as client:
        resp = client.post("/chat/completions", json=payload)

    assert resp.status_code == 200, f"status={resp.status_code} body={resp.text}"

    data = resp.json()
    assert "choices" in data and data["choices"], "missing or empty choices"

    message = data["choices"][0]["message"]
    assert isinstance(message, dict)
    assert isinstance(message.get("content"), str) and message["content"].strip(), "empty content"

    # Verify annotations are present
    annotations = message.get("annotations")
    assert isinstance(annotations, list), "expected annotations list on message"
    assert annotations, "expected at least one annotation for web search response"

    # Verify url_citation structure
    url_citations = [a for a in annotations if a.get("type") == "url_citation"]
    assert url_citations, "expected at least one url_citation annotation"

    for ann in url_citations:
        assert isinstance(ann.get("url"), str) and ann["url"], "url_citation must have non-empty url"
        assert isinstance(ann.get("title"), str) and ann["title"], "url_citation must have non-empty title"
        assert isinstance(ann.get("start_index"), int), "url_citation must have start_index"
        assert isinstance(ann.get("end_index"), int), "url_citation must have end_index"
