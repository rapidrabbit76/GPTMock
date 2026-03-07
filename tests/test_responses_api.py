"""Integration tests: Responses API for all supported models.

Validates that every model returns a valid response via POST /v1/responses.
No running server needed — uses FastAPI TestClient in-process.
"""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from tests.conftest import ALL_MODELS, TEST_PROMPT, TIMEOUT


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_responses_non_stream(client: TestClient, model: str) -> None:
    """POST /v1/responses (non-streaming) returns valid response for each model."""
    payload = {
        "model": model,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": TEST_PROMPT}],
            }
        ],
        "stream": False,
    }

    resp = client.post("/v1/responses", json=payload, timeout=TIMEOUT)

    assert resp.status_code == 200, (
        f"[{model}] status={resp.status_code} body={resp.text}"
    )

    data = resp.json()
    assert "output" in data, f"[{model}] missing 'output' in response"
    assert len(data["output"]) > 0, f"[{model}] empty output"

    # Find the message output item
    messages = [o for o in data["output"] if o.get("type") == "message"]
    assert messages, f"[{model}] no message in output"

    content_parts = messages[0].get("content", [])
    assert content_parts, f"[{model}] no content parts in message"

    text_parts = [p for p in content_parts if p.get("type") == "output_text"]
    assert text_parts, f"[{model}] no output_text parts"
    assert len(text_parts[0].get("text", "").strip()) > 0, (
        f"[{model}] output text is empty"
    )


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_responses_stream(client: TestClient, model: str) -> None:
    """POST /v1/responses (streaming) returns valid SSE events for each model."""
    payload = {
        "model": model,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": TEST_PROMPT}],
            }
        ],
        "stream": True,
    }

    with client.stream("POST", "/v1/responses", json=payload, timeout=TIMEOUT) as resp:
        assert resp.status_code == 200, f"[{model}] status={resp.status_code}"

        events: list[dict] = []
        got_completed = False

        for line in resp.iter_lines():
            if not line:
                continue
            if line.startswith("event: "):
                continue
            if line.startswith("data: "):
                data = json.loads(line[6:])
                events.append(data)
                if data.get("type") == "response.completed":
                    got_completed = True

    assert got_completed, f"[{model}] stream never received response.completed"
    assert len(events) > 0, f"[{model}] no events received"
