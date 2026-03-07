"""Integration tests: OpenAI SDK client for all supported models.

Validates that every model returns a valid response via the openai Python client.
No running server needed — uses FastAPI TestClient in-process.
"""

from __future__ import annotations

import openai
import pytest
from starlette.testclient import TestClient

from tests.conftest import ALL_MODELS, TEST_PROMPT, TIMEOUT


def _make_openai_client(http_client: TestClient) -> openai.OpenAI:
    return openai.OpenAI(
        base_url="http://testserver/v1",
        api_key="test-key",
        timeout=TIMEOUT,
        http_client=http_client,
    )


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_openai_chat_non_stream(client: TestClient, model: str) -> None:
    """OpenAI client chat.completions.create (non-streaming) returns content for each model."""
    oc = _make_openai_client(client)

    resp = oc.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": TEST_PROMPT}],
        stream=False,
    )

    assert resp.choices, f"[{model}] no choices returned"
    message = resp.choices[0].message
    assert message.content is not None, f"[{model}] message.content is None"
    assert len(message.content.strip()) > 0, f"[{model}] content is empty"


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_openai_chat_stream(client: TestClient, model: str) -> None:
    """OpenAI client chat.completions.create (streaming) returns chunks for each model."""
    oc = _make_openai_client(client)

    stream = oc.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": TEST_PROMPT}],
        stream=True,
    )

    chunks: list[str] = []
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                chunks.append(delta.content)

    full_text = "".join(chunks)
    assert len(full_text.strip()) > 0, f"[{model}] streamed content is empty"


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_openai_completions(client: TestClient, model: str) -> None:
    """OpenAI client completions.create (legacy text) returns text for each model."""
    oc = _make_openai_client(client)

    resp = oc.completions.create(
        model=model,
        prompt=TEST_PROMPT,
        stream=False,
    )

    assert resp.choices, f"[{model}] no choices returned"
    text = resp.choices[0].text
    assert isinstance(text, str) and len(text.strip()) > 0, (
        f"[{model}] completion text is empty"
    )
