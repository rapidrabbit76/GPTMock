"""Integration tests: LangChain client for all supported models.

Validates that every model returns a valid response via langchain-openai ChatOpenAI.
No running server needed — uses FastAPI TestClient in-process.
"""

from __future__ import annotations

import pytest
from langchain_openai import ChatOpenAI
from starlette.testclient import TestClient

from tests.conftest import ALL_MODELS, TEST_PROMPT, TIMEOUT


def _make_llm(
    model: str, http_client: TestClient, *, streaming: bool = False
) -> ChatOpenAI:
    return ChatOpenAI(
        base_url="http://testserver/v1",
        api_key="test-key",  # type: ignore[arg-type]
        model=model,
        streaming=streaming,
        timeout=TIMEOUT,
        http_client=http_client,
    )


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_langchain_invoke(client: TestClient, model: str) -> None:
    """ChatOpenAI.invoke (non-streaming) returns content for each model."""
    llm = _make_llm(model, client, streaming=False)
    response = llm.invoke(TEST_PROMPT)

    assert response.content is not None, f"[{model}] response.content is None"
    content = str(response.content)
    assert len(content.strip()) > 0, f"[{model}] content is empty"


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_langchain_stream(client: TestClient, model: str) -> None:
    """ChatOpenAI.stream returns chunks for each model."""
    llm = _make_llm(model, client, streaming=True)

    chunks: list[str] = []
    for chunk in llm.stream(TEST_PROMPT):
        if chunk.content:
            chunks.append(str(chunk.content))

    full_text = "".join(chunks)
    assert len(full_text.strip()) > 0, f"[{model}] streamed content is empty"
