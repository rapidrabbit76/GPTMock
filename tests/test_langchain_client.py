"""Integration tests: LangChain client for all supported models.

Validates that every model returns a valid response via langchain-openai ChatOpenAI.
"""

from __future__ import annotations

import pytest
from langchain_openai import ChatOpenAI

from tests.conftest import ALL_MODELS, OPENAI_BASE_URL, TEST_PROMPT, TIMEOUT


def _make_llm(model: str, *, streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=OPENAI_BASE_URL,
        api_key="test-key",  # type: ignore[arg-type]
        model=model,
        streaming=streaming,
        request_timeout=TIMEOUT,
    )


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_langchain_invoke(model: str) -> None:
    """ChatOpenAI.invoke (non-streaming) returns content for each model."""
    llm = _make_llm(model, streaming=False)
    response = llm.invoke(TEST_PROMPT)

    assert response.content is not None, f"[{model}] response.content is None"
    content = str(response.content)
    assert len(content.strip()) > 0, f"[{model}] content is empty"


@pytest.mark.parametrize("model", ALL_MODELS, ids=ALL_MODELS)
def test_langchain_stream(model: str) -> None:
    """ChatOpenAI.stream returns chunks for each model."""
    llm = _make_llm(model, streaming=True)

    chunks: list[str] = []
    for chunk in llm.stream(TEST_PROMPT):
        if chunk.content:
            chunks.append(str(chunk.content))

    full_text = "".join(chunks)
    assert len(full_text.strip()) > 0, f"[{model}] streamed content is empty"
