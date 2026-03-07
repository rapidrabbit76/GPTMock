"""Shared fixtures for gptmock integration tests.

No running server needed — uses FastAPI TestClient in-process.

Usage:
    uv run pytest tests/ -v
    uv run pytest tests/ -v -k "gpt-5.4"
"""

from __future__ import annotations

from typing import Generator, List

import pytest
from starlette.testclient import TestClient

from gptmock.app import create_app
from gptmock.services.model_registry import get_model_list

TEST_PROMPT = "Say 'hello' and nothing else."
TIMEOUT = 120


def _get_all_models() -> List[str]:
    """Return all base model IDs from the registry (computed once at import time)."""
    return get_model_list(expose_reasoning=False)


ALL_MODELS: List[str] = _get_all_models()


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """In-process TestClient — no external server required."""
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture(scope="session")
def all_models() -> List[str]:
    """Session-scoped fixture providing the model list."""
    return ALL_MODELS
