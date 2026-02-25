"""Shared fixtures for gptmock integration tests.

These tests require a running gptmock server.
Configure the server URL via GPTMOCK_BASE_URL env var (default: http://127.0.0.1:8000).

Usage:
    # Start server first
    gptmock serve

    # Run tests
    pytest tests/ -v
"""

from __future__ import annotations

import os
from typing import List

import pytest

from gptmock.services.model_registry import get_model_list

BASE_URL = os.getenv("GPTMOCK_BASE_URL", "http://127.0.0.1:8000")
OPENAI_BASE_URL = f"{BASE_URL}/v1"
TEST_PROMPT = "Say 'hello' and nothing else."
TIMEOUT = 120


def _get_all_models() -> List[str]:
    """Return all base model IDs from the registry (computed once at import time)."""
    return get_model_list(expose_reasoning=False)


ALL_MODELS: List[str] = _get_all_models()


@pytest.fixture(scope="session")
def all_models() -> List[str]:
    """Session-scoped fixture providing the model list."""
    return ALL_MODELS


@pytest.fixture(scope="session")
def base_url() -> str:
    return BASE_URL


@pytest.fixture(scope="session")
def openai_base_url() -> str:
    return OPENAI_BASE_URL
