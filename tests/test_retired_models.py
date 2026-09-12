from __future__ import annotations

import importlib
import json
from collections.abc import AsyncIterator

import httpx
import pytest
from starlette.testclient import TestClient

from gptmock.app import create_app
from gptmock.core.dependencies import get_http_client
from gptmock.core.settings import Settings
from gptmock.services.model_registry import (
    FAST_MODEL_ALIASES,
    MODEL_GROUPS,
    SYNTHETIC_MODEL_GROUPS,
    normalize_model_name,
    resolve_upstream_model,
)

RETIRED_MODELS = ("gpt-5.4", "gpt-5.4-mini", "gpt-5.4-fast", "gpt-5.4-mini-fast")
GENERATION_ROUTES = (
    "/v1/chat/completions", "/v1/completions", "/v1/responses", "/api/chat", "/api/generate",
)


@pytest.mark.parametrize("expose_reasoning", [False, True])
def test_retired_models_are_absent_from_discovery(expose_reasoning: bool) -> None:
    with TestClient(create_app(Settings(expose_reasoning_models=expose_reasoning))) as client:
        models = client.get("/v1/models")
        tags = client.get("/api/tags")
        assert models.status_code == tags.status_code == 200
        model_ids = {item["id"] for item in models.json()["data"]}
        tag_ids = {item["name"] for item in tags.json()["models"]}
        assert model_ids == tag_ids
        assert "gpt-5.5" in model_ids
        assert not any(name.startswith("gpt-5.4") for name in model_ids)
        for model in RETIRED_MODELS:
            assert client.post("/api/show", json={"model": model}).status_code == 404
            assert client.post("/api/show", json={"name": model}).status_code == 404
        if expose_reasoning:
            assert client.post("/api/show", json={"model": "gpt-5.4-high"}).status_code == 404


@pytest.mark.parametrize("model", RETIRED_MODELS)
def test_retired_models_have_no_registered_alias_or_substitution(model: str) -> None:
    assert model not in {name for name, _ in (*MODEL_GROUPS, *SYNTHETIC_MODEL_GROUPS)}
    assert model not in FAST_MODEL_ALIASES
    assert normalize_model_name(model) == model
    assert resolve_upstream_model(model) == (model, {})


@pytest.mark.parametrize("model", RETIRED_MODELS)
@pytest.mark.parametrize("route", GENERATION_ROUTES)
@pytest.mark.parametrize("stream", [False, True])
def test_explicit_retired_requests_preserve_upstream_rejection(
    monkeypatch: pytest.MonkeyPatch, model: str, route: str, stream: bool,
) -> None:
    captured: list[dict] = []

    async def fake_auth() -> tuple[str, str]:
        return "test-token", "test-account"

    def fake_transport(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(400, json={"error": {"message": "retired-model-probe-rejected"}})

    async def test_http_client() -> AsyncIterator[httpx.AsyncClient]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake_transport)) as client:
            yield client

    for module in ("gptmock.services.chat", "gptmock.services.responses"):
        monkeypatch.setattr(importlib.import_module(module), "get_effective_chatgpt_auth", fake_auth)
    app = create_app(Settings(debug_model=None))
    app.dependency_overrides[get_http_client] = test_http_client
    payload = {"model": model, "stream": stream, "messages": [{"role": "user", "content": "hello"}],
               "prompt": "hello", "input": "hello"}
    with TestClient(app) as client:
        response = client.post(route, json=payload)
    assert response.status_code == 400
    assert "retired-model-probe-rejected" in response.text
    assert len(captured) == 1
    assert captured[0]["model"] == model
    assert captured[0].get("service_tier") != "priority"


@pytest.mark.parametrize("model", [None, "", " ", "\t\n"])
def test_empty_model_has_no_implicit_default(model: str | None) -> None:
    with pytest.raises(ValueError, match="non-empty model"):
        normalize_model_name(model)


def test_explicit_debug_model_override_remains_available() -> None:
    assert normalize_model_name("", debug_model="gpt-5.5") == "gpt-5.5"


@pytest.mark.parametrize("route", GENERATION_ROUTES)
@pytest.mark.parametrize("model", ["", " \t\n"])
@pytest.mark.parametrize("stream", [False, True])
def test_empty_model_fails_before_authentication(
    monkeypatch: pytest.MonkeyPatch, route: str, model: str, stream: bool,
) -> None:
    async def unexpected_auth() -> tuple[str, str]:
        pytest.fail("An empty model must not authenticate or call upstream")

    for module in ("gptmock.services.chat", "gptmock.services.responses"):
        monkeypatch.setattr(importlib.import_module(module), "get_effective_chatgpt_auth", unexpected_auth)
    payload = {"model": model, "stream": stream, "messages": [{"role": "user", "content": "hello"}],
               "prompt": "hello", "input": "hello"}
    with TestClient(create_app(Settings(debug_model=None))) as client:
        response = client.post(route, json=payload)
    assert response.status_code == 400
    assert "non-empty model" in response.text


@pytest.mark.parametrize("route", GENERATION_ROUTES)
def test_missing_model_remains_a_validation_error(route: str) -> None:
    with TestClient(create_app(Settings(debug_model=None))) as client:
        response = client.post(route, json={"messages": [], "prompt": "hello", "input": "hello"})
    assert response.status_code == 422
