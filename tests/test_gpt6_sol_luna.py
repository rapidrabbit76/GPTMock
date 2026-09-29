from __future__ import annotations

import importlib
import json
from collections.abc import AsyncIterator
from typing import Final

import httpx
import pytest
from starlette.testclient import TestClient

from gptmock.app import create_app
from gptmock.core.settings import Settings
from gptmock.services.model_registry import get_model_list, normalize_model_name, resolve_upstream_model
from gptmock.services.reasoning import (
    allowed_efforts_for_model,
    build_reasoning_param,
    extract_reasoning_from_model_name,
)

MODELS: Final = ("gpt-6-sol", "gpt-6-luna")
EFFORTS: Final = ("none", "low", "medium", "high", "xhigh", "max")
ROUTES: Final = ("/v1/chat/completions", "/v1/completions", "/v1/responses", "/api/chat", "/api/generate")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("expose_reasoning", [False, True])
def test_gpt6_discovery_advertises_concrete_models(model: str, expose_reasoning: bool) -> None:
    with TestClient(create_app(Settings(expose_reasoning_models=expose_reasoning))) as client:
        models = {item["id"]: item for item in client.get("/v1/models").json()["data"]}
        tags = {item["name"]: item for item in client.get("/api/tags").json()["models"]}
        assert models[model]["reasoning"] == {
            "supported_efforts": list(EFFORTS), "default_effort": "medium",
        }
        assert tags[model]["remote_model"] == model
        assert tags[model]["details"]["format"] == "remote"
        assert tags[model]["size"] == 0
        assert tags[model]["digest"] == ""
        assert set(models) == set(tags)
        assert not any(name.startswith(f"{model}-fast") for name in models)
        for suffix in ("", "-fast"):
            response = client.post("/api/show", json={"model": model + suffix})
            assert response.status_code == 200
            assert response.json()["model_info"]["gptmock.upstream_model"] == model
        for effort in EFFORTS:
            name = f"{model}-{effort}"
            assert (name in models) is expose_reasoning
            if expose_reasoning:
                assert models[name]["reasoning"]["preset_effort"] == effort


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("effort", EFFORTS)
@pytest.mark.parametrize("fast", [False, True])
def test_gpt6_aliases_preserve_model_and_reasoning(model: str, effort: str, fast: bool) -> None:
    requested = model + ("-fast" if fast else "")
    for alias in (requested, requested.replace("gpt-6", "gpt6"), f"{requested}-latest"):
        assert normalize_model_name(alias) == requested
    for separator in ("-", "_", ":"):
        alias = f"{requested}{separator}{effort}"
        assert normalize_model_name(alias) == requested
        assert extract_reasoning_from_model_name(alias) == {"effort": effort}
        assert allowed_efforts_for_model(alias) == set(EFFORTS)
    overrides = {"service_tier": "priority"} if fast else {}
    assert resolve_upstream_model(requested) == (model, overrides)
    assert build_reasoning_param(effort, allowed_efforts=allowed_efforts_for_model(requested))["effort"] == effort


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("effort", ["minimal", "ultra"])
def test_gpt6_unsupported_efforts_are_rejected(model: str, effort: str) -> None:
    assert f"{model}-{effort}" not in get_model_list(expose_reasoning=True)
    with pytest.raises(ValueError, match="Unsupported reasoning effort"):
        build_reasoning_param(effort, allowed_efforts=allowed_efforts_for_model(model))


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("effort", (*EFFORTS, "minimal", "ultra"))
def test_gpt6_routes_preserve_model_and_validate_effort(
    monkeypatch: pytest.MonkeyPatch, model: str, effort: str,
) -> None:
    captured = []

    async def fake_auth() -> tuple[str, str]:
        assert effort in EFFORTS, "Unsupported efforts must fail before authentication"
        return "test-token", "test-account"

    def fake_transport(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        event = {"type": "response.completed", "response": {
            "id": "resp_gpt6", "model": model, "status": "completed", "service_tier": "default",
            "output": [{"type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "OK", "annotations": []}]}],
        }}
        return httpx.Response(200, content=f"data: {json.dumps(event)}\n\n".encode())

    async def test_http_client() -> AsyncIterator[httpx.AsyncClient]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake_transport)) as client:
            yield client

    for module in ("gptmock.services.chat", "gptmock.services.responses"):
        monkeypatch.setattr(importlib.import_module(module), "get_effective_chatgpt_auth", fake_auth)
    app = importlib.import_module("gptmock.app").create_app(Settings())
    dependency = importlib.import_module("gptmock.core.dependencies").get_http_client
    app.dependency_overrides[dependency] = test_http_client
    with TestClient(app) as client:
        for route in ROUTES:
            for suffix in ("", "-fast"):
                payload = {
                    "model": model + suffix, "stream": False,
                    "messages": [{"role": "user", "content": "hello"}],
                    "input": "hello", "prompt": "hello", "reasoning_effort": effort,
                    "reasoning": {"effort": effort},
                }
                response = client.post(route, json=payload)
                if effort not in EFFORTS:
                    assert response.status_code == 400
                    assert "Unsupported reasoning effort" in response.text
                    assert captured == []
                    continue
                assert response.status_code == 200
                assert response.json()["model"] == model
                assert response.json()["service_tier"] == "default"
                assert captured[-1]["model"] == model
                assert captured[-1]["reasoning"]["effort"] == effort
                assert (captured[-1].get("service_tier") == "priority") is bool(suffix)
    assert len(captured) == (len(ROUTES) * 2 if effort in EFFORTS else 0)
