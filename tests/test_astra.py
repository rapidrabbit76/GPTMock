"""Astra discovery and request validation through all compatibility routes."""

from __future__ import annotations

import importlib
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from starlette.testclient import TestClient

from gptmock.app import create_app
from gptmock.core.dependencies import get_http_client
from gptmock.core.settings import Settings
from gptmock.services.model_registry import (
    get_model_list,
    normalize_model_name,
    resolve_upstream_model,
)
from gptmock.services.reasoning import (
    allowed_efforts_for_model,
    extract_reasoning_from_model_name,
)
from gptmock.services.upstream import _adapt_astra_system_messages

ASTRA_EFFORTS = ["low", "medium", "high", "xhigh", "max"]


def test_astra_discovery_and_remote_metadata() -> None:
    with TestClient(create_app(Settings())) as client:
        models = {item["id"]: item for item in client.get("/v1/models").json()["data"]}
        assert models["gpt-6-astra"]["reasoning"] == {
            "supported_efforts": ASTRA_EFFORTS,
            "default_effort": "medium",
        }
        tags = {item["name"]: item for item in client.get("/api/tags").json()["models"]}
        assert tags["gpt-6-astra"]["remote_model"] == "gpt-6-astra"
        assert tags["gpt-6-astra"]["details"]["format"] == "remote"
        assert tags["gpt-6-astra"]["digest"] == ""
        for name in ("gpt-6-astra-fast", "gpt-6-astra-pro", "gpt-6"):
            assert name not in models
            assert name not in tags
        response = client.post("/api/show", json={"model": "gpt-6-astra-fast"})
        assert response.status_code == 200
        assert response.json()["model_info"]["gptmock.upstream_model"] == "gpt-6-astra"


@pytest.mark.parametrize("effort", ASTRA_EFFORTS)
@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6-astra-fast"])
def test_astra_effort_aliases_keep_the_concrete_model(model: str, effort: str) -> None:
    name = f"{model}-{effort}"
    if effort == "max":
        with pytest.raises(ValueError, match="Removed model alias"):
            normalize_model_name(name)
        assert name not in get_model_list(expose_reasoning=True)
        assert allowed_efforts_for_model(model) == set(ASTRA_EFFORTS)
        return
    assert normalize_model_name(name) == model
    assert normalize_model_name(model.replace("gpt-6", "gpt6")) == model
    assert normalize_model_name(f"{model}-latest") == model
    assert extract_reasoning_from_model_name(name) == {"effort": effort}
    assert allowed_efforts_for_model(name) == set(ASTRA_EFFORTS)
    overrides = {"service_tier": "priority"} if model.endswith("-fast") else {}
    assert resolve_upstream_model(model) == ("gpt-6-astra", overrides)
    if not model.endswith("-fast"):
        assert name in get_model_list(expose_reasoning=True)


@pytest.mark.parametrize("effort", ["none", "minimal", "ultra"])
@pytest.mark.parametrize("route", ["/v1/responses", "/v1/chat/completions", "/api/chat", "/api/generate"])
def test_astra_rejects_unsupported_efforts_before_auth(
    monkeypatch: pytest.MonkeyPatch, effort: str, route: str,
) -> None:
    async def unexpected_auth() -> tuple[str, str]:
        pytest.fail("unsupported effort must not reach authentication or upstream")

    for module in ("gptmock.services.chat", "gptmock.services.responses"):
        monkeypatch.setattr(importlib.import_module(module), "get_effective_chatgpt_auth", unexpected_auth)
    payload: dict[str, Any] = {"model": "gpt-6-astra", "stream": False}
    if route == "/v1/responses":
        payload.update(input="hello", reasoning={"effort": effort})
    elif route == "/api/generate":
        payload.update(prompt="hello", think=effort)
    else:
        payload["messages"] = [{"role": "user", "content": "hello"}]
        payload["think" if route == "/api/chat" else "reasoning_effort"] = effort
    with TestClient(create_app(Settings())) as client:
        response = client.post(route, json=payload)
    assert response.status_code == 400
    assert "Unsupported reasoning effort" in response.text


def test_astra_rejected_options_are_not_registered() -> None:
    models = get_model_list(expose_reasoning=True)
    for suffix in ("none", "minimal", "ultra", "pro"):
        assert f"gpt-6-astra-{suffix}" not in models
    assert resolve_upstream_model("gpt-6-astra-pro") == ("gpt-6-astra-pro", {})


@pytest.mark.parametrize("route", ["/v1/responses", "/v1/chat/completions", "/api/chat", "/api/generate"])
@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6-astra-fast"])
def test_astra_system_messages_reach_upstream_as_instructions(
    monkeypatch: pytest.MonkeyPatch, route: str, model: str,
) -> None:
    captured: list[dict[str, Any]] = []

    async def fake_auth() -> tuple[str, str]:
        return "test-token", "test-account"

    def fake_transport(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        event = {"type": "response.completed", "response": {
            "id": "resp_astra",
            "model": "gpt-6-astra",
            "status": "completed",
            "service_tier": "default",
            "output": [{"type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "OK", "annotations": []}]}],
        }}
        return httpx.Response(200, content=f"data: {json.dumps(event)}\n\n".encode())

    async def test_http_client() -> AsyncIterator[httpx.AsyncClient]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake_transport)) as client:
            yield client

    for module in ("gptmock.services.chat", "gptmock.services.responses"):
        loaded = importlib.import_module(module)
        monkeypatch.setattr(loaded, "get_effective_chatgpt_auth", fake_auth)
    payload: dict[str, Any] = {"model": model, "stream": False}
    if route == "/api/generate":
        payload.update(system="system authority", prompt="hello")
    else:
        input_key = "input" if route == "/v1/responses" else "messages"
        payload[input_key] = [
            {"role": "system", "content": "system authority"},
            {"role": "developer", "content": "developer authority"},
            {"role": "user", "content": "hello"},
        ]
    app = create_app(Settings())
    app.dependency_overrides[get_http_client] = test_http_client
    with TestClient(app) as client:
        response = client.post(route, json=payload)
    assert response.status_code == 200
    assert response.json()["model"] == "gpt-6-astra"
    assert response.json()["service_tier"] == "default"
    assert len(captured) == 1
    assert captured[0]["model"] == "gpt-6-astra"
    assert captured[0]["instructions"].endswith("system authority")
    expected_roles = ["user"] if route == "/api/generate" else ["developer", "user"]
    assert [item["role"] for item in captured[0]["input"]] == expected_roles
    if model.endswith("-fast"):
        assert captured[0]["service_tier"] == "priority"


def test_astra_instruction_adaptation_preserves_text_and_input_payload() -> None:
    payload = {
        "model": "gpt-6-astra", "instructions": "existing instructions",
        "input": [
            {"role": "system", "content": "first system"},
            {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "second system"}]},
            {"role": "user", "content": "hello"},
        ],
    }
    adapted = _adapt_astra_system_messages(payload)
    assert adapted["instructions"] == "existing instructions\n\nfirst system\n\nsecond system"
    assert adapted["input"] == [{"role": "user", "content": "hello"}]
    assert payload["instructions"] == "existing instructions"
    assert len(payload["input"]) == 3
    assert _adapt_astra_system_messages(adapted) == adapted


def test_astra_instruction_adaptation_does_not_discard_nontext_content() -> None:
    payload = {"model": "gpt-6-astra", "input": [
        {"role": "system", "content": [{"type": "input_image", "image_url": "https://example.com/image.png"}]},
    ]}
    assert _adapt_astra_system_messages(payload) is payload
    other_model = {"model": "gpt-5.6-luna", "input": [{"role": "system", "content": "system authority"}]}
    assert _adapt_astra_system_messages(other_model) is other_model
