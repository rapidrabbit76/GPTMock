from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from openai import OpenAI
from starlette.testclient import TestClient

from gptmock.app import create_app
from gptmock.core.constants import CHATGPT_IMAGES_GENERATIONS_URL
from gptmock.core.dependencies import get_http_client
from gptmock.core.settings import Settings
from gptmock.services import images

ROUTE = "/v1/images/generations"
IMAGE_MODELS = (
    "gpt-image-1.5", "gpt-image-2", "gpt-image-2.5", "gpt-image-2.5-flare", "gpt-image-2.5-sunburst",
    "gpt-image-2.6", "gpt-image-2.6-flare", "gpt-image-2.6-sunburst",
)
PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\nsimulated").decode()
UPSTREAM_RESULT: dict[str, Any] = {
    "created": 1794182400,
    "background": "opaque",
    "output_format": "png",
    "size": "1024x1024",
    "quality": "low",
    "data": [{"b64_json": PNG_B64, "revised_prompt": "A red circle on white"}],
    "usage": {"input_tokens": 12, "output_tokens": 272, "total_tokens": 284},
}


@dataclass
class FakeUpstream:
    response: httpx.Response | Exception = field(default_factory=lambda: httpx.Response(200, json=UPSTREAM_RESULT))
    credentials: tuple[str | None, str | None] = ("test-token", "test-account")
    requests: list[httpx.Request] = field(default_factory=list)
    auth_calls: int = 0

    async def auth(self) -> tuple[str | None, str | None]:
        self.auth_calls += 1
        return self.credentials

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def client(self, monkeypatch: pytest.MonkeyPatch, settings: Settings | None = None) -> TestClient:
        monkeypatch.setattr(images, "get_effective_chatgpt_auth", self.auth)

        async def http_client() -> AsyncIterator[httpx.AsyncClient]:
            async with httpx.AsyncClient(transport=httpx.MockTransport(self.handle)) as client:
                yield client

        app = create_app(settings or Settings(api_key=None, debug_model=None))
        app.dependency_overrides[get_http_client] = http_client
        return TestClient(app)


@pytest.mark.parametrize("extra", [{}, {"stream": False}])
def test_forwards_body_to_codex_images_endpoint_and_returns_upstream_json(
    monkeypatch: pytest.MonkeyPatch, extra: dict[str, Any],
) -> None:
    upstream = FakeUpstream()
    body = {
        "model": "  gpt-image-2.6\t",
        "prompt": "A red circle on white",
        "n": 1,
        "size": "1024x1024",
        "quality": "low",
        "background": "opaque",
        "output_format": "png",
        "output_compression": 80,
        "moderation": "low",
        "user": "user-123",
        "future_option": {"nested": [1, None, "x"]},
        **extra,
    }
    with upstream.client(monkeypatch) as client:
        response = client.post(ROUTE, json=body)

    assert response.status_code == 200
    assert response.json() == UPSTREAM_RESULT
    assert upstream.auth_calls == 1
    assert len(upstream.requests) == 1
    sent = upstream.requests[0]
    assert sent.method == "POST"
    assert str(sent.url) == CHATGPT_IMAGES_GENERATIONS_URL
    assert sent.headers["authorization"] == "Bearer test-token"
    assert sent.headers["chatgpt-account-id"] == "test-account"
    assert sent.headers["content-type"] == "application/json"
    assert sent.headers["accept"] == "application/json"
    assert json.loads(sent.content) == {**body, "model": "gpt-image-2.6"}


def test_openai_sdk_images_generate_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = FakeUpstream()
    with upstream.client(monkeypatch) as test_client:
        sdk = OpenAI(base_url="http://testserver/v1", api_key="unused", http_client=test_client, max_retries=0)
        result = sdk.images.generate(model="gpt-image-2.6", prompt="A red circle on white", size="1024x1024")

    assert result.data is not None
    assert base64.b64decode(result.data[0].b64_json or "") == base64.b64decode(PNG_B64)
    assert json.loads(upstream.requests[0].content) == {
        "model": "gpt-image-2.6", "prompt": "A red circle on white", "size": "1024x1024",
    }


@pytest.mark.parametrize(
    ("status", "error_body"),
    [
        (400, {"error": {"message": "Invalid size", "type": "invalid_request_error", "param": "size",
                         "code": "invalid_value"}}),
        (401, {"detail": "Unauthorized"}),
        (429, {"error": {"message": "Rate limit reached", "code": "rate_limit_exceeded"}}),
        (500, {"error": {"message": "internal"}, "request_id": "req_123"}),
    ],
)
def test_upstream_error_status_and_body_are_preserved(
    monkeypatch: pytest.MonkeyPatch, status: int, error_body: dict[str, Any],
) -> None:
    upstream = FakeUpstream(response=httpx.Response(status, json=error_body))
    with upstream.client(monkeypatch) as client:
        response = client.post(ROUTE, json={"model": "gpt-image-2.6", "prompt": "cat", "size": "7x7"})

    assert response.status_code == status
    assert response.json() == error_body
    assert len(upstream.requests) == 1


def test_non_json_upstream_error_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = FakeUpstream(response=httpx.Response(503, text="upstream overloaded"))
    with upstream.client(monkeypatch) as client:
        response = client.post(ROUTE, json={"model": "gpt-image-2.6", "prompt": "cat"})

    assert response.status_code == 503
    assert response.json() == {"error": {"message": "upstream overloaded"}}


def test_non_json_upstream_success_is_a_bad_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = FakeUpstream(response=httpx.Response(200, text="<html>not json</html>"))
    with upstream.client(monkeypatch) as client:
        response = client.post(ROUTE, json={"model": "gpt-image-2.6", "prompt": "cat"})

    assert response.status_code == 502
    assert "non-JSON" in response.json()["error"]["message"]


def test_network_error_returns_bad_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = FakeUpstream(response=httpx.ConnectError("connection refused"))
    with upstream.client(monkeypatch) as client:
        response = client.post(ROUTE, json={"model": "gpt-image-2.6", "prompt": "cat"})

    assert response.status_code == 502
    message = response.json()["error"]["message"]
    assert "Upstream ChatGPT request failed" in message
    assert "connection refused" in message


@pytest.mark.parametrize("credentials", [(None, None), ("test-token", None), (None, "test-account")])
def test_missing_credentials_return_401_without_upstream_call(
    monkeypatch: pytest.MonkeyPatch, credentials: tuple[str | None, str | None],
) -> None:
    upstream = FakeUpstream(credentials=credentials)
    with upstream.client(monkeypatch) as client:
        response = client.post(ROUTE, json={"model": "gpt-image-2.6", "prompt": "cat"})

    assert response.status_code == 401
    assert "Missing ChatGPT credentials" in response.json()["error"]["message"]
    assert upstream.requests == []


@pytest.mark.parametrize("model", ["", " ", " \t\n"])
def test_empty_model_fails_before_authentication(monkeypatch: pytest.MonkeyPatch, model: str) -> None:
    upstream = FakeUpstream()
    with upstream.client(monkeypatch) as client:
        response = client.post(ROUTE, json={"model": model, "prompt": "cat"})

    assert response.status_code == 400
    assert "non-empty model" in response.json()["error"]["message"]
    assert upstream.auth_calls == 0
    assert upstream.requests == []


def test_missing_model_is_a_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = FakeUpstream()
    with upstream.client(monkeypatch) as client:
        response = client.post(ROUTE, json={"prompt": "cat"})

    assert response.status_code == 422
    assert upstream.auth_calls == 0
    assert upstream.requests == []


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"model": "gpt-image-2.6"}, 422),
        ({"model": "gpt-image-2.6", "prompt": None}, 422),
        ({"model": "gpt-image-2.6", "prompt": ""}, 400),
        ({"model": "gpt-image-2.6", "prompt": " \n\t"}, 400),
    ],
)
def test_missing_or_empty_prompt_is_rejected_before_authentication(
    monkeypatch: pytest.MonkeyPatch, body: dict[str, Any], status: int,
) -> None:
    upstream = FakeUpstream()
    with upstream.client(monkeypatch) as client:
        response = client.post(ROUTE, json=body)

    assert response.status_code == status
    if status == 400:
        assert "non-empty prompt" in response.json()["error"]["message"]
    assert upstream.auth_calls == 0
    assert upstream.requests == []


def test_streaming_is_rejected_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = FakeUpstream()
    with upstream.client(monkeypatch) as client:
        response = client.post(ROUTE, json={"model": "gpt-image-2.6", "prompt": "cat", "stream": True})

    assert response.status_code == 400
    assert "streaming image generation is not supported" in response.json()["error"]["message"]
    assert upstream.auth_calls == 0
    assert upstream.requests == []


def test_proxy_api_key_protects_images_route(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = FakeUpstream()
    settings = Settings(api_key="proxy-secret", debug_model=None)
    with upstream.client(monkeypatch, settings) as client:
        rejected = client.post(ROUTE, json={"model": "gpt-image-2.6", "prompt": "cat"})
        assert upstream.auth_calls == 0
        assert upstream.requests == []
        accepted = client.post(
            ROUTE,
            json={"model": "gpt-image-2.6", "prompt": "cat"},
            headers={"Authorization": "Bearer proxy-secret"},
        )

    assert rejected.status_code == 401
    assert rejected.headers["www-authenticate"] == "Bearer"
    assert accepted.status_code == 200
    assert accepted.json() == UPSTREAM_RESULT
    assert len(upstream.requests) == 1


@pytest.mark.parametrize("expose_reasoning", [False, True])
def test_image_models_are_absent_from_discovery(expose_reasoning: bool) -> None:
    settings = Settings(api_key=None, expose_reasoning_models=expose_reasoning)
    with TestClient(create_app(settings)) as client:
        models = client.get("/v1/models")
        tags = client.get("/api/tags")

    assert models.status_code == tags.status_code == 200
    model_ids = {item["id"] for item in models.json()["data"]}
    tag_ids = {item["name"] for item in tags.json()["models"]}
    assert model_ids == tag_ids
    assert not any("image" in name for name in model_ids)
    assert model_ids.isdisjoint(IMAGE_MODELS)
