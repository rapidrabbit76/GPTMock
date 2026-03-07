"""Unit tests: app bootstrap, routing, Pydantic validation, and model registry.

These tests do NOT require ChatGPT credentials.
They verify that the application boots, routes are wired, Pydantic models
parse/reject correctly, and the model registry returns expected data.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from gptmock.app import create_app
from gptmock.schemas.requests import (
    ChatCompletionRequest,
    OllamaChatRequest,
    OllamaShowRequest,
    ResponsesCreateRequest,
    TextCompletionRequest,
)
from gptmock.services.model_registry import get_model_list, get_ollama_models, get_openai_models

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    """In-process TestClient — no credentials required for these tests."""
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestAppBootstrap:
    """Verify that create_app() produces a working FastAPI application."""

    def test_health_endpoint(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_root_endpoint(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_models_endpoint(self, client: TestClient) -> None:
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) > 0
        # Each model entry must have id and object fields
        for m in data["data"]:
            assert "id" in m, f"model entry missing 'id': {m}"
            assert m["object"] == "model"

    def test_ollama_version(self, client: TestClient) -> None:
        resp = client.get("/api/version")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
        assert isinstance(data["version"], str)

    def test_ollama_tags(self, client: TestClient) -> None:
        resp = client.get("/api/tags")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) > 0

    def test_ollama_show_valid_model(self, client: TestClient) -> None:
        resp = client.post("/api/show", json={"model": "gpt-5"})
        assert resp.status_code == 200
        data = resp.json()
        assert "details" in data
        assert "capabilities" in data

    def test_ollama_show_missing_model(self, client: TestClient) -> None:
        """POST /api/show without model field returns 422 (Pydantic validation)."""
        resp = client.post("/api/show", json={})
        assert resp.status_code == 422

    def test_ollama_show_empty_model(self, client: TestClient) -> None:
        """POST /api/show with empty string model returns 400."""
        resp = client.post("/api/show", json={"model": ""})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Pydantic validation: valid payloads parse, invalid ones reject
# ---------------------------------------------------------------------------


class TestPydanticRequestModels:
    """Verify Pydantic request models parse and validate correctly."""

    # -- ChatCompletionRequest --

    def test_chat_completion_valid_minimal(self) -> None:
        req = ChatCompletionRequest(model="gpt-5")
        assert req.model == "gpt-5"
        assert req.messages is None
        assert req.stream is False

    def test_chat_completion_with_messages(self) -> None:
        req = ChatCompletionRequest(
            model="gpt-5",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
        assert req.model == "gpt-5"
        assert len(req.messages) == 1
        assert req.stream is True

    def test_chat_completion_extra_fields_preserved(self) -> None:
        """Extra fields must be preserved (backward compatibility)."""
        req = ChatCompletionRequest(
            model="gpt-5",
            messages=[],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        dump = req.model_dump()
        assert dump["temperature"] == 0.7
        assert dump["response_format"] == {"type": "json_object"}

    def test_chat_completion_missing_model_rejects(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            ChatCompletionRequest()

    # -- TextCompletionRequest --

    def test_text_completion_valid(self) -> None:
        req = TextCompletionRequest(model="gpt-5", prompt="hello")
        assert req.model == "gpt-5"
        assert req.prompt == "hello"

    def test_text_completion_missing_model_rejects(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            TextCompletionRequest(prompt="hello")

    # -- ResponsesCreateRequest --

    def test_responses_create_valid(self) -> None:
        req = ResponsesCreateRequest(
            model="gpt-5",
            input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        )
        assert req.model == "gpt-5"
        assert isinstance(req.input, list)

    def test_responses_create_extra_fields(self) -> None:
        req = ResponsesCreateRequest(
            model="gpt-5",
            input=[],
            tools=[{"type": "web_search"}],
            instructions="Be helpful",
        )
        dump = req.model_dump()
        assert dump["tools"] == [{"type": "web_search"}]
        assert dump["instructions"] == "Be helpful"

    # -- OllamaShowRequest --

    def test_ollama_show_valid(self) -> None:
        req = OllamaShowRequest(model="gpt-5")
        assert req.model == "gpt-5"

    def test_ollama_show_missing_model_rejects(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            OllamaShowRequest()

    # -- OllamaChatRequest --

    def test_ollama_chat_valid(self) -> None:
        req = OllamaChatRequest(
            model="gpt-5",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert req.model == "gpt-5"
        assert req.stream is True  # Ollama defaults to streaming

    def test_ollama_chat_stream_false(self) -> None:
        req = OllamaChatRequest(
            model="gpt-5",
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
        )
        assert req.stream is False

    def test_ollama_chat_missing_messages_rejects(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            OllamaChatRequest(model="gpt-5")

    def test_ollama_chat_extra_fields(self) -> None:
        """Ollama extensions like tools, images must be preserved."""
        req = OllamaChatRequest(
            model="gpt-5",
            messages=[{"role": "user", "content": "hello"}],
            tools=[{"type": "function", "function": {"name": "test"}}],
            images=["base64data"],
        )
        dump = req.model_dump()
        assert "tools" in dump
        assert "images" in dump


# ---------------------------------------------------------------------------
# Pydantic validation via HTTP: invalid payloads get 422
# ---------------------------------------------------------------------------


class TestPydanticHTTPValidation:
    """Verify that invalid request bodies return 422 via FastAPI."""

    def test_chat_completions_missing_model(self, client: TestClient) -> None:
        resp = client.post("/v1/chat/completions", json={"messages": []})
        assert resp.status_code == 422

    def test_chat_completions_invalid_json(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat/completions",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_completions_missing_model(self, client: TestClient) -> None:
        resp = client.post("/v1/completions", json={"prompt": "hello"})
        assert resp.status_code == 422

    def test_responses_missing_model(self, client: TestClient) -> None:
        resp = client.post("/v1/responses", json={"input": []})
        assert resp.status_code == 422

    def test_ollama_chat_missing_model(self, client: TestClient) -> None:
        resp = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 422

    def test_ollama_chat_missing_messages(self, client: TestClient) -> None:
        resp = client.post("/api/chat", json={"model": "gpt-5"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class TestModelRegistry:
    """Verify model registry returns consistent data."""

    def test_get_model_list_non_empty(self) -> None:
        models = get_model_list(expose_reasoning=False)
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(m, str) for m in models)

    def test_get_model_list_with_reasoning(self) -> None:
        without = get_model_list(expose_reasoning=False)
        with_reasoning = get_model_list(expose_reasoning=True)
        # With reasoning variants exposed, list should be >= base list
        assert len(with_reasoning) >= len(without)

    def test_get_openai_models_structure(self) -> None:
        models = get_openai_models(expose_reasoning=False)
        assert isinstance(models, list)
        for m in models:
            assert isinstance(m, dict)
            assert "id" in m
            assert m["object"] == "model"
            assert "owned_by" in m

    def test_get_ollama_models_structure(self) -> None:
        models = get_ollama_models(expose_reasoning=False)
        assert isinstance(models, list)
        for m in models:
            assert isinstance(m, dict)
            assert "name" in m
            assert "model" in m

    def test_gpt5_in_model_list(self) -> None:
        models = get_model_list(expose_reasoning=False)
        assert "gpt-5" in models, f"gpt-5 not in model list: {models}"


# ---------------------------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------------------------


class TestCORSConfig:
    """Verify CORS middleware is applied."""

    def test_cors_headers_present(self, client: TestClient) -> None:
        resp = client.options(
            "/v1/models",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" in resp.headers


# ---------------------------------------------------------------------------
# Structured output (response_format / json_schema)
# ---------------------------------------------------------------------------


class TestStructuredOutput:
    """Verify structured output helpers without upstream auth.

    Tests cover:
    - _build_text_format: response_format -> upstream text format conversion
    - _is_strict_json_text_format (chat.py): checks chat-style text_format dicts
    - _is_strict_json_text_format (responses.py): checks responses-style text_obj dicts
    """

    # -- _build_text_format (chat.py) ------------------------------------------

    def test_build_text_format_json_object(self) -> None:
        from gptmock.services.chat import _build_text_format

        result = _build_text_format({"type": "json_object"})
        assert result == {"type": "json_object"}

    def test_build_text_format_json_schema_valid(self) -> None:
        from gptmock.services.chat import _build_text_format

        result = _build_text_format({
            "type": "json_schema",
            "json_schema": {
                "name": "my_schema",
                "schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
            },
        })
        assert result is not None
        assert result["type"] == "json_schema"
        assert result["name"] == "my_schema"
        assert result["schema"] == {"type": "object", "properties": {"x": {"type": "integer"}}}
        assert "strict" not in result

    def test_build_text_format_json_schema_with_strict(self) -> None:
        from gptmock.services.chat import _build_text_format

        result = _build_text_format({
            "type": "json_schema",
            "json_schema": {
                "name": "strict_schema",
                "schema": {"type": "object"},
                "strict": True,
            },
        })
        assert result is not None
        assert result["strict"] is True

    def test_build_text_format_json_schema_missing_name_rejects(self) -> None:
        from gptmock.services.chat import ChatCompletionError, _build_text_format

        with pytest.raises(ChatCompletionError):
            _build_text_format({
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object"}},
            })

    def test_build_text_format_json_schema_missing_schema_rejects(self) -> None:
        from gptmock.services.chat import ChatCompletionError, _build_text_format

        with pytest.raises(ChatCompletionError):
            _build_text_format({
                "type": "json_schema",
                "json_schema": {"name": "broken"},
            })

    def test_build_text_format_json_schema_empty_name_rejects(self) -> None:
        from gptmock.services.chat import ChatCompletionError, _build_text_format

        with pytest.raises(ChatCompletionError):
            _build_text_format({
                "type": "json_schema",
                "json_schema": {"name": "  ", "schema": {"type": "object"}},
            })

    def test_build_text_format_text_type(self) -> None:
        from gptmock.services.chat import _build_text_format

        assert _build_text_format({"type": "text"}) == {"type": "text"}

    def test_build_text_format_unsupported_type_rejects(self) -> None:
        from gptmock.services.chat import ChatCompletionError, _build_text_format

        with pytest.raises(ChatCompletionError):
            _build_text_format({"type": "xml"})

    def test_build_text_format_none_input(self) -> None:
        from gptmock.services.chat import _build_text_format

        assert _build_text_format(None) is None

    def test_build_text_format_non_dict_input(self) -> None:
        from gptmock.services.chat import _build_text_format

        assert _build_text_format("json") is None

    def test_build_text_format_missing_type_key(self) -> None:
        from gptmock.services.chat import _build_text_format

        assert _build_text_format({"format": "json"}) is None

    # -- _is_strict_json_text_format (chat.py) ---------------------------------

    @pytest.mark.parametrize(
        ("text_format", "expected"),
        [
            ({"type": "json_schema"}, True),
            ({"type": "json_object"}, True),
            ({"type": "text"}, False),
            ({}, False),
            (None, False),
            ("json_schema", False),
        ],
    )
    def test_chat_is_strict_json_text_format(
        self, text_format: object, expected: bool,
    ) -> None:
        from gptmock.services.chat import _is_strict_json_text_format

        assert _is_strict_json_text_format(text_format) is expected

    # -- _is_strict_json_text_format (responses.py) ----------------------------
    # NOTE: responses.py version checks text_obj["format"]["type"] (nested dict)

    @pytest.mark.parametrize(
        ("text_obj", "expected"),
        [
            ({"format": {"type": "json_schema"}}, True),
            ({"format": {"type": "json_object"}}, True),
            ({"format": {"type": "text"}}, False),
            ({"format": {}}, False),
            ({}, False),
            (None, False),
            ("json_schema", False),
        ],
    )
    def test_responses_is_strict_json_text_format(
        self, text_obj: object, expected: bool,
    ) -> None:
        from gptmock.services.responses import (
            _is_strict_json_text_format as resp_is_strict,
        )

        assert resp_is_strict(text_obj) is expected

    # -- Pydantic models accept response_format (HTTP level) -------------------

    def test_chat_completion_with_json_schema_format(
        self, client: TestClient,
    ) -> None:
        """POST with response_format passes Pydantic (not 422)."""
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5",
                "messages": [{"role": "user", "content": "return JSON"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "test_schema",
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                        },
                        "strict": True,
                    },
                },
            },
        )
        assert resp.status_code != 422

    def test_responses_with_json_schema_text(
        self, client: TestClient,
    ) -> None:
        """POST /v1/responses with text.format passes Pydantic."""
        resp = client.post(
            "/v1/responses",
            json={
                "model": "gpt-5",
                "input": "return JSON",
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "test_schema",
                        "schema": {"type": "object"},
                    },
                },
            },
        )
        assert resp.status_code != 422
