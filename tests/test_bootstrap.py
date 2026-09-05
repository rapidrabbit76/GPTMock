"""Unit tests: app bootstrap, routing, Pydantic validation, and model registry.

These tests do NOT require ChatGPT credentials.
They verify that the application boots, routes are wired, Pydantic models
parse/reject correctly, and the model registry returns expected data.
"""

from __future__ import annotations

import logging

import pytest
from starlette.testclient import TestClient

from gptmock.app import create_app
from gptmock.core.settings import Settings
from gptmock.schemas.requests import (
    ChatCompletionRequest,
    OllamaChatRequest,
    OllamaGenerateRequest,
    OllamaShowRequest,
    ResponsesCreateRequest,
    TextCompletionRequest,
)
from gptmock.services.model_registry import (
    apply_model_overrides,
    get_model_list,
    get_ollama_models,
    get_openai_models,
    normalize_model_name,
)

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------


def test_verbose_settings_enable_package_debug_logging() -> None:
    package_logger = logging.getLogger("gptmock")
    original_level = package_logger.level
    original_handlers = list(package_logger.handlers)
    try:
        create_app(Settings(verbose=True))
        assert package_logger.level == logging.DEBUG
        assert any(
            getattr(handler, "_gptmock_verbose_handler", False)
            for handler in package_logger.handlers
        )
    finally:
        package_logger.setLevel(original_level)
        package_logger.handlers[:] = original_handlers


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

    def test_root_head_supports_ollama_cli_probe(self, client: TestClient) -> None:
        resp = client.head("/")
        assert resp.status_code == 200

    def test_models_endpoint(self, client: TestClient) -> None:
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) > 0
        for m in data["data"]:
            assert "id" in m, f"model entry missing 'id': {m}"
            assert m["object"] == "model"
            assert "reasoning" in m, f"model entry missing 'reasoning': {m}"
            r = m["reasoning"]
            assert isinstance(r.get("supported_efforts"), list)
            assert r.get("default_effort") in r["supported_efforts"]

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
        resp = client.post("/api/show", json={"model": "gpt-5.4"})
        assert resp.status_code == 200
        data = resp.json()
        assert "details" in data
        assert "capabilities" in data
        assert data["details"]["format"] == "remote"
        assert data["model_info"]["gptmock.remote"] is True

    def test_ollama_show_accepts_hidden_fast_alias(self, client: TestClient) -> None:
        resp = client.post("/api/show", json={"model": "gpt-5.6-luna-fast"})
        assert resp.status_code == 200
        model_info = resp.json()["model_info"]
        assert model_info["gptmock.upstream_model"] == "gpt-5.6-luna"
        assert model_info["gptmock.request_overrides"] == {"service_tier": "priority"}

    def test_ollama_show_accepts_cli_name_field(self, client: TestClient) -> None:
        resp = client.post("/api/show", json={"name": "gpt-5.6-luna"})
        assert resp.status_code == 200
        assert resp.json()["model_info"]["gptmock.upstream_model"] == "gpt-5.6-luna"

    def test_ollama_show_unknown_model(self, client: TestClient) -> None:
        resp = client.post("/api/show", json={"model": "definitely-not-a-model"})
        assert resp.status_code == 404

    def test_ollama_show_missing_model(self, client: TestClient) -> None:
        resp = client.post("/api/show", json={})
        assert resp.status_code == 400

    def test_ollama_show_empty_model(self, client: TestClient) -> None:
        """POST /api/show with empty string model returns 400."""
        resp = client.post("/api/show", json={"model": ""})
        assert resp.status_code == 400

    def test_ollama_chat_rejects_unsupported_options_as_client_error(
        self, client: TestClient,
    ) -> None:
        resp = client.post(
            "/api/chat",
            json={
                "model": "gpt-5.6-luna",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
                "options": {"temperature": 0.2},
            },
        )
        assert resp.status_code == 400
        assert resp.json() == {
            "error": "Unsupported Ollama option(s): options.temperature",
        }


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

    def test_ollama_generate_request_preserves_native_fields(self) -> None:
        req = OllamaGenerateRequest(
            model="gpt-5.6-luna",
            prompt="hello",
            think="high",
            options={"num_predict": 128},
        )
        assert req.prompt == "hello"
        assert req.model_extra == {
            "think": "high",
            "options": {"num_predict": 128},
        }

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

    def test_ollama_show_missing_model_defaults_empty(self) -> None:
        req = OllamaShowRequest()
        assert req.model == ""

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
            assert "reasoning" in m
            r = m["reasoning"]
            assert isinstance(r["supported_efforts"], list)
            assert len(r["supported_efforts"]) > 0
            assert r["default_effort"] in r["supported_efforts"]

    def test_get_openai_models_reasoning_per_family(self) -> None:
        models = {m["id"]: m for m in get_openai_models(expose_reasoning=False)}

        assert "xhigh" in models["gpt-5.4"]["reasoning"]["supported_efforts"]
        assert "minimal" not in models["gpt-5.4"]["reasoning"]["supported_efforts"]
        assert models["gpt-5.6-luna"]["reasoning"]["supported_efforts"] == [
            "none", "low", "medium", "high", "xhigh", "max",
        ]

    def test_get_openai_models_reasoning_for_variants(self) -> None:
        models = {m["id"]: m for m in get_openai_models(expose_reasoning=True)}

        gpt54_high = models["gpt-5.4-high"]["reasoning"]
        assert gpt54_high["supported_efforts"] == ["low", "medium", "high", "xhigh"]
        assert gpt54_high["preset_effort"] == "high"

        gpt54_xhigh = models["gpt-5.4-xhigh"]["reasoning"]
        assert gpt54_xhigh["supported_efforts"] == ["low", "medium", "high", "xhigh"]
        assert gpt54_xhigh["preset_effort"] == "xhigh"

    def test_get_openai_models_default_effort_reflects_setting(self) -> None:
        models_medium = {m["id"]: m for m in get_openai_models(default_effort="medium")}
        assert models_medium["gpt-5.4"]["reasoning"]["default_effort"] == "medium"

        models_high = {m["id"]: m for m in get_openai_models(default_effort="high")}
        assert models_high["gpt-5.4"]["reasoning"]["default_effort"] == "high"

        models_xhigh = {m["id"]: m for m in get_openai_models(default_effort="xhigh")}
        assert models_xhigh["gpt-5.4"]["reasoning"]["default_effort"] == "xhigh"

    def test_get_openai_models_endpoint_reflects_configured_default_effort(
        self, client: TestClient
    ) -> None:
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        models = {m["id"]: m for m in data["data"]}
        assert models["gpt-5.4"]["reasoning"]["default_effort"] in {
            "low", "medium", "high", "xhigh",
        }

    def test_strip_effort_suffix_handles_both_separators(self) -> None:
        from gptmock.services.reasoning import strip_effort_suffix

        assert strip_effort_suffix("gpt-5-high") == "gpt-5"
        assert strip_effort_suffix("gpt-5_high") == "gpt-5"
        assert strip_effort_suffix("gpt-5_minimal") == "gpt-5"
        assert strip_effort_suffix("gpt-5.4-xhigh") == "gpt-5.4"
        assert strip_effort_suffix("gpt-5") == "gpt-5"

    def test_variant_detection_requires_known_base(self) -> None:
        from gptmock.services.model_registry import _detect_preset_effort

        assert _detect_preset_effort("gpt-5.4-high") == "high"
        assert _detect_preset_effort("gpt-5.4_medium") == "medium"
        assert _detect_preset_effort("gpt-5.4") is None
        assert _detect_preset_effort("unknown-model-high") is None

    def test_allowed_efforts_handles_variant_ids(self) -> None:
        from gptmock.services.reasoning import allowed_efforts_for_model

        base_efforts = allowed_efforts_for_model("gpt-5")
        variant_efforts = allowed_efforts_for_model("gpt-5-high")
        assert variant_efforts == base_efforts, (
            "Variant IDs must yield the same family set as their base"
        )

    def test_get_ollama_models_structure(self) -> None:
        models = get_ollama_models(expose_reasoning=False)
        assert isinstance(models, list)
        for m in models:
            assert isinstance(m, dict)
            assert "name" in m
            assert "model" in m
            assert m["details"]["format"] == "remote"
            assert m["size"] == 0
            assert m["digest"] == ""
            assert m["modified_at"] == "2026-09-05T00:00:00Z"
            assert isinstance(m["remote_model"], str)
            assert m["capabilities"] == ["completion", "tools", "thinking"]

    def test_rejected_models_are_not_advertised(self) -> None:
        models = get_model_list(expose_reasoning=False)
        rejected = {
            "gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5-codex", "gpt-5.1-codex",
            "gpt-5.1-codex-mini", "gpt-5.1-codex-max", "gpt-5.2-codex", "gpt-5.3-codex",
        }
        assert rejected.isdisjoint(models)

    def test_normalize_codex_mini_aliases(self) -> None:
        assert normalize_model_name("codex-mini") == "gpt-5.1-codex-mini"
        assert normalize_model_name("gpt5.1-codex-mini") == "gpt-5.1-codex-mini"
        assert normalize_model_name("gpt-5.1-codex-mini") == "gpt-5.1-codex-mini"
        assert normalize_model_name("gpt-5.1-codex-mini-latest") == "gpt-5.1-codex-mini"


class TestFastModelVariants:
    """Verify synthetic 'fast' aliases translate to base model + service_tier=priority."""

    def test_fast_variants_are_not_advertised(self) -> None:
        models = get_model_list(expose_reasoning=False)
        assert "gpt-5.5" in models
        assert "gpt-5.6" in models
        for family in ("sol", "terra", "luna"):
            assert f"gpt-5.6-{family}" in models
        assert not any("-fast" in model for model in models)

    def test_fast_reasoning_variants_are_not_advertised(self) -> None:
        models = get_model_list(expose_reasoning=True)
        assert not any("-fast" in model for model in models)

    def test_verified_variants_reasoning_metadata(self) -> None:
        models = {m["id"]: m for m in get_openai_models(expose_reasoning=False)}
        expected_efforts = ["low", "medium", "high", "xhigh"]
        gpt56_efforts = ["none", "low", "medium", "high", "xhigh", "max"]
        assert models["gpt-5.5"]["reasoning"]["supported_efforts"] == expected_efforts
        for family in ("sol", "terra", "luna"):
            assert models[f"gpt-5.6-{family}"]["reasoning"]["supported_efforts"] == gpt56_efforts

    def test_normalize_fast_aliases(self) -> None:
        assert normalize_model_name("gpt-5.4-fast") == "gpt-5.4-fast"
        assert normalize_model_name("gpt5.4-fast") == "gpt-5.4-fast"
        assert normalize_model_name("gpt-5.4-fast-latest") == "gpt-5.4-fast"
        assert normalize_model_name("gpt-5.5") == "gpt-5.5"
        assert normalize_model_name("gpt5.5") == "gpt-5.5"
        assert normalize_model_name("gpt-5.5-latest") == "gpt-5.5"
        assert normalize_model_name("gpt-5.6") == "gpt-5.6"
        assert normalize_model_name("gpt5.6") == "gpt-5.6"
        assert normalize_model_name("gpt-5.6-latest") == "gpt-5.6"
        for family in ("sol", "terra", "luna"):
            assert normalize_model_name(f"gpt-5.6-{family}") == f"gpt-5.6-{family}"
            assert normalize_model_name(f"gpt5.6-{family}") == f"gpt-5.6-{family}"
            assert normalize_model_name(f"gpt-5.6-{family}-latest") == f"gpt-5.6-{family}"
        assert normalize_model_name("gpt-5.5-fast") == "gpt-5.5-fast"
        assert normalize_model_name("gpt5.5-fast") == "gpt-5.5-fast"
        assert normalize_model_name("gpt-5.5-fast-latest") == "gpt-5.5-fast"
        assert normalize_model_name("gpt-5.6-fast") == "gpt-5.6-fast"
        assert normalize_model_name("gpt5.6-fast") == "gpt-5.6-fast"
        assert normalize_model_name("gpt-5.6-fast-latest") == "gpt-5.6-fast"
        for family in ("sol", "terra", "luna"):
            assert normalize_model_name(f"gpt-5.6-{family}-fast") == f"gpt-5.6-{family}-fast"
            assert normalize_model_name(f"gpt5.6-{family}-fast") == f"gpt-5.6-{family}-fast"
            assert normalize_model_name(f"gpt-5.6-{family}-fast-latest") == f"gpt-5.6-{family}-fast"
        assert normalize_model_name("gpt-5.4-mini-fast") == "gpt-5.4-mini-fast"
        assert normalize_model_name("gpt5.4-mini-fast") == "gpt-5.4-mini-fast"
        assert normalize_model_name("gpt-5.4-mini-fast-latest") == "gpt-5.4-mini-fast"

    def test_normalize_fast_with_effort_suffix_strips_effort(self) -> None:
        assert normalize_model_name("gpt-5.4-fast-medium") == "gpt-5.4-fast"
        assert normalize_model_name("gpt-5.4-fast-xhigh") == "gpt-5.4-fast"
        assert normalize_model_name("gpt-5.5-fast-medium") == "gpt-5.5-fast"
        assert normalize_model_name("gpt-5.5-fast-xhigh") == "gpt-5.5-fast"
        assert normalize_model_name("gpt-5.6-sol-fast-medium") == "gpt-5.6-sol-fast"
        assert normalize_model_name("gpt-5.6-sol-fast-xhigh") == "gpt-5.6-sol-fast"
        assert normalize_model_name("gpt-5.4-mini-fast-high") == "gpt-5.4-mini-fast"
        assert normalize_model_name("gpt-5.4-mini-fast-low") == "gpt-5.4-mini-fast"

    def test_resolve_upstream_model_for_fast_aliases(self) -> None:
        from gptmock.services.model_registry import resolve_upstream_model

        priority = {"service_tier": "priority"}
        assert resolve_upstream_model("gpt-5.4-fast") == ("gpt-5.4", priority)
        assert resolve_upstream_model("gpt-5.5-fast") == ("gpt-5.5", priority)
        assert resolve_upstream_model("gpt-5.6-fast") == ("gpt-5.6-sol", priority)
        assert resolve_upstream_model("gpt-5.6-sol-fast") == ("gpt-5.6-sol", priority)
        assert resolve_upstream_model("gpt-5.6-terra-fast") == ("gpt-5.6-terra", priority)
        assert resolve_upstream_model("gpt-5.6-luna-fast") == ("gpt-5.6-luna", priority)
        assert resolve_upstream_model("gpt-5.4-mini-fast") == ("gpt-5.4-mini", priority)

    def test_apply_model_overrides_sets_service_tier(self) -> None:
        payload = {"service_tier": "default"}

        apply_model_overrides(payload, {"service_tier": "priority"})

        assert payload["service_tier"] == "priority"

    def test_resolve_upstream_model_passes_through_regular_models(self) -> None:
        from gptmock.services.model_registry import resolve_upstream_model

        assert resolve_upstream_model("gpt-5.4") == ("gpt-5.4", {})
        assert resolve_upstream_model("gpt-5.5") == ("gpt-5.5", {})
        assert resolve_upstream_model("gpt-5.6") == ("gpt-5.6-sol", {})
        for family in ("sol", "terra", "luna"):
            model = f"gpt-5.6-{family}"
            assert resolve_upstream_model(model) == (model, {})
        assert resolve_upstream_model("gpt-5.4-mini") == ("gpt-5.4-mini", {})
        assert resolve_upstream_model("gpt-5") == ("gpt-5", {})
        assert resolve_upstream_model("gpt-5.1-codex-max") == ("gpt-5.1-codex-max", {})

    def test_fast_variants_allowed_efforts_match_base(self) -> None:
        from gptmock.services.reasoning import allowed_efforts_for_model

        base_efforts = allowed_efforts_for_model("gpt-5.4")
        gpt55_efforts = allowed_efforts_for_model("gpt-5.5")
        fast_efforts = allowed_efforts_for_model("gpt-5.4-fast")
        gpt55_fast_efforts = allowed_efforts_for_model("gpt-5.5-fast")
        mini_fast_efforts = allowed_efforts_for_model("gpt-5.4-mini-fast")
        gpt56_efforts = {"none", "low", "medium", "high", "xhigh", "max"}
        assert gpt55_efforts == base_efforts
        assert allowed_efforts_for_model("gpt-5.6") == gpt56_efforts
        assert allowed_efforts_for_model("gpt-5.6-fast") == gpt56_efforts
        for family in ("sol", "terra", "luna"):
            assert allowed_efforts_for_model(f"gpt-5.6-{family}") == gpt56_efforts
            assert allowed_efforts_for_model(f"gpt-5.6-{family}-fast") == gpt56_efforts
        assert fast_efforts == base_efforts
        assert gpt55_fast_efforts == base_efforts
        assert mini_fast_efforts == base_efforts

    def test_gpt56_reasoning_requires_known_family_variant(self) -> None:
        from gptmock.services.reasoning import allowed_efforts_for_model

        assert "minimal" not in allowed_efforts_for_model("gpt-5.6")
        assert "minimal" in allowed_efforts_for_model("gpt-5.6-unknown")

    def test_fast_variants_use_base_instructions_not_codex(self) -> None:
        from gptmock.services.model_registry import get_instructions_for_model

        base = "base instructions"
        codex = "codex instructions"
        assert get_instructions_for_model("gpt-5.4-fast", base, codex) == base
        assert get_instructions_for_model("gpt-5.5-fast", base, codex) == base
        assert get_instructions_for_model("gpt-5.6-sol-fast", base, codex) == base
        assert get_instructions_for_model("gpt-5.4-mini-fast", base, codex) == base

    def test_fast_variants_hidden_from_openai_models_endpoint(self, client: TestClient) -> None:
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        ids = {m["id"] for m in resp.json()["data"]}
        assert "gpt-5.5" in ids
        assert "gpt-5.6" in ids
        for family in ("sol", "terra", "luna"):
            assert f"gpt-5.6-{family}" in ids
        assert not any("-fast" in model_id for model_id in ids)

    def test_fast_variants_hidden_from_ollama_tags_endpoint(self, client: TestClient) -> None:
        resp = client.get("/api/tags")
        assert resp.status_code == 200
        names = {m["name"] for m in resp.json()["models"]}
        assert "gpt-5.5" in names
        assert "gpt-5.6" in names
        for family in ("sol", "terra", "luna"):
            assert f"gpt-5.6-{family}" in names
        assert not any("-fast" in name for name in names)


# ---------------------------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------------------------


class TestCORSConfig:
    """Verify browser access is opt-in."""

    def test_cors_headers_absent_by_default(self, client: TestClient) -> None:
        resp = client.options(
            "/v1/models",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" not in resp.headers

    def test_configured_cors_origin_is_allowed(self) -> None:
        app = create_app(Settings(cors_origins="http://localhost:3000"))
        with TestClient(app, raise_server_exceptions=False) as configured_client:
            resp = configured_client.options(
                "/v1/models",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "GET",
                },
            )
        assert resp.headers["access-control-allow-origin"] == "http://localhost:3000"


class TestProxyAuthentication:
    def test_api_key_protects_model_routes_but_not_health(self) -> None:
        app = create_app(Settings(api_key="proxy-secret"))
        with TestClient(app, raise_server_exceptions=False) as protected_client:
            assert protected_client.get("/health").status_code == 200
            rejected = protected_client.get("/v1/models")
            accepted = protected_client.get(
                "/v1/models",
                headers={"Authorization": "Bearer proxy-secret"},
            )
        assert rejected.status_code == 401
        assert rejected.headers["www-authenticate"] == "Bearer"
        assert accepted.status_code == 200


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
