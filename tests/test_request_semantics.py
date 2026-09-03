from __future__ import annotations

import importlib
import json
from typing import Any

import httpx
import pytest

from gptmock.core.settings import Settings
from gptmock.infra.sse import sse_translate_chat, sse_translate_text


def _completed_response(response: dict[str, Any]) -> httpx.Response:
    event = {"type": "response.completed", "response": response}
    return httpx.Response(200, content=f"data: {json.dumps(event)}\n\n".encode())


def _incomplete_response(response: dict[str, Any]) -> httpx.Response:
    event = {"type": "response.incomplete", "response": response}
    return httpx.Response(200, content=f"data: {json.dumps(event)}\n\n".encode())


@pytest.mark.asyncio
async def test_chat_request_preserves_roles_tools_and_supported_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_module = importlib.import_module("gptmock.services.chat")
    captured: list[dict[str, Any]] = []

    async def fake_auth() -> tuple[str, str]:
        return "token", "account"

    async def fake_send(payload: dict[str, Any], *args: Any, **kwargs: Any) -> httpx.Response:
        del args, kwargs
        captured.append(payload)
        return _completed_response(
            {
                "id": "resp_chat",
                "status": "completed",
                "model": "gpt-5.6-sol",
                "service_tier": "default",
            },
        )

    monkeypatch.setattr(chat_module, "get_effective_chatgpt_auth", fake_auth)
    monkeypatch.setattr(chat_module, "send_upstream_request", fake_send)

    payload = {
        "model": "gpt-5.6",
        "messages": [
            {"role": "system", "content": "system authority"},
            {"role": "developer", "content": "developer authority"},
            {"role": "user", "content": "hello"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "strict": True,
                    "parameters": {"type": "object"},
                },
            },
        ],
        "tool_choice": "required",
        "service_tier": "priority",
        "reasoning_effort": "max",
    }
    async with httpx.AsyncClient() as client:
        result, is_stream = await chat_module.process_chat_completion(
            payload, Settings(), client,
        )

    assert is_stream is False
    assert result["id"] == "resp_chat"
    assert result["model"] == "gpt-5.6-sol"
    assert result["service_tier"] == "default"
    assert captured[0]["model"] == "gpt-5.6-sol"
    assert [item["role"] for item in captured[0]["input"]] == ["system", "developer", "user"]
    assert captured[0]["tools"][0]["strict"] is True
    assert captured[0]["tool_choice"] == "required"
    assert captured[0]["service_tier"] == "priority"
    assert captured[0]["reasoning"] == {
        "effort": "max",
        "summary": "auto",
    }


@pytest.mark.asyncio
async def test_conflicting_chat_reasoning_fields_are_rejected() -> None:
    chat_module = importlib.import_module("gptmock.services.chat")
    payload = {
        "model": "gpt-5.6-sol",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning": {"effort": "low"},
        "reasoning_effort": "high",
    }

    async with httpx.AsyncClient() as client:
        with pytest.raises(chat_module.ChatCompletionError) as exc_info:
            await chat_module.process_chat_completion(payload, Settings(), client)

    assert exc_info.value.status_code == 400
    assert exc_info.value.error_data["error"]["code"] == "conflicting_parameters"


def test_chat_reasoning_effort_overrides_synthetic_model_suffix() -> None:
    chat_module = importlib.import_module("gptmock.services.chat")
    result = chat_module._chat_reasoning_overrides(
        {"reasoning_effort": "low"},
        {"effort": "high"},
    )
    assert result == {"effort": "low"}


@pytest.mark.asyncio
async def test_responses_request_preserves_input_options_and_actual_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses_module = importlib.import_module("gptmock.services.responses")
    captured: list[dict[str, Any]] = []

    async def fake_auth() -> tuple[str, str]:
        return "token", "account"

    final_response = {
        "id": "resp_semantics",
        "object": "response",
        "status": "completed",
        "model": "gpt-5.6-luna",
        "service_tier": "default",
        "previous_response_id": "resp_previous",
        "reasoning": {"effort": "high"},
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hello", "annotations": []}],
            },
        ],
    }

    async def fake_send(payload: dict[str, Any], *args: Any, **kwargs: Any) -> httpx.Response:
        del args, kwargs
        captured.append(payload)
        return _completed_response(final_response)

    monkeypatch.setattr(responses_module, "get_effective_chatgpt_auth", fake_auth)
    monkeypatch.setattr(responses_module, "send_upstream_request", fake_send)

    payload = {
        "model": "gpt-5.6-luna",
        "input": "hello as a string",
        "tool_choice": "required",
        "previous_response_id": "resp_previous",
        "service_tier": "priority",
        "metadata": {"trace": "semantic-test"},
        "reasoning": {"effort": "max"},
    }
    async with httpx.AsyncClient() as client:
        result, is_stream = await responses_module.process_responses_api(
            payload, Settings(), client,
        )

    assert is_stream is False
    assert captured[0]["input"][0]["content"][0]["text"] == "hello as a string"
    assert captured[0]["model"] == "gpt-5.6-luna"
    assert captured[0]["tool_choice"] == "required"
    assert captured[0]["previous_response_id"] == "resp_previous"
    assert captured[0]["service_tier"] == "priority"
    assert captured[0]["metadata"] == {"trace": "semantic-test"}
    assert captured[0]["reasoning"] == {
        "effort": "max",
        "summary": "auto",
    }
    assert result["model"] == "gpt-5.6-luna"
    assert result["service_tier"] == "default"
    assert result["previous_response_id"] == "resp_previous"
    assert result["reasoning"] == {"effort": "high"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module_name", "processor_name", "parameter", "base_payload"),
    [
        (
            "gptmock.services.chat",
            "process_chat_completion",
            "max_completion_tokens",
            {"model": "gpt-5.6-sol", "messages": [{"role": "user", "content": "hello"}]},
        ),
        (
            "gptmock.services.chat",
            "process_chat_completion",
            "max_tokens",
            {"model": "gpt-5.6-sol", "messages": [{"role": "user", "content": "hello"}]},
        ),
        (
            "gptmock.services.chat",
            "process_text_completion",
            "max_tokens",
            {"model": "gpt-5.6-sol", "prompt": "hello"},
        ),
        (
            "gptmock.services.responses",
            "process_responses_api",
            "max_output_tokens",
            {"model": "gpt-5.6-sol", "input": "hello"},
        ),
    ],
)
async def test_output_token_limits_are_rejected_in_strict_policy(
    module_name: str,
    processor_name: str,
    parameter: str,
    base_payload: dict[str, Any],
) -> None:
    module = importlib.import_module(module_name)
    processor = getattr(module, processor_name)
    payload = {**base_payload, parameter: 32}

    async with httpx.AsyncClient() as client:
        with pytest.raises(module.ChatCompletionError) as exc_info:
            await processor(payload, Settings(output_token_policy="reject"), client)

    error = exc_info.value
    assert error.status_code == 400
    assert error.error_data["error"] == {
        "message": f"Unsupported parameter: {parameter}",
        "type": "invalid_request_error",
        "param": parameter,
        "code": "unsupported_parameter",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module_name", "processor_name", "parameter", "base_payload"),
    [
        (
            "gptmock.services.chat",
            "process_chat_completion",
            "max_completion_tokens",
            {"model": "gpt-5.6-sol", "messages": [{"role": "user", "content": "hello"}]},
        ),
        (
            "gptmock.services.chat",
            "process_chat_completion",
            "max_tokens",
            {"model": "gpt-5.6-sol", "messages": [{"role": "user", "content": "hello"}]},
        ),
        (
            "gptmock.services.chat",
            "process_text_completion",
            "max_tokens",
            {"model": "gpt-5.6-sol", "prompt": "hello"},
        ),
        (
            "gptmock.services.responses",
            "process_responses_api",
            "max_output_tokens",
            {"model": "gpt-5.6-sol", "input": "hello"},
        ),
    ],
)
async def test_output_token_limits_are_omitted_before_upstream_by_default(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    module_name: str,
    processor_name: str,
    parameter: str,
    base_payload: dict[str, Any],
) -> None:
    module = importlib.import_module(module_name)
    processor = getattr(module, processor_name)
    captured: list[dict[str, Any]] = []

    async def fake_auth() -> tuple[str, str]:
        return "token", "account"

    async def fake_send(payload: dict[str, Any], *args: Any, **kwargs: Any) -> httpx.Response:
        del args, kwargs
        captured.append(payload)
        return _completed_response(
            {
                "id": "resp_omit",
                "status": "completed",
                "model": "gpt-5.6-sol",
                "output": [],
            },
        )

    monkeypatch.setattr(module, "get_effective_chatgpt_auth", fake_auth)
    monkeypatch.setattr(module, "send_upstream_request", fake_send)

    with caplog.at_level("WARNING"):
        async with httpx.AsyncClient() as client:
            await processor({**base_payload, parameter: 32}, Settings(), client)

    assert captured
    assert parameter not in captured[0]
    assert parameter in caplog.text


@pytest.mark.asyncio
async def test_rejected_responses_tool_is_not_silently_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_module = importlib.import_module("gptmock.services.chat")
    calls = 0

    async def fake_auth() -> tuple[str, str]:
        return "token", "account"

    async def fake_send(payload: dict[str, Any], *args: Any, **kwargs: Any) -> httpx.Response:
        nonlocal calls
        del payload, args, kwargs
        calls += 1
        return httpx.Response(400, json={"error": {"message": "tool rejected"}})

    monkeypatch.setattr(chat_module, "get_effective_chatgpt_auth", fake_auth)
    monkeypatch.setattr(chat_module, "send_upstream_request", fake_send)

    async with httpx.AsyncClient() as client:
        with pytest.raises(chat_module.ChatCompletionError, match="tool rejected"):
            await chat_module.process_chat_completion(
                {
                    "model": "gpt-5.6-luna",
                    "messages": [{"role": "user", "content": "search"}],
                    "responses_tools": [{"type": "web_search"}],
                },
                Settings(),
                client,
            )
    assert calls == 1


@pytest.mark.asyncio
async def test_responses_non_stream_preserves_incomplete_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses_module = importlib.import_module("gptmock.services.responses")

    async def fake_auth() -> tuple[str, str]:
        return "token", "account"

    async def fake_send(payload: dict[str, Any], *args: Any, **kwargs: Any) -> httpx.Response:
        del payload, args, kwargs
        return _incomplete_response(
            {
                "id": "resp_incomplete",
                "status": "incomplete",
                "model": "gpt-5.6-luna",
                "service_tier": "default",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [],
            },
        )

    monkeypatch.setattr(responses_module, "get_effective_chatgpt_auth", fake_auth)
    monkeypatch.setattr(responses_module, "send_upstream_request", fake_send)

    async with httpx.AsyncClient() as client:
        result, is_stream = await responses_module.process_responses_api(
            {"model": "gpt-5.6-luna", "input": "hello"},
            Settings(),
            client,
        )

    assert is_stream is False
    assert result["status"] == "incomplete"
    assert result["incomplete_details"] == {"reason": "max_output_tokens"}
    assert result["model"] == "gpt-5.6-luna"
    assert result["service_tier"] == "default"


@pytest.mark.asyncio
async def test_chat_non_stream_maps_incomplete_to_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat_module = importlib.import_module("gptmock.services.chat")

    async def fake_auth() -> tuple[str, str]:
        return "token", "account"

    async def fake_send(payload: dict[str, Any], *args: Any, **kwargs: Any) -> httpx.Response:
        del payload, args, kwargs
        return _incomplete_response(
            {
                "id": "resp_incomplete",
                "status": "incomplete",
                "model": "gpt-5.6-terra",
                "service_tier": "default",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
        )

    monkeypatch.setattr(chat_module, "get_effective_chatgpt_auth", fake_auth)
    monkeypatch.setattr(chat_module, "send_upstream_request", fake_send)

    async with httpx.AsyncClient() as client:
        result, is_stream = await chat_module.process_chat_completion(
            {
                "model": "gpt-5.6-terra",
                "messages": [{"role": "user", "content": "hello"}],
            },
            Settings(),
            client,
        )

    assert is_stream is False
    assert result["choices"][0]["finish_reason"] == "length"
    assert result["model"] == "gpt-5.6-terra"
    assert result["service_tier"] == "default"


@pytest.mark.asyncio
@pytest.mark.parametrize("translator", [sse_translate_chat, sse_translate_text])
async def test_stream_failure_emits_error_and_done(translator: Any) -> None:
    event = {
        "type": "response.failed",
        "response": {"id": "resp_failed", "status": "failed", "error": {"message": "boom"}},
    }
    upstream = httpx.Response(
        200,
        content=f"data: {json.dumps(event)}\n\n".encode(),
    )
    frames = [frame async for frame in translator(upstream, "gpt-5.6-luna", 1)]
    rendered = [frame.decode() if isinstance(frame, bytes) else frame for frame in frames]
    assert any('"message": "boom"' in frame for frame in rendered)
    assert rendered[-1].strip() == "data: [DONE]"


@pytest.mark.asyncio
@pytest.mark.parametrize("translator", [sse_translate_chat, sse_translate_text])
async def test_stream_incomplete_emits_length_and_done(translator: Any) -> None:
    event = {
        "type": "response.incomplete",
        "response": {
            "id": "resp_incomplete",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
        },
    }
    upstream = httpx.Response(
        200,
        content=f"data: {json.dumps(event)}\n\n".encode(),
    )
    frames = [frame async for frame in translator(upstream, "gpt-5.6-luna", 1)]
    rendered = [frame.decode() if isinstance(frame, bytes) else frame for frame in frames]
    assert any('"finish_reason": "length"' in frame for frame in rendered)
    assert rendered[-1].strip() == "data: [DONE]"


@pytest.mark.asyncio
async def test_responses_stream_incomplete_is_a_terminal_event() -> None:
    responses_module = importlib.import_module("gptmock.services.responses")
    event = {
        "type": "response.incomplete",
        "response": {"id": "resp_incomplete", "status": "incomplete"},
    }
    upstream = httpx.Response(
        200,
        content=f"data: {json.dumps(event)}\n\n".encode(),
    )

    frames = [frame async for frame in responses_module._proxy_stream(upstream)]

    assert any("response.incomplete" in frame for frame in frames)
    assert not any("response.failed" in frame for frame in frames)


def test_ollama_non_stream_preserves_actual_model_and_tier() -> None:
    ollama_module = importlib.import_module("gptmock.routers.ollama")
    result = ollama_module._convert_openai_to_ollama_response(
        {
            "model": "gpt-5.6-sol",
            "service_tier": "default",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                },
            ],
        },
        "gpt-5.6-fast",
    )

    assert result["model"] == "gpt-5.6-sol"
    assert result["service_tier"] == "default"


def test_ollama_request_maps_thinking_tier_and_structured_output() -> None:
    ollama_module = importlib.import_module("gptmock.routers.ollama")
    payload = ollama_module._build_openai_payload(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "think": "max",
            "service_tier": "priority",
            "format": {"type": "object", "properties": {"answer": {"type": "string"}}},
        },
        "gpt-5.6-luna",
    )

    assert payload["reasoning_effort"] == "max"
    assert payload["service_tier"] == "priority"
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "ollama_response",
            "strict": True,
            "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
        },
    }


def test_ollama_options_reject_unsupported_runtime_controls() -> None:
    ollama_module = importlib.import_module("gptmock.routers.ollama")
    with pytest.raises(ollama_module.ChatCompletionError, match="options.temperature"):
        ollama_module._ollama_policy_headers(
            {"options": {"temperature": 0.2}},
            Settings(),
        )
    with pytest.raises(ollama_module.ChatCompletionError, match="must be an object"):
        ollama_module._ollama_policy_headers(
            {"options": "temperature=0.2"},
            Settings(),
        )


def test_ollama_num_predict_uses_output_token_policy() -> None:
    ollama_module = importlib.import_module("gptmock.routers.ollama")
    payload = {"options": {"num_predict": 128}}

    assert ollama_module._ollama_policy_headers(payload, Settings()) == {
        "X-GPTMock-Omitted-Parameters": "options.num_predict",
    }
    with pytest.raises(ollama_module.ChatCompletionError, match="options.num_predict"):
        ollama_module._ollama_policy_headers(
            payload,
            Settings(output_token_policy="reject"),
        )


def test_ollama_non_stream_maps_reasoning_to_thinking() -> None:
    ollama_module = importlib.import_module("gptmock.routers.ollama")
    result = ollama_module._convert_openai_to_ollama_response(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning_content": "analysis",
                    },
                    "finish_reason": "stop",
                },
            ],
        },
        "gpt-5.6-luna",
    )

    assert result["message"]["thinking"] == "analysis"
    assert "reasoning_content" not in result["message"]


@pytest.mark.asyncio
async def test_ollama_stream_preserves_incomplete_reason_model_and_tier() -> None:
    ollama_module = importlib.import_module("gptmock.routers.ollama")

    async def openai_frames():
        terminal = {
            "model": "gpt-5.6-sol",
            "service_tier": "default",
            "choices": [{"delta": {}, "finish_reason": "length"}],
        }
        yield f"data: {json.dumps(terminal)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    frames = [
        json.loads(frame)
        async for frame in ollama_module._convert_openai_to_ollama_stream(
            openai_frames(),
            "gpt-5.6-fast",
        )
    ]

    assert frames[-1]["done"] is True
    assert frames[-1]["done_reason"] == "length"
    assert frames[-1]["model"] == "gpt-5.6-sol"
    assert frames[-1]["service_tier"] == "default"


@pytest.mark.asyncio
async def test_ollama_stream_reports_missing_terminal_event() -> None:
    ollama_module = importlib.import_module("gptmock.routers.ollama")

    async def openai_frames():
        yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'

    frames = [
        json.loads(frame)
        async for frame in ollama_module._convert_openai_to_ollama_stream(
            openai_frames(),
            "gpt-5.6-luna",
        )
    ]

    assert frames[-1] == {"error": "Upstream stream ended before a terminal event"}


def test_ollama_generate_response_uses_native_response_fields() -> None:
    ollama_module = importlib.import_module("gptmock.routers.ollama")
    result = ollama_module._convert_openai_to_ollama_generate_response(
        {
            "model": "gpt-5.6-luna",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning_content": "analysis",
                    },
                    "finish_reason": "stop",
                },
            ],
        },
        "gpt-5.6-luna",
    )

    assert result["response"] == "answer"
    assert result["thinking"] == "analysis"
    assert "message" not in result


@pytest.mark.asyncio
async def test_responses_stream_truncation_emits_failure_and_done() -> None:
    responses_module = importlib.import_module("gptmock.services.responses")
    event = {"type": "response.output_text.delta", "delta": "partial"}
    upstream = httpx.Response(
        200,
        content=f"data: {json.dumps(event)}\n\n".encode(),
    )

    frames = [frame async for frame in responses_module._proxy_stream(upstream)]

    assert any("terminal response event" in frame for frame in frames)
    assert frames[-1].strip() == "data: [DONE]"


@pytest.mark.asyncio
async def test_non_stream_truncation_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    chat_module = importlib.import_module("gptmock.services.chat")

    async def fake_auth() -> tuple[str, str]:
        return "token", "account"

    async def fake_send(payload: dict[str, Any], *args: Any, **kwargs: Any) -> httpx.Response:
        del payload, args, kwargs
        event = {"type": "response.output_text.delta", "delta": "partial"}
        return httpx.Response(200, content=f"data: {json.dumps(event)}\n\n".encode())

    monkeypatch.setattr(chat_module, "get_effective_chatgpt_auth", fake_auth)
    monkeypatch.setattr(chat_module, "send_upstream_request", fake_send)

    async with httpx.AsyncClient() as client:
        with pytest.raises(
            chat_module.ChatCompletionError, match="terminal response event",
        ):
            await chat_module.process_chat_completion(
                {"model": "gpt-5.6-luna", "messages": [{"role": "user", "content": "hello"}]},
                Settings(),
                client,
            )
