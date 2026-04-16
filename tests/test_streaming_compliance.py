from __future__ import annotations

import importlib
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from gptmock.core.settings import Settings
from gptmock.schemas.messages import build_short_name_map


def _sse_response(events: list[dict[str, Any]]) -> httpx.Response:
    payload = b"".join(
        f"data: {json.dumps(event)}\n\n".encode() for event in events
    )
    return httpx.Response(200, content=payload)


async def _make_client(
    monkeypatch: pytest.MonkeyPatch,
    events: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
    captured_payloads: list[dict[str, Any]] | None = None,
) -> AsyncIterator[httpx.AsyncClient]:
    async def fake_send_upstream_request(
        payload: dict[str, Any],
        access_token: str,
        account_id: str,
        session_id: str,
        http_client: httpx.AsyncClient,
        *,
        verbose: bool = False,
    ) -> httpx.Response:
        del access_token, account_id, session_id, http_client, verbose
        if captured_payloads is not None:
            captured_payloads.append(payload)
        return _sse_response(events)

    async def fake_auth() -> tuple[str, str]:
        return "token", "account"

    chat_module = importlib.import_module("gptmock.services.chat")
    app_module = importlib.import_module("gptmock.app")
    dependencies_module = importlib.import_module("gptmock.core.dependencies")

    monkeypatch.setattr(chat_module, "send_upstream_request", fake_send_upstream_request)
    monkeypatch.setattr(chat_module, "get_effective_chatgpt_auth", fake_auth)

    effective_settings = settings or Settings()
    app = app_module.create_app(settings=effective_settings)
    app.dependency_overrides[dependencies_module.get_settings] = lambda: effective_settings
    app.state.http_client = httpx.AsyncClient(timeout=300.0)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
    await app.state.http_client.aclose()


async def _stream_lines(
    monkeypatch: pytest.MonkeyPatch,
    events: list[dict[str, Any]],
    payload: dict[str, Any],
    *,
    settings: Settings | None = None,
    captured_payloads: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    lines: list[str] = []
    async for client in _make_client(
        monkeypatch,
        events,
        settings=settings,
        captured_payloads=captured_payloads,
    ):
        async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            assert response.status_code == 200, response.text
            lines = [line async for line in response.aiter_lines() if line]
    chunks = [json.loads(line[6:]) for line in lines if line.startswith("data: {")]
    return lines, chunks


async def _non_stream_json(
    monkeypatch: pytest.MonkeyPatch,
    events: list[dict[str, Any]],
    payload: dict[str, Any],
    *,
    settings: Settings | None = None,
    captured_payloads: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    response: httpx.Response | None = None
    async for client in _make_client(
        monkeypatch,
        events,
        settings=settings,
        captured_payloads=captured_payloads,
    ):
        response = await client.post("/v1/chat/completions", json=payload)
    assert response is not None
    assert response.status_code == 200, response.text
    return response.json()


def _base_payload(**extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(extra)
    return payload


@pytest.mark.asyncio
async def test_first_chunk_has_role_on_text(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {"type": "response.output_text.delta", "delta": "Hello", "response": {"id": "resp_text"}},
        {"type": "response.output_text.delta", "delta": " world"},
        {"type": "response.completed", "response": {"id": "resp_text"}},
    ]
    lines, chunks = await _stream_lines(monkeypatch, events, _base_payload(stream=True))

    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": "Hello"}
    assert chunks[1]["choices"][0]["delta"] == {"content": " world"}
    assert chunks[-1]["choices"][0]["delta"] == {}
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert all("role" not in chunk["choices"][0]["delta"] for chunk in chunks[1:])
    assert lines[-1] == "data: [DONE]"


@pytest.mark.asyncio
async def test_first_chunk_has_role_on_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "id": "item_1", "call_id": "call_1", "name": "lookup"},
            "response": {"id": "resp_tool"},
        },
        {"type": "response.completed", "response": {"id": "resp_tool"}},
    ]
    _, chunks = await _stream_lines(monkeypatch, events, _base_payload(stream=True, tools=[]))

    first_delta = chunks[0]["choices"][0]["delta"]
    assert first_delta["role"] == "assistant"
    assert first_delta["tool_calls"][0]["id"] == "call_1"
    assert all("role" not in chunk["choices"][0]["delta"] for chunk in chunks[1:])


@pytest.mark.asyncio
async def test_text_and_tool_call_coexist(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {"type": "response.output_text.delta", "delta": "Let me ", "response": {"id": "resp_mix"}},
        {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "id": "item_mix", "call_id": "call_mix", "name": "lookup"},
        },
        {"type": "response.function_call_arguments.delta", "delta": '{"q":', "item_id": "item_mix"},
        {"type": "response.output_text.delta", "delta": "check.", "response": {"id": "resp_mix"}},
        {"type": "response.function_call_arguments.delta", "delta": '"weather"}', "item_id": "item_mix"},
        {"type": "response.output_item.done", "item": {"type": "function_call", "call_id": "call_mix", "name": "lookup", "arguments": '{"q":"weather"}'}},
        {"type": "response.completed", "response": {"id": "resp_mix"}},
    ]
    _, chunks = await _stream_lines(monkeypatch, events, _base_payload(stream=True))

    content = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
    tool_chunks = [chunk for chunk in chunks if chunk["choices"][0]["delta"].get("tool_calls")]
    assert content == "Let me check."
    assert len(tool_chunks) == 3


@pytest.mark.asyncio
async def test_args_done_no_duplication(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "id": "item_dup", "call_id": "call_dup", "name": "lookup"},
            "response": {"id": "resp_dup"},
        },
        {"type": "response.function_call_arguments.delta", "delta": "a", "item_id": "item_dup"},
        {"type": "response.function_call_arguments.delta", "delta": "b", "item_id": "item_dup"},
        {"type": "response.function_call_arguments.done", "arguments": "ab", "item_id": "item_dup"},
        {"type": "response.completed", "response": {"id": "resp_dup"}},
    ]
    _, chunks = await _stream_lines(monkeypatch, events, _base_payload(stream=True))

    args = [
        tc["function"]["arguments"]
        for chunk in chunks
        for tc in chunk["choices"][0]["delta"].get("tool_calls", [])
        if tc.get("function", {}).get("arguments")
    ]
    assert args == ["a", "b"]


@pytest.mark.asyncio
async def test_args_done_fallback_when_no_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "id": "item_done", "call_id": "call_done", "name": "lookup"},
            "response": {"id": "resp_done"},
        },
        {"type": "response.function_call_arguments.done", "arguments": '{"q":"done"}', "item_id": "item_done"},
        {"type": "response.completed", "response": {"id": "resp_done"}},
    ]
    _, chunks = await _stream_lines(monkeypatch, events, _base_payload(stream=True))

    args = [
        tc["function"]["arguments"]
        for chunk in chunks
        for tc in chunk["choices"][0]["delta"].get("tool_calls", [])
        if tc.get("function", {}).get("arguments")
    ]
    assert args == ['{"q":"done"}']


@pytest.mark.asyncio
async def test_args_delta_before_item_added_is_buffered(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {"type": "response.function_call_arguments.delta", "delta": "a", "item_id": "item_buf"},
        {"type": "response.function_call_arguments.delta", "delta": "b", "item_id": "item_buf"},
        {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "id": "item_buf", "call_id": "call_buf", "name": "lookup"},
            "response": {"id": "resp_buf"},
        },
        {"type": "response.function_call_arguments.delta", "delta": "c", "item_id": "item_buf"},
        {"type": "response.completed", "response": {"id": "resp_buf"}},
    ]
    _, chunks = await _stream_lines(monkeypatch, events, _base_payload(stream=True))

    args = [
        tc["function"]["arguments"]
        for chunk in chunks
        for tc in chunk["choices"][0]["delta"].get("tool_calls", [])
        if tc.get("function", {}).get("arguments")
    ]
    assert args == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_reasoning_content_in_streaming_standard_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {"type": "response.reasoning_text.delta", "delta": "think", "response": {"id": "resp_reason"}},
        {"type": "response.output_text.delta", "delta": "answer"},
        {"type": "response.completed", "response": {"id": "resp_reason"}},
    ]
    _, chunks = await _stream_lines(monkeypatch, events, _base_payload(stream=True), settings=Settings(reasoning_compat="standard"))

    first_delta = chunks[0]["choices"][0]["delta"]
    assert first_delta == {"role": "assistant", "reasoning_content": "think"}
    assert all("<think>" not in json.dumps(chunk) for chunk in chunks)


@pytest.mark.asyncio
async def test_reasoning_content_paragraph_break(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {"type": "response.reasoning_summary_part.added", "response": {"id": "resp_para"}},
        {"type": "response.reasoning_summary_text.delta", "delta": "one"},
        {"type": "response.reasoning_summary_part.added"},
        {"type": "response.reasoning_summary_text.delta", "delta": "two"},
        {"type": "response.completed", "response": {"id": "resp_para"}},
    ]
    _, chunks = await _stream_lines(monkeypatch, events, _base_payload(stream=True), settings=Settings(reasoning_compat="standard"))

    reasoning_parts = [chunk["choices"][0]["delta"].get("reasoning_content") for chunk in chunks if "reasoning_content" in chunk["choices"][0]["delta"]]
    assert reasoning_parts == ["one", "\n\n", "two"]


@pytest.mark.asyncio
async def test_content_clean_in_standard_mode_with_json_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {"type": "response.reasoning_text.delta", "delta": "internal", "response": {"id": "resp_json"}},
        {"type": "response.output_text.delta", "delta": '{"ok":true}'},
        {"type": "response.completed", "response": {"id": "resp_json"}},
    ]
    _, chunks = await _stream_lines(
        monkeypatch,
        events,
        _base_payload(
            stream=True,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": {"type": "object"}},
            },
        ),
        settings=Settings(reasoning_compat="standard"),
    )

    content = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
    assert content == '{"ok":true}'
    assert "<think>" not in content


@pytest.mark.asyncio
async def test_think_tags_opt_in_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {"type": "response.reasoning_text.delta", "delta": "internal", "response": {"id": "resp_think"}},
        {"type": "response.output_text.delta", "delta": "answer"},
        {"type": "response.completed", "response": {"id": "resp_think"}},
    ]
    _, chunks = await _stream_lines(monkeypatch, events, _base_payload(stream=True), settings=Settings(reasoning_compat="think-tags"))

    content = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
    assert content.startswith("<think>internal</think>answer")


@pytest.mark.asyncio
async def test_finish_reason_tool_calls_non_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {"type": "response.output_item.done", "item": {"type": "function_call", "call_id": "call_ns", "name": "lookup", "arguments": '{"q":"x"}'}, "response": {"id": "resp_ns"}},
        {"type": "response.completed", "response": {"id": "resp_ns"}},
    ]
    data = await _non_stream_json(monkeypatch, events, _base_payload(stream=False))
    assert data["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_finish_reason_tool_calls_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {"type": "response.output_item.added", "item": {"type": "function_call", "id": "item_fr", "call_id": "call_fr", "name": "lookup"}, "response": {"id": "resp_fr"}},
        {"type": "response.completed", "response": {"id": "resp_fr"}},
    ]
    _, chunks = await _stream_lines(monkeypatch, events, _base_payload(stream=True))
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[-1]["choices"][0]["delta"] == {}


@pytest.mark.asyncio
async def test_usage_chunk_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {"type": "response.output_text.delta", "delta": "hello", "response": {"id": "resp_usage"}},
        {"type": "response.completed", "response": {"id": "resp_usage", "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}}},
    ]
    _, chunks = await _stream_lines(
        monkeypatch,
        events,
        _base_payload(stream=True, stream_options={"include_usage": True}),
    )

    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"] == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}


@pytest.mark.asyncio
async def test_done_sentinel_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {"type": "response.output_text.delta", "delta": "hi", "response": {"id": "resp_done_sentinel"}},
        {"type": "response.completed", "response": {"id": "resp_done_sentinel"}},
    ]
    lines, _ = await _stream_lines(monkeypatch, events, _base_payload(stream=True))
    assert lines[-1] == "data: [DONE]"


def test_build_short_name_map_unit() -> None:
    name = "mcp__atlassian_mcp_server_cloud__jira_create_issue_with_many_extra_characters_to_trim"
    mapping = build_short_name_map([name])
    assert len(mapping[name]) <= 64
    assert mapping[name].startswith("mcp__")


def test_shortening_resolves_conflicts_unit() -> None:
    names = [
        "mcp__foo__same_name_that_is_long_enough_to_force_collision_and_truncation_alpha",
        "mcp__bar__same_name_that_is_long_enough_to_force_collision_and_truncation_alpha",
    ]
    mapping = build_short_name_map(names)
    assert mapping[names[0]] != mapping[names[1]]
    assert mapping[names[1]].endswith("_1")


@pytest.mark.asyncio
async def test_tool_name_roundtrip_restores_original(monkeypatch: pytest.MonkeyPatch) -> None:
    original_name = "mcp__atlassian_mcp_server_cloud__jira_create_issue_with_many_extra_characters_to_trim"
    short_name = build_short_name_map([original_name])[original_name]
    captured_payloads: list[dict[str, Any]] = []

    streaming_events = [
        {"type": "response.output_item.done", "item": {"type": "function_call", "call_id": "call_rt", "name": short_name, "arguments": '{"x":1}'}, "response": {"id": "resp_rt_stream"}},
        {"type": "response.completed", "response": {"id": "resp_rt_stream"}},
    ]
    _, chunks = await _stream_lines(
        monkeypatch,
        streaming_events,
        _base_payload(
            stream=True,
            tools=[{"type": "function", "function": {"name": original_name, "description": "d", "parameters": {"type": "object"}}}],
            tool_choice={"type": "function", "function": {"name": original_name}},
        ),
        captured_payloads=captured_payloads,
    )

    tool_delta = chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    assert tool_delta["function"]["name"] == original_name
    assert captured_payloads[0]["tools"][0]["name"] == short_name
    assert captured_payloads[0]["tool_choice"]["function"]["name"] == short_name

    non_stream_events = [
        {"type": "response.output_item.done", "item": {"type": "function_call", "call_id": "call_rt_ns", "name": short_name, "arguments": '{"x":2}'}, "response": {"id": "resp_rt_nonstream"}},
        {"type": "response.completed", "response": {"id": "resp_rt_nonstream"}},
    ]
    data = await _non_stream_json(
        monkeypatch,
        non_stream_events,
        _base_payload(
            stream=False,
            tools=[{"type": "function", "function": {"name": original_name, "description": "d", "parameters": {"type": "object"}}}],
        ),
    )
    assert data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == original_name
