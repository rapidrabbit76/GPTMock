"""Regression coverage for the pre-PR protocol and deployment review."""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from starlette.testclient import TestClient

from gptmock.app import create_app
from gptmock.core.settings import Settings
from gptmock.infra.sse import sse_translate_chat, sse_translate_text
from gptmock.routers.ollama import _convert_openai_to_ollama_response, _convert_openai_to_ollama_stream
from gptmock.schemas.transform import convert_ollama_messages
from gptmock.services.chat import ChatCompletionError
from gptmock.services.responses import _proxy_stream


def response(events: list[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(200, content="".join(f"data: {json.dumps(e)}\n\n" for e in events).encode())


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"requests": [], "events": [{"type": "response.completed", "response": {
        "model": "gpt-6-astra", "status": "completed", "service_tier": "default", "output": [],
    }}]}

    async def auth() -> tuple[str, str]:
        return "test-token", "test-account"

    async def send(payload: dict[str, Any], *args: Any, **kwargs: Any) -> httpx.Response:
        state["requests"].append(payload)
        return response(state["events"])

    for name in ("gptmock.services.chat", "gptmock.services.responses"):
        module = importlib.import_module(name)
        monkeypatch.setattr(module, "get_effective_chatgpt_auth", auth)
        monkeypatch.setattr(module, "send_upstream_request", send)
    # Bootstrap tests reload modules; already-registered endpoints can retain the old globals.
    for route in create_app(Settings()).routes:
        endpoint = getattr(route, "endpoint", None)
        for name in ("process_chat_completion", "process_text_completion", "process_responses_api"):
            processor = getattr(endpoint, "__globals__", {}).get(name)
            if processor is not None:
                monkeypatch.setitem(processor.__globals__, "get_effective_chatgpt_auth", auth)
                monkeypatch.setitem(processor.__globals__, "send_upstream_request", send)
    return state


@pytest.mark.parametrize("route", ["/v1/responses", "/v1/chat/completions", "/v1/completions", "/api/chat", "/api/generate"])
@pytest.mark.parametrize("model", ["gpt-6-astra-max", "gpt-6-astra_max", "gpt-6-astra-fast-max"])
def test_removed_astra_alias_never_reaches_upstream(upstream: dict[str, Any], route: str, model: str) -> None:
    payload = {"model": model, "stream": False, "input": "hi", "prompt": "hi", "messages": [{"role": "user", "content": "hi"}]}
    with TestClient(create_app(Settings())) as client:
        result = client.post(route, json=payload)
        assert result.status_code == 400
        assert "reasoning" in result.text
    assert not upstream["requests"]


def test_factory_settings_are_app_local() -> None:
    with TestClient(create_app(Settings(reasoning_effort="low", ollama_version="9.9.9", expose_reasoning_models=True))) as low:
        with TestClient(create_app(Settings(reasoning_effort="high", ollama_version="8.8.8"))) as high:
            assert low.get("/api/version").json()["version"] == "9.9.9"
            assert high.get("/api/version").json()["version"] == "8.8.8"
            models = {m["id"]: m for m in low.get("/v1/models").json()["data"]}
            assert models["gpt-6-astra"]["reasoning"]["default_effort"] == "low"
            assert "gpt-6-astra-max" not in models
            assert "gpt-5.6-luna-max" in models
            assert all(m["id"] != "gpt-5.6-luna-max" for m in high.get("/v1/models").json()["data"])


@pytest.mark.parametrize("explicit,expected", [(None, "max"), ("low", "low")])
def test_responses_keeps_remaining_effort_alias_semantics(upstream: dict[str, Any], explicit: str | None, expected: str) -> None:
    payload: dict[str, Any] = {"model": "gpt-5.6-luna-max", "input": "hi"}
    if explicit:
        payload["reasoning"] = {"effort": explicit}
    with TestClient(create_app(Settings())) as client:
        assert client.post("/v1/responses", json=payload).status_code == 200
    assert upstream["requests"][0]["reasoning"]["effort"] == expected


def test_named_tool_and_history_share_mapping(upstream: dict[str, Any]) -> None:
    name = "mcp__" + "x" * 70 + "__echo"
    payload = {"model": "gpt-6-astra", "messages": [
        {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": name, "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "OK"},
    ], "tools": [{"type": "function", "function": {"name": name, "strict": True, "parameters": {"type": "object"}}}],
        "tool_choice": {"type": "function", "function": {"name": name}}}
    with TestClient(create_app(Settings())) as client:
        assert client.post("/v1/chat/completions", json=payload).status_code == 200
    sent = upstream["requests"][0]
    assert sent["tool_choice"] == {"type": "function", "name": sent["tools"][0]["name"]}
    assert sent["input"][0]["name"] == sent["tools"][0]["name"]
    assert sent["tools"][0]["strict"] is True


def incomplete_tool_events() -> list[dict[str, Any]]:
    item = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read", "arguments": '{"path":', "status": "incomplete"}
    return [{"type": "response.output_item.added", "output_index": 0, "item": item},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.incomplete", "response": {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": [item]}}]


@pytest.mark.asyncio
async def test_incomplete_tool_stream_is_error() -> None:
    frames = [frame async for frame in sse_translate_chat(response(incomplete_tool_events()), "gpt-6-astra", 1)]
    assert any(b'"error"' in frame for frame in frames)
    assert not any(b'"finish_reason": "tool_calls"' in frame for frame in frames)


def test_incomplete_tool_nonstream_is_error(upstream: dict[str, Any]) -> None:
    upstream["events"] = incomplete_tool_events()
    with TestClient(create_app(Settings())) as client:
        result = client.post("/v1/chat/completions", json={"model": "gpt-6-astra", "messages": [{"role": "user", "content": "hi"}]})
    assert result.status_code == 502


class TimeoutStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"type":"response.output_text.delta","delta":"partial"}\n\n'
        raise httpx.ReadTimeout("synthetic timeout")


@pytest.mark.asyncio
@pytest.mark.parametrize("translator", [sse_translate_chat, sse_translate_text])
async def test_transport_timeout_emits_error(translator: Any) -> None:
    frames = [frame async for frame in translator(httpx.Response(200, stream=TimeoutStream()), "gpt-6-astra", 1)]
    assert any(b'"error"' in frame for frame in frames)
    assert frames[-1] == b"data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_bare_done_is_not_responses_completion() -> None:
    frames = [frame async for frame in _proxy_stream(httpx.Response(200, content=b"data: [DONE]\n\n"))]
    assert '"type": "error"' in "".join(frames)
    error = json.loads(frames[0].split("data: ", 1)[1])
    assert error["error"]["message"] == error["message"]
    assert error["error"]["code"] == error["code"] == "upstream_stream_incomplete"
    assert error["sequence_number"] == 0
    assert "".join(frames).count("[DONE]") == 1


@pytest.mark.asyncio
async def test_text_item_done_does_not_finish_the_response() -> None:
    events = [{"type": "response.output_text.done", "text": "partial"}, incomplete_tool_events()[-1]]
    frames = [frame async for frame in sse_translate_text(response(events), "gpt-6-astra", 1)]
    assert not any(b'"finish_reason": "stop"' in frame for frame in frames)
    assert sum(b'"finish_reason": "length"' in frame for frame in frames) == 1


@pytest.mark.asyncio
async def test_ollama_buffers_interleaved_tool_arguments() -> None:
    async def chunks():
        deltas = [
            [{"index": 0, "function": {"name": "first", "arguments": '{"x":'}}],
            [{"index": 1, "function": {"name": "second", "arguments": '{"y":2}'}}, {"index": 0, "function": {"arguments": "1}"}}],
        ]
        for calls in deltas:
            yield ("data: " + json.dumps({"choices": [{"delta": {"tool_calls": calls}}]}) + "\n\n").encode()
        yield b'data: {"model":"gpt-6-astra","service_tier":"default","choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
        yield b"data: [DONE]\n\n"
    frames = [json.loads(frame) async for frame in _convert_openai_to_ollama_stream(chunks(), "gpt-6-astra-fast")]
    assert len(frames) == 1
    assert frames[0]["message"]["tool_calls"] == [
        {"function": {"name": "first", "arguments": {"x": 1}}},
        {"function": {"name": "second", "arguments": {"y": 2}}},
    ]
    assert frames[0]["done"] is True
    assert frames[0]["service_tier"] == "default"


def test_ollama_nonstream_arguments_are_objects() -> None:
    result = _convert_openai_to_ollama_response({"choices": [{"message": {"role": "assistant", "content": None,
        "tool_calls": [{"function": {"name": "echo", "arguments": '{"x":1}'}}]}}]}, "gpt-6-astra")
    assert result["message"]["content"] == ""
    assert result["message"]["tool_calls"][0]["function"]["arguments"] == {"x": 1}


def test_ollama_invalid_arguments_are_not_emitted() -> None:
    with pytest.raises(ChatCompletionError):
        _convert_openai_to_ollama_response({"choices": [{"message": {"tool_calls": [{"function": {"name": "echo", "arguments": "{"}}]}}]}, "gpt-6-astra")


@pytest.mark.asyncio
async def test_concurrent_auth_refresh_is_single_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    from gptmock.infra import auth
    assert auth.write_auth_file({"tokens": {"access_token": "old", "refresh_token": "refresh-old", "account_id": "test-account"}})
    calls = []

    async def refresh(token: str, client_id: str) -> dict[str, str]:
        calls.append(token)
        await asyncio.sleep(0.04)
        return {"access_token": "new", "refresh_token": "refresh-new", "account_id": "test-account"}

    monkeypatch.setattr(auth, "_refresh_chatgpt_tokens", refresh)
    monkeypatch.setattr(auth, "_should_refresh_access_token", lambda token, last: token == "old")
    results = await asyncio.gather(*(auth.load_chatgpt_tokens() for _ in range(5)))
    assert calls == ["refresh-old"]
    assert all(result[0] == "new" for result in results)


def test_cli_explicit_false_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    from gptmock import cli
    monkeypatch.setenv("GPTMOCK_DEFAULT_WEB_SEARCH", "true")
    monkeypatch.setenv("GPTMOCK_CORS_ORIGINS", "https://example.com")
    monkeypatch.setattr(cli, "read_auth_file", lambda: {"tokens": {"access_token": "test"}})
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    # Restore variables written by cmd_serve after this test.
    for name in ("VERBOSE", "VERBOSE_OBFUSCATION", "REASONING_EFFORT", "REASONING_SUMMARY", "REASONING_COMPAT", "EXPOSE_REASONING_MODELS", "OUTPUT_TOKEN_POLICY", "HOST", "PORT"):
        monkeypatch.setenv(f"GPTMOCK_{name}", "")
    cli.cmd_serve("127.0.0.1", 8000, False, False, "medium", "auto", "standard", None, False, False, "omit", "")
    assert Settings().default_web_search is False
    assert Settings().cors_origins == ""


@pytest.mark.asyncio
async def test_real_send_records_rate_limit_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    from gptmock.services import upstream
    recorded = []
    monkeypatch.setattr(upstream, "record_rate_limits_from_response", lambda result: recorded.append(result.status_code))
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        result = await upstream.send_upstream_request({"model": "gpt-6-astra"}, "test", "test", "test", client)
        await result.aclose()
    assert recorded == [200]


def test_ollama_out_of_order_results_match_tool_names() -> None:
    messages = [{"role": "assistant", "tool_calls": [
        {"function": {"name": "first", "arguments": {}}},
        {"function": {"name": "second", "arguments": {}}},
    ]}, {"role": "tool", "tool_name": "second", "content": "2"},
        {"role": "tool", "tool_name": "first", "content": "1"}]
    converted = convert_ollama_messages(messages, None)
    assert converted[1]["tool_call_id"] == converted[0]["tool_calls"][1]["id"]
    assert converted[2]["tool_call_id"] == converted[0]["tool_calls"][0]["id"]


def test_cli_rejects_unwritable_storage_before_login(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from gptmock import cli
    def unavailable() -> None:
        raise PermissionError("synthetic read-only mount")
    monkeypatch.setattr(cli, "validate_auth_storage", unavailable)
    monkeypatch.setattr(cli, "read_auth_file", lambda: pytest.fail("must fail before reading auth"))
    assert cli.cmd_serve("127.0.0.1", 8000, False, False, "medium", "auto", "standard", None, False, False, "omit", "") == 1
    assert "10001" in capsys.readouterr().err


def test_cli_honors_dotenv_defaults_and_explicit_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from gptmock import cli
    for name in list(os.environ):
        if name.startswith(("GPTMOCK_", "CHATGPT_LOCAL_")) and not name.endswith("HOME"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("GPTMOCK_REASONING_EFFORT=high\nGPTMOCK_DEFAULT_WEB_SEARCH=true\nGPTMOCK_OUTPUT_TOKEN_POLICY=reject\n", encoding="utf-8")
    captured = {}
    def serve(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 0
    monkeypatch.setattr(cli, "cmd_serve", serve)
    monkeypatch.setattr(sys, "argv", ["gptmock", "serve", "--no-enable-web-search"])
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 0
    assert captured["reasoning_effort"] == "high"
    assert captured["output_token_policy"] == "reject"
    assert captured["default_web_search"] is False


def test_refresh_lock_serializes_processes() -> None:
    code = """
import asyncio, os
from pathlib import Path
from gptmock.infra.auth import _refresh_lock, get_home_dir
async def run():
    async with _refresh_lock():
        marker = Path(get_home_dir()) / 'synthetic-refresh-overlap'
        assert not marker.exists(), 'Overlapping refresh processes'
        marker.touch()
        try:
            await asyncio.sleep(0.08)
        finally:
            marker.unlink()
asyncio.run(run())
"""
    workers = []
    try:
        for _ in range(3):
            worker = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            workers.append(worker)
            print({"pid": worker.pid, "ppid": os.getpid(), "command": "python refresh-lock test", "cwd": os.getcwd(), "ports": []})
        for worker in workers:
            _, error = worker.communicate(timeout=15)
            assert worker.returncode == 0, error
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=5)


@pytest.mark.asyncio
async def test_cancelled_refresh_waiter_does_not_leak_lock() -> None:
    from gptmock.infra.auth import _refresh_lock
    async def wait() -> None:
        async with _refresh_lock():
            pass
    async with _refresh_lock():
        waiter = asyncio.create_task(wait())
        await asyncio.sleep(0.02)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
    await asyncio.wait_for(wait(), timeout=2)
