from __future__ import annotations

import base64
import io
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import pytest

from gptmock.core.badges import make_pct_badge, read_coverage_pct, reset_coverage_files, update_gist_badges
from gptmock.core.logging import log_json
from gptmock.core.models import AuthBundle, TokenData
from gptmock.core.settings import Settings
from gptmock.core.utils import extract_usage, parse_datetime
from gptmock.infra.auth import (
    _derive_account_id,
    _now_iso8601,
    _should_refresh_access_token,
    generate_pkce,
    get_home_dir,
    parse_jwt_claims,
    read_auth_file,
    write_auth_file,
)
from gptmock.infra.limits import (
    RateLimitSnapshot,
    RateLimitWindow,
    compute_reset_at,
    load_rate_limit_snapshot,
    parse_rate_limit_headers,
    record_rate_limits_from_response,
    store_rate_limit_snapshot,
)
from gptmock.infra.oauth import LOGIN_SUCCESS_HTML, OAuthHandler, OAuthHTTPServer
from gptmock.infra.sse import (
    SSEChatContext,
    _handle_completed,
    _handle_content_part_done,
    _handle_output_item_done,
    _handle_reasoning_delta,
    _handle_summary_part_added,
    _handle_web_search,
    _merge_ws_params,
    _serialize_tool_args,
    sse_translate_text,
)
from gptmock.schemas.messages import (
    _normalize_image_data_url,
    convert_chat_messages_to_responses_input,
    convert_tools_chat_to_responses,
)
from gptmock.schemas.transform import convert_ollama_messages, normalize_ollama_tools, to_data_url
from gptmock.services.reasoning import (
    allowed_efforts_for_model,
    apply_reasoning_to_message,
    build_reasoning_param,
    extract_reasoning_from_model_name,
)
from gptmock.services.responses import (
    CollectorState,
    _apply_reasoning_text,
    _build_responses_api_result,
    _extract_output_text_from_response,
    _extract_usage,
    _handle_response_terminal,
    _is_strict_json_text_format,
    _merge_instructions,
    _parse_sse_data,
    _safe_tool_choice,
    _update_response_metadata,
)
from gptmock.services.responses import (
    _handle_content_part_done as _responses_handle_content_part_done,
)
from gptmock.services.responses import (
    _handle_output_item_done as _responses_handle_output_item_done,
)


def _make_jwt(payload: dict[str, object]) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).rstrip(b"=")
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=")
    return f"{header.decode()}.{body.decode()}.sig"


class _Response:
    def __init__(self, headers: dict[str, object] | None = None) -> None:
        self.headers = headers


class _AsyncResponse:
    def __init__(self, lines: Sequence[str | bytes]) -> None:
        self._lines = lines
        self.closed = False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aclose(self) -> None:
        self.closed = True


class _ExplodingLogger:
    def __call__(self, message: str) -> None:
        raise RuntimeError("boom")


class _BadRepr:
    def __repr__(self) -> str:
        raise RuntimeError("bad repr")


class _ShutdownServer:
    def __init__(self) -> None:
        self.shutdown_called = False
        self.exit_code = 1

    def shutdown(self) -> None:
        self.shutdown_called = True


def _make_oauth_handler(path: str) -> Any:
    handler = cast(Any, OAuthHandler.__new__(OAuthHandler))
    handler.path = path
    handler.server = _ShutdownServer()
    handler.wfile = io.BytesIO()
    handler.error_calls = []
    handler.response_calls = []
    handler.header_calls = []
    handler.end_headers_called = False
    handler.send_error = lambda code, message=None, explain=None: handler.error_calls.append((code, message))
    handler.send_response = lambda code, message=None: handler.response_calls.append(code)
    handler.send_header = lambda keyword, value: handler.header_calls.append((keyword, value))
    handler.end_headers = lambda: setattr(handler, "end_headers_called", True)
    return handler


class TestCoreHelpers:
    def test_extract_usage_maps_response_fields(self) -> None:
        evt = {"response": {"usage": {"input_tokens": 10, "output_tokens": 7}}}
        assert extract_usage(evt) == {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17}

    def test_extract_usage_returns_none_for_invalid_payload(self) -> None:
        assert extract_usage({"response": {"usage": "bad"}}) is None

    def test_parse_datetime_handles_zulu_and_naive(self) -> None:
        assert parse_datetime("2026-03-08T12:00:00Z") == datetime(2026, 3, 8, 12, 0, 0, tzinfo=UTC)
        assert parse_datetime("2026-03-08T12:00:00") == datetime(2026, 3, 8, 12, 0, 0, tzinfo=UTC)

    def test_parse_datetime_rejects_invalid_values(self) -> None:
        assert parse_datetime(123) is None
        assert parse_datetime("not-a-date") is None

    def test_log_json_falls_back_for_non_serializable_payload(self) -> None:
        lines: list[str] = []
        log_json("PREFIX", {"bad": {1, 2}}, logger=lines.append)
        assert lines and lines[0].startswith("PREFIX\n")

    def test_log_json_uses_print_when_logger_is_missing(self, capsys: pytest.CaptureFixture[str]) -> None:
        log_json("PREFIX", {"ok": True})
        captured = capsys.readouterr()
        assert "PREFIX" in captured.out

    def test_log_json_swallows_exploding_fallback_logger(self) -> None:
        log_json("PREFIX", {"bad": _BadRepr()}, logger=_ExplodingLogger())


class TestBadgeHelpers:
    def test_make_pct_badge_uses_expected_colors(self) -> None:
        assert make_pct_badge("tests", 100)["color"] == "brightgreen"
        assert make_pct_badge("tests", 95)["color"] == "yellow"
        assert make_pct_badge("tests", 70)["color"] == "red"

    def test_reset_and_read_coverage_files(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".coverage").write_text("x")
        (tmp_path / "coverage.json").write_text(json.dumps({"totals": {"percent_covered": 72.7}}))
        assert read_coverage_pct() == 73
        reset_coverage_files()
        assert not (tmp_path / ".coverage").exists()
        assert not (tmp_path / "coverage.json").exists()
        assert read_coverage_pct() is None

    def test_update_gist_badges_without_token_prints_payload(self, tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("GIST_TOKEN", raising=False)
        (tmp_path / "coverage.json").write_text(json.dumps({"totals": {"percent_covered": 70}}))
        update_gist_badges(tests_label="tests", tests_pct=100, tests_collected=10, tests_skipped=0)
        captured = capsys.readouterr()
        assert "gptmock-coverage.json" in captured.err
        assert '"message": "70%"' in captured.out
        assert '"message": "100%"' in captured.out

    def test_update_gist_badges_handles_missing_coverage_and_all_skipped(self, tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("GIST_TOKEN", raising=False)
        update_gist_badges(tests_label="tests", tests_pct=0, tests_collected=5, tests_skipped=5)
        captured = capsys.readouterr()
        assert '"message": "no tests"' in captured.out


class TestAuthHelpers:
    def test_generate_pkce_returns_expected_shapes(self) -> None:
        pkce = generate_pkce()
        assert len(pkce.code_verifier) == 128
        assert pkce.code_challenge

    def test_parse_jwt_claims_round_trip(self) -> None:
        token = _make_jwt({"sub": "user-1", "exp": 123})
        assert parse_jwt_claims(token) == {"sub": "user-1", "exp": 123}

    def test_parse_jwt_claims_rejects_bad_token(self) -> None:
        assert parse_jwt_claims("bad-token") is None

    def test_derive_account_id_reads_nested_claim(self) -> None:
        token = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct_123"}})
        assert _derive_account_id(token) == "acct_123"

    def test_should_refresh_access_token_when_missing_or_expiring(self) -> None:
        soon = datetime.now(UTC) + timedelta(minutes=4)
        token = _make_jwt({"exp": soon.timestamp()})
        assert _should_refresh_access_token(None, None) is True
        assert _should_refresh_access_token(token, None) is True

    def test_should_refresh_access_token_uses_last_refresh_fallback(self) -> None:
        last_refresh = (datetime.now(UTC) - timedelta(minutes=56)).isoformat()
        token = _make_jwt({"sub": "user-1"})
        assert _should_refresh_access_token(token, last_refresh) is True

    def test_should_refresh_access_token_keeps_fresh_token(self) -> None:
        later = datetime.now(UTC) + timedelta(hours=1)
        token = _make_jwt({"exp": later.timestamp()})
        assert _should_refresh_access_token(token, None) is False

    def test_auth_file_round_trip(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        home = tmp_path / "auth-home"
        home.mkdir()
        monkeypatch.setenv("GPTMOCK_HOME", str(home))
        payload = {"tokens": {"access_token": "a", "refresh_token": "b"}}
        assert get_home_dir() == str(home)
        assert write_auth_file(payload) is True
        assert read_auth_file() == payload

    def test_now_iso8601_uses_z_suffix(self) -> None:
        assert _now_iso8601().endswith("Z")

    def test_read_auth_file_returns_none_for_missing_or_invalid_json(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("CHATGPT_LOCAL_HOME", raising=False)
        monkeypatch.delenv("CODEX_HOME", raising=False)
        monkeypatch.setenv("GPTMOCK_HOME", str(tmp_path))
        assert read_auth_file() is None
        (tmp_path / "auth.json").write_text("not-json")
        assert read_auth_file() is None

    def test_write_auth_file_fails_when_home_is_a_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        bad_home = tmp_path / "not-a-dir"
        bad_home.write_text("x")
        monkeypatch.setenv("GPTMOCK_HOME", str(bad_home))
        assert write_auth_file({"tokens": {}}) is False


class TestOAuthHelpers:
    def test_auth_url_contains_expected_query(self) -> None:
        server = OAuthHTTPServer.__new__(OAuthHTTPServer)
        server.client_id = "client-1"
        server.redirect_uri = "http://localhost:1455/auth/callback"
        server.issuer = "https://example.com"
        server.pkce = generate_pkce()
        server.state = "abc123"
        url = server.auth_url()
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        assert parsed.scheme == "https"
        assert qs["client_id"] == ["client-1"]
        assert qs["state"] == ["abc123"]
        assert qs["code_challenge_method"] == ["S256"]

    def test_maybe_obtain_api_key_without_org_or_project_returns_setup_url(self) -> None:
        server = OAuthHTTPServer.__new__(OAuthHTTPServer)
        token_data = TokenData(id_token="idtok", access_token="acctok", refresh_token="reftok", account_id="acct")
        api_key, success_url = server.maybe_obtain_api_key({}, {"chatgpt_plan_type": "plus"}, token_data)
        assert api_key is None
        assert success_url is not None and success_url.startswith("http://localhost:1455/success?")
        qs = parse_qs(urlparse(success_url).query)
        assert qs["id_token"] == ["idtok"]
        assert qs["plan_type"] == ["plus"]

    def test_persist_auth_writes_expected_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("GPTMOCK_HOME", str(tmp_path))
        server = OAuthHTTPServer.__new__(OAuthHTTPServer)
        bundle = AuthBundle(
            api_key="sk-test",
            last_refresh="2026-03-08T12:00:00Z",
            token_data=TokenData(
                id_token="id",
                access_token="access",
                refresh_token="refresh",
                account_id="acct",
            ),
        )
        assert server.persist_auth(bundle) is True
        auth = json.loads((tmp_path / "auth.json").read_text())
        assert auth["OPENAI_API_KEY"] == "sk-test"
        assert auth["tokens"]["account_id"] == "acct"

    def test_oauth_handler_send_html_and_post_not_found(self) -> None:
        handler = _make_oauth_handler("/ignored")
        handler._send_html("hello")
        assert handler.response_calls == [200]
        assert handler.end_headers_called is True
        assert handler.wfile.getvalue() == b"hello"

        handler = _make_oauth_handler("/ignored")
        handler._shutdown = lambda: setattr(handler.server, "shutdown_called", True)
        handler.do_POST()
        assert handler.error_calls == [(404, "Not Found")]
        assert handler.server.shutdown_called is True

    def test_oauth_handler_get_success_missing_code_and_404_paths(self) -> None:
        handler = _make_oauth_handler("/success")
        handler._shutdown_after_delay = lambda seconds=2.0: setattr(handler.server, "shutdown_called", True)
        handler.do_GET()
        assert handler.response_calls == [200]
        assert LOGIN_SUCCESS_HTML.encode() in handler.wfile.getvalue()

        handler = _make_oauth_handler("/auth/callback")
        handler._shutdown = lambda: setattr(handler.server, "shutdown_called", True)
        handler.do_GET()
        assert handler.error_calls == [(400, "Missing auth code")]

        handler = _make_oauth_handler("/nope")
        handler._shutdown = lambda: setattr(handler.server, "shutdown_called", True)
        handler.do_GET()
        assert handler.error_calls == [(404, "Not Found")]

    def test_oauth_handler_exchange_code_and_persist_failure_paths(self) -> None:
        handler = _make_oauth_handler("/auth/callback?code=abc")
        handler._shutdown = lambda: setattr(handler.server, "shutdown_called", True)
        handler._exchange_code = lambda code: (_ for _ in ()).throw(RuntimeError("boom"))
        handler.do_GET()
        assert handler.error_calls == [(500, "Token exchange failed: boom")]

        handler = _make_oauth_handler("/auth/callback?code=abc")
        handler._shutdown_after_delay = lambda seconds=2.0: setattr(handler.server, "shutdown_called", True)
        token_data = TokenData(id_token="id", access_token="access", refresh_token="refresh", account_id="acct")
        bundle = AuthBundle(api_key="sk", token_data=token_data, last_refresh="2026-03-08T12:00:00Z")
        handler._exchange_code = lambda code: (bundle, "http://localhost:1455/success")
        original_write_auth_file = OAuthHandler.do_GET.__globals__["write_auth_file"]
        OAuthHandler.do_GET.__globals__["write_auth_file"] = lambda payload: False
        try:
            handler.do_GET()
        finally:
            OAuthHandler.do_GET.__globals__["write_auth_file"] = original_write_auth_file
        assert handler.error_calls == [(500, "Unable to persist auth file")]


class TestResponsesHelpers:
    def test_merge_tool_choice_and_json_text_helpers(self) -> None:
        assert _merge_instructions("base", "user") == "base\n\nuser"
        assert _merge_instructions(None, "user") == "user"
        assert _safe_tool_choice("none") == "none"
        assert _safe_tool_choice(123) == "auto"
        assert _is_strict_json_text_format({"format": {"type": "json_schema"}}) is True
        assert _is_strict_json_text_format({"format": {"type": "text"}}) is False

    def test_apply_reasoning_text_extract_usage_and_output_helpers(self) -> None:
        assert _apply_reasoning_text("answer", "sum", "full", "think-tags", strict_json_text=False).startswith("<think>sum\n\nfull</think>answer")
        assert _apply_reasoning_text("answer", "sum", "full", "o3", strict_json_text=False) == "answer"
        assert _apply_reasoning_text("answer", "sum", "full", "think-tags", strict_json_text=True) == "answer"
        assert _extract_usage({"usage": {"prompt_tokens": 1}}) == {"prompt_tokens": 1}
        assert _extract_usage(None) is None
        response_obj = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "hello"}, {"type": "output_text", "text": " world"}]}]}
        assert _extract_output_text_from_response(response_obj) == "hello world"
        assert _extract_output_text_from_response(None) == ""

    def test_parse_and_collect_response_metadata_helpers(self) -> None:
        assert _parse_sse_data("event: noop") is None
        assert _parse_sse_data("data: hi") == "hi"
        assert _parse_sse_data(b"data: bye") == "bye"
        state = CollectorState()
        _update_response_metadata(state, {"id": "resp_1", "created_at": 123.0, "status": "done"})
        assert state.response_id == "resp_1"
        assert state.created_at == 123.0
        assert state.status == "done"

    def test_response_event_handlers_and_result_builder(self) -> None:
        state = CollectorState(created_at=1.0)
        _responses_handle_output_item_done(state, {"item": {"type": "function_call", "id": "fc_1", "name": "lookup", "arguments": "{}"}})
        _responses_handle_content_part_done(state, {"part": {"type": "output_text", "annotations": [{"kind": "url"}]}})
        should_break = _handle_response_terminal(state, {"response": {"error": {"message": "boom"}}}, "response.failed")
        assert should_break is True
        assert state.error_message == "boom"

        state = CollectorState(
            response_id="resp_1",
            created_at=2.0,
            status="completed",
            final_response_obj={
                "usage": {"total_tokens": 3},
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "final", "annotations": [{"kind": "url"}]}]}],
            },
            function_calls=[{"type": "function_call", "name": "lookup"}],
            reasoning_summary_text="sum",
            reasoning_full_text="full",
        )
        settings = Settings(reasoning_compat="o3")
        result = _build_responses_api_result(state, "gpt-5", "gpt-5", settings, {"format": {"type": "text"}})
        assert result["id"] == "resp_1"
        assert result["usage"] == {"total_tokens": 3}
        assert result["output"][0]["content"][0]["annotations"] == [{"kind": "url"}]
        assert result["output"][1] == {"type": "function_call", "name": "lookup"}
        assert result["reasoning"]["content"][0]["text"] == "sum\n\nfull"


class TestLimitsHelpers:
    def test_parse_rate_limit_headers_reads_primary_and_secondary(self) -> None:
        headers = {
            "x-codex-primary-used-percent": "25.5",
            "x-codex-primary-window-minutes": "60",
            "x-codex-primary-reset-after-seconds": "120",
            "x-codex-secondary-used-percent": "10",
            "x-codex-secondary-window-minutes": "10080",
            "x-codex-secondary-reset-after-seconds": "3600",
        }
        snapshot = parse_rate_limit_headers(headers)
        assert snapshot is not None
        assert snapshot.primary is not None and snapshot.primary.used_percent == 25.5
        assert snapshot.secondary is not None and snapshot.secondary.window_minutes == 10080

    def test_parse_rate_limit_headers_rejects_missing_usage(self) -> None:
        assert parse_rate_limit_headers({"x-codex-primary-window-minutes": "60"}) is None

    def test_rate_limit_snapshot_round_trip(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        home = tmp_path / "limits-home"
        home.mkdir()
        monkeypatch.setenv("GPTMOCK_HOME", str(home))
        snapshot = RateLimitSnapshot(
            primary=RateLimitWindow(used_percent=12.5, window_minutes=60, resets_in_seconds=30),
            secondary=RateLimitWindow(used_percent=80.0, window_minutes=10080, resets_in_seconds=600),
        )
        captured_at = datetime(2026, 3, 8, 12, 0, tzinfo=UTC)
        store_rate_limit_snapshot(snapshot, captured_at=captured_at)
        stored = load_rate_limit_snapshot()
        assert stored is not None
        assert stored.captured_at == captured_at
        assert stored.snapshot.primary is not None and stored.snapshot.primary.used_percent == 12.5
        assert stored.snapshot.secondary is not None and stored.snapshot.secondary.resets_in_seconds == 600

    def test_record_rate_limits_from_response_persists_data(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        home = tmp_path / "limits-response-home"
        home.mkdir()
        monkeypatch.setenv("GPTMOCK_HOME", str(home))
        response = _Response(
            {
                "x-codex-primary-used-percent": "50",
                "x-codex-primary-window-minutes": "60",
                "x-codex-primary-reset-after-seconds": "90",
            }
        )
        record_rate_limits_from_response(response)
        stored = load_rate_limit_snapshot()
        assert stored is not None
        assert stored.snapshot.primary is not None and stored.snapshot.primary.used_percent == 50.0

    def test_compute_reset_at_handles_invalid_and_valid_seconds(self) -> None:
        captured_at = datetime(2026, 3, 8, 12, 0, tzinfo=UTC)
        assert compute_reset_at(captured_at, RateLimitWindow(10.0, 60, 30)) == captured_at + timedelta(seconds=30)
        assert compute_reset_at(captured_at, RateLimitWindow(10.0, 60, None)) is None

    def test_load_rate_limit_snapshot_handles_missing_invalid_and_empty_data(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("GPTMOCK_HOME", str(tmp_path))
        assert load_rate_limit_snapshot() is None
        (tmp_path / "usage_limits.json").write_text("bad")
        assert load_rate_limit_snapshot() is None
        (tmp_path / "usage_limits.json").write_text(json.dumps({"captured_at": "2026-03-08T12:00:00Z"}))
        assert load_rate_limit_snapshot() is None

    def test_record_rate_limits_ignores_missing_headers(self) -> None:
        record_rate_limits_from_response(None)
        record_rate_limits_from_response(_Response(None))


class TestMessageTransforms:
    def test_normalize_image_data_url_repads_base64(self) -> None:
        raw = "data:image/png;base64,aGVsbG8"
        assert _normalize_image_data_url(raw) == "data:image/png;base64,aGVsbG8="

    def test_convert_chat_messages_to_responses_input_handles_tools_images_and_text(self) -> None:
        messages = [
            {"role": "system", "content": "skip"},
            {"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {"url": "https://x/y.png"}}]},
            {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}], "content": "done"},
            {"role": "tool", "tool_call_id": "call_1", "content": [{"text": "tool output"}]},
        ]
        converted = convert_chat_messages_to_responses_input(messages)
        assert converted[0]["type"] == "message"
        assert converted[0]["content"][0] == {"type": "input_text", "text": "hello"}
        assert converted[1] == {"type": "function_call", "name": "lookup", "arguments": "{}", "call_id": "call_1"}
        assert converted[2]["content"][0] == {"type": "output_text", "text": "done"}
        assert converted[3] == {"type": "function_call_output", "call_id": "call_1", "output": "tool output"}

    def test_convert_tools_chat_to_responses_normalizes_invalid_parameters(self) -> None:
        tools = [{"type": "function", "function": {"name": "lookup", "description": "d", "parameters": "bad"}}]
        assert convert_tools_chat_to_responses(tools) == [{"type": "function", "name": "lookup", "description": "d", "strict": False, "parameters": {"type": "object", "properties": {}}}]


class TestOllamaTransforms:
    def test_to_data_url_detects_existing_urls_and_common_base64_prefixes(self) -> None:
        assert to_data_url("https://example.com/a.png") == "https://example.com/a.png"
        assert to_data_url("iVBORw0KGgoAAA") == "data:image/png;base64,iVBORw0KGgoAAA"
        assert to_data_url("/9j/AAA") == "data:image/jpeg;base64,/9j/AAA"

    def test_convert_ollama_messages_adds_tool_calls_and_top_images(self) -> None:
        messages = [
            {"role": "assistant", "content": "thinking", "tool_calls": [{"function": {"name": "search", "arguments": {"q": "weather"}}}]},
            {"role": "tool", "content": "sunny"},
        ]
        converted = convert_ollama_messages(messages, top_images=["R0lGODabc"])
        assert converted[0]["tool_calls"][0]["function"]["name"] == "search"
        assert converted[1]["tool_call_id"] == "ollama_call_1"
        assert converted[2]["content"][0]["image_url"]["url"].startswith("data:image/gif;base64,")

    def test_normalize_ollama_tools_supports_both_tool_shapes(self) -> None:
        tools = [
            {"function": {"name": "lookup", "description": "d", "parameters": {"type": "object"}}},
            {"name": "ping", "description": "p"},
            {"function": {"name": ""}},
        ]
        assert normalize_ollama_tools(tools) == [
            {"type": "function", "function": {"name": "lookup", "description": "d", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "ping", "description": "p", "parameters": {"type": "object", "properties": {}}}},
        ]


class TestSSEHelpers:
    def _ctx(self, compat: str = "think-tags", *, include_usage: bool = False) -> SSEChatContext:
        return SSEChatContext(model="gpt-5", created=123, compat=compat, verbose=False, vlog=None, include_usage=include_usage)

    def test_serialize_tool_args_and_merge_params(self) -> None:
        assert _serialize_tool_args({"q": "weather"}) == '{"q": "weather"}'
        assert _serialize_tool_args("hello") == '{"query": "hello"}'
        params: dict[str, object] = {}
        _merge_ws_params({"q": "weather", "limit": 3, "include": ["a.com"]}, params)
        assert params == {"query": "weather", "max_results": 3, "domains": ["a.com"]}

    def test_handle_web_search_and_output_item_done_emit_tool_calls(self) -> None:
        ctx = self._ctx()
        frames = _handle_web_search(ctx, {"item_id": "call_1", "query": "weather"}, "response.output_item.added.web_search_call")
        assert b'"name": "web_search"' in frames[0]
        done_frames = _handle_web_search(ctx, {"item_id": "call_1", "query": "weather"}, "response.output_item.done.web_search_call.completed")
        assert b'"finish_reason": "tool_calls"' in done_frames[-1]

        item_frames = _handle_output_item_done(
            ctx,
            {"item": {"type": "function_call", "id": "call_2", "name": "lookup", "arguments": {"city": "seoul"}}},
            "response.output_item.done",
        )
        assert len(item_frames) == 2
        assert b'"name": "lookup"' in item_frames[0]
        assert b'"finish_reason": "tool_calls"' in item_frames[1]
        assert _handle_output_item_done(ctx, {"item": {"type": "message"}}, "response.output_item.done") == []

    def test_handle_reasoning_deltas_for_o3_think_tags_and_legacy(self) -> None:
        o3 = self._ctx("o3")
        _handle_summary_part_added(o3, {}, "response.reasoning_summary_part.added")
        frames = _handle_reasoning_delta(o3, {"delta": "first"}, "response.reasoning_summary_text.delta")
        assert b'"reasoning"' in frames[0]

        think = self._ctx("think-tags")
        frames = _handle_reasoning_delta(think, {"delta": "hidden"}, "response.reasoning_text.delta")
        assert b"<think>" in frames[0]
        assert b"hidden" in frames[1]

        legacy = self._ctx("legacy")
        frames = _handle_reasoning_delta(legacy, {"delta": "summary"}, "response.reasoning_summary_text.delta")
        assert b'"reasoning_summary": "summary"' in frames[0]

    def test_handle_content_done_and_completed_emit_annotations_usage_and_done(self) -> None:
        ctx = self._ctx(include_usage=True)
        ann = _handle_content_part_done(ctx, {"part": {"type": "output_text", "annotations": [{"kind": "url"}]}}, "response.content_part.done")
        assert b'"annotations"' in ann[0]
        assert _handle_content_part_done(ctx, {"part": {"type": "output_text"}}, "response.content_part.done") == []

        ctx.think_open = True
        completed = _handle_completed(
            ctx,
            {"response": {"usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}}},
            "response.completed",
        )
        assert any(b"</think>" in frame for frame in completed)
        assert any(b'"usage"' in frame for frame in completed)
        assert completed[-1] == b"data: [DONE]\n\n"

    def test_handle_completed_and_failed_cover_additional_paths(self) -> None:
        ctx = self._ctx("legacy", include_usage=False)
        completed = _handle_completed(ctx, {"response": {}}, "response.completed")
        assert any(b'"finish_reason": "stop"' in frame for frame in completed)
        failed = _handle_completed(self._ctx(), {"response": {"usage": "bad"}}, "response.completed")
        assert failed[-1] == b"data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_sse_translate_text_streams_delta_stop_usage_and_done(self) -> None:
        lines = [
            'data: {"type":"response.output_text.delta","delta":"hi","response":{"id":"resp_1"}}',
            'data: {"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}}',
            'data: [DONE]',
        ]
        upstream = _AsyncResponse(lines)
        frames = [frame async for frame in sse_translate_text(cast(Any, upstream), model="gpt-5", created=123, include_usage=True)]
        assert any(b'"id": "resp_1"' in frame for frame in frames)
        assert any(b'"text": "hi"' in frame for frame in frames)
        assert any(b'"usage"' in frame for frame in frames)
        assert frames[-1] == b"data: [DONE]\n\n"
        assert upstream.closed is True

    @pytest.mark.asyncio
    async def test_sse_translate_text_skips_invalid_lines_and_handles_done_event(self) -> None:
        upstream = _AsyncResponse([
            "event: ignored",
            "data: not-json",
            b'data: {"type":"response.output_text.done"}',
            "data: [DONE]",
        ])
        frames = [frame async for frame in sse_translate_text(cast(Any, upstream), model="gpt-5", created=123)]
        assert any(b'"finish_reason": "stop"' in frame for frame in frames)
        assert upstream.closed is True


class TestReasoningHelpers:
    def test_allowed_efforts_for_known_model_families(self) -> None:
        assert allowed_efforts_for_model(None) == {"minimal", "low", "medium", "high", "xhigh"}
        assert allowed_efforts_for_model("gpt-5") == {"minimal", "low", "medium", "high"}
        assert allowed_efforts_for_model("gpt-5.2") == {"low", "medium", "high", "xhigh"}
        assert allowed_efforts_for_model("gpt-5.1") == {"low", "medium", "high"}
        assert allowed_efforts_for_model("gpt-5.1-codex-max") == {"low", "medium", "high", "xhigh"}

    def test_build_reasoning_param_applies_valid_overrides_and_fallbacks(self) -> None:
        assert build_reasoning_param("bad", "bad", None, allowed_efforts={"low", "medium"}) == {"effort": "medium", "summary": "auto"}
        assert build_reasoning_param("medium", "auto", {"effort": "low", "summary": "detailed"}, allowed_efforts={"low", "medium"}) == {"effort": "low", "summary": "detailed"}
        assert build_reasoning_param("medium", "none", None) == {"effort": "medium"}

    def test_apply_reasoning_to_message_handles_all_compat_modes(self) -> None:
        assert apply_reasoning_to_message({"content": "hello"}, "sum", "full", "o3")["reasoning"]["content"][0]["text"] == "sum\n\nfull"
        legacy = apply_reasoning_to_message({}, "sum", "full", "legacy")
        assert legacy["reasoning_summary"] == "sum"
        assert legacy["reasoning"] == "full"
        think = apply_reasoning_to_message({"content": "hello"}, "sum", "full", cast(Any, object()))
        assert think["content"].startswith("<think>sum\n\nfull</think>hello")

    def test_extract_reasoning_from_model_name_supports_colon_and_suffixes(self) -> None:
        assert extract_reasoning_from_model_name("gpt-5:high") == {"effort": "high"}
        assert extract_reasoning_from_model_name("gpt-5_xhigh") == {"effort": "xhigh"}
        assert extract_reasoning_from_model_name("gpt-5-low") == {"effort": "low"}
        assert extract_reasoning_from_model_name("") is None
        assert extract_reasoning_from_model_name("gpt-5") is None
