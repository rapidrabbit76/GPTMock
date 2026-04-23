from __future__ import annotations

import base64
import importlib
import json
from typing import Any

import httpx
import pytest

from gptmock.core.settings import Settings


def _sse_response(events: list[dict[str, Any]]) -> httpx.Response:
    payload = b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events)
    return httpx.Response(200, content=payload)


@pytest.mark.asyncio
async def test_responses_view_image_executes_tool_and_follows_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    image_path = tmp_path / "cat.png"
    image_path.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="))
    monkeypatch.setenv("GPTMOCK_VIEW_IMAGE_ROOTS", str(tmp_path))

    responses_module = importlib.import_module("gptmock.services.responses")
    captured_payloads: list[dict[str, Any]] = []

    async def fake_auth() -> tuple[str, str]:
        return "token", "account"

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
        captured_payloads.append(payload)
        if len(captured_payloads) == 1:
            return _sse_response(
                [
                    {
                        "type": "response.output_item.done",
                        "item": {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_view",
                            "name": "view_image",
                            "arguments": json.dumps({"path": str(image_path), "detail": "original"}),
                            "status": "completed",
                        },
                        "response": {"id": "resp_tool"},
                    },
                    {"type": "response.completed", "response": {"id": "resp_tool"}},
                ],
            )
        return _sse_response(
            [
                {"type": "response.output_text.delta", "delta": "saw image", "response": {"id": "resp_final"}},
                {"type": "response.completed", "response": {"id": "resp_final"}},
            ],
        )

    monkeypatch.setattr(responses_module, "get_effective_chatgpt_auth", fake_auth)
    monkeypatch.setattr(responses_module, "send_upstream_request", fake_send_upstream_request)

    async with httpx.AsyncClient() as client:
        result, is_streaming = await responses_module.process_responses_api(
            {
                "model": "gpt-5.4-mini",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "What is in this image?"}],
                    }
                ],
                "tools": [{"type": "view_image"}],
                "stream": False,
            },
            Settings(),
            client,
        )

    assert is_streaming is False
    assert result["output"][0]["content"][0]["text"] == "saw image"
    assert len(captured_payloads) == 2
    assert captured_payloads[0]["tools"][0]["type"] == "function"
    assert captured_payloads[0]["tools"][0]["name"] == "view_image"

    followup_items = captured_payloads[1]["input"]
    function_output = next(item for item in followup_items if item.get("type") == "function_call_output")
    assert function_output["call_id"] == "call_view"
    output_items = function_output["output"]
    assert output_items[0]["type"] == "input_image"
    assert output_items[0]["detail"] == "original"
    assert output_items[0]["image_url"].startswith("data:image/png;base64,")
