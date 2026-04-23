from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient


@pytest.mark.skipif(
    os.getenv("GPTMOCK_RUN_IMAGE_GENERATION_TEST") != "1",
    reason="Live image generation is opt-in because it consumes ChatGPT/Codex credits.",
)
def test_responses_image_generation_live(client: TestClient, tmp_path: Path) -> None:
    payload: dict[str, Any] = {
        "model": os.getenv("GPTMOCK_IMAGE_GENERATION_MODEL", "gpt-5.4-mini"),
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Use the image_generation tool to create a simple 64x64 PNG icon: "
                            "a red circle centered on a white background. Return only the generated image."
                        ),
                    }
                ],
            }
        ],
        "tools": [{"type": "image_generation", "output_format": "png"}],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "stream": False,
    }

    response = client.post("/v1/responses", json=payload, timeout=300)

    assert response.status_code == 200, response.text
    data = response.json()
    image_items = [
        item
        for item in data.get("output", [])
        if isinstance(item, dict) and item.get("type") == "image_generation_call"
    ]
    assert image_items, data

    result = image_items[0].get("result")
    assert isinstance(result, str) and result

    image_bytes = base64.b64decode(result)
    output_path = tmp_path / "probe.png"
    output_path.write_bytes(image_bytes)
    assert output_path.stat().st_size > 100
