from __future__ import annotations

import pytest

from gptmock.services.upstream_errors import extract_upstream_error_message


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"error": {"message": "standard error"}}, "standard error"),
        ({"detail": "System messages are not allowed"}, "System messages are not allowed"),
        ({"detail": [{"loc": ["body", "input"], "msg": "required"}]}, '[{"loc":["body","input"],"msg":"required"}]'),
        ({"message": "top-level message"}, "top-level message"),
        ({"raw": "plain upstream failure"}, "plain upstream failure"),
        ("plain upstream failure", "plain upstream failure"),
    ],
)
def test_extract_upstream_error_message_preserves_known_shapes(
    body: object,
    expected: str,
) -> None:
    assert extract_upstream_error_message(body, status_code=400) == expected


def test_extract_upstream_error_message_uses_status_fallback() -> None:
    assert extract_upstream_error_message({}, status_code=503) == "Upstream HTTP 503"
