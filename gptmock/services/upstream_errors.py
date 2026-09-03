"""Helpers for preserving useful upstream error details."""

from __future__ import annotations

import json
from typing import Any


def _render_error_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if value is None:
        return None
    try:
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        rendered = str(value)
    return rendered.strip() or None


def extract_upstream_error_message(
    error_body: Any,
    *,
    status_code: int | None = None,
) -> str:
    """Extract an upstream error without assuming a single response envelope."""
    if isinstance(error_body, dict):
        error = error_body.get("error")
        if isinstance(error, dict):
            message = _render_error_value(error.get("message"))
            if message:
                return message
        elif isinstance(error, str) and error.strip():
            return error.strip()

        for key in ("detail", "message", "raw"):
            message = _render_error_value(error_body.get(key))
            if message:
                return message
    else:
        message = _render_error_value(error_body)
        if message:
            return message

    if status_code is not None:
        return f"Upstream HTTP {status_code}"
    return "Upstream error"
