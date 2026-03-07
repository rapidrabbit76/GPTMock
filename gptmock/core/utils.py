from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def extract_usage(evt: dict[str, Any]) -> dict[str, int] | None:
    """Extract and normalise usage from an upstream ChatGPT Responses-API event.

    The upstream payload nests usage under ``evt["response"]["usage"]`` with
    keys ``input_tokens`` / ``output_tokens`` / ``total_tokens``.  This helper
    maps them to the OpenAI Chat-Completions naming convention
    (``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``).
    """
    try:
        usage = (evt.get("response") or {}).get("usage")
        if not isinstance(usage, dict):
            return None
        pt = int(usage.get("input_tokens") or 0)
        ct = int(usage.get("output_tokens") or 0)
        tt = int(usage.get("total_tokens") or (pt + ct))
        return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}
    except Exception:
        return None


def parse_datetime(value: Any) -> datetime | None:
    """Parse an ISO-8601 string into a timezone-aware UTC *datetime*.

    Accepts any type; non-strings and empty strings return ``None``.
    Trailing ``"Z"`` is normalised to ``+00:00`` before parsing.
    Naive datetimes are assumed UTC; aware datetimes are converted to UTC.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return None
