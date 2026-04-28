from __future__ import annotations

from typing import Any

DEFAULT_REASONING_EFFORTS: set[str] = {"minimal", "low", "medium", "high", "xhigh"}

EFFORT_ORDER: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh")


def sort_efforts(efforts: set[str] | list[str]) -> list[str]:
    order = {e: i for i, e in enumerate(EFFORT_ORDER)}
    return sorted(efforts, key=lambda e: order.get(e, len(EFFORT_ORDER)))


def strip_effort_suffix(model: str) -> str:
    base = model.split(":", 1)[0]
    for sep in ("-", "_"):
        for effort in EFFORT_ORDER:
            suffix = f"{sep}{effort}"
            if base.endswith(suffix):
                return base[: -len(suffix)]
    return base


def allowed_efforts_for_model(model: str | None) -> set[str]:
    raw = (model or "").strip().lower()
    if not raw:
        return DEFAULT_REASONING_EFFORTS
    normalized = strip_effort_suffix(raw)
    if normalized.startswith(("gpt-5.5", "gpt-5.4", "gpt-5.3", "gpt-5.2")):
        return {"low", "medium", "high", "xhigh"}
    if normalized.startswith("gpt-5.1-codex-max"):
        return {"low", "medium", "high", "xhigh"}
    if normalized.startswith("gpt-5.1"):
        return {"low", "medium", "high"}
    if normalized.startswith("gpt-5-codex"):
        return {"low", "medium", "high"}
    if normalized.startswith("gpt-5"):
        return {"minimal", "low", "medium", "high"}
    return DEFAULT_REASONING_EFFORTS


def build_reasoning_param(
    base_effort: str = "medium",
    base_summary: str = "auto",
    overrides: dict[str, Any] | None = None,
    *,
    allowed_efforts: set[str] | None = None,
) -> dict[str, Any]:
    effort = (base_effort or "").strip().lower()
    summary = (base_summary or "").strip().lower()

    valid_efforts = allowed_efforts or DEFAULT_REASONING_EFFORTS
    valid_summaries = {"auto", "concise", "detailed", "none"}

    if isinstance(overrides, dict):
        o_eff = str(overrides.get("effort", "")).strip().lower()
        o_sum = str(overrides.get("summary", "")).strip().lower()
        if o_eff in valid_efforts and o_eff:
            effort = o_eff
        if o_sum in valid_summaries and o_sum:
            summary = o_sum
    if effort not in valid_efforts:
        effort = "medium"
    if summary not in valid_summaries:
        summary = "auto"

    reasoning: dict[str, Any] = {"effort": effort}
    if summary != "none":
        reasoning["summary"] = summary
    return reasoning


def apply_reasoning_to_message(
    message: dict[str, Any],
    reasoning_summary_text: str,
    reasoning_full_text: str,
    compat: str,
) -> dict[str, Any]:
    try:
        compat = (compat or "standard").strip().lower()
    except Exception:
        compat = "standard"

    if compat == "o3":
        rtxt_parts: list[str] = []
        if isinstance(reasoning_summary_text, str) and reasoning_summary_text.strip():
            rtxt_parts.append(reasoning_summary_text)
        if isinstance(reasoning_full_text, str) and reasoning_full_text.strip():
            rtxt_parts.append(reasoning_full_text)
        rtxt = "\n\n".join([p for p in rtxt_parts if p])
        if rtxt:
            message["reasoning"] = {"content": [{"type": "text", "text": rtxt}]}
        return message

    if compat in ("legacy", "current"):
        if reasoning_summary_text:
            message["reasoning_summary"] = reasoning_summary_text
        if reasoning_full_text:
            message["reasoning"] = reasoning_full_text
        return message

    if compat in ("standard", "openai"):
        rtxt_parts = [
            part
            for part in (reasoning_summary_text, reasoning_full_text)
            if isinstance(part, str) and part.strip()
        ]
        rtxt = "\n\n".join(rtxt_parts)
        if rtxt:
            message["reasoning_content"] = rtxt
        return message

    rtxt_parts: list[str] = []
    if isinstance(reasoning_summary_text, str) and reasoning_summary_text.strip():
        rtxt_parts.append(reasoning_summary_text)
    if isinstance(reasoning_full_text, str) and reasoning_full_text.strip():
        rtxt_parts.append(reasoning_full_text)
    rtxt = "\n\n".join([p for p in rtxt_parts if p])
    if rtxt:
        think_block = f"<think>{rtxt}</think>"
        content_text = message.get("content") or ""
        if isinstance(content_text, str):
            message["content"] = think_block + (content_text or "")
    return message


def extract_reasoning_from_model_name(model: str | None) -> dict[str, Any] | None:
    """Infer reasoning overrides from a model."""
    if not isinstance(model, str) or not model:
        return None
    s = model.strip().lower()
    if not s:
        return None
    efforts = {"minimal", "low", "medium", "high", "xhigh"}

    if ":" in s:
        maybe = s.rsplit(":", 1)[-1].strip()
        if maybe in efforts:
            return {"effort": maybe}

    for sep in ("-", "_"):
        if s.endswith(sep + "minimal"):
            return {"effort": "minimal"}
        if s.endswith(sep + "low"):
            return {"effort": "low"}
        if s.endswith(sep + "medium"):
            return {"effort": "medium"}
        if s.endswith(sep + "high"):
            return {"effort": "high"}
        if s.endswith(sep + "xhigh"):
            return {"effort": "xhigh"}

    return None
