from __future__ import annotations

from typing import Any

from gptmock.services.reasoning import (
    EFFORT_ORDER,
    allowed_efforts_for_model,
    sort_efforts,
    strip_effort_suffix,
)

MODEL_GROUPS: list[tuple[str, list[str]]] = [
    ("gpt-5.3-codex-spark", ["xhigh", "high", "medium", "low"]),
    ("gpt-5.4", ["xhigh", "high", "medium", "low"]),
    ("gpt-5.5", ["xhigh", "high", "medium", "low"]),
    ("gpt-5.6", ["max", "xhigh", "high", "medium", "low", "none"]),
    ("gpt-5.6-sol", ["max", "xhigh", "high", "medium", "low", "none"]),
    ("gpt-5.6-terra", ["max", "xhigh", "high", "medium", "low", "none"]),
    ("gpt-5.6-luna", ["max", "xhigh", "high", "medium", "low", "none"]),
    ("gpt-6-astra", ["max", "xhigh", "high", "medium", "low"]),
    ("gpt-5.4-mini", ["xhigh", "high", "medium", "low"]),
]

SYNTHETIC_MODEL_GROUPS: list[tuple[str, list[str]]] = [
    ("gpt-5.4-fast", ["xhigh", "high", "medium", "low"]),
    ("gpt-5.5-fast", ["xhigh", "high", "medium", "low"]),
    ("gpt-5.6-fast", ["max", "xhigh", "high", "medium", "low", "none"]),
    ("gpt-5.6-sol-fast", ["max", "xhigh", "high", "medium", "low", "none"]),
    ("gpt-5.6-terra-fast", ["max", "xhigh", "high", "medium", "low", "none"]),
    ("gpt-5.6-luna-fast", ["max", "xhigh", "high", "medium", "low", "none"]),
    ("gpt-6-astra-fast", ["max", "xhigh", "high", "medium", "low"]),
    ("gpt-5.4-mini-fast", ["xhigh", "high", "medium", "low"]),
]

_BASE_MODEL_IDS: frozenset[str] = frozenset(
    base for base, _ in (*MODEL_GROUPS, *SYNTHETIC_MODEL_GROUPS)
)

MODEL_REGISTRY_VERIFIED_AT = "2026-09-05T00:00:00Z"

UPSTREAM_MODEL_ALIASES: dict[str, str] = {
    "gpt-5.6": "gpt-5.6-sol",
}

FAST_MODEL_ALIASES: dict[str, str] = {
    "gpt-5.4-fast": "gpt-5.4",
    "gpt-5.5-fast": "gpt-5.5",
    "gpt-5.6-fast": "gpt-5.6-sol",
    "gpt-5.6-sol-fast": "gpt-5.6-sol",
    "gpt-5.6-terra-fast": "gpt-5.6-terra",
    "gpt-5.6-luna-fast": "gpt-5.6-luna",
    "gpt-6-astra-fast": "gpt-6-astra",
    "gpt-5.4-mini-fast": "gpt-5.4-mini",
}

FAST_SERVICE_TIER: str = "priority"


def normalize_model_name(name: str | None, debug_model: str | None = None) -> str:
    raw = name if isinstance(name, str) else ""
    raw = raw.strip().lower().split(":", 1)[0].replace("_", "-").replace("gpt6-", "gpt-6-")
    if raw in {"gpt-6-astra-max", "gpt-6-astra-fast-max"}:
        raise ValueError("Removed model alias: use gpt-6-astra with reasoning.effort='max' (or reasoning_effort='max')")
    if isinstance(debug_model, str) and debug_model.strip():
        return debug_model.strip()
    if not isinstance(name, str) or not name.strip():
        return "gpt-5.4"
    base = name.split(":", 1)[0].strip()
    for sep in ("-", "_"):
        lowered = base.lower()
        if base == "gpt-5.1-codex-max":
            return base
        for effort in EFFORT_ORDER:
            suffix = f"{sep}{effort}"
            if lowered.endswith(suffix):
                base = base[: -len(suffix)]
                break
    mapping = {
        "gpt5": "gpt-5",
        "gpt-5-latest": "gpt-5",
        "gpt-5": "gpt-5",
        "gpt-5.1": "gpt-5.1",
        "gpt5.2": "gpt-5.2",
        "gpt-5.2": "gpt-5.2",
        "gpt-5.2-latest": "gpt-5.2",
        "gpt5.2-codex": "gpt-5.2-codex",
        "gpt-5.2-codex": "gpt-5.2-codex",
        "gpt-5.2-codex-latest": "gpt-5.2-codex",
        "gpt5.3-codex": "gpt-5.3-codex",
        "gpt-5.3-codex": "gpt-5.3-codex",
        "gpt-5.3-codex-latest": "gpt-5.3-codex",
        "gpt5.3-codex-spark": "gpt-5.3-codex-spark",
        "gpt-5.3-codex-spark": "gpt-5.3-codex-spark",
        "gpt-5.3-codex-spark-latest": "gpt-5.3-codex-spark",
        "gpt5-codex": "gpt-5-codex",
        "gpt-5-codex": "gpt-5-codex",
        "gpt-5-codex-latest": "gpt-5-codex",
        "gpt-5.1-codex": "gpt-5.1-codex",
        "gpt-5.1-codex-max": "gpt-5.1-codex-max",
        "codex-mini": "gpt-5.1-codex-mini",
        "gpt5.1-codex-mini": "gpt-5.1-codex-mini",
        "gpt-5.1-codex-mini": "gpt-5.1-codex-mini",
        "gpt-5.1-codex-mini-latest": "gpt-5.1-codex-mini",
        "gpt5.4": "gpt-5.4",
        "gpt-5.4": "gpt-5.4",
        "gpt-5.4-latest": "gpt-5.4",
        "gpt5.5": "gpt-5.5",
        "gpt-5.5": "gpt-5.5",
        "gpt-5.5-latest": "gpt-5.5",
        "gpt5.6": "gpt-5.6",
        "gpt-5.6": "gpt-5.6",
        "gpt-5.6-latest": "gpt-5.6",
        "gpt5.6-sol": "gpt-5.6-sol",
        "gpt-5.6-sol": "gpt-5.6-sol",
        "gpt-5.6-sol-latest": "gpt-5.6-sol",
        "gpt5.6-terra": "gpt-5.6-terra",
        "gpt-5.6-terra": "gpt-5.6-terra",
        "gpt-5.6-terra-latest": "gpt-5.6-terra",
        "gpt5.6-luna": "gpt-5.6-luna",
        "gpt-5.6-luna": "gpt-5.6-luna",
        "gpt-5.6-luna-latest": "gpt-5.6-luna",
        "gpt6-astra": "gpt-6-astra",
        "gpt-6-astra": "gpt-6-astra",
        "gpt-6-astra-latest": "gpt-6-astra",
        "gpt5.4-mini": "gpt-5.4-mini",
        "gpt-5.4-mini": "gpt-5.4-mini",
        "gpt-5.4-mini-latest": "gpt-5.4-mini",
        "gpt5.4-fast": "gpt-5.4-fast",
        "gpt-5.4-fast": "gpt-5.4-fast",
        "gpt-5.4-fast-latest": "gpt-5.4-fast",
        "gpt5.5-fast": "gpt-5.5-fast",
        "gpt-5.5-fast": "gpt-5.5-fast",
        "gpt-5.5-fast-latest": "gpt-5.5-fast",
        "gpt5.6-fast": "gpt-5.6-fast",
        "gpt-5.6-fast": "gpt-5.6-fast",
        "gpt-5.6-fast-latest": "gpt-5.6-fast",
        "gpt5.6-sol-fast": "gpt-5.6-sol-fast",
        "gpt-5.6-sol-fast": "gpt-5.6-sol-fast",
        "gpt-5.6-sol-fast-latest": "gpt-5.6-sol-fast",
        "gpt5.6-terra-fast": "gpt-5.6-terra-fast",
        "gpt-5.6-terra-fast": "gpt-5.6-terra-fast",
        "gpt-5.6-terra-fast-latest": "gpt-5.6-terra-fast",
        "gpt5.6-luna-fast": "gpt-5.6-luna-fast",
        "gpt-5.6-luna-fast": "gpt-5.6-luna-fast",
        "gpt-5.6-luna-fast-latest": "gpt-5.6-luna-fast",
        "gpt6-astra-fast": "gpt-6-astra-fast",
        "gpt-6-astra-fast": "gpt-6-astra-fast",
        "gpt-6-astra-fast-latest": "gpt-6-astra-fast",
        "gpt5.4-mini-fast": "gpt-5.4-mini-fast",
        "gpt-5.4-mini-fast": "gpt-5.4-mini-fast",
        "gpt-5.4-mini-fast-latest": "gpt-5.4-mini-fast",
    }
    return mapping.get(base, base)


def get_instructions_for_model(
    model: str, base_instructions: str, gpt5_codex_instructions: str | None,
) -> str:
    """Return system instructions for a given model."""
    if (
        model.startswith("gpt-5-codex")
        or model.startswith("gpt-5.1-codex")
        or model.startswith("gpt-5.2-codex")
        or model.startswith("gpt-5.3-codex")
    ):
        if isinstance(gpt5_codex_instructions, str) and gpt5_codex_instructions.strip():
            return gpt5_codex_instructions
    return base_instructions


def resolve_upstream_model(model: str) -> tuple[str, dict[str, Any]]:
    """Return the upstream model ID and provider body overrides."""
    if model in FAST_MODEL_ALIASES:
        return FAST_MODEL_ALIASES[model], {"service_tier": FAST_SERVICE_TIER}
    if model in UPSTREAM_MODEL_ALIASES:
        return UPSTREAM_MODEL_ALIASES[model], {}
    return model, {}


def apply_model_overrides(payload: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Merge model-specific provider body overrides into an upstream payload."""
    for key, value in overrides.items():
        current = payload.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            payload[key] = {**current, **value}
        else:
            payload[key] = value


def get_model_list(
    expose_reasoning: bool = False,
) -> list[str]:
    """Return unified model list for both OpenAI and Ollama formats."""
    model_ids: list[str] = []
    for base, efforts in MODEL_GROUPS:
        model_ids.append(base)
        if expose_reasoning:
            model_ids.extend(f"{base}-{effort}" for effort in efforts if not (base == "gpt-6-astra" and effort == "max"))

    return model_ids


def _detect_preset_effort(model_id: str) -> str | None:
    if model_id in _BASE_MODEL_IDS:
        return None
    for sep in ("-", "_"):
        for effort in reversed(EFFORT_ORDER):
            suffix = f"{sep}{effort}"
            if model_id.endswith(suffix):
                base = model_id[: -len(suffix)]
                if base in _BASE_MODEL_IDS:
                    return effort
                return None
    return None


def _reasoning_metadata(model_id: str, default_effort: str) -> dict[str, Any]:
    base = strip_effort_suffix(model_id)
    supported = sort_efforts(allowed_efforts_for_model(base))

    if default_effort in supported:
        default = default_effort
    elif "medium" in supported:
        default = "medium"
    elif supported:
        default = supported[0]
    else:
        default = default_effort

    metadata: dict[str, Any] = {
        "supported_efforts": supported,
        "default_effort": default,
    }

    preset = _detect_preset_effort(model_id)
    if preset is not None and preset in supported:
        metadata["preset_effort"] = preset

    return metadata


def get_openai_models(
    expose_reasoning: bool = False,
    default_effort: str = "medium",
) -> list[dict[str, Any]]:
    """Return OpenAI-formatted model list."""
    model_ids = get_model_list(expose_reasoning)
    return [
        {
            "id": mid,
            "object": "model",
            "owned_by": "owner",
            "reasoning": _reasoning_metadata(mid, default_effort),
        }
        for mid in model_ids
    ]


def get_ollama_models(expose_reasoning: bool = False) -> list[dict[str, Any]]:
    """Return Ollama-formatted model list."""
    model_ids = get_model_list(expose_reasoning)
    models = []
    for model_id in model_ids:
        upstream_model, _ = resolve_upstream_model(model_id)
        models.append(
            {
                "name": model_id,
                "model": model_id,
                "remote_model": upstream_model,
                "modified_at": MODEL_REGISTRY_VERIFIED_AT,
                "size": 0,
                "digest": "",
                "details": {
                    "parent_model": "",
                    "format": "remote",
                    "family": "openai",
                    "families": ["openai"],
                },
                "capabilities": ["completion", "tools", "thinking"],
            },
        )
    return models
