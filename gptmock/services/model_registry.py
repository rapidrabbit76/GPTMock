from __future__ import annotations

from typing import Any

from gptmock.services.reasoning import (
    EFFORT_ORDER,
    allowed_efforts_for_model,
    sort_efforts,
    strip_effort_suffix,
)

OLLAMA_FAKE_EVAL = {
    "total_duration": 8497226791,
    "load_duration": 1747193958,
    "prompt_eval_count": 24,
    "prompt_eval_duration": 269219750,
    "eval_count": 247,
    "eval_duration": 6413802458,
}

MODEL_GROUPS: list[tuple[str, list[str]]] = [
    ("gpt-5", ["high", "medium", "low", "minimal"]),
    ("gpt-5.1", ["high", "medium", "low"]),
    ("gpt-5.2", ["xhigh", "high", "medium", "low"]),
    ("gpt-5-codex", ["high", "medium", "low"]),
    ("gpt-5.2-codex", ["xhigh", "high", "medium", "low"]),
    ("gpt-5.3-codex", ["xhigh", "high", "medium", "low"]),
    ("gpt-5.3-codex-spark", ["xhigh", "high", "medium", "low"]),
    ("gpt-5.1-codex", ["high", "medium", "low"]),
    ("gpt-5.1-codex-mini", ["high", "medium", "low"]),
    ("gpt-5.1-codex-max", ["xhigh", "high", "medium", "low"]),
    ("gpt-5.4", ["xhigh", "high", "medium", "low"]),
    ("gpt-5.4-mini", ["xhigh", "high", "medium", "low"]),
]

_BASE_MODEL_IDS: frozenset[str] = frozenset(base for base, _ in MODEL_GROUPS)


def normalize_model_name(name: str | None, debug_model: str | None = None) -> str:
    if isinstance(debug_model, str) and debug_model.strip():
        return debug_model.strip()
    if not isinstance(name, str) or not name.strip():
        return "gpt-5"
    base = name.split(":", 1)[0].strip()
    for sep in ("-", "_"):
        lowered = base.lower()
        for effort in ("minimal", "low", "medium", "high", "xhigh"):
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
        "gpt5.4-mini": "gpt-5.4-mini",
        "gpt-5.4-mini": "gpt-5.4-mini",
        "gpt-5.4-mini-latest": "gpt-5.4-mini",
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


def get_model_list(
    expose_reasoning: bool = False,
) -> list[str]:
    """Return unified model list for both OpenAI and Ollama formats."""
    model_ids: list[str] = []
    for base, efforts in MODEL_GROUPS:
        model_ids.append(base)
        if expose_reasoning:
            model_ids.extend([f"{base}-{effort}" for effort in efforts])

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
        models.append(
            {
                "name": model_id,
                "model": model_id,
                "modified_at": "2023-10-01T00:00:00Z",
                "size": 815319791,
                "digest": "8648f39daa8fbf5b18c7b4e6a8fb4990c692751d49917417b8842ca5758e7ffc",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "llama",
                    "families": ["llama"],
                    "parameter_size": "8.0B",
                    "quantization_level": "Q4_0",
                },
            },
        )
    return models
