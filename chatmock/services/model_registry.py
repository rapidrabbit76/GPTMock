from __future__ import annotations

from typing import Any, Dict, List


OLLAMA_FAKE_EVAL = {
    "total_duration": 8497226791,
    "load_duration": 1747193958,
    "prompt_eval_count": 24,
    "prompt_eval_duration": 269219750,
    "eval_count": 247,
    "eval_duration": 6413802458,
}


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
        "gpt5-codex": "gpt-5-codex",
        "gpt-5-codex": "gpt-5-codex",
        "gpt-5-codex-latest": "gpt-5-codex",
        "gpt-5.1-codex": "gpt-5.1-codex",
        "gpt-5.1-codex-max": "gpt-5.1-codex-max",
        "codex": "codex-mini-latest",
        "codex-mini": "codex-mini-latest",
        "codex-mini-latest": "codex-mini-latest",
        "gpt-5.1-codex-mini": "gpt-5.1-codex-mini",
    }
    return mapping.get(base, base)


def get_instructions_for_model(model: str, base_instructions: str, gpt5_codex_instructions: str | None) -> str:
    """Return system instructions for a given model."""
    if model.startswith("gpt-5-codex") or model.startswith("gpt-5.1-codex") or model.startswith("gpt-5.2-codex"):
        if isinstance(gpt5_codex_instructions, str) and gpt5_codex_instructions.strip():
            return gpt5_codex_instructions
    return base_instructions


def get_model_list(
    expose_reasoning: bool = False,
) -> List[Dict[str, Any]]:
    """Return unified model list for both OpenAI and Ollama formats."""
    model_groups = [
        ("gpt-5", ["high", "medium", "low", "minimal"]),
        ("gpt-5.1", ["high", "medium", "low"]),
        ("gpt-5.2", ["xhigh", "high", "medium", "low"]),
        ("gpt-5-codex", ["high", "medium", "low"]),
        ("gpt-5.2-codex", ["xhigh", "high", "medium", "low"]),
        ("gpt-5.1-codex", ["high", "medium", "low"]),
        ("gpt-5.1-codex-max", ["xhigh", "high", "medium", "low"]),
        ("gpt-5.1-codex-mini", []),
        ("codex-mini", []),
    ]
    
    model_ids: List[str] = []
    for base, efforts in model_groups:
        model_ids.append(base)
        if expose_reasoning:
            model_ids.extend([f"{base}-{effort}" for effort in efforts])
    
    return model_ids


def get_openai_models(expose_reasoning: bool = False) -> List[Dict[str, Any]]:
    """Return OpenAI-formatted model list."""
    model_ids = get_model_list(expose_reasoning)
    return [{"id": mid, "object": "model", "owned_by": "owner"} for mid in model_ids]


def get_ollama_models(expose_reasoning: bool = False) -> List[Dict[str, Any]]:
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
            }
        )
    return models
