from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _read_prompt_text(filename: str) -> str | None:
    candidates = [
        Path(__file__).parent.parent.parent / filename,
        Path(__file__).parent.parent / filename,
        Path(getattr(sys, "_MEIPASS", "")) / filename if getattr(sys, "_MEIPASS", None) else None,
        Path.cwd() / filename,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            if candidate.exists():
                content = candidate.read_text(encoding="utf-8")
                if isinstance(content, str) and content.strip():
                    return content
        except Exception:
            continue
    return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHATMOCK_",
        env_file=".env",
        case_sensitive=False,
    )

    verbose: bool = False
    verbose_obfuscation: bool = False
    reasoning_effort: str = "medium"
    reasoning_summary: str = "auto"
    reasoning_compat: str = "think-tags"
    debug_model: str | None = None
    expose_reasoning_models: bool = False
    default_web_search: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    ollama_version: str = "0.12.10"

    @computed_field
    @property
    def base_instructions(self) -> str:
        content = _read_prompt_text("prompt.md")
        if content is None:
            raise FileNotFoundError("Failed to read prompt.md; expected adjacent to package or CWD.")
        return content

    @computed_field
    @property
    def gpt5_codex_instructions(self) -> str:
        content = _read_prompt_text("prompt_gpt5_codex.md")
        return content if isinstance(content, str) and content.strip() else self.base_instructions
