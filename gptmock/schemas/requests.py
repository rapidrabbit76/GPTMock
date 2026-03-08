"""Pydantic request models for API endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ChatCompletionRequest(BaseModel):
    """POST /v1/chat/completions request body."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]] | None = None
    prompt: str | None = None
    input: str | None = None
    stream: bool = False


class TextCompletionRequest(BaseModel):
    """POST /v1/completions request body."""

    model_config = ConfigDict(extra="allow")

    model: str
    prompt: str | None = None
    stream: bool = False


class ResponsesCreateRequest(BaseModel):
    """POST /v1/responses request body."""

    model_config = ConfigDict(extra="allow")

    model: str
    input: list[dict[str, Any]] | Any = None
    stream: bool = False


class OllamaShowRequest(BaseModel):
    """POST /api/show request body."""

    model_config = ConfigDict(extra="allow")

    model: str = ""


class OllamaChatRequest(BaseModel):
    """POST /api/chat request body."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]]
    stream: bool = True
