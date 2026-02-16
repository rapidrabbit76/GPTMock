# AGENTS.md - Development Guide for ChatMock

This guide provides coding agents with essential information about building, testing, and contributing to ChatMock.

## Overview

ChatMock is an OpenAI & Ollama compatible API server powered by ChatGPT accounts. It's a **FastAPI-based** Python application that proxies requests to ChatGPT's Codex backend, enabling usage of GPT-5 models through a local API endpoint.

**Python Version**: `>=3.13` (strictly enforced in pyproject.toml)
**Build System**: `uv` (replaces pip/setuptools)
**Framework**: `FastAPI` (async/await throughout)
**HTTP Client**: `httpx` (async)
**Configuration**: `pydantic-settings` (environment variables)

## Project Structure

```
chatmock/
├── app.py              # FastAPI app factory + lifespan
├── cli.py              # CLI commands (login, serve, info) — click-based
│
├── core/               # Shared core
│   ├── settings.py     # pydantic-settings BaseSettings
│   ├── models.py       # Data classes (TokenData, AuthBundle, PkceCodes)
│   ├── dependencies.py # FastAPI DI (get_settings, get_http_client)
│   ├── logging.py      # Common logging utilities
│   └── constants.py    # OAuth and API URL constants
│
├── routers/            # Router Layer (API endpoints)
│   ├── openai.py       # /v1/chat/completions, /v1/completions, /v1/models
│   ├── ollama.py       # /api/version, /api/tags, /api/show, /api/chat
│   └── health.py       # /, /health
│
├── services/           # Service Layer (business logic)
│   ├── chat.py         # Chat completion business logic (OpenAI + Ollama shared)
│   ├── reasoning.py    # Reasoning parameter building
│   └── model_registry.py # Model list/normalization management
│
├── infra/              # Infrastructure Layer (external I/O)
│   ├── upstream.py     # httpx upstream requests (async)
│   ├── sse.py          # SSE stream translation (async generators)
│   ├── auth.py         # Token load/refresh/save (async httpx)
│   ├── oauth.py        # OAuth PKCE flow
│   ├── session.py      # Session ID management
│   └── limits.py       # Rate limit tracking
│
├── schemas/            # Request/response schemas
│   ├── messages.py     # Message conversion functions
│   └── transform.py    # Ollama ↔ OpenAI format conversion
│
prompt.md               # System prompt (DO NOT MODIFY)
prompt_gpt5_codex.md    # Codex prompt (DO NOT MODIFY)
```

## Commands

### Installation & Setup

```bash
# Install dependencies (using uv)
uv sync

# Authenticate with ChatGPT account
uv run python chatmock.py login

# Verify authentication
uv run python chatmock.py info
```

### Running the Server

```bash
# Basic server start (default: http://127.0.0.1:8000)
uv run python chatmock.py serve

# With custom port and host
uv run python chatmock.py serve --port 9000 --host 0.0.0.0

# With reasoning effort and summary settings
uv run python chatmock.py serve --reasoning-effort high --reasoning-summary detailed

# Enable verbose logging
uv run python chatmock.py serve --verbose

# Enable web search by default
uv run python chatmock.py serve --enable-web-search

# Expose reasoning models as separate queryable models
uv run python chatmock.py serve --expose-reasoning-models
```

### Testing

**No automated test suite exists.** Manual testing:
1. Run `uv run python chatmock.py login` to authenticate
2. Start server with `uv run python chatmock.py serve`
3. Make test requests to `http://127.0.0.1:8000/v1/chat/completions`

### Building

```bash
# Build package distribution
uv build

# Build script (for macOS app)
python build.py
```

### Docker

See [DOCKER.md](DOCKER.md) for container-based deployment.

## Code Style Guidelines

### Import Order

All Python files use `from __future__ import annotations` as the **first import**, followed by:
1. Standard library imports (grouped)
2. Third-party imports (grouped)
3. Local/relative imports (grouped)

**Example:**
```python
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

from flask import Flask, jsonify, request

from .config import BASE_INSTRUCTIONS
from .utils import eprint
```

### Type Hints

- **Always use type hints** for function signatures
- Use modern `|` syntax for Optional types: `str | None` (not `Optional[str]`)
- Use `Dict`, `List`, `Any` from `typing` module
- Return type annotations are mandatory for public functions

**Example:**
```python
def load_chatgpt_tokens(ensure_fresh: bool = True) -> tuple[str | None, str | None, str | None]:
    ...

def build_reasoning_param(
    base_effort: str = "medium",
    base_summary: str = "auto",
    overrides: Dict[str, Any] | None = None,
    *,
    allowed_efforts: Set[str] | None = None,
) -> Dict[str, Any]:
    ...
```

### Naming Conventions

- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private/internal**: Prefix with `_` (e.g., `_log_json`, `_derive_account_id`)
- **Type-safe conversions**: Use explicit type checks before operations

### Error Handling

- Use specific exception types when possible
- Broad `except Exception` is acceptable for graceful degradation
- **Always provide context** in error messages with `eprint()` for user-facing errors
- Silent failures are acceptable for non-critical operations (e.g., logging, cleanup)

**Example:**
```python
try:
    data = json.loads(raw)
except (json.JSONDecodeError, UnicodeDecodeError):
    return jsonify({"error": {"message": "Invalid JSON body"}}), 400

try:
    resp = requests.post(OAUTH_TOKEN_URL, json=payload, timeout=30)
except requests.RequestException as exc:
    eprint(f"ERROR: failed to refresh ChatGPT token: {exc}")
    return None
```

### Docstrings & Comments

- **No docstrings** are used in the current codebase (pragmatic approach)
- Comments are minimal; prefer self-documenting code
- Use inline comments only for complex/non-obvious logic
- Document public APIs in README.md, not in code

### String Handling

- Use double quotes `"` for strings (consistent throughout codebase)
- Use f-strings for formatting
- Multi-line strings: triple double-quotes `"""`

### Data Structures

- Prefer `dataclasses` for DTOs (see `models.py`)
- Use `dict` for dynamic/config data
- Type `Dict[str, Any]` for JSON-like structures

### Async/Await Patterns

- **All route handlers are async**: `async def handler(...)`
- **All service functions are async**: `async def process_chat_completion(...)`
- **SSE translation uses async generators**: `async def sse_translate_chat(...) -> AsyncGenerator[str, None]`
- **HTTP client is async**: `httpx.AsyncClient()` managed by FastAPI lifespan
- **Request body reading is async**: `await request.body()`

**Example:**
```python
@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    settings: Settings = Depends(get_settings),
):
    raw_body = await request.body()
    response, is_streaming = await process_chat_completion(settings, payload)
    return response
```

### FastAPI Patterns

- Use APIRouter for route grouping (replaces Flask Blueprints)
- App config via `Settings` class with pydantic-settings
- CORS handled via `CORSMiddleware` at app level
- Dependency injection via `Depends()` for settings and HTTP client
- Health checks at `/` and `/health`

## Critical Files - DO NOT MODIFY Without Approval

Per [CONTRIBUTING.md](CONTRIBUTING.md):

- **`prompt.md`** and **`prompt_gpt5_codex.md`**: System prompts for models. These are sensitive harness files.
- **Entry points**: CLI commands, route signatures, response payloads must maintain backward compatibility.
- **Parameter names**: Changing public API parameter names breaks downstream clients.

Always coordinate with maintainers before touching these areas.

## Common Patterns

### SSE (Server-Sent Events) Translation

The `sse_translate_chat()` and `sse_translate_text()` functions in `utils.py` convert upstream ChatGPT responses to OpenAI-compatible SSE format. They handle:
- Reasoning summaries (`<think>` tags or `reasoning` field based on `compat` mode)
- Tool calls (including web search)
- Usage statistics
- Graceful connection failures

### Reasoning Configuration

Three modes controlled by `--reasoning-compat`:
- `"think-tags"` (default): Wrap reasoning in `<think>...</think>` tags
- `"o3"`: Use `reasoning.content` structure
- `"legacy"`: Use separate `reasoning` and `reasoning_summary` fields

Reasoning effort levels: `minimal`, `low`, `medium`, `high`, `xhigh` (model-dependent)

### Message Conversion

`convert_chat_messages_to_responses_input()` transforms OpenAI chat format to ChatGPT's `responses` API format:
- Filters out system messages
- Handles tool calls and tool results
- Normalizes image data URLs
- Supports multi-part content (text + images)

## Development Workflow

1. **Make changes** to source files
2. **Test manually** by running server and making requests
3. **Verify** with multiple clients if API changes were made
4. **Update docs** (README.md, DOCKER.md) for user-facing changes
5. **Keep commits focused** - one logical change per commit

## Important Notes

- **No linting/formatting enforced**: Project has no `.flake8`, `ruff.toml`, or similar configs. Match existing style.
- **No type checking**: No `mypy.ini` present. Type hints are for documentation, not enforced.
- **Auth files location**: `~/.chatgpt-local/auth.json` (or `CHATGPT_LOCAL_HOME` env var)
- **OAuth flow**: Uses PKCE with local HTTP server on port 1338
- **Supported models**: `gpt-5`, `gpt-5.1`, `gpt-5.2`, and their `-codex` variants

## Feature Flags

When adding new features:
- Prefer **opt-in flags** (CLI arguments or config switches)
- Leave defaults unchanged until approved by maintainers
- Document flags in README.md with examples
- Consider compatibility with popular clients (Jan, Raycast, OpenAI SDKs)

## Debugging

Enable verbose logging with `--verbose`:
```bash
uv run python chatmock.py serve --verbose
```

This logs:
- Incoming request bodies
- Outgoing upstream payloads
- SSE stream chunks
- Tool call events

Use `--verbose-obfuscation` to reduce sensitive data in logs.

## Dependencies

Core dependencies (from `pyproject.toml`):
- `fastapi==0.115.0` - Web framework
- `uvicorn[standard]==0.32.0` - ASGI server
- `httpx==0.27.2` - Async HTTP client
- `pydantic-settings==2.6.1` - Settings management
- `click==8.2.1` - CLI framework
- `certifi==2024.12.14` - SSL certificates (OAuth)

No testing frameworks, linters, or dev tools are currently installed.

## PR Checklist

Before submitting changes:
- [ ] Rebased on latest `main`
- [ ] Manual testing completed with sample requests
- [ ] README.md updated if user-facing features changed
- [ ] DOCKER.md updated if Docker setup changed
- [ ] No generated files committed (`__pycache__/`, `.pytest_cache/`, `build/`, `dist/`)
- [ ] Critical paths reviewed (`prompt.md`, routing, parameter names)
- [ ] Issue reference included in PR description


## Git Mastor 
- Author is "yslee.dev@gmail.com"
- You must Atomic Commits and write a Udacity-style commit message with a header, body, and footer. The header should be in the format of "type(scope): subject", where type is one of "feat", "fix", "docs", "style", "refactor", "perf", "test", or "chore". The scope is optional and can be anything that describes the area of the code being changed. The subject should be a brief description of the change. The body should provide more detailed information about the change, and the footer should include any relevant issue references or breaking change notes.