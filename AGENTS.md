# Repository Guidelines

## Project Structure & Module Organization

`gptmock/` contains the application package. Use `app.py` for FastAPI app creation and `cli.py` for the `gptmock` entry point. Keep HTTP endpoints in `gptmock/routers/`, request orchestration and response shaping in `gptmock/services/`, external integrations and runtime plumbing in `gptmock/infra/`, and shared settings/utilities in `gptmock/core/`. Request/response conversion helpers live in `gptmock/schemas/`.

Tests live in `tests/`. The default suite covers core API and client behavior; `tests/tools/` contains opt-in tool-calling coverage and is ignored by default. Documentation assets are in `assets/`, and container files are in `docker/`.

## Build, Test, and Development Commands

- `uv sync --dev` — install runtime and development dependencies.
- `uv run ruff check gptmock/ tests/` — run the same lint check used in CI.
- `uv run pytest` — run the default local test suite with `pytest-testmon` enabled.
- `uv run pytest tests/ --no-testmon -q` — run the full deterministic suite.
- `./scripts/test.sh` — run coverage and badge-refresh workflow; requires `GIST_TOKEN`.
- `uv run gptmock serve --port 8000` — start the local API server.
- `uv build` — build distribution artifacts.

## Coding Style & Naming Conventions

Target Python 3.13+ and follow the existing async FastAPI style. Use 4-space indentation, explicit type hints for new or changed code, and concise docstrings where the module already uses them. Ruff enforces import sorting and lint rules with a 120-character line length.

Use `snake_case` for modules, functions, and variables; `PascalCase` for classes and Pydantic models; and `UPPER_SNAKE_CASE` for constants. Keep routers thin: business logic belongs in `services/`, and network/auth/session side effects belong in `infra/`.

## Testing Guidelines

Use `pytest`, `pytest-asyncio`, `pytest-cov`, and `pytest-testmon`. Name tests `test_*.py`, mirroring the feature under test, for example `tests/test_responses_api.py`. Prefer in-process `TestClient` coverage over external server processes. Add focused tests for touched routes, service transforms, and environment precedence. Run `tests/tools/` only when changing tool-calling behavior.

## Commit & Pull Request Guidelines

Follow the commit style already in history: `fix(core): ...`, `test(pytest): ...`, `docs: ...`, `refactor(reasoning): ...`. Use imperative, scoped subjects and keep each commit focused.

Pull requests should summarize behavior changes, list lint/test commands run, link related issues, and include request/response samples or screenshots when CLI or API behavior changes.

## Security & Configuration Tips

Start from `.env.example` and keep secrets local. Do not commit `.env`, auth files, coverage artifacts, or badge tokens. Preserve the repository’s configuration precedence: CLI flags override `GPTMOCK_*`, which override legacy `CHATGPT_LOCAL_*` variables.
