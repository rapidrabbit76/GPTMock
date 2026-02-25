FROM python:3.13-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv


ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT="/app/.venv" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install dependencies first (cache-friendly layer)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY gptmock/ gptmock/
COPY prompt.md prompt_gpt5_codex.md ./

# Install the project itself
RUN uv sync --frozen --no-dev

EXPOSE 8000
ENTRYPOINT ["uv", "run" ,"--no-sync","gptmock"]
CMD ["serve"]
