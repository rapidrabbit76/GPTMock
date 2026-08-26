FROM python:3.13.15-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.12.6 /uv /usr/local/bin/uv


ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT="/app/.venv" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN groupadd --system --gid 10001 gptmock \
    && useradd --system --uid 10001 --gid 10001 --create-home gptmock \
    && mkdir -p /data \
    && chown gptmock:gptmock /data

# Install dependencies first (cache-friendly layer)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY gptmock/ gptmock/
COPY prompt.md prompt_gpt5_codex.md ./

# Install the project itself
RUN uv sync --frozen --no-dev

EXPOSE 8000
USER gptmock:gptmock
ENTRYPOINT ["/app/.venv/bin/gptmock"]
CMD ["serve"]
