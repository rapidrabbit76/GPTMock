from __future__ import annotations

import hmac
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from gptmock.core.dependencies import get_settings
from gptmock.core.settings import Settings
from gptmock.routers.health import router as health_router
from gptmock.routers.ollama import router as ollama_router
from gptmock.routers.openai import router as openai_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Manage a single httpx.AsyncClient for the application lifetime."""
    app.state.http_client = httpx.AsyncClient(timeout=300.0)
    try:
        yield
    finally:
        await app.state.http_client.aclose()

def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        settings: Optional Settings instance. If None, creates from environment.

    Returns:
        Configured FastAPI application.
    """
    if settings is None:
        settings = get_settings()

    package_logger = logging.getLogger("gptmock")
    package_logger.setLevel(
        logging.DEBUG if settings.verbose else logging.WARNING,
    )
    if settings.verbose and not any(
        getattr(handler, "_gptmock_verbose_handler", False)
        for handler in package_logger.handlers
    ):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        handler._gptmock_verbose_handler = True  # type: ignore[attr-defined]
        package_logger.addHandler(handler)

    # Create FastAPI app with lifespan
    app = FastAPI(
        title="gptmock",
        description="OpenAI & Ollama compatible API powered by ChatGPT",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.dependency_overrides[get_settings] = lambda: settings

    # Browser access is disabled by default. API clients do not require CORS.
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials="*" not in origins,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )

    @app.middleware("http")
    async def authenticate_proxy_requests(request: Request, call_next):
        """Require the configured proxy bearer token for model API routes."""
        api_key = settings.api_key
        protected = (
            request.method != "OPTIONS"
            and request.url.path.startswith(("/v1/", "/api/"))
        )
        if protected and isinstance(api_key, str) and api_key:
            scheme, _, credential = request.headers.get("Authorization", "").partition(" ")
            valid = scheme.lower() == "bearer" and hmac.compare_digest(credential, api_key)
            if not valid:
                return JSONResponse(
                    {"error": {"message": "Invalid or missing GPTMock API key"}},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
        return await call_next(request)

    # Register routers
    app.include_router(health_router)
    app.include_router(openai_router)
    app.include_router(ollama_router)

    return app
