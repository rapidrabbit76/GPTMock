from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gptmock.core.settings import Settings
from gptmock.routers.health import router as health_router
from gptmock.routers.ollama import router as ollama_router
from gptmock.routers.openai import router as openai_router


# Global httpx client (managed by lifespan)
_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """Get the global httpx client (must be called within lifespan context)."""
    if _http_client is None:
        raise RuntimeError("HTTP client not initialized. App lifespan not started.")
    return _http_client


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.
    
    Manages httpx.AsyncClient lifecycle for the entire application.
    """
    global _http_client
    
    # Startup: create httpx client
    _http_client = httpx.AsyncClient(timeout=300.0)
    
    try:
        yield
    finally:
        # Shutdown: close httpx client
        if _http_client is not None:
            await _http_client.aclose()
            _http_client = None


def create_app(settings: Settings | None = None) -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Args:
        settings: Optional Settings instance. If None, creates from environment.
    
    Returns:
        Configured FastAPI application.
    """
    if settings is None:
        from gptmock.core.dependencies import get_settings
        settings = get_settings()
    
    # Create FastAPI app with lifespan
    app = FastAPI(
        title="gptmock",
        description="OpenAI & Ollama compatible API powered by ChatGPT",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Register routers
    app.include_router(health_router)
    app.include_router(openai_router)
    app.include_router(ollama_router)
    
    return app
