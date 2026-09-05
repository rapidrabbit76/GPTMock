from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint - health check."""
    return JSONResponse({"status": "ok"})


@router.head("/")
async def root_head():
    """Support the Ollama CLI server-availability probe."""
    return Response(status_code=200, media_type="text/plain")


@router.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse({"status": "ok"})
