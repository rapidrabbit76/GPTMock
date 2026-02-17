from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint - health check."""
    return JSONResponse({"status": "ok"})


@router.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse({"status": "ok"})
