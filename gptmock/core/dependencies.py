from __future__ import annotations

from functools import lru_cache

from .settings import Settings


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_http_client():
    from gptmock.app import get_http_client as _get_http_client
    return _get_http_client()
