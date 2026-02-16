from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenData:
    id_token: str
    access_token: str
    refresh_token: str
    account_id: str


@dataclass
class AuthBundle:
    api_key: Optional[str]
    token_data: TokenData
    last_refresh: str


@dataclass
class PkceCodes:
    code_verifier: str
    code_challenge: str

