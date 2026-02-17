from __future__ import annotations

import base64
import datetime
import hashlib
import json
import os
import secrets
import sys
from typing import Any, Dict, Optional, Tuple

import httpx

from gptmock.core.constants import CLIENT_ID_DEFAULT, OAUTH_TOKEN_URL
from gptmock.core.models import PkceCodes


def eprint(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, **kwargs)


def get_home_dir() -> str:
    home = os.getenv("GPTMOCK_HOME") or os.getenv("CHATGPT_LOCAL_HOME") or os.getenv("CODEX_HOME")
    if not home:
        home = os.path.expanduser("~/.config/gptmock")
    return home


def read_auth_file() -> Dict[str, Any] | None:
    for base in [
        os.getenv("GPTMOCK_HOME"),
        os.getenv("CHATGPT_LOCAL_HOME"),
        os.getenv("CODEX_HOME"),
        os.path.expanduser("~/.config/gptmock"),
        os.path.expanduser("~/.chatgpt-local"),
        os.path.expanduser("~/.codex"),
    ]:
        if not base:
            continue
        path = os.path.join(base, "auth.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return None


def write_auth_file(auth: Dict[str, Any]) -> bool:
    home = get_home_dir()
    try:
        os.makedirs(home, exist_ok=True)
    except Exception as exc:
        eprint(f"ERROR: unable to create auth home directory {home}: {exc}")
        return False
    path = os.path.join(home, "auth.json")
    try:
        with open(path, "w", encoding="utf-8") as fp:
            if hasattr(os, "fchmod"):
                os.fchmod(fp.fileno(), 0o600)
            json.dump(auth, fp, indent=2)
        return True
    except Exception as exc:
        eprint(f"ERROR: unable to write auth file: {exc}")
        return False


def parse_jwt_claims(token: str) -> Dict[str, Any] | None:
    if not token or token.count(".") != 2:
        return None
    try:
        _, payload, _ = token.split(".")
        padded = payload + "=" * (-len(payload) % 4)
        data = base64.urlsafe_b64decode(padded.encode())
        return json.loads(data.decode())
    except Exception:
        return None


def generate_pkce() -> PkceCodes:
    code_verifier = secrets.token_hex(64)
    digest = hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return PkceCodes(code_verifier=code_verifier, code_challenge=code_challenge)


async def load_chatgpt_tokens(ensure_fresh: bool = True) -> tuple[str | None, str | None, str | None]:
    auth = read_auth_file()
    if not isinstance(auth, dict):
        return None, None, None

    tokens = auth.get("tokens") if isinstance(auth.get("tokens"), dict) else {}
    access_token: Optional[str] = tokens.get("access_token")
    account_id: Optional[str] = tokens.get("account_id")
    id_token: Optional[str] = tokens.get("id_token")
    refresh_token: Optional[str] = tokens.get("refresh_token")
    last_refresh = auth.get("last_refresh")

    if ensure_fresh and isinstance(refresh_token, str) and refresh_token and CLIENT_ID_DEFAULT:
        needs_refresh = _should_refresh_access_token(access_token, last_refresh)
        if needs_refresh or not (isinstance(access_token, str) and access_token):
            refreshed = await _refresh_chatgpt_tokens(refresh_token, CLIENT_ID_DEFAULT)
            if refreshed:
                access_token = refreshed.get("access_token") or access_token
                id_token = refreshed.get("id_token") or id_token
                refresh_token = refreshed.get("refresh_token") or refresh_token
                account_id = refreshed.get("account_id") or account_id

                updated_tokens = dict(tokens)
                if isinstance(access_token, str) and access_token:
                    updated_tokens["access_token"] = access_token
                if isinstance(id_token, str) and id_token:
                    updated_tokens["id_token"] = id_token
                if isinstance(refresh_token, str) and refresh_token:
                    updated_tokens["refresh_token"] = refresh_token
                if isinstance(account_id, str) and account_id:
                    updated_tokens["account_id"] = account_id

                persisted = _persist_refreshed_auth(auth, updated_tokens)
                if persisted is not None:
                    auth, tokens = persisted
                else:
                    tokens = updated_tokens

    if not isinstance(account_id, str) or not account_id:
        account_id = _derive_account_id(id_token)

    access_token = access_token if isinstance(access_token, str) and access_token else None
    id_token = id_token if isinstance(id_token, str) and id_token else None
    account_id = account_id if isinstance(account_id, str) and account_id else None
    return access_token, account_id, id_token


def _should_refresh_access_token(access_token: Optional[str], last_refresh: Any) -> bool:
    if not isinstance(access_token, str) or not access_token:
        return True

    claims = parse_jwt_claims(access_token) or {}
    exp = claims.get("exp") if isinstance(claims, dict) else None
    now = datetime.datetime.now(datetime.timezone.utc)
    if isinstance(exp, (int, float)):
        try:
            expiry = datetime.datetime.fromtimestamp(float(exp), datetime.timezone.utc)
        except (OverflowError, OSError, ValueError):
            expiry = None
        if expiry is not None:
            return expiry <= now + datetime.timedelta(minutes=5)

    if isinstance(last_refresh, str):
        refreshed_at = _parse_iso8601(last_refresh)
        if refreshed_at is not None:
            return refreshed_at <= now - datetime.timedelta(minutes=55)
    return False


async def _refresh_chatgpt_tokens(refresh_token: str, client_id: str) -> Optional[Dict[str, Optional[str]]]:
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "scope": "openid profile email offline_access",
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(OAUTH_TOKEN_URL, json=payload, timeout=30)
    except httpx.RequestError as exc:
        eprint(f"ERROR: failed to refresh ChatGPT token: {exc}")
        return None

    if resp.status_code >= 400:
        eprint(f"ERROR: refresh token request returned status {resp.status_code}")
        return None

    try:
        data = resp.json()
    except ValueError as exc:
        eprint(f"ERROR: unable to parse refresh token response: {exc}")
        return None

    id_token = data.get("id_token")
    access_token = data.get("access_token")
    new_refresh_token = data.get("refresh_token") or refresh_token
    if not isinstance(id_token, str) or not isinstance(access_token, str):
        eprint("ERROR: refresh token response missing expected tokens")
        return None

    account_id = _derive_account_id(id_token)
    new_refresh_token = new_refresh_token if isinstance(new_refresh_token, str) and new_refresh_token else refresh_token
    return {
        "id_token": id_token,
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "account_id": account_id,
    }


def _persist_refreshed_auth(auth: Dict[str, Any], updated_tokens: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    updated_auth = dict(auth)
    updated_auth["tokens"] = updated_tokens
    updated_auth["last_refresh"] = _now_iso8601()
    if write_auth_file(updated_auth):
        return updated_auth, updated_tokens
    eprint("ERROR: unable to persist refreshed auth tokens")
    return None


def _derive_account_id(id_token: Optional[str]) -> Optional[str]:
    if not isinstance(id_token, str) or not id_token:
        return None
    claims = parse_jwt_claims(id_token) or {}
    auth_claims = claims.get("https://api.openai.com/auth") if isinstance(claims, dict) else None
    if isinstance(auth_claims, dict):
        account_id = auth_claims.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id:
            return account_id
    return None


def _parse_iso8601(value: str) -> Optional[datetime.datetime]:
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(datetime.timezone.utc)
    except Exception:
        return None


def _now_iso8601() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


async def get_effective_chatgpt_auth() -> tuple[str | None, str | None]:
    access_token, account_id, id_token = await load_chatgpt_tokens()
    if not account_id:
        account_id = _derive_account_id(id_token)
    return access_token, account_id
