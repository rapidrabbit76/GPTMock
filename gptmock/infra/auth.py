from __future__ import annotations

import asyncio
import base64
import datetime
import hashlib
import json
import os
import secrets
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from gptmock.core.constants import CLIENT_ID_DEFAULT, OAUTH_TOKEN_URL
from gptmock.core.models import PkceCodes
from gptmock.core.utils import parse_datetime


def eprint(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, **kwargs)


def get_home_dir() -> str:
    home = os.getenv("GPTMOCK_HOME") or os.getenv("CHATGPT_LOCAL_HOME")
    if not home:
        home = os.path.expanduser("~/.config/gptmock")
    return home


def validate_auth_storage() -> None:
    """Fail before login/serve when a migrated bind mount is inaccessible."""
    home = get_home_dir()
    os.makedirs(home, exist_ok=True)
    path = os.path.join(home, "auth.json")
    if os.path.exists(path):
        with open(path, "rb"):
            pass
    with tempfile.TemporaryFile(dir=home):
        pass


def read_auth_file() -> dict[str, Any] | None:
    configured_home = os.getenv("GPTMOCK_HOME") or os.getenv("CHATGPT_LOCAL_HOME")
    candidates = (
        [configured_home]
        if configured_home
        else [
            os.path.expanduser("~/.config/gptmock"),
            os.path.expanduser("~/.chatgpt-local"),
        ]
    )
    for base in candidates:
        if not base:
            continue
        path = os.path.join(base, "auth.json")
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return None


def write_auth_file(auth: dict[str, Any]) -> bool:
    home = get_home_dir()
    try:
        os.makedirs(home, exist_ok=True)
    except Exception as exc:
        eprint(f"ERROR: unable to create auth home directory {home}: {exc}")
        return False
    path = os.path.join(home, "auth.json")
    temp_path = os.path.join(home, f".auth.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    try:
        with open(temp_path, "x", encoding="utf-8") as fp:
            if hasattr(os, "fchmod"):
                os.fchmod(fp.fileno(), 0o600)
            json.dump(auth, fp, indent=2)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(temp_path, path)
        return True
    except Exception as exc:
        eprint(f"ERROR: unable to write auth file: {exc}")
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        except OSError:
            pass
        return False


def parse_jwt_claims(token: str) -> dict[str, Any] | None:
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


@asynccontextmanager
async def _refresh_lock() -> AsyncIterator[None]:
    """Serialize refreshes across event loops and processes sharing an auth home."""
    os.makedirs(get_home_dir(), exist_ok=True)
    with open(os.path.join(get_home_dir(), ".refresh.lock"), "a+b") as lock_file:
        if os.name == "nt":
            import msvcrt
            if lock_file.seek(0, os.SEEK_END) == 0:
                lock_file.write(b"\0")
                lock_file.flush()
        else:
            import fcntl
        acquired = False
        deadline = time.monotonic() + 65
        try:
            while not acquired:
                try:
                    if os.name == "nt":
                        lock_file.seek(0)
                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except (BlockingIOError, PermissionError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Timed out waiting for credential refresh") from None
                    await asyncio.sleep(0.05)
            yield
        finally:
            if acquired:
                if os.name == "nt":
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


async def load_chatgpt_tokens(ensure_fresh: bool = True) -> tuple[str | None, str | None, str | None]:
    auth = await asyncio.to_thread(read_auth_file)
    tokens = auth.get("tokens", {}) if isinstance(auth, dict) else {}
    if (ensure_fresh and isinstance(tokens, dict) and tokens.get("refresh_token")
            and _should_refresh_access_token(tokens.get("access_token"), auth.get("last_refresh"))):
        async with _refresh_lock():
            # A previous waiter may have rotated the token. Always re-read under the lock.
            return await _load_chatgpt_tokens(ensure_fresh=True)
    return await _load_chatgpt_tokens(ensure_fresh=False)


async def _load_chatgpt_tokens(ensure_fresh: bool = True) -> tuple[str | None, str | None, str | None]:
    auth = await asyncio.to_thread(read_auth_file)
    if not isinstance(auth, dict):
        return None, None, None

    tokens = auth.get("tokens") if isinstance(auth.get("tokens"), dict) else {}
    access_token: str | None = tokens.get("access_token")
    account_id: str | None = tokens.get("account_id")
    id_token: str | None = tokens.get("id_token")
    refresh_token: str | None = tokens.get("refresh_token")
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

                persisted = await asyncio.to_thread(_persist_refreshed_auth, auth, updated_tokens)
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


def _should_refresh_access_token(access_token: str | None, last_refresh: Any) -> bool:
    if not isinstance(access_token, str) or not access_token:
        return True

    claims = parse_jwt_claims(access_token) or {}
    exp = claims.get("exp") if isinstance(claims, dict) else None
    now = datetime.datetime.now(datetime.UTC)
    if isinstance(exp, (int, float)):
        try:
            expiry = datetime.datetime.fromtimestamp(float(exp), datetime.UTC)
        except (OverflowError, OSError, ValueError):
            expiry = None
        if expiry is not None:
            return expiry <= now + datetime.timedelta(minutes=5)

    if isinstance(last_refresh, str):
        refreshed_at = parse_datetime(last_refresh)
        if refreshed_at is not None:
            return refreshed_at <= now - datetime.timedelta(minutes=55)
    return False


async def _refresh_chatgpt_tokens(refresh_token: str, client_id: str) -> dict[str, str | None] | None:
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


def _persist_refreshed_auth(auth: dict[str, Any], updated_tokens: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    updated_auth = dict(auth)
    updated_auth["tokens"] = updated_tokens
    updated_auth["last_refresh"] = _now_iso8601()
    if write_auth_file(updated_auth):
        return updated_auth, updated_tokens
    eprint("ERROR: unable to persist refreshed auth tokens")
    return None


def _derive_account_id(id_token: str | None) -> str | None:
    if not isinstance(id_token, str) or not id_token:
        return None
    claims = parse_jwt_claims(id_token) or {}
    auth_claims = claims.get("https://api.openai.com/auth") if isinstance(claims, dict) else None
    if isinstance(auth_claims, dict):
        account_id = auth_claims.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id:
            return account_id
    return None




def _now_iso8601() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")


async def get_effective_chatgpt_auth() -> tuple[str | None, str | None]:
    access_token, account_id, id_token = await load_chatgpt_tokens()
    if not account_id:
        account_id = _derive_account_id(id_token)
    return access_token, account_id
