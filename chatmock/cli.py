from __future__ import annotations

import errno
import argparse
import json
import os
import sys
import webbrowser
from datetime import datetime

from chatmock.core.constants import CLIENT_ID_DEFAULT
from chatmock.infra.limits import RateLimitWindow, compute_reset_at, load_rate_limit_snapshot
from chatmock.infra.oauth import OAuthHTTPServer, OAuthHandler, REQUIRED_PORT, URL_BASE
from chatmock.infra.auth import eprint, get_home_dir, parse_jwt_claims, read_auth_file


_STATUS_LIMIT_BAR_SEGMENTS = 30
_STATUS_LIMIT_BAR_FILLED = "█"
_STATUS_LIMIT_BAR_EMPTY = "░"
_STATUS_LIMIT_BAR_PARTIAL = "▓"


def _clamp_percent(value: float) -> float:
    try:
        percent = float(value)
    except Exception:
        return 0.0
    if percent != percent:
        return 0.0
    if percent < 0.0:
        return 0.0
    if percent > 100.0:
        return 100.0
    return percent


def _render_progress_bar(percent_used: float) -> str:
    ratio = max(0.0, min(1.0, percent_used / 100.0))
    filled_exact = ratio * _STATUS_LIMIT_BAR_SEGMENTS
    filled = int(filled_exact)
    partial = filled_exact - filled
    
    has_partial = partial > 0.5
    if has_partial:
        filled += 1
    
    filled = max(0, min(_STATUS_LIMIT_BAR_SEGMENTS, filled))
    empty = _STATUS_LIMIT_BAR_SEGMENTS - filled
    
    if has_partial and filled > 0:
        bar = _STATUS_LIMIT_BAR_FILLED * (filled - 1) + _STATUS_LIMIT_BAR_PARTIAL + _STATUS_LIMIT_BAR_EMPTY * empty
    else:
        bar = _STATUS_LIMIT_BAR_FILLED * filled + _STATUS_LIMIT_BAR_EMPTY * empty
    
    return f"[{bar}]"


def _get_usage_color(percent_used: float) -> str:
    if percent_used >= 90:
        return "\033[91m" 
    elif percent_used >= 75:
        return "\033[93m"  
    elif percent_used >= 50:
        return "\033[94m"  
    else:
        return "\033[92m" 


def _reset_color() -> str:
    """ANSI reset color code"""
    return "\033[0m"


def _format_window_duration(minutes: int | None) -> str | None:
    if minutes is None:
        return None
    try:
        total = int(minutes)
    except Exception:
        return None
    if total <= 0:
        return None
    minutes = total
    weeks, remainder = divmod(minutes, 7 * 24 * 60)
    days, remainder = divmod(remainder, 24 * 60)
    hours, remainder = divmod(remainder, 60)
    parts = []
    if weeks:
        parts.append(f"{weeks} week" + ("s" if weeks != 1 else ""))
    if days:
        parts.append(f"{days} day" + ("s" if days != 1 else ""))
    if hours:
        parts.append(f"{hours} hour" + ("s" if hours != 1 else ""))
    if remainder:
        parts.append(f"{remainder} minute" + ("s" if remainder != 1 else ""))
    if not parts:
        parts.append(f"{minutes} minute" + ("s" if minutes != 1 else ""))
    return " ".join(parts)


def _format_reset_duration(seconds: int | None) -> str | None:
    if seconds is None:
        return None
    try:
        value = int(seconds)
    except Exception:
        return None
    if value < 0:
        value = 0
    days, remainder = divmod(value, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, remainder = divmod(remainder, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if not parts and remainder:
        parts.append("under 1m")
    if not parts:
        parts.append("0m")
    return " ".join(parts)


def _format_local_datetime(dt: datetime) -> str:
    local = dt.astimezone()
    tz_name = local.tzname() or "local"
    return f"{local.strftime('%b %d, %Y %H:%M')} {tz_name}"


def _print_usage_limits_block() -> None:
    stored = load_rate_limit_snapshot()
    
    print("📊 Usage Limits")
    
    if stored is None:
        print("  No usage data available yet. Send a request through ChatMock first.")
        print()
        return

    update_time = _format_local_datetime(stored.captured_at)
    print(f"Last updated: {update_time}")
    print()

    windows: list[tuple[str, str, RateLimitWindow]] = []
    if stored.snapshot.primary is not None:
        windows.append(("⚡", "5 hour limit", stored.snapshot.primary))
    if stored.snapshot.secondary is not None:
        windows.append(("📅", "Weekly limit", stored.snapshot.secondary))

    if not windows:
        print("  Usage data was captured but no limit windows were provided.")
        print()
        return

    for i, (icon_label, desc, window) in enumerate(windows):
        if i > 0:
            print()
        
        percent_used = _clamp_percent(window.used_percent)
        remaining = max(0.0, 100.0 - percent_used)
        color = _get_usage_color(percent_used)
        reset = _reset_color()
        
        progress = _render_progress_bar(percent_used)
        usage_text = f"{percent_used:5.1f}% used"
        remaining_text = f"{remaining:5.1f}% left"
        
        print(f"{icon_label} {desc}")
        print(f"{color}{progress}{reset} {color}{usage_text}{reset} | {remaining_text}")
        
        reset_in = _format_reset_duration(window.resets_in_seconds)
        reset_at = compute_reset_at(stored.captured_at, window)
        
        if reset_in and reset_at:
            reset_at_str = _format_local_datetime(reset_at)
            print(f"    ⏳ Resets in: {reset_in} at {reset_at_str}")
        elif reset_in:
            print(f"    ⏳ Resets in: {reset_in}")
        elif reset_at:
            reset_at_str = _format_local_datetime(reset_at)
            print(f"    ⏳ Resets at: {reset_at_str}")

    print()

def _format_token_expiry(exp_timestamp: int | float | None) -> tuple[str, bool]:
    if exp_timestamp is None:
        return "unknown", False
    try:
        expiry = datetime.fromtimestamp(float(exp_timestamp), tz=_UTC)
        now = datetime.now(tz=_UTC)
        expired = expiry <= now

        if expired:
            delta = now - expiry
        else:
            delta = expiry - now

        days = delta.days
        hours, rem = divmod(delta.seconds, 3600)
        minutes = rem // 60

        parts: list[str] = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes and not days:
            parts.append(f"{minutes}m")
        time_str = " ".join(parts) if parts else "< 1m"

        local_expiry = expiry.astimezone()
        tz_name = local_expiry.tzname() or "local"
        date_str = local_expiry.strftime("%Y-%m-%d %H:%M")

        if expired:
            return f"{date_str} {tz_name} ({time_str} ago) \033[91m[EXPIRED]\033[0m", True
        else:
            return f"{date_str} {tz_name} ({time_str} left)", False
    except Exception:
        return "unknown", False


_UTC = __import__("datetime").timezone.utc


def cmd_info(auth: dict | None) -> int:
    if not isinstance(auth, dict):
        print("  Not signed in")
        print(f"  Run: uv run python chatmock.py login")
        print()
        _print_usage_limits_block()
        return 0

    tokens = auth.get("tokens") if isinstance(auth.get("tokens"), dict) else {}
    access_token = tokens.get("access_token") if isinstance(tokens, dict) else None
    id_token = tokens.get("id_token") if isinstance(tokens, dict) else None
    account_id = tokens.get("account_id") if isinstance(tokens, dict) else None
    last_refresh = auth.get("last_refresh")

    if not access_token and not id_token:
        print("  Not signed in")
        print(f"  Run: uv run python chatmock.py login")
        print()
        _print_usage_limits_block()
        return 0

    id_claims = parse_jwt_claims(id_token) if id_token else {}
    access_claims = parse_jwt_claims(access_token) if access_token else {}
    id_claims = id_claims or {}
    access_claims = access_claims or {}

    auth_data = id_claims.get("https://api.openai.com/auth") or {}
    access_auth_data = access_claims.get("https://api.openai.com/auth") or {}

    email = id_claims.get("email") or id_claims.get("preferred_username") or "<unknown>"
    auth_provider = id_claims.get("auth_provider", "unknown")

    plan_raw = auth_data.get("chatgpt_plan_type") or access_auth_data.get("chatgpt_plan_type") or "unknown"
    plan_map = {"plus": "Plus", "pro": "Pro", "free": "Free", "team": "Team", "enterprise": "Enterprise"}
    plan = plan_map.get(str(plan_raw).lower(), str(plan_raw).title() if isinstance(plan_raw, str) else "Unknown")

    if not account_id:
        account_id = auth_data.get("chatgpt_account_id") or access_auth_data.get("chatgpt_account_id")
    user_id = auth_data.get("chatgpt_user_id") or access_auth_data.get("chatgpt_user_id")

    sub_start = auth_data.get("chatgpt_subscription_active_start")
    sub_until = auth_data.get("chatgpt_subscription_active_until")

    access_exp = access_claims.get("exp")
    id_exp = id_claims.get("exp")

    print("Account")
    print(f"  Email     : {email}")
    print(f"  Provider  : {auth_provider}")
    print(f"  Plan      : {plan}")
    if account_id:
        print(f"  Account   : {account_id}")
    if user_id:
        print(f"  User      : {user_id}")
    print()

    if sub_start or sub_until:
        print("Subscription")
        if sub_start:
            try:
                start_dt = datetime.fromisoformat(sub_start).astimezone()
                print(f"  Start     : {start_dt.strftime('%Y-%m-%d')}")
            except Exception:
                print(f"  Start     : {sub_start}")
        if sub_until:
            try:
                until_dt = datetime.fromisoformat(sub_until).astimezone()
                now = datetime.now(tz=_UTC)
                active = until_dt > now
                days_left = (until_dt - now).days if active else 0
                status = f"\033[92mActive\033[0m ({days_left}d left)" if active else "\033[91mExpired\033[0m"
                print(f"  Until     : {until_dt.strftime('%Y-%m-%d')}  {status}")
            except Exception:
                print(f"  Until     : {sub_until}")
        print()

    print("Tokens")
    access_exp_str, access_expired = _format_token_expiry(access_exp)
    id_exp_str, id_expired = _format_token_expiry(id_exp)
    print(f"  Access    : {access_exp_str}")
    print(f"  ID        : {id_exp_str}")
    if last_refresh:
        try:
            lr_dt = datetime.fromisoformat(last_refresh.replace("Z", "+00:00")).astimezone()
            tz_name = lr_dt.tzname() or "local"
            print(f"  Refreshed : {lr_dt.strftime('%Y-%m-%d %H:%M')} {tz_name}")
        except Exception:
            print(f"  Refreshed : {last_refresh}")

    if access_expired:
        print()
        print(f"\033[93m  Access token expired. It will auto-refresh on next server request.\033[0m")
        print(f"\033[93m  Or re-login: uv run python chatmock.py login\033[0m")
    print()

    print(f"Storage")
    print(f"  Path      : {get_home_dir()}/auth.json")
    print()

    _print_usage_limits_block()
    return 0


def cmd_login(no_browser: bool, verbose: bool) -> int:
    home_dir = get_home_dir()
    client_id = CLIENT_ID_DEFAULT
    if not client_id:
        eprint("ERROR: No OAuth client id configured. Set CHATGPT_LOCAL_CLIENT_ID.")
        return 1

    try:
        bind_host = os.getenv("CHATGPT_LOCAL_LOGIN_BIND", "127.0.0.1")
        httpd = OAuthHTTPServer((bind_host, REQUIRED_PORT), OAuthHandler, home_dir=home_dir, client_id=client_id, verbose=verbose)
    except OSError as e:
        eprint(f"ERROR: {e}")
        if e.errno == errno.EADDRINUSE:
            return 13
        return 1

    auth_url = httpd.auth_url()
    with httpd:
        eprint(f"Starting local login server on {URL_BASE}")
        if not no_browser:
            try:
                webbrowser.open(auth_url, new=1, autoraise=True)
            except Exception as e:
                eprint(f"Failed to open browser: {e}")
        eprint(f"If your browser did not open, navigate to:\n{auth_url}")

        def _stdin_paste_worker() -> None:
            try:
                eprint(
                    "If the browser can't reach this machine, paste the full redirect URL here and press Enter (or leave blank to keep waiting):"
                )
                line = sys.stdin.readline().strip()
                if not line:
                    return
                try:
                    from urllib.parse import urlparse, parse_qs

                    parsed = urlparse(line)
                    params = parse_qs(parsed.query)
                    code = (params.get("code") or [None])[0]
                    state = (params.get("state") or [None])[0]
                    if not code:
                        eprint("Input did not contain an auth code. Ignoring.")
                        return
                    if state and state != httpd.state:
                        eprint("State mismatch. Ignoring pasted URL for safety.")
                        return
                    eprint("Received redirect URL. Completing login without callback…")
                    bundle, _ = httpd.exchange_code(code)
                    if httpd.persist_auth(bundle):
                        httpd.exit_code = 0
                        eprint("Login successful. Tokens saved.")
                    else:
                        eprint("ERROR: Unable to persist auth file.")
                    httpd.shutdown()
                except Exception as exc:
                    eprint(f"Failed to process pasted redirect URL: {exc}")
            except Exception:
                pass

        try:
            import threading

            threading.Thread(target=_stdin_paste_worker, daemon=True).start()
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            eprint("\nKeyboard interrupt received, exiting.")
        return httpd.exit_code


def cmd_serve(
    host: str,
    port: int,
    verbose: bool,
    verbose_obfuscation: bool,
    reasoning_effort: str,
    reasoning_summary: str,
    reasoning_compat: str,
    debug_model: str | None,
    expose_reasoning_models: bool,
    default_web_search: bool,
) -> int:
    auth = read_auth_file()
    if not isinstance(auth, dict) or not auth.get("tokens"):
        eprint("No credentials found. Starting login flow...")
        login_result = cmd_login(no_browser=False, verbose=verbose)
        if login_result != 0:
            eprint("Login failed. Cannot start server without credentials.")
            return login_result
        eprint("Login successful. Starting server...\n")

    if verbose:
        os.environ["CHATMOCK_VERBOSE"] = "true"
    if verbose_obfuscation:
        os.environ["CHATMOCK_VERBOSE_OBFUSCATION"] = "true"
    if reasoning_effort:
        os.environ["CHATMOCK_REASONING_EFFORT"] = reasoning_effort
    if reasoning_summary:
        os.environ["CHATMOCK_REASONING_SUMMARY"] = reasoning_summary
    if reasoning_compat:
        os.environ["CHATMOCK_REASONING_COMPAT"] = reasoning_compat
    if debug_model:
        os.environ["CHATMOCK_DEBUG_MODEL"] = debug_model
    if expose_reasoning_models:
        os.environ["CHATMOCK_EXPOSE_REASONING_MODELS"] = "true"
    if default_web_search:
        os.environ["CHATMOCK_DEFAULT_WEB_SEARCH"] = "true"
    
    os.environ["CHATMOCK_HOST"] = host
    os.environ["CHATMOCK_PORT"] = str(port)
    
    import uvicorn
    uvicorn.run(
        "chatmock.app:create_app",
        factory=True,
        host=host,
        port=port,
        log_level="info" if verbose else "warning",
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="ChatGPT Local: login & OpenAI-compatible proxy")
    sub = parser.add_subparsers(dest="command", required=True)

    p_login = sub.add_parser("login", help="Authorize with ChatGPT and store tokens")
    p_login.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically")
    p_login.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    p_serve = sub.add_parser("serve", help="Run local OpenAI-compatible server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    p_serve.add_argument(
        "--verbose-obfuscation",
        action="store_true",
        help="Also dump raw SSE/obfuscation events (in addition to --verbose request/response logs).",
    )
    p_serve.add_argument(
        "--debug-model",
        dest="debug_model",
        default=os.getenv("CHATGPT_LOCAL_DEBUG_MODEL"),
        help="Forcibly override requested 'model' with this value",
    )
    p_serve.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high", "xhigh"],
        default=os.getenv("CHATGPT_LOCAL_REASONING_EFFORT", "medium").lower(),
        help="Reasoning effort level for Responses API (default: medium)",
    )
    p_serve.add_argument(
        "--reasoning-summary",
        choices=["auto", "concise", "detailed", "none"],
        default=os.getenv("CHATGPT_LOCAL_REASONING_SUMMARY", "auto").lower(),
        help="Reasoning summary verbosity (default: auto)",
    )
    p_serve.add_argument(
        "--reasoning-compat",
        choices=["legacy", "o3", "think-tags", "current"],
        default=os.getenv("CHATGPT_LOCAL_REASONING_COMPAT", "think-tags").lower(),
        help=(
            "Compatibility mode for exposing reasoning to clients (legacy|o3|think-tags). "
            "'current' is accepted as an alias for 'legacy'"
        ),
    )
    p_serve.add_argument(
        "--expose-reasoning-models",
        action="store_true",
        default=(os.getenv("CHATGPT_LOCAL_EXPOSE_REASONING_MODELS") or "").strip().lower() in ("1", "true", "yes", "on"),
        help=(
            "Expose GPT-5 family reasoning effort variants (minimal|low|medium|high|xhigh where supported) "
            "as separate models from /v1/models. This allows choosing effort via model selection in compatible UIs."
        ),
    )
    p_serve.add_argument(
        "--enable-web-search",
        action=argparse.BooleanOptionalAction,
        default=(os.getenv("CHATGPT_LOCAL_ENABLE_WEB_SEARCH") or "").strip().lower() in ("1", "true", "yes", "on"),
        help=(
            "Enable default web_search tool when a request omits responses_tools (off by default). "
            "Also configurable via CHATGPT_LOCAL_ENABLE_WEB_SEARCH."
        ),
    )

    p_info = sub.add_parser("info", help="Print current stored tokens and derived account id")
    p_info.add_argument("--json", action="store_true", help="Output raw auth.json contents")

    args = parser.parse_args()

    if args.command == "login":
        sys.exit(cmd_login(no_browser=args.no_browser, verbose=args.verbose))
    elif args.command == "serve":
        sys.exit(
            cmd_serve(
                host=args.host,
                port=args.port,
                verbose=args.verbose,
                verbose_obfuscation=args.verbose_obfuscation,
                reasoning_effort=args.reasoning_effort,
                reasoning_summary=args.reasoning_summary,
                reasoning_compat=args.reasoning_compat,
                debug_model=args.debug_model,
                expose_reasoning_models=args.expose_reasoning_models,
                default_web_search=args.enable_web_search,
            )
        )
    elif args.command == "info":
        auth = read_auth_file()
        if getattr(args, "json", False):
            print(json.dumps(auth or {}, indent=2))
            sys.exit(0)
        sys.exit(cmd_info(auth))
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
