#!/usr/bin/env python3
"""
Model matrix runner for GPTMock tool-calling tests.

Sends 3 tool-call scenarios to every model and reports results with timing.

Usage:
  python3 tools/tool_tests/run_tool_tests_by_model.py
  python3 tools/tool_tests/run_tool_tests_by_model.py --models gpt-5,gpt-5.3-codex-spark --scenarios web_search
  python3 tools/tool_tests/run_tool_tests_by_model.py --scenarios all --print-payload --out results.json
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Callable, Dict, List

import requests


URL = "http://127.0.0.1:8000/v1/chat/completions"
TIMEOUT_SECONDS_DEFAULT = 90

# All base models from model_registry.py get_model_list()
ALL_MODELS = [
    "gpt-5",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5-codex",
    "gpt-5.1-codex",
    "gpt-5.1-codex-max",
    "gpt-5.2-codex",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
]


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------


def build_case_web_search(model: str) -> Dict:
    """responses_tools 경로: web_search (Responses API native tool)."""
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Find the latest news about GPTMock and summarize.",
            }
        ],
        "responses_tools": [{"type": "web_search"}],
        "stream": False,
    }


def build_case_calculator(model: str) -> Dict:
    """tools[] 경로: function tool (OpenAI Chat Completions 호환)."""
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Calculate 37 * 42 and explain the steps briefly.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Performs basic arithmetic",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Arithmetic expression",
                            }
                        },
                        "required": ["expression"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "stream": False,
    }


def build_case_parallel(model: str) -> Dict:
    """parallel_tool_calls=True 로 여러 function tool 동시 호출 유도."""
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Gather the latest AI headlines and compute a sentiment "
                    "score (0-100) for one headline."
                ),
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "sentiment",
                    "description": "Return sentiment score for given text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to score"},
                        },
                        "required": ["text"],
                    },
                },
            },
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "stream": False,
    }


TEST_CASES: Dict[str, Callable[[str], Dict]] = {
    "web_search": build_case_web_search,
    "calculator": build_case_calculator,
    "parallel": build_case_parallel,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run tool-calling test matrix by model")
    p.add_argument(
        "--models",
        default=",".join(ALL_MODELS),
        help="Comma-separated model list (default: all registered models)",
    )
    p.add_argument(
        "--scenarios",
        default="all",
        help="Comma-separated scenarios or 'all' (web_search,calculator,parallel)",
    )
    p.add_argument("--url", default=URL, help="GPTMock endpoint")
    p.add_argument(
        "--timeout",
        type=int,
        default=TIMEOUT_SECONDS_DEFAULT,
        help="Request timeout (seconds)",
    )
    p.add_argument(
        "--print-payload", action="store_true", help="Print payload before each request"
    )
    p.add_argument(
        "--print-body", action="store_true", help="Print full response body on success"
    )
    p.add_argument("--out", metavar="FILE", help="Save results to JSON file")
    return p.parse_args()


def _csv(value: str) -> List[str]:
    if not value or not isinstance(value, str):
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


# ---------------------------------------------------------------------------
# Request / summarize
# ---------------------------------------------------------------------------


def send(payload: Dict, url: str, timeout: int) -> Dict:
    t0 = time.monotonic()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as e:
        elapsed = time.monotonic() - t0
        return {
            "ok": False,
            "status": None,
            "error": str(e),
            "elapsed_s": round(elapsed, 2),
        }

    elapsed = time.monotonic() - t0
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text[:500]}

    return {
        "ok": resp.ok,
        "status": resp.status_code,
        "body": body,
        "elapsed_s": round(elapsed, 2),
    }


def summarize(body: Dict) -> str:
    """한 줄 요약: tool_calls 개수, finish_reason, 에러 메시지."""
    if not isinstance(body, dict):
        return "(non-dict body)"

    # 에러 응답
    err = body.get("error")
    if isinstance(err, dict):
        msg = err.get("message") or err.get("code") or str(err)
        return f"ERROR: {msg}"
    if isinstance(err, str):
        return f"ERROR: {err}"

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return "NO choices"

    first = choices[0]
    if not isinstance(first, dict):
        return "MALFORMED choice"

    message = first.get("message") or {}
    finish = first.get("finish_reason", "?")
    tc = message.get("tool_calls") if isinstance(message, dict) else None

    if isinstance(tc, list) and tc:
        names = [
            (t.get("function") or {}).get("name", "?")
            for t in tc
            if isinstance(t, dict)
        ]
        return f"tool_calls={len(tc)} [{', '.join(names)}] finish={finish}"

    if finish == "tool_calls":
        return "finish_reason=tool_calls but tool_calls field missing/empty"

    content = (message.get("content") or "")[:80] if isinstance(message, dict) else ""
    return f"finish={finish}, no tool_calls, content={content!r}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    models = _csv(args.models)
    scenarios_raw = _csv(args.scenarios)

    if not models:
        print("No models specified. Use --models gpt-5,gpt-5.1,...")
        raise SystemExit(2)

    scenario_names = (
        list(TEST_CASES.keys())
        if "all" in scenarios_raw or not scenarios_raw
        else scenarios_raw
    )
    unknown = [s for s in scenario_names if s not in TEST_CASES]
    if unknown:
        print(f"Unknown scenario(s): {', '.join(unknown)}")
        print(f"Available: {', '.join(sorted(TEST_CASES.keys()))}")
        raise SystemExit(2)

    print(f"URL:       {args.url}")
    print(f"Models:    {', '.join(models)}")
    print(f"Scenarios: {', '.join(scenario_names)}")
    print(f"Timeout:   {args.timeout}s")
    print()

    results: List[Dict] = []
    total = 0
    failed = 0

    for model in models:
        print(f"=== MODEL: {model} ===")
        for scenario in scenario_names:
            payload = TEST_CASES[scenario](model)
            if args.print_payload:
                print(json.dumps(payload, indent=2, ensure_ascii=False))

            result = send(payload, args.url, args.timeout)
            total += 1

            row = {
                "model": model,
                "scenario": scenario,
                "ok": result["ok"],
                "status": result.get("status"),
                "elapsed_s": result.get("elapsed_s"),
            }

            if not result["ok"]:
                failed += 1
                err_msg = result.get("error", "")
                err_body = result.get("body")
                if isinstance(err_body, dict):
                    upstream_msg = err_body.get("error") or {}
                    if isinstance(upstream_msg, dict):
                        upstream_msg = upstream_msg.get("message", "")
                    err_msg = (
                        f"{err_msg} | upstream: {upstream_msg}"
                        if upstream_msg
                        else err_msg
                    )
                row["summary"] = f"FAIL: {err_msg}"
                print(
                    f"  ✗ {scenario:12s}  status={result.get('status')}  {result.get('elapsed_s')}s  {err_msg}"
                )
            else:
                body = result.get("body", {})
                s = summarize(body)
                row["summary"] = s
                print(
                    f"  ✓ {scenario:12s}  status={result.get('status')}  {result.get('elapsed_s')}s  {s}"
                )
                if args.print_body:
                    print(json.dumps(body, indent=2, ensure_ascii=False))

            results.append(row)
        print()

    # Summary table
    print("=" * 60)
    print(f"Total: {total}  Passed: {total - failed}  Failed: {failed}")
    print("=" * 60)

    if args.out:
        out_data = {
            "url": args.url,
            "models": models,
            "scenarios": scenario_names,
            "total": total,
            "failed": failed,
            "results": results,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
