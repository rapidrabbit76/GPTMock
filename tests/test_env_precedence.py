"""Unit tests: env var precedence helpers and argparse defaults.

These tests do NOT require a running server.
They verify that GPTMOCK_* canonical env vars take precedence over
CHATGPT_LOCAL_* legacy aliases in the CLI layer.
"""

from __future__ import annotations

import importlib
import sys
from typing import Generator
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers: re-import cli module with patched env
# ---------------------------------------------------------------------------


def _reload_cli(env: dict[str, str]):
    """Reload gptmock.cli with a clean env snapshot."""
    # Must clear cached module so module-level code re-executes
    for key in list(sys.modules):
        if key.startswith("gptmock"):
            del sys.modules[key]
    with patch.dict("os.environ", env, clear=True):
        return importlib.import_module("gptmock.cli")


# ---------------------------------------------------------------------------
# _env_with_legacy
# ---------------------------------------------------------------------------


class TestEnvWithLegacy:
    def test_canonical_wins_over_legacy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GPTMOCK_REASONING_EFFORT", "high")
        monkeypatch.setenv("CHATGPT_LOCAL_REASONING_EFFORT", "low")
        from gptmock.cli import _env_with_legacy

        assert (
            _env_with_legacy(
                "GPTMOCK_REASONING_EFFORT", "CHATGPT_LOCAL_REASONING_EFFORT"
            )
            == "high"
        )

    def test_legacy_used_when_canonical_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GPTMOCK_REASONING_EFFORT", raising=False)
        monkeypatch.setenv("CHATGPT_LOCAL_REASONING_EFFORT", "low")
        from gptmock.cli import _env_with_legacy

        assert (
            _env_with_legacy(
                "GPTMOCK_REASONING_EFFORT", "CHATGPT_LOCAL_REASONING_EFFORT"
            )
            == "low"
        )

    def test_default_returned_when_both_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GPTMOCK_REASONING_EFFORT", raising=False)
        monkeypatch.delenv("CHATGPT_LOCAL_REASONING_EFFORT", raising=False)
        from gptmock.cli import _env_with_legacy

        assert (
            _env_with_legacy(
                "GPTMOCK_REASONING_EFFORT", "CHATGPT_LOCAL_REASONING_EFFORT", "medium"
            )
            == "medium"
        )

    def test_empty_canonical_falls_through_to_legacy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GPTMOCK_REASONING_EFFORT", "")
        monkeypatch.setenv("CHATGPT_LOCAL_REASONING_EFFORT", "xhigh")
        from gptmock.cli import _env_with_legacy

        assert (
            _env_with_legacy(
                "GPTMOCK_REASONING_EFFORT", "CHATGPT_LOCAL_REASONING_EFFORT"
            )
            == "xhigh"
        )

    def test_no_legacy_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GPTMOCK_PORT", raising=False)
        from gptmock.cli import _env_with_legacy

        assert _env_with_legacy("GPTMOCK_PORT", default="8000") == "8000"


# ---------------------------------------------------------------------------
# _env_truthy
# ---------------------------------------------------------------------------


class TestEnvTruthy:
    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "YES", "on", "ON"])
    def test_truthy_values_canonical(
        self, monkeypatch: pytest.MonkeyPatch, val: str
    ) -> None:
        monkeypatch.setenv("GPTMOCK_DEFAULT_WEB_SEARCH", val)
        monkeypatch.delenv("CHATGPT_LOCAL_ENABLE_WEB_SEARCH", raising=False)
        from gptmock.cli import _env_truthy

        assert (
            _env_truthy("GPTMOCK_DEFAULT_WEB_SEARCH", "CHATGPT_LOCAL_ENABLE_WEB_SEARCH")
            is True
        )

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", ""])
    def test_falsy_values(self, monkeypatch: pytest.MonkeyPatch, val: str) -> None:
        monkeypatch.setenv("GPTMOCK_DEFAULT_WEB_SEARCH", val)
        monkeypatch.delenv("CHATGPT_LOCAL_ENABLE_WEB_SEARCH", raising=False)
        from gptmock.cli import _env_truthy

        assert (
            _env_truthy("GPTMOCK_DEFAULT_WEB_SEARCH", "CHATGPT_LOCAL_ENABLE_WEB_SEARCH")
            is False
        )

    def test_canonical_truthy_beats_legacy_falsy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GPTMOCK_DEFAULT_WEB_SEARCH", "true")
        monkeypatch.setenv("CHATGPT_LOCAL_ENABLE_WEB_SEARCH", "false")
        from gptmock.cli import _env_truthy

        assert (
            _env_truthy("GPTMOCK_DEFAULT_WEB_SEARCH", "CHATGPT_LOCAL_ENABLE_WEB_SEARCH")
            is True
        )

    def test_legacy_truthy_when_canonical_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GPTMOCK_DEFAULT_WEB_SEARCH", raising=False)
        monkeypatch.setenv("CHATGPT_LOCAL_ENABLE_WEB_SEARCH", "1")
        from gptmock.cli import _env_truthy

        assert (
            _env_truthy("GPTMOCK_DEFAULT_WEB_SEARCH", "CHATGPT_LOCAL_ENABLE_WEB_SEARCH")
            is True
        )


# ---------------------------------------------------------------------------
# _env_int
# ---------------------------------------------------------------------------


class TestEnvInt:
    def test_canonical_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GPTMOCK_PORT", "9000")
        from gptmock.cli import _env_int

        assert _env_int("GPTMOCK_PORT", 8000) == 9000

    def test_default_when_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GPTMOCK_PORT", raising=False)
        from gptmock.cli import _env_int

        assert _env_int("GPTMOCK_PORT", 8000) == 8000

    def test_invalid_value_returns_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GPTMOCK_PORT", "not-a-number")
        from gptmock.cli import _env_int

        assert _env_int("GPTMOCK_PORT", 8000) == 8000


# ---------------------------------------------------------------------------
# Reasoning env vars: GPTMOCK_* canonical precedence via argparse defaults
# ---------------------------------------------------------------------------


class TestArgparseEnvDefaults:
    """Verify that argparse defaults honour GPTMOCK_* over CHATGPT_LOCAL_*."""

    def _parse_serve(self, env: dict[str, str]) -> object:
        import argparse

        with patch.dict("os.environ", env, clear=True):
            # Re-import to force fresh evaluation of argparse defaults
            for key in list(sys.modules):
                if key.startswith("gptmock.cli"):
                    del sys.modules[key]
            from gptmock import cli

            parser = argparse.ArgumentParser()
            sub = parser.add_subparsers(dest="command")
            p = sub.add_parser("serve")
            p.add_argument(
                "--reasoning-effort",
                choices=["minimal", "low", "medium", "high", "xhigh"],
                default=(
                    cli._env_with_legacy(
                        "GPTMOCK_REASONING_EFFORT",
                        "CHATGPT_LOCAL_REASONING_EFFORT",
                        "medium",
                    )
                    or "medium"
                ).lower(),
            )
            p.add_argument(
                "--reasoning-summary",
                choices=["auto", "concise", "detailed", "none"],
                default=(
                    cli._env_with_legacy(
                        "GPTMOCK_REASONING_SUMMARY",
                        "CHATGPT_LOCAL_REASONING_SUMMARY",
                        "auto",
                    )
                    or "auto"
                ).lower(),
            )
            p.add_argument(
                "--reasoning-compat",
                choices=["legacy", "o3", "think-tags", "current"],
                default=(
                    cli._env_with_legacy(
                        "GPTMOCK_REASONING_COMPAT",
                        "CHATGPT_LOCAL_REASONING_COMPAT",
                        "think-tags",
                    )
                    or "think-tags"
                ).lower(),
            )
            p.add_argument(
                "--expose-reasoning-models",
                action="store_true",
                default=cli._env_truthy(
                    "GPTMOCK_EXPOSE_REASONING_MODELS",
                    "CHATGPT_LOCAL_EXPOSE_REASONING_MODELS",
                ),
            )
            p.add_argument(
                "--enable-web-search",
                action="store_true",
                default=cli._env_truthy(
                    "GPTMOCK_DEFAULT_WEB_SEARCH", "CHATGPT_LOCAL_ENABLE_WEB_SEARCH"
                ),
            )
            return parser.parse_args(["serve"])

    def test_gptmock_reasoning_effort_wins(self) -> None:
        args = self._parse_serve(
            {
                "GPTMOCK_REASONING_EFFORT": "high",
                "CHATGPT_LOCAL_REASONING_EFFORT": "low",
            }
        )
        assert args.reasoning_effort == "high"

    def test_legacy_reasoning_effort_fallback(self) -> None:
        args = self._parse_serve(
            {
                "CHATGPT_LOCAL_REASONING_EFFORT": "minimal",
            }
        )
        assert args.reasoning_effort == "minimal"

    def test_gptmock_reasoning_summary_wins(self) -> None:
        args = self._parse_serve(
            {
                "GPTMOCK_REASONING_SUMMARY": "detailed",
                "CHATGPT_LOCAL_REASONING_SUMMARY": "none",
            }
        )
        assert args.reasoning_summary == "detailed"

    def test_gptmock_reasoning_compat_wins(self) -> None:
        args = self._parse_serve(
            {
                "GPTMOCK_REASONING_COMPAT": "o3",
                "CHATGPT_LOCAL_REASONING_COMPAT": "legacy",
            }
        )
        assert args.reasoning_compat == "o3"

    def test_gptmock_web_search_wins(self) -> None:
        args = self._parse_serve(
            {
                "GPTMOCK_DEFAULT_WEB_SEARCH": "true",
                "CHATGPT_LOCAL_ENABLE_WEB_SEARCH": "false",
            }
        )
        assert args.enable_web_search is True

    def test_legacy_web_search_fallback(self) -> None:
        args = self._parse_serve(
            {
                "CHATGPT_LOCAL_ENABLE_WEB_SEARCH": "1",
            }
        )
        assert args.enable_web_search is True

    def test_defaults_when_no_env(self) -> None:
        args = self._parse_serve({})
        assert args.reasoning_effort == "medium"
        assert args.reasoning_summary == "auto"
        assert args.reasoning_compat == "think-tags"
        assert args.expose_reasoning_models is False
        assert args.enable_web_search is False
