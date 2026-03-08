from __future__ import annotations

import os
import shutil
from collections.abc import Generator
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from gptmock.app import create_app
from gptmock.core.badges import update_gist_badges
from gptmock.infra.auth import get_home_dir, read_auth_file
from gptmock.services.model_registry import get_model_list

TEST_PROMPT = "Say 'hello' and nothing else."
TIMEOUT = 120

_INTEGRATION_TEST_FILES = {
    "test_rest_api.py",
    "test_openai_client.py",
    "test_responses_api.py",
    "test_ollama_client.py",
    "test_langchain_client.py",
}


def _get_all_models() -> list[str]:
    return get_model_list(expose_reasoning=False)


ALL_MODELS: list[str] = _get_all_models()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-gist",
        action="store_true",
        default=False,
        help="Update gist badges after a full tests run",
    )


def _is_full_tests_run(config: pytest.Config) -> bool:
    args = [str(arg).rstrip("/") for arg in config.args]
    return args in ([], ["tests"])


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not session.config.getoption("update_gist"):
        return
    if not _is_full_tests_run(session.config):
        terminal = session.config.pluginmanager.get_plugin("terminalreporter")
        if terminal is not None:
            terminal.write_line("Skipping gist update for non-full test target")
        return

    terminal = session.config.pluginmanager.get_plugin("terminalreporter")
    stats = getattr(terminal, "stats", {}) if terminal is not None else {}
    passed = len(stats.get("passed", []))
    failed = len(stats.get("failed", [])) + len(stats.get("error", []))
    skipped = len(stats.get("skipped", []))
    collected = session.testscollected or passed + failed + skipped
    ran = collected - skipped
    tests_pct = round(passed / ran * 100) if ran > 0 else 0
    update_gist_badges(
        tests_label="tests",
        tests_pct=tests_pct,
        tests_collected=collected,
        tests_skipped=skipped,
    )


@pytest.fixture(scope="session", autouse=True)
def isolated_gptmock_home(tmp_path_factory: pytest.TempPathFactory) -> Generator[Path]:
    real_home = Path(get_home_dir())
    tmp_home = tmp_path_factory.mktemp("gptmock")

    real_auth = real_home / "auth.json"
    if real_auth.exists():
        shutil.copy2(real_auth, tmp_home / "auth.json")

    prev = os.environ.get("GPTMOCK_HOME")
    os.environ["GPTMOCK_HOME"] = str(tmp_home)

    yield tmp_home

    tmp_auth = tmp_home / "auth.json"
    if tmp_auth.exists():
        real_home.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp_auth, real_auth)

    if prev is None:
        os.environ.pop("GPTMOCK_HOME", None)
    else:
        os.environ["GPTMOCK_HOME"] = prev


@pytest.fixture(autouse=True)
def skip_without_auth(request: pytest.FixtureRequest) -> None:
    filename = os.path.basename(str(request.fspath))
    if filename in _INTEGRATION_TEST_FILES and read_auth_file() is None:
        pytest.skip("No ChatGPT auth — run `gptmock login` first")


@pytest.fixture(scope="session")
def client() -> Generator[TestClient]:
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture(scope="session")
def all_models() -> list[str]:
    return ALL_MODELS
