"""Dependency-light container checks; mount this file read-only into the image."""
from __future__ import annotations

import asyncio
import json
import os
import stat
from pathlib import Path

from starlette.testclient import TestClient

from gptmock.app import create_app
from gptmock.core.settings import Settings
from gptmock.infra.auth import _refresh_lock, get_home_dir, validate_auth_storage, write_auth_file


def main() -> None:
    print(json.dumps({"pid": os.getpid(), "ppid": os.getppid(), "cwd": os.getcwd(), "command": "python /checks/docker_smoke.py", "ports": []}), flush=True)
    assert os.getuid() == 10001
    validate_auth_storage()
    assert write_auth_file({"tokens": {"access_token": "synthetic-container-token"}})
    assert stat.S_IMODE((Path(get_home_dir()) / "auth.json").stat().st_mode) == 0o600
    try:
        Path("/app/write-probe").touch()
    except OSError:
        pass
    else:
        raise AssertionError("Root filesystem must not be writable")

    async def locking() -> None:
        active = 0
        peak = 0
        async def worker() -> None:
            nonlocal active, peak
            async with _refresh_lock():
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.01)
                active -= 1
        await asyncio.gather(*(worker() for _ in range(5)))
        assert peak == 1
    asyncio.run(locking())

    with TestClient(create_app(Settings(api_key="synthetic-proxy-key", reasoning_effort="low", expose_reasoning_models=True))) as client:
        headers = {"Authorization": "Bearer synthetic-proxy-key"}
        assert client.get("/health").status_code == 200
        assert client.get("/v1/models").status_code == 401
        models = {item["id"]: item for item in client.get("/v1/models", headers=headers).json()["data"]}
        assert "gpt-6-astra-max" not in models
        assert models["gpt-6-astra"]["reasoning"]["default_effort"] == "low"
        assert models["gpt-6-astra"]["reasoning"]["supported_efforts"][-1] == "max"
        tags = client.get("/api/tags", headers=headers).json()["models"]
        assert all(item["details"]["format"] == "remote" for item in tags)
        assert client.post("/v1/responses", headers=headers, json={"model": "gpt-6-astra-max", "input": "hi"}).status_code == 400
        preflight = client.options("/v1/models", headers={"Origin": "https://example.com", "Access-Control-Request-Method": "GET"})
        assert "access-control-allow-origin" not in preflight.headers
    print("DOCKER_SMOKE_OK: nonroot read-only auth CORS discovery alias storage locking", flush=True)


if __name__ == "__main__":
    main()
