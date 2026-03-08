#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

GIST_ID = "255a945245d92c731d002ee3be93a74c"
COVERAGE_FILENAME = "gptmock-coverage.json"
TESTS_FILENAME = "gptmock-tests.json"
TEST_TARGETS = ["tests/"]


def run_pytest(paths: list[str], extra_args: list[str] | None = None) -> tuple[int, int, int, int]:
    cmd = ["uv", "run", "pytest", *paths, "--tb=short", "-q", *(extra_args or [])]
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    passed = failed = skipped = 0
    for line in result.stdout.splitlines():
        if not ("passed" in line or "failed" in line or "skipped" in line):
            continue
        parts = line.split()
        for i, part in enumerate(parts):
            if part in ("passed", "passed,") and i > 0 and parts[i - 1].isdigit():
                passed = int(parts[i - 1])
            elif part in ("failed", "failed,") and i > 0 and parts[i - 1].isdigit():
                failed = int(parts[i - 1])
            elif part in ("skipped", "skipped,") and i > 0 and parts[i - 1].isdigit():
                skipped = int(parts[i - 1])

    collected = passed + failed + skipped or 1
    return collected, passed, failed, skipped


def read_coverage_pct() -> int | None:
    try:
        with open("coverage.json", encoding="utf-8") as f:
            return round(json.load(f)["totals"]["percent_covered"])
    except Exception:
        return None


def reset_coverage_files() -> None:
    for path in [Path(".coverage"), Path("coverage.json")]:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def patch_gist(filename: str, data: dict) -> None:
    token = os.environ.get("GIST_TOKEN", "")
    if not token:
        print(f"GIST_TOKEN not set — skipping {filename} update", file=sys.stderr)
        print(f"Badge data: {json.dumps(data, indent=2)}")
        return

    payload = json.dumps({"files": {filename: {"content": json.dumps(data)}}}).encode()
    req = Request(
        f"https://api.github.com/gists/{GIST_ID}",
        data=payload,
        method="PATCH",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
    )
    with urlopen(req) as resp:
        if resp.status == 200:
            print(f"  {filename}: {data['message']}")
        else:
            print(f"  {filename}: Gist update failed ({resp.status})", file=sys.stderr)


def make_pct_badge(label: str, pct: int) -> dict:
    if pct == 100:
        color = "brightgreen"
    elif pct >= 90:
        color = "yellow"
    else:
        color = "red"
    return {"schemaVersion": 1, "label": label, "message": f"{pct}%", "color": color}


def main() -> None:
    reset_coverage_files()

    print("=" * 60)
    print("Running tests + coverage")
    print("=" * 60)
    collected, passed, failed, skipped = run_pytest(
        TEST_TARGETS,
        extra_args=["--cov", "--cov-report=json"],
    )
    cov_pct = read_coverage_pct()

    ran = collected - skipped
    tests_pct = round(passed / ran * 100) if ran > 0 else 0

    print("=" * 60)
    print("Updating badges...")
    if cov_pct is not None:
        patch_gist(COVERAGE_FILENAME, make_pct_badge("coverage", cov_pct))
    else:
        print(f"  {COVERAGE_FILENAME}: skipped (no coverage.json)")

    if skipped == collected:
        patch_gist(TESTS_FILENAME, {"schemaVersion": 1, "label": "tests", "message": "no tests", "color": "lightgrey"})
    else:
        patch_gist(TESTS_FILENAME, make_pct_badge("tests", tests_pct))

    print("=" * 60)
    print(f"Coverage  : {cov_pct}%")
    print(f"Tests     : {passed} passed, {failed} failed, {skipped} skipped")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
