from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

GIST_ID = "255a945245d92c731d002ee3be93a74c"
COVERAGE_FILENAME = "gptmock-coverage.json"
TESTS_FILENAME = "gptmock-tests.json"


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


def make_pct_badge(label: str, pct: int) -> dict[str, Any]:
    if pct == 100:
        color = "brightgreen"
    elif pct >= 90:
        color = "yellow"
    else:
        color = "red"
    return {"schemaVersion": 1, "label": label, "message": f"{pct}%", "color": color}


def patch_gist(filename: str, data: dict[str, Any]) -> None:
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


def update_gist_badges(*, tests_label: str, tests_pct: int, tests_collected: int, tests_skipped: int) -> None:
    cov_pct = read_coverage_pct()
    if cov_pct is not None:
        patch_gist(COVERAGE_FILENAME, make_pct_badge("coverage", cov_pct))

    if tests_skipped == tests_collected:
        patch_gist(TESTS_FILENAME, {"schemaVersion": 1, "label": tests_label, "message": "no tests", "color": "lightgrey"})
    else:
        patch_gist(TESTS_FILENAME, make_pct_badge(tests_label, tests_pct))
