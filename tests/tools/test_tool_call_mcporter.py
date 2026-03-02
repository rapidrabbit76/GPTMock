#!/usr/bin/env python3
"""
Test: use the `mcporter` (MCP) CLI to forward a chat completion request that requests a web_search tool.

This script assumes `mcporter` is installed and configured.
Do NOT run this script from the assistant — run it on your local machine.
"""

import json
import shutil
import subprocess
import sys

# Adjust if your mcporter CLI name differs
MCP_CLI = "mcporter"
URL = "http://127.0.0.1:8000/v1/chat/completions"

payload = {
    "model": "gpt-5",
    "messages": [
        {"role": "user", "content": "Search for recent GPTMock releases and summarize."}
    ],
    "responses_tools": [{"type": "web_search"}],
    "stream": False,
}

if not shutil.which(MCP_CLI):
    print(f"`{MCP_CLI}` not found in PATH. Install/configure mcporter and try again.")
    print(
        "This script expects an mcporter CLI that can forward an HTTP POST to the given URL."
    )
    sys.exit(2)

# Try a generic invocation first.
cmd = [MCP_CLI, "request", "--url", URL, "--json", json.dumps(payload)]

print("Running:", " ".join(cmd))
try:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
except Exception as e:
    print("Failed to run mcporter:", e)
    sys.exit(3)

print("Exit code:", proc.returncode)
print("--- STDOUT ---")
print(proc.stdout)
print("--- STDERR ---")
print(proc.stderr)

# If mcporter has a different subcommand (e.g., `call`), try a second variant.
if proc.returncode != 0:
    alt_cmd = [MCP_CLI, "call", "--url", URL, "--body", json.dumps(payload)]
    print("Attempting alternate invocation:", " ".join(alt_cmd))
    try:
        proc2 = subprocess.run(alt_cmd, capture_output=True, text=True, timeout=120)
        print("Alt exit code:", proc2.returncode)
        print("--- ALT STDOUT ---")
        print(proc2.stdout)
        print("--- ALT STDERR ---")
        print(proc2.stderr)
    except Exception as e:
        print("Alternate invocation failed:", e)

print("Done. Inspect output for tool_calls or upstream error details.")
