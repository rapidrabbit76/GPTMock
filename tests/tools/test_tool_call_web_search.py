#!/usr/bin/env python3
"""
Simple test: request a web_search tool via responses_tools payload.
Non-streaming request to make observation simpler.
"""

import json
import requests

URL = "http://127.0.0.1:8000/v1/chat/completions"

payload = {
    "model": "gpt-5",
    "messages": [
        {"role": "user", "content": "Find the latest news about GPTMock and summarize."}
    ],
    # Ask the server to include web_search as a Responses-style tool
    "responses_tools": [{"type": "web_search"}],
    "stream": False,
}

print("Payload:")
print(json.dumps(payload, indent=2))

try:
    resp = requests.post(URL, json=payload, timeout=60)
    print("Status:", resp.status_code)
    print(json.dumps(resp.json(), indent=2))
except requests.RequestException as e:
    print("Request failed:", e)
except Exception:
    print(resp.text if "resp" in locals() else "No response object available")
