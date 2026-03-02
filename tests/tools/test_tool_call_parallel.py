#!/usr/bin/env python3
"""
Parallel tool calls test: request parallel_tool_calls=True and multiple functions.
This helps observe how GPTMock/upstream handle parallel tool invocation behavior.
"""

import json
import requests

URL = "http://127.0.0.1:8000/v1/chat/completions"

payload = {
    "model": "gpt-5",
    "messages": [
        {
            "role": "user",
            "content": "Gather the latest AI headlines and compute a sentiment score (0-100) for one headline.",
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

print("Payload:")
print(json.dumps(payload, indent=2))

try:
    resp = requests.post(URL, json=payload, timeout=120)
    print("Status:", resp.status_code)
    print(json.dumps(resp.json(), indent=2))
except requests.RequestException as e:
    print("Request failed:", e)
except Exception:
    print(resp.text if "resp" in locals() else "No response object available")
