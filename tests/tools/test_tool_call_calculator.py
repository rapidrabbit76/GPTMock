#!/usr/bin/env python3
"""
Test: request a function-style tool call via tools field.
This uses OpenAI Responses-compatible tool schema used by the server's converter.
"""

import json
import requests

URL = "http://127.0.0.1:8000/v1/chat/completions"

payload = {
    "model": "gpt-5",
    "messages": [
        {"role": "user", "content": "Calculate 37 * 42 and explain the steps briefly."}
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
