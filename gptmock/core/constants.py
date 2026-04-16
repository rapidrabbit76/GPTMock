from __future__ import annotations

import os

# OAuth Configuration
CLIENT_ID_DEFAULT = (
    os.getenv("GPTMOCK_CLIENT_ID")
    or os.getenv("CHATGPT_LOCAL_CLIENT_ID")
    or "app_EMoamEEZ73f0CkXaXp7hrann"
)
OAUTH_ISSUER_DEFAULT = (
    os.getenv("GPTMOCK_ISSUER")
    or os.getenv("CHATGPT_LOCAL_ISSUER")
    or "https://auth.openai.com"
)
OAUTH_TOKEN_URL = f"{OAUTH_ISSUER_DEFAULT}/oauth/token"

# ChatGPT API Endpoints
CHATGPT_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"

# SSE Event Types (Responses API)
SSE_OUTPUT_TEXT_DELTA = "response.output_text.delta"
SSE_OUTPUT_TEXT_DONE = "response.output_text.done"
SSE_OUTPUT_ITEM_DONE = "response.output_item.done"
SSE_CONTENT_PART_DONE = "response.content_part.done"
SSE_REASONING_SUMMARY_PART_ADDED = "response.reasoning_summary_part.added"
SSE_REASONING_SUMMARY_TEXT_DELTA = "response.reasoning_summary_text.delta"
SSE_REASONING_TEXT_DELTA = "response.reasoning_text.delta"
SSE_OUTPUT_ITEM_ADDED = "response.output_item.added"
SSE_FUNCTION_CALL_ARGS_DELTA = "response.function_call_arguments.delta"
SSE_FUNCTION_CALL_ARGS_DONE = "response.function_call_arguments.done"
SSE_RESPONSE_COMPLETED = "response.completed"
SSE_RESPONSE_FAILED = "response.failed"
