# GPT-6 Astra validation

Validated on 2026-09-05 against the ChatGPT Codex Responses backend using the
existing local GPTMock login. Results are account/backend observations, not a
guarantee for every account or the public OpenAI API. Credentials, account IDs,
response IDs, and private paths are omitted from this report.

## Upstream request contract

The [official Astra model page](https://developers.openai.com/api/docs/models/gpt-6-astra)
lists `low`, `medium`, `high`, `xhigh`, and `max`. GPTMock uses the concrete
`gpt-6-astra` ID without substituting another model.

Direct probes used `store: false`, `stream: true`, a short user message asking
for `OK`, and `reasoning: {"effort": "<tested effort>"}`. Successful cases reached
`response.completed`; acceptance was not inferred from HTTP 200 alone.

| Request | HTTP | Observed result |
| --- | --- | --- |
| `gpt-6-astra`, each of `low`, `medium`, `high`, `xhigh`, `max` | 200 | All five completed; actual model and reasoning effort matched the request |
| `reasoning.effort: none` | 400 | Unsupported effort for Astra |
| `reasoning.effort: minimal` | 400 | Unsupported effort for Astra |
| `reasoning.effort: ultra` | 400 | Invalid effort value |
| `service_tier: priority`, effort `low` | 200 | Completed as `gpt-6-astra`; actual tier was `default` |
| `reasoning.mode: pro`, effort `low` | 400 | Mode unsupported by this backend |
| Model `gpt-6-astra-pro` | 400 | Model request rejected |

`gpt-6-astra-fast` is a request alias for the concrete model plus
`service_tier: priority`. It is not advertised in `/v1/models` or `/api/tags`.
No Astra Pro alias or unsupported reasoning effort is registered. The existing
policy of rejecting explicit `reasoning.mode` remains in place.

## System-prompt compatibility

The backend rejects a literal `system` role with `System messages are not
allowed`. GPTMock accepts text system prompts from clients and moves their text
into the upstream `instructions` field, after existing instructions and in
original message order. Developer/user messages and tool results stay in
`input`. This happens before the first request, without a failed-request retry.

The conversion is scoped to the concrete Astra upstream model, so it also
covers fast aliases. It applies to Chat, Responses, Ollama chat, and Ollama
generate's `system` field. It does not mutate the caller's payload or discard
unsupported nontext system content. Tests cover multiple system messages,
existing instructions, developer-role preservation, and repeated adaptation.

## Live protocol validation

An in-process FastAPI TestClient exercised real upstream requests without
starting a listening GPTMock server. All 16 final cases passed:

- `/v1/responses`: all five reasoning efforts, fast alias, and SSE completion.
- `/v1/chat/completions`: non-streaming and SSE with a system/user conversation.
- `/api/chat`: non-streaming `think: max` and streaming with a system prompt.
- `/api/generate`: non-streaming and streaming with the native `system` field,
  plus strict JSON-schema output.
- Required strict function call through Responses with a system instruction,
  followed by a function result and completed final answer.
- `options.num_predict` omitted with `X-GPTMock-Omitted-Parameters` as before.

Non-streaming system-prompt checks require the `OK` marker specified only in the
system message. Successful responses retain `gpt-6-astra` and the upstream tier
`default`. These checks exercise GPTMock's HTTP formats; the OpenCode and Ollama
CLI programs were not rerun for Astra.

## Repository and Docker validation

- Unit/regression coverage checks discovery, efforts, aliases, exact model
  routing, system-prompt adaptation, strict tools, and actual model/tier output.
- Ruff passed. The deterministic pytest suite reported 299 passed and 114
  skipped; its live integration tests remain opt-in.
- `uv build` produced the wheel and source distribution.
- `docker-compose.yml` matches upstream `88f1ba9f08` exactly (Git blob
  `3fda26da52e6c3c358b910d8000eadf9126983da`). It retains
  `rapidrabbit76/gptmock:latest` and the original bind mount.
- The previous hardened configuration is preserved separately as
  `docker-compose.local.yml`. Both configurations pass Compose validation
  (the original is checked with `--no-env-resolution` because `.env` is local).
- The Docker engine is stopped, so no Astra container build or runtime test is
  claimed. Earlier Docker tests apply to the earlier revision, not this one.
