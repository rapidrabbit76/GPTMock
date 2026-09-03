<p align="center">
  <img src="assets/banner.png" alt="GPTMock banner" />
</p>

<h1 align="center">GPTMock</h1>

<p align="center"><strong>OpenAI &amp; Ollama compatible API powered by your ChatGPT account.</strong></p>

<p align="center">
  <a href="https://github.com/binary1215/GPTMock"><img alt="Tests" src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rapidrabbit76/255a945245d92c731d002ee3be93a74c/raw/gptmock-tests.json"></a>
  <a href="https://github.com/binary1215/GPTMock"><img alt="Coverage" src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rapidrabbit76/255a945245d92c731d002ee3be93a74c/raw/gptmock-coverage.json"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.13+" src="https://img.shields.io/badge/python-3.13%2B-blue.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green.svg"></a>
</p>

> **This is a fork of [RayBytes/chatmock](https://github.com/RayBytes/chatmock).**
> The original Flask + synchronous `requests` stack has been replaced with **FastAPI + async `httpx`**, a layered architecture (router / service / infra), `pydantic-settings` configuration, and `uv` as the build system.

Integration and coverage badges are updated from local runs. Refresh both by running `scripts/test.sh` with `GIST_TOKEN` available in your environment or `.env`.

GPTMock runs a local protocol adapter in front of the ChatGPT Codex backend. OpenAI SDKs, OpenAI-compatible frontends and gateways, and Ollama-compatible clients can use the same authenticated backend without GPTMock pretending that remote models are local weights. It advertises only model names verified against that backend; availability still depends on your paid ChatGPT account.

GPTMock preserves request intent rather than silently repairing rejected requests. Model IDs, roles, strict function schemas, tool choice, reasoning controls, and service-tier requests are forwarded according to their documented meaning. The model name, service tier, terminal status, and upstream errors returned by the backend remain authoritative.

> **Migration note:** `--reasoning-compat` now defaults to `standard`, which emits reasoning via `delta.reasoning_content` / `message.reasoning_content` instead of injecting `<think>` tags into `content`. Set `--reasoning-compat think-tags` (or `GPTMOCK_REASONING_COMPAT=think-tags`) to keep the old behavior.

## Requirements

- **Docker Engine 24+ with Docker Compose v2** (recommended deployment)
- **Paid ChatGPT account** (Plus / Pro / Team / Enterprise)
- **Python 3.13+** and [`uv`](https://docs.astral.sh/uv/getting-started/installation/) only for direct, non-Docker usage

---

## Quick Start (Docker, recommended)

The repository compose file builds the exact checked-out source and applies the hardened runtime defaults.

### 1. Clone and build

```bash
git clone https://github.com/binary1215/GPTMock.git
cd GPTMock
docker compose build
```

### 2. Start (first run — login + serve in one step)

Run the container interactively. If no credentials are found, the login flow starts automatically:

```bash
docker compose run --rm --service-ports serve
```

A URL will be printed in the terminal:

```
No credentials found. Starting login flow...
Starting local login server on http://localhost:1455
If your browser did not open, navigate to:
  https://auth.openai.com/oauth/authorize?...

If the browser can't reach this machine, paste the full redirect URL here and press Enter:
```

**Two ways to complete login:**

1. **Browser on the same machine** — the URL opens automatically and the OAuth callback is caught on port 1455.
2. **Browser on a different machine** — open the URL, complete login, then copy the full redirect URL from the browser address bar (starts with `http://localhost:1455/auth/callback?code=...`) and paste it into the terminal.

Once login succeeds, the server starts automatically.

### 3. Subsequent starts

Once credentials are saved in the volume, just run in the background:

```bash
docker compose up -d serve
```

The repository compose file uses the named volume `gptmock-data`. If you used an older `./volumes/gptmock` bind mount, copy its `auth.json` into the named volume before removing the old directory.

### 4. Verify

```bash
curl -s http://localhost:8000/health | jq .
```

### Docker Environment Variables

All server options below are also available as environment variables. Use the `GPTMOCK_*` canonical names (see [Server Options](#server-options)).

Additional Docker-specific variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GPTMOCK_HOME` | `/data` | Auth file directory — mount a volume here |
| `GPTMOCK_LOGIN_BIND` | `0.0.0.0` | OAuth callback server bind address |
| `GPTMOCK_API_KEY` | unset | Optional Bearer token required by `/v1/*` and `/api/*` routes |
| `GPTMOCK_OUTPUT_TOKEN_POLICY` | `omit` | Handle unenforceable client output limits: `omit` with warning/header or `reject` with HTTP 400 |
| `GPTMOCK_OLLAMA_VERSION` | `0.12.10` | Ollama API compatibility header version |

The published ports bind to host loopback by default. If you expose them on a LAN, configure `GPTMOCK_API_KEY` and an explicit `GPTMOCK_CORS_ORIGINS` allowlist first.

### Docker Security Defaults

The Compose deployment runs as UID/GID `10001`, mounts the root filesystem read-only, drops every Linux capability, enables `no-new-privileges`, applies a PID limit, and persists credentials only in the `gptmock-data` volume. Ports `8000` and `1455` are published to host loopback only.

API authentication is optional for local loopback use. To require a Bearer token, copy `.env.example` to `.env` and set:

```dotenv
GPTMOCK_API_KEY=replace-with-a-long-random-value
GPTMOCK_CORS_ORIGINS=
```

Clients must then send `Authorization: Bearer <GPTMOCK_API_KEY>` to `/v1/*` and `/api/*`. The health endpoint remains unauthenticated for container health checks. Browser CORS access stays disabled until an explicit origin allowlist is configured.

---

## Direct Install (uvx)

This is the non-Docker development path. No clone or persistent installation is needed.

### 1. Login

```bash
uvx gptmock login
```

A browser window will open for ChatGPT OAuth. After login, tokens are saved to `~/.config/gptmock/auth.json`.

### 2. Start the server

```bash
uvx gptmock serve
```

The server starts at `http://127.0.0.1:8000`. Use `http://127.0.0.1:8000/v1` as your OpenAI base URL.

### 3. Verify

```bash
uvx gptmock info
```

### Tip: Shell Alias

```bash
alias gptmock='uvx gptmock'

gptmock login
gptmock serve --port 9000
gptmock info
```

> **Note:** To install directly from the GitHub repository instead of PyPI:
> ```bash
> uvx --from "git+https://github.com/binary1215/GPTMock" gptmock login
> uvx --from "git+https://github.com/binary1215/GPTMock" gptmock serve
> ```

---

## Usage Examples

### OpenCode

Use OpenCode's Responses provider, not its Chat Completions compatibility provider. OpenCode 1.x configuration:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "gptmock": {
      "npm": "@ai-sdk/openai",
      "name": "GPTMock",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "replace-with-your-GPTMOCK_API_KEY"
      },
      "models": {
        "gpt-5.6-luna": {
          "name": "GPT-5.6 Luna",
          "reasoning": true,
          "tool_call": true,
          "variants": {
            "none": { "reasoningEffort": "none" },
            "low": { "reasoningEffort": "low" },
            "medium": { "reasoningEffort": "medium" },
            "high": { "reasoningEffort": "high" },
            "xhigh": { "reasoningEffort": "xhigh" },
            "max": { "reasoningEffort": "max" }
          }
        }
      }
    }
  }
}
```

Select `gptmock/gpt-5.6-luna` and the desired variant. OpenCode automatically sends `max_output_tokens`; GPTMock's default `omit` policy keeps the request compatible while explicitly reporting that the limit was not enforced.

Do not use `@ai-sdk/openai-compatible` for OpenCode with this backend. That adapter calls `/v1/chat/completions` and sends OpenCode's system prompt as a `system` message. GPTMock preserves the role as requested, and the connected ChatGPT Codex backend currently rejects it with `System messages are not allowed`. The Responses adapter sends the prompt as `instructions` and was verified end to end.

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="gptmock-local"  # use GPTMOCK_API_KEY when it is configured
)

resp = client.chat.completions.create(
    model="gpt-5.4",
    messages=[{"role": "user", "content": "hello world"}]
)
print(resp.choices[0].message.content)
```

### Python (LangChain)

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="gptmock-local",  # use GPTMOCK_API_KEY when it is configured
    model="gpt-5.4",
)
response = llm.invoke("hello world")
print(response.content)
```

### curl

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.4",
    "messages": [{"role": "user", "content": "hello world"}]
  }'
```

### Image Generation (Responses API)

GPTMock can expose the ChatGPT Codex backend's built-in image generation tool through `POST /v1/responses`. This uses your existing GPTMock / Codex OAuth credentials; no separate OpenAI API key is required.

Pass an `image_generation` tool in the Responses API request:

```bash
curl http://127.0.0.1:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.4",
    "input": [
      {
        "type": "message",
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": "Use the image_generation tool to create a cute illustration of a fluffy orange tabby cat sitting on a white background. Return only the generated image."
          }
        ]
      }
    ],
    "tools": [{"type": "image_generation", "output_format": "png"}],
    "tool_choice": "auto",
    "stream": false
  }'
```

For non-streaming requests, generated images are returned as `image_generation_call` items in `output`. The `result` field is a base64-encoded PNG payload:

```json
{
  "output": [
    {
      "type": "message",
      "status": "completed",
      "role": "assistant",
      "content": [{"type": "output_text", "text": ""}]
    },
    {
      "type": "image_generation_call",
      "id": "ig_...",
      "status": "generating",
      "output_format": "png",
      "revised_prompt": "A cute illustration of a fluffy orange tabby cat...",
      "result": "<base64 png>"
    }
  ]
}
```

Decode and save the first generated image with Python:

```python
import base64

image_b64 = response["output"][1]["result"]
with open("cat.png", "wb") as fp:
    fp.write(base64.b64decode(image_b64))
```

You can also run the included live probe script from a checked-out repository:

```bash
uv run python scripts/probe_image_generation.py \
  --model gpt-5.4 \
  --prompt "Use the image_generation tool to create a cute cat illustration. Return only the generated image." \
  --output .omx/logs/cat.png
```

> **Notes:** `gpt-5.4` and `gpt-5.4-mini` have been verified with this flow. The model interprets the request and invokes the built-in tool; the image bytes come back in the `image_generation_call.result` field. Model availability and image-generation entitlements are controlled by the upstream ChatGPT Codex backend and can vary by account.

### Local Image Inspection (`view_image`)

GPTMock also supports a Codex-compatible `view_image` client-side tool for `POST /v1/responses`. Unlike `image_generation`, this is not executed by the upstream backend: GPTMock reads the local file, returns it to the model as an `input_image` function-call output, and then continues the Responses turn.

Enable it per request by passing the shorthand tool:

```bash
curl http://127.0.0.1:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.4-mini",
    "input": [
      {
        "type": "message",
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": "Use view_image to inspect this local image path: assets/banner.png. Describe it briefly."
          }
        ]
      }
    ],
    "tools": [{"type": "view_image"}],
    "tool_choice": "auto",
    "stream": false
  }'
```

The shorthand is normalized to a Responses function tool named `view_image`. You can also provide an explicit function tool with the same name.

By default, `view_image` can read files under the server's current working directory only. Configure the readable roots when needed:

```bash
GPTMOCK_VIEW_IMAGE_ROOTS="/path/to/images:/another/root" gptmock serve
```

Additional knobs:

| Variable | Default | Description |
|----------|---------|-------------|
| `GPTMOCK_VIEW_IMAGE_ROOTS` | server cwd | `os.pathsep`-separated list of readable image roots |
| `GPTMOCK_VIEW_IMAGE_ALLOW_ANY_PATH` | off | Set to `1` to allow any local path readable by the server process |
| `GPTMOCK_VIEW_IMAGE_MAX_BYTES` | `20971520` | Maximum image file size in bytes |

Supported image content types are PNG, JPEG, GIF, and WebP. `detail: "original"` is accepted when the model requests original-resolution handling.

---

## Supported Models

| Model | Reasoning Efforts | Status |
|-------|-------------------|--------|
| `gpt-5.3-codex-spark` | `low` / `medium` / `high` / `xhigh` | ✅ Verified upstream |
| `gpt-5.4` | `low` / `medium` / `high` / `xhigh` | ✅ Verified upstream |
| `gpt-5.5` | `low` / `medium` / `high` / `xhigh` | ✅ Verified upstream |
| `gpt-5.6` | `none` / `low` / `medium` / `high` / `xhigh` / `max` | ✅ Verified alias; upstream resolves it to `gpt-5.6-sol` |
| `gpt-5.6-sol` | `none` / `low` / `medium` / `high` / `xhigh` / `max` | ✅ Verified upstream |
| `gpt-5.6-terra` | `none` / `low` / `medium` / `high` / `xhigh` / `max` | ✅ Verified upstream |
| `gpt-5.6-luna` | `none` / `low` / `medium` / `high` / `xhigh` / `max` | ✅ Verified upstream |
| `gpt-5.4-mini` | `low` / `medium` / `high` / `xhigh` | ✅ Verified upstream |

Direct Docker probes on 2026-09-03 confirmed every listed GPT-5.6 reasoning effort (`none` through `max`) for Sol, Terra, and Luna through `/v1/responses`. All 18 requests completed successfully and returned the requested concrete model name.

> **Fast compatibility aliases:** `*-fast` names are accepted when requested directly and add `service_tier="priority"` to the same verified base-model request. They are not separate models, are not advertised by `/v1/models` or `/api/tags`, and do not guarantee priority service. New integrations should prefer the verified base model plus an explicit `service_tier="priority"` request. GPTMock returns the actual upstream `service_tier` unchanged. All seven compatibility aliases were accepted on 2026-08-26, but every ChatGPT Codex backend probe reported `service_tier="default"`.

> **GPT-5.6 compatibility note:** GPTMock exposes only verified GPT-5.6 model names. `gpt-5.6` follows OpenAI's documented alias to `gpt-5.6-sol`. OpenAI documents Pro as `reasoning.mode="pro"` on the same model rather than as a `*-pro` model ID, but the ChatGPT Codex backend rejected that mode in direct probes. GPTMock therefore advertises no Pro model slug and rejects `reasoning.mode` locally instead of sending a request known to be unsupported by this upstream.

> **Upstream availability note:** model availability can change independently of GPTMock releases. The advertised list reflects direct probes made on 2026-08-26. `gpt-5`, `gpt-5.1`, `gpt-5.2`, `gpt-5-codex`, `gpt-5.1-codex`, `gpt-5.1-codex-mini`, `gpt-5.1-codex-max`, `gpt-5.2-codex`, and `gpt-5.3-codex` were rejected and are therefore not advertised.

### Deprecated / Unsupported Models

Rejected names are omitted from `/v1/models` and `/api/tags`. A client can still send an arbitrary model identifier; GPTMock forwards it unchanged and preserves the upstream rejection instead of silently routing it to a different model.

### Request and Response Semantics

| Input or event | GPTMock behavior |
|----------------|------------------|
| `system` and `developer` messages | Preserved as distinct input roles |
| Function tools with `strict: true` | Strict schema flag and parameters are preserved |
| `tool_choice: "required"` | Forwarded as required; never weakened to `auto` |
| Rejected tools or model options | Upstream error is returned; GPTMock does not remove tools and retry |
| `reasoning.effort` or Chat `reasoning_effort` | Validated against the selected model family and forwarded; conflicting values are rejected |
| Explicit `reasoning.mode` | Rejected locally; the connected ChatGPT Codex backend rejected Pro mode |
| `max_output_tokens`, `max_completion_tokens`, or `max_tokens` | Not forwarded because the ChatGPT Codex upstream cannot enforce them. Default `omit` mode logs a warning and returns `X-GPTMock-Omitted-Parameters`; `reject` mode returns HTTP 400 |
| `service_tier` or `*-fast` request | Requested tier is sent, while the tier actually returned by upstream is exposed unchanged |
| `response.incomplete` | Preserved by `/v1/responses`; mapped to `length` or `content_filter` by Chat/Text/Ollama compatibility responses |
| `response.failed` or interrupted SSE | Returned as an explicit error; an interrupted stream is never converted into a successful completion |
| Upstream response model | Returned unchanged instead of being replaced with the requested alias |
| Ollama model metadata | Marked as remote with zero/empty unknown size and digest values; no GGUF, Llama family, parameter size, quantization, or local evaluation timings are fabricated |

The GPT-5.6 alias and reasoning-effort range follow [OpenAI's GPT-5.6 model guidance](https://developers.openai.com/api/docs/guides/latest-model). GPTMock's accepted options are narrower because they reflect what the ChatGPT Codex backend accepted during the dated probes above, not what may be available through a separate OpenAI API account.

The default output-token policy is intentionally compatibility-oriented. OpenAI SDKs and agent frontends commonly add an output-token field even when the user did not set one. GPTMock accepts such requests but does not claim that the limit was honored: the field is omitted only from the upstream request, a warning names the omitted field, and the HTTP response exposes the same fact. Use `--output-token-policy reject` when failing closed is preferable to frontend compatibility.

Docker-backed OpenCode 1.17.18 probes on 2026-09-03 verified direct Luna/Sol/Terra model selection, `none`/`high`/`max` reasoning variants, `service_tier="priority"` request preservation, streaming, tool definitions, `tool_choice="auto"`, and a complete function-call/result/final-answer loop. The upstream response still reported the actual tier as `default`. OpenCode did not emit configured `top_p`, `tool_choice="required"`, or strict tool schemas in these runs, so those semantics remain covered by GPTMock's direct request tests rather than claimed as OpenCode-verified.

### Ollama Request Semantics

GPTMock supports both `/api/chat` and `/api/generate`. The native Ollama `think` field maps string levels to reasoning effort; GPTMock additionally accepts the full model-specific effort range (`none` through `max`) and the explicit `reasoning_effort` field. `think: false` suppresses the returned thinking summary. The non-standard `service_tier: "priority"` field requests priority service, while the actual upstream tier remains authoritative. Ollama `format: "json"` and JSON-schema objects map to structured output.

`options.num_predict` cannot be enforced by the ChatGPT Codex backend, so it follows `GPTMOCK_OUTPUT_TOKEN_POLICY`: default `omit` mode returns `X-GPTMock-Omitted-Parameters: options.num_predict`, while `reject` mode returns HTTP 400. Other non-empty Ollama runtime `options` are rejected instead of being silently ignored. Generate-only `suffix`, non-empty `template`, `context`, and `raw: true` are also rejected because this upstream cannot preserve their meaning.

Docker probes on 2026-09-03 verified authenticated raw HTTP requests for tags, show, structured output, reasoning, priority tier requests, streaming, strict required tools, tool-result continuation, output-limit policy, and upstream errors. Ollama CLI 0.33.1 also completed `ollama show` and a streaming `ollama run` through the loopback-only unauthenticated container. Two CLI limitations remain explicit: the CLI did not send `OLLAMA_API_KEY` to a custom local host, and `ollama list` panicked when it encountered the intentionally empty remote-model digest. GPTMock will not fabricate a weight digest to satisfy that client assumption; use `/api/tags`, `ollama show <model>`, or an Ollama-compatible client that accepts remote metadata.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | OpenAI Chat Completions (stream / non-stream) |
| POST | `/v1/completions` | OpenAI Text Completions |
| POST | `/v1/responses` | OpenAI Responses API semantics |
| GET | `/v1/models` | List available models |
| GET | `/api/version` | Ollama-compatible version info |
| POST | `/api/chat` | Ollama-compatible chat |
| POST | `/api/generate` | Ollama-compatible text generation |
| POST | `/api/show` | Ollama-compatible model details |
| GET | `/api/tags` | Ollama model list |
| GET | `/health` | Health check |

---

## Features

- **Streaming & Non-streaming** — real-time SSE and buffered JSON responses, including explicit incomplete terminal states
- **Structured Output** — `response_format` with `json_schema` / `json_object` support
- **Tool / Function Calling** — including web search with URL citation annotations via `responses_tools`
- **Image Generation** — Responses API `image_generation` tool support with base64 PNG output
- **Local Image Inspection** — Codex-compatible `view_image` function tool for allowed local image paths
- **Thinking Summaries** — `<think>` tags, `o3` reasoning format, or legacy mode
- **Responses API** — `POST /v1/responses` for LangChain and other clients that auto-route codex models
- **Ollama Compatibility** — chat and generate APIs with remote-model metadata, without fabricated GGUF sizes, digests, or local evaluation timings
- **Auto Token Refresh** — JWT tokens are refreshed automatically before expiry

---

## Server Options

```
gptmock serve [OPTIONS]
```

Each option can also be set via environment variable. Precedence: **CLI flag > `GPTMOCK_*` env > `CHATGPT_LOCAL_*` legacy env > default**.

| Option | Env var | Default | Description |
|--------|---------|---------|-------------|
| `--host` | `GPTMOCK_HOST` | `127.0.0.1` | Bind address |
| `--port` | `GPTMOCK_PORT` | `8000` | Bind port |
| `--verbose` | `GPTMOCK_VERBOSE` | off | Log request/response payloads |
| `--verbose-obfuscation` | `GPTMOCK_VERBOSE_OBFUSCATION` | off | Also dump raw SSE/obfuscation events |
| `--debug-model` | `GPTMOCK_DEBUG_MODEL` | — | Force all requests to use this model name |
| `--reasoning-effort` | `GPTMOCK_REASONING_EFFORT` | `medium` | `none` / `minimal` / `low` / `medium` / `high` / `xhigh` / `max`; availability is model-specific |
| `--reasoning-summary` | `GPTMOCK_REASONING_SUMMARY` | `auto` | `auto` / `concise` / `detailed` / `none` |
| `--reasoning-compat` | `GPTMOCK_REASONING_COMPAT` | `standard` | How reasoning is exposed: `standard` / `think-tags` / `o3` / `legacy` (`openai` is accepted as an alias for `standard`, `current` as an alias for `legacy`) |
| `--expose-reasoning-models` | `GPTMOCK_EXPOSE_REASONING_MODELS` | off | Show effort variants as separate models in `/v1/models` |
| `--enable-web-search` | `GPTMOCK_DEFAULT_WEB_SEARCH` | off | Enable web search by default when `responses_tools` is omitted |
| `--output-token-policy` | `GPTMOCK_OUTPUT_TOKEN_POLICY` | `omit` | `omit` unenforceable output limits with warning/header, or `reject` them with HTTP 400 |
| `--cors-origins` | `GPTMOCK_CORS_ORIGINS` | disabled | Comma-separated allowed CORS origins |
| — | `GPTMOCK_API_KEY` | unset | Optional Bearer token for `/v1/*` and `/api/*`; strongly recommended before non-loopback exposure |

> **Legacy aliases**: `CHATGPT_LOCAL_REASONING_EFFORT`, `CHATGPT_LOCAL_REASONING_SUMMARY`, `CHATGPT_LOCAL_REASONING_COMPAT`, `CHATGPT_LOCAL_EXPOSE_REASONING_MODELS`, `CHATGPT_LOCAL_ENABLE_WEB_SEARCH`, `CHATGPT_LOCAL_DEBUG_MODEL` are still accepted as fallbacks.

---

## Web Search

Use `--enable-web-search` to enable the web search tool by default for all requests. When enabled, the model decides autonomously whether a query needs a web search. You can also enable web search per-request without the server flag by passing the parameters below.

### Request Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `responses_tools` | `[{"type":"web_search"}]` | Enable web search for this request |
| `responses_tool_choice` | `"auto"` / `"none"` | Let the model decide, or disable |

### Annotations (URL Citations)

When web search is active, the model may return `annotations` containing source URLs. These are included automatically in responses:

**Non-streaming** (`stream: false`) — annotations are attached to the message:

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "SpaceX launched 29 Starlink satellites...",
        "annotations": [
          {
            "type": "url_citation",
            "start_index": 0,
            "end_index": 150,
            "url": "https://spaceflightnow.com/...",
            "title": "SpaceX Falcon 9 launch"
          }
        ]
      }
    }
  ]
}
```

**Streaming** (`stream: true`) — annotations arrive as a dedicated chunk before the final `stop` chunk:

```json
data: {"choices": [{"delta": {"annotations": [{"type": "url_citation", "start_index": 0, "end_index": 150, "url": "https://...", "title": "..."}]}, "finish_reason": null}]}
data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}
```

**Responses API** (`POST /v1/responses`, non-streaming) — annotations are nested inside the output content:

```json
{
  "output": [
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "SpaceX launched 29 Starlink satellites...",
          "annotations": [
            {
              "type": "url_citation",
              "start_index": 0,
              "end_index": 150,
              "url": "https://spaceflightnow.com/...",
              "title": "SpaceX Falcon 9 launch"
            }
          ]
        }
      ]
    }
  ]
}
```

### Example Request

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.4",
    "messages": [{"role":"user","content":"Find current METAR rules"}],
    "stream": true,
    "responses_tools": [{"type": "web_search"}],
    "responses_tool_choice": "auto"
  }'
```

---

## Notes & Limits

- Requires an active, paid ChatGPT account.
- Context length may be partially used by internal system instructions.
- For the lowest-reasoning latency baseline, use the lowest effort supported by the selected model (`none` for GPT-5.6, otherwise usually `low`) and set `--reasoning-summary` to `none`.
- Context limits and account entitlements are controlled by the ChatGPT Codex backend and may differ from the ChatGPT app or OpenAI API.
- When the model returns a thinking summary, the default `standard` mode emits `reasoning_content` fields without polluting `content`. Set `--reasoning-compat think-tags` to keep `<think>` tags for older chat apps, or `--reasoning-compat legacy` for the older reasoning fields.
- This project is not affiliated with OpenAI. Use responsibly and at your own risk.

## Credits

- Original project: [RayBytes/chatmock](https://github.com/RayBytes/chatmock)
- This fork: [rapidrabbit76/GPTMock](https://github.com/rapidrabbit76/GPTMock)
