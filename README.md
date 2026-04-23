<p align="center">
  <img src="assets/banner.png" alt="GPTMock banner" />
</p>

<h1 align="center">GPTMock</h1>

<p align="center"><strong>OpenAI &amp; Ollama compatible API powered by your ChatGPT account.</strong></p>

<p align="center">
  <a href="https://github.com/rapidrabbit76/GPTMock"><img alt="Tests" src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rapidrabbit76/255a945245d92c731d002ee3be93a74c/raw/gptmock-tests.json"></a>
  <a href="https://github.com/rapidrabbit76/GPTMock"><img alt="Coverage" src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rapidrabbit76/255a945245d92c731d002ee3be93a74c/raw/gptmock-coverage.json"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.13+" src="https://img.shields.io/badge/python-3.13%2B-blue.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green.svg"></a>
</p>

> **This is a fork of [RayBytes/chatmock](https://github.com/RayBytes/chatmock).**
> The original Flask + synchronous `requests` stack has been replaced with **FastAPI + async `httpx`**, a layered architecture (router / service / infra), `pydantic-settings` configuration, and `uv` as the build system.

Integration and coverage badges are updated from local runs. Refresh both by running `scripts/test.sh` with `GIST_TOKEN` available in your environment or `.env`.

gptmock runs a local server that proxies requests to the ChatGPT Codex backend, exposing an OpenAI/Ollama compatible API. Use GPT-5, GPT-5-Codex, and other models directly from your ChatGPT Plus/Pro subscription — no API key required.

> **Migration note:** `--reasoning-compat` now defaults to `standard`, which emits reasoning via `delta.reasoning_content` / `message.reasoning_content` instead of injecting `<think>` tags into `content`. Set `--reasoning-compat think-tags` (or `GPTMOCK_REASONING_COMPAT=think-tags`) to keep the old behavior.

## Requirements

- **Python 3.13+**
- **Paid ChatGPT account** (Plus / Pro / Team / Enterprise)
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (for uvx usage)

---

## Quick Start (uvx)

The fastest way to run gptmock. No clone, no install — just `uvx`.

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
> uvx --from "git+https://github.com/rapidrabbit76/GPTMock" gptmock login
> uvx --from "git+https://github.com/rapidrabbit76/GPTMock" gptmock serve
> ```

---

## Quick Start (Docker)

No build required — pull the pre-built image and run.

### 1. Create `docker-compose.yml`

```yaml
services:
  serve:
    image: rapidrabbit76/gptmock:latest
    container_name: gptmock
    command: ["serve", "--verbose", "--host", "0.0.0.0"]
    ports:
      - "8000:8000"
      - "1455:1455"  # OAuth callback port (needed during first-time login)
    volumes:
      - gptmock-data:/data
    environment:
      - GPTMOCK_HOME=/data
      - GPTMOCK_LOGIN_BIND=0.0.0.0
    healthcheck:
      test: ["CMD-SHELL", "python -c \"import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)\""]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 120s  # Allows time for first-time login before health checks begin

volumes:
  gptmock-data:
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
| `GPTMOCK_OLLAMA_VERSION` | `0.12.10` | Ollama API compatibility header version |

---

## Usage Examples

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="anything"  # ignored by gptmock
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
    api_key="anything",
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
| `gpt-5` | `minimal` / `low` / `medium` / `high` | ⚠️ Recognized by GPTMock, currently rejected upstream for ChatGPT Codex accounts |
| `gpt-5.1` | `low` / `medium` / `high` | ⚠️ Recognized by GPTMock, currently rejected upstream for ChatGPT Codex accounts |
| `gpt-5.2` | `low` / `medium` / `high` / `xhigh` | ✅ Verified upstream |
| `gpt-5-codex` | `low` / `medium` / `high` | ⚠️ Recognized by GPTMock, currently rejected upstream for ChatGPT Codex accounts |
| `gpt-5.1-codex` | `low` / `medium` / `high` | ⚠️ Recognized by GPTMock, currently rejected upstream for ChatGPT Codex accounts |
| `gpt-5.1-codex-mini` | `low` / `medium` / `high` | ⚠️ Recognized by GPTMock, currently rejected upstream for ChatGPT Codex accounts |
| `gpt-5.1-codex-max` | `low` / `medium` / `high` / `xhigh` | ⚠️ Recognized by GPTMock, currently rejected upstream for ChatGPT Codex accounts |
| `gpt-5.2-codex` | `low` / `medium` / `high` / `xhigh` | ⚠️ Recognized by GPTMock, currently rejected upstream for ChatGPT Codex accounts |
| `gpt-5.3-codex` | `low` / `medium` / `high` / `xhigh` | ✅ Verified upstream |
| `gpt-5.3-codex-spark` | `low` / `medium` / `high` / `xhigh` | ✅ Verified upstream |
| `gpt-5.4` | `low` / `medium` / `high` / `xhigh` | ✅ Verified upstream |
| `gpt-5.4-mini` | `low` / `medium` / `high` / `xhigh` | ✅ Verified upstream |
| `gpt-5.4-fast` | `low` / `medium` / `high` / `xhigh` | ✅ Supported (priority tier alias of `gpt-5.4`) |
| `gpt-5.4-mini-fast` | `low` / `medium` / `high` / `xhigh` | ✅ Supported (priority tier alias of `gpt-5.4-mini`) |

> **Fast variants** (`*-fast`) are synthetic aliases that map to the base model plus `service_tier="priority"` in the upstream payload. No separate endpoint or auth is required — the ChatGPT backend accepts them as paid-tier priority requests.

> **Upstream availability note:** model availability can change independently of GPTMock releases. GPTMock may recognize a model ID even when the current ChatGPT Codex backend rejects it for a specific account or subscription. On 2026-04-17, direct probe requests against the current upstream accepted `gpt-5.2`, `gpt-5.3-codex`, `gpt-5.3-codex-spark`, `gpt-5.4`, and `gpt-5.4-mini`, while rejecting `gpt-5`, `gpt-5.1`, `gpt-5-codex`, `gpt-5.1-codex`, `gpt-5.1-codex-mini`, `gpt-5.1-codex-max`, and `gpt-5.2-codex` with: `The '<model>' model is not supported when using Codex with a ChatGPT account.`

### Deprecated / Unsupported Models

None hardcoded in GPTMock at this time. See the upstream availability note above for models that are currently rejected by the ChatGPT Codex backend.
---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | OpenAI Chat Completions (stream / non-stream) |
| POST | `/v1/completions` | OpenAI Text Completions |
| POST | `/v1/responses` | OpenAI Responses API (for LangChain codex routing) |
| GET | `/v1/models` | List available models |
| GET | `/api/version` | Ollama-compatible version info |
| POST | `/api/chat` | Ollama-compatible chat |
| POST | `/api/show` | Ollama-compatible model details |
| GET | `/api/tags` | Ollama model list |
| GET | `/health` | Health check |

---

## Features

- **Streaming & Non-streaming** — real-time SSE and buffered JSON responses
- **Structured Output** — `response_format` with `json_schema` / `json_object` support
- **Tool / Function Calling** — including web search with URL citation annotations via `responses_tools`
- **Image Generation** — Responses API `image_generation` tool support with base64 PNG output
- **Local Image Inspection** — Codex-compatible `view_image` function tool for allowed local image paths
- **Thinking Summaries** — `<think>` tags, `o3` reasoning format, or legacy mode
- **Responses API** — `POST /v1/responses` for LangChain and other clients that auto-route codex models
- **Ollama Compatibility** — drop-in replacement for Ollama API consumers
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
| `--reasoning-effort` | `GPTMOCK_REASONING_EFFORT` | `medium` | `minimal` / `low` / `medium` / `high` / `xhigh` |
| `--reasoning-summary` | `GPTMOCK_REASONING_SUMMARY` | `auto` | `auto` / `concise` / `detailed` / `none` |
| `--reasoning-compat` | `GPTMOCK_REASONING_COMPAT` | `standard` | How reasoning is exposed: `standard` / `think-tags` / `o3` / `legacy` (`openai` is accepted as an alias for `standard`, `current` as an alias for `legacy`) |
| `--expose-reasoning-models` | `GPTMOCK_EXPOSE_REASONING_MODELS` | off | Show effort variants as separate models in `/v1/models` |
| `--enable-web-search` | `GPTMOCK_DEFAULT_WEB_SEARCH` | off | Enable web search by default when `responses_tools` is omitted |
| `--cors-origins` | `GPTMOCK_CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins |

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
    "model": "gpt-5",
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
- For the fastest responses, set `--reasoning-effort` to `low` and `--reasoning-summary` to `none`.
- The context size of this route is larger than what you get in the regular ChatGPT app.
- When the model returns a thinking summary, the default `standard` mode emits `reasoning_content` fields without polluting `content`. Set `--reasoning-compat think-tags` to keep `<think>` tags for older chat apps, or `--reasoning-compat legacy` for the older reasoning fields.
- This project is not affiliated with OpenAI. Use responsibly and at your own risk.

## Credits

- Original project: [RayBytes/chatmock](https://github.com/RayBytes/chatmock)
- This fork: [rapidrabbit76/GPTMock](https://github.com/rapidrabbit76/GPTMock)
