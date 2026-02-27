# GPTMock

> **This is a fork of [RayBytes/chatmock](https://github.com/RayBytes/chatmock).**
> The original Flask + synchronous `requests` stack has been replaced with **FastAPI + async `httpx`**, a layered architecture (router / service / infra), `pydantic-settings` configuration, and `uv` as the build system.

**OpenAI & Ollama compatible API powered by your ChatGPT account.**

gptmock runs a local server that proxies requests to the ChatGPT Codex backend, exposing an OpenAI/Ollama compatible API. Use GPT-5, GPT-5-Codex, and other models directly from your ChatGPT Plus/Pro subscription — no API key required.

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
      - CHATGPT_LOCAL_LOGIN_BIND=0.0.0.0
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

Configure via `.env` file or docker-compose environment:

| Variable | Default | Description |
|----------|---------|-------------|
| `GPTMOCK_PORT` | `8000` | Server port |
| `GPTMOCK_VERBOSE` | `false` | Enable request/response logging |
| `GPTMOCK_REASONING_EFFORT` | `medium` | `minimal` / `low` / `medium` / `high` / `xhigh` |
| `GPTMOCK_REASONING_SUMMARY` | `auto` | `auto` / `concise` / `detailed` / `none` |
| `GPTMOCK_REASONING_COMPAT` | `think-tags` | `think-tags` / `o3` / `legacy` |
| `GPTMOCK_EXPOSE_REASONING_MODELS` | `false` | Expose reasoning levels as separate models |
| `GPTMOCK_DEFAULT_WEB_SEARCH` | `false` | Enable web search tool by default |

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
    model="gpt-5",
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
    model="gpt-5",
)
response = llm.invoke("hello world")
print(response.content)
```

### curl

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5",
    "messages": [{"role": "user", "content": "hello world"}]
  }'
```

---

## Supported Models

| Model | Reasoning Efforts | Status |
|-------|-------------------|--------|
| `gpt-5` | `minimal` / `low` / `medium` / `high` | ✅ Supported |
| `gpt-5.1` | `low` / `medium` / `high` | ✅ Supported |
| `gpt-5.2` | `low` / `medium` / `high` / `xhigh` | ✅ Supported |
| `gpt-5-codex` | `low` / `medium` / `high` | ✅ Supported |
| `gpt-5.1-codex` | `low` / `medium` / `high` | ✅ Supported |
| `gpt-5.1-codex-max` | `low` / `medium` / `high` / `xhigh` | ✅ Supported |
| `gpt-5.2-codex` | `low` / `medium` / `high` / `xhigh` | ✅ Supported |
| `gpt-5.3-codex` | `low` / `medium` / `high` / `xhigh` | ✅ Supported |
| `gpt-5.3-codex-spark` | `low` / `medium` / `high` / `xhigh` | ✅ Supported |

### Deprecated / Unsupported Models

| Model | Reason |
|-------|--------|
| `codex-mini` / `gpt-5.1-codex-mini` | ❌ Discontinued by Codex Backend — removed |
---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | OpenAI Chat Completions (stream / non-stream) |
| POST | `/v1/completions` | OpenAI Text Completions |
| POST | `/v1/responses` | OpenAI Responses API (for LangChain codex routing) |
| GET | `/v1/models` | List available models |
| POST | `/api/chat` | Ollama-compatible chat |
| GET | `/api/tags` | Ollama model list |
| GET | `/health` | Health check |

---

## Features

- **Streaming & Non-streaming** — real-time SSE and buffered JSON responses
- **Structured Output** — `response_format` with `json_schema` / `json_object` support
- **Tool / Function Calling** — including web search with URL citation annotations via `responses_tools`
- **Thinking Summaries** — `<think>` tags, `o3` reasoning format, or legacy mode
- **Responses API** — `POST /v1/responses` for LangChain and other clients that auto-route codex models
- **Ollama Compatibility** — drop-in replacement for Ollama API consumers
- **Auto Token Refresh** — JWT tokens are refreshed automatically before expiry

---

## Server Options

```
gptmock serve [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8000` | Bind port |
| `--verbose` | off | Log request/response payloads |
| `--reasoning-effort` | `medium` | Default reasoning effort level |
| `--reasoning-summary` | `auto` | Reasoning summary verbosity |
| `--reasoning-compat` | `think-tags` | How reasoning is exposed (`think-tags` / `o3` / `legacy`) |
| `--expose-reasoning-models` | off | Show each reasoning level as a separate model in `/v1/models` |
| `--enable-web-search` | off | Enable web search tool by default |

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
- When the model returns a thinking summary, it sends back thinking tags for compatibility with chat apps. Set `--reasoning-compat` to `legacy` to use the reasoning tag instead of inline text.
- This project is not affiliated with OpenAI. Use responsibly and at your own risk.

## Credits

- Original project: [RayBytes/chatmock](https://github.com/RayBytes/chatmock)
- This fork: [rapidrabbit76/GPTMock](https://github.com/rapidrabbit76/GPTMock)
