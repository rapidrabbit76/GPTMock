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
uvx --from "git+https://github.com/rapidrabbit76/GPTMock" gptmock login
```

A browser window will open for ChatGPT OAuth. After login, tokens are saved to `~/.config/gptmock/auth.json`.

### 2. Start the server

```bash
uvx --from "git+https://github.com/rapidrabbit76/GPTMock" gptmock serve
```

The server starts at `http://127.0.0.1:8000`. Use `http://127.0.0.1:8000/v1` as your OpenAI base URL.

### 3. Verify

```bash
uvx --from "git+https://github.com/rapidrabbit76/GPTMock" gptmock info
```

### Tip: Shell Alias

```bash
alias gptmock='uvx --from "git+https://github.com/rapidrabbit76/GPTMock" gptmock'

gptmock login
gptmock serve --port 9000
gptmock info
```

---

## Quick Start (Docker)

### 1. Setup

```bash
git clone https://github.com/rapidrabbit76/GPTMock.git
cd gptmock
cp .env.example .env
docker compose build
```

### 2. Login

```bash
docker compose run --rm --service-ports login login
```

A URL will be printed. Open it in your browser and complete the OAuth flow. If your browser can't reach the container, copy the full redirect URL from the browser address bar and paste it into the terminal.

### 3. Start the server

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

| Model | Reasoning Efforts |
|-------|-------------------|
| `gpt-5` | `minimal` / `low` / `medium` / `high` |
| `gpt-5.1` | `low` / `medium` / `high` |
| `gpt-5.2` | `low` / `medium` / `high` / `xhigh` |
| `gpt-5-codex` | `low` / `medium` / `high` |
| `gpt-5.1-codex` | `low` / `medium` / `high` |
| `gpt-5.1-codex-max` | `low` / `medium` / `high` / `xhigh` |
| `gpt-5.1-codex-mini` | `low` / `medium` / `high` |
| `gpt-5.2-codex` | `low` / `medium` / `high` / `xhigh` |
| `gpt-5.3-codex` | `low` / `medium` / `high` / `xhigh` |
| `codex-mini` | `low` / `medium` / `high` |

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
- **Tool / Function Calling** — including web search via `responses_tools`
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

## Notes & Limits

- Requires an active, paid ChatGPT account.
- Context length may be partially used by internal system instructions.
- This project is not affiliated with OpenAI. Use responsibly and at your own risk.

## Credits

- Original project: [RayBytes/gptmock](https://github.com/RayBytes/gptmock)
- This fork: [rapidrabbit76/GPTMockFastAPI](https://github.com/rapidrabbit76/GPTMockFastAPI)

  <p><b>OpenAI & Ollama compatible API powered by your ChatGPT plan.</b></p>
  <p>Use your ChatGPT Plus/Pro account to call OpenAI models from code or alternate chat UIs.</p>
  <br>
</div>

## What It Does

gptmock runs a local server that creates an OpenAI/Ollama compatible API, and requests are then fulfilled using your authenticated ChatGPT login with the oauth client of Codex, OpenAI's coding CLI tool. This allows you to use GPT-5, GPT-5-Codex, and other models right through your OpenAI account, without requiring an api key. You are then able to use it in other chat apps or other coding tools. <br>
This does require a paid ChatGPT account.

> **Fork note:** This fork migrates the original [RayBytes/gptmock](https://github.com/RayBytes/gptmock) from Flask + synchronous `requests` to **FastAPI + async `httpx`**, adds a layered architecture (router / service / infra), manages configuration via `pydantic-settings`, and uses `uv` as the build system. It also adds the OpenAI Responses API endpoint (`POST /v1/responses`) and structured output (`response_format`) support.

## Quickstart

### Homebrew (macOS)

Install gptmock as a command-line tool using [Homebrew](https://brew.sh/):
```
brew tap RayBytes/gptmock
brew install gptmock
```

### Python
If you wish to run this as a local Python server, you are also freely welcome to.

Clone or download this repository, then cd into the project directory. Then follow the instrunctions listed below.

1. Sign in with your ChatGPT account and follow the prompts
```bash
python gptmock.py login
```
You can make sure this worked by running `python gptmock.py info`

2. After the login completes successfully, you can just simply start the local server

```bash
python gptmock.py serve
```
Then, you can simply use the address and port as the baseURL as you require (http://127.0.0.1:8000 by default)

**Reminder:** When setting a baseURL in other applications, make you sure you include /v1/ at the end of the URL if you're using this as a OpenAI compatible endpoint (e.g http://127.0.0.1:8000/v1)

### Docker

Read [the docker instrunctions here](https://github.com/RayBytes/gptmock/blob/main/DOCKER.md)

# Examples

### Python 

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="key"  # ignored
)

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[{"role": "user", "content": "hello world"}]
)

print(resp.choices[0].message.content)
```

### curl

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5",
    "messages": [{"role":"user","content":"hello world"}]
  }'
```

# What's supported

- Tool/Function calling 
- Vision/Image understanding
- Thinking summaries (through thinking tags)
- Thinking effort

## Notes & Limits

- Requires an active, paid ChatGPT account.
- Some context length might be taken up by internal instructions (but they dont seem to degrade the model) 
- Use responsibly and at your own risk. This project is not affiliated with OpenAI, and is a educational exercise.

# Supported models
- `gpt-5`
- `gpt-5.1`
- `gpt-5.2`
- `gpt-5-codex`
- `gpt-5.2-codex`
- `gpt-5.1-codex`
- `gpt-5.1-codex-max`
- `gpt-5.1-codex-mini`
- `gpt-5.3-codex`
- `codex-mini`

# Customisation / Configuration

### Thinking effort

- `--reasoning-effort` (choice of minimal,low,medium,high,xhigh)<br>
GPT-5 has a configurable amount of "effort" it can put into thinking, which may cause it to take more time for a response to return, but may overall give a smarter answer. Applying this parameter after `serve` forces the server to use this reasoning effort by default, unless overrided by the API request with a different effort set. The default reasoning effort without setting this parameter is `medium`.<br>
    The `gpt-5.1` family (including codex) supports `low`, `medium`, and `high` while `gpt-5.1-codex-max` adds `xhigh`. The `gpt-5.2` family (including codex) supports `low`, `medium`, `high`, and `xhigh`. 

### Thinking summaries

- `--reasoning-summary` (choice of auto,concise,detailed,none)<br>
Models like GPT-5 do not return raw thinking content, but instead return thinking summaries. These can also be customised by you.

### OpenAI Tools

- `--enable-web-search`<br>
You can also access OpenAI tools through this project. Currently, only web search is available.
You can enable it by starting the server with this parameter, which will allow OpenAI to determine when a request requires a web search, or you can use the following parameters during a request to the API to enable web search:
<br><br>
`responses_tools`: supports `[{"type":"web_search"}]` / `{ "type": "web_search_preview" }`<br>
`responses_tool_choice`: `"auto"` or `"none"`

#### Example usage
```json
{
  "model": "gpt-5",
  "messages": [{"role":"user","content":"Find current METAR rules"}],
  "stream": true,
  "responses_tools": [{"type": "web_search"}],
  "responses_tool_choice": "auto"
}
```

### Expose reasoning models

- `--expose-reasoning-models`<br>
If your preferred app doesn’t support selecting reasoning effort, or you just want a simpler approach, this parameter exposes each reasoning level as a separate, queryable model. Each reasoning level also appears individually under ⁠/v1/models, so model pickers in your favorite chat apps will list all reasoning options as distinct models you can switch between.

## Notes
If you wish to have the fastest responses, I'd recommend setting `--reasoning-effort` to low, and `--reasoning-summary` to none. <br>
All parameters and choices can be seen by sending `python gptmock.py serve --h`<br>
The context size of this route is also larger than what you get access to in the regular ChatGPT app.<br>

When the model returns a thinking summary, the model will send back thinking tags to make it compatible with chat apps. **If you don't like this behavior, you can instead set `--reasoning-compat` to legacy, and reasoning will be set in the reasoning tag instead of being returned in the actual response text.**
