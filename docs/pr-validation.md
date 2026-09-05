# Pre-PR validation

Date: 2026-09-05. Upstream baseline: `rapidrabbit76/GPTMock:main` at
`88f1ba9f089056d4ed992c6cc84e0f6638cea6dc`.

This report distinguishes real backend observations from deterministic and
injected-failure tests. Live availability is specific to the tested ChatGPT
account/backend, not a promise about every account or the public OpenAI API.
No credentials, account identifiers, private paths, or raw conversations are
included. Runtime credentials and OpenCode sessions stay outside the PR.

## Results

| Verification | Result |
| --- | --- |
| `uv run ruff check gptmock/ tests/` | Passed |
| `uv run pytest tests/ --no-testmon -q` | 335 passed, 114 skipped |
| `uv build` | Wheel and source distribution built |
| OpenCode 1.17.18, temporary loopback GPTMock server | 18/18 scenarios passed, including expected rejection and synthetic failures |
| Ollama Python SDK 0.6.2, native chat tools | 2/2 streaming/non-streaming tool-and-result loops passed |
| Direct Chat API, strict named long tools | 2/2 streaming/non-streaming calls and follow-up answers passed |
| Docker Engine 29.1.3, Linux amd64 image | Build and hardened runtime smoke checks passed |
| Original and local Compose configurations | Syntax validation passed; original used `--no-env-resolution` |

The deterministic suite deliberately skips live integration tests and excludes
the opt-in `tests/tools/` directory. Its pass count does not mean that all live
models, clients, or tools were tested. The separately listed live tests used
existing authentication without publishing it or changing personal client
configuration. Successful model probes required `response.completed`, the
expected model/effort, and the requested marker or completed tool loop.

## OpenCode and protocol observations

- Responses and Chat adapters both completed Astra read-tool/result/final-answer
  loops. Read access was limited to a synthetic local fixture; editing, shell
  tools, delegation, and sharing were disabled.
- Astra completed at `low`, `medium`, `high`, `xhigh`, and `max`. Luna, Terra,
  and Sol retained their concrete upstream model IDs at `low`; Luna also
  completed at `none` and `max`.
- Astra fast requested `service_tier: priority`; both upstream and client
  responses retained the actual `default` tier. No priority execution is claimed.
- `gpt-6-astra-max` returned HTTP 400 without an upstream call. Explicit `max`
  reasoning on `gpt-6-astra` remained successful.
- OpenCode-added output limits were absent upstream and named in the
  `X-GPTMock-Omitted-Parameters` response header.
- Bare `[DONE]` on Responses, Chat read timeout, and incomplete Chat tool
  arguments were injected locally, without real upstream requests. All three
  surfaced an OpenCode error and exit code 1, without tool execution, timeout,
  or the harness's repetition guard. Earlier failing runs are not counted.
- Generated Responses stream errors include top-level code/message and the
  equivalent nested error expected by AI SDK 3.x. Neither form claims success.
- Native Ollama tool arguments were JSON objects, not strings/fragments, and
  follow-up tool results completed in both modes.
- Direct strict/named Chat tests verified the flattened upstream tool choice,
  consistent long-name mapping, restored client names, and follow-up history.
  This is direct API evidence, not an OpenCode strict-tool claim.

## Docker and regression scope

The image uses Python 3.13.15, uv 0.12.6, and UID/GID 10001. The repeatable
`tests/docker_smoke.py` check runs in a read-only container with all capabilities
dropped, `no-new-privileges`, a PID limit, and writable temporary credential
storage. It verifies auth-file mode 0600, serialized Linux file locks, configured
API authentication, default-blocked CORS, per-app settings, and discovery.
Both CI verification workflows invoke it before publishing can occur.

A separate synthetic root-owned Linux Docker volume reproduced the startup
permission error. After changing only that test volume's ownership to 10001,
storage validation and credential writes passed. A temporary loopback-bound
CLI server returned health 200, unauthenticated models 401, authenticated models
200, and removed-alias 400. Test containers and the synthetic volume were removed.
This does not claim fresh interactive OAuth or live model calls inside Docker;
those live protocol checks ran on the host. Windows bind-mount ACLs and network
filesystems require deployment-specific validation.

Additional regressions cover concurrent refresh single-flight, multiprocess
locking and cancellation, CLI/environment precedence, separate app settings,
rate-limit recording, out-of-order Ollama tool results, transport failures,
and response-vs-item terminal state handling.

## Preservation and limitations

- `docker-compose.yml` is byte-identical to upstream (Git blob
  `3fda26da52e6c3c358b910d8000eadf9126983da`). Hardened deployment remains
  opt-in through the separate `docker-compose.local.yml`.
- Original maintainer attribution, LICENSE, and package author/project metadata
  are retained. Authentication remains optional for loopback use; configure an
  API key before exposing the service. CORS is disabled by default.
- Rejected Pro names/mode were not re-probed in this pass and remain unregistered.
  Terra/Sol's full effort matrix, all gateways/frontends, native Ollama CLI,
  fresh OAuth login, and concurrent manual re-login were not tested here.
- A current-tree secret/PII check is not a certification that the entire Git
  history is free of personal information. Historical commits retain author
  emails and a previously removed contributor instruction file. History was
  not rewritten; raw local validation data is excluded from Git and Docker context.
