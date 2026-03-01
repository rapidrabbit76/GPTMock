#!/usr/bin/env bash
set -euo pipefail

export GPTMOCK_HOME="${GPTMOCK_HOME:-${CHATGPT_LOCAL_HOME:-/data}}"
export CHATGPT_LOCAL_HOME="${CHATGPT_LOCAL_HOME:-${GPTMOCK_HOME}}"

cmd="${1:-serve}"
shift || true

bool() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0;;
    *) return 1;;
  esac
}

if [[ "$cmd" == "serve" ]]; then
  PORT="${GPTMOCK_PORT:-${PORT:-8000}}"
  export GPTMOCK_PORT="${PORT}"
  ARGS=(serve --host 0.0.0.0 --port "${PORT}")

  if bool "${GPTMOCK_VERBOSE:-${VERBOSE:-${CHATGPT_LOCAL_VERBOSE:-}}}"; then
    ARGS+=(--verbose)
  fi
  if bool "${GPTMOCK_VERBOSE_OBFUSCATION:-${VERBOSE_OBFUSCATION:-${CHATGPT_LOCAL_VERBOSE_OBFUSCATION:-}}}"; then
    ARGS+=(--verbose-obfuscation)
  fi

  if [[ "$#" -gt 0 ]]; then
    ARGS+=("$@")
  fi

  exec gptmock "${ARGS[@]}"
elif [[ "$cmd" == "login" ]]; then
  ARGS=(login --no-browser)
  if bool "${GPTMOCK_VERBOSE:-${VERBOSE:-${CHATGPT_LOCAL_VERBOSE:-}}}"; then
    ARGS+=(--verbose)
  fi

  exec gptmock "${ARGS[@]}"
else
  exec "$cmd" "$@"
fi
