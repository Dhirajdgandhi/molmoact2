#!/usr/bin/env bash
# Run offline val evaluation (teacher-forcing loss + open-loop action MSE).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$PROJECT_ROOT/.env"
  set +a
fi

HF_TOKEN_FILE="${HOME}/.cache/huggingface/token"
if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
fi

export MOLMOACT2_OUTPUT_DIR="${MOLMOACT2_OUTPUT_DIR:-/tmp/molmoact2-record-test/outputs}"
export HF_HOME="${HF_HOME:-/tmp/huggingface}"
mkdir -p "$MOLMOACT2_OUTPUT_DIR" "$HF_HOME" "${HF_HOME}/hub"

cd "$SCRIPT_DIR"
exec python eval_offline.py "$@"
