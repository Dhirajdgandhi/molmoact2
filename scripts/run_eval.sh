#!/usr/bin/env bash
# Run offline val evaluation (teacher-forcing loss + open-loop action MSE).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

"$SCRIPT_DIR/sync_secrets.sh" >/dev/null 2>&1 || true

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

if ! python -c "from huggingface_hub import HfApi; HfApi().whoami()" >/dev/null 2>&1; then
  echo "Hugging Face not authenticated."
  echo "  hf auth login"
  echo "  or set HF_TOKEN in .env and run ./scripts/sync_secrets.sh"
  exit 1
fi

export MOLMOACT2_OUTPUT_DIR="${MOLMOACT2_OUTPUT_DIR:-/tmp/molmoact2-record-test/outputs}"
export HF_HOME="${HF_HOME:-/tmp/huggingface}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONUNBUFFERED=1
mkdir -p "$MOLMOACT2_OUTPUT_DIR" "$PROJECT_ROOT/outputs/eval_runs" "$HF_HOME" "${HF_HOME}/hub"

if [[ $# -eq 0 && -z "${MOLMOACT2_CHECKPOINT:-}" ]]; then
  echo "No --checkpoint passed; auto-selecting latest Hub best (dhirajdg/molmoact2-record-test-step*)."
fi

cd "$SCRIPT_DIR"
exec python eval_offline.py "$@"
