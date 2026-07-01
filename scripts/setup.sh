#!/usr/bin/env bash
# One-time (or repeat on fresh SageMaker) setup: deps, dirs, credentials.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
  echo "Tip: copy .env.example to .env and set HF_TOKEN before running eval on private checkpoints."
fi

"$SCRIPT_DIR/sync_secrets.sh" || true

pip install -r "$SCRIPT_DIR/requirements.txt"

mkdir -p \
  "$PROJECT_ROOT/outputs/eval_runs" \
  "${MOLMOACT2_OUTPUT_DIR:-/tmp/molmoact2-record-test/outputs}" \
  "${HF_HOME:-/tmp/huggingface}/hub"

if python -c "from huggingface_hub import HfApi; print('HF user:', HfApi().whoami()['name'])" 2>/dev/null; then
  echo "Hugging Face: authenticated"
else
  echo "Hugging Face: not authenticated — run 'hf auth login' or set HF_TOKEN in .env"
fi

echo ""
echo "Ready. Examples:"
echo "  ./scripts/run_eval.sh                    # latest Hub best checkpoint"
echo "  ./scripts/run_eval.sh --max-batches 5    # quick smoke test"
echo "  ./scripts/docker_eval.sh                 # Docker (after first build)"
