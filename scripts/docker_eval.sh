#!/usr/bin/env bash
# Build (if needed) and run offline eval in Docker with cached HF downloads.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE="${MOLMOACT2_DOCKER_IMAGE:-molmoact2-eval}"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$PROJECT_ROOT/.env"
  set +a
fi

if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN required. Set it in .env or run ./scripts/sync_secrets.sh after hf auth login."
  exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Building $IMAGE (first time only)..."
  docker build -t "$IMAGE" "$PROJECT_ROOT"
fi

mkdir -p "$PROJECT_ROOT/outputs/eval_runs"

exec docker run --rm --gpus all \
  -e HF_TOKEN \
  -e MOLMOACT2_CHECKPOINT="${MOLMOACT2_CHECKPOINT:-}" \
  -v molmoact2-hf-cache:/tmp/huggingface \
  -v "$PROJECT_ROOT/outputs:/app/outputs" \
  "$IMAGE" \
  "$@"
