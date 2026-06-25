#!/usr/bin/env bash
# Start MolmoAct2 training detached with nohup (survives IDE/session disconnects).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=train_process.sh
source "$SCRIPT_DIR/train_process.sh"

OUTPUT_ROOT="${MOLMOACT2_OUTPUT_DIR:-/tmp/molmoact2-record-test/outputs}"
LOG_FILE="$OUTPUT_ROOT/train.log"
PID_FILE="$OUTPUT_ROOT/train.pid"
HF_TOKEN_FILE="${HOME}/.cache/huggingface/token"
TRAIN_LOCK="/tmp/molmoact2-record-test.train.lock"

mkdir -p "$OUTPUT_ROOT"
export MOLMOACT2_OUTPUT_DIR="$OUTPUT_ROOT"
export MOLMOACT2_TRAIN_LOCK="$TRAIN_LOCK"

if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
fi

if [[ "${FORCE_RESTART:-0}" == "1" ]]; then
  stop_train_processes
else
  assert_single_training
fi

# Stale PID file from a crashed run.
if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE")"
  if ! kill -0 "$OLD_PID" 2>/dev/null; then
    rm -f "$PID_FILE"
  fi
fi

wait_for_gpu_idle

cd "$SCRIPT_DIR"
nohup env \
  MOLMOACT2_OUTPUT_DIR="$OUTPUT_ROOT" \
  MOLMOACT2_TRAIN_LOCK="$TRAIN_LOCK" \
  HF_TOKEN="${HF_TOKEN:-}" \
  python train.py >> "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
disown "$TRAIN_PID" 2>/dev/null || true

echo "Training started in background (PID $TRAIN_PID)"
echo "Log:  tail -f $LOG_FILE"
echo "Stop: ./stop_train.sh"
