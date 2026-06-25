#!/usr/bin/env bash
# Start MolmoAct2 training detached with nohup (survives IDE/session disconnects).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="$PROJECT_ROOT/outputs/train.log"
PID_FILE="$PROJECT_ROOT/outputs/train.pid"

mkdir -p "$PROJECT_ROOT/outputs"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Training already running (PID $OLD_PID)."
    echo "Log: $LOG_FILE"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

cd "$SCRIPT_DIR"
nohup python train.py >> "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
disown "$TRAIN_PID" 2>/dev/null || true

echo "Training started in background (PID $TRAIN_PID)"
echo "Log:  tail -f $LOG_FILE"
echo "Stop: kill $TRAIN_PID"
