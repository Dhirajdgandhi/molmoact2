#!/usr/bin/env bash
# Shared helpers to ensure only one molmoact2 training job uses the GPU.

list_train_pids() {
  # Match train.py launched from this project (cwd or absolute path).
  { pgrep -f "python.*train\.py" 2>/dev/null || true; } | while read -r pid; do
    if [[ -r "/proc/${pid}/cwd" ]]; then
      cwd="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null || true)"
      if [[ "$cwd" == "$SCRIPT_DIR" ]]; then
        echo "$pid"
      fi
    fi
  done
}

stop_train_processes() {
  local pids pid
  pids="$(list_train_pids | tr '\n' ' ')"
  if [[ -z "${pids// /}" ]]; then
    echo "No molmoact2 training processes found."
    cleanup_train_artifacts
    return 0
  fi

  echo "Stopping training process(es): ${pids}"
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true

  local waited=0
  while (( waited < 30 )); do
    pids="$(list_train_pids | tr '\n' ' ')"
    [[ -z "${pids// /}" ]] && break
    sleep 1
    waited=$((waited + 1))
  done

  pids="$(list_train_pids | tr '\n' ' ')"
  if [[ -n "${pids// /}" ]]; then
    echo "Force-killing: ${pids}"
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
    sleep 2
  fi

  cleanup_train_artifacts
}

cleanup_train_artifacts() {
  rm -f /tmp/molmoact2-record-test.train.lock
  rm -f /tmp/molmoact2-record-test/outputs/train.pid
  rm -f "${HOME}/molmoact2-record-test/outputs/train.pid" 2>/dev/null || true

  if command -v nvidia-smi &>/dev/null; then
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
    if [[ -n "$used" && "$used" -gt 500 ]]; then
      echo "GPU memory still in use: ${used} MiB (may clear in a few seconds)."
    fi
  fi
}

wait_for_gpu_idle() {
  if ! command -v nvidia-smi &>/dev/null; then
    return 0
  fi

  local max_wait=30 waited=0 used
  while (( waited < max_wait )); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
    if [[ -z "$used" || "$used" -lt 500 ]]; then
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done

  echo "Warning: GPU still has ${used} MiB allocated after ${max_wait}s."
  echo "Another process may still be using the GPU."
}

assert_single_training() {
  local pids
  pids="$(list_train_pids | tr '\n' ' ')"
  if [[ -n "${pids// /}" ]]; then
    echo "Training already running (PIDs: ${pids})."
    echo "Log: ${LOG_FILE:-<see run_train.sh>}"
    echo "Stop: ./stop_train.sh"
    echo "Restart: FORCE_RESTART=1 ./run_train.sh"
    exit 0
  fi
}
