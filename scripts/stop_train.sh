#!/usr/bin/env bash
# Stop background molmoact2 training and release GPU / lock files.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=train_process.sh
source "$SCRIPT_DIR/train_process.sh"

stop_train_processes
echo "Done."
