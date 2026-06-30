#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/sync_secrets.sh" || true

cd "$SCRIPT_DIR"
exec python eval_offline.py "$@"
