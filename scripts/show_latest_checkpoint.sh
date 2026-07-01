#!/usr/bin/env bash
# Print the latest deployed best-checkpoint Hub repo id (or exit 1).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$PROJECT_ROOT/.env"
  set +a
fi

cd "$SCRIPT_DIR"
exec python - <<'PY'
import checkpoint_utils
import config

repo = checkpoint_utils.find_latest_hub_best_repo(config.HUB_BEST_REPO_PREFIX)
if not repo:
    raise SystemExit(
        "No Hub best checkpoint found. Authenticate with HF and ensure repos exist under "
        f"{config.HUB_BEST_REPO_PREFIX}-step*"
    )
print(repo)
PY
