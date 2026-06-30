#!/usr/bin/env bash
# Push commits to GitHub using credentials from project .env (non-interactive).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

"$SCRIPT_DIR/sync_secrets.sh" >/dev/null

if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$PROJECT_ROOT/.env"
  set +a
fi

if [[ -z "${GIT_USERNAME:-}" || -z "${GIT_TOKEN:-}" ]]; then
  echo "Set GIT_USERNAME and GIT_TOKEN in $PROJECT_ROOT/.env (copy from .env.example)." >&2
  echo "GIT_TOKEN must be a GitHub Personal Access Token with repo scope." >&2
  exit 1
fi

REMOTE_URL="$(git -C "$PROJECT_ROOT" remote get-url origin)"
case "$REMOTE_URL" in
  https://github.com/*)
    REPO_PATH="${REMOTE_URL#https://github.com/}"
    ;;
  git@github.com:*)
    REPO_PATH="${REMOTE_URL#git@github.com:}"
    ;;
  *)
    echo "Unsupported origin URL: $REMOTE_URL" >&2
    exit 1
    ;;
esac
REPO_PATH="${REPO_PATH%.git}"

# One-shot authenticated URL; credentials are not written to git config.
AUTH_URL="https://${GIT_USERNAME}:${GIT_TOKEN}@github.com/${REPO_PATH}.git"

cd "$PROJECT_ROOT"
if [[ $# -gt 0 ]]; then
  git push "$AUTH_URL" "$@"
else
  git push "$AUTH_URL"
fi
