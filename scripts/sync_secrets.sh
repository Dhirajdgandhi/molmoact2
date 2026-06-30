#!/usr/bin/env bash
# Persist HF + Git credentials from the environment or project .env to local storage.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$PROJECT_ROOT/.env"
  set +a
fi

# Shell environment overrides .env when both are set.
HF_TOKEN="${HF_TOKEN:-}"
GIT_USERNAME="${GIT_USERNAME:-}"
GIT_TOKEN="${GIT_TOKEN:-}"
GIT_USER_NAME="${GIT_USER_NAME:-}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-}"

write_env=false
if [[ -n "$HF_TOKEN" || -n "$GIT_USERNAME" || -n "$GIT_TOKEN" || -n "$GIT_USER_NAME" || -n "$GIT_USER_EMAIL" ]]; then
  write_env=true
  umask 077
  cat >"$PROJECT_ROOT/.env" <<EOF
HF_TOKEN=${HF_TOKEN}

GIT_USERNAME=${GIT_USERNAME}
GIT_TOKEN=${GIT_TOKEN}
GIT_USER_NAME="${GIT_USER_NAME}"
GIT_USER_EMAIL=${GIT_USER_EMAIL}
EOF
fi

if [[ -n "$HF_TOKEN" ]]; then
  HF_DIR="${HOME}/.cache/huggingface"
  mkdir -p "$HF_DIR"
  umask 077
  printf '%s' "$HF_TOKEN" >"$HF_DIR/token"
  chmod 600 "$HF_DIR/token"
  echo "Saved Hugging Face token to $HF_DIR/token"
fi

if [[ -n "$GIT_USERNAME" && -n "$GIT_TOKEN" ]]; then
  CRED_FILE="$PROJECT_ROOT/.git/credentials"
  umask 077
  printf 'https://%s:%s@github.com\n' "$GIT_USERNAME" "$GIT_TOKEN" >"$CRED_FILE"
  chmod 600 "$CRED_FILE"
  git -C "$PROJECT_ROOT" config credential.helper "store --file=$CRED_FILE"
  echo "Saved Git credentials for github.com (repo-local store)"
fi

if [[ -n "$GIT_USER_NAME" ]]; then
  git config --global user.name "$GIT_USER_NAME"
  echo "Configured git user.name"
fi

if [[ -n "$GIT_USER_EMAIL" ]]; then
  git config --global user.email "$GIT_USER_EMAIL"
  echo "Configured git user.email"
fi

if [[ "$write_env" == true ]]; then
  echo "Updated $PROJECT_ROOT/.env"
fi
