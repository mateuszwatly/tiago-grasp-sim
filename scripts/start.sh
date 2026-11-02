#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_dir}"

mkdir -p datasets recordings

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ "${TIAGO_HEADLESS:-false}" != "true" ]]; then
  "${repo_dir}/scripts/setup-xauth.sh"
else
  touch .docker.xauth
fi

exec "${repo_dir}/scripts/compose.sh" up --build
