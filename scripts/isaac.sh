#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="$repo_root/.env.isaac"
compose_file="$repo_root/compose.isaac.yaml"

usage() {
  cat <<'EOF'
Usage: ./scripts/isaac.sh COMMAND

Commands:
  export-tiago  Export and sanitize TIAGo's URDF and mesh dependencies
  pull          Pull the pinned Isaac Sim container image
  start         Generate data with one Isaac Sim worker
  start-dual    Generate data with one worker on each of two GPUs
  stop          Stop and remove Isaac Sim workers
  logs          Follow logs from all enabled workers
EOF
}

command_name="${1:-}"

if [[ "$command_name" == "export-tiago" ]]; then
  exec "$repo_root/isaac_sim/scripts/export_tiago_urdf.sh"
fi

if [[ ! -f "$env_file" ]]; then
  echo "Missing $env_file. Copy .env.isaac.example to .env.isaac and explicitly accept the NVIDIA EULA." >&2
  exit 2
fi

if [[ "$command_name" == "start" || "$command_name" == "start-dual" ]]; then
  if ! grep -Eq '^ACCEPT_EULA=Y([[:space:]]*)$' "$env_file"; then
    echo "Isaac Sim was not started: .env.isaac must contain ACCEPT_EULA=Y after you review and accept NVIDIA's terms." >&2
    exit 2
  fi
fi

compose=(docker compose --env-file "$env_file" -f "$compose_file")
mkdir -p \
  "$repo_root/.isaac_cache"/{main,computecache,kit,logs,config,data} \
  "$repo_root/ready_datasets/isaac_clutter_pick"

case "$command_name" in
  pull)
    "${compose[@]}" pull
    ;;
  start)
    WORKER_COUNT=1 "${compose[@]}" up -d isaac-worker-0
    ;;
  start-dual)
    WORKER_COUNT=2 "${compose[@]}" --profile dual-gpu up -d
    ;;
  stop)
    "${compose[@]}" --profile dual-gpu down
    ;;
  logs)
    "${compose[@]}" --profile dual-gpu logs -f
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
