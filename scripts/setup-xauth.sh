#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
xauth_file="${repo_dir}/.docker.xauth"
display="${DISPLAY:-:0}"
target_uid="${USER_ID:-${SUDO_UID:-$(id -u)}}"
target_gid="${GROUP_ID:-${SUDO_GID:-$(id -g)}}"

if ! command -v xauth >/dev/null 2>&1; then
  echo "xauth is required. Install it with: sudo apt install xauth" >&2
  exit 1
fi

cookie=$(xauth nlist "${display}" 2>/dev/null | sed -e 's/^..../ffff/' | head -n 1)
if [[ -z "${cookie}" ]]; then
  echo "No Xauthority cookie found for DISPLAY=${display}." >&2
  echo "Log in to the graphical session or set TIAGO_HEADLESS=true in .env." >&2
  exit 1
fi

temporary_file=$(mktemp "${repo_dir}/.docker.xauth.XXXXXX")
trap 'rm -f "${temporary_file}"' EXIT
xauth -b -f "${temporary_file}" nmerge - <<<"${cookie}"
chown "${target_uid}:${target_gid}" "${temporary_file}"
chmod 600 "${temporary_file}"
mv -f "${temporary_file}" "${xauth_file}"
trap - EXIT
echo "Prepared ${xauth_file} for DISPLAY=${display}"
