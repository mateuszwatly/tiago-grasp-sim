#!/usr/bin/env bash
set -eo pipefail

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source /opt/tiago_ws/install/setup.bash
# shellcheck disable=SC1091
source /opt/grasp_ws/install/setup.bash
set -u

session_name="${1:-grasp_$(date -u +%Y%m%dT%H%M%SZ)}"
output="/data/rosbags/${session_name}"

if [[ ! "${session_name}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Session name may contain only letters, digits, dots, underscores, and hyphens." >&2
  exit 1
fi

if [[ -e "${output}" ]]; then
  echo "Recording path already exists: ${output}" >&2
  exit 1
fi

echo "Recording TIAGo observations and commands to ${output}"
exec ros2 bag record \
  --storage mcap \
  --storage-preset-profile zstd_fast \
  --output "${output}" \
  --regex '^/(clock|joint_states|tf|tf_static|robot_description|tiago_grasp_sim/object|head_front_camera/(camera_info|image/compressed|depth_image/zstd|points)|arm_controller/.*|gripper_controller/.*|torso_controller/.*)$'
