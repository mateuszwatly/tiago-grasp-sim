#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/jazzy/setup.bash
source /opt/tiago_ws/install/setup.bash
source /opt/grasp_ws/install/setup.bash

session="calibration_$(date -u +%Y%m%dT%H%M%SZ)"
output="/data/rosbags/${session}"
mkdir -p "${output}"
exec >"${output}/capture.log" 2>&1
echo "Starting dense calibration capture ${session}"

controllers_ready=false
for _ in $(seq 1 120); do
  controllers=$(timeout 5s ros2 control list_controllers 2>/dev/null || true)
  if grep -Eq '^arm_controller[[:space:]].*[[:space:]]active$' <<<"${controllers}" \
      && grep -Eq '^gripper_controller[[:space:]].*[[:space:]]active$' <<<"${controllers}"; then
    controllers_ready=true
    break
  fi
  sleep 2
done
if [[ "${controllers_ready}" != "true" ]]; then
  echo "Arm and gripper controllers did not become active" >&2
  exit 1
fi

sleep 30
ros2 bag record \
  --storage mcap \
  --storage-preset-profile zstd_fast \
  --output "${output}/sensors" \
  --regex '^/(clock|joint_states|tf|tf_static|robot_description|tiago_grasp_sim/object|head_front_camera/(camera_info|image/compressed|depth_image/zstd|points)|arm_controller/.*|gripper_controller/.*|torso_controller/.*)$' \
  >"${output}/rosbag.log" 2>&1 &
bag_pid=$!

cleanup() {
  kill -INT "${bag_pid}" 2>/dev/null || true
  wait "${bag_pid}" 2>/dev/null || true
}
trap cleanup EXIT
sleep 3

ros2 run tiago_grasp_dataset capture_calibration \
  --ros-args -p output_dir:="${output}"

cleanup
trap - EXIT
