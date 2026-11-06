#!/usr/bin/env bash
set -eo pipefail

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source /opt/tiago_ws/install/setup.bash
# shellcheck disable=SC1091
source /opt/grasp_ws/install/setup.bash
set -u

echo "Waiting for TIAGo controllers before starting the optional grasp heuristic..."
for _ in $(seq 1 120); do
  controllers=$(timeout 5s ros2 control list_controllers 2>/dev/null || true)
  if grep -Eq '^arm_controller[[:space:]].*[[:space:]]active$' <<<"${controllers}" \
      && grep -Eq '^gripper_controller[[:space:]].*[[:space:]]active$' <<<"${controllers}"; then
    # Let the upstream tuck motion finish before this process starts sending
    # goals to the same arm controller. This one-time delay avoids starting
    # MoveIt from the raw spawn configuration.
    sleep 30
    ros2 param set /arm_controller constraints.goal_time 15.0 >/dev/null
    exec ros2 launch tiago_grasp_dataset run_heuristic.launch.py \
      use_sim_time:=True \
      base_type:=pmb2 \
      arm_type:=tiago-arm \
      end_effector:=pal-gripper \
      output_dir:="${HEURISTIC_OUTPUT_DIR:-/data/grasp_dataset}" \
      seed:="${HEURISTIC_SEED:-0}" \
      example_count:="${HEURISTIC_EPISODES:-1}" \
      finger_pad_z_correction:="${HEURISTIC_FINGER_PAD_Z_CORRECTION:-0.0}" \
      lift_distance:="${HEURISTIC_LIFT_DISTANCE:-0.05}" \
      speed_scale:="${HEURISTIC_SPEED_SCALE:-1.0}" \
      record_diagnostics:="${HEURISTIC_RECORD_DIAGNOSTICS:-true}"
  fi
  sleep 2
done

echo "The heuristic stopped because the arm and gripper controllers did not become active." >&2
exit 1
