#!/usr/bin/env bash
set -eo pipefail

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source /opt/tiago_ws/install/setup.bash
# shellcheck disable=SC1091
source /opt/grasp_ws/install/setup.bash
set -u

exec ros2 run tiago_grasp_dataset scene_manager --ros-args \
  -p use_sim_time:=true \
  -p seed:="${SCENE_SEED:-0}" \
  -p object_type:="${SCENE_OBJECT_TYPE:-}" \
  -p spawn_object_on_start:="${SCENE_SPAWN_OBJECT:-true}"
