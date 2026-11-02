#!/usr/bin/env bash
set -eo pipefail

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source /opt/tiago_ws/install/setup.bash
# shellcheck disable=SC1091
source /opt/grasp_ws/install/setup.bash
set -u

scene_args=(
  --ros-args
  -p use_sim_time:=true
  -p seed:="${SCENE_SEED:-0}"
  -p clutter_min_objects:="${SCENE_CLUTTER_MIN_OBJECTS:-4}"
  -p clutter_max_objects:="${SCENE_CLUTTER_MAX_OBJECTS:-7}"
  -p spawn_object_on_start:="${SCENE_SPAWN_OBJECT:-true}"
)
if [[ -n "${SCENE_OBJECT_TYPE:-}" ]]; then
  scene_args+=(-p object_type:="${SCENE_OBJECT_TYPE}")
fi

exec ros2 run tiago_grasp_dataset scene_manager "${scene_args[@]}"
