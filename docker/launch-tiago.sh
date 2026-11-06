#!/usr/bin/env bash
set -euo pipefail

if [[ "${TIAGO_HEADLESS:-false}" == "true" ]]; then
  export DISPLAY=:99
  export LIBGL_ALWAYS_SOFTWARE=1
  Xvfb :99 -screen 0 1280x800x24 -nolisten tcp &
fi

start-tabletop-scene &
(
  until gz topic -l 2>/dev/null | grep -qx '/external_camera/image'; do
    sleep 1
  done
  ros2 run ros_gz_bridge parameter_bridge \
    '/external_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo' &
  exec ros2 run ros_gz_image image_bridge /external_camera/image
) &

if [[ "${TIAGO_RUN_HEURISTIC:-false}" == "true" ]]; then
  run-grasp-heuristic &
fi

tuck_arm=True
if [[ "${TIAGO_RUN_HEURISTIC:-false}" == "true" ]]; then
  tuck_arm=False
fi

exec ros2 launch tiago_gazebo tiago_gazebo.launch.py \
  is_public_sim:=True \
  world_name:="${TIAGO_WORLD:-empty}" \
  base_type:=pmb2 \
  arm_type:=tiago-arm \
  end_effector:=pal-gripper \
  camera_model:=orbbec-astra \
  tuck_arm:="${tuck_arm}" \
  moveit:=True
