FROM osrf/ros:jazzy-desktop-full@sha256:ae7ad3ac243da1dfd8bd402a2fa08e149ffbf7a385013bd8ef798d0800accdc8

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG USER_ID=1000
ARG GROUP_ID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    GZ_VERSION=harmonic \
    ROS_WS=/opt/tiago_ws

RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      mesa-utils \
      python3-colcon-common-extensions \
      python3-rosdep \
      python3-vcstool \
      ros-jazzy-gz-ros2-control \
      ros-jazzy-moveit-py \
      ros-jazzy-ros-gz \
      ros-jazzy-ros2-controllers \
      ros-jazzy-rosbag2-storage-mcap \
      xauth \
      xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${ROS_WS}
COPY docker/tiago.repos /tmp/tiago.repos

RUN mkdir -p src \
    && vcs import --recursive src < /tmp/tiago.repos \
    && rm /tmp/tiago.repos \
    && rosdep update --rosdistro jazzy \
    && source /opt/ros/jazzy/setup.bash \
    && apt-get update \
    && rosdep install --from-paths src --ignore-src --rosdistro jazzy -r -y \
    && rm -rf /var/lib/apt/lists/* \
    && colcon build \
         --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release \
         --event-handlers console_direct+ \
    && rm -rf build log

# The stock simulated Astra publishes 640x480. Use a dense organized cloud for
# calibration captures while reducing the rate enough to keep recording stable.
RUN camera_xacro="${ROS_WS}/install/tiago_description/share/tiago_description/urdf/sensors/openni.gazebo.xacro" \
    && sed -i 's#<update_rate>30.0</update_rate>#<update_rate>10.0</update_rate>#' "${camera_xacro}" \
    && sed -i 's#<width>640</width>#<width>1280</width>#' "${camera_xacro}" \
    && sed -i 's#<height>480</height>#<height>960</height>#' "${camera_xacro}"

COPY ros_ws/src/tiago_grasp_dataset /opt/grasp_ws/src/tiago_grasp_dataset
RUN source /opt/ros/jazzy/setup.bash \
    && source /opt/tiago_ws/install/setup.bash \
    && cd /opt/grasp_ws \
    && colcon build --packages-select tiago_grasp_dataset \
         --event-handlers console_direct+ \
    && rm -rf build log

RUN existing_user="$(getent passwd "${USER_ID}" | cut -d: -f1 || true)" \
    && if [[ -n "${existing_user}" ]]; then userdel --remove "${existing_user}"; fi \
    && existing_group="$(getent group "${GROUP_ID}" | cut -d: -f1 || true)" \
    && if [[ -n "${existing_group}" ]]; then groupmod --new-name tiago "${existing_group}"; else groupadd --gid "${GROUP_ID}" tiago; fi \
    && useradd --uid "${USER_ID}" --gid "${GROUP_ID}" --create-home --shell /bin/bash tiago \
    && mkdir -p /data/rosbags /data/grasp_dataset \
    && chown -R tiago:tiago /data/rosbags /data/grasp_dataset /home/tiago

COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/tiago-entrypoint
COPY --chmod=755 docker/launch-tiago.sh /usr/local/bin/launch-tiago
COPY --chmod=755 docker/start-tabletop-scene.sh /usr/local/bin/start-tabletop-scene
COPY --chmod=755 docker/run-grasp-heuristic.sh /usr/local/bin/run-grasp-heuristic
COPY --chmod=755 docker/record-dataset.sh /usr/local/bin/record-dataset
COPY --chmod=755 docker/run-calibration-capture.sh /usr/local/bin/run-calibration-capture

ENV DEBIAN_FRONTEND= \
    HOME=/home/tiago \
    ROS_DOMAIN_ID=42 \
    TIAGO_WORLD=empty \
    TIAGO_HEADLESS=false \
    TIAGO_RUN_HEURISTIC=false \
    SCENE_SEED=0 \
    SCENE_SPAWN_OBJECT=true \
    SCENE_OBJECT_TYPE= \
    SCENE_CLUTTER_MIN_OBJECTS=4 \
    SCENE_CLUTTER_MAX_OBJECTS=7 \
    HEURISTIC_SEED=0 \
    HEURISTIC_EPISODES=1 \
    HEURISTIC_OUTPUT_DIR=/data/grasp_dataset \
    HEURISTIC_RECORD_DIAGNOSTICS=true \
    HEURISTIC_FINGER_PAD_Z_CORRECTION=0.0 \
    HEURISTIC_LIFT_DISTANCE=0.05 \
    HEURISTIC_SPEED_SCALE=1.0

USER tiago
WORKDIR /home/tiago
ENTRYPOINT ["/usr/local/bin/tiago-entrypoint"]
CMD ["launch-tiago"]
