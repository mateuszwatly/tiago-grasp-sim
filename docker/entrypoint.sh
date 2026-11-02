#!/usr/bin/env bash
set -e

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source /opt/tiago_ws/install/setup.bash
# shellcheck disable=SC1091
source /opt/grasp_ws/install/setup.bash

exec "$@"
