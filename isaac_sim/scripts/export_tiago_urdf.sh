#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
output_dir="${repo_root}/isaac_sim/assets/tiago_source"
source_image="${TIAGO_SOURCE_IMAGE:-tiago-grasp:jazzy-harmonic}"

mkdir -p "${output_dir}/packages"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --entrypoint bash \
  -v "${output_dir}:/out" \
  "${source_image}" -lc '
    source /opt/ros/jazzy/setup.bash
    source /opt/tiago_ws/install/setup.bash
    xacro_file=/opt/tiago_ws/install/tiago_description/share/tiago_description/robots/tiago.urdf.xacro
    xacro "${xacro_file}" \
      base_type:=pmb2 \
      laser_model:=no-laser \
      arm_type:=tiago-arm \
      ft_sensor:=schunk-ft \
      wrist_model:=wrist-2010 \
      end_effector:=pal-gripper \
      camera_model:=no-camera \
      is_public_sim:=true \
      use_sim_time:=true > /out/tiago_raw.urdf

    { grep -oE "package://[^/]+" /out/tiago_raw.urdf | cut -d/ -f3; \
      grep -oE "file:///opt/tiago_ws/install/[^/]+/share" /out/tiago_raw.urdf | cut -d/ -f7; } \
      | sort -u \
      | while read -r package_name; do
          package_prefix=$(ros2 pkg prefix "${package_name}")
          mkdir -p "/out/packages/${package_name}"
          cp -a "${package_prefix}/share/${package_name}/." "/out/packages/${package_name}/"
        done
  '

python3 "${repo_root}/isaac_sim/tools/sanitize_urdf.py" \
  "${output_dir}/tiago_raw.urdf" \
  "${output_dir}/tiago.urdf"
rm -f "${output_dir}/tiago_raw.urdf"

echo "Exported TIAGo URDF and meshes to ${output_dir}"
