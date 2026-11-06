import os
from dataclasses import dataclass

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_pal.arg_utils import LaunchArgumentsBase, read_launch_argument
from launch_pal.robot_arguments import CommonArgs
from launch_ros.actions import Node
from tiago_description.launch_arguments import TiagoArgs
from tiago_description.tiago_launch_utils import get_tiago_hw_suffix
from moveit_configs_utils import MoveItConfigsBuilder


@dataclass(frozen=True)
class LaunchArguments(LaunchArgumentsBase):
    base_type: DeclareLaunchArgument = TiagoArgs.base_type
    arm_type: DeclareLaunchArgument = TiagoArgs.arm_type
    end_effector: DeclareLaunchArgument = TiagoArgs.end_effector
    ft_sensor: DeclareLaunchArgument = TiagoArgs.ft_sensor
    wrist_model: DeclareLaunchArgument = TiagoArgs.wrist_model
    camera_model: DeclareLaunchArgument = TiagoArgs.camera_model
    laser_model: DeclareLaunchArgument = TiagoArgs.laser_model
    use_sim_time: DeclareLaunchArgument = CommonArgs.use_sim_time


def generate_launch_description():
    description = LaunchDescription()
    arguments = LaunchArguments()
    arguments.add_to_launch_description(description)
    description.add_action(
        DeclareLaunchArgument("output_dir", default_value="/data/grasp_dataset")
    )
    description.add_action(DeclareLaunchArgument("seed", default_value="0"))
    description.add_action(DeclareLaunchArgument("example_count", default_value="1"))
    description.add_action(
        DeclareLaunchArgument("finger_pad_z_correction", default_value="0.0")
    )
    description.add_action(DeclareLaunchArgument("lift_distance", default_value="0.05"))
    description.add_action(DeclareLaunchArgument("speed_scale", default_value="1.0"))
    description.add_action(
        DeclareLaunchArgument("record_diagnostics", default_value="true")
    )
    description.add_action(OpaqueFunction(function=start_heuristic))
    return description


def start_heuristic(context, *args, **kwargs):
    base_type = read_launch_argument("base_type", context)
    arm_type = read_launch_argument("arm_type", context)
    end_effector = read_launch_argument("end_effector", context)
    ft_sensor = read_launch_argument("ft_sensor", context)
    use_sim_time = read_launch_argument("use_sim_time", context).lower() == "true"

    hardware_suffix = get_tiago_hw_suffix(
        arm=arm_type,
        end_effector=end_effector,
        ft_sensor=ft_sensor,
    )
    moveit_share = get_package_share_directory("tiago_moveit_config")
    srdf_path = os.path.join(moveit_share, "config", "srdf", "tiago.srdf.xacro")
    mappings = {
        "arm_type": arm_type,
        "end_effector": end_effector,
        "ft_sensor": ft_sensor,
        "base_type": base_type,
    }

    moveit_config = (
        MoveItConfigsBuilder("tiago")
        .robot_description_semantic(file_path=srdf_path, mappings=mappings)
        .robot_description_kinematics(file_path="config/kinematics_kdl.yaml")
        .trajectory_execution(
            file_path=f"config/controllers/controllers{hardware_suffix}.yaml"
        )
        .planning_scene_monitor(
            publish_planning_scene=True,
            publish_geometry_updates=True,
            publish_state_updates=True,
            publish_transforms_updates=True,
            publish_robot_description=True,
        )
        .joint_limits(file_path="config/joint_limits.yaml")
        .moveit_cpp(
            file_path=os.path.join(
                get_package_share_directory("tiago_grasp_dataset"),
                "config",
                "pipelines.yaml",
            )
        )
        .pilz_cartesian_limits(file_path="config/pilz_cartesian_limits.yaml")
        .to_moveit_configs()
    )

    return [
        Node(
            package="tiago_grasp_dataset",
            executable="run_heuristic",
            output="screen",
            parameters=[
                moveit_config.to_dict(),
                {
                    "use_sim_time": use_sim_time,
                    "output_dir": LaunchConfiguration("output_dir"),
                    "seed": LaunchConfiguration("seed"),
                    "example_count": LaunchConfiguration("example_count"),
                    "finger_pad_z_correction": LaunchConfiguration(
                        "finger_pad_z_correction"
                    ),
                    "lift_distance": LaunchConfiguration("lift_distance"),
                    "speed_scale": LaunchConfiguration("speed_scale"),
                    "record_diagnostics": LaunchConfiguration(
                        "record_diagnostics"
                    ),
                },
            ],
        )
    ]
