#!/usr/bin/env python3
import json
from pathlib import Path

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def to_duration(seconds):
    whole_seconds = int(seconds)
    return Duration(
        sec=whole_seconds,
        nanosec=int((seconds - whole_seconds) * 1_000_000_000),
    )


class EpisodeReplayer(Node):
    def __init__(self):
        super().__init__(
            "grasp_episode_replayer",
            automatically_declare_parameters_from_overrides=True,
        )
        episode_file = str(self.get_parameter("episode_file").value)
        if not episode_file:
            raise ValueError("Set the episode_file ROS parameter")
        self.episode_path = Path(episode_file)
        self.episode = json.loads(self.episode_path.read_text(encoding="utf-8"))
        self.clients = {
            "arm_controller": ActionClient(
                self,
                FollowJointTrajectory,
                "/arm_controller/follow_joint_trajectory",
            ),
            "gripper_controller": ActionClient(
                self,
                FollowJointTrajectory,
                "/gripper_controller/follow_joint_trajectory",
            ),
        }

    def send(self, controller, joint_names, points):
        client = self.clients[controller]
        if not client.wait_for_server(timeout_sec=20.0):
            raise RuntimeError(f"Controller action is unavailable: {controller}")
        trajectory = JointTrajectory(joint_names=list(joint_names))
        for source in points:
            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in source["positions"]]
            point.velocities = [float(value) for value in source.get("velocities", [])]
            point.accelerations = [float(value) for value in source.get("accelerations", [])]
            point.effort = [float(value) for value in source.get("effort", [])]
            point.time_from_start = to_duration(float(source["time_from_start"]))
            trajectory.points.append(point)
        goal = FollowJointTrajectory.Goal(trajectory=trajectory)
        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError(f"Trajectory was rejected by {controller}")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        if result is None or result.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(f"Trajectory execution failed on {controller}")

    def move_to_recorded_start(self):
        segments = self.episode.get("planned_segments", [])
        arm_segment = next(
            segment for segment in segments if segment["controller"] == "arm_controller"
        )
        first_point = arm_segment["points"][0]
        self.send(
            "arm_controller",
            arm_segment["joint_names"],
            [
                {
                    "positions": first_point["positions"],
                    "velocities": [],
                    "accelerations": [],
                    "effort": [],
                    "time_from_start": 4.0,
                }
            ],
        )

    def replay(self):
        if self.episode.get("status") != "completed":
            raise ValueError("Only completed episodes can be replayed")
        self.get_logger().info(f"Replaying {self.episode_path}")
        preparation = self.episode.get("preparation_segment")
        if preparation is not None:
            self.send(
                preparation["controller"],
                preparation["joint_names"],
                preparation["points"],
            )
        self.move_to_recorded_start()
        for segment in self.episode["planned_segments"]:
            self.get_logger().info(f"Executing phase {segment['phase']}")
            self.send(segment["controller"], segment["joint_names"], segment["points"])
        self.get_logger().info("Replay completed without a lift command")


def main():
    rclpy.init()
    node = None
    try:
        node = EpisodeReplayer()
        node.replay()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
