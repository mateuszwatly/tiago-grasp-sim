#!/usr/bin/env python3
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState, PointCloud2
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


ARM_JOINTS = [f"arm_{index}_joint" for index in range(1, 8)]
FINGER_JOINTS = ["gripper_left_finger_joint", "gripper_right_finger_joint"]
ARM_POSES = [
    ("table_center", [0.17, -0.27, -1.35, 2.28, 1.22, 0.92, 1.57]),
    ("table_left", [0.23, -0.31, -1.30, 2.24, 1.16, 0.98, 1.50]),
    ("table_right", [0.11, -0.23, -1.40, 2.30, 1.28, 0.86, 1.64]),
    ("table_low", [0.19, -0.33, -1.42, 2.31, 1.18, 0.90, 1.58]),
    ("table_center_return", [0.17, -0.27, -1.35, 2.28, 1.22, 0.92, 1.57]),
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class CalibrationCapture(Node):
    def __init__(self):
        super().__init__(
            "tiago_calibration_capture",
            automatically_declare_parameters_from_overrides=True,
        )
        self.output_dir = Path(str(self.get_parameter("output_dir").value))
        self.arm_client = ActionClient(
            self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory"
        )
        self.gripper_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/gripper_controller/follow_joint_trajectory",
        )
        self.latest_joints = None
        self.cloud = {"messages": 0, "width": 0, "height": 0, "points": 0}
        self.create_subscription(JointState, "/joint_states", self.on_joints, 20)
        self.create_subscription(
            PointCloud2, "/head_front_camera/points", self.on_cloud, 2
        )
        self.manifest = {
            "schema_version": 1,
            "created_at": utc_now(),
            "purpose": "dense_rgbd_gripper_workspace_calibration",
            "point_cloud_topic": "/head_front_camera/points",
            "phases": [],
        }

    def on_joints(self, message):
        self.latest_joints = dict(zip(message.name, message.position))

    def on_cloud(self, message):
        self.cloud = {
            "messages": self.cloud["messages"] + 1,
            "width": int(message.width),
            "height": int(message.height),
            "points": int(message.width * message.height),
            "point_step": int(message.point_step),
            "frame_id": message.header.frame_id,
        }

    def send(self, client, phase, joints, positions, seconds):
        if not client.wait_for_server(timeout_sec=20.0):
            raise RuntimeError(f"Controller is unavailable for {phase}")
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in positions]
        point.time_from_start = Duration(sec=int(seconds))
        goal = FollowJointTrajectory.Goal(
            trajectory=JointTrajectory(joint_names=joints, points=[point])
        )
        sent = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, sent, timeout_sec=10.0)
        handle = sent.result()
        if handle is None or not handle.accepted:
            raise RuntimeError(f"Controller rejected {phase}")
        result = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result, timeout_sec=seconds + 15.0)
        outcome = result.result()
        if outcome is None:
            raise RuntimeError(f"Controller timed out during {phase}")
        record = {
            "phase": phase,
            "wall_time": utc_now(),
            "joint_names": joints,
            "commanded_positions": positions,
            "controller_error_code": int(outcome.result.error_code),
            "cloud": dict(self.cloud),
        }
        if self.latest_joints is not None:
            record["measured_positions"] = {
                name: float(self.latest_joints[name])
                for name in joints
                if name in self.latest_joints
            }
        self.manifest["phases"].append(record)
        self.write_manifest()
        time.sleep(2.0)

    def write_manifest(self):
        (self.output_dir / "calibration_manifest.json").write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

    def run(self):
        deadline = time.monotonic() + 30.0
        while (self.latest_joints is None or self.cloud["messages"] < 3) and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
        if self.latest_joints is None or self.cloud["messages"] < 3:
            raise RuntimeError("Joint state or dense point cloud did not become available")
        self.manifest["initial_cloud"] = dict(self.cloud)
        self.write_manifest()
        self.send(self.gripper_client, "fingers_open", FINGER_JOINTS, [0.044, 0.044], 2)
        for name, positions in ARM_POSES:
            self.send(self.arm_client, name, ARM_JOINTS, positions, 8)
            opening = 0.020 if name in {"table_left", "table_low"} else 0.044
            self.send(
                self.gripper_client,
                f"fingers_{name}",
                FINGER_JOINTS,
                [opening, opening],
                2,
            )
        self.send(self.gripper_client, "fingers_final_open", FINGER_JOINTS, [0.044, 0.044], 2)
        self.manifest["completed_at"] = utc_now()
        self.manifest["final_cloud"] = dict(self.cloud)
        self.manifest["status"] = "completed"
        self.write_manifest()


def main():
    rclpy.init()
    node = CalibrationCapture()
    try:
        node.run()
    except Exception as error:
        node.manifest["status"] = "failed"
        node.manifest["error"] = str(error)
        node.write_manifest()
        raise
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
