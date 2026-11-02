#!/usr/bin/env python3
import json
import math
import os
import random
import secrets
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from moveit_msgs.msg import CollisionObject
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from tiago_grasp_dataset.geometry_policy import GeometryContactPolicy

ARM_GROUP = "arm"
GRIPPER_GROUP = "gripper"
EE_FRAME = "gripper_grasping_frame"
# Scene object poses are expressed in Gazebo's world frame.  The stationary
# robot's base_footprint coincides with that frame; base_link is roughly 0.1 m
# above it and caused every Cartesian target to be executed 0.1 m too high.
BASE_FRAME = "base_footprint"
TRACKED_FRAMES = (
    EE_FRAME,
    "gripper_left_finger_link",
    "gripper_right_finger_link",
    "arm_7_link",
)
ARM_JOINTS = tuple(f"arm_{index}_joint" for index in range(1, 8))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def duration_seconds(duration):
    return float(duration.sec) + float(duration.nanosec) / 1_000_000_000.0


def duration_message(seconds):
    whole = int(seconds)
    return Duration(sec=whole, nanosec=int((seconds - whole) * 1_000_000_000))


def quaternion_from_matrix(matrix):
    rotation = np.asarray(matrix, dtype=float)[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / scale
        qx = 0.25 * scale
        qy = (rotation[0, 1] + rotation[1, 0]) / scale
        qz = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / scale
        qx = (rotation[0, 1] + rotation[1, 0]) / scale
        qy = 0.25 * scale
        qz = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / scale
        qx = (rotation[0, 2] + rotation[2, 0]) / scale
        qy = (rotation[1, 2] + rotation[2, 1]) / scale
        qz = 0.25 * scale
    return {"x": qx, "y": qy, "z": qz, "w": qw}


def quaternion_delta_rotvec(current, following):
    ax, ay, az, aw = (current[key] for key in ("x", "y", "z", "w"))
    bx, by, bz, bw = (following[key] for key in ("x", "y", "z", "w"))
    # following * inverse(current)
    x = -bw * ax + bx * aw - by * az + bz * ay
    y = -bw * ay + bx * az + by * aw - bz * ax
    z = -bw * az - bx * ay + by * ax + bz * aw
    w = bw * aw + bx * ax + by * ay + bz * az
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = (value / max(norm, 1e-12) for value in (x, y, z, w))
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    vector_norm = math.sqrt(x * x + y * y + z * z)
    if vector_norm <= 1e-9:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(vector_norm, max(w, 1e-12))
    return [angle * x / vector_norm, angle * y / vector_norm, angle * z / vector_norm]


def matrix_pose(matrix):
    transform = np.asarray(matrix, dtype=float)
    return {
        "position": {
            "x": float(transform[0, 3]),
            "y": float(transform[1, 3]),
            "z": float(transform[2, 3]),
        },
        "orientation": quaternion_from_matrix(transform),
    }


def pose_message(x, y, z, yaw=0.0):
    pose = PoseStamped()
    pose.header.frame_id = BASE_FRAME
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    pose.pose.position.z = float(z)
    pose.pose.orientation.z = math.sin(float(yaw) * 0.5)
    pose.pose.orientation.w = math.cos(float(yaw) * 0.5)
    return pose


class ObservationRecorder(Node):
    def __init__(self):
        super().__init__(
            "heuristic_grasp_recorder",
            automatically_declare_parameters_from_overrides=True,
        )
        self.tf_buffer = Buffer(node=self)
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.latest_joints = None
        self.active_path = None
        self.active_phase = "idle"
        self.lock = threading.Lock()
        self.create_subscription(JointState, "/joint_states", self.on_joint_state, 20)
        self.collision_publisher = self.create_publisher(
            CollisionObject, "/collision_object", 20
        )
        self.published_collision_ids = set()
        self.create_timer(0.05, self.sample_end_effector)

    def on_joint_state(self, message):
        with self.lock:
            self.latest_joints = {
                "name": list(message.name),
                "position": [float(value) for value in message.position],
                "velocity": [float(value) for value in message.velocity],
                "effort": [float(value) for value in message.effort],
            }

    def begin_path(self):
        with self.lock:
            self.active_path = []
            self.active_phase = "approach"

    def set_phase(self, phase):
        with self.lock:
            self.active_phase = phase

    def finish_path(self):
        with self.lock:
            result = list(self.active_path or [])
            self.active_path = None
            self.active_phase = "idle"
        return result

    def current_joint_state(self):
        with self.lock:
            return dict(self.latest_joints) if self.latest_joints is not None else None

    @staticmethod
    def transform_pose(transform):
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return {
            "position": {
                "x": float(translation.x),
                "y": float(translation.y),
                "z": float(translation.z),
            },
            "orientation": {
                "x": float(rotation.x),
                "y": float(rotation.y),
                "z": float(rotation.z),
                "w": float(rotation.w),
            },
        }

    def sample_end_effector(self):
        with self.lock:
            if self.active_path is None:
                return
            phase = self.active_phase
            joints = self.latest_joints
        frame_poses = {}
        for frame in TRACKED_FRAMES:
            try:
                transform = self.tf_buffer.lookup_transform(BASE_FRAME, frame, Time())
                frame_poses[frame] = self.transform_pose(transform)
            except TransformException:
                frame_poses[frame] = None
        if frame_poses[EE_FRAME] is None:
            return
        sample = {
            "sim_time": self.get_clock().now().nanoseconds / 1_000_000_000.0,
            "wall_time": utc_now(),
            "phase": phase,
            "ee_pose": frame_poses[EE_FRAME],
            "tracked_frame_poses": frame_poses,
            "joint_state": joints,
        }
        with self.lock:
            if self.active_path is not None:
                self.active_path.append(sample)

    def current_end_effector_pose(self):
        transform = self.tf_buffer.lookup_transform(BASE_FRAME, EE_FRAME, Time())
        return self.transform_pose(transform)

    def publish_clutter(self, scene):
        desired_ids = {
            item["instance_id"] for item in scene.get("objects", [])
            if not item.get("is_target", False)
        }
        for object_id in self.published_collision_ids - desired_ids:
            message = CollisionObject()
            message.header.frame_id = BASE_FRAME
            message.id = object_id
            message.operation = CollisionObject.REMOVE
            self.collision_publisher.publish(message)
        for item in scene.get("objects", []):
            if item.get("is_target", False):
                continue
            geometry = item["geometry"]
            primitive = SolidPrimitive()
            if geometry["shape"] == "sphere":
                primitive.type = SolidPrimitive.SPHERE
                primitive.dimensions = [float(geometry["half_width"])]
            else:
                primitive.type = SolidPrimitive.CYLINDER
                primitive.dimensions = [
                    2.0 * float(geometry["center_height"]),
                    float(geometry["half_width"]),
                ]
            pose = PoseStamped().pose
            pose.position.x = float(item["pose"]["x"])
            pose.position.y = float(item["pose"]["y"])
            pose.position.z = (
                float(item["table_height"])
                + float(geometry["center_height"])
            )
            pose.orientation.z = math.sin(float(item["pose"]["yaw"]) * 0.5)
            pose.orientation.w = math.cos(float(item["pose"]["yaw"]) * 0.5)
            message = CollisionObject()
            message.header.frame_id = BASE_FRAME
            message.id = item["instance_id"]
            message.primitives = [primitive]
            message.primitive_poses = [pose]
            message.operation = CollisionObject.ADD
            self.collision_publisher.publish(message)
        self.published_collision_ids = desired_ids


class ArtifactLog:
    def __init__(self, session_dir):
        self.path = session_dir / "events.jsonl"
        self.lock = threading.Lock()

    def write(self, event, message, **data):
        record = {"wall_time": utc_now(), "event": event, "message": message, **data}
        line = json.dumps(record, sort_keys=True)
        print(f"[grasp-heuristic] {message}", flush=True)
        with self.lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")



class HeuristicRecorder:
    def __init__(self, node):
        self.node = node
        seed_parameter = str(node.get_parameter("seed").value)
        requested_seed = int(seed_parameter)
        self.seed = requested_seed if requested_seed != 0 else secrets.randbits(31)
        self.random = random.Random(self.seed)
        self.example_count = int(node.get_parameter("example_count").value)
        self.finger_pad_z_correction = float(
            node.get_parameter("finger_pad_z_correction").value
        )
        self.lift_distance = float(node.get_parameter("lift_distance").value)
        self.speed_scale = float(node.get_parameter("speed_scale").value)
        diagnostics_value = node.get_parameter("record_diagnostics").value
        self.record_diagnostics = (
            diagnostics_value
            if isinstance(diagnostics_value, bool)
            else str(diagnostics_value).lower() in ("1", "true", "yes", "on")
        )
        if self.example_count < 1:
            raise ValueError("example_count must be at least 1")
        if self.speed_scale <= 0.0:
            raise ValueError("speed_scale must be greater than zero")
        output_root = Path(str(node.get_parameter("output_dir").value))
        session_name = datetime.now(timezone.utc).strftime("grasp_%Y%m%dT%H%M%SZ")
        self.session_dir = output_root / f"{session_name}_seed{self.seed}"
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.log = ArtifactLog(self.session_dir)
        self.robot = MoveItPy(node_name="moveit_py_grasp_dataset")
        self.robot_model = self.robot.get_robot_model()
        self.arm = self.robot.get_planning_component(ARM_GROUP)
        self.arm_client = ActionClient(
            node,
            FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
        )
        self.gripper_client = ActionClient(
            node,
            FollowJointTrajectory,
            "/gripper_controller/follow_joint_trajectory",
        )
        self.reset_scene_client = node.create_client(
            Trigger, "/tiago_grasp_sim/reset_scene"
        )
        self.clear_scene_client = node.create_client(
            Trigger, "/tiago_grasp_sim/clear_scene"
        )
        self.current_object_client = node.create_client(
            Trigger, "/tiago_grasp_sim/current_object"
        )
        self.stabilize_grasp_client = node.create_client(
            Trigger, "/tiago_grasp_sim/stabilize_grasp"
        )
        self.release_grasp_client = node.create_client(
            Trigger, "/tiago_grasp_sim/release_grasp"
        )
        self.policy = GeometryContactPolicy(
            self.random,
            lift_distance=self.lift_distance,
            finger_pad_z_correction=self.finger_pad_z_correction,
        )
        self.bag_process = None
        self.bag_log_stream = None
        self.gz_diagnostic_processes = []
        self.gz_diagnostic_streams = []

    def start_gazebo_diagnostics(self):
        topics = {
            "gazebo_dynamic_poses.pbtxt": "/world/default/dynamic_pose/info",
            "gazebo_all_poses.pbtxt": "/world/default/pose/info",
            "gazebo_physics_stats.pbtxt": "/world/default/stats",
            "gazebo_contacts.pbtxt": "/tiago_grasp_sim/contacts",
        }
        for filename, topic in topics.items():
            stream = (self.session_dir / filename).open("w", encoding="utf-8")
            process = subprocess.Popen(
                ["gz", "topic", "-e", "-t", topic],
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.gz_diagnostic_streams.append(stream)
            self.gz_diagnostic_processes.append(process)
        self.log.write(
            "gazebo_diagnostics_started",
            "Started raw Gazebo pose, physics, and contact streams",
            topics=list(topics.values()),
        )

    def stop_gazebo_diagnostics(self):
        for process in self.gz_diagnostic_processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        for process in self.gz_diagnostic_processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=3)
        for stream in self.gz_diagnostic_streams:
            stream.close()
        self.log.write(
            "gazebo_diagnostics_stopped",
            "Finished raw Gazebo diagnostic streams",
        )

    def start_bag(self):
        bag_path = self.session_dir / "sensors"
        self.bag_log_stream = (self.session_dir / "rosbag.log").open("w", encoding="utf-8")
        command = [
            "ros2",
            "bag",
            "record",
            "--storage",
            "mcap",
            "--storage-preset-profile",
            "zstd_fast",
            "--max-bag-size",
            str(4 * 1024 * 1024 * 1024),
            "--include-hidden-topics",
            "--output",
            str(bag_path),
            "--regex",
            "^/(clock|joint_states|tf|tf_static|tiago_grasp_sim/object|external_camera/(image/compressed|camera_info)|ft_sensor_controller/(wrench|wrench_filtered)|head_front_camera/(camera_info|image/compressed)|arm_controller/.*|gripper_controller/.*|torso_controller/.*)$",
        ]
        self.bag_process = subprocess.Popen(
            command,
            stdout=self.bag_log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.log.write("bag_started", "Started compressed RGB-D, joint, TF, and controller recording", path=str(bag_path))
        time.sleep(2)

    def stop_bag(self):
        if self.bag_process is not None and self.bag_process.poll() is None:
            os.killpg(self.bag_process.pid, signal.SIGINT)
            try:
                self.bag_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.bag_process.pid, signal.SIGTERM)
                self.bag_process.wait(timeout=5)
        if self.bag_log_stream is not None:
            self.bag_log_stream.close()
        self.log.write("bag_stopped", "Finished sensor recording")

    def call_scene_service(self, client, operation):
        if not client.wait_for_service(timeout_sec=60.0):
            raise RuntimeError(f"Scene manager is unavailable for {operation}")
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + 30.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not future.done() or future.result() is None:
            raise RuntimeError(f"Scene manager timed out during {operation}")
        response = future.result()
        if not response.success:
            raise RuntimeError(f"Scene manager failed during {operation}: {response.message}")
        return response.message

    def request_random_object(self):
        payload = self.call_scene_service(self.reset_scene_client, "reset_scene")
        scene = json.loads(payload)
        self.node.publish_clutter(scene)
        time.sleep(1.0)
        return scene

    def clear_scene_object(self):
        self.call_scene_service(self.clear_scene_client, "clear_scene")

    def current_scene_object(self):
        payload = self.call_scene_service(
            self.current_object_client, "current_object"
        )
        return json.loads(payload)

    def record_phase_snapshot(self, episode, phase):
        snapshot = {
            "phase": phase,
            "wall_time": utc_now(),
            "sim_time": self.node.get_clock().now().nanoseconds / 1_000_000_000.0,
            "end_effector_pose": self.node.current_end_effector_pose(),
            "joint_state": self.node.current_joint_state(),
            "object_state": self.current_scene_object(),
        }
        episode["phase_snapshots"].append(snapshot)
        return snapshot

    def stabilize_grasp(self):
        payload = self.call_scene_service(
            self.stabilize_grasp_client, "stabilize_grasp"
        )
        return json.loads(payload)

    def release_grasp(self):
        self.call_scene_service(self.release_grasp_client, "release_grasp")

    def serialize_trajectory(self, phase, trajectory, controller):
        message = trajectory.get_robot_trajectory_msg()
        points = []
        for point in message.joint_trajectory.points:
            points.append(
                {
                    "time_from_start": duration_seconds(point.time_from_start),
                    "positions": [float(value) for value in point.positions],
                    "velocities": [float(value) for value in point.velocities],
                    "accelerations": [float(value) for value in point.accelerations],
                    "effort": [float(value) for value in point.effort],
                }
            )
        ee_path = []
        for index, waypoint in enumerate(trajectory):
            state = waypoint
            if isinstance(waypoint, (tuple, list)):
                state = next(
                    (
                        item
                        for item in waypoint
                        if hasattr(item, "get_global_link_transform")
                    ),
                    None,
                )
            if state is None or not hasattr(state, "get_global_link_transform"):
                raise TypeError(
                    f"Unsupported MoveIt trajectory waypoint type: {type(waypoint)!r}"
                )
            ee_path.append(
                {
                    "waypoint": index,
                    "ee_pose": matrix_pose(state.get_global_link_transform(EE_FRAME)),
                }
            )
        return {
            "phase": phase,
            "controller": controller,
            "joint_names": list(message.joint_trajectory.joint_names),
            "points": points,
            "planned_ee_path": ee_path,
        }

    def plan_arm(self, phase, pose):
        target = (
            float(pose.pose.position.x),
            float(pose.pose.position.y),
            float(pose.pose.position.z),
        )
        attempts = []
        for attempt in range(1, 4):
            self.arm.set_start_state_to_current_state()
            self.arm.set_goal_state(pose_stamped_msg=pose, pose_link=EE_FRAME)
            result = self.arm.plan()
            if not result:
                attempts.append(
                    {"attempt": attempt, "planning_status": "FAILED"}
                )
                continue
            segment = self.execute_arm_plan(phase, result.trajectory)
            measured = self.node.current_end_effector_pose()
            measured_position = measured["position"]
            position_error = math.sqrt(
                (measured_position["x"] - target[0]) ** 2
                + (measured_position["y"] - target[1]) ** 2
                + (measured_position["z"] - target[2]) ** 2
            )
            attempt_record = {
                "attempt": attempt,
                "planning_status": "SUCCESSFUL",
                "controller_status": segment["execution_status"],
                "controller_error": segment.get("controller_error", ""),
                "target_position": dict(zip(("x", "y", "z"), target)),
                "measured_end_effector_pose": measured,
                "position_error": position_error,
                "trajectory": {
                    key: segment[key]
                    for key in (
                        "phase", "controller", "joint_names", "points",
                        "planned_ee_path", "execution_status",
                    )
                },
            }
            attempts.append(attempt_record)
            if segment["execution_status"] == "SUCCESSFUL":
                segment["execution_attempts"] = attempts
                return segment
            self.log.write(
                "arm_execution_retry",
                f"Correcting {phase} after controller failure; Cartesian "
                f"error is {position_error:.4f} m",
                phase=phase,
                attempt=attempt,
                position_error=position_error,
                target_position=attempt_record["target_position"],
                measured_end_effector_pose=measured,
                controller_error=segment.get("controller_error", ""),
            )
            # A failed trajectory action may still have reached the Cartesian
            # goal.  Accept only a tightly measured endpoint; otherwise replan
            # from the state the robot actually reached.
            if position_error <= 0.010:
                segment["execution_status"] = "MEASURED_TARGET_REACHED"
                segment["execution_attempts"] = attempts
                return segment
            time.sleep(0.5)
        errors = [
            item.get("position_error") for item in attempts
            if item.get("position_error") is not None
        ]
        raise RuntimeError(
            f"Arm failed phase {phase} after {len(attempts)} planned attempts; "
            f"measured Cartesian errors: {errors}"
        )

    def plan_arm_joints(self, phase, positions):
        state = RobotState(self.robot_model)
        state.set_to_default_values()
        state.set_joint_group_positions(ARM_GROUP, positions)
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(robot_state=state)
        result = self.arm.plan()
        if not result:
            raise RuntimeError(f"MoveIt failed to plan phase {phase}")
        segment = self.execute_arm_plan(phase, result.trajectory)
        if segment["execution_status"] != "SUCCESSFUL":
            raise RuntimeError(
                f"Controller failed phase {phase}: "
                f"{segment.get('controller_error', 'unknown controller error')}"
            )
        return segment

    def execute_arm_plan(self, phase, trajectory):
        segment = self.serialize_trajectory(phase, trajectory, "arm_controller")
        speed_scale = self.speed_scale
        for point in segment["points"]:
            point["time_from_start"] *= speed_scale
            point["velocities"] = [value / speed_scale for value in point["velocities"]]
            point["accelerations"] = [
                value / (speed_scale * speed_scale) for value in point["accelerations"]
            ]
        try:
            self.execute_segment(self.arm_client, segment)
            segment["execution_status"] = "SUCCESSFUL"
        except RuntimeError as error:
            segment["execution_status"] = "FAILED"
            segment["controller_error"] = str(error)
        time.sleep(0.5)
        return segment

    def execute_segment(self, client, segment):
        if not client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"Controller action is unavailable: {segment['controller']}")
        trajectory = JointTrajectory(joint_names=segment["joint_names"])
        for source in segment["points"]:
            point = JointTrajectoryPoint()
            point.positions = source["positions"]
            point.velocities = source["velocities"]
            point.accelerations = source["accelerations"]
            point.effort = source["effort"]
            point.time_from_start = duration_message(source["time_from_start"])
            trajectory.points.append(point)
        goal = FollowJointTrajectory.Goal(trajectory=trajectory)
        goal_future = client.send_goal_async(goal)
        timeout = max(30.0, segment["points"][-1]["time_from_start"] + 25.0)
        deadline = time.monotonic() + timeout
        while not goal_future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not goal_future.done() or goal_future.result() is None or not goal_future.result().accepted:
            raise RuntimeError(f"Controller rejected phase {segment['phase']}")
        result_future = goal_future.result().get_result_async()
        while not result_future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not result_future.done() or result_future.result() is None:
            raise RuntimeError(f"Controller timed out during phase {segment['phase']}")
        result = result_future.result().result
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(
                f"Controller failed phase {segment['phase']}: {result.error_string}"
            )

    def plan_gripper(self, phase, opening, allow_contact=False):
        if not self.gripper_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("Gripper controller action is unavailable")
        joint_names = ["gripper_left_finger_joint", "gripper_right_finger_joint"]
        point = JointTrajectoryPoint()
        point.positions = [float(opening), float(opening)]
        point.time_from_start = Duration(sec=1)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory(joint_names=joint_names, points=[point])
        goal_future = self.gripper_client.send_goal_async(goal)
        deadline = time.monotonic() + 12.0
        while not goal_future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not goal_future.done() or goal_future.result() is None or not goal_future.result().accepted:
            raise RuntimeError(f"Gripper rejected phase {phase}")
        result_future = goal_future.result().get_result_async()
        while not result_future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not result_future.done() or result_future.result() is None:
            raise RuntimeError(f"Gripper timed out during phase {phase}")
        result = result_future.result().result
        succeeded = result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
        if not succeeded and not allow_contact:
            raise RuntimeError(f"Gripper failed phase {phase}: {result.error_string}")
        time.sleep(0.5)
        return {
            "phase": phase,
            "controller": "gripper_controller",
            "joint_names": joint_names,
            "points": [
                {
                    "time_from_start": 1.0,
                    "positions": [float(opening), float(opening)],
                    "velocities": [],
                    "accelerations": [],
                    "effort": [],
                }
            ],
            "planned_ee_path": [],
            "execution_status": "SUCCESSFUL" if succeeded else "CONTACT_STOP",
            "controller_error": result.error_string if not succeeded else "",
        }

    def validate_motion(self, target):
        joint_state = self.node.current_joint_state()
        if joint_state is None:
            return {"validated": False, "reason": "No joint state was received"}
        positions = dict(zip(joint_state["name"], joint_state["position"]))
        efforts = dict(zip(joint_state["name"], joint_state["effort"]))
        names = ["gripper_left_finger_joint", "gripper_right_finger_joint"]
        fingers = [float(positions[name]) for name in names]
        finger_efforts = [float(efforts[name]) for name in names]
        symmetry_error = abs(fingers[0] - fingers[1])
        ee_pose = self.node.current_end_effector_pose()
        position = ee_pose["position"]
        position_error = math.sqrt(
            (position["x"] - target[0]) ** 2
            + (position["y"] - target[1]) ** 2
            + (position["z"] - target[2]) ** 2
        )
        validated = position_error <= 0.015
        reason = (
            "Measured end-effector pose reached the requested target"
            if validated
            else "Measured end-effector pose missed its requested target"
        )
        return {
            "validated": validated,
            "reason": reason,
            "validation_method": "measured_target_pose",
            "finger_positions": dict(zip(names, fingers)),
            "finger_efforts": dict(zip(names, finger_efforts)),
            "symmetry_error": symmetry_error,
            "measured_ee_pose": ee_pose,
            "position_error": position_error,
            "maximum_position_error": 0.015,
        }

    def choose_random_start_joints(self):
        anchor = [0.17, -0.27, -1.35, 2.28, 1.22, 0.92, 1.57]
        jitter = [0.06, 0.05, 0.06, 0.04, 0.06, 0.05, 0.06]
        return [
            value + self.random.uniform(-spread, spread)
            for value, spread in zip(anchor, jitter)
        ]

    def move_to_viable_start(self):
        last_error = None
        for _ in range(5):
            try:
                segment = self.plan_arm_joints(
                    "position_random_start", self.choose_random_start_joints()
                )
                return self.node.current_end_effector_pose(), segment
            except RuntimeError as error:
                last_error = error
        raise RuntimeError(f"Unable to find a viable random start pose: {last_error}")

    def reverse_arm_segment(self, source, phase):
        points = list(reversed(source["points"]))
        total_time = max(4.0, source["points"][-1]["time_from_start"])
        count = len(points)
        reset = {
            "phase": phase,
            "controller": "arm_controller",
            "joint_names": source["joint_names"],
            "points": [],
            "planned_ee_path": list(reversed(source["planned_ee_path"])),
        }
        for index, point in enumerate(points, start=1):
            reset["points"].append(
                {
                    "time_from_start": total_time * index / count,
                    "positions": point["positions"],
                    "velocities": [],
                    "accelerations": [],
                    "effort": [],
                }
            )
        self.execute_segment(self.arm_client, reset)
        reset["execution_status"] = "SUCCESSFUL"
        return reset

    def write_training_steps(self, episode, path):
        samples = episode.get("actual_ee_path", [])
        task = episode["object"].get("task", {})
        distractors = [
            item["instance_id"]
            for item in episode["object"].get("objects", [])
            if not item.get("is_target", False)
        ]
        with path.open("w", encoding="utf-8") as stream:
            for index, sample in enumerate(samples):
                following = samples[min(index + 1, len(samples) - 1)]
                current_pose = sample["ee_pose"]
                next_pose = following["ee_pose"]
                delta_position = [
                    next_pose["position"][axis] - current_pose["position"][axis]
                    for axis in ("x", "y", "z")
                ]
                delta_rotation = quaternion_delta_rotvec(
                    current_pose["orientation"], next_pose["orientation"]
                )
                joints = sample.get("joint_state") or {
                    "name": [], "position": [], "velocity": [], "effort": []
                }
                positions = dict(zip(joints["name"], joints["position"]))
                velocities = dict(zip(joints["name"], joints["velocity"]))
                efforts = dict(zip(joints["name"], joints["effort"]))
                next_joints = following.get("joint_state") or joints
                next_positions = dict(
                    zip(next_joints["name"], next_joints["position"])
                )
                arm_names = list(ARM_JOINTS)
                finger_names = [
                    "gripper_left_finger_joint", "gripper_right_finger_joint"
                ]
                gripper = sum(float(positions.get(name, 0.0)) for name in finger_names) / 2.0
                next_gripper = sum(
                    float(next_positions.get(name, gripper)) for name in finger_names
                ) / 2.0
                record = {
                    "step_index": index,
                    "sim_time": sample["sim_time"],
                    "phase": sample["phase"],
                    "task": task,
                    "observation": {
                        "rgb": {
                            "head": {"topic": "/head_front_camera/image/compressed", "nearest_timestamp": sample["sim_time"]},
                            "external": {"topic": "/external_camera/image/compressed", "nearest_timestamp": sample["sim_time"]},
                        },
                        "end_effector_pose_base": current_pose,
                        "arm_joint_position": [float(positions.get(name, 0.0)) for name in arm_names],
                        "arm_joint_velocity": [float(velocities.get(name, 0.0)) for name in arm_names],
                        "arm_joint_effort": [float(efforts.get(name, 0.0)) for name in arm_names],
                        "arm_joint_names": arm_names,
                        "gripper_position": gripper,
                    },
                    "action": {
                        "representation": "delta_end_effector_rotvec_gripper_terminate",
                        "values": delta_position + delta_rotation + [next_gripper, float(index == len(samples) - 1)],
                    },
                    "supervision": {
                        "target_instance_id": task.get("target_instance_id"),
                        "distractor_instance_ids": distractors,
                        "success": bool(episode.get("grasp_success")),
                    },
                }
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        return {
            "path": path.name,
            "step_count": len(samples),
            "primary_action_dimension": 8,
            "primary_action": "delta xyz + delta rotation-vector + gripper + terminate",
            "full_arm_state_retained": True,
        }

    def generate_episode(self, index, object_info):
        policy_result = self.policy.compute(object_info)
        pregrasp = policy_result["pregrasp_pose"]
        grasp = policy_result["grasp_pose"]
        lift = policy_result["lift_pose"]
        initial_object_state = self.current_scene_object()
        episode = {
            "schema_version": 2,
            "episode_index": index,
            "policy_seed": self.seed,
            "object": object_info,
            "initial_object_state": initial_object_state,
            "policy": policy_result,
            "inverse_kinematics": {
                "solver": "MoveIt KDL kinematics plugin",
                "planning_group": ARM_GROUP,
                "end_effector_frame": EE_FRAME,
            },
            "base_motion_commanded": False,
            "lift_commanded": True,
            "status": "running",
            "grasp_success": False,
            "grasp_label": "failed",
            "physical_contact_verified": False,
            "obstacle_avoidance": {
                "planner": "MoveIt OMPL RRTConnect with distractor collision objects",
                "distractor_instance_ids": [
                    item["instance_id"] for item in object_info.get("objects", [])
                    if not item.get("is_target", False)
                ],
                "whole_arm_collision_checked": True,
            },
            "planned_segments": [],
            "reset_segments": [],
            "actual_ee_path": [],
            "phase_snapshots": [],
        }
        path = self.session_dir / f"episode_{index:03d}_{object_info['type']}.json"
        self.log.write(
            "episode_started",
            f"Starting geometry-based grasp of {object_info['type']}",
            episode=index,
            object=object_info,
            contacts=policy_result["contacts"],
        )
        try:
            episode["preparation_segment"] = self.plan_gripper(
                "open_before_episode", 0.044
            )
            start_pose, positioning_segment = self.move_to_viable_start()
            episode["random_start_pose"] = start_pose
            episode["positioning_segment"] = positioning_segment
            self.record_phase_snapshot(episode, "random_start_reached")
            self.node.begin_path()
            self.node.set_phase("pregrasp")
            episode["planned_segments"].append(
                self.plan_arm(
                    "move_to_pregrasp",
                    pose_message(
                        pregrasp["position"]["x"],
                        pregrasp["position"]["y"],
                        pregrasp["position"]["z"],
                        pregrasp["yaw"],
                    ),
                )
            )
            self.record_phase_snapshot(episode, "pregrasp_reached")
            self.node.set_phase("approach")
            episode["planned_segments"].append(
                self.plan_arm(
                    "approach_contacts",
                    pose_message(
                        grasp["position"]["x"],
                        grasp["position"]["y"],
                        grasp["position"]["z"],
                        grasp["yaw"],
                    ),
                )
            )
            self.record_phase_snapshot(episode, "contact_pose_reached")
            self.node.set_phase("close_gripper")
            # The trajectory controller does not stop on contact. Command the
            # PAL pad centres to the two sampled object surfaces plus the
            # 5 mm pad half-thickness, instead of driving through the object.
            contact_opening = min(
                0.044, 0.5 * policy_result["contact_separation"] + 0.005
            )
            episode["contact_limited_finger_opening"] = contact_opening
            episode["planned_segments"].append(
                self.plan_gripper(
                    "close_to_object_surface", contact_opening,
                    allow_contact=True,
                )
            )
            time.sleep(0.75)
            self.record_phase_snapshot(episode, "fingers_closed")
            episode["contact_pose_validation"] = self.validate_motion(
                (
                    grasp["position"]["x"],
                    grasp["position"]["y"],
                    grasp["position"]["z"],
                )
            )
            finger_positions = list(
                episode["contact_pose_validation"]["finger_positions"].values()
            )
            finger_opening_error = max(
                abs(value - contact_opening) for value in finger_positions
            )
            episode["contact_pose_validation"][
                "contact_limited_opening_error"
            ] = finger_opening_error
            if (
                not episode["contact_pose_validation"]["validated"]
                or finger_opening_error > 0.003
            ):
                raise RuntimeError(
                    "The gripper did not reach the selected contact-limited pose"
                )
            episode["grasp_stabilization"] = self.stabilize_grasp()
            episode["geometric_enclosure_verified"] = True
            self.record_phase_snapshot(episode, "grasp_stabilized")
            self.node.set_phase("lift")
            episode["planned_segments"].append(
                self.plan_arm(
                    "lift_grasped_object",
                    pose_message(
                        lift["position"]["x"],
                        lift["position"]["y"],
                        lift["position"]["z"],
                        lift["yaw"],
                    ),
                )
            )
            self.record_phase_snapshot(episode, "lift_pose_reached")
            self.node.set_phase("hold_after_lift")
            time.sleep(1.0)
            episode["actual_ee_path"] = self.node.finish_path()
            episode["lift_motion_validation"] = self.validate_motion(
                (
                    lift["position"]["x"],
                    lift["position"]["y"],
                    lift["position"]["z"],
                )
            )
            final_object_state = self.current_scene_object()
            episode["final_object_state"] = final_object_state
            initial_z = float(initial_object_state["actual_pose"]["z"])
            final_z = float(final_object_state["actual_pose"]["z"])
            object_rise = final_z - initial_z
            minimum_rise = max(0.02, self.lift_distance * 0.5)
            lift_verified = (
                episode["lift_motion_validation"]["validated"]
                and object_rise >= minimum_rise
            )
            episode["lift_validation"] = {
                "validated": lift_verified,
                "initial_object_z": initial_z,
                "final_object_z": final_z,
                "object_rise": object_rise,
                "minimum_object_rise": minimum_rise,
                "method": "gazebo_object_pose_delta_after_gripper_lift",
            }
            if not lift_verified:
                raise RuntimeError(
                    f"Object rose {object_rise:.4f} m; at least "
                    f"{minimum_rise:.4f} m is required"
                )
            episode["status"] = "completed"
            episode["grasp_success"] = True
            episode["grasp_label"] = "success"
            episode["simulated_lift_verified"] = True
            self.log.write(
                "episode_completed",
                f"Grasped and lifted {object_info['type']} by {object_rise:.3f} m",
                episode=index,
                actual_samples=len(episode["actual_ee_path"]),
                object_rise=object_rise,
            )
        except Exception as error:
            if not episode["actual_ee_path"]:
                episode["actual_ee_path"] = self.node.finish_path()
            episode["status"] = "failed"
            episode["error"] = str(error)
            self.log.write(
                "episode_failed",
                f"Geometry-based grasp of {object_info['type']} failed: {error}",
                episode=index,
            )
        finally:
            try:
                self.release_grasp()
            except Exception as error:
                self.log.write(
                    "grasp_release_failed",
                    f"Could not release grasp stabilization: {error}",
                    episode=index,
                )
            try:
                self.plan_gripper("release_after_episode", 0.044)
            except Exception as error:
                self.log.write(
                    "release_failed",
                    f"Could not reopen gripper: {error}",
                    episode=index,
                )
            try:
                arm_segments = [
                    segment
                    for segment in episode["planned_segments"]
                    if segment["controller"] == "arm_controller"
                ]
                for segment in reversed(arm_segments):
                    episode["reset_segments"].append(
                        self.reverse_arm_segment(segment, "reset_after_release")
                    )
                if "positioning_segment" in episode:
                    episode["reset_segments"].append(
                        self.reverse_arm_segment(
                            episode["positioning_segment"], "return_to_ready"
                        )
                    )
            except Exception as error:
                self.log.write(
                    "reset_failed",
                    f"Could not return arm to its ready pose: {error}",
                    episode=index,
                )
            steps_path = self.session_dir / (
                f"episode_{index:03d}_{object_info['type']}_steps.jsonl"
            )
            episode["training_steps"] = self.write_training_steps(
                episode, steps_path
            )
            path.write_text(
                json.dumps(episode, indent=2, sort_keys=True), encoding="utf-8"
            )
        return episode

    def run(self):
        manifest = {
            "schema_version": 2,
            "created_at": utc_now(),
            "policy_seed": self.seed,
            "policy_name": "pca_antipodal_two_finger",
            "reference_doi": "10.1177/1729881419831846",
            "frame": BASE_FRAME,
            "end_effector_frame": EE_FRAME,
            "requested_episode_count": self.example_count,
            "policy_parameters": {
                "finger_pad_z_correction": self.finger_pad_z_correction,
                "finger_pad_forward_offset": self.policy.finger_pad_forward_offset,
                "finger_pad_half_height": self.policy.finger_pad_half_height,
                "pregrasp_distance": self.policy.pregrasp_distance,
                "lift_distance": self.lift_distance,
                "arm_trajectory_speed_scale": self.speed_scale,
                "raw_gazebo_diagnostics_recorded": self.record_diagnostics,
            },
            "base_motion_commanded": False,
            "lift_commanded": True,
            "grasp_labels": "verified_by_gazebo_object_lift",
            "objects": [],
        }
        dataset_schema = {
            "schema_version": 1,
            "task_input": {
                "verb": "pick_up",
                "target": "category + color referring expression",
                "simulator_instance_id_is_supervision_only": True,
            },
            "observations": {
                "rgb_topics": [
                    "/head_front_camera/image/compressed",
                    "/external_camera/image/compressed",
                ],
                "camera_calibration_topics": [
                    "/head_front_camera/camera_info",
                    "/external_camera/camera_info",
                ],
                "proprioception": [
                    "7 arm joint positions, velocities, efforts",
                    "absolute end-effector pose in base_footprint",
                    "two-finger gripper position",
                ],
            },
            "primary_action": {
                "dimension": 8,
                "layout": ["dx", "dy", "dz", "drot_x", "drot_y", "drot_z", "gripper", "terminate"],
                "rotation": "axis-angle rotation vector",
                "frame": BASE_FRAME,
            },
            "native_actions_retained": "full planned joint trajectories and measured joint states",
            "storage": {
                "sensors": "MCAP with zstd compression",
                "aligned_policy_steps": "JSON Lines keyed by simulation timestamp",
            },
        }
        (self.session_dir / "dataset_schema.json").write_text(
            json.dumps(dataset_schema, indent=2, sort_keys=True), encoding="utf-8"
        )
        (self.session_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        self.log.write(
            "session_started",
            "Optional heuristic recording session started",
            policy_seed=self.seed,
            directory=str(self.session_dir),
        )
        episodes = []
        try:
            if self.record_diagnostics:
                self.start_gazebo_diagnostics()
            self.start_bag()
            for index in range(1, self.example_count + 1):
                object_info = self.request_random_object()
                manifest["objects"].append(object_info)
                (self.session_dir / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
                )
                episodes.append(self.generate_episode(index, object_info))
                self.write_summary(episodes, final=False)
        finally:
            try:
                self.clear_scene_object()
            except Exception as error:
                self.log.write(
                    "scene_clear_failed",
                    f"Could not clear the scene object: {error}",
                )
            self.stop_bag()
            if self.record_diagnostics:
                self.stop_gazebo_diagnostics()
        summary = self.write_summary(episodes, final=True)
        self.log.write("session_completed", "Heuristic recording session completed", **summary)
        return summary

    def write_summary(self, episodes, final):
        summary = {
            "policy_seed": self.seed,
            "session_directory": str(self.session_dir),
            "requested_episode_count": self.example_count,
            "episode_count": len(episodes),
            "completed": sum(
                episode["status"] == "completed" for episode in episodes
            ),
            "failed": sum(episode["status"] == "failed" for episode in episodes),
            "simulated_lift_verified_successes": sum(
                bool(episode.get("simulated_lift_verified")) for episode in episodes
            ),
            "final": final,
            "episodes": [
                {
                    "index": episode["episode_index"],
                    "object_type": episode["object"]["type"],
                    "status": episode["status"],
                    "actual_path_samples": len(episode["actual_ee_path"]),
                    "error": episode.get("error"),
                }
                for episode in episodes
            ],
        }
        (self.session_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        index_records = []
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            split_bucket = episode_index % 10
            split = "test" if split_bucket == 0 else "validation" if split_bucket == 1 else "train"
            object_type = episode["object"]["type"]
            index_records.append({
                "episode_index": episode_index,
                "split": split,
                "status": episode["status"],
                "success": bool(episode.get("grasp_success")),
                "instruction": episode["object"].get("instruction"),
                "object_type": object_type,
                "episode_path": f"episode_{episode_index:03d}_{object_type}.json",
                "steps_path": f"episode_{episode_index:03d}_{object_type}_steps.jsonl",
            })
        (self.session_dir / "dataset_index.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in index_records),
            encoding="utf-8",
        )
        return summary


def main():
    rclpy.init()
    node = ObservationRecorder()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    exit_code = 0
    generator = None
    try:
        generator = HeuristicRecorder(node)
        summary = generator.run()
        if summary["completed"] == 0:
            exit_code = 1
    except Exception as error:
        node.get_logger().error(f"Heuristic recording stopped: {error}")
        exit_code = 1
        if generator is not None:
            try:
                generator.stop_bag()
            except Exception:
                pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()

    # The Jazzy MoveItPy bindings can fault while their C++ objects are torn
    # down by Python. The one-shot generator has closed every artifact here.
    os._exit(exit_code)


if __name__ == "__main__":
    main()
