#!/usr/bin/env python3
import json
import math
import random
import re
import secrets
import subprocess
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String
from std_srvs.srv import Trigger


TABLE_HEIGHT = 0.55
OBJECT_TYPES = ("apple", "cup", "can")
OBJECT_GEOMETRY = {
    "apple": {"shape": "sphere", "half_width": 0.032, "center_height": 0.032},
    "cup": {"shape": "cylinder", "half_width": 0.032, "center_height": 0.0475},
    "can": {"shape": "cylinder", "half_width": 0.028, "center_height": 0.0575},
}
COLOR_PALETTE = (
    ("red", (0.82, 0.10, 0.08)),
    ("orange", (0.92, 0.34, 0.06)),
    ("yellow", (0.88, 0.76, 0.08)),
    ("green", (0.10, 0.68, 0.18)),
    ("cyan", (0.08, 0.70, 0.72)),
    ("blue", (0.08, 0.25, 0.84)),
    ("purple", (0.52, 0.14, 0.76)),
)


def rgba_text(color):
    return " ".join(f"{value:.5f}" for value in (*color, 1.0))


def surface_xml():
    return """
      <surface>
        <friction><ode><mu>1.4</mu><mu2>1.4</mu2></ode></friction>
        <contact><ode><kp>1000000</kp><kd>10</kd></ode></contact>
      </surface>"""


def inertial_xml(mass):
    inertia = max(0.00001, mass * 0.001)
    return f"""
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{inertia}</ixx><iyy>{inertia}</iyy><izz>{inertia}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>"""


def table_sdf():
    leg_poses = [(0.50, -0.40), (0.50, 0.40), (1.02, -0.40), (1.02, 0.40)]
    legs = []
    for index, (x_pos, y_pos) in enumerate(leg_poses):
        legs.append(
            f"""
        <collision name='leg_{index}_collision'><pose>{x_pos} {y_pos} 0.265 0 0 0</pose><geometry><box><size>0.055 0.055 0.53</size></box></geometry></collision>
        <visual name='leg_{index}_visual'><pose>{x_pos} {y_pos} 0.265 0 0 0</pose><geometry><box><size>0.055 0.055 0.53</size></box></geometry><material><ambient>0.20 0.16 0.11 1</ambient><diffuse>0.28 0.21 0.14 1</diffuse></material></visual>"""
        )
    return f"""<?xml version='1.0'?>
<sdf version='1.9'>
  <model name='grasp_table'>
    <static>true</static>
    <link name='table_link'>
      <collision name='top_collision'><pose>0.76 0 0.53 0 0 0</pose><geometry><box><size>0.66 0.92 0.04</size></box></geometry></collision>
      <visual name='top_visual'><pose>0.76 0 0.53 0 0 0</pose><geometry><box><size>0.66 0.92 0.04</size></box></geometry><material><ambient>0.42 0.28 0.15 1</ambient><diffuse>0.58 0.39 0.20 1</diffuse></material></visual>
      {''.join(legs)}
    </link>
  </model>
</sdf>"""


def external_camera_sdf():
    return """<?xml version='1.0'?>
<sdf version='1.9'>
  <model name='external_rgb_camera'>
    <static>true</static>
    <pose>0.45 -0.85 1.35 0 0.694 1.232</pose>
    <link name='camera_link'>
      <sensor name='external_rgb' type='camera'>
        <always_on>true</always_on>
        <update_rate>10</update_rate>
        <topic>/external_camera/image</topic>
        <gz_frame_id>external_camera_optical_frame</gz_frame_id>
        <camera>
          <horizontal_fov>1.0472</horizontal_fov>
          <image><width>1280</width><height>720</height><format>R8G8B8</format></image>
          <clip><near>0.08</near><far>5.0</far></clip>
        </camera>
      </sensor>
    </link>
  </model>
</sdf>"""


def object_sdf(kind, name, color):
    material = rgba_text(color)
    surface = surface_xml()
    if kind == "apple":
        geometry = f"""
      <collision name='body_collision'><geometry><sphere><radius>0.032</radius></sphere></geometry>{surface}</collision>
      <visual name='body_visual'><geometry><sphere><radius>0.032</radius></sphere></geometry><material><ambient>{material}</ambient><diffuse>{material}</diffuse></material></visual>
      <visual name='stem_visual'><pose>0 0 0.037 0 0 0</pose><geometry><cylinder><radius>0.0035</radius><length>0.020</length></cylinder></geometry><material><ambient>0.20 0.09 0.03 1</ambient><diffuse>0.28 0.12 0.04 1</diffuse></material></visual>"""
        mass = 0.12
    elif kind == "cup":
        geometry = f"""
      <collision name='body_collision'><geometry><cylinder><radius>0.032</radius><length>0.095</length></cylinder></geometry>{surface}</collision>
      <visual name='body_visual'><geometry><cylinder><radius>0.032</radius><length>0.095</length></cylinder></geometry><material><ambient>{material}</ambient><diffuse>{material}</diffuse></material></visual>
      <visual name='opening_visual'><pose>0 0 0.0485 0 0 0</pose><geometry><cylinder><radius>0.025</radius><length>0.002</length></cylinder></geometry><material><ambient>0.025 0.025 0.025 1</ambient><diffuse>0.04 0.04 0.04 1</diffuse></material></visual>"""
        mass = 0.10
    elif kind == "can":
        geometry = f"""
      <collision name='body_collision'><geometry><cylinder><radius>0.028</radius><length>0.115</length></cylinder></geometry>{surface}</collision>
      <visual name='body_visual'><geometry><cylinder><radius>0.028</radius><length>0.115</length></cylinder></geometry><material><ambient>{material}</ambient><diffuse>{material}</diffuse></material></visual>
      <visual name='top_visual'><pose>0 0 0.058 0 0 0</pose><geometry><cylinder><radius>0.026</radius><length>0.002</length></cylinder></geometry><material><ambient>0.65 0.68 0.70 1</ambient><diffuse>0.78 0.80 0.82 1</diffuse></material></visual>"""
        mass = 0.18
    else:
        raise ValueError(f"Unsupported object type: {kind}")
    return f"""<?xml version='1.0'?>
<sdf version='1.9'>
  <model name='{name}'>
    <link name='object_link'>
      {inertial_xml(mass)}
      {geometry}
      <sensor name='object_contact_sensor' type='contact'>
        <always_on>true</always_on>
        <update_rate>200</update_rate>
        <topic>/tiago_grasp_sim/contacts</topic>
        <contact><collision>body_collision</collision></contact>
      </sensor>
    </link>
  </model>
</sdf>"""


class GazeboScene:
    def __init__(self, node):
        self.node = node

    def spawn(self, name, sdf, x=0.0, y=0.0, z=0.0, yaw=0.0):
        command = [
            "ros2", "run", "ros_gz_sim", "create", "-world", "default",
            "-name", name, "-string", sdf, "-x", str(x), "-y", str(y),
            "-z", str(z), "-Y", str(yaw),
        ]
        last_error = ""
        for attempt in range(1, 31):
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=15
                )
            except subprocess.TimeoutExpired as error:
                last_error = f"timed out after {error.timeout} seconds"
                # The create request can succeed in Gazebo even when its ROS
                # client times out waiting for the response. Accept that case
                # instead of retrying the same model name indefinitely.
                try:
                    probe = subprocess.run(
                        ["gz", "model", "-m", name, "-p"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if probe.returncode == 0 and "Pose [ XYZ (m) ]" in probe.stdout:
                        self.node.get_logger().info(
                            f"Spawned {name}; create response timed out"
                        )
                        return
                except subprocess.TimeoutExpired:
                    pass
                if attempt < 30:
                    time.sleep(2)
                    continue
                raise RuntimeError(f"Unable to spawn {name}: {last_error}")
            if result.returncode == 0:
                self.node.get_logger().info(f"Spawned {name}")
                return
            last_error = (result.stderr or result.stdout).strip()
            if attempt == 30:
                raise RuntimeError(f"Unable to spawn {name}: {last_error}")
            time.sleep(2)

    def remove(self, name):
        command = [
            "ros2", "run", "ros_gz_sim", "remove", "--ros-args",
            "-p", "world:=default", "-p", f"entity_name:={name}",
        ]
        last_error = ""
        for attempt in range(5):
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=15
                )
                if result.returncode == 0:
                    self.node.get_logger().info(f"Removed {name}")
                    return
                last_error = (result.stderr or result.stdout).strip()
            except subprocess.TimeoutExpired as error:
                last_error = f"timed out after {error.timeout} seconds"
            if attempt < 4:
                time.sleep(1)
        raise RuntimeError(f"Unable to remove {name}: {last_error}")

    def pose(self, name):
        command = ["gz", "model", "-m", name, "-p"]
        result = None
        match = None
        for attempt in range(5):
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=15
                )
            except subprocess.TimeoutExpired:
                if attempt < 4:
                    time.sleep(0.2)
                continue
            if result.returncode == 0:
                match = re.search(
                    r"Pose \[ XYZ \(m\) \] \[ RPY \(rad\) \]:\s*"
                    r"\[\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\]\s*"
                    r"\[\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\]",
                    result.stdout,
                )
                if match is not None:
                    break
            if attempt < 4:
                time.sleep(0.2)
        if match is None:
            output = (result.stderr or result.stdout).strip() if result else ""
            raise RuntimeError(
                f"Unable to read pose for {name} after 5 attempts: {output!r}"
            )
        values = [float(value) for value in match.groups()]
        return {
            "x": values[0],
            "y": values[1],
            "z": values[2],
            "roll": values[3],
            "pitch": values[4],
            "yaw": values[5],
        }

    def link_pose(self, model_name, link_name):
        command = ["gz", "model", "-m", model_name, "-l", link_name, "-p"]
        result = None
        link_block = None
        for attempt in range(5):
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=15
                )
            except subprocess.TimeoutExpired:
                if attempt < 4:
                    time.sleep(0.2)
                continue
            if result.returncode == 0:
                link_block = re.search(
                    rf"- Name:\s*{re.escape(link_name)}\s.*?"
                    r"- Pose \[ XYZ \(m\) \] \[ RPY \(rad\) \]:\s*"
                    r"\[\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\]\s*"
                    r"\[\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\]",
                    result.stdout,
                    re.DOTALL,
                )
                if link_block is not None:
                    break
            if attempt < 4:
                time.sleep(0.2)
        if link_block is None:
            output = (result.stderr or result.stdout).strip() if result else ""
            raise RuntimeError(
                f"Unable to read link pose for {model_name}/{link_name} "
                f"after 5 attempts: {output!r}"
            )
        values = [float(value) for value in link_block.groups()]
        return {
            "x": values[0], "y": values[1], "z": values[2],
            "roll": values[3], "pitch": values[4], "yaw": values[5],
        }

    @staticmethod
    def finger_pad_center(link_pose):
        # The PAL finger collision box is centred 0.165 m along local -Z.
        roll = link_pose["roll"]
        pitch = link_pose["pitch"]
        yaw = link_pose["yaw"]
        local_z_world = (
            math.cos(yaw) * math.sin(pitch) * math.cos(roll)
            + math.sin(yaw) * math.sin(roll),
            math.sin(yaw) * math.sin(pitch) * math.cos(roll)
            - math.cos(yaw) * math.sin(roll),
            math.cos(pitch) * math.cos(roll),
        )
        return {
            axis: link_pose[axis] - 0.165 * component
            for axis, component in zip(("x", "y", "z"), local_z_world)
        }

    def validate_finger_enclosure(self, object_name, geometry):
        object_pose = self.pose(object_name)
        left_pose = self.link_pose("tiago", "gripper_left_finger_link")
        right_pose = self.link_pose("tiago", "gripper_right_finger_link")
        left = self.finger_pad_center(left_pose)
        right = self.finger_pad_center(right_pose)
        object_point = [object_pose[axis] for axis in ("x", "y", "z")]
        left_point = [left[axis] for axis in ("x", "y", "z")]
        right_point = [right[axis] for axis in ("x", "y", "z")]
        midpoint = [(a + b) * 0.5 for a, b in zip(left_point, right_point)]
        separation_vector = [a - b for a, b in zip(left_point, right_point)]
        separation = math.sqrt(sum(value * value for value in separation_vector))
        horizontal_separation = math.hypot(
            separation_vector[0], separation_vector[1]
        )
        midpoint_xy_error = math.hypot(
            midpoint[0] - object_point[0], midpoint[1] - object_point[1]
        )
        midpoint_z_error = abs(midpoint[2] - object_point[2])
        vertical_overlap = (
            float(geometry["center_height"]) + 0.025 - midpoint_z_error
        )
        object_diameter = 2.0 * float(geometry["half_width"])
        minimum_pad_separation = object_diameter + 0.004
        maximum_pad_separation = object_diameter + 0.018
        validated = (
            horizontal_separation >= minimum_pad_separation
            and horizontal_separation <= maximum_pad_separation
            and abs(separation_vector[2]) <= 0.012
            and midpoint_xy_error <= 0.012
            and vertical_overlap >= 0.005
        )
        details = {
            "validated": validated,
            "method": "gazebo_actual_finger_pad_geometric_enclosure",
            "object_pose": object_pose,
            "left_finger_link_pose": left_pose,
            "right_finger_link_pose": right_pose,
            "left_pad_center": left,
            "right_pad_center": right,
            "pad_midpoint": dict(zip(("x", "y", "z"), midpoint)),
            "pad_separation": separation,
            "horizontal_pad_separation": horizontal_separation,
            "object_diameter": object_diameter,
            "minimum_pad_separation": minimum_pad_separation,
            "maximum_pad_separation": maximum_pad_separation,
            "midpoint_xy_error": midpoint_xy_error,
            "midpoint_z_error": midpoint_z_error,
            "vertical_pad_object_overlap": vertical_overlap,
        }
        if not validated:
            raise RuntimeError(
                "Refusing artificial attachment because the object is not "
                f"enclosed by both finger pads: {json.dumps(details, sort_keys=True)}"
            )
        return details

    def stabilize_grasp(self, object_name):
        topic_root = f"/{object_name}/grasp_joint"
        model_result = subprocess.run(
            ["gz", "model", "-m", "tiago", "-p"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        model_match = re.search(r"Model:\s*\[(\d+)\]", model_result.stdout)
        if model_result.returncode != 0 or model_match is None:
            raise RuntimeError(
                "Unable to resolve TIAGo's Gazebo entity ID: "
                f"{model_result.stderr or model_result.stdout}"
            )
        model_id = int(model_match.group(1))
        inner_xml = (
            "<parent_link>arm_7_link</parent_link>"
            f"<child_model>{object_name}</child_model>"
            "<child_link>object_link</child_link>"
            f"<detach_topic>{topic_root}/detach</detach_topic>"
            f"<attach_topic>{topic_root}/attach</attach_topic>"
            f"<output_topic>{topic_root}/state</output_topic>"
        )
        request = (
            f'entity: {{id: {model_id}, type: MODEL}} '
            'plugins: {name: "gz::sim::systems::DetachableJoint" '
            'filename: "gz-sim-detachable-joint-system" '
            f'innerxml: "{inner_xml}"}}'
        )
        state_process = subprocess.Popen(
            ["gz", "topic", "-e", "-t", f"{topic_root}/state", "-n", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        result = subprocess.run(
            [
                "gz", "service", "-s", "/world/default/entity/system/add",
                "--reqtype", "gz.msgs.EntityPlugin_V",
                "--reptype", "gz.msgs.Boolean", "--timeout", "5000",
                "--req", request,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or "true" not in result.stdout.lower():
            state_process.terminate()
            raise RuntimeError(
                "Unable to enable grasp stabilization: "
                f"{result.stderr or result.stdout}"
            )
        try:
            state_output, state_error = state_process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            state_process.terminate()
            state_output, state_error = state_process.communicate(timeout=2)
        if not re.search(r'data:\s*"attached"|\btrue\b', state_output.lower()):
            raise RuntimeError(
                "Gazebo accepted the stabilization plugin but did not report an "
                f"attached joint: {state_error or state_output}"
            )
        return {
            "attached": True,
            "detach_topic": f"{topic_root}/detach",
            "method": "gz_detachable_fixed_joint",
            "parent_model_entity_id": model_id,
            "state_topic": f"{topic_root}/state",
        }

    def release_grasp(self, detach_topic):
        result = subprocess.run(
            [
                "gz", "topic", "-t", detach_topic,
                "-m", "gz.msgs.Empty", "-p", "unused: true",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Unable to release grasp stabilization: {result.stderr or result.stdout}"
            )


class TabletopSceneManager(Node):
    def __init__(self):
        super().__init__("tabletop_scene_manager")
        self.declare_parameter("seed", 0)
        self.declare_parameter("spawn_object_on_start", True)
        self.declare_parameter("object_type", "")
        self.declare_parameter("clutter_min_objects", 4)
        self.declare_parameter("clutter_max_objects", 7)
        requested_seed = int(self.get_parameter("seed").value)
        self.seed = requested_seed if requested_seed else secrets.randbits(31)
        self.random = random.Random(self.seed)
        self.fixed_object_type = str(self.get_parameter("object_type").value).strip()
        if self.fixed_object_type and self.fixed_object_type not in OBJECT_TYPES:
            raise ValueError(
                f"object_type must be one of {', '.join(OBJECT_TYPES)}"
            )
        self.scene = GazeboScene(self)
        self.current_object = None
        self.scene_objects = []
        self.object_index = 0
        self.object_deck = []
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.object_publisher = self.create_publisher(
            String, "/tiago_grasp_sim/object", qos
        )
        self.create_service(Trigger, "/tiago_grasp_sim/reset_scene", self.reset_scene)
        self.create_service(Trigger, "/tiago_grasp_sim/clear_scene", self.clear_scene)
        self.create_service(
            Trigger, "/tiago_grasp_sim/current_object", self.current_object_state
        )
        self.create_service(
            Trigger, "/tiago_grasp_sim/stabilize_grasp", self.stabilize_grasp
        )
        self.create_service(
            Trigger, "/tiago_grasp_sim/release_grasp", self.release_grasp
        )

    def initialize(self):
        self.scene.spawn("grasp_table", table_sdf())
        self.scene.spawn("external_rgb_camera", external_camera_sdf())
        if bool(self.get_parameter("spawn_object_on_start").value):
            self.replace_object()
        self.get_logger().info(
            f"Tabletop simulation ready with scene seed {self.seed}"
        )

    def random_colors(self, count):
        colors = list(COLOR_PALETTE)
        self.random.shuffle(colors)
        return [colors[index % len(colors)] for index in range(count)]

    def next_kind(self):
        if self.fixed_object_type:
            return self.fixed_object_type
        if not self.object_deck:
            self.object_deck = list(OBJECT_TYPES)
            self.random.shuffle(self.object_deck)
        return self.object_deck.pop()

    def replace_object(self):
        self.remove_scene_objects()
        minimum = int(self.get_parameter("clutter_min_objects").value)
        maximum = int(self.get_parameter("clutter_max_objects").value)
        if minimum < 2 or maximum < minimum:
            raise ValueError("clutter object range must satisfy 2 <= min <= max")
        count = self.random.randint(minimum, maximum)
        colors = self.random_colors(count)
        candidates = [
            (x + self.random.uniform(-0.008, 0.008),
             y + self.random.uniform(-0.008, 0.008))
            for x in (0.69, 0.78, 0.87)
            for y in (-0.27, -0.135, 0.0, 0.135, 0.27)
        ]
        target_positions = [
            position for position in candidates
            if position[0] <= 0.79 and abs(position[1]) <= 0.16
        ]
        anchor = self.random.choice(target_positions)
        remaining = [
            position for position in candidates
            if position != anchor
            and math.hypot(
                position[0] - anchor[0], position[1] - anchor[1]
            ) > 0.14
        ]
        self.random.shuffle(remaining)
        selected_positions = [anchor, *remaining[:count - 1]]
        self.random.shuffle(selected_positions)
        objects = []
        for item_index, ((x_pos, y_pos), (color_name, color)) in enumerate(
            zip(selected_positions, colors), start=1
        ):
            self.object_index += 1
            kind = self.next_kind()
            geometry = dict(OBJECT_GEOMETRY[kind])
            yaw = self.random.uniform(-math.pi, math.pi)
            z_pos = TABLE_HEIGHT + geometry["center_height"] + 0.003
            name = f"scene_{self.object_index:04d}_{kind}"
            self.scene.spawn(
                name, object_sdf(kind, name, color), x_pos, y_pos, z_pos, yaw
            )
            objects.append({
                "instance_id": name,
                "name": name,
                "type": kind,
                "color_name": color_name,
                "color_rgb": list(color),
                "pose": {"x": x_pos, "y": y_pos, "z": z_pos, "yaw": yaw},
                "geometry": geometry,
                "table_height": TABLE_HEIGHT,
                "is_target": False,
            })
            time.sleep(0.15)
        target = min(
            objects,
            key=lambda item: math.hypot(
                item["pose"]["x"] - anchor[0],
                item["pose"]["y"] - anchor[1],
            ),
        )
        target["is_target"] = True
        self.current_object = dict(target)
        self.scene_objects = objects
        self.current_object.update({
            "scene_seed": self.seed,
            "scene_index": self.object_index,
            "target_object_id": self.current_object["instance_id"],
            "instruction": f"pick up the {self.current_object['color_name']} {self.current_object['type']}",
            "task": {
                "verb": "pick_up",
                "target_instance_id": self.current_object["instance_id"],
                "category": self.current_object["type"],
                "color": self.current_object["color_name"],
                "referring_expression": f"the {self.current_object['color_name']} {self.current_object['type']}",
            },
            "objects": objects,
            "camera_setup": {
                "model_inputs": ["head_rgb", "external_rgb"],
                "head_rgb_topic": "/head_front_camera/image",
                "external_rgb_topic": "/external_camera/image",
                "external_camera_pose_world": {
                    "x": 0.45, "y": -0.85, "z": 1.35,
                    "roll": 0.0, "pitch": 0.694, "yaw": 1.232,
                },
            },
        })
        self.publish_current_object()
        return self.current_object

    def remove_scene_objects(self):
        for item in self.scene_objects:
            try:
                self.scene.remove(item["name"])
            except RuntimeError as error:
                self.get_logger().warning(str(error))
        self.scene_objects = []
        self.current_object = None

    def publish_current_object(self):
        message = String()
        message.data = json.dumps(self.current_object, sort_keys=True)
        self.object_publisher.publish(message)

    def reset_scene(self, _request, response):
        try:
            response.message = json.dumps(self.replace_object(), sort_keys=True)
            response.success = True
        except Exception as error:
            response.success = False
            response.message = str(error)
        return response

    def clear_scene(self, _request, response):
        try:
            self.remove_scene_objects()
            self.publish_current_object()
            response.success = True
            response.message = "Scene object cleared"
        except Exception as error:
            response.success = False
            response.message = str(error)
        return response

    def current_object_state(self, _request, response):
        try:
            if self.current_object is None:
                raise RuntimeError("There is no object in the scene")
            state = dict(self.current_object)
            state["actual_pose"] = self.scene.pose(self.current_object["name"])
            state["objects"] = [
                {**item, "actual_pose": self.scene.pose(item["name"])}
                for item in self.scene_objects
            ]
            response.success = True
            response.message = json.dumps(state, sort_keys=True)
        except Exception as error:
            response.success = False
            response.message = str(error)
        return response

    def stabilize_grasp(self, _request, response):
        try:
            if self.current_object is None:
                raise RuntimeError("There is no object in the scene")
            if self.current_object.get("grasp_stabilization") is not None:
                raise RuntimeError("The current object is already stabilized")
            enclosure = self.scene.validate_finger_enclosure(
                self.current_object["name"], self.current_object["geometry"]
            )
            details = self.scene.stabilize_grasp(self.current_object["name"])
            details["pre_attachment_enclosure"] = enclosure
            self.current_object["grasp_stabilization"] = details
            response.success = True
            response.message = json.dumps(details, sort_keys=True)
        except Exception as error:
            response.success = False
            response.message = str(error)
        return response

    def release_grasp(self, _request, response):
        try:
            if self.current_object is None:
                raise RuntimeError("There is no object in the scene")
            details = self.current_object.get("grasp_stabilization")
            if details is not None:
                self.scene.release_grasp(details["detach_topic"])
                self.current_object["grasp_stabilization"] = None
            response.success = True
            response.message = "Grasp stabilization released"
        except Exception as error:
            response.success = False
            response.message = str(error)
        return response


def main():
    rclpy.init()
    node = TabletopSceneManager()
    try:
        node.initialize()
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
