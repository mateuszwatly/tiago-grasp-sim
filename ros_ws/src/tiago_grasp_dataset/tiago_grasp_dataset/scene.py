#!/usr/bin/env python3
import colorsys
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
    "apple": {"shape": "sphere", "half_width": 0.038, "center_height": 0.038},
    "cup": {"shape": "cylinder", "half_width": 0.032, "center_height": 0.0475},
    "can": {"shape": "cylinder", "half_width": 0.028, "center_height": 0.0575},
}


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


def object_sdf(kind, name, color):
    material = rgba_text(color)
    surface = surface_xml()
    if kind == "apple":
        geometry = f"""
      <collision name='body_collision'><geometry><sphere><radius>0.038</radius></sphere></geometry>{surface}</collision>
      <visual name='body_visual'><geometry><sphere><radius>0.038</radius></sphere></geometry><material><ambient>{material}</ambient><diffuse>{material}</diffuse></material></visual>
      <visual name='stem_visual'><pose>0 0 0.043 0 0 0</pose><geometry><cylinder><radius>0.004</radius><length>0.022</length></cylinder></geometry><material><ambient>0.20 0.09 0.03 1</ambient><diffuse>0.28 0.12 0.04 1</diffuse></material></visual>"""
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
        for attempt in range(1, 31):
            result = subprocess.run(command, capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                self.node.get_logger().info(f"Spawned {name}")
                return
            if attempt == 30:
                raise RuntimeError(f"Unable to spawn {name}: {result.stderr or result.stdout}")
            time.sleep(2)

    def remove(self, name):
        command = [
            "ros2", "run", "ros_gz_sim", "remove", "--ros-args",
            "-p", "world:=default", "-p", f"entity_name:={name}",
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            raise RuntimeError(f"Unable to remove {name}: {result.stderr or result.stdout}")
        self.node.get_logger().info(f"Removed {name}")

    def pose(self, name):
        result = subprocess.run(
            ["gz", "model", "-m", name, "-p"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Unable to read pose for {name}: {result.stderr or result.stdout}"
            )
        match = re.search(
            r"Pose \[ XYZ \(m\) \] \[ RPY \(rad\) \]:\s*"
            r"\[\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\]\s*"
            r"\[\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\]",
            result.stdout,
        )
        if match is None:
            raise RuntimeError(f"Pose output for {name} was not understood")
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
        result = subprocess.run(
            ["gz", "model", "-m", model_name, "-l", link_name, "-p"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Unable to read pose for {model_name}/{link_name}: "
                f"{result.stderr or result.stdout}"
            )
        link_block = re.search(
            rf"- Name:\s*{re.escape(link_name)}\s.*?"
            r"- Pose \[ XYZ \(m\) \] \[ RPY \(rad\) \]:\s*"
            r"\[\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\]\s*"
            r"\[\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\]",
            result.stdout,
            re.DOTALL,
        )
        if link_block is None:
            raise RuntimeError(
                f"Link pose output for {model_name}/{link_name} was not understood"
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
        object_diameter = 2.0 * float(geometry["half_width"])
        minimum_pad_separation = object_diameter + 0.004
        maximum_pad_separation = object_diameter + 0.018
        validated = (
            horizontal_separation >= minimum_pad_separation
            and horizontal_separation <= maximum_pad_separation
            and abs(separation_vector[2]) <= 0.012
            and midpoint_xy_error <= 0.012
            and midpoint_z_error <= 0.020
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
        if bool(self.get_parameter("spawn_object_on_start").value):
            self.replace_object()
        self.get_logger().info(
            f"Tabletop simulation ready with scene seed {self.seed}"
        )

    def random_color(self):
        hue = self.random.random()
        saturation = self.random.uniform(0.58, 0.90)
        value = self.random.uniform(0.62, 0.92)
        return tuple(float(channel) for channel in colorsys.hsv_to_rgb(hue, saturation, value))

    def next_kind(self):
        if self.fixed_object_type:
            return self.fixed_object_type
        if not self.object_deck:
            self.object_deck = list(OBJECT_TYPES)
            self.random.shuffle(self.object_deck)
        return self.object_deck.pop()

    def replace_object(self):
        if self.current_object is not None:
            self.scene.remove(self.current_object["name"])
        self.object_index += 1
        kind = self.next_kind()
        geometry = dict(OBJECT_GEOMETRY[kind])
        color = self.random_color()
        x_pos = 0.74 + self.random.uniform(-0.018, 0.018)
        y_pos = self.random.uniform(-0.12, 0.12)
        yaw = self.random.uniform(-0.22, 0.22)
        z_pos = TABLE_HEIGHT + geometry["center_height"] + 0.003
        name = f"scene_{self.object_index:04d}_{kind}"
        self.scene.spawn(name, object_sdf(kind, name, color), x_pos, y_pos, z_pos, yaw)
        self.current_object = {
            "scene_seed": self.seed,
            "scene_index": self.object_index,
            "name": name,
            "type": kind,
            "color_rgb": list(color),
            "pose": {"x": x_pos, "y": y_pos, "z": z_pos, "yaw": yaw},
            "geometry": geometry,
            "table_height": TABLE_HEIGHT,
        }
        self.publish_current_object()
        return self.current_object

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
            if self.current_object is not None:
                self.scene.remove(self.current_object["name"])
                self.current_object = None
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
