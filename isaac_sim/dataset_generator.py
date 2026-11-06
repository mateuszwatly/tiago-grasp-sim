#!/usr/bin/env python3
"""Parallel TIAGo clutter-grasp dataset generation for Isaac Sim 6.2.

The simulator runs several independent table scenes in one PhysX process. Each
episode records two RGB-D camera streams through Replicator plus the full arm,
gripper, object and end-effector path. A grasp is labelled successful only
after geometric enclosure and a measured object rise.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import secrets
import signal
import shutil
import time
import sys
import traceback
from typing import Any

from isaacsim import SimulationApp


ARM_JOINTS = [f"arm_{index}_joint" for index in range(1, 8)]
FINGER_JOINTS = ["gripper_left_finger_joint", "gripper_right_finger_joint"]
EE_LINK_CANDIDATES = ["gripper_grasping_frame", "gripper_link", "arm_7_link"]
COLORS = {
    "red": [0.82, 0.10, 0.08],
    "orange": [0.92, 0.34, 0.06],
    "yellow": [0.88, 0.76, 0.08],
    "green": [0.10, 0.68, 0.18],
    "cyan": [0.08, 0.70, 0.72],
    "blue": [0.08, 0.25, 0.84],
    "purple": [0.52, 0.14, 0.76],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--tiago-usd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--renderer", default="RaytracedLighting")
    parser.add_argument("--multi-gpu-rendering", action="store_true")
    parser.add_argument("--profile", choices=["quality", "fast"], default="quality")
    parser.add_argument("--duration-hours", type=float, default=0.0)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parsed = parser.parse_args()
    if parsed.episodes < 1 or parsed.num_envs < 1 or parsed.worker_count < 1:
        parser.error("episodes, num-envs, and worker-count must be positive")
    if not 0 <= parsed.worker_id < parsed.worker_count:
        parser.error("worker-id must be within worker-count")
    if parsed.duration_hours < 0 or parsed.min_free_gb < 0:
        parser.error("duration-hours and min-free-gb cannot be negative")
    return parsed


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_value(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return list(value)


args = parse_args()
base_config = json.loads(args.config.read_text(encoding="utf-8"))
seed = args.seed if args.seed else secrets.randbits(31)
app = SimulationApp(
    {
        "headless": True,
        "renderer": args.renderer,
        "multi_gpu": args.multi_gpu_rendering,
        "active_gpu": 0,
        "physics_gpu": 0,
        "sync_loads": True,
    }
)


import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import GroundPlane  # noqa: E402
from isaacsim.core.cloner import GridCloner  # noqa: E402
from isaacsim.core.prims import Articulation, RigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from pxr import Gf, Sdf, Semantics, UsdGeom, UsdLux, UsdPhysics  # noqa: E402


def set_xform(prim, translation=None, orientation=None, scale=None) -> None:
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    if translation is not None:
        xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    if orientation is not None:
        xform.AddOrientOp().Set(Gf.Quatf(*orientation))
    if scale is not None:
        xform.AddScaleOp().Set(Gf.Vec3f(*scale))


def look_at_matrix(eye, target) -> Gf.Matrix4d:
    return Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0.0, 0.0, 1.0)
    ).GetInverse()


def create_camera(stage, path: str, eye, target, focal_length: float) -> None:
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 20.0))
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(look_at_matrix(eye, target))


def create_table(stage, env_path: str, config: dict[str, Any]) -> None:
    height = float(config["table_height"])
    size = config["table_size"]
    top = UsdGeom.Cube.Define(stage, f"{env_path}/table/top")
    top.CreateSizeAttr(1.0)
    set_xform(
        top.GetPrim(),
        translation=(0.76, 0.0, height - float(size[2]) / 2.0),
        scale=size,
    )
    UsdPhysics.CollisionAPI.Apply(top.GetPrim())
    top.CreateDisplayColorAttr([Gf.Vec3f(0.58, 0.39, 0.20)])

    for index, (x_pos, y_pos) in enumerate(
        ((0.50, -0.40), (0.50, 0.40), (1.02, -0.40), (1.02, 0.40))
    ):
        leg = UsdGeom.Cube.Define(stage, f"{env_path}/table/leg_{index}")
        leg.CreateSizeAttr(1.0)
        set_xform(
            leg.GetPrim(),
            translation=(x_pos, y_pos, (height - float(size[2])) / 2.0),
            scale=(0.055, 0.055, height - float(size[2])),
        )
        UsdPhysics.CollisionAPI.Apply(leg.GetPrim())
        leg.CreateDisplayColorAttr([Gf.Vec3f(0.28, 0.21, 0.14)])


def create_object_slot(stage, path: str, object_assets: dict[str, Any]) -> None:
    root = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(0.15)

    for kind, definition in object_assets.items():
        shape_path = f"{path}/proxy_{kind}"
        collision = definition["collision"]
        if collision["shape"] == "sphere":
            geom = UsdGeom.Sphere.Define(stage, shape_path)
            geom.CreateRadiusAttr(float(collision["radius"]))
        elif collision["shape"] == "cylinder":
            geom = UsdGeom.Cylinder.Define(stage, shape_path)
            geom.CreateAxisAttr("Z")
            geom.CreateRadiusAttr(float(collision["radius"]))
            geom.CreateHeightAttr(float(collision["height"]))
        else:
            raise ValueError(f"Unsupported collision proxy: {collision['shape']}")
        geom.CreateDisplayColorAttr([Gf.Vec3f(0.5, 0.5, 0.5)])
        imageable = UsdGeom.Imageable(geom.GetPrim())
        imageable.MakeInvisible()
        collision_api = UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        collision_api.CreateCollisionEnabledAttr(False)

        appearance = str(definition.get("appearance_usd", "")).strip()
        if appearance:
            appearance_path = f"{path}/appearance_{kind}"
            appearance_prim = UsdGeom.Xform.Define(stage, appearance_path).GetPrim()
            appearance_prim.GetReferences().AddReference(str(Path(appearance).resolve()))
            appearance_transform = definition.get("appearance_transform", {})
            set_xform(
                appearance_prim,
                translation=appearance_transform.get("translation", [0.0, 0.0, 0.0]),
                orientation=appearance_transform.get(
                    "orientation_wxyz", [1.0, 0.0, 0.0, 0.0]
                ),
                scale=appearance_transform.get("scale", [1.0, 1.0, 1.0]),
            )
            UsdGeom.Imageable(appearance_prim).MakeInvisible()

    semantics = Semantics.SemanticsAPI.Apply(root, "Class")
    semantics.CreateSemanticTypeAttr("class")
    semantics.CreateSemanticDataAttr("inactive")
    color_semantics = Semantics.SemanticsAPI.Apply(root, "Color")
    color_semantics.CreateSemanticTypeAttr("color")
    color_semantics.CreateSemanticDataAttr("inactive")


def activate_object(stage, slot_path: str, kind: str, color_name: str, assets) -> None:
    slot = stage.GetPrimAtPath(slot_path)
    UsdPhysics.MassAPI(slot).GetMassAttr().Set(float(assets[kind]["mass_kg"]))
    for candidate in assets:
        proxy = stage.GetPrimAtPath(f"{slot_path}/proxy_{candidate}")
        enabled = candidate == kind
        UsdPhysics.CollisionAPI(proxy).GetCollisionEnabledAttr().Set(enabled)
        imageable = UsdGeom.Imageable(proxy)
        imageable.MakeVisible() if enabled and not assets[kind].get("appearance_usd") else imageable.MakeInvisible()
        appearance = stage.GetPrimAtPath(f"{slot_path}/appearance_{candidate}")
        if appearance.IsValid():
            appearance_imageable = UsdGeom.Imageable(appearance)
            appearance_imageable.MakeVisible() if enabled else appearance_imageable.MakeInvisible()
    proxy_geom = UsdGeom.Gprim(stage.GetPrimAtPath(f"{slot_path}/proxy_{kind}"))
    proxy_geom.GetDisplayColorAttr().Set([Gf.Vec3f(*COLORS[color_name])])
    Semantics.SemanticsAPI.Get(slot, "Class").GetSemanticDataAttr().Set(kind)
    Semantics.SemanticsAPI.Get(slot, "Color").GetSemanticDataAttr().Set(color_name)


def find_relative_prim(stage, root_path: str, names: list[str], require_articulation=False):
    root = Sdf.Path(root_path)
    for prim in stage.Traverse():
        path = prim.GetPath()
        if not path.HasPrefix(root):
            continue
        if require_articulation and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return path.MakeRelativePath(root).pathString
        if not require_articulation and prim.GetName() in names:
            return path.MakeRelativePath(root).pathString
    requested = "articulation root" if require_articulation else f"one of {names}"
    raise RuntimeError(f"TIAGo USD does not contain {requested}")


def quaternion_error(target: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    target = target / torch.linalg.norm(target, dim=-1, keepdim=True).clamp_min(1e-9)
    current = current / torch.linalg.norm(current, dim=-1, keepdim=True).clamp_min(1e-9)
    tw, tx, ty, tz = target.unbind(-1)
    cw, cx, cy, cz = current.unbind(-1)
    error = torch.stack(
        (
            -tw * cx + tx * cw - ty * cz + tz * cy,
            -tw * cy + tx * cz + ty * cw - tz * cx,
            -tw * cz - tx * cy + ty * cx + tz * cw,
        ),
        dim=-1,
    )
    scalar = tw * cw + tx * cx + ty * cy + tz * cz
    return 2.0 * torch.where((scalar < 0.0).unsqueeze(-1), -error, error)


def quaternion_multiply(first, second) -> np.ndarray:
    aw, ax, ay, az = np.asarray(first, dtype=float)
    bw, bx, by, bz = np.asarray(second, dtype=float)
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def quaternion_rotate(quaternion, vector) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=float)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
    conjugate = quaternion * np.asarray([1.0, -1.0, -1.0, -1.0])
    pure_vector = np.asarray([0.0, *np.asarray(vector, dtype=float)])
    return quaternion_multiply(
        quaternion_multiply(quaternion, pure_vector), conjugate
    )[1:]


def quaternion_delta_rotvec(current, following) -> list[float]:
    current = np.asarray(current, dtype=float)
    following = np.asarray(following, dtype=float)
    current /= max(float(np.linalg.norm(current)), 1e-12)
    following /= max(float(np.linalg.norm(following)), 1e-12)
    delta = quaternion_multiply(
        following, current * np.asarray([1.0, -1.0, -1.0, -1.0])
    )
    if delta[0] < 0.0:
        delta = -delta
    vector_norm = float(np.linalg.norm(delta[1:]))
    if vector_norm <= 1e-9:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(vector_norm, max(float(delta[0]), 1e-12))
    return (angle * delta[1:] / vector_norm).tolist()


class IsaacDatasetGenerator:
    def __init__(self) -> None:
        self.config = dict(base_config)
        if args.profile == "fast":
            self.config.update(camera_width=640, camera_height=480,
                               capture_every_control_steps=6)
        self.started_at = time.monotonic()
        self.stop_requested = False
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        self.config["num_envs"] = args.num_envs
        self.config["episodes"] = args.episodes
        self.object_assets = json.loads(args.objects.read_text(encoding="utf-8"))
        self.random = random.Random(seed + args.worker_id * 1_000_003)
        self.device = "cpu"
        self.stage = omni.usd.get_context().get_stage()
        self.frame_index = 0
        self.render_writers = []
        self.render_products = []
        self.env_paths: list[str] = []
        self.env_origins = None
        self.active_episode_ids: list[int] = []
        self.trajectories: list[list[dict[str, Any]]] = []
        self.episode_records: list[dict[str, Any]] = []
        session_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.session_dir = args.output / f"isaac_{session_stamp}_worker{args.worker_id}_seed{seed}"
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.events_path = self.session_dir / "events.jsonl"
        (self.session_dir / "config.json").write_text(json.dumps(
            {"generator": self.config, "objects": self.object_assets,
             "seed": seed, "profile": args.profile}, indent=2))
        self.log(
            "session_started",
            episodes=args.episodes,
            num_envs=args.num_envs,
            worker_id=args.worker_id,
            worker_count=args.worker_count,
            seed=seed,
        )

    def request_stop(self, *_args) -> None:
        self.stop_requested = True

    def log(self, event: str, **fields) -> None:
        record = {"wall_time": utc_now(), "event": event, **fields}
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[isaac-dataset] {event}: {fields}", flush=True)

    def create_world(self) -> None:
        SimulationManager.set_device("cpu")
        settings = carb.settings.get_settings()
        settings.set("/renderer/multiGpu/enabled", args.multi_gpu_rendering)
        settings.set("/renderer/multiGpu/maxGpuCount", 2)
        settings.set("/physics/cudaDevice", 0)
        settings.set("/physics/useGpu", False)
        settings.set("/persistent/physics/useFabric", False)

        self.world = World(
            physics_dt=float(self.config["physics_dt"]),
            rendering_dt=float(self.config["rendering_dt"]),
            backend="torch",
            device=self.device,
        )
        self.world.scene.add(GroundPlane("/World/groundPlane", size=100.0))
        dome_light = UsdLux.DomeLight.Define(self.stage, "/World/domeLight")
        dome_light.CreateIntensityAttr(1000.0)
        dome_light.CreateColorTemperatureAttr(5500.0)
        dome_light.CreateEnableColorTemperatureAttr(True)
        source_env = "/World/envs/env_0"
        UsdGeom.Scope.Define(self.stage, "/World/envs")
        UsdGeom.Xform.Define(self.stage, source_env)
        add_reference_to_stage(str(args.tiago_usd.resolve()), f"{source_env}/Tiago")
        tiago_prim = self.stage.GetPrimAtPath(f"{source_env}/Tiago")
        physics_variant = tiago_prim.GetVariantSets().GetVariantSet("Physics")
        available_physics_variants = physics_variant.GetVariantNames()
        if "physx" in available_physics_variants:
            physics_variant.SetVariantSelection("physx")
        elif available_physics_variants:
            raise RuntimeError(
                "TIAGo USD does not provide the required PhysX variant; "
                f"available variants are {available_physics_variants}"
            )
        robot_root = Sdf.Path(f"{source_env}/Tiago")
        # URDF damping (and Newton attributes) is not a PhysX position servo.
        # Author gains before cloning/reset so every environment gets active drives.
        for prim in self.stage.Traverse():
            if not prim.GetPath().HasPrefix(robot_root):
                continue
            if prim.IsA(UsdPhysics.FixedJoint) and prim.GetName() == "root_joint":
                # A world joint has no body0; the imported Xform is not a rigid body.
                UsdPhysics.Joint(prim).GetBody0Rel().ClearTargets(True)
            if prim.IsA(UsdPhysics.RevoluteJoint):
                axis, stiffness, damping = "angular", 200.0, 10.0
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                axis, stiffness, damping = "linear", 200000.0, 3000.0
            else:
                continue
            drive = UsdPhysics.DriveAPI.Apply(prim, axis)
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(stiffness)
            drive.CreateDampingAttr(damping)
            drive.CreateTargetVelocityAttr(0.0)

        enabled_robot_colliders = []
        disabled_robot_colliders = []
        for prim in self.stage.Traverse():
            if not prim.GetPath().HasPrefix(robot_root):
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            prim_path = str(prim.GetPath())
            is_finger_pad = (
                "gripper_left_finger_link" in prim_path
                or "gripper_right_finger_link" in prim_path
            ) and prim.GetName() == "box"
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(is_finger_pad)
            destination = (
                enabled_robot_colliders if is_finger_pad else disabled_robot_colliders
            )
            destination.append(prim_path)
        self.log(
            "robot_colliders_configured",
            enabled=enabled_robot_colliders,
            disabled_count=len(disabled_robot_colliders),
        )
        create_table(self.stage, source_env, self.config)
        for slot in range(int(self.config["clutter_max_objects"])):
            create_object_slot(
                self.stage, f"{source_env}/objects/slot_{slot}", self.object_assets
            )
        create_camera(
            self.stage,
            f"{source_env}/cameras/head",
            (0.28, 0.0, 1.44),
            (0.76, 0.0, 0.61),
            17.5,
        )
        create_camera(
            self.stage,
            f"{source_env}/cameras/external",
            (0.45, -0.85, 1.35),
            (0.76, 0.0, 0.61),
            24.0,
        )

        cloner = GridCloner(spacing=float(self.config["environment_spacing"]))
        cloner.define_base_env("/World/envs")
        self.env_paths = cloner.generate_paths("/World/envs/env", args.num_envs)
        if args.num_envs == 1:
            self.env_origins = np.zeros((1, 3), dtype=np.float32)
        else:
            self.env_origins = np.asarray(
                cloner.clone(
                    source_prim_path=source_env,
                    prim_paths=self.env_paths,
                    replicate_physics=False,
                    base_env_path="/World/envs",
                    root_path="/World/envs/env_",
                    copy_from_source=True,
                    enable_env_ids=True,
                ),
                dtype=np.float32,
            )
        # The grid spacing keeps environments physically disjoint. Applying
        # collision groups to this nested mobile-manipulator articulation makes
        # PhysX attempt to attach the imported collision shapes more than once.

        articulation_rel = find_relative_prim(
            self.stage, f"{source_env}/Tiago", [], require_articulation=True
        )
        ee_rel = find_relative_prim(
            self.stage, f"{source_env}/Tiago", EE_LINK_CANDIDATES
        )
        left_finger_rel = find_relative_prim(
            self.stage, f"{source_env}/Tiago", ["gripper_left_finger_link"]
        )
        right_finger_rel = find_relative_prim(
            self.stage, f"{source_env}/Tiago", ["gripper_right_finger_link"]
        )
        articulation_paths = [
            f"{env}/Tiago/{articulation_rel}".rstrip("/") for env in self.env_paths
        ]
        ee_paths = [f"{env}/Tiago/{ee_rel}".rstrip("/") for env in self.env_paths]
        self.ee_paths = ee_paths
        left_finger_paths = [
            f"{env}/Tiago/{left_finger_rel}".rstrip("/") for env in self.env_paths
        ]
        right_finger_paths = [
            f"{env}/Tiago/{right_finger_rel}".rstrip("/") for env in self.env_paths
        ]
        self.robots = Articulation(
            prim_paths_expr=articulation_paths,
            name="tiago_view",
            reset_xform_properties=False,
        )
        self.world.scene.add(self.robots)
        self.object_views = []
        for slot in range(int(self.config["clutter_max_objects"])):
            view = RigidPrim(
                prim_paths_expr=[f"{env}/objects/slot_{slot}" for env in self.env_paths],
                name=f"object_slot_{slot}",
                reset_xform_properties=False,
            )
            self.world.scene.add(view)
            self.object_views.append(view)

        self.world.reset()
        self.reset_joint_positions = self.robots.get_joint_positions().clone()
        self.arm_indices = [self.robots.get_dof_index(name) for name in ARM_JOINTS]
        self.finger_indices = [self.robots.get_dof_index(name) for name in FINGER_JOINTS]
        self.ee_link_name = self.stage.GetPrimAtPath(ee_paths[0]).GetName()
        self.ee_link_index = self.robots.get_link_index(self.ee_link_name)
        self.left_finger_link_index = self.robots.get_link_index(
            self.stage.GetPrimAtPath(left_finger_paths[0]).GetName()
        )
        self.right_finger_link_index = self.robots.get_link_index(
            self.stage.GetPrimAtPath(right_finger_paths[0]).GetName()
        )
        limits = self.robots.get_dof_limits()
        self.lower_limits = limits[:, :, 0]
        self.upper_limits = limits[:, :, 1]
        self.log(
            "world_ready",
            articulation_paths=articulation_paths,
            end_effector_paths=ee_paths,
            dof_names=list(self.robots.dof_names),
            body_names=list(self.robots.body_names),
        )
        self.setup_replicator()

    def get_link_world_poses(self, link_index: int):
        transforms = self.robots._physics_view.get_link_transforms()
        positions = transforms[:, link_index, :3]
        xyzw = transforms[:, link_index, 3:7]
        orientations = torch.cat((xyzw[:, 3:4], xyzw[:, :3]), dim=-1)
        return positions, orientations

    def setup_replicator(self) -> None:
        width = int(self.config["camera_width"])
        height = int(self.config["camera_height"])
        for env_index, env_path in enumerate(self.env_paths):
            for camera_name in ("head", "external"):
                camera_path = f"{env_path}/cameras/{camera_name}"
                render_product = rep.create.render_product(camera_path, (width, height))
                writer = rep.WriterRegistry.get("BasicWriter")
                writer.initialize(
                    output_dir=str(
                        self.session_dir / "cameras" / f"env_{env_index:02d}" / camera_name
                    ),
                    rgb=True,
                    distance_to_image_plane=True,
                    semantic_segmentation=True,
                    colorize_semantic_segmentation=False,
                    instance_id_segmentation=True,
                    colorize_instance_id_segmentation=False,
                    bounding_box_2d_tight=True,
                    bounding_box_3d=True,
                    camera_params=True,
                )
                writer.attach([render_product])
                self.render_products.append(render_product)
                self.render_writers.append(writer)

    def configure_batch(self, episode_ids: list[int]) -> None:
        self.active_episode_ids = episode_ids
        self.trajectories = [[] for _ in episode_ids]
        self.episode_records = []
        count_positions = [
            (x, y)
            for x in (0.69, 0.78, 0.87)
            for y in (-0.27, -0.135, 0.0, 0.135, 0.27)
        ]
        object_positions = [np.zeros((len(episode_ids), 3), dtype=np.float32) for _ in self.object_views]
        object_orientations = [
            np.tile(
                np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                (len(episode_ids), 1),
            )
            for _ in self.object_views
        ]
        active_counts = []
        for env_index, episode_id in enumerate(episode_ids):
            env_random = random.Random(seed + episode_id * 104729)
            count = env_random.randint(
                int(self.config["clutter_min_objects"]),
                int(self.config["clutter_max_objects"]),
            )
            active_counts.append(count)
            anchors = [p for p in count_positions if p[0] <= 0.79 and abs(p[1]) <= 0.16]
            anchor = env_random.choice(anchors)
            candidates = [
                p
                for p in count_positions
                if p != anchor
                and math.hypot(p[0] - anchor[0], p[1] - anchor[1])
                > float(self.config["target_clearance"])
            ]
            env_random.shuffle(candidates)
            selected = [anchor, *candidates[: count - 1]]
            kinds = [env_random.choice(sorted(self.object_assets)) for _ in range(count)]
            color_names = list(COLORS)
            env_random.shuffle(color_names)
            scene_objects = []
            for slot, position in enumerate(selected):
                kind = kinds[slot]
                color_name = color_names[slot % len(color_names)]
                collision = self.object_assets[kind]["collision"]
                center_height = (
                    float(collision["radius"])
                    if collision["shape"] == "sphere"
                    else float(collision["height"]) / 2.0
                )
                local = np.array(
                    [
                        position[0] + env_random.uniform(-0.008, 0.008),
                        position[1] + env_random.uniform(-0.008, 0.008),
                        float(self.config["table_height"]) + center_height + 0.002,
                    ],
                    dtype=np.float32,
                )
                world_position = self.env_origins[env_index] + local
                object_positions[slot][env_index] = world_position
                object_yaw = env_random.uniform(-math.pi, math.pi)
                object_orientations[slot][env_index] = np.asarray(
                    [math.cos(object_yaw / 2.0), 0.0, 0.0, math.sin(object_yaw / 2.0)],
                    dtype=np.float32,
                )
                slot_path = f"{self.env_paths[env_index]}/objects/slot_{slot}"
                activate_object(self.stage, slot_path, kind, color_name, self.object_assets)
                scene_objects.append(
                    {
                        "instance_id": f"env{env_index}_slot{slot}",
                        "slot": slot,
                        "type": kind,
                        "color": color_name,
                        "is_target": slot == 0,
                        "initial_position": local.tolist(),
                        "initial_orientation_wxyz": object_orientations[slot][
                            env_index
                        ].tolist(),
                        "collision_proxy": collision,
                        "appearance_usd": self.object_assets[kind].get("appearance_usd", ""),
                    }
                )
            far_position = self.env_origins[env_index] + np.array([0.0, 0.0, -2.0])
            for slot in range(count, len(self.object_views)):
                object_positions[slot][env_index] = far_position
                for kind in self.object_assets:
                    proxy = self.stage.GetPrimAtPath(
                        f"{self.env_paths[env_index]}/objects/slot_{slot}/proxy_{kind}"
                    )
                    UsdPhysics.CollisionAPI(proxy).GetCollisionEnabledAttr().Set(False)
                    UsdGeom.Imageable(proxy).MakeInvisible()
                    appearance = self.stage.GetPrimAtPath(
                        f"{self.env_paths[env_index]}/objects/slot_{slot}/appearance_{kind}"
                    )
                    if appearance.IsValid():
                        UsdGeom.Imageable(appearance).MakeInvisible()

            target = scene_objects[0]
            target_world = object_positions[0][env_index]
            yaw_offset = env_random.uniform(-math.pi, math.pi)
            yaw_candidates = [
                yaw_offset + index * (math.pi / 8.0) for index in range(16)
            ]
            approach_scores = []
            for candidate_yaw in yaw_candidates:
                candidate_approach = np.asarray(
                    [math.cos(candidate_yaw), math.sin(candidate_yaw)], dtype=float
                )
                wrist_xy = target_world[:2] - candidate_approach * 0.241575
                distractor_clearances = [
                    float(
                        np.linalg.norm(
                            wrist_xy - object_positions[slot][env_index, :2]
                        )
                    )
                    for slot in range(1, count)
                ]
                approach_scores.append(
                    min(distractor_clearances, default=float("inf"))
                )
            best_candidate = int(np.argmax(approach_scores))
            yaw = yaw_candidates[best_candidate]
            approach = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
            grasp_position = target_world - approach * 0.241575
            grasp_position[2] = max(
                float(self.config["table_height"]) + self.env_origins[env_index, 2] + 0.065,
                # Upper-side grasp retains pad overlap while avoiding the
                # poorly tracked low-wrist pose of the imported TIAGo.
                target_world[2] + min(0.035, 0.5 * float(
                    target["collision_proxy"].get("height", 2 * target["collision_proxy"]["radius"])
                )),
            )
            pregrasp = grasp_position.copy()
            pregrasp[2] += float(self.config["pregrasp_height"])
            lift = grasp_position.copy()
            lift[2] += float(self.config["lift_distance"])
            # arm_7 is the imported wrist frame, not Gazebo's grasp tool frame.
            # The merged gripper pads lie 0.241575 m along wrist +Z.
            c, s = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
            orientation = [c / math.sqrt(2), -s / math.sqrt(2),
                           c / math.sqrt(2), s / math.sqrt(2)]
            target["task"] = {
                "verb": "pick_up",
                "target_instance_id": target["instance_id"],
                "category": target["type"],
                "color": target["color"],
                "referring_expression": f"the {target['color']} {target['type']}",
            }
            self.episode_records.append(
                {
                    "schema_version": 3,
                    "episode_id": episode_id,
                    "worker_id": args.worker_id,
                    "environment_index": env_index,
                    "seed": seed + episode_id * 104729,
                    "objects": scene_objects,
                    "target": target,
                    "instruction": f"pick up the {target['color']} {target['type']}",
                    "waypoints": {
                        "pregrasp": pregrasp.tolist(),
                        "grasp": grasp_position.tolist(),
                        "lift": lift.tolist(),
                        "orientation_wxyz": orientation,
                    },
                    "approach_selection": {
                        "method": "maximum_clutter_clearance_over_16_horizontal_yaws",
                        "selected_yaw": yaw,
                        "minimum_wrist_clearance": approach_scores[best_candidate],
                    },
                    "camera_inputs": ["head_rgb", "external_rgb"],
                    "camera_modalities": [
                        "rgb",
                        "distance_to_image_plane",
                        "semantic_segmentation",
                        "instance_id_segmentation",
                        "bounding_box_2d_tight",
                        "bounding_box_3d",
                        "camera_params",
                    ],
                    "base_motion_commanded": False,
                    "ik_solver": "batched damped-least-squares Jacobian IK",
                    "obstacle_evasion": (
                        "clutter-aware horizontal yaw, overhead pregrasp, and vertical descent; "
                        "no full-arm motion planner"
                    ),
                    "status": "running",
                    "success": False,
                    "replicator_frames": [],
                }
            )

        for slot, view in enumerate(self.object_views):
            padded = np.zeros((args.num_envs, 3), dtype=np.float32)
            padded_orientations = np.zeros((args.num_envs, 4), dtype=np.float32)
            padded_orientations[:, 0] = 1.0
            padded[: len(episode_ids)] = object_positions[slot]
            padded_orientations[: len(episode_ids)] = object_orientations[slot]
            if len(episode_ids) < args.num_envs:
                padded[len(episode_ids) :] = self.env_origins[len(episode_ids) :] + [0, 0, -2]
            view.set_world_poses(
                positions=torch.as_tensor(padded, device=self.device),
                orientations=torch.as_tensor(
                    padded_orientations, device=self.device
                ),
            )
            view.set_velocities(
                torch.zeros((args.num_envs, 6), device=self.device)
            )

        q = self.reset_joint_positions.clone()
        anchor = torch.tensor(
            [0.17, -0.27, -1.35, 2.28, 1.22, 0.92, 1.57],
            device=q.device,
            dtype=q.dtype,
        )
        for env_index, episode_id in enumerate(episode_ids):
            generator = torch.Generator(device=q.device)
            generator.manual_seed(seed + episode_id)
            jitter = (torch.rand(7, generator=generator, device=q.device) - 0.5) * 0.06
            q[env_index, self.arm_indices] = anchor + jitter
            q[env_index, self.finger_indices] = 0.044
        self.joint_targets = q.clone()
        self.robots.set_joint_positions(q)
        self.robots.set_joint_velocities(torch.zeros_like(q))
        self.robots.set_joint_position_targets(q)
        for _ in range(int(self.config["settle_steps"])):
            self.world.step(render=False)

        initial_positions, _ = self.object_views[0].get_world_poses()
        for env_index, record in enumerate(self.episode_records):
            record["initial_target_world_position"] = json_value(initial_positions[env_index])
            self.log("episode_started", episode=record["episode_id"], target=record["target"])

    def capture(self, phase: str) -> None:
        rep.orchestrator.step(rt_subframes=1, pause_timeline=False)
        q = self.robots.get_joint_positions()
        qd = self.robots.get_joint_velocities()
        ee_positions, ee_orientations = self.get_link_world_poses(self.ee_link_index)
        object_positions, object_orientations = self.object_views[0].get_world_poses()
        for env_index, record in enumerate(self.episode_records):
            sample = {
                "step_index": len(self.trajectories[env_index]),
                "physics_step": int(self.world.current_time_step_index),
                "replicator_frame": self.frame_index,
                "phase": phase,
                "end_effector_position": json_value(ee_positions[env_index]),
                "end_effector_orientation_wxyz": json_value(ee_orientations[env_index]),
                "joint_names": list(self.robots.dof_names),
                "joint_positions": json_value(q[env_index]),
                "joint_velocities": json_value(qd[env_index]),
                "target_object_position": json_value(object_positions[env_index]),
                "target_object_orientation_wxyz": json_value(object_orientations[env_index]),
                "camera_paths": {
                    "head": f"cameras/env_{env_index:02d}/head",
                    "external": f"cameras/env_{env_index:02d}/external",
                },
            }
            self.trajectories[env_index].append(sample)
            record["replicator_frames"].append(self.frame_index)
        self.frame_index += 1

    def move_end_effectors(self, targets, orientations, phase: str, active=None) -> list[bool]:
        count = len(self.episode_records)
        target_pos = torch.as_tensor(targets, device=self.device, dtype=torch.float32)
        target_quat = torch.as_tensor(orientations, device=self.device, dtype=torch.float32)
        active = torch.as_tensor(
            [True] * count if active is None else active, device=self.device, dtype=torch.bool
        )
        converged = torch.zeros(count, dtype=torch.bool, device=self.device)
        if not bool(torch.any(active)):
            return [False] * count
        eye = torch.eye(6, device=self.device).unsqueeze(0).repeat(count, 1, 1)
        damping = float(self.config["ik_damping"])
        gain = float(self.config["ik_gain"])
        for step in range(int(self.config["phase_timeout_steps"])):
            positions, quaternions = self.get_link_world_poses(self.ee_link_index)
            positions = positions[:count]
            quaternions = quaternions[:count]
            position_error = target_pos - positions
            rotation_error = quaternion_error(target_quat, quaternions)
            converged = (
                torch.linalg.norm(position_error, dim=-1)
                <= float(self.config["ik_position_tolerance"])
            ) & (
                torch.linalg.norm(rotation_error, dim=-1)
                <= float(self.config["ik_orientation_tolerance"])
            )
            if bool(torch.all(converged | ~active)):
                break
            jacobians = self.robots.get_jacobians()[:count]
            fixed_base = jacobians.shape[1] == len(self.robots.body_names) - 1
            jacobian_index = self.ee_link_index - int(fixed_base)
            column_offset = 0 if fixed_base else 6
            jacobian = jacobians[:, jacobian_index, :, :][:, :, [i + column_offset for i in self.arm_indices]]
            # Bound Cartesian corrections so descent does not sweep a finger
            # sideways through the object while solving the final waypoint.
            position_error = position_error * torch.clamp(
                0.01 / torch.linalg.norm(position_error, dim=-1, keepdim=True).clamp_min(1e-9), max=1.0)
            rotation_error = rotation_error * torch.clamp(
                0.05 / torch.linalg.norm(rotation_error, dim=-1, keepdim=True).clamp_min(1e-9), max=1.0)
            error = torch.cat((position_error, rotation_error), dim=-1).unsqueeze(-1)
            jj_t = jacobian @ jacobian.transpose(1, 2)
            delta = jacobian.transpose(1, 2) @ torch.linalg.solve(
                jj_t + (damping * damping) * eye, error
            )
            q = self.robots.get_joint_positions()
            arm_q = q[:count, self.arm_indices] + torch.clamp(gain * delta.squeeze(-1), -0.08, 0.08)
            arm_q = torch.maximum(
                torch.minimum(arm_q, self.upper_limits[:count, self.arm_indices]),
                self.lower_limits[:count, self.arm_indices],
            )
            q[:count, self.arm_indices] = torch.where(
                (converged | ~active).unsqueeze(-1), self.joint_targets[:count, self.arm_indices], arm_q
            )
            self.joint_targets[:count, self.arm_indices] = q[:count, self.arm_indices]
            self.robots.set_joint_position_targets(self.joint_targets)
            for _ in range(int(self.config["control_decimation"])):
                self.world.step(render=False)
            if step % int(self.config["capture_every_control_steps"]) == 0:
                self.capture(phase)
        positions, quaternions = self.get_link_world_poses(self.ee_link_index)
        converged = (
            torch.linalg.norm(target_pos - positions[:count], dim=-1)
            <= float(self.config["ik_position_tolerance"])
        ) & (
            torch.linalg.norm(quaternion_error(target_quat, quaternions[:count]), dim=-1)
            <= float(self.config["ik_orientation_tolerance"])
        )
        self.log("motion_phase_finished", phase=phase,
                 converged=json_value(converged & active),
                 joint_positions=json_value(self.robots.get_joint_positions()),
                 joint_targets=json_value(self.joint_targets))
        return json_value(converged & active)

    def command_gripper(self, openings: list[float], phase: str, active=None) -> None:
        if active is not None and not any(active):
            return
        q = self.joint_targets.clone()
        target = torch.as_tensor(openings, device=q.device, dtype=q.dtype).unsqueeze(-1)
        if active is not None:
            mask = torch.as_tensor(active, device=q.device).unsqueeze(-1)
            target = torch.where(mask, target, q[:len(openings), self.finger_indices])
        q[: len(openings), self.finger_indices] = target
        self.joint_targets = q
        self.robots.set_joint_position_targets(q)
        for step in range(30):
            self.world.step(render=False)
            if step % 2 == 0:
                self.capture(phase)

    def attach_targets(self, active=None) -> list[bool]:
        object_positions, object_orientations = self.object_views[0].get_world_poses()
        left_positions, left_orientations = self.get_link_world_poses(
            self.left_finger_link_index
        )
        right_positions, right_orientations = self.get_link_world_poses(
            self.right_finger_link_index
        )
        attached = []
        for env_index, record in enumerate(self.episode_records):
            object_position = np.asarray(
                json_value(object_positions[env_index]), dtype=float
            )
            left_link_position = np.asarray(
                json_value(left_positions[env_index]), dtype=float
            )
            right_link_position = np.asarray(
                json_value(right_positions[env_index]), dtype=float
            )
            left_pad = left_link_position + quaternion_rotate(
                json_value(left_orientations[env_index]), [0.0, 0.0, -0.165]
            )
            right_pad = right_link_position + quaternion_rotate(
                json_value(right_orientations[env_index]), [0.0, 0.0, -0.165]
            )
            midpoint = 0.5 * (left_pad + right_pad)
            separation_vector = left_pad - right_pad
            separation = float(np.linalg.norm(separation_vector))
            horizontal_separation = float(np.linalg.norm(separation_vector[:2]))
            midpoint_xy_error = float(
                np.linalg.norm(midpoint[:2] - object_position[:2])
            )
            midpoint_z_error = abs(float(midpoint[2] - object_position[2]))
            collision = record["target"]["collision_proxy"]
            radius = float(collision["radius"])
            center_height = (
                radius
                if collision["shape"] == "sphere"
                else 0.5 * float(collision["height"])
            )
            vertical_overlap = center_height + 0.025 - midpoint_z_error
            object_diameter = 2.0 * radius
            minimum_separation = object_diameter + 0.004
            maximum_separation = object_diameter + 0.018
            valid = (
                (active is None or active[env_index])
                and minimum_separation <= horizontal_separation <= maximum_separation
                and abs(float(separation_vector[2])) <= 0.012
                and midpoint_xy_error <= 0.012
                and vertical_overlap >= 0.005
            )
            record["pre_attachment_enclosure"] = {
                "validated": valid,
                "method": "actual_finger_pad_geometric_enclosure",
                "object_position": object_position.tolist(),
                "left_finger_link_position": left_link_position.tolist(),
                "right_finger_link_position": right_link_position.tolist(),
                "left_pad_center": left_pad.tolist(),
                "right_pad_center": right_pad.tolist(),
                "pad_midpoint": midpoint.tolist(),
                "pad_separation": separation,
                "horizontal_pad_separation": horizontal_separation,
                "object_diameter": object_diameter,
                "minimum_pad_separation": minimum_separation,
                "maximum_pad_separation": maximum_separation,
                "midpoint_xy_error": midpoint_xy_error,
                "midpoint_z_error": midpoint_z_error,
                "vertical_pad_object_overlap": vertical_overlap,
            }
            attached.append(valid)
            if not valid:
                continue
            ee_path = self.ee_paths[env_index]
            body0 = self.stage.GetPrimAtPath(ee_path)
            while body0.IsValid() and not body0.HasAPI(UsdPhysics.RigidBodyAPI):
                body0 = body0.GetParent()
            if not body0.IsValid():
                attached[-1] = False
                continue
            joint_path = f"{self.env_paths[env_index]}/grasp_attachment"
            self.stage.RemovePrim(joint_path)
            joint = UsdPhysics.FixedJoint.Define(self.stage, joint_path)
            joint.CreateBody0Rel().SetTargets([body0.GetPath()])
            joint.CreateBody1Rel().SetTargets(
                [Sdf.Path(f"{self.env_paths[env_index]}/objects/slot_0")]
            )
            body_transform = Gf.Transform(
                UsdGeom.XformCache().GetLocalToWorldTransform(body0)
            )
            body_position = np.asarray(body_transform.GetTranslation(), dtype=float)
            body_quaternion = body_transform.GetRotation().GetQuat()
            body_orientation = np.asarray(
                [body_quaternion.GetReal(), *body_quaternion.GetImaginary()],
                dtype=float,
            )
            object_orientation = np.asarray(
                json_value(object_orientations[env_index]), dtype=float
            )
            inverse_body = body_orientation * np.asarray(
                [1.0, -1.0, -1.0, -1.0]
            )
            local_position = quaternion_rotate(
                inverse_body, object_position - body_position
            )
            local_orientation = quaternion_multiply(
                inverse_body, object_orientation
            )
            joint.CreateLocalPos0Attr(Gf.Vec3f(*local_position.tolist()))
            joint.CreateLocalRot0Attr(Gf.Quatf(*local_orientation.tolist()))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        return attached

    def finish_batch(self, phase_results: dict[str, list[bool]]) -> None:
        final_positions, _ = self.object_views[0].get_world_poses()
        final_positions = final_positions.clone()
        # Replicator completion can stop the timeline and invalidate tensor views.
        # Snapshot the physical outcome first, then drain writes before indexing.
        rep.orchestrator.wait_until_complete()
        for env_index, record in enumerate(self.episode_records):
            initial = np.asarray(record["initial_target_world_position"], dtype=float)
            final = np.asarray(json_value(final_positions[env_index]), dtype=float)
            rise = float(final[2] - initial[2])
            success = (
                all(result[env_index] for result in phase_results.values())
                and rise >= float(self.config["minimum_verified_lift"])
            )
            record.update(
                {
                    "status": "completed" if success else "failed",
                    "success": success,
                    "simulated_lift_verified": success,
                    "final_target_world_position": final.tolist(),
                    "object_rise": rise,
                    "minimum_verified_lift": float(
                        self.config["minimum_verified_lift"]
                    ),
                    "failure_phase": next(
                        (key for key, value in phase_results.items() if not value[env_index]),
                        None if success else "lift_verification",
                    ),
                    "phase_results": {
                        key: bool(value[env_index])
                        for key, value in phase_results.items()
                    },
                    "trajectory_samples": len(self.trajectories[env_index]),
                }
            )
            trajectory = self.trajectories[env_index]
            for sample_index, sample in enumerate(trajectory):
                following = trajectory[min(sample_index + 1, len(trajectory) - 1)]
                delta_position = (
                    np.asarray(following["end_effector_position"])
                    - np.asarray(sample["end_effector_position"])
                ).tolist()
                delta_rotation = quaternion_delta_rotvec(
                    sample["end_effector_orientation_wxyz"],
                    following["end_effector_orientation_wxyz"],
                )
                sample["action"] = {
                    "representation": "delta_end_effector_rotvec_gripper_terminate",
                    "values": delta_position
                    + delta_rotation
                    + [
                        float(following["joint_positions"][self.finger_indices[0]]),
                        float(sample_index == len(trajectory) - 1),
                    ],
                }
                sample["supervision"] = {
                    "target_instance_id": record["target"]["instance_id"],
                    "success": success,
                }
            episode_dir = self.session_dir / "episodes" / f"episode_{record['episode_id']:06d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            (episode_dir / "metadata.json").write_text(
                json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
            )
            with (episode_dir / "trajectory.jsonl").open("w", encoding="utf-8") as stream:
                for sample in trajectory:
                    stream.write(json.dumps(sample, sort_keys=True) + "\n")
            self.log(
                "episode_completed" if success else "episode_failed",
                episode=record["episode_id"],
                object_rise=rise,
                trajectory_samples=len(trajectory),
            )
            index_record = {
                "episode_id": record["episode_id"], "success": success,
                "failure_phase": record["failure_phase"], "object_rise": rise,
                "target": record["target"]["type"],
                "metadata": str(episode_dir.relative_to(self.session_dir) / "metadata.json"),
                "trajectory": str(episode_dir.relative_to(self.session_dir) / "trajectory.jsonl"),
                "environment_index": env_index, "replicator_frames": record["replicator_frames"],
            }
            for index_name in (["episodes.jsonl", "successes.jsonl"] if success else ["episodes.jsonl"]):
                with (self.session_dir / index_name).open("a") as stream:
                    stream.write(json.dumps(index_record) + "\n")
            self.all_records.append({"success": success})

    def run_batch(self, episode_ids: list[int]) -> None:
        # Removing a successful grasp's fixed joint invalidates PhysX views.
        # Rebuild handles before touching articulation/object tensors again.
        self.world.reset()
        self.configure_batch(episode_ids)
        count = len(episode_ids)
        orientations = [record["waypoints"]["orientation_wxyz"] for record in self.episode_records]
        pregrasp = [record["waypoints"]["pregrasp"] for record in self.episode_records]
        grasp = [record["waypoints"]["grasp"] for record in self.episode_records]
        lift = [record["waypoints"]["lift"] for record in self.episode_records]
        # Clear the tabletop before rotating/sweeping the long imported fingers.
        current_positions, current_orientations = self.get_link_world_poses(self.ee_link_index)
        clearance = current_positions[:count].clone()
        clearance[:, 2] = torch.maximum(clearance[:, 2],
            torch.as_tensor([p[2] + 0.10 for p in pregrasp], device=self.device))
        phase_results = {}
        phase_results["clearance"] = self.move_end_effectors(
            clearance, current_orientations[:count], "clearance"
        )
        overhead = [[p[0], p[1], p[2] + 0.10] for p in pregrasp]
        phase_results["overhead"] = self.move_end_effectors(
            overhead, orientations, "overhead", phase_results["clearance"]
        )
        phase_results["pregrasp"] = self.move_end_effectors(
            pregrasp, orientations, "pregrasp", phase_results["overhead"]
        )
        phase_results["approach"] = self.move_end_effectors(
            grasp, orientations, "approach", phase_results["pregrasp"]
        )
        openings = [
            min(0.044, float(record["target"]["collision_proxy"].get("radius", 0.03)) + 0.005)
            for record in self.episode_records
        ]
        self.command_gripper(openings, "close_gripper", phase_results["approach"])
        phase_results["enclosure"] = self.attach_targets(phase_results["approach"])
        phase_results["lift"] = self.move_end_effectors(lift, orientations, "lift", phase_results["enclosure"])
        for _ in range(int(self.config["settle_steps"])):
            self.world.step(render=False)
        self.capture("hold_after_lift")
        self.finish_batch(phase_results)
        for env_index in range(count):
            path = f"{self.env_paths[env_index]}/grasp_attachment"
            if self.stage.GetPrimAtPath(path).IsValid():
                self.stage.RemovePrim(path)

    def write_summary(self, final: bool) -> None:
        summary = {
            "dataset_version": int(self.config["dataset_version"]),
            "session_directory": str(self.session_dir),
            "worker_id": args.worker_id,
            "worker_count": args.worker_count,
            "requested_global_episode_count": args.episodes,
            "assigned_episode_count": len(self.assigned_ids),
            "episode_count": len(self.all_records),
            "completed": sum(bool(record["success"]) for record in self.all_records),
            "failed": sum(not bool(record["success"]) for record in self.all_records),
            "final": final,
            "profile": args.profile,
            "elapsed_seconds": time.monotonic() - self.started_at,
            "stop_reason": getattr(self, "stop_reason", None),
            "num_parallel_environments": args.num_envs,
            "renderer": args.renderer,
            "camera_resolution": [
                int(self.config["camera_width"]),
                int(self.config["camera_height"]),
            ],
            "modalities": [
                "two-view RGB",
                "depth",
                "semantic segmentation",
                "instance segmentation",
                "2D and 3D bounding boxes",
                "camera calibration",
                "joint state",
                "end-effector path",
                "object pose",
            ],
        }
        summary_path = self.session_dir / "summary.json"
        temporary_path = summary_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )

        temporary_path.replace(summary_path)

    def run(self) -> None:
        self.all_records: list[dict[str, Any]] = []
        self.assigned_ids = list(range(args.worker_id + 1, args.episodes + 1, args.worker_count))
        self.create_world()
        for start in range(0, len(self.assigned_ids), args.num_envs):
            if self.stop_requested:
                self.stop_reason = "signal"
                break
            if args.duration_hours and time.monotonic() - self.started_at >= args.duration_hours * 3600:
                self.stop_reason = "duration_limit"
                break
            if shutil.disk_usage(self.session_dir).free < args.min_free_gb * 1024**3:
                self.stop_reason = "disk_reserve"
                break
            self.run_batch(self.assigned_ids[start : start + args.num_envs])
            self.write_summary(final=False)
        self.write_summary(final=True)
        self.log(
            "session_completed",
            completed=sum(bool(record["success"]) for record in self.all_records),
            failed=sum(not bool(record["success"]) for record in self.all_records),
        )

    def close(self) -> None:
        for writer in self.render_writers:
            writer.detach()
        rep.orchestrator.wait_until_complete()
        if hasattr(self, "world"):
            self.world.stop()


def main() -> int:
    generator = IsaacDatasetGenerator()
    exit_code = 0
    try:
        generator.run()
    except Exception as error:
        generator.log(
            "session_failed",
            error=repr(error),
            traceback=traceback.format_exc(),
        )
        generator.write_summary(final=False)
        exit_code = 1
    finally:
        try:
            generator.close()
        finally:
            app.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
