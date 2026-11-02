# TIAGo tabletop simulation

This project builds a pinned ROS 2 Jazzy + Gazebo Harmonic environment with a single-arm TIAGo, the PAL two-finger gripper, MoveIt 2, a 1280x960 RGB-D head camera, and a randomized tabletop scene.

The simulation and grasp policy are separate processes. Starting the stack launches Gazebo, TIAGo, and the scene manager only. The scene manager owns the table and object lifecycle and publishes object geometry and pose; it does not choose an end-effector target, gripper aperture, or motion. A heuristic, learned policy, teleoperation node, or another controller can consume the same interface without changing the simulator.

## Build and start the simulation

```bash
git clone https://github.com/mateuszwatly/tiago-grasp-sim.git
cd tiago-grasp-sim
cp .env.example .env
mkdir -p datasets recordings
./scripts/start.sh
```

Gazebo opens on the host display. Set `TIAGO_HEADLESS=true` only when a GUI is not needed.

By default, the scene contains the table and exactly one randomized apple, cup, or can. Bowls are excluded because their low profile requires a separate top-grasp strategy. The robot does not move because `TIAGO_RUN_HEURISTIC=false`.

## Policy-independent scene interface

The scene manager publishes the current object as JSON on a transient-local topic:

```bash
./scripts/compose.sh exec tiago ros2 topic echo \
  /tiago_grasp_sim/object std_msgs/msg/String --once \
  --qos-durability transient_local --qos-reliability reliable
```

The JSON contains the scene seed, object type, RGB color, world pose, table height, and physical geometry such as `half_width` and `center_height`. It contains no grasp target or policy output.

Replace the current object with another randomized object:

```bash
./scripts/compose.sh exec tiago ros2 service call \
  /tiago_grasp_sim/reset_scene std_srvs/srv/Trigger '{}'
```

Remove the current object while leaving the table and robot running:

```bash
./scripts/compose.sh exec tiago ros2 service call \
  /tiago_grasp_sim/clear_scene std_srvs/srv/Trigger '{}'
```

Read the object's current Gazebo pose, including any displacement caused by a grasp:

```bash
./scripts/compose.sh exec tiago ros2 service call \
  /tiago_grasp_sim/current_object std_srvs/srv/Trigger '{}'
```

These endpoints are the boundary for a replacement policy. A new policy can subscribe to `/tiago_grasp_sim/object`, observe the camera and robot topics, and command the existing ROS controllers.

Recording is also independent of the policy. Start or stop it around any controller:

```bash
./scripts/compose.sh exec tiago record-dataset my_session
```

This records RGB-D, the complete organized point cloud, TF, joint state, controller, and scene-metadata topics to `recordings/my_session/`.

Capture a calibration motion independently of the grasp policy:

```bash
./scripts/compose.sh exec -d tiago run-calibration-capture
```

The command moves the arm through conservative tabletop poses, varies the two-finger opening, and saves the complete 1280x960 point cloud and robot state to a timestamped `recordings/calibration_*` MCAP bag. Its `calibration_manifest.json` records every commanded phase and result.

## Optional reference heuristic

The optional policy adapts the PCA, cutting-plane, and antipodal contact ranking described in [Fast geometry-based computation of grasping points on three-dimensional point clouds](https://doi.org/10.1177/1729881419831846). It samples the simulated object's surface, chooses a two-finger contact pair, and uses MoveIt's KDL inverse-kinematics solver for every arm motion. It is disabled by default. Run it explicitly against the already-running simulation:

```bash
./scripts/compose.sh exec tiago run-grasp-heuristic
```

Or set `TIAGO_RUN_HEURISTIC=true` before startup. `HEURISTIC_EPISODES` defaults to one. Each episode starts from a randomized viable arm pose, moves to a 10 cm pregrasp, approaches the ranked contacts, fully closes both fingers, and lifts 5 cm. It records the planned segments and the measured end-effector path under `datasets/`.

The contact pose accounts for the PAL gripper's URDF geometry: its finger-pad center is 45 mm forward of `gripper_grasping_frame`, and the policy keeps the full pad collision box above the table. The position controller closes only to the sampled object surfaces, including the 5 mm pad half-thickness. Before simulated grasp stabilization, the scene manager independently reads the actual Gazebo poses of the object and both finger links and requires the object to be enclosed between the pads. A miss is rejected without attaching or lifting the object. The episode records the measurements in `grasp_stabilization.pre_attachment_enclosure`.

An attempt is labeled `success` only when the end effector reaches its lift target and Gazebo reports that the object rose by at least half the commanded lift (with a 2 cm minimum). A miss remains in the dataset as `failed`; controller completion alone cannot produce a successful label.

## Configuration

| Variable | Default | Owner |
| --- | --- | --- |
| `TIAGO_WORLD` | `empty` | Gazebo base world. |
| `TIAGO_HEADLESS` | `false` | Display mode. |
| `SCENE_SEED` | `0` | Scene manager; zero selects and publishes a random seed. |
| `SCENE_SPAWN_OBJECT` | `true` | Scene manager; spawn one object at startup. |
| `SCENE_OBJECT_TYPE` | empty | Fix every scene to `apple`, `cup`, or `can`; empty selects among them. |
| `TIAGO_RUN_HEURISTIC` | `false` | Start the optional reference policy automatically. |
| `HEURISTIC_SEED` | `0` | Reference policy random-start seed. |
| `HEURISTIC_EPISODES` | `1` | Reference policy recording count. |
| `HEURISTIC_FINGER_PAD_Z_CORRECTION` | `0.0` | Optional vertical correction from the object's settled center to the gripper pad center, in metres. |
| `HEURISTIC_LIFT_DISTANCE` | `0.05` | Post-close lift distance in metres. |
| `HEURISTIC_SPEED_SCALE` | `2.5` | Reference policy trajectory-duration scale. |
| `ROS_DOMAIN_ID` | `42` | ROS graph isolation. |

Stop the stack with Ctrl+C in the startup terminal or run `./scripts/compose.sh down`.
