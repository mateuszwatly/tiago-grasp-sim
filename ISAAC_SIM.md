# Isaac Sim dataset generator

This repository now has an Isaac Sim 6.2 data-generation path alongside the established Gazebo simulation. It imports the same TIAGo description, keeps the base fixed, generates randomized tabletop clutter, and records two-camera observations while a Jacobian inverse-kinematics controller approaches, closes around, and slightly lifts a selected object. The controller evaluates 16 horizontal grasp yaws, uses the one with the largest local distractor clearance, then follows an overhead pregrasp and vertical descent. It does not claim full-arm collision planning; that remains learnable from the retained observations and supervision or can later be replaced with cuRobo/RMPflow.

The Isaac generator is independent of the Gazebo/ROS heuristic. Its primary action is eight-dimensional: end-effector translation delta, rotation-vector delta, gripper command, and termination flag. Every step also preserves the full arm joint state. The scene can therefore train an end-effector policy without discarding the joint-level supervision needed for collision-aware control.

## Host prerequisites

Isaac Sim runs in NVIDIA's container and needs:

- An NVIDIA GPU supported by Isaac Sim 6.2 and the proprietary NVIDIA driver.
- `nvidia-smi` working on the host.
- Docker Engine and NVIDIA Container Toolkit configured for Docker.
- Enough disk space for the Isaac Sim image and caches.

On Debian 13, install the proprietary driver by following NVIDIA's [Debian driver guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/latest/debian.html), reboot, and verify `nvidia-smi`. Then follow the [NVIDIA Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and configure Docker:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

First verify GPU access from Docker:

```bash
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

Do not continue if this reports `could not select device driver ... gpu`; that means NVIDIA Container Toolkit is not yet configured for Docker.

The current host must expose every intended GPU in that output. `start-dual` assigns worker 0 to GPU 0 and worker 1 to GPU 1.

## Configure and start

Export the robot from the existing TIAGo image. This writes generated robot assets under the ignored `isaac_sim/assets/tiago_source/` directory:

```bash
./scripts/isaac.sh export-tiago
```

Create the local Isaac environment file and explicitly accept NVIDIA's license terms:

```bash
cp .env.isaac.example .env.isaac
```

Open `.env.isaac`, review NVIDIA's [Isaac Sim container terms and launch requirements](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/install_container.html), and change `ACCEPT_EULA=N` to `ACCEPT_EULA=Y` only if accepted. Then pull and run one worker:

```bash
./scripts/isaac.sh pull
./scripts/isaac.sh start
./scripts/isaac.sh logs
```

For two visible GPUs, start one process per GPU. The wrapper sets the worker count to two so the episode ranges do not overlap:

```bash
./scripts/isaac.sh start-dual
```

Each process also creates `NUM_ENVS` vectorized environments. Four environments per process is the conservative default for 1280x960 RGB-D and segmentation output. Increase it only after observing VRAM use. Episode identifiers are deterministically sharded across workers, so workers never write the same episode.

Generated data is saved under timestamped worker sessions in `ready_datasets/isaac_clutter_pick/`. Each episode contains:

- a metadata JSON record with object poses, target identity, seed, outcome, and lift measurement;
- a JSONL trajectory containing timestamps, phase, end-effector pose and delta, seven arm joint positions and velocities, gripper command, and termination flag;
- two Replicator camera streams containing RGB, metric depth, semantic and instance segmentation, 2D/3D boxes, and camera calibration. The organized 1280x960 point cloud can be reconstructed without loss from metric depth and intrinsics rather than duplicating it on disk.

The generator labels success only after the gripper has closed and the target has risen by the configured minimum. A fixed joint is used only after geometric finger enclosure is established, avoiding the false-positive attachment behavior previously found in Gazebo.

## Gaussian-splat object appearance

Isaac Sim 6.x can render native OpenUSD Gaussian-splat particle fields. Convert a 3DGS PLY or SPZ file inside the Isaac container:

```bash
docker compose --env-file .env.isaac -f compose.isaac.yaml run --rm \
  --entrypoint /isaac-sim/python.sh isaac-worker-0 \
  /workspace/isaac_sim/convert_splat.py \
  /workspace/assets/my_can.ply \
  /workspace/isaac_sim/assets/generated/my_can.usd
```

Set an object's `appearance_usd` in `isaac_sim/config/object_assets.json` to the generated USD path, then use `appearance_transform` to align its scale, orientation, and center with the collision proxy. The splat supplies high-quality appearance only. PhysX still uses the object's invisible analytic collision proxy, because rendered splats are not a reliable contact surface. The proxy dimensions and mass remain explicit in the same configuration file.

## Configuration

Primary generation settings are in `isaac_sim/config/generator.json`. They include episode count, seed, vector environment count, camera resolution, control timing, clutter count, lift height, and success threshold. Object proxy dimensions, masses, grasp widths, and optional appearance USDs are in `isaac_sim/config/object_assets.json`.

Stop all workers with:

```bash
./scripts/isaac.sh stop
```

The Gazebo workflow remains available and unchanged; Isaac output is written to a separate dataset tree.

## Isaac control conventions

The imported `arm_7_link` is the wrist frame, not Gazebo's grasp tool frame.
For the bundled merged TIAGo asset, the pad midpoint is 0.241575 m along wrist
+Z. The controller rotates that axis horizontally and compensates the waypoint
position accordingly. Recheck this transform when replacing the robot asset.

The generator explicitly authors PhysX force-drive stiffness and damping before
cloning the robot (angular: 200/10, linear: 200000/3000 in USD drive units), retaining
the imported force limits. URDF/Newton damping does not configure PhysX position
servos. Non-arm joints retain commanded positions during IK. Cartesian and joint
corrections are bounded; a vertical clearance move precedes overhead positioning
and descent. The upper-side grasp uses up to 35 mm above object center.

Failed motion phases disable subsequent grasp actions for that environment;
episode metadata includes `failure_phase`, and events contain measured and commanded
joint positions at phase completion. Success still requires geometric enclosure
and measured object rise; the subsequent lift uses a fixed attachment, so it is
not evidence of a friction-only grasp or calibrated real-robot actuator behavior.

## Longer runs and selecting successes

For a bounded fast recording run on one GPU:

```bash
ISAAC_PROFILE=fast NUM_ENVS=2 EPISODES=100000 DURATION_HOURS=4 \
  MIN_FREE_GB=20 DATASET_SEED=240645649 ./scripts/isaac.sh start
```

`fast` records both cameras at 640x480 every six control steps; `quality` retains
1280x960 every three steps. Physics timestep and control decimation are unchanged.
The simulation runs without real-time pacing. Throughput depends on GPU, scene,
failed-phase timeouts, and recording I/O; faster capture is not higher physics speed.

The duration limit is wall-clock time from generator initialization (after Isaac
application startup), checked between batches.
The episode count is an additional cap. Free disk space is checked before every
batch; reserve enough for a whole batch. Completed episodes are retained on stop.
SIGINT/SIGTERM request a stop between batches. `scripts/isaac.sh stop` allows up to
ten minutes for shutdown and camera writes. Restarting creates a new session;
it does not resume or overwrite the previous session.

Each session stores the effective configuration, an atomically updated summary,
`episodes.jsonl` for every finished attempt, and `successes.jsonl` for successful
attempts (created on the first success). Index entries point to metadata and
trajectories and identify camera environment and frame numbers. Camera files are
shared across the session: retain the referenced frames when extracting episodes.
Failed attempts remain available for analysis. These success labels include the
geometrically gated attachment described above, not friction-only grasp validation.

Each batch rebuilds physics handles after removing grasp attachments, resets joint
positions and velocities, and drains camera writes before publishing its episode
indexes. Environment clones have independent USD/physics state. Failed or waiting
environments retain their motor targets while other environments finish.

## Machine-learning exports and richer capture

See [the dataset contract](isaac_sim/DATASET.md) for export commands, the NumPy loader,
action semantics, time alignment, splits, privileged-state rules, and modality units.
Use `ISAAC_PROFILE=rich` to retain 1280x960 RGB-D plus normals, motion vectors and
radial depth. New sessions also retain control-rate commands, all-object state,
robot-link poses and joint reaction measurements. Existing sessions can be exported
without inventing commands or modalities they did not record.
