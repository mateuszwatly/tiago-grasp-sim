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
