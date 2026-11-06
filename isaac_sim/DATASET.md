# Training dataset contract

Raw sessions are the source of truth. Do not delete them after exporting: exports
reference the original lossless image, depth, segmentation and calibration files.
The format is framework-neutral and supports image policies, point-cloud policies,
sequence models, dynamics models, and success/failure classification. It is not a
claim that every architecture can be trained without preprocessing.

## Export and load

Requires Python 3 and NumPy (already installed in the Isaac container):

```bash
python isaac_sim/export_dataset.py --source ready_datasets/isaac_clutter_pick \
  --output ready_datasets/ml_snapshot_001
```

Use a new output directory for each snapshot. Completed episodes are validated;
missing required camera files, invalid numeric data, inconsistent success labels,
and explicitly invalid sessions are rejected into `rejected.jsonl`. Keep both
successes and failures by default. `--successes-only` is available for imitation.
Repeated scene seeds are deduplicated, newest valid recording first. Use
`--keep-replays` to retain all completed variants while grouping their splits. Hashing the
scene seed assigns train/validation/test splits (80/10/10); all replays stay in the
same split. Small datasets may have empty splits. This does not enforce object- or
category-disjoint evaluation; construct those splits separately for generalization.

```python
from isaac_sim.export_dataset import EpisodeDataset
samples = EpisodeDataset('ready_datasets/ml_snapshot_001', split='train', successes_only=True)
episode = samples[0]
q = episode['arrays']['joint_position']  # [T, number_of_joints]
mask = episode['arrays']['transition_valid']
# Media paths are relative to episode['media_root']; load images lazily.
```

`episodes.jsonl` carries instruction, split, success, array/media paths, joint names,
command availability, and whether timestamps were inferred for older data. Each
episode contains compressed `arrays.npz` (no pickle), `frames.jsonl`, metadata,
raw trajectory, and controls when recorded. Paths are relative; preserve directory
relationships when relocating the dataset. There is no padding, normalization,
resizing, train-fitted statistics, or imposed sequence length. Batch/window sampling
and padding masks belong in the training pipeline.

## Timing, actions, and masks

Positions are meters, angles radians, time seconds. End-effector and object poses
use the simulator world frame; subtract the environment origin for local scenes.
`ee_quaternion_wxyz` uses scalar-first ordering; robot-link poses explicitly use
`xyz + xyzw`. Depth `distance_to_image_plane` is axial depth in meters; rich
`distance_to_camera` is radial distance. Calibration files retain native Replicator
matrices and camera conventions. Use their projection/extrinsics when reconstructing
point clouds; do not assume an OpenCV axis convention or treat radial depth as Z.

`achieved_motion[T,7]` is observed next-frame delta XYZ, world-frame rotation vector,
and next observed left-finger position. It is **not the commanded action**. Its final
row has no future observation: use `transition_valid` (false on the final row).
`imitation_mask` additionally requires episode success. `is_first` and `is_last`
mark episode boundaries, not environment MDP termination. No arbitrary rewards,
discounts, or terminal/truncation labels are invented.

New raw `controls.jsonl` records actual absolute joint-position commands, before/
after joint states, measured velocities, phase and interval timestamps at command
execution rate. Camera sampling and control sampling are separate time series.
Capture-time targets are snapshots, not next-step labels: use the control intervals
for command-supervised policies. Recorded renderer advances/settling can occur
between control intervals; do not assume uniform spacing or interpolate commands
across episode boundaries. Legacy data has no commands; `has_commands=false`.
Legacy observation timestamps are estimated from recorded physics steps and the
session timestep (1/120 s fallback), flagged by `timestamps_inferred`.

## Modalities and privileged state

All profiles record two cameras: RGB, axial depth, semantic and instance IDs,
2D/3D boxes, labels/mappings, calibration, joint positions/velocities, and target
poses. New schema 4 additionally records all active objects' poses and velocities,
robot-link poses, commanded joint targets, attachment flags, joint effort and
incoming-joint reaction wrench when available. Reaction wrench is expressed in the
local child-joint frame (force XYZ then torque XYZ), ordered by link incoming joint;
it is not a tactile pad/contact sensor. Object arrays follow metadata object order.
Raw state is retained even when not promoted to a numeric export array.

The `rich` profile adds normals, motion vectors, and radial depth at 1280x960 with
capture every three control steps. `fast` retains 640x480 every six steps. `rich-fast` uses that lower resolution
and sampling frequency while retaining all rich sensor channels. Physics
integration remains 120 Hz; commanded IK updates normally run every two physics
steps, gripper updates every step. Rendering can advance simulation too. Rich data
costs substantially more disk space and throughput; the disk reserve still applies.
Missing modalities in old recordings are not fabricated. Camera parameters plus
metric depth permit point-cloud reconstruction without storing redundant XYZ images.

Object states, labels, success, attachment status and future motion are privileged
supervision. Exclude them from policy observations unless intentionally training a
state-based/simulation policy. Keep failure examples for recovery/dynamics/value
learning; successful examples alone do not provide failure coverage.

Success means geometric enclosure followed by an attachment-assisted measured lift.
It does not demonstrate friction-only grasping, real-robot dynamics, calibrated
contacts, or full-arm collision avoidance. Keep this distinction in model evaluation.

While an Isaac worker is running, export using its installed NumPy environment:

```bash
docker exec tiago_grasp-isaac-worker-0-1 /isaac-sim/python.sh \
  /workspace/isaac_sim/export_dataset.py \
  --source /workspace/ready_datasets/isaac_clutter_pick \
  --output /workspace/ready_datasets/ml_snapshot_002 --keep-replays
```

For bulk collection retaining all rich channels:

```bash
ISAAC_PROFILE=rich-fast NUM_ENVS=2 EPISODES=100000 DURATION_HOURS=4 \
  MIN_FREE_GB=20 DATASET_SEED=0 ./scripts/isaac.sh start
```

Seed zero requests a fresh recorded random seed. A session configuration includes
object definitions, generator settings, and a generator-source SHA-256 for new runs.
Exports preserve that configuration per episode when available. Re-export into a
new snapshot to include newly completed episodes; snapshots do not change while a
training job reads them. All raw variants remain available even when deduplicating.
