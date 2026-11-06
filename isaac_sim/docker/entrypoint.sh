#!/usr/bin/env bash
set -euo pipefail

if [[ "${ACCEPT_EULA:-N}" != "Y" ]]; then
  echo "Set ACCEPT_EULA=Y after reviewing NVIDIA's Isaac Sim license." >&2
  exit 2
fi

mkdir -p \
  /workspace/isaac_sim/assets/generated/tiago \
  "${ISAAC_OUTPUT_ROOT:-/workspace/ready_datasets/isaac_clutter_pick}" \
  /workspace/.isaac_cache/{main,computecache,kit,logs,config,data}

tiago_usd=/workspace/isaac_sim/assets/generated/tiago_direct/tiago.usd
tiago_urdf=/workspace/isaac_sim/assets/tiago_source/tiago.urdf
if [[ ! -f "${tiago_urdf}" ]]; then
  echo "TIAGo source asset is missing; run ./scripts/isaac.sh export-tiago first." >&2
  exit 2
fi
if [[ ! -f "${tiago_usd}" ]]; then
  /isaac-sim/python.sh /workspace/isaac_sim/import_tiago.py \
    --urdf "${tiago_urdf}" \
    --output "${tiago_usd}"
fi

generator_args=(
  /workspace/isaac_sim/dataset_generator.py
  --config /workspace/isaac_sim/config/generator.json
  --objects /workspace/isaac_sim/config/object_assets.json
  --tiago-usd "${tiago_usd}"
  --output "${ISAAC_OUTPUT_ROOT:-/workspace/ready_datasets/isaac_clutter_pick}"
  --episodes "${EPISODES:-300}"
  --num-envs "${NUM_ENVS:-4}"
  --seed "${DATASET_SEED:-0}"
  --worker-id "${WORKER_ID:-0}"
  --worker-count "${WORKER_COUNT:-1}"
  --renderer "${ISAAC_RENDERER:-RaytracedLighting}"
)
if [[ "${ISAAC_MULTI_GPU_RENDERING:-false}" == "true" ]]; then
  generator_args+=(--multi-gpu-rendering)
fi

exec /isaac-sim/python.sh "${generator_args[@]}"
