#!/usr/bin/env bash
# Train state-only ACT on an exported teleop dataset (IL venv).
set -euo pipefail

DATASET_ROOT="${1:?usage: train_act_smoke.sh DATASET_ROOT [OUTPUT_DIR]}"
OUTPUT_DIR="${2:-outputs/train/act_table_wipe_smoke}"
REPO_ID="${REPO_ID:-local/table_wipe_act_smoke}"

# chunk_size / n_action_steps; n_action_steps must match the node's --n-action-steps (25).
lerobot-train \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.repo_id="${REPO_ID}" \
  --policy.type=act \
  --policy.push_to_hub=false \
  --policy.chunk_size="${CHUNK_SIZE:-50}" \
  --policy.n_action_steps="${N_ACTION_STEPS:-25}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=act_table_wipe_smoke \
  --policy.device=cuda \
  --batch_size=8 \
  --steps=20000 \
  --eval_freq=0 \
  --save_freq=5000 \
  --wandb.enable=false
