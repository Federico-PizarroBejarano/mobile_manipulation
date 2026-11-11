#!/usr/bin/env bash
# Train state-only ACT on an exported teleop dataset (conda env mm-il).
#
# Sized for joint-state (no images): smaller transformer than LeRobot's
# vision-oriented defaults so full trains finish faster on a 4GB GPU.
# chunk_size / n_action_steps must match act_teleop --n-action-steps (25).
#
# Always tees console output to OUTPUT_DIR/train_console.log and parses a
# local metrics.csv / metrics.jsonl (no wandb required).
#
# Fresh train:
#   ./train_act_smoke.sh DATASET_ROOT OUTPUT_DIR
#
# Resume an existing run (weights + optimizer + step) to a higher STEPS:
#   RESUME_CONFIG=.../checkpoints/020000/pretrained_model/train_config.json \
#     STEPS=40000 ./train_act_smoke.sh DATASET_ROOT OUTPUT_DIR
# DATASET_ROOT is ignored on resume (config carries dataset + output_dir);
# OUTPUT_DIR must match the resumed run's output_dir.
set -euo pipefail

DATASET_ROOT="${1:?usage: train_act_smoke.sh DATASET_ROOT [OUTPUT_DIR]}"
OUTPUT_DIR="${2:-outputs/train/act_table_wipe_smoke_tight}"
REPO_ID="${REPO_ID:-local/table_wipe_act_smoke}"
RESUME_CONFIG="${RESUME_CONFIG:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Unbuffered python so tee sees loss lines promptly.
export PYTHONUNBUFFERED=1
set -o pipefail

run_train() {
  if [[ -n "${RESUME_CONFIG}" ]]; then
    if [[ ! -f "${RESUME_CONFIG}" ]]; then
      echo "RESUME_CONFIG not found: ${RESUME_CONFIG}" >&2
      exit 1
    fi
    if [[ ! -d "${OUTPUT_DIR}" ]]; then
      echo "resume requires existing OUTPUT_DIR: ${OUTPUT_DIR}" >&2
      exit 1
    fi
    # Append to the existing console log.
    LOG_PATH="${OUTPUT_DIR}/train_console.log"
    lerobot-train \
      --config_path="${RESUME_CONFIG}" \
      --resume=true \
      --steps="${STEPS:?STEPS is required when resuming (e.g. STEPS=40000)}" \
      --policy.device=cuda \
      --wandb.enable=false \
      2>&1 | tee -a "${LOG_PATH}"
  else
    # LeRobot refuses a pre-existing output_dir when resume=false, so do not mkdir it.
    # Tee to a sibling log; move into OUTPUT_DIR after train creates it.
    mkdir -p "$(dirname "${OUTPUT_DIR}")"
    LOG_PATH="${OUTPUT_DIR}.train_console.log"
    lerobot-train \
      --dataset.root="${DATASET_ROOT}" \
      --dataset.repo_id="${REPO_ID}" \
      --policy.type=act \
      --policy.push_to_hub=false \
      --policy.chunk_size="${CHUNK_SIZE:-50}" \
      --policy.n_action_steps="${N_ACTION_STEPS:-25}" \
      --policy.dim_model="${DIM_MODEL:-256}" \
      --policy.dim_feedforward="${DIM_FF:-1024}" \
      --policy.n_encoder_layers="${N_ENC_LAYERS:-2}" \
      --policy.n_decoder_layers="${N_DEC_LAYERS:-1}" \
      --policy.n_heads="${N_HEADS:-8}" \
      --output_dir="${OUTPUT_DIR}" \
      --job_name="${JOB_NAME:-act_table_wipe_smoke_tight}" \
      --policy.device=cuda \
      --batch_size="${BATCH_SIZE:-8}" \
      --steps="${STEPS:-20000}" \
      --eval_freq=0 \
      --save_freq="${SAVE_FREQ:-5000}" \
      --log_freq="${LOG_FREQ:-200}" \
      --wandb.enable=false \
      2>&1 | tee "${LOG_PATH}"
    mv -f "${LOG_PATH}" "${OUTPUT_DIR}/train_console.log"
    LOG_PATH="${OUTPUT_DIR}/train_console.log"
  fi
}

run_train
python3 "${SCRIPT_DIR}/parse_lerobot_train_log.py" "${LOG_PATH}" --out-dir "${OUTPUT_DIR}"
echo "console log: ${LOG_PATH}"
