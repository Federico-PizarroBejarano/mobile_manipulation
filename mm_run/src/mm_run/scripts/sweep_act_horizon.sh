#!/usr/bin/env bash
# Overnight ACT horizon sweep wrapper (chunk_size × n_action_steps).
#
# Prerequisites (eval):
#   source /opt/ros/noetic/setup.bash && source ~/catkin_ws/devel/setup.bash
#   export ACADOS_SOURCE_DIR=$HOME/safe_control/acados
#   export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
# Train uses MM_IL_PYTHON (default ~/miniconda3/envs/mm-il/bin/python).
#
# Usage (from repo root):
#   ./mm_run/src/mm_run/scripts/sweep_act_horizon.sh
#   ./mm_run/src/mm_run/scripts/sweep_act_horizon.sh --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
# Do NOT prepend mm_run/src: it shadows catkin-generated mm_run.msg and breaks eval.
# Keep devel/ROS PYTHONPATH first; only add mm_utils for the orchestrator imports.
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${REPO_ROOT}/mm_utils/src"
export MM_IL_PYTHON="${MM_IL_PYTHON:-$HOME/miniconda3/envs/mm-il/bin/python}"

cd "${REPO_ROOT}"
SWEEP_ROOT="${REPO_ROOT}/mm_run/results/il/sweeps/horizon_ee_mpc3"
mkdir -p "${SWEEP_ROOT}"
exec /usr/bin/python3 "${SCRIPT_DIR}/sweep_act_horizon.py" \
  --dataset "${REPO_ROOT}/mm_run/results/il/datasets/table_wipe_act_ee_mpc3" \
  --repo-id local/table_wipe_act_ee_mpc3 \
  --sweep-root "${SWEEP_ROOT}" \
  --trials 10 \
  --steps 40000 \
  --skip-existing \
  "$@"
