# ACT PyBullet Smoke Runbook

Run all commands from the repository root:

```bash
cd /home/federico/catkin_ws/src/mobile_manipulation
```

## IL environment setup

Keep LeRobot and its ML dependencies outside the catkin Python environment.
Use the `mm-il` conda env (Python ≥3.10, PyTorch **cu118** so Maxwell GPUs
like Quadro M2000M work):

```bash
conda activate mm-il
# If recreating: install torch from the cu118 index first, then lerobot.
#   pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 \
#     --index-url https://download.pytorch.org/whl/cu118
#   pip install -r mm_run/requirements-il.txt
python -c "import lerobot, torch; print(lerobot.__version__, torch.__version__, torch.cuda.is_available())"
```

`mm_run/requirements-il.txt` pins `lerobot>=0.4.0,<0.5` (needed for
`LeRobotDataset.finalize` and checkpoint pre/post-processors). Train with
`--policy.device=cuda` (see `train_act_smoke.sh`). State-only defaults use a
smaller transformer than LeRobot's vision ACT
(`dim_model=256`, `dim_feedforward=1024`, `n_encoder_layers=2`).

**Frozen defaults (baked 2026-11-12):** `CHUNK_SIZE=25`, `N_ACTION_STEPS=12`,
`STEPS=40000`, export `--action desired_ee_vel` / `--obs q` /
`--env-state obs_duplicate` / `--default-fps 10`, launch/eval
`--boundary-blend 0.5 --boundary-blend-mode decay`, `ctrl_rate=10`.
Joint-space (`u_cmd` / `teleop:=act`) still works if you override `--action u_cmd`.

To continue an existing run to more steps (weights + optimizer):
`RESUME_CONFIG=.../checkpoints/020000/pretrained_model/train_config.json STEPS=40000 ./train_act_smoke.sh DATASET_ROOT OUTPUT_DIR`.

Deploy architecture: `act_ee_mpsf_ros.py` (`teleop:=act_ee`, EE recipe) or
`act_teleop_ros.py` (`teleop:=act`, joint). Both stay on ROS Noetic (system
Python 3.8) and spawn `python -m mm_utils.act_policy_worker` under
`MM_IL_PYTHON` (default `~/miniconda3/envs/mm-il/bin/python`).

## Operator checklist

### Step 1: Collect or locate demos

- [ ] Prefer the **Tomorrow** plan below (`demo_mpc10`, ≥20 EE MPSF trials @ 10 Hz).
- [ ] Or locate existing `teleop:=mpsf` trials with `control/data.npz` logs.
- [ ] Synthetic data is for pipeline debug only — not learning evidence.

### Step 2: Export and train — operator must run

- [ ] Activate the IL environment described above.
- [ ] Export (defaults are already EE / fps=10):

```bash
python mm_run/src/mm_run/scripts/export_teleop_to_lerobot.py \
  --root mm_run/results/mpsf_experiments/teleop_table_wipe/demo_mpc10 \
  --out mm_run/results/il/datasets/table_wipe_act_ee_mpc10 \
  --repo-id local/table_wipe_act_ee_mpc10
```

- [ ] Train ACT (defaults: chunk 25 / na 12 / 40k steps):

```bash
conda activate mm-il
REPO_ID=local/table_wipe_act_ee_mpc10 JOB_NAME=act_table_wipe_ee_mpc10 \
  ./mm_run/src/mm_run/scripts/train_act_smoke.sh \
  mm_run/results/il/datasets/table_wipe_act_ee_mpc10 \
  mm_run/results/il/outputs/act_table_wipe_ee_mpc10
```

The train script tees console output to `OUTPUT_DIR/train_console.log` and
writes `metrics.csv` / `metrics.jsonl` / `metrics_summary.txt` (loss vs step).
Re-parse a pasted log anytime with
`python mm_run/src/mm_run/scripts/parse_lerobot_train_log.py LOG --out-dir DIR`.

- [ ] Confirm that training produced a loadable checkpoint, typically:
      `.../act_table_wipe_smoke_tight/checkpoints/last/pretrained_model`.
      Keep the original `act_table_wipe_smoke` run for A/B comparison.
- [ ] Skim `metrics_summary.txt` / `metrics.csv` for plateau vs still-falling loss.
- [ ] Save the exact LeRobot version, command line, dataset path, and checkpoint
      path with the run notes.

### Step 3: Launch PyBullet with ACT — operator must run

- [ ] Use a **local** ROS master (lab robot URI will hang). Before launch:

```bash
conda deactivate   # keep roslaunch on system Python
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
# Ensure ROS_MASTER_URI=http://localhost:11311 and ROS_IP/ROS_HOSTNAME
# are unset or local (bashrc may already do this for sim).
```

- [ ] Launch the simulator with ACT:

```bash
roslaunch mm_run run_pybullet_sim.launch \
  config:="$(rospack find mm_run)/config/mpsf_experiments/teleop_table_wipe.yaml" \
  teleop:=act \
  act_checkpoint:="$(pwd)/mm_run/results/il/outputs/act_table_wipe_smoke/checkpoints/last/pretrained_model" \
  gui:=true
```

- [ ] Confirm the ACT controller starts and loads the requested checkpoint.
- [ ] Confirm the robot receives non-zero plans while the policy runs.
- [ ] Confirm no MPSF node is running.

### Step 4: Evaluate the smoke bar — operator must run

- [ ] Record coverage, collisions, and elapsed time using the existing metrics.
- [ ] Inspect the motion in PyBullet for recognizable table wiping.
- [ ] Treat recognizable wiping as the initial success bar.
- [ ] After that first success, iterate toward high coverage, no collisions, and
      completion in approximately 30 seconds or less.
- [ ] Preserve logs, metrics, configuration, checkpoint, and observations; do
      not invent or infer results for runs that were not performed.

### Step 5: Note follow-ons

- [ ] Case 1 smoke path should be repeatable before iterating paper splits.
- [ ] Record failures and the next concrete pipeline or policy adjustment before
      collecting more data or retraining.

## Case 2: ACT EE → MPSF (`desired_ee_vel`)

Obs stays `q`; action is logged `desired_ee_vel` (6D, same MPC EE-vel frame).
Deploy uses `teleop:=act_ee` (thin MPSF subclass). EE-only; left-stick
teleop refs are ignored after Square.

### Export + train

```bash
conda activate mm-il
python mm_run/src/mm_run/scripts/export_teleop_to_lerobot.py \
  --root mm_run/results/mpsf_experiments/teleop_table_wipe/demo_mpc \
  --out mm_run/results/il/datasets/table_wipe_act_ee_mpc \
  --repo-id local/table_wipe_act_ee_mpc \
  --obs q \
  --action desired_ee_vel

REPO_ID=local/table_wipe_act_ee_mpc JOB_NAME=act_table_wipe_ee_mpc \
  ./mm_run/src/mm_run/scripts/train_act_smoke.sh \
  mm_run/results/il/datasets/table_wipe_act_ee_mpc \
  mm_run/results/il/outputs/act_table_wipe_ee_mpc
```

### Deploy (sim)

PyBullet sim defaults to **lockstep** (`sim_lockstep:=true`): physics advances only
after each control cycle, so GUI / slow ACT inference stretch wall time without
skipping sim ticks. Disable with `sim_lockstep:=false` if needed.

```bash
conda deactivate
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ACADOS_SOURCE_DIR=/home/federico/safe_control/acados
export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
# catkin build mm_run  # once, after adding act_ee_mpsf_ros.py
roslaunch mm_run run_pybullet_sim.launch \
  config:="$(rospack find mm_run)/config/mpsf_experiments/teleop_table_wipe.yaml" \
  teleop:=act_ee \
  act_checkpoint:="$(pwd)/mm_run/results/il/outputs/act_table_wipe_ee_mpc/checkpoints/last/pretrained_model" \
  gui:=true
```

Worker `--nu` must match the trained action width (6 for `desired_ee_vel`).
`low_level_cmd_node` keeps MPC kp for `act_ee` (unlike case-1 `act`).

## Unattended eval

Runs N headless PyBullet trials (default 5) against `checkpoints/last`, infers
`teleop` from action width (9→`act`, 3/6→`act_ee`), auto-starts, stops at
`simulation.duration` (20s for table wipe), and writes:

`OUTPUT_DIR/evals/<stamp>_last/trial_XX/` plus `summary.json`.

```bash
conda deactivate
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
# local ROS master only
python3 mm_run/src/mm_run/scripts/eval_act_smoke.py \
  mm_run/results/il/outputs/act_table_wipe_ee_mpc
# optional: --trials 5 --gui --checkpoint last
# (use python3 — plain ``python`` is often 2.7 on this box)
```

`act_ee` reads `--nu` and `--n-action-steps` from the checkpoint when left at `0`
(default), so horizon-sweep checkpoints execute with the trained open-loop length.

## Frozen deploy recipe (table wipe ACT-EE) — 2026-11-12

Decided after MPSF timing + rate×na sweep on `cs25_na25` (lockstep eval).

| Knob | Value | Why |
|---|---|---|
| `controller.ctrl_rate` | **10 Hz** | Hard-constraint MPSF work ≈ 80–90 ms; cannot honestly hold 20 Hz on hardware. Set in `teleop_table_wipe.yaml`. |
| Collisions | **hard** (softened false) | Softening caused odd behavior; keep safety. |
| Deploy `n_action_steps` | **12** | ≈ 1.2 s open-loop @ 10 Hz; best 10 Hz cell in rate×na sweep (~77% hard-switch). |
| Deploy boundary blend | **`decay` @ 0.5** | Baked into launch/eval defaults. |
| Demo / dataset fps | **10** | Match deploy clock when collecting new data. |
| Train `chunk_size` / `n_action_steps` | **25 / 12** | Baked into `train_act_smoke.sh`. |
| Action / obs | `desired_ee_vel` (6) / `q` + `env_state=obs_duplicate` | Winning stack. |

**Eval (defaults already match the recipe):**
```bash
python3 mm_run/src/mm_run/scripts/eval_act_smoke.py \
  mm_run/results/il/sweeps/horizon_ee_mpc3/runs/cs25_na25/train
```

**Do not** treat 20 Hz lockstep scores as hardware truth — lockstep advances sim by `1/ctrl_rate` even when wall work overruns.

## Tomorrow: exact data-collection plan (2026-11-13)

Goal: matched **10 Hz** demos for the paper IL section (MPC-assisted EE primary;
matched direct set for the source comparison).

### A. MPC-assisted EE demos (primary — train the deploy policy)

| Item | Spec |
|---|---|
| Config | `teleop_table_wipe.yaml` (`ctrl_rate: 10`, hard collisions) |
| Teleop | **`teleop:=mpsf`**, stay in **EE mode** the whole trial |
| Platform | Hardware preferred; same YAML in sim only if hardware blocked |
| Logging root | `mm_run/results/mpsf_experiments/teleop_table_wipe/demo_mpc10/` (new folder; do not mix with `demo_mpc3`) |
| Count | **≥ 20** successful full wipes (target **25–30**); prior smoke used 13 |
| Duration | Finish **both** tables; expect ~20–30 s enabled time (tight needed ~30 s) |
| Quality gate | Path coverage high on both tables; no protective stop; no stuck base; discard bad trials |
| Variation | Vary start base pose slightly; wipe both tables in consistent order; natural stick pace |
| Do **not** | Soften collisions; run at 20 Hz; switch to base mode mid-wipe; mix operators |

Launch sketch:
```bash
roslaunch mobile_manipulation_central thing.launch   # hardware
roslaunch mm_run hardware_teleop.launch teleop:=mpsf \
  config:=$(rospack find mm_run)/config/mpsf_experiments/teleop_table_wipe.yaml \
  logging_sub_folder:=demo_mpc10
```

### B. Matched direct demos (source comparison only)

| Item | Spec |
|---|---|
| Same task / YAML / 10 Hz / folder sibling | `.../demo_direct10/` |
| Teleop | **`teleop:=direct`** |
| Count | **≥ 20** successful trials (same bar as A) |
| Purpose | Train a matched direct-source policy for the paper A/B — **not** the deploy recipe |

### C. After collection (same day or overnight)

```bash
conda activate mm-il
# Export MPC EE (defaults are already desired_ee_vel / fps=10)
python mm_run/src/mm_run/scripts/export_teleop_to_lerobot.py \
  --root mm_run/results/mpsf_experiments/teleop_table_wipe/demo_mpc10 \
  --out mm_run/results/il/datasets/table_wipe_act_ee_mpc10 \
  --repo-id local/table_wipe_act_ee_mpc10
# Confirm meta fps == 10 and episodes >= 20

REPO_ID=local/table_wipe_act_ee_mpc10 JOB_NAME=act_table_wipe_ee_mpc10 \
  ./mm_run/src/mm_run/scripts/train_act_smoke.sh \
  mm_run/results/il/datasets/table_wipe_act_ee_mpc10 \
  mm_run/results/il/outputs/act_table_wipe_ee_mpc10

# Eval (defaults = recipe)
source /opt/ros/noetic/setup.bash && source ~/catkin_ws/devel/setup.bash
export ACADOS_SOURCE_DIR=$HOME/safe_control/acados
export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
python3 mm_run/src/mm_run/scripts/eval_act_smoke.py \
  mm_run/results/il/outputs/act_table_wipe_ee_mpc10
```

Repeat export/train/eval for `demo_direct10` with `--action u_cmd` and `teleop:=act` only if reporting the joint/direct IL arm in the paper.

### D. What you do **not** need tomorrow

- Temporal ensemble / async inference work
- More blend or horizon sweeps
- Soft-collision experiments

### Boundary blend note (keep) — 2026-11-12

Deploy-only: on each `n_action_steps` refresh, mix the new chunk with the previous chunk’s leftover (`--boundary-blend` in `[0,1]`, 0=hard switch). Modes: `constant` | `decay` | `head`. Still **one ACT infer per refresh** — 10 Hz / MPSF-friendly (no every-tick ensemble).

Matrix @ 10 Hz / na=12 (`cs25_na25`, n=10). Same-session hard-switch re-run ≈ 76.6% (matches prior 77.1%).

| Setting | Mean | Min–Max | Note |
|---|---|---|---|
| **`decay 0.5`** | **94.4%** | 92–96% | **Preferred** — tight, high |
| `constant 0.2` | 95.6% | 87–100% | Similar mean, wider spread |
| `head 0.5 / h=3` | 84.2% | 77–92% | Partial help |
| hard switch `0` | ~77% | ~69–86% | Baseline |
| `constant 0.3` / `0.5` | 72% / 70% | worse | Too much old chunk |

**Decision:** `decay @ 0.5` is the frozen open-loop deploy blend (baked into defaults).

### Temporal ensemble note (lockstep) — 2026-11-12

Paper / LeRobot `ACTTemporalEnsembler` ported to numpy (`ActTemporalEnsembler`); deploy via `--temporal-ensemble-coeff` (forces `na=1`, no boundary blend).

Lockstep @ 10 Hz, `cs25_na25`, n=10:

| Setting | Mean | Min–Max |
|---|---|---|
| **`na=12` decay @ 0.5** | **94.4%** | 92–96% |
| `na=1` TE `m=-0.01` (favor new) | 50.5% | 44–72% |
| `na=1` TE `m=0.01` (paper) | 48.4% | 44–57% |
| `na=1` TE `m=0.0` (uniform) | 48.3% | 44–68% |
| `na=1` TE `m=0.05` | 45.3% | 45–46% |
| `na=1` hard (no TE) | 4.5% | 4.4–4.6% |

**Decision:** TE helps vs bare `na=1`, but does **not** beat open-loop `na=12` + decay under lockstep. **Do not chase async TE** for this stack; keep decay deploy. Leaderboard: `mm_run/results/il/sweeps/horizon_ee_mpc3/temporal_ensemble_eval/`.

## Overnight horizon sweep (`chunk_size` × `n_action_steps`)

Trains + evals the fixed ee_mpc3 stack (`desired_ee_vel`, `env_state=obs_duplicate`)
over the grid in `sweep_act_horizon.py` (baseline `50/25` included). Default
`n=10` trials, `STEPS=40000`. Writes
`mm_run/results/il/sweeps/horizon_ee_mpc3/leaderboard.{json,csv}`.

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export ACADOS_SOURCE_DIR=$HOME/safe_control/acados
export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
export MM_IL_PYTHON=$HOME/miniconda3/envs/mm-il/bin/python

cd ~/catkin_ws/src/mobile_manipulation
mkdir -p mm_run/results/il/sweeps   # needed before nohup redirect
# dry-run first:
./mm_run/src/mm_run/scripts/sweep_act_horizon.sh --dry-run
# overnight:
nohup ./mm_run/src/mm_run/scripts/sweep_act_horizon.sh \
  > mm_run/results/il/sweeps/horizon_ee_mpc3_nohup.log 2>&1 &
```


Or call `python3 mm_run/src/mm_run/scripts/sweep_act_horizon.py ...` directly.
Train uses `MM_IL_PYTHON`; eval uses `/usr/bin/python3` with the sourced ROS env.
