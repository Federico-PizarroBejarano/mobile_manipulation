# ACT PyBullet Smoke Runbook

Run all commands from the repository root:

```bash
cd /home/federico/catkin_ws/src/mobile_manipulation
```

## IL environment setup

Keep LeRobot and its ML dependencies outside the catkin Python environment.
Use the `mm-il` conda env (Python ≥3.10), or an equivalent venv:

```bash
conda activate mm-il   # or: source .venv-il/bin/activate
python -m pip install -r mm_run/requirements-il.txt
python -c "import lerobot, torch; print(lerobot.__version__, torch.__version__)"
```

`mm_run/requirements-il.txt` pins `lerobot>=0.4.0,<0.5` (needed for
`LeRobotDataset.finalize` and checkpoint pre/post-processors). The training
`N_ACTION_STEPS` (default 25) must match the node's `--n-action-steps` (default
25); `CHUNK_SIZE` defaults to 50. Confirm the installed LeRobot version before
deploy (`lerobot-train --help` and the installed `ACTPolicy` API).

State-only export writes joint `q` to both `observation.state` and
`observation.environment_state` (LeRobot ACT 0.4 requires env_state for config
validation but still reads `observation.state` in the VAE forward path). Re-export
the dataset after changing feature keys.

Deploy architecture: `act_teleop_ros.py` stays on ROS Noetic (system Python
3.8). It spawns `python -m mm_utils.act_policy_worker` under
`MM_IL_PYTHON` (default `~/miniconda3/envs/mm-il/bin/python`) for checkpoint
load and action-chunk inference.

## Operator checklist

### Step 1: Collect or locate demos

- [ ] Locate existing table-wipe trials collected with `teleop:=mpsf`, using one
      operator and the usual `control/data.npz` logs.
- [ ] For a learning-quality smoke test, prefer at least 10 real trials.
- [ ] Synthetic data is acceptable only for debugging the export/train/runtime
      pipeline; do not treat it as evidence of learned wiping quality.
- [ ] Record the results root containing the selected trials:
      `RESULTS_ROOT=/path/to/results/<user>`.

### Step 2: Export and train — operator must run

- [ ] Activate the IL environment described above.
- [ ] Export the demonstrations with the repository script:

```bash
python mm_run/src/mm_run/scripts/export_teleop_to_lerobot.py \
  --root /path/to/results/<user> \
  --out /path/to/datasets/table_wipe_act_smoke \
  --repo-id local/table_wipe_act_smoke
```

- [ ] Train ACT with the repository script:

```bash
./mm_run/src/mm_run/scripts/train_act_smoke.sh \
  /path/to/datasets/table_wipe_act_smoke \
  /path/to/outputs/act_table_wipe_smoke
```

- [ ] Confirm that training produced a loadable checkpoint, typically:
      `/path/to/outputs/act_table_wipe_smoke/checkpoints/last/pretrained_model`.
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

- [ ] Do not begin case 2 or paper train/test splits until this complete smoke
      path is repeatable and reliable.
- [ ] Record failures and the next concrete pipeline or policy adjustment before
      collecting more data or retraining.
