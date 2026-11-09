# ACT PyBullet Smoke Runbook

Run all commands from the repository root:

```bash
cd /home/federico/catkin_ws/src/mobile_manipulation
```

## IL environment setup

Keep LeRobot and its ML dependencies outside the catkin Python environment.

```bash
python3 -m venv .venv-il
source .venv-il/bin/activate
python -m pip install --upgrade pip
python -m pip install -r mm_run/requirements-il.txt
python -c "import lerobot, torch; print(lerobot.__version__, torch.__version__)"
```

`mm_run/requirements-il.txt` pins `lerobot>=0.4.0,<0.5` (needed for
`LeRobotDataset.finalize` and checkpoint pre/post-processors). The training
`N_ACTION_STEPS` (default 25) must match the node's `--n-action-steps` (default
25); `CHUNK_SIZE` defaults to 50. Before the first simulation or hardware
deployment, confirm the installed LeRobot version. Verify that
`mm_utils/src/mm_utils/act_policy_runtime.py`'s `_predict_chunk` behavior and the
training CLI flags match that exact installation (`lerobot-train --help` and the
installed `ACTPolicy` API). Do not assume compatibility across LeRobot releases.

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

- [ ] Source the catkin workspace and ensure the runtime can import the verified
      LeRobot installation.
- [ ] Launch the simulator with ACT:

```bash
roslaunch mm_run run_pybullet_sim.launch \
  config:=/path/to/table_wipe_config.yaml \
  teleop:=act \
  act_checkpoint:=/path/to/outputs/act_table_wipe_smoke/checkpoints/last/pretrained_model \
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
