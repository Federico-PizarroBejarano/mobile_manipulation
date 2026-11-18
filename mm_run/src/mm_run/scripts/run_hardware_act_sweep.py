#!/usr/bin/env python3
"""Hardware ACT sweep loop.

Square starts a trial and Circle ends it. This script then asks what to do
with the folder, backs the base up, and homes to ``table``.

After Circle:

  Enter   keep the trial
  c       keep the trial and mark a collision
  d       delete that folder and repeat the cell
  s       leave the shortfall; resume later fills it
  q       back up, home, and stop

Sweeps:

  decay    blends 0, 0.3, 0.5, 0.7 on the three full 25-demo models
           (direct u_cmd, MPC u_cmd, MPC EE), 3 trials each.
           Rank by smoothness at the chunk refresh, not path coverage.
  models   15 last-checkpoint models, 5 trials each, decay blend 0.5.
           --n-action-steps sets the execute horizon (12 for direct and
           MPC EE, 25 for MPC u_cmd). Complete cells are skipped.
  ensemble coefficients 0.01, 0.05, 0.1, -0.05. Two trials each on two
           checkpoints per policy (the horizon-12 miss, then the covered
           cell): 48 trials. Horizon 1 and blend 0. Startup must log those,
           not the launch defaults.

Requires thing.launch (robot + joystick) already running.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from mm_utils.hardware_sweep import (
    DECAY_TRIALS,
    ENSEMBLE_TRIALS,
    MODEL_TRIALS,
    TABLE_WIPE_LOG_DIR,
    SweepCell,
    after_trial_action,
    decay_blend_cells,
    ensemble_coeff_cells,
    model_cells,
    new_trial_dir,
    trial_dirs,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sweep", choices=("decay", "models", "ensemble"))
    p.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Override trials per cell (decay 3, models 5, ensemble 2)",
    )
    p.add_argument(
        "--boundary-blend",
        type=float,
        default=0.5,
        help="Decay blend for every models-sweep policy (chosen: 0.5)",
    )
    p.add_argument(
        "--n-action-steps",
        type=int,
        default=12,
        help="ACT execute horizon for every models-sweep policy "
        "(12 for direct and MPC EE, 25 for MPC u_cmd)",
    )
    p.add_argument("--backup-distance", type=float, default=0.2)
    p.add_argument("--backup-speed", type=float, default=0.25)
    p.add_argument(
        "--home-max-vel",
        type=float,
        default=0.6,
        help="home.py --max-vel after the backup (default home.py cap is 0.3)",
    )
    p.add_argument("--home-name", default="table")
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Experiment YAML (default: teleop_table_wipe.yaml)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the queue and folder counts, then exit",
    )
    return p.parse_args(argv)


def _rospack(package: str) -> Path:
    out = subprocess.check_output(
        ["rospack", "find", package], universal_newlines=True
    ).strip()
    return Path(out)


def _results_root() -> Path:
    return _rospack("mm_run") / "results" / TABLE_WIPE_LOG_DIR


def _outputs_dir() -> Path:
    return _rospack("mm_run") / "results" / "il" / "outputs"


def _default_config() -> Path:
    return _rospack("mm_run") / "config" / "mpsf_experiments" / "teleop_table_wipe.yaml"


def _tty_prompt(msg: str) -> str:
    with open("/dev/tty", "r") as tin, open("/dev/tty", "w") as tout:
        tout.write(msg)
        tout.flush()
        line = tin.readline()
        if line == "":
            raise EOFError("EOF on /dev/tty")
        return line.rstrip("\n")


def _cells(args):
    if args.sweep == "decay":
        trials = DECAY_TRIALS if args.trials is None else args.trials
        return decay_blend_cells(trials=trials)
    if args.sweep == "ensemble":
        trials = ENSEMBLE_TRIALS if args.trials is None else args.trials
        return ensemble_coeff_cells(trials=trials)
    trials = MODEL_TRIALS if args.trials is None else args.trials
    return model_cells(
        boundary_blend=args.boundary_blend,
        trials=trials,
        n_action_steps=args.n_action_steps,
    )


def _launch(cell: SweepCell, config: Path, checkpoint: Path) -> int:
    cmd = [
        "roslaunch",
        "mm_run",
        "hardware_teleop.launch",
        "config:=%s" % config,
        "teleop:=%s" % cell.teleop,
        "act_checkpoint:=%s" % checkpoint,
        "logging_sub_folder:=%s" % cell.logging_sub_folder,
        "act_n_action_steps:=%d" % int(cell.n_action_steps),
        "act_boundary_blend:=%s" % ("%g" % float(cell.boundary_blend)),
        "act_boundary_blend_mode:=%s" % cell.boundary_blend_mode,
        "act_temporal_ensemble_coeff:=%s" % cell.temporal_ensemble_coeff,
        "record_bag:=true",
    ]
    print("Square starts the trial. Circle ends it.", flush=True)
    print(" ".join(cmd), flush=True)
    return int(subprocess.call(cmd, stdin=subprocess.DEVNULL))


def _backup(distance: float, speed: float) -> int:
    script = Path(__file__).resolve().parent / "backup_base.py"
    return int(
        subprocess.call(
            [
                sys.executable,
                str(script),
                "--distance",
                "%g" % float(distance),
                "--speed",
                "%g" % float(speed),
            ]
        )
    )


def _home(name: str, max_vel: float) -> int:
    home_py = (
        _rospack("mobile_manipulation_central") / "scripts" / "control" / "home.py"
    )
    return int(
        subprocess.call(
            [
                sys.executable,
                str(home_py),
                name,
                "--max-vel",
                "%g" % float(max_vel),
            ]
        )
    )


def _reset(args) -> None:
    print(
        "Backing up %.2fm, then homing to %s at max-vel %g."
        % (args.backup_distance, args.home_name, args.home_max_vel),
        flush=True,
    )
    rc = _backup(args.backup_distance, args.backup_speed)
    if rc != 0:
        print(
            "Backup failed (exit %d). Clear the arm yourself, then Enter to home." % rc,
            flush=True,
        )
        _tty_prompt("")
    rc = _home(args.home_name, args.home_max_vel)
    if rc != 0:
        print("home.py exited %d." % rc, flush=True)


def _ask(new_dir: Path) -> str:
    if new_dir is None or not Path(new_dir).is_dir():
        hint = (
            "\nNo trial folder was saved.\n"
            "Enter = retry this cell, s = skip, q = stop\n> "
        )
        allowed = {"keep", "skip", "quit"}
    else:
        hint = (
            "\nTrial folder: %s\n"
            "Enter = keep, c = keep and mark a collision, "
            "d = delete and retry, s = skip for later, q = stop\n> " % new_dir
        )
        allowed = {"keep", "collision", "delete", "skip", "quit"}
    while True:
        action = after_trial_action(_tty_prompt(hint))
        if action in allowed:
            return action
        if "collision" in allowed:
            print("Enter, c, d, s, or q.", flush=True)
        else:
            print("Enter, s, or q.", flush=True)


def run_sweep(args) -> int:
    config = Path(args.config) if args.config else _default_config()
    if not config.is_file():
        raise SystemExit("config not found: %s" % config)
    outputs = _outputs_dir()
    root = _results_root()
    cells = _cells(args)

    print("Results root: %s" % root, flush=True)
    for cell in cells:
        done = len(trial_dirs(root / cell.logging_sub_folder))
        print(
            "  %s  %d/%d  teleop=%s  na=%d  blend=%g  ens=%s  %s"
            % (
                cell.name,
                done,
                cell.trials,
                cell.teleop,
                cell.n_action_steps,
                cell.boundary_blend,
                cell.temporal_ensemble_coeff,
                cell.checkpoint(outputs),
            ),
            flush=True,
        )
    if args.dry_run:
        return 0

    for cell in cells:
        cell_dir = root / cell.logging_sub_folder
        checkpoint = cell.checkpoint(outputs)
        if not (checkpoint / "config.json").is_file():
            print(
                "Missing checkpoint, skipping %s: %s" % (cell.name, checkpoint),
                flush=True,
            )
            continue
        while len(trial_dirs(cell_dir)) < cell.trials:
            done = len(trial_dirs(cell_dir))
            print(
                "\n=== %s  trial %d/%d  na=%d  blend=%g  ens=%s  %s ==="
                % (
                    cell.name,
                    done + 1,
                    cell.trials,
                    cell.n_action_steps,
                    cell.boundary_blend,
                    cell.temporal_ensemble_coeff,
                    cell.teleop,
                ),
                flush=True,
            )
            before = trial_dirs(cell_dir)
            _launch(cell, config, checkpoint)
            created = new_trial_dir(cell_dir, before)
            action = _ask(created)
            if action == "collision" and created is not None and created.is_dir():
                (created / "collision.txt").write_text("collision\n")
                print("Marked collision in %s" % created, flush=True)
            if action == "delete" and created is not None and created.is_dir():
                shutil.rmtree(created)
                print("Deleted %s" % created, flush=True)
            _reset(args)
            if action == "quit":
                print("Stopped. Run the same command again to resume.", flush=True)
                return 0
            if action == "skip":
                print("Leaving %s short. Resume will fill it." % cell.name, flush=True)
                break
    print("Sweep finished.", flush=True)
    return 0


def main(argv=None) -> int:
    return run_sweep(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
