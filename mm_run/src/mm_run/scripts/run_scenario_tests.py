#!/usr/bin/env python3
"""Run the MPC scenario configs and report task completion plus solve timing.

Source the workspace (and acados ``LD_LIBRARY_PATH``) before running; the child
processes inherit this environment.

    rosrun mm_run run_scenario_tests.py                      # ROS + PyBullet stack
    rosrun mm_run run_scenario_tests.py --mode pybullet      # experiment.py, no roslaunch
    rosrun mm_run run_scenario_tests.py --trials 5

Exits non-zero if any trial fails.
"""

import argparse
import contextlib
import ctypes
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rospkg

from mm_utils import parsing

# Scenarios driven by mpc_ros/experiment.py. The MPSF configs are excluded: they
# need the mpsf plan node and a teleop velocity source.
DEFAULT_CONFIGS = [
    "tests/test_position_only.yaml",
    "tests/test_orientation_only.yaml",
    "tests/test_complex_waypoints.yaml",
    "tests/test_path_planning.yaml",
    "tests/test_upright_payload.yaml",
    "simple_experiment.yaml",
    "3d_collision.yaml",
]

FAILURE_MARKERS = (
    "MPC solve failed",
    "ACADOS_MINSTEP",
    "acados_ocp_solver returned status",
    "Collision E-stop",
    "undefined symbol",
    "TIMEOUT after",
)

# sim_ros exits on its own at the end of the duration, so only the control nodes
# dying indicates a real failure.
CONTROL_NODE_DIED = re.compile(
    r"process \[(controller_mpc|low_level_cmd_node)-\d+\] has died"
)


def preflight():
    """Fail fast if acados cannot be loaded, which would kill every run.

    ``source devel/setup.bash`` can drop the acados lib dir from
    ``LD_LIBRARY_PATH``, leaving libacados unable to resolve HPIPM symbols.
    """
    acados_dir = os.environ.get("ACADOS_SOURCE_DIR")
    if not acados_dir:
        return
    lib = Path(acados_dir) / "lib" / "libacados.so"
    if not lib.is_file():
        return
    try:
        ctypes.CDLL(str(lib))
    except OSError as exc:
        sys.exit(
            f"cannot load {lib}: {exc}\n"
            f"export LD_LIBRARY_PATH={lib.parent}:$LD_LIBRARY_PATH "
            "after sourcing the workspace, then rerun."
        )


def parse_run(text):
    """Extract completion markers and solve times from a run's output."""
    solve = [float(x) for x in re.findall(r"Controller Run Time: ([0-9.]+)", text)]
    failures = [m for m in FAILURE_MARKERS if m in text]
    died = CONTROL_NODE_DIED.search(text)
    if died:
        failures.append(f"{died.group(1)} died")
    return {
        "solve": np.asarray(solve, dtype=float),
        "tasks_completed": "All tasks completed" in text,
        "reach_errs": re.findall(
            r"(?:EE|base) reached \(pos_err: ([0-9.]+), ori_err: [0-9.]+\)", text
        ),
        "failures": failures,
    }


def _terminate_group(proc):
    """SIGINT the process group so roslaunch unwinds its nodes, then SIGKILL."""
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return proc.communicate()[0]
    try:
        return proc.communicate(timeout=20)[0]
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        return proc.communicate()[0]


def run_trial(cfg_path, mode, trial_dir, gui, timeout_s, pkg):
    """Run one scenario once; return (returncode, parsed metrics)."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    if mode == "ros":
        cmd = [
            "roslaunch",
            "mm_run",
            "run_pybullet_sim.launch",
            f"config:={cfg_path}",
            f"gui:={'true' if gui else 'false'}",
        ]
    else:
        exp = pkg / "src/mm_run/scripts/experiment.py"
        cmd = [sys.executable, str(exp), "--config", str(cfg_path)]
        if gui:
            cmd.append("--GUI")

    env = dict(os.environ)
    if mode == "ros":
        env["ROS_LOG_DIR"] = str(trial_dir)

    # stdin must not be a tty: mpc_ros waits on an Enter prompt when it is one.
    # Own process group so a timeout tears down every child, not just the parent.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        out = proc.communicate(timeout=timeout_s)[0] or ""
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        out = (_terminate_group(proc) or "") + f"\nTIMEOUT after {timeout_s:.0f}s"
        rc = -1
    except KeyboardInterrupt:
        print("\ninterrupted: shutting down...", flush=True)
        _terminate_group(proc)
        raise
    (trial_dir / "run.out").write_text(out)

    node_log = "\n".join(
        p.read_text(errors="replace")
        for p in sorted(trial_dir.rglob("controller_mpc*.log"))
    )
    m = parse_run(node_log if "Controller Run Time" in node_log else out)
    m["failures"] = sorted(set(m["failures"]) | set(parse_run(out)["failures"]))
    m["tasks_completed"] = m["tasks_completed"] or "All tasks completed" in out
    return rc, m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("ros", "pybullet", "sync"),
        default="ros",
        help="ros: roslaunch stack. pybullet: experiment.py in-process (sync is an alias).",
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--configs",
        default=",".join(DEFAULT_CONFIGS),
        help="Comma-separated configs, absolute or relative to mm_run/config",
    )
    parser.add_argument("--gui", action="store_true", help="Show the PyBullet GUI")
    parser.add_argument(
        "--timeout-factor",
        type=float,
        default=6.0,
        help="Per-trial timeout = 60 s + factor * simulation.duration",
    )
    args = parser.parse_args()
    if args.mode == "sync":
        args.mode = "pybullet"
    preflight()

    pkg = Path(rospkg.RosPack().get_path("mm_run"))
    cfg_root = pkg / "config"
    out_root = (
        pkg
        / "results"
        / "scenario_tests"
        / args.mode
        / time.strftime("%Y-%m-%d_%H-%M-%S")
    )

    configs = []
    for entry in (c.strip() for c in args.configs.split(",")):
        if not entry:
            continue
        path = Path(entry) if Path(entry).is_absolute() else cfg_root / entry
        if not path.is_file():
            parser.error(f"config not found: {path}")
        configs.append(path)

    print(f"mode={args.mode} trials={args.trials} out={out_root}\n", flush=True)
    rows = []
    n_failed = 0

    for cfg in configs:
        loaded = parsing.load_config(str(cfg))
        dt = float(loaded["controller"]["dt"])
        ctrl_rate = float(loaded["controller"]["ctrl_rate"])
        period = 1.0 / ctrl_rate
        duration = float(loaded["simulation"]["duration"])
        timeout_s = 60.0 + args.timeout_factor * duration
        label = str(cfg.relative_to(cfg_root)) if cfg_root in cfg.parents else cfg.name

        print(
            f"===== {label} (duration={duration:g}s dt={dt:g}s ctrl_rate={ctrl_rate:g}Hz) =====",
            flush=True,
        )
        pooled = []
        n_pass = 0
        slug = re.sub(r"\.yaml$", "", label).replace("/", "_")
        for trial in range(1, args.trials + 1):
            trial_dir = out_root / slug / f"trial{trial}"
            rc, m = run_trial(cfg, args.mode, trial_dir, args.gui, timeout_s, pkg)
            solve = m["solve"]
            ok = rc == 0 and m["tasks_completed"] and not m["failures"]
            n_pass += ok
            pooled += list(solve)
            over = int((solve > period).sum()) if solve.size else 0
            stats = (
                f"ticks={solve.size:3d} med={np.median(solve):.4f} "
                f"p95={np.percentile(solve, 95):.4f} max={solve.max():.4f} "
                f"over_period={over}"
                if solve.size
                else "no solve data"
            )
            note = ""
            if m["failures"]:
                note = "  " + ", ".join(m["failures"])
            elif rc != 0:
                note = f"  exit={rc}"
            elif not m["tasks_completed"]:
                note = "  tasks NOT completed"
            print(
                f"  trial{trial}: {'PASS' if ok else 'FAIL'}  {stats}"
                f"  final_err={m['reach_errs'][-1] if m['reach_errs'] else 'n/a'}{note}",
                flush=True,
            )

        pooled = np.asarray(pooled, dtype=float)
        summary = f"{label}: {n_pass}/{args.trials} passed"
        if pooled.size:
            summary += (
                f" | solve med={np.median(pooled):.4f} "
                f"p95={np.percentile(pooled, 95):.4f} max={pooled.max():.4f} "
                f"over_period={int((pooled > period).sum())}/{pooled.size}"
            )
        print(f"  -> {summary}\n", flush=True)
        rows.append(summary)
        n_failed += args.trials - n_pass

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.txt").write_text("\n".join(rows) + "\n")
    print("===== SUMMARY =====")
    print("\n".join(rows))
    print(f"\nlogs: {out_root}")
    print("ALL PASSED" if n_failed == 0 else f"{n_failed} trial(s) FAILED")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
