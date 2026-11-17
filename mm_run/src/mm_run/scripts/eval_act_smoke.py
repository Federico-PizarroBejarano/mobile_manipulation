#!/usr/bin/env python3
"""Unattended ACT PyBullet eval: run N trials against a train output checkpoint.

Defaults: checkpoint=last, N=5, gui=false, teleop inferred from action width
(9->act, 3/6->act_ee). Trial length is ``simulation.duration`` (20s for table wipe);
sim exit tears down the launch (``required:=true``).

Example:
  python3 mm_run/src/mm_run/scripts/eval_act_smoke.py \\
    mm_run/results/il/outputs/act_table_wipe_ee_mpc

Writes ``OUTPUT_DIR/evals/<stamp>_<ckpt>/trial_XX/`` plus ``summary.json``.
"""

import sys

if sys.version_info[0] < 3:
    sys.stderr.write(
        "eval_act_smoke.py requires Python 3 (got Python %s). Use python3.\n"
        % (sys.version.split()[0],)
    )
    sys.exit(1)

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from mm_utils.act_eval import (
    aggregate_eval_trials,
    find_session_metrics,
    infer_teleop_from_checkpoint,
    parse_metrics_summary,
    resolve_pretrained_model,
    write_eval_overlay,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "output_dir",
        type=Path,
        help="Train output directory containing checkpoints/",
    )
    p.add_argument(
        "--checkpoint",
        default="last",
        help="Checkpoint folder name under checkpoints/ (default: last)",
    )
    p.add_argument("-n", "--trials", type=int, default=10, help="Number of trials")
    p.add_argument(
        "--gui",
        action="store_true",
        help="Enable PyBullet GUI (default: headless)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Experiment YAML (default: teleop_table_wipe.yaml via rospack)",
    )
    p.add_argument(
        "--boot-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for roslaunch to finish beyond duration",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Override simulation.duration seconds (default: YAML value)",
    )
    p.add_argument(
        "--env-state",
        default="obs_duplicate",
        choices=("obs_duplicate", "ee_pose", "zeros"),
        help="ACT observation.environment_state for teleop:=act_ee",
    )
    p.add_argument(
        "--n-action-steps",
        type=int,
        default=12,
        help="ACT execute horizon (frozen recipe 12; 0=from checkpoint config.json)",
    )
    p.add_argument(
        "--ctrl-rate",
        type=float,
        default=10.0,
        help="controller.ctrl_rate [Hz] via eval overlay (frozen recipe 10)",
    )
    p.add_argument(
        "--boundary-blend",
        type=float,
        default=0.5,
        help="ACT boundary blend weight on prior leftover [0,1] (frozen: 0.5)",
    )
    p.add_argument(
        "--boundary-blend-mode",
        default="decay",
        choices=("constant", "decay", "head"),
        help="constant|decay|head (frozen: decay)",
    )
    p.add_argument(
        "--boundary-blend-head",
        type=int,
        default=3,
        help="For mode=head: steps at the seam to blend",
    )
    p.add_argument(
        "--temporal-ensemble-coeff",
        default=None,
        type=float,
        help="Paper-style temporal ensembling coeff (omit=off; forces na=1)",
    )
    return p.parse_args(argv)


def _default_config():
    try:
        out = subprocess.check_output(
            ["rospack", "find", "mm_run"], universal_newlines=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise SystemExit("rospack find mm_run failed; source ROS + catkin devel first")
    return Path(out) / "config" / "mpsf_experiments" / "teleop_table_wipe.yaml"


def _duration_from_config(config_path, override):
    if override is not None:
        return float(override)
    # Avoid a hard YAML dependency: table-wipe default is 20; scan the file.
    text = Path(config_path).read_text()
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("duration:"):
            return float(s.split(":", 1)[1].strip().split()[0])
    return 20.0


def _acados_lib_dir():
    """Directory that must lead LD_LIBRARY_PATH for HPIPM symbols used by acados."""
    for key in ("ACADOS_SOURCE_DIR", "ACADOS_INSTALL_DIR"):
        root = os.environ.get(key)
        if root:
            lib = Path(root) / "lib"
            if (lib / "libacados.so").is_file():
                return lib
    fallback = Path.home() / "safe_control" / "acados" / "lib"
    if (fallback / "libacados.so").is_file():
        return fallback
    return None


def _env_with_acados():
    """Prepend acados lib so catkin devel's older libhpipm does not win."""
    env = os.environ.copy()
    env.setdefault("ROS_MASTER_URI", "http://localhost:11311")
    # Source mm_run/src shadows catkin-generated mm_run.msg — drop it for roslaunch.
    parts = []
    for p in env.get("PYTHONPATH", "").split(os.pathsep):
        if not p:
            continue
        if p.rstrip("/").endswith("mm_run/src"):
            continue
        parts.append(p)
    # pinocchio (and friends) ship in noetic's python3.8 site-packages; setup.bash
    # does not always put that dir on PYTHONPATH in non-interactive shells.
    noetic_site = "/opt/ros/noetic/lib/python3.8/site-packages"
    if os.path.isdir(noetic_site) and noetic_site not in parts:
        parts.insert(0, noetic_site)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    acados_lib = _acados_lib_dir()
    if acados_lib is not None:
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            str(acados_lib) if not prev else "%s:%s" % (acados_lib, prev)
        )
    return env


def _run_trial(
    *,
    trial_dir,
    config,
    overlay,
    teleop,
    checkpoint,
    gui,
    timeout_s,
    duration_s=None,
    env_state="obs_duplicate",
    n_action_steps=0,
    ctrl_rate=None,
    boundary_blend=0.5,
    boundary_blend_mode="decay",
    boundary_blend_head=3,
    temporal_ensemble_coeff=None,
):
    trial_dir.mkdir(parents=True, exist_ok=True)
    write_eval_overlay(
        overlay,
        log_dir=trial_dir,
        duration_s=duration_s,
        ctrl_rate=ctrl_rate,
    )
    env = _env_with_acados()
    te = (
        "none"
        if temporal_ensemble_coeff is None
        else f"{float(temporal_ensemble_coeff):g}"
    )
    cmd = [
        "roslaunch",
        "mm_run",
        "run_pybullet_sim.launch",
        f"config:={config}",
        f"ctrl_config:={overlay}",
        f"teleop:={teleop}",
        f"act_checkpoint:={checkpoint}",
        f"act_env_state:={env_state}",
        f"act_n_action_steps:={int(n_action_steps)}",
        f"act_boundary_blend:={float(boundary_blend)}",
        f"act_boundary_blend_mode:={boundary_blend_mode}",
        f"act_boundary_blend_head:={int(boundary_blend_head)}",
        f"act_temporal_ensemble_coeff:={te}",
        "auto_start:=true",
        f"gui:={'true' if gui else 'false'}",
        "record_bag:=false",
    ]
    log_path = trial_dir / "roslaunch.log"
    print(f"[eval] launching trial → {trial_dir}", flush=True)
    with log_path.open("w") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )
        try:
            return int(proc.wait(timeout=timeout_s))
        except subprocess.TimeoutExpired:
            print(
                f"[eval] trial timed out after {timeout_s:.0f}s; killing launch",
                flush=True,
            )
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                return int(proc.wait(timeout=30))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                return int(proc.wait())


def main(argv=None):
    args = parse_args(argv)
    if args.trials < 1:
        raise SystemExit("--trials must be >= 1")

    output_dir = Path(args.output_dir).resolve()
    pretrained = resolve_pretrained_model(output_dir, args.checkpoint)
    teleop, nu = infer_teleop_from_checkpoint(pretrained)
    config = Path(args.config).resolve() if args.config else _default_config()
    if not config.is_file():
        raise SystemExit(f"config not found: {config}")
    duration = _duration_from_config(config, args.duration)
    timeout_s = float(duration) + float(args.boot_timeout)

    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    eval_root = output_dir / "evals" / f"{stamp}_{args.checkpoint}"
    eval_root.mkdir(parents=True, exist_ok=False)

    meta = {
        "output_dir": str(output_dir),
        "checkpoint": args.checkpoint,
        "pretrained_model": str(pretrained),
        "teleop": teleop,
        "action_width": nu,
        "env_state": args.env_state,
        "n_action_steps": int(args.n_action_steps),
        "ctrl_rate": (float(args.ctrl_rate) if args.ctrl_rate is not None else None),
        "boundary_blend": float(args.boundary_blend),
        "boundary_blend_mode": str(args.boundary_blend_mode),
        "boundary_blend_head": int(args.boundary_blend_head),
        "temporal_ensemble_coeff": (
            None
            if args.temporal_ensemble_coeff is None
            else float(args.temporal_ensemble_coeff)
        ),
        "config": str(config),
        "gui": bool(args.gui),
        "duration_s": duration,
        "n_requested": int(args.trials),
        "started_at": stamp,
    }
    (eval_root / "eval_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    acados_lib = _acados_lib_dir()
    rate_s = f"{float(args.ctrl_rate):g}Hz" if args.ctrl_rate is not None else "yaml"
    te_s = (
        "off"
        if args.temporal_ensemble_coeff is None
        else f"{float(args.temporal_ensemble_coeff):g}"
    )
    print(
        f"[eval] teleop={teleop} (nu={nu}) env_state={args.env_state} "
        f"n_action_steps={int(args.n_action_steps) or 'checkpoint'} "
        f"ctrl_rate={rate_s} boundary_blend={float(args.boundary_blend):g} "
        f"mode={args.boundary_blend_mode} head={int(args.boundary_blend_head)} "
        f"temporal_ensemble={te_s} "
        f"checkpoint={pretrained}\n"
        f"[eval] writing to {eval_root}\n"
        f"[eval] LD_LIBRARY_PATH prepend={acados_lib}",
        flush=True,
    )

    if acados_lib is None:
        print(
            "[eval] warning: acados lib not found; MPSF/act_ee may fail to load",
            flush=True,
        )

    trial_records = []
    for i in range(1, args.trials + 1):
        trial_dir = eval_root / f"trial_{i:02d}"
        overlay = trial_dir / "eval_overlay.yaml"
        rc = _run_trial(
            trial_dir=trial_dir,
            config=config,
            overlay=overlay,
            teleop=teleop,
            checkpoint=pretrained,
            gui=bool(args.gui),
            timeout_s=timeout_s,
            duration_s=args.duration,
            env_state=args.env_state,
            n_action_steps=int(args.n_action_steps),
            ctrl_rate=args.ctrl_rate,
            boundary_blend=float(args.boundary_blend),
            boundary_blend_mode=str(args.boundary_blend_mode),
            boundary_blend_head=int(args.boundary_blend_head),
            temporal_ensemble_coeff=args.temporal_ensemble_coeff,
        )
        # Brief pause so ports / masters settle between launches.
        time.sleep(2.0)
        summary_path = find_session_metrics(trial_dir)
        record = {
            "trial": i,
            "trial_dir": str(trial_dir),
            "roslaunch_exit": rc,
            "summary_path": str(summary_path) if summary_path else None,
            "path_coverage": None,
            "path_coverage_by_name": {},
            "state_violations": None,
        }
        if summary_path is not None:
            parsed = parse_metrics_summary(summary_path.read_text())
            record.update(parsed)
        trial_records.append(record)
        cov = record.get("path_coverage")
        cov_s = f"{100.0 * cov:.1f}%" if cov is not None else "n/a"
        print(
            f"[eval] trial {i}/{args.trials} exit={rc} coverage={cov_s}",
            flush=True,
        )

    summary = aggregate_eval_trials(trial_records)
    summary["meta"] = meta
    summary_path = eval_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    mean = summary.get("path_coverage_mean")
    mean_s = f"{100.0 * mean:.1f}%" if mean is not None else "n/a"
    print(f"[eval] done → {summary_path} (mean coverage {mean_s})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
