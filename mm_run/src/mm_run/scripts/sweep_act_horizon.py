#!/usr/bin/env python3
"""Overnight ACT horizon A/B sweep: train + eval (chunk_size × n_action_steps).

Fixed stack (not swept):
  - dataset: table_wipe_act_ee_mpc3 (desired_ee_vel / obs=q / env=obs_duplicate)
  - STEPS default 40000, model size from train_act_smoke.sh defaults
  - eval: headless, env_state=obs_duplicate, n=10

Deploy reads ``n_action_steps`` from the checkpoint (act_ee --n-action-steps 0).

Example:
  source /opt/ros/noetic/setup.bash && source ~/catkin_ws/devel/setup.bash
  export ACADOS_SOURCE_DIR=$HOME/safe_control/acados
  export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
  export MM_IL_PYTHON=$HOME/miniconda3/envs/mm-il/bin/python

  python3 mm_run/src/mm_run/scripts/sweep_act_horizon.py \\
    --dataset mm_run/results/il/datasets/table_wipe_act_ee_mpc3 \\
    --repo-id local/table_wipe_act_ee_mpc3 \\
    --sweep-root mm_run/results/il/sweeps/horizon_ee_mpc3 \\
    --trials 10 --steps 40000 --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# (chunk_size, n_action_steps); baseline is (50, 25).
HORIZON_GRID: Tuple[Tuple[int, int], ...] = (
    (25, 10),
    (25, 25),
    (50, 10),
    (50, 25),
    (50, 50),
    (75, 25),
    (75, 50),
    (100, 25),
    (100, 50),
    (100, 100),
)


def _repo_root() -> Path:
    # .../mm_run/src/mm_run/scripts/this.py → mobile_manipulation
    return Path(__file__).resolve().parents[4]


def _default_mm_il_python() -> str:
    return os.environ.get(
        "MM_IL_PYTHON",
        os.path.expanduser("~/miniconda3/envs/mm-il/bin/python"),
    )


def parse_args(argv=None):
    root = _repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        default=root / "mm_run/results/il/datasets/table_wipe_act_ee_mpc3",
        help="LeRobot dataset root",
    )
    p.add_argument(
        "--repo-id",
        default="local/table_wipe_act_ee_mpc3",
        help="LeRobot repo_id for train",
    )
    p.add_argument(
        "--sweep-root",
        type=Path,
        default=root / "mm_run/results/il/sweeps/horizon_ee_mpc3",
        help="Directory for runs/ + leaderboard",
    )
    p.add_argument("--trials", type=int, default=10, help="Eval trials per config")
    p.add_argument("--steps", type=int, default=40000, help="Train steps per config")
    p.add_argument(
        "--boot-timeout",
        type=float,
        default=180.0,
        help="Passed to eval_act_smoke.py",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip configs that already have train checkpoint + eval summary",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned configs and exit",
    )
    p.add_argument(
        "--mm-il-python",
        default=None,
        help="Interpreter for train (default: MM_IL_PYTHON or ~/miniconda3/envs/mm-il/bin/python)",
    )
    p.add_argument(
        "--eval-python",
        default="/usr/bin/python3",
        help="Interpreter for eval_act_smoke (ROS Noetic system Python)",
    )
    return p.parse_args(argv)


def _run_id(chunk_size: int, n_action_steps: int) -> str:
    return f"cs{int(chunk_size)}_na{int(n_action_steps)}"


def _log(sweep_root: Path, msg: str) -> None:
    line = f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    log_path = sweep_root / "sweep.log"
    with log_path.open("a") as f:
        f.write(line + "\n")


def _check_env(*, mm_il_python: str, eval_python: str) -> None:
    errors = []
    if not Path(mm_il_python).is_file():
        errors.append(f"mm-il python missing: {mm_il_python} (set MM_IL_PYTHON)")
    if not Path(eval_python).is_file():
        errors.append(f"eval python missing: {eval_python}")
    try:
        subprocess.check_output(["rospack", "find", "mm_run"], universal_newlines=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        errors.append("rospack find mm_run failed; source ROS + catkin devel")
    acados = os.environ.get("ACADOS_SOURCE_DIR") or os.environ.get("ACADOS_INSTALL_DIR")
    if not acados:
        home_fb = Path.home() / "safe_control" / "acados"
        if not (home_fb / "lib" / "libacados.so").is_file():
            errors.append(
                "ACADOS_SOURCE_DIR not set and ~/safe_control/acados/lib missing"
            )
    if errors:
        raise SystemExit("env check failed:\n  - " + "\n  - ".join(errors))


def _train_done(train_dir: Path) -> bool:
    return (train_dir / "checkpoints/last/pretrained_model/config.json").is_file()


def _latest_eval_summary(train_dir: Path) -> Optional[Path]:
    summaries = (
        sorted(
            (train_dir / "evals").glob("*/summary.json"),
            key=lambda p: p.stat().st_mtime,
        )
        if (train_dir / "evals").is_dir()
        else []
    )
    return summaries[-1] if summaries else None


def _load_eval_summary(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _eval_usable(summary: Optional[Dict[str, Any]], *, min_trials: int = 1) -> bool:
    """True if enough trials produced path_coverage (empty overnight evals fail this)."""
    if not summary:
        return False
    if int(summary.get("n_trials") or 0) < int(min_trials):
        return False
    if int(summary.get("n_with_coverage") or 0) < 1:
        return False
    return summary.get("path_coverage_mean") is not None


def _env_for_eval() -> Dict[str, str]:
    """ROS/catkin PYTHONPATH must win over source mm_run/src (needs mm_run.msg)."""
    env = os.environ.copy()
    parts = []
    for p in env.get("PYTHONPATH", "").split(os.pathsep):
        if not p:
            continue
        if p.rstrip("/").endswith("mm_run/src"):
            continue
        parts.append(p)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _write_hparams(run_dir: Path, hparams: Dict[str, Any]) -> None:
    (run_dir / "hparams.json").write_text(json.dumps(hparams, indent=2) + "\n")


def _run_train(
    *,
    dataset: Path,
    train_dir: Path,
    repo_id: str,
    chunk_size: int,
    n_action_steps: int,
    steps: int,
    mm_il_python: str,
    script_dir: Path,
) -> int:
    train_sh = script_dir / "train_act_smoke.sh"
    env = os.environ.copy()
    env["CHUNK_SIZE"] = str(int(chunk_size))
    env["N_ACTION_STEPS"] = str(int(n_action_steps))
    env["STEPS"] = str(int(steps))
    env["REPO_ID"] = repo_id
    env["JOB_NAME"] = f"act_horizon_{_run_id(chunk_size, n_action_steps)}"
    env["MM_IL_PYTHON"] = mm_il_python
    # train_act_smoke.sh invokes `python -m mm_run.scripts.lerobot_train_act`
    env["PATH"] = str(Path(mm_il_python).parent) + os.pathsep + env.get("PATH", "")
    cmd = ["bash", str(train_sh), str(dataset), str(train_dir)]
    log_path = train_dir.parent / "train_subprocess.log"
    train_dir.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_repo_root()),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
        return int(proc.wait())


def _run_eval(
    *,
    train_dir: Path,
    trials: int,
    boot_timeout: float,
    eval_python: str,
    script_dir: Path,
) -> int:
    eval_py = script_dir / "eval_act_smoke.py"
    cmd = [
        eval_python,
        str(eval_py),
        str(train_dir),
        "--trials",
        str(int(trials)),
        "--env-state",
        "obs_duplicate",
        "--boot-timeout",
        str(float(boot_timeout)),
        "--checkpoint",
        "last",
    ]
    log_path = train_dir.parent / "eval_subprocess.log"
    with log_path.open("w") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_repo_root()),
            env=_env_for_eval(),
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
        return int(proc.wait())


def _leaderboard_row(
    *,
    chunk_size: int,
    n_action_steps: int,
    steps: int,
    train_dir: Path,
    eval_summary: Optional[Dict[str, Any]],
    eval_summary_path: Optional[Path],
    status: str,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    mean = eval_summary.get("path_coverage_mean") if eval_summary else None
    return {
        "run_id": _run_id(chunk_size, n_action_steps),
        "chunk_size": int(chunk_size),
        "n_action_steps": int(n_action_steps),
        "steps": int(steps),
        "status": status,
        "error": error,
        "path_coverage_mean": mean,
        "path_coverage_min": (
            eval_summary.get("path_coverage_min") if eval_summary else None
        ),
        "path_coverage_max": (
            eval_summary.get("path_coverage_max") if eval_summary else None
        ),
        "n_trials": eval_summary.get("n_trials") if eval_summary else None,
        "n_with_coverage": (
            eval_summary.get("n_with_coverage") if eval_summary else None
        ),
        "train_dir": str(train_dir),
        "eval_summary": str(eval_summary_path) if eval_summary_path else None,
    }


def _write_leaderboard(sweep_root: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ranked = sorted(
        rows,
        key=lambda r: (
            r.get("path_coverage_mean") is None,
            -(r.get("path_coverage_mean") or 0.0),
            r.get("run_id") or "",
        ),
    )
    (sweep_root / "leaderboard.json").write_text(
        json.dumps(
            {"updated_at": dt.datetime.now().isoformat(), "runs": ranked}, indent=2
        )
        + "\n"
    )
    fields = [
        "run_id",
        "chunk_size",
        "n_action_steps",
        "steps",
        "status",
        "path_coverage_mean",
        "path_coverage_min",
        "path_coverage_max",
        "n_trials",
        "n_with_coverage",
        "train_dir",
        "eval_summary",
        "error",
    ]
    with (sweep_root / "leaderboard.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in ranked:
            w.writerow(row)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.trials < 1:
        raise SystemExit("--trials must be >= 1")
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")

    repo = _repo_root()
    dataset = Path(args.dataset).resolve()
    sweep_root = Path(args.sweep_root).resolve()
    script_dir = Path(__file__).resolve().parent
    mm_il_python = args.mm_il_python or _default_mm_il_python()

    for cs, na in HORIZON_GRID:
        if not (1 <= int(na) <= int(cs)):
            raise SystemExit(f"invalid grid pair chunk={cs} n_action={na}")

    if args.dry_run:
        print(f"dataset={dataset}")
        print(f"sweep_root={sweep_root}")
        print(f"trials={args.trials} steps={args.steps}")
        print(f"mm_il_python={mm_il_python}")
        print(f"eval_python={args.eval_python}")
        for cs, na in HORIZON_GRID:
            print(f"  {_run_id(cs, na)}")
        return 0

    if not dataset.is_dir():
        raise SystemExit(f"dataset not found: {dataset}")
    _check_env(mm_il_python=mm_il_python, eval_python=args.eval_python)

    sweep_root.mkdir(parents=True, exist_ok=True)
    runs_root = sweep_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    (sweep_root / "sweep_meta.json").write_text(
        json.dumps(
            {
                "dataset": str(dataset),
                "repo_id": args.repo_id,
                "trials": int(args.trials),
                "steps": int(args.steps),
                "grid": [
                    {"chunk_size": c, "n_action_steps": n} for c, n in HORIZON_GRID
                ],
                "repo_root": str(repo),
                "started_at": dt.datetime.now().isoformat(),
            },
            indent=2,
        )
        + "\n"
    )

    _log(
        sweep_root,
        f"start sweep grid={len(HORIZON_GRID)} trials={args.trials} steps={args.steps}",
    )
    rows: List[Dict[str, Any]] = []

    for chunk_size, n_action_steps in HORIZON_GRID:
        rid = _run_id(chunk_size, n_action_steps)
        run_dir = runs_root / rid
        train_dir = run_dir / "train"
        run_dir.mkdir(parents=True, exist_ok=True)
        hparams = {
            "chunk_size": int(chunk_size),
            "n_action_steps": int(n_action_steps),
            "steps": int(args.steps),
            "trials": int(args.trials),
            "dataset": str(dataset),
            "repo_id": args.repo_id,
            "env_state": "obs_duplicate",
        }
        _write_hparams(run_dir, hparams)

        eval_summary_path = (
            _latest_eval_summary(train_dir) if train_dir.is_dir() else None
        )
        if (
            args.skip_existing
            and _train_done(train_dir)
            and eval_summary_path is not None
        ):
            summary = _load_eval_summary(eval_summary_path)
            if _eval_usable(summary, min_trials=args.trials):
                rows.append(
                    _leaderboard_row(
                        chunk_size=chunk_size,
                        n_action_steps=n_action_steps,
                        steps=args.steps,
                        train_dir=train_dir,
                        eval_summary=summary,
                        eval_summary_path=eval_summary_path,
                        status="skipped_existing",
                    )
                )
                _write_leaderboard(sweep_root, rows)
                _log(
                    sweep_root,
                    f"SKIP {rid} mean=" f"{summary.get('path_coverage_mean')}",
                )
                continue
            _log(
                sweep_root,
                f"RE-EVAL {rid}: prior summary incomplete "
                f"(n_trials={summary.get('n_trials')} "
                f"n_with_coverage={summary.get('n_with_coverage')})",
            )

        _log(sweep_root, f"TRAIN {rid} chunk={chunk_size} n_action={n_action_steps}")
        t0 = time.perf_counter()
        try:
            import shutil

            if _train_done(train_dir):
                _log(sweep_root, f"TRAIN {rid} checkpoint present; eval only")
                rc_train = 0
            else:
                # Fresh train: LeRobot refuses existing output_dir.
                if train_dir.exists():
                    _log(sweep_root, f"TRAIN {rid} removing incomplete {train_dir}")
                    shutil.rmtree(train_dir)
                rc_train = _run_train(
                    dataset=dataset,
                    train_dir=train_dir,
                    repo_id=args.repo_id,
                    chunk_size=chunk_size,
                    n_action_steps=n_action_steps,
                    steps=args.steps,
                    mm_il_python=mm_il_python,
                    script_dir=script_dir,
                )
            if rc_train != 0 or not _train_done(train_dir):
                raise RuntimeError(
                    f"train failed rc={rc_train} (see {run_dir / 'train_subprocess.log'})"
                )
        except Exception as exc:  # noqa: BLE001 - keep sweep going
            rows.append(
                _leaderboard_row(
                    chunk_size=chunk_size,
                    n_action_steps=n_action_steps,
                    steps=args.steps,
                    train_dir=train_dir,
                    eval_summary=None,
                    eval_summary_path=None,
                    status="train_failed",
                    error=str(exc),
                )
            )
            _write_leaderboard(sweep_root, rows)
            _log(sweep_root, f"FAIL TRAIN {rid}: {exc}")
            continue

        _log(
            sweep_root,
            f"TRAIN {rid} ok in {time.perf_counter() - t0:.0f}s; EVAL n={args.trials}",
        )
        t1 = time.perf_counter()
        try:
            rc_eval = _run_eval(
                train_dir=train_dir,
                trials=args.trials,
                boot_timeout=args.boot_timeout,
                eval_python=args.eval_python,
                script_dir=script_dir,
            )
            eval_summary_path = _latest_eval_summary(train_dir)
            if rc_eval != 0 or eval_summary_path is None:
                raise RuntimeError(
                    f"eval failed rc={rc_eval} (see {run_dir / 'eval_subprocess.log'})"
                )
            summary = _load_eval_summary(eval_summary_path)
            if not _eval_usable(summary, min_trials=args.trials):
                raise RuntimeError(
                    f"eval incomplete "
                    f"(n_trials={summary.get('n_trials')} "
                    f"n_with_coverage={summary.get('n_with_coverage')}; "
                    f"see {run_dir / 'eval_subprocess.log'} and trial roslaunch.log)"
                )
            rows.append(
                _leaderboard_row(
                    chunk_size=chunk_size,
                    n_action_steps=n_action_steps,
                    steps=args.steps,
                    train_dir=train_dir,
                    eval_summary=summary,
                    eval_summary_path=eval_summary_path,
                    status="ok",
                )
            )
            mean = summary.get("path_coverage_mean")
            mean_s = f"{100.0 * mean:.1f}%" if mean is not None else "n/a"
            _log(
                sweep_root,
                f"OK {rid} mean={mean_s} eval_s={time.perf_counter() - t1:.0f}",
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                _leaderboard_row(
                    chunk_size=chunk_size,
                    n_action_steps=n_action_steps,
                    steps=args.steps,
                    train_dir=train_dir,
                    eval_summary=None,
                    eval_summary_path=None,
                    status="eval_failed",
                    error=str(exc),
                )
            )
            _log(sweep_root, f"FAIL EVAL {rid}: {exc}")
        _write_leaderboard(sweep_root, rows)

    _write_leaderboard(sweep_root, rows)
    _log(sweep_root, f"done → {sweep_root / 'leaderboard.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
