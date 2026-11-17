#!/usr/bin/env python3
"""Batch-convert teleop control/data.npz trials into a LeRobot dataset (ACT smoke).

Requires the IL env (see mm_run/requirements-il.txt):
  python export_teleop_to_lerobot.py --root /path/to/results/user --out /path/to/lerobot_ds \\
    --repo-id local/table_wipe_act_smoke [--obs q] [--action u_cmd|desired_ee_vel|...] \\
    [--env-state obs_duplicate|ee_pose|zeros]

Run export and training in the IL env. Deploy keeps ROS on system Python and
runs ACT inference via ``mm_utils.act_policy_worker`` under that same IL env.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mm_utils.il_episode import (
    ACTION_SPECS,
    ENV_STATE_SPECS,
    OBS_SPECS,
    extract_episode,
    find_control_npzs,
    resolve_action_spec,
    resolve_env_state_spec,
    resolve_obs_spec,
)
from mm_utils.lerobot_hf_features import get_hf_features_keep_len1_obs


def build_features(*, obs: str, action: str, env_state: str) -> dict:
    obs_spec = resolve_obs_spec(obs)
    action_spec = resolve_action_spec(action)
    env_spec = resolve_env_state_spec(env_state)
    obs_w = int(obs_spec["width"])
    act_w = int(action_spec["width"])
    env_w = obs_w if env_spec is None else int(env_spec["width"])
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (obs_w,),
            "names": None,
        },
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (env_w,),
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (act_w,),
            "names": None,
        },
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Results root to scan for control/data.npz",
    )
    p.add_argument("--out", type=Path, required=True, help="Output dataset directory")
    p.add_argument("--repo-id", type=str, default="local/table_wipe_act_smoke")
    p.add_argument("--task", type=str, default="table_wipe")
    p.add_argument(
        "--obs",
        type=str,
        default="q",
        choices=sorted(OBS_SPECS),
        help="Observation vector exported as observation.state",
    )
    p.add_argument(
        "--action",
        type=str,
        default="desired_ee_vel",
        choices=sorted(ACTION_SPECS),
        help="Action vector (frozen EE: desired_ee_vel; joint: u_cmd)",
    )
    p.add_argument(
        "--env-state",
        type=str,
        default="obs_duplicate",
        choices=sorted(ENV_STATE_SPECS),
        help="observation.environment_state (obs_duplicate|ee_pose|zeros)",
    )
    p.add_argument("--min-enabled-seconds", type=float, default=0.5)
    p.add_argument(
        "--default-fps",
        type=float,
        default=10.0,
        help="Fallback fps when episode dt is missing (frozen recipe: 10)",
    )
    p.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Keep only the first N successfully extracted episodes "
        "(sorted control/data.npz paths). Nested subsets: use the same "
        "--root with increasing N.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise SystemExit(
            "lerobot is required for export; install mm_run/requirements-il.txt in a venv"
        ) from exc

    paths = list(find_control_npzs(args.root))
    if not paths:
        raise SystemExit(f"no control/data.npz under {args.root}")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise SystemExit(f"--max-episodes must be >= 1, got {args.max_episodes}")

    features = build_features(
        obs=args.obs, action=args.action, env_state=args.env_state
    )
    episodes = []
    fps_votes = []
    for path in paths:
        if args.max_episodes is not None and len(episodes) >= args.max_episodes:
            break
        try:
            ep = extract_episode(
                path,
                obs=args.obs,
                action=args.action,
                env_state=args.env_state,
                min_enabled_seconds=args.min_enabled_seconds,
                default_fps=args.default_fps,
            )
        except (KeyError, ValueError) as err:
            print(f"skip {path}: {err}", file=sys.stderr)
            continue
        episodes.append((path, ep))
        fps_votes.append(int(ep["fps"]))

    if not episodes:
        raise SystemExit("no episodes extracted")
    if args.max_episodes is not None and len(episodes) < args.max_episodes:
        raise SystemExit(
            f"only extracted {len(episodes)} episodes, need {args.max_episodes}"
        )

    # Single fps for the dataset: median of per-episode fps
    fps = int(np.median(np.asarray(fps_votes, dtype=float)))
    # LeRobotDataset.create requires a non-existent root (exist_ok=False).
    if args.out.exists():
        raise SystemExit(
            f"output dataset path already exists: {args.out}\n"
            "Remove it or choose a new --out, then re-run."
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Patch for the whole write: create and save_episode both resolve hf_features
    # via get_hf_features_from_features (width-1 must stay a Sequence).
    with patch(
        "lerobot.datasets.lerobot_dataset.get_hf_features_from_features",
        get_hf_features_keep_len1_obs,
    ):
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=fps,
            features=features,
            root=str(args.out),
            robot_type="thing",
            use_videos=False,
        )

        provenance = []
        for path, ep in episodes:
            for i in range(ep["n_frames"]):
                dataset.add_frame(
                    {
                        "observation.state": ep["observation"][i],
                        "observation.environment_state": np.asarray(
                            ep["environment_state"][i], dtype=np.float32
                        ).reshape(-1),
                        "action": ep["action"][i],
                        "task": args.task,
                    }
                )
            dataset.save_episode()
            provenance.append(
                {
                    "source": str(path),
                    "n_frames": int(ep["n_frames"]),
                    "fps_episode": int(ep["fps"]),
                }
            )

        dataset.finalize()
    prov_path = Path(args.out) / "mm_export_provenance.json"
    prov_path.write_text(
        json.dumps(
            {
                "fps": fps,
                "obs": args.obs,
                "action": args.action,
                "env_state": args.env_state,
                "max_episodes": args.max_episodes,
                "episodes": provenance,
            },
            indent=2,
        )
    )
    print(
        f"wrote {len(episodes)} episodes to {args.out} "
        f"(fps={fps}, obs={args.obs}, action={args.action}, "
        f"env_state={args.env_state}"
        f"{'' if args.max_episodes is None else f', max_episodes={args.max_episodes}'})"
    )


if __name__ == "__main__":
    main()
