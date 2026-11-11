#!/usr/bin/env python3
"""Batch-convert teleop control/data.npz trials into a LeRobot dataset (ACT smoke).

Requires the IL env (see mm_run/requirements-il.txt):
  python export_teleop_to_lerobot.py --root /path/to/results/user --out /path/to/lerobot_ds --repo-id local/table_wipe_act_smoke

Run export and training in the IL env. Deploy keeps ROS on system Python and
runs ACT inference via ``mm_utils.act_policy_worker`` under that same IL env.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from mm_utils.il_episode import extract_episode, find_control_npzs

# observation.state = joint q (9). observation.environment_state = q again (9).
# LeRobot ACT 0.4 requires env_state for config validation; the smoke path
# duplicates q (tight baseline that beat the EE-pose A/B).
FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (9,),
        "names": None,
    },
    "observation.environment_state": {
        "dtype": "float32",
        "shape": (9,),
        "names": None,
    },
    "action": {
        "dtype": "float32",
        "shape": (9,),
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
    p.add_argument("--min-enabled-seconds", type=float, default=0.5)
    p.add_argument("--default-fps", type=float, default=50.0)
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

    episodes = []
    fps_votes = []
    for path in paths:
        try:
            ep = extract_episode(
                path,
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

    # Single fps for the dataset: median of per-episode fps
    fps = int(np.median(np.asarray(fps_votes, dtype=float)))
    # LeRobotDataset.create requires a non-existent root (exist_ok=False).
    if args.out.exists():
        raise SystemExit(
            f"output dataset path already exists: {args.out}\n"
            "Remove it or choose a new --out, then re-run."
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        features=FEATURES,
        root=str(args.out),
        robot_type="thing",
        use_videos=False,
    )

    provenance = []
    for path, ep in episodes:
        for i in range(ep["n_frames"]):
            dataset.add_frame(
                {
                    "observation.state": ep["q"][i],
                    "observation.environment_state": ep["q"][i],
                    "action": ep["u_cmd"][i],
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
                "env_state": "q_duplicate",
                "episodes": provenance,
            },
            indent=2,
        )
    )
    print(f"wrote {len(episodes)} episodes to {args.out} (fps={fps})")


if __name__ == "__main__":
    main()
