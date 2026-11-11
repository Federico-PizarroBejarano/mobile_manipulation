"""Standalone ACT inference worker for the ``mm-il`` conda env (Python >=3.10).

ROS Noetic nodes stay on system Python 3.8 and talk to this process over
newline-delimited JSON on stdin/stdout so they never import torch/lerobot
(or pinocchio/casadi built for 3.8) under the wrong interpreter.
"""

from __future__ import annotations

import argparse
import json
import sys


def _reply(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--nu", type=int, required=True)
    args = parser.parse_args(argv)

    try:
        import numpy as np
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
    except ImportError as exc:
        _reply({"ok": False, "error": f"worker import failed: {exc}"})
        return 1

    try:
        # LeRobot prints banners to stdout; keep the JSONL channel clean.
        import contextlib

        with contextlib.redirect_stdout(sys.stderr):
            policy = ACTPolicy.from_pretrained(args.checkpoint)
            policy.eval()
            preprocessor, postprocessor = make_pre_post_processors(
                policy.config, pretrained_path=args.checkpoint
            )
    except Exception as exc:  # noqa: BLE001 - surface load errors to the ROS node
        _reply({"ok": False, "error": f"failed to load checkpoint: {exc}"})
        return 1

    _reply({"ok": True, "ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _reply({"ok": False, "error": f"bad json: {exc}"})
            continue

        cmd = req.get("cmd")
        if cmd == "shutdown":
            _reply({"ok": True, "shutdown": True})
            return 0
        if cmd != "predict":
            _reply({"ok": False, "error": f"unknown cmd: {cmd!r}"})
            continue

        try:
            q = np.asarray(req["q"], dtype=np.float32).reshape(1, -1)
            if "env_state" in req:
                env = np.asarray(req["env_state"], dtype=np.float32).reshape(1, -1)
            else:
                # Tight smoke path: env_state duplicates q (width 9).
                env = q
            q_t = torch.as_tensor(q)
            env_t = torch.as_tensor(env)
            obs = {
                "observation.state": q_t,
                "observation.environment_state": env_t,
            }
            with torch.no_grad():
                obs = preprocessor(obs)
                chunk = policy.predict_action_chunk(obs)
                actions = [postprocessor(chunk[:, i]) for i in range(chunk.shape[1])]
            out = torch.cat(actions, dim=0).detach().cpu().numpy().astype(float)
            if out.ndim != 2 or out.shape[1] != int(args.nu):
                raise ValueError(f"ACT chunk must be (H, {args.nu}), got {out.shape}")
            _reply({"ok": True, "chunk": out.tolist()})
        except Exception as exc:  # noqa: BLE001 - keep worker alive across bad requests
            _reply({"ok": False, "error": str(exc)})

    return 0


if __name__ == "__main__":
    sys.exit(main())
