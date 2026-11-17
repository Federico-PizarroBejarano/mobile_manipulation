"""Bench ACT inference latency on the deploy path (ROS node → mm-il worker).

Reports end-to-end ``ActPolicySubprocess.predict_chunk`` wall time (JSON+IPC+torch),
which is what the hardware/control loop pays on replan ticks. Optionally also
times in-process torch-only predict under ``MM_IL_PYTHON``.

Example:
  export MM_IL_PYTHON=$HOME/miniconda3/envs/mm-il/bin/python
  python3 mm_run/src/mm_run/scripts/bench_act_infer.py \\
    --checkpoints \\
      mm_run/results/il/sweeps/horizon_ee_mpc3/runs/cs25_na25/train \\
      mm_run/results/il/sweeps/horizon_ee_mpc3/runs/cs50_na25/train \\
      mm_run/results/il/sweeps/horizon_ee_mpc3/runs/cs100_na25/train \\
    --iters 50 --warmup 10 --also-inprocess
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np

from mm_utils.act_eval import resolve_pretrained_model
from mm_utils.act_policy_runtime import ActPolicySubprocess, default_mm_il_python


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        required=True,
        help="Train output dirs (…/train) or pretrained_model dirs",
    )
    p.add_argument("--nu", type=int, default=6)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument(
        "--also-inprocess",
        action="store_true",
        help="Also time torch-only predict via MM_IL_PYTHON subprocess script",
    )
    p.add_argument(
        "--mm-il-python",
        default=None,
        help="mm-il interpreter (default: MM_IL_PYTHON / ~/miniconda3/envs/mm-il/bin/python)",
    )
    return p.parse_args(argv)


def _resolve(path: Path) -> Path:
    path = path.resolve()
    if (path / "config.json").is_file():
        return path
    return resolve_pretrained_model(path, "last")


def _ckpt_meta(pretrained: Path) -> dict:
    cfg = json.loads((pretrained / "config.json").read_text())
    return {
        "chunk_size": int(cfg.get("chunk_size") or 0),
        "n_action_steps": int(cfg.get("n_action_steps") or 0),
    }


def _stats_ms(samples_s: Sequence[float]) -> dict:
    xs = [1000.0 * s for s in samples_s]
    xs_sorted = sorted(xs)
    n = len(xs_sorted)

    def pct(p: float) -> float:
        if n == 1:
            return xs_sorted[0]
        i = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
        return xs_sorted[i]

    return {
        "n": n,
        "mean_ms": float(statistics.fmean(xs)),
        "std_ms": float(statistics.stdev(xs)) if n > 1 else 0.0,
        "p50_ms": float(pct(50)),
        "p95_ms": float(pct(95)),
        "min_ms": float(xs_sorted[0]),
        "max_ms": float(xs_sorted[-1]),
    }


def _bench_e2e(
    pretrained: Path, *, nu: int, iters: int, warmup: int, mm_il_python: str
) -> dict:
    policy = ActPolicySubprocess(str(pretrained), nu=nu, python_executable=mm_il_python)
    q = np.zeros(9, dtype=float)
    env = np.zeros(9, dtype=float)  # obs_duplicate-style
    try:
        for _ in range(int(warmup)):
            policy.predict_chunk(q, env_state=env)
        samples = []
        for _ in range(int(iters)):
            t0 = time.perf_counter()
            chunk = policy.predict_chunk(q, env_state=env)
            samples.append(time.perf_counter() - t0)
        meta = _ckpt_meta(pretrained)
        return {
            "path": "e2e_subprocess",
            "chunk_shape": list(chunk.shape),
            **meta,
            **_stats_ms(samples),
        }
    finally:
        policy.close()


_INPROCESS_CODE = r"""
import json, sys, time
import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

ckpt, nu, warmup, iters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
policy = ACTPolicy.from_pretrained(ckpt)
policy.eval()
preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=ckpt)
q = torch.zeros(1, 9)
env = torch.zeros(1, 9)
obs0 = {"observation.state": q, "observation.environment_state": env}

def once():
    with torch.no_grad():
        obs = preprocessor({k: v.clone() for k, v in obs0.items()})
        chunk = policy.predict_action_chunk(obs)
        actions = [postprocessor(chunk[:, i]) for i in range(chunk.shape[1])]
        out = torch.cat(actions, dim=0)
    return out

for _ in range(warmup):
    once()
if torch.cuda.is_available():
    torch.cuda.synchronize()
samples = []
for _ in range(iters):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = once()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    samples.append(time.perf_counter() - t0)
print(json.dumps({"samples_s": samples, "chunk_shape": list(out.shape), "cuda": bool(torch.cuda.is_available())}))
"""


def _bench_inprocess(
    pretrained: Path, *, nu: int, iters: int, warmup: int, mm_il_python: str
) -> dict:
    proc = subprocess.run(
        [
            mm_il_python,
            "-c",
            _INPROCESS_CODE,
            str(pretrained),
            str(int(nu)),
            str(int(warmup)),
            str(int(iters)),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    # Last JSON line (ignore any banners).
    line = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")][-1]
    payload = json.loads(line)
    meta = _ckpt_meta(pretrained)
    return {
        "path": "inprocess_torch",
        "chunk_shape": payload["chunk_shape"],
        "cuda": payload.get("cuda"),
        **meta,
        **_stats_ms(payload["samples_s"]),
    }


def _print_row(label: str, row: dict) -> None:
    print(
        f"{label:28s} path={row['path']:16s} "
        f"H={row.get('chunk_size')} na={row.get('n_action_steps')} "
        f"shape={row.get('chunk_shape')} "
        f"mean={row['mean_ms']:6.2f}ms  p50={row['p50_ms']:6.2f}  "
        f"p95={row['p95_ms']:6.2f}  min={row['min_ms']:6.2f}  max={row['max_ms']:6.2f}",
        flush=True,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    mm_il = args.mm_il_python or default_mm_il_python()
    if not Path(mm_il).is_file():
        raise SystemExit(f"mm-il python missing: {mm_il}")

    print(
        f"[bench] mm_il={mm_il} nu={args.nu} warmup={args.warmup} iters={args.iters}",
        flush=True,
    )
    print(
        "[bench] e2e = ActPolicySubprocess (same path as act_ee on hardware); "
        "50ms budget @ 20Hz",
        flush=True,
    )

    results: List[dict] = []
    for raw in args.checkpoints:
        pretrained = _resolve(Path(raw))
        label = Path(raw).name if Path(raw).name != "train" else Path(raw).parent.name
        if Path(raw).name == "train":
            label = Path(raw).parent.name
        print(f"\n=== {label}  {pretrained} ===", flush=True)
        e2e = _bench_e2e(
            pretrained,
            nu=args.nu,
            iters=args.iters,
            warmup=args.warmup,
            mm_il_python=mm_il,
        )
        e2e["label"] = label
        e2e["pretrained"] = str(pretrained)
        _print_row(label, e2e)
        results.append(e2e)
        # Rough hardware implication at 20 Hz.
        period_ms = 50.0
        print(
            f"{'':28s} → e2e uses {100.0 * e2e['mean_ms'] / period_ms:5.1f}% of a "
            f"20Hz tick; replan every n_action steps so ACT duty ≈ "
            f"{e2e['mean_ms'] / (e2e.get('n_action_steps') or 1):.2f}ms/tick amortized",
            flush=True,
        )
        if args.also_inprocess:
            ip = _bench_inprocess(
                pretrained,
                nu=args.nu,
                iters=args.iters,
                warmup=args.warmup,
                mm_il_python=mm_il,
            )
            ip["label"] = label
            ip["pretrained"] = str(pretrained)
            _print_row(label + " (torch)", ip)
            results.append(ip)

    out = Path("mm_run/results/il/sweeps/horizon_ee_mpc3/bench_act_infer.json")
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"results": results}, indent=2) + "\n")
        print(f"\n[bench] wrote {out.resolve()}", flush=True)
    except OSError:
        print("[bench] (could not write results json)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
