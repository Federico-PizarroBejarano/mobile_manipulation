#!/usr/bin/env python3
"""Parse LeRobot train console logs into a local metrics table.

Extracts lines like:
  step:5K smpl:40K ep:168 epch:33.53 loss:0.592 grdn:18.969 lr:1.0e-05 ...

Writes:
  metrics.jsonl  — one JSON object per log row
  metrics.csv    — same columns for quick plotting
  metrics_summary.txt — short plateau / last-loss notes
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# Prefer tqdm's exact step (N/TOTAL) when present; LeRobot's step:1K is ambiguous.
_TQDM_STEP_RE = re.compile(r"\|\s*(?P<tqdm_step>\d+)/(?P<tqdm_total>\d+)\s*\[")
# LeRobot prints abbreviated ints (5K) and floats; accept both.
_STEP_RE = re.compile(
    r"\bstep:(?P<step>\d+(?:\.\d+)?K?)\s+"
    r"smpl:(?P<smpl>\d+(?:\.\d+)?K?)\s+"
    r"ep:(?P<ep>\d+(?:\.\d+)?K?)\s+"
    r"epch:(?P<epch>[\d.]+)\s+"
    r"loss:(?P<loss>[\d.eE+-]+)\s+"
    r"grdn:(?P<grdn>[\d.eE+-]+)\s+"
    r"lr:(?P<lr>[\d.eE+-]+)"
)


def _parse_count(token: str) -> float:
    token = token.strip()
    if token.endswith(("K", "k")):
        return float(token[:-1]) * 1_000.0
    if token.endswith(("M", "m")):
        return float(token[:-1]) * 1_000_000.0
    return float(token)


def parse_log(text: str) -> list[dict]:
    rows: list[dict] = []
    for line in text.splitlines():
        m = _STEP_RE.search(line)
        if not m:
            continue
        step = int(_parse_count(m.group("step")))
        tq = _TQDM_STEP_RE.search(line)
        if tq:
            step = int(tq.group("tqdm_step"))
        rows.append(
            {
                "step": step,
                "samples": int(_parse_count(m.group("smpl"))),
                "episodes": int(_parse_count(m.group("ep"))),
                "epoch": float(m.group("epch")),
                "loss": float(m.group("loss")),
                "grad_norm": float(m.group("grdn")),
                "lr": float(m.group("lr")),
            }
        )
    return rows


def summarize(rows: list[dict]) -> str:
    if not rows:
        return "No training metric rows found.\n"
    first, last = rows[0], rows[-1]
    # Compare last 25% of points vs previous 25% for plateau signal.
    n = len(rows)
    q = max(1, n // 4)
    early_tail = rows[-(2 * q) : -q] if n >= 2 * q else rows[: max(1, n // 2)]
    late = rows[-q:]
    early_loss = sum(r["loss"] for r in early_tail) / len(early_tail)
    late_loss = sum(r["loss"] for r in late) / len(late)
    rel = (early_loss - late_loss) / early_loss if early_loss > 0 else 0.0
    lines = [
        f"rows: {n}",
        f"step range: {first['step']} -> {last['step']}",
        f"loss: {first['loss']:.4f} -> {last['loss']:.4f}",
        f"grad_norm: {first['grad_norm']:.3f} -> {last['grad_norm']:.3f}",
        f"final epoch (dataset passes): {last['epoch']:.2f}",
        f"last-quarter mean loss: {late_loss:.4f}",
        f"prev-quarter mean loss: {early_loss:.4f}",
        f"relative drop last vs prev quarter: {100.0 * rel:.2f}%",
    ]
    if rel < 0.05:
        lines.append(
            "note: train loss looks mostly flat late in the run; "
            "extra steps may mainly overfit (confirm with PyBullet eval)."
        )
    elif last["epoch"] > 50:
        lines.append(
            "note: many dataset epochs; watch for overfitting even if loss still drops."
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log", type=Path, help="Console log file (tee of lerobot-train)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for metrics.* (default: same dir as the log)",
    )
    args = p.parse_args(argv)
    text = args.log.read_text(errors="replace")
    rows = parse_log(text)
    out_dir = args.out_dir or args.log.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "metrics.jsonl"
    with jsonl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    csv_path = out_dir / "metrics.csv"
    fields = ["step", "samples", "episodes", "epoch", "loss", "grad_norm", "lr"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = summarize(rows)
    (out_dir / "metrics_summary.txt").write_text(summary)
    print(summary, end="")
    print(f"wrote {jsonl_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
