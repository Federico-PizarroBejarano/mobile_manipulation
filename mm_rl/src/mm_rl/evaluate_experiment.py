"""Analyze experiment pickle outputs and provide shared episode telemetry helpers."""

import argparse
import pickle

import numpy as np

from mm_utils import math as mm_math
from mm_utils.metrics import compute_jerkiness


class EpisodeTelemetry:
    """Accumulate per-step traces and scalar metrics for one rollout."""

    def __init__(self, robot):
        """Initialize traces from current base pose.

        Args:
            robot: Robot object with ``joint_states()`` and ``link_pose(link_idx=-1)``.
        """
        base_pos, _ = robot.link_pose(link_idx=-1)
        self._base_pos_prev = np.asarray(base_pos, dtype=np.float64).copy()
        self._prev_dq = None
        self.base_effort = 0.0
        self.arm_effort = 0.0
        self.base_smoothness = 0.0
        self.arm_smoothness = 0.0
        self.base_path_length = 0.0
        self.base_cmd_trace = []
        self.arm_cmd_trace = []
        self.full_cmd_trace = []
        self.base_pos_trace = [self._base_pos_prev.copy()]
        self.pos_error_trace = []
        self.orn_error_trace = []
        self.reward_trace = []

    def accumulate_step(self, robot, *, goal_pos=None, goal_orn=None, reward=None):
        """Record command/pose samples and optional goal errors and reward.

        Args:
            robot: Same robot interface used in ``__init__``.
            goal_pos (ndarray, optional): Goal position for error traces.
            goal_orn (ndarray, optional): Goal quaternion (xyzs) for error traces.
            reward (float, optional): RL reward for this step.
        """
        _, dq = robot.joint_states(add_noise=False)
        dq = np.asarray(dq, dtype=np.float64)
        base_u, arm_u = dq[:3], dq[3:]
        self.base_cmd_trace.append(base_u.copy())
        self.arm_cmd_trace.append(arm_u.copy())
        self.full_cmd_trace.append(dq.copy())
        self.base_effort += float(np.dot(base_u, base_u))
        self.arm_effort += float(np.dot(arm_u, arm_u))
        if self._prev_dq is not None:
            d_base = base_u - self._prev_dq[:3]
            d_arm = arm_u - self._prev_dq[3:]
            self.base_smoothness += float(np.dot(d_base, d_base))
            self.arm_smoothness += float(np.dot(d_arm, d_arm))
        self._prev_dq = dq.copy()

        base_now, _ = robot.link_pose(link_idx=-1)
        b = np.asarray(base_now, dtype=np.float64)
        self.base_path_length += float(np.linalg.norm(b - self._base_pos_prev))
        self._base_pos_prev = b.copy()
        self.base_pos_trace.append(b.copy())

        if goal_pos is not None and goal_orn is not None:
            ee_pos, ee_orn = robot.link_pose()
            self.pos_error_trace.append(
                float(np.linalg.norm(np.asarray(ee_pos) - np.asarray(goal_pos)))
            )
            self.orn_error_trace.append(
                float(mm_math.quat_orientation_error(ee_orn, goal_orn))
            )
        if reward is not None:
            self.reward_trace.append(float(reward))

    def as_dict(self):
        """Serialize all telemetry to pickle-friendly arrays.

        Returns:
            dict: Scalar metrics and trajectory arrays.
        """
        out = {
            "base_effort": self.base_effort,
            "arm_effort": self.arm_effort,
            "base_smoothness": self.base_smoothness,
            "arm_smoothness": self.arm_smoothness,
            "base_path_length": self.base_path_length,
            "base_cmd_trace": np.asarray(self.base_cmd_trace, dtype=np.float64),
            "arm_cmd_trace": np.asarray(self.arm_cmd_trace, dtype=np.float64),
            "full_cmd_trace": np.asarray(self.full_cmd_trace, dtype=np.float64),
            "base_pos_trace": np.asarray(self.base_pos_trace, dtype=np.float64),
            "pos_error_trace": np.asarray(self.pos_error_trace, dtype=np.float64),
            "orn_error_trace": np.asarray(self.orn_error_trace, dtype=np.float64),
        }
        if self.reward_trace:
            out["reward_trace"] = np.asarray(self.reward_trace, dtype=np.float64)
        return out


def _stats(values):
    """Compute mean/std/min/max over numeric values."""
    a = np.asarray(values, dtype=np.float64)
    return float(np.mean(a)), float(np.std(a)), float(np.min(a)), float(np.max(a))


def _fmt_stats(values, suffix=""):
    """Format mean/std/min/max into one readable line."""
    m, s, lo, hi = _stats(values)
    return f"{m:.3f} ± {s:.3f}{suffix}  (min {lo:.3f}{suffix}, max {hi:.3f}{suffix})"


def _episode_jerk(episodes, key, dt):
    """Compute jerkiness for each episode trace."""
    jerks = []
    for ep in episodes:
        trace = np.asarray(ep.get(key, []), dtype=np.float64)
        jerks.append(float(compute_jerkiness(trace, dt)))
    return jerks


def _print_block(label, episodes, dt):
    """Print aggregate metrics for one episode set."""
    if not episodes:
        print(f"{label}: no episodes.")
        return
    succ = [bool(ep["success"]) for ep in episodes]
    print(label)
    print(f"  Episodes: {len(episodes)}")
    print(f"  Success: {sum(succ)}/{len(succ)} ({100.0 * np.mean(succ):.1f}%)")
    print(
        f"  Final position error (m): {_fmt_stats([ep['pos_error'] for ep in episodes])}"
    )
    print(
        f"  Final orientation error: {_fmt_stats([ep['orn_error'] for ep in episodes])}"
    )
    if "reward" in episodes[0]:
        print(f"  Episode reward: {_fmt_stats([ep['reward'] for ep in episodes])}")
    print(f"  Episode length (steps): {_fmt_stats([ep['length'] for ep in episodes])}")
    print(f"  Wall time (s): {_fmt_stats([ep['elapsed_time'] for ep in episodes])}")
    print(
        f"  Base control effort: {_fmt_stats([ep['base_effort'] for ep in episodes])}"
    )
    print(f"  Arm control effort: {_fmt_stats([ep['arm_effort'] for ep in episodes])}")
    print(
        f"  Base control smoothness: {_fmt_stats([ep['base_smoothness'] for ep in episodes])}"
    )
    print(
        f"  Arm control smoothness: {_fmt_stats([ep['arm_smoothness'] for ep in episodes])}"
    )
    print(
        f"  Base path length (m): {_fmt_stats([ep['base_path_length'] for ep in episodes])}"
    )
    print(
        "  Base jerkiness (from mm_utils.metrics): "
        f"{_fmt_stats(_episode_jerk(episodes, 'base_cmd_trace', dt))}"
    )
    print(
        "  Arm jerkiness (from mm_utils.metrics): "
        f"{_fmt_stats(_episode_jerk(episodes, 'arm_cmd_trace', dt))}"
    )
    print(
        "  Full-command jerkiness (from mm_utils.metrics): "
        f"{_fmt_stats(_episode_jerk(episodes, 'full_cmd_trace', dt))}"
    )


def _print_episode_table(data):
    """Print per-episode rows, paired RL vs MPSF when both exist.

    Args:
        data (dict): Pickle payload with ``rl_episodes`` and ``mpsf_episodes`` lists.
    """
    rl = data.get("rl_episodes", [])
    mpsf = data.get("mpsf_episodes", [])
    if rl and mpsf:
        n = min(len(rl), len(mpsf))
        print("-" * 80)
        print("Per-episode comparison")
        print("-" * 80)
        print(
            f"{'Ep':>4} {'RL ok':>7} {'RL pos':>8} {'RL orn':>8} {'RL armE':>9} {'RL armS':>9} | "
            f"{'MPSF ok':>8} {'MPSF pos':>9} {'MPSF orn':>9} {'MPSF armE':>11} {'MPSF armS':>11}"
        )
        for i in range(n):
            r, m = rl[i], mpsf[i]
            print(
                f"{i+1:4d} {str(r['success']):>7} {r['pos_error']:8.3f} {r['orn_error']:8.3f} "
                f"{r['arm_effort']:9.3f} {r['arm_smoothness']:9.3f} | "
                f"{str(m['success']):>8} {m['pos_error']:9.3f} {m['orn_error']:9.3f} "
                f"{m['arm_effort']:11.3f} {m['arm_smoothness']:11.3f}"
            )
        return
    for label, episodes in (("RL", rl), ("MPSF", mpsf)):
        if not episodes:
            continue
        print("-" * 80)
        print(f"Per-episode results — {label}")
        print("-" * 80)
        print(
            f"{'Ep':>4} {'ok':>7} {'pos':>8} {'orn':>8} {'len':>6} "
            f"{'baseE':>9} {'armE':>9} {'baseS':>9} {'armS':>9} {'basePath':>9}"
        )
        for i, ep in enumerate(episodes, 1):
            print(
                f"{i:4d} {str(ep['success']):>7} {ep['pos_error']:8.3f} {ep['orn_error']:8.3f} "
                f"{ep['length']:6d} {ep['base_effort']:9.3f} {ep['arm_effort']:9.3f} "
                f"{ep['base_smoothness']:9.3f} {ep['arm_smoothness']:9.3f} {ep['base_path_length']:9.3f}"
            )


def print_report(data, pkl_path="<in-memory>"):
    """Print full aggregate and per-episode experiment report.

    Args:
        data (dict): Evaluation payload as produced by ``mm_rl.evaluate_model``.
        pkl_path (str): Source path label shown in the report header.
    """
    meta = data.get("meta", {})
    dt = float(meta.get("sim_timestep", 0.01))
    print("=" * 80)
    print("EVALUATION METRICS REPORT")
    print("=" * 80)
    print(f"pickle: {pkl_path}")
    print(f"episodes: {meta.get('n_episodes')}, goal_seed: {meta.get('goal_seed')}")
    print(
        "compare_mpsf: "
        f"{meta.get('compare_mpsf')} | use_ik_solver: {meta.get('use_ik_solver')} | "
        f"planner_mode: {meta.get('planner_mode')}"
    )
    print(f"sim_timestep: {dt}")
    print("-" * 80)
    _print_block("RL", data.get("rl_episodes", []), dt)
    print("-" * 80)
    _print_block("MPSF", data.get("mpsf_episodes", []), dt)
    _print_episode_table(data)
    print("=" * 80)


def main():
    """Load an evaluation pickle and print aggregated RL/MPSF metrics."""
    parser = argparse.ArgumentParser(description="Print metrics from evaluation pickle")
    parser.add_argument("--pkl", required=True, help="Path to evaluation_data_*.pkl")
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    print_report(data, pkl_path=args.pkl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
