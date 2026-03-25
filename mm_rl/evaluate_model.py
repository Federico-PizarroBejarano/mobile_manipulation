"""Evaluate RL checkpoint and optionally compare to MPSF on identical goals.

```bash
cd /home/federico/catkin_ws/src/mobile_manipulation
python3 -m mm_rl.evaluate_model -c mm_rl/config/train_config.yaml \\
  --checkpoint checkpoints/final_model.pth --n-episodes 10 --seed 3

# RL vs MPSF, same goals (from --seed), RL runs full horizon (no early term):
python3 -m mm_rl.evaluate_model -c mm_rl/config/train_config.yaml \\
  --checkpoint checkpoints/final_model.pth --compare-mpsf --seed 3 --n-episodes 5

# Optional: use closed-loop planner for MPSF rollouts
python3 -m mm_rl.evaluate_model -c mm_rl/config/train_config.yaml \\
  --checkpoint checkpoints/final_model.pth --compare-mpsf --closed-loop-planner --seed 3 --n-episodes 5
```
"""

import argparse
import datetime
import pickle
import time
from pathlib import Path

import numpy as np

import mm_control.MPC as MPC
from mm_rl.env.ee_planner import EEPlanner
from mm_rl.env.simple_goal_env import SimpleGoalEnv
from mm_rl.evaluate_experiment import EpisodeTelemetry, print_report
from mm_rl.mpsf_helpers import (
    build_ik_params_from_config,
    ensure_controller_config,
    generate_goal,
    setup_mpsf_config,
    solve_ik,
)
from mm_rl.sac.sac import SAC
from mm_simulator import simulation
from mm_utils import math as mm_math
from mm_utils import parsing


def parse_args():
    """Parse command-line arguments for RL evaluation and optional MPSF comparison.

    Returns:
        argparse.Namespace: Parsed arguments (``config``, ``checkpoint``, ``n_episodes``,
        ``gui``, ``seed``, ``compare_mpsf``, ``use_ik_solver``, ``closed_loop_planner``).
    """
    parser = argparse.ArgumentParser(description="Evaluate RL and/or compare to MPSF")
    parser.add_argument("-c", "--config", required=True, help="Training config YAML")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="RL checkpoint (.pth)",
    )
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="RL / env seed")
    parser.add_argument(
        "--compare-mpsf",
        action="store_true",
        help="After RL, run MPSF on the same goals and print side-by-side summary",
    )
    parser.add_argument(
        "--use-ik-solver",
        action="store_true",
        help="MPSF: use Jacobian IK path like RL (MPC base vel + IK arm)",
    )
    parser.add_argument(
        "--closed-loop-planner",
        action="store_true",
        help="Use closed-loop EEPlanner for MPSF rollouts (default: open-loop)",
    )
    return parser.parse_args()


def sample_goals(config, n_episodes, seed):
    """Build a reproducible list of sampled goals for evaluation.

    Args:
        config (dict): Must contain ``goal.pos_range`` and ``goal.orn_range``.
        n_episodes (int): Number of goals to draw.
        seed (int): Seed for :class:`numpy.random.RandomState`.

    Returns:
        list: Length ``n_episodes``; each element is ``(goal_pos, goal_orn)`` as
        ``numpy`` arrays (position shape ``(3,)``, quaternion shape ``(4,)``).
    """
    g = config["goal"]
    rng = np.random.RandomState(seed)
    goals = []
    for _ in range(n_episodes):
        goal_pos, goal_orn = generate_goal(g["pos_range"], g["orn_range"], rng)
        goals.append((goal_pos.copy(), goal_orn.copy()))
    return goals


def run_mpsf_episode(
    mpsf_runtime,
    episode_num,
    goal_pos,
    goal_orn,
    max_episode_steps,
    ee_max_linear_vel,
    ee_max_angular_vel,
    use_ik_solver,
):
    """Roll out one MPSF episode using a prebuilt runtime (no sim/controller rebuild).

    Args:
        mpsf_runtime (dict): From :func:`create_mpsf_runtime` (``sim``, ``robot``,
            ``controller``, ``ik_params``, ``ctrl_period``, etc.).
        episode_num (int): Index used only for logging.
        goal_pos (ndarray): Goal position in world frame, shape ``(3,)``.
        goal_orn (ndarray): Goal orientation quaternion (xyzs), shape ``(4,)``.
        max_episode_steps (int): Maximum simulation steps before stopping.
        ee_max_linear_vel (float): EE linear-norm clamp when ``use_ik_solver`` is True.
        ee_max_angular_vel (float): EE angular-norm clamp when ``use_ik_solver`` is True.
        use_ik_solver (bool): If True, apply :func:`mm_rl.mpsf_helpers.solve_ik` with
            MPC base velocity; if False, command full ``v_bar`` from MPC.

    Returns:
        dict: Episode summary plus raw traces for offline analysis.
    """
    print(f"\nEpisode {episode_num}:")
    print(f"  Goal position: {goal_pos}")
    print(f"  Goal orientation: {goal_orn}")

    config = mpsf_runtime["config"]
    sim = mpsf_runtime["sim"]
    robot = mpsf_runtime["robot"]
    controller = mpsf_runtime["controller"]
    nu = mpsf_runtime["nu"]
    ik_params = mpsf_runtime["ik_params"]
    ctrl_period = mpsf_runtime["ctrl_period"]
    max_lin = mpsf_runtime["planner_max_linear_speed"]
    dt = mpsf_runtime["dt"]
    planner_mode = mpsf_runtime["planner_mode"]

    gcfg = config["goal"]
    success_pos_threshold = float(gcfg["success_pos_threshold"])
    success_orn_threshold = float(gcfg["success_orn_threshold"])

    robot.reset_joint_configuration(robot.home)
    ee_pos, ee_orn = robot.link_pose()
    ee_planner = EEPlanner(
        goal_pos,
        goal_orn,
        ee_pos,
        ee_orn,
        max_lin,
        dt,
        planner_mode=planner_mode,
    )
    controller.reset()

    start_time = time.time()
    t = 0.0
    last_controller_time = -ctrl_period
    v_bar = None

    episode_length = 0
    pos_err = 0.0
    orn_err = 0.0
    tel = EpisodeTelemetry(robot)
    for _step in range(max_episode_steps):
        robot_states = robot.joint_states(add_noise=False)
        if planner_mode == "closed_loop":
            ee_pos_now, ee_orn_now = robot.link_pose()
            desired_lin, desired_ang_w = ee_planner.step(ee_pos_now, ee_orn_now)
        else:
            _, ee_orn_now = robot.link_pose()
            desired_lin, desired_ang_w = ee_planner.step()
        clamp = (ee_max_linear_vel, ee_max_angular_vel) if use_ik_solver else None
        desired_ee_vel_world, desired_ee_vel_mpc = (
            mm_math.ee_twist_world_and_mpc_reference(
                desired_lin, desired_ang_w, ee_orn_now, clamp_limits=clamp
            )
        )

        if t - last_controller_time >= ctrl_period:
            references = {
                "base_pose": None,
                "base_velocity": None,
                "ee_pose": None,
                "ee_velocity": None,
                "desired_velocity": {"ee_velocity": desired_ee_vel_mpc},
            }
            try:
                v_bar, _ = controller.control(t, robot_states, references)
                last_controller_time = t
            except Exception:
                v_bar = None

        if use_ik_solver and v_bar is not None:
            mpc_base = v_bar[1, :3]
            u = solve_ik(robot, desired_ee_vel_world, mpc_base, ik_params)
        elif v_bar is not None:
            u = v_bar[1, :]
        else:
            u = np.zeros(nu)

        robot.command_velocity(u)
        t, _ = sim.step(t)
        tel.accumulate_step(robot, goal_pos=goal_pos, goal_orn=goal_orn)

        episode_length = _step + 1
        ee_pos_f, ee_orn_f = robot.link_pose()
        pos_err = float(np.linalg.norm(ee_pos_f - goal_pos))
        orn_err = float(mm_math.quat_orientation_error(ee_orn_f, goal_orn))
        if pos_err <= success_pos_threshold and orn_err <= success_orn_threshold:
            break

    success = pos_err <= success_pos_threshold and orn_err <= success_orn_threshold

    elapsed_time = time.time() - start_time
    print("  Episode completed:")
    print(f"    Length: {episode_length} steps ({elapsed_time:.2f} seconds)")
    print(f"    Final position error: {pos_err:.3f} m")
    print(f"    Final orientation error: {orn_err:.3f}")
    print(f"    Success: {'Yes' if success else 'No'}")
    print(
        f"    Effort [base/arm]: {tel.base_effort:.3f} / {tel.arm_effort:.3f} "
        f" Smoothness [base/arm]: {tel.base_smoothness:.3f} / {tel.arm_smoothness:.3f}"
    )
    print(f"    Base path length: {tel.base_path_length:.3f} m")

    return {
        "episode_num": episode_num,
        "pos_error": pos_err,
        "orn_error": orn_err,
        "success": success,
        "length": episode_length,
        "elapsed_time": elapsed_time,
        **tel.as_dict(),
    }


def create_mpsf_runtime(config):
    """Instantiate Bullet sim, MPC controller, and IK metadata for MPSF rollouts.

    Expects ``ensure_controller_config`` and (for MPSF tests) ``setup_mpsf_config``
    to have been applied to ``config`` beforehand.

    Args:
        config (dict): Full merged YAML including ``controller``, ``simulation``,
            ``planner``, and ``planner_mode`` (optional).

    Returns:
        dict: Keys ``config``, ``sim``, ``robot``, ``controller``, ``nu``, ``ik_params``,
            ``ctrl_period``, ``planner_max_linear_speed``, ``dt``, ``planner_mode``.
    """
    ctrl_config = config["controller"]
    sim_config = config["simulation"]
    nu = sim_config["robot"]["dims"]["v"]
    nq = sim_config["robot"]["dims"]["q"]
    ik_params = build_ik_params_from_config(config, nu, nq)

    timestamp = datetime.datetime.now()
    controller = MPC.MPC(ctrl_config)
    sim = simulation.BulletSimulation(sim_config, timestamp, cli_args=None)
    return {
        "config": config,
        "sim": sim,
        "robot": sim.robot,
        "controller": controller,
        "nu": nu,
        "ik_params": ik_params,
        "ctrl_period": 1.0 / ctrl_config.get("ctrl_rate", 10.0),
        "planner_max_linear_speed": float(config["planner"]["max_linear_speed"]),
        "dt": float(sim_config["timestep"]),
        "planner_mode": config.get("planner_mode"),
    }


def evaluate_episode(env, agent, episode_num, reset_options=None):
    """Run one RL episode with deterministic actions and print per-episode progress.

    Args:
        env (SimpleGoalEnv): Initialized environment.
        agent (SAC): Loaded agent; ``select_action(..., deterministic=True)`` is used.
        episode_num (int): Label for logs only.
        reset_options (dict, optional): Passed to ``env.reset(options=...)`` (e.g.
            ``goal_pos``, ``goal_orn``, ``disable_early_termination``).

    Returns:
        dict: Episode summary plus raw traces for offline analysis.
    """
    reset_options = reset_options or {}
    obs, info = env.reset(options=reset_options)
    episode_reward = 0
    episode_length = 0
    done = False
    tel = EpisodeTelemetry(env.sim.robot)

    print(f"\nEpisode {episode_num}:")
    print(f"  Goal position: {info['goal_pos']}")
    print(f"  Goal orientation: {info['goal_orn']}")

    start_time = time.time()

    while not done:
        action = agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_length += 1
        done = terminated or truncated
        tel.accumulate_step(
            env.sim.robot,
            goal_pos=env.goal_pos,
            goal_orn=env.goal_orn,
            reward=reward,
        )

        if episode_length % 100 == 0:
            ee_pos, ee_orn = env.sim.robot.link_pose()
            pos_error = np.linalg.norm(ee_pos - env.goal_pos)
            orn_error = mm_math.quat_orientation_error(ee_orn, env.goal_orn)
            print(
                f"  Step {episode_length}: Reward={reward:.3f}, "
                f"Pos error={pos_error:.3f}, Orn error={orn_error:.3f}"
            )

    elapsed_time = time.time() - start_time
    ee_pos, ee_orn = env.sim.robot.link_pose()
    pos_error = np.linalg.norm(ee_pos - env.goal_pos)
    orn_error = mm_math.quat_orientation_error(ee_orn, env.goal_orn)
    success = (
        pos_error <= env.success_pos_threshold
        and orn_error <= env.success_orn_threshold
    )

    print("  Episode completed:")
    print(f"    Length: {episode_length} steps ({elapsed_time:.2f} seconds)")
    print(f"    Total reward: {episode_reward:.3f}")
    print(f"    Final position error: {pos_error:.3f} m")
    print(f"    Final orientation error: {orn_error:.3f}")
    print(f"    Success: {'Yes' if success else 'No'}")
    print(
        f"    Effort [base/arm]: {tel.base_effort:.3f} / {tel.arm_effort:.3f} "
        f" Smoothness [base/arm]: {tel.base_smoothness:.3f} / {tel.arm_smoothness:.3f}"
    )
    print(f"    Base path length: {tel.base_path_length:.3f} m")

    return {
        "episode_num": episode_num,
        "reward": episode_reward,
        "length": episode_length,
        "success": success,
        "pos_error": pos_error,
        "orn_error": orn_error,
        "elapsed_time": elapsed_time,
        **tel.as_dict(),
    }


def main():
    """Load config/checkpoint, run episodes, and dump full results to pickle."""
    args = parse_args()
    config = parsing.load_config(args.config)

    if args.gui:
        config["simulation"]["gui"] = True
        print("GUI visualization enabled")

    goal_seed = args.seed if args.seed is not None else 0
    run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("./logs")
    output_dir.mkdir(parents=True, exist_ok=True)

    max_steps = int(config["simulation"]["max_episode_steps"])
    # Evaluation always runs full horizon; no early termination.
    rl_reset_extras = {"disable_early_termination": True}

    ensure_controller_config(config)
    config["planner_mode"] = "closed_loop" if args.closed_loop_planner else "open_loop"
    goals = sample_goals(config, args.n_episodes, goal_seed)

    if args.compare_mpsf:
        setup_mpsf_config(config["controller"])
        print(f"Compare mode: {args.n_episodes} goals from goal_seed={goal_seed}")

    print("Initializing RL environment...")
    env = SimpleGoalEnv(config)
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    state_dim = obs_shape[0]
    action_dim = action_shape[0]
    rl_config = config.get("rl")
    sac_config = rl_config.get("sac")
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        action_range=(-1.0, 1.0),
        lr=sac_config.get("lr"),
        gamma=sac_config.get("gamma"),
        tau=sac_config.get("tau"),
        alpha=sac_config.get("alpha"),
        auto_alpha=sac_config.get("auto_alpha"),
        hidden_layers=sac_config["hidden_layers"],
        buffer_size=sac_config.get("buffer_size", 100000),
        device="cpu",
    )
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"Loading checkpoint: {checkpoint_path}")
    agent.load(str(checkpoint_path))

    if args.seed is not None:
        env.reset(seed=args.seed)
        print(f"Env seed: {args.seed}")

    print(f"\n{'='*80}\nRL evaluation ({args.n_episodes} episodes)\n{'='*80}")
    all_rl = []
    for ep in range(1, args.n_episodes + 1):
        ro = {**rl_reset_extras}
        gp, go = goals[ep - 1]
        ro["goal_pos"] = gp
        ro["goal_orn"] = go
        all_rl.append(
            evaluate_episode(
                env,
                agent,
                ep,
                reset_options=ro if ro else None,
            )
        )
        if args.gui:
            time.sleep(0.5)

    env.close()

    mpsf_results = []
    if args.compare_mpsf:
        print(
            f"\n{'='*80}\nMPSF rollouts (same goals, early exit on success, max {max_steps} steps)\n{'='*80}"
        )
        import pybullet as pyb

        robot_cfg = config.get("robot", {})
        ee_max_linear_vel = float(robot_cfg.get("ee_linear_vel_limit", 0.5))
        ee_max_angular_vel = float(robot_cfg.get("ee_angular_vel_limit", 0.75))
        mpsf_runtime = create_mpsf_runtime(config)
        try:
            for i, (gp, go) in enumerate(goals, 1):
                mpsf_results.append(
                    run_mpsf_episode(
                        mpsf_runtime,
                        i,
                        gp,
                        go,
                        max_steps,
                        ee_max_linear_vel=ee_max_linear_vel,
                        ee_max_angular_vel=ee_max_angular_vel,
                        use_ik_solver=args.use_ik_solver,
                    )
                )
        finally:
            pyb.disconnect()

    payload = {
        "meta": {
            "timestamp": run_stamp,
            "config_path": str(args.config),
            "checkpoint": str(checkpoint_path),
            "n_episodes": int(args.n_episodes),
            "seed": args.seed,
            "goal_seed": int(goal_seed),
            "compare_mpsf": bool(args.compare_mpsf),
            "use_ik_solver": bool(args.use_ik_solver),
            "closed_loop_planner": bool(args.closed_loop_planner),
            "planner_mode": config.get("planner_mode"),
            "sim_timestep": float(config["simulation"]["timestep"]),
        },
        "goals": [
            {
                "goal_pos": np.asarray(gp, dtype=np.float64),
                "goal_orn": np.asarray(go, dtype=np.float64),
            }
            for gp, go in goals
        ],
        "rl_episodes": all_rl,
        "mpsf_episodes": mpsf_results,
    }
    pkl_file = output_dir / f"evaluation_data_{run_stamp}.pkl"
    with open(pkl_file, "wb") as f:
        pickle.dump(payload, f)
    print_report(payload, pkl_path=str(pkl_file))
    print(f"\nSaved evaluation data to: {pkl_file}")
    print("Use: python3 -m mm_rl.evaluate_experiment --pkl <path>")


if __name__ == "__main__":
    raise SystemExit(main())
