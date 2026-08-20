"""Test EE planner with MPSF controller.

This test uses the EE planner as a "faux teleoperator" that sends EE velocity
commands to MPSF. MPSF controls the base while tracking the planner's EE velocities.
If MPSF can track the planner's trajectory without large deviations, then the
planner works correctly.

Test Setup:
- EE planner generates desired EE velocities (as teleoperator)
- MPSF tracks EE velocities (mpsf_ee_mask = [1,1,1,1,1,1])
- MPC controls base freely (base_mask = [0,0,0])
- Measure tracking error: deviation between planner's desired pose and actual EE pose
"""

import argparse
import datetime
from pathlib import Path

import numpy as np

import mm_control.MPC as MPC
from mm_rl.env.ee_planner import EEPlanner
from mm_rl.mpsf_helpers import (
    build_ik_params_from_config,
    ensure_controller_config,
    generate_goal,
    setup_mpsf_config,
    solve_ik,
)
from mm_simulator import simulation
from mm_utils import math as mm_math
from mm_utils import parsing


def _resolve_success_thresholds(goal_cfg, success_pos_threshold, success_orn_threshold):
    """Fill missing success thresholds from the goal section of the config.

    Args:
        goal_cfg (dict): Typically ``config["goal"]``.
        success_pos_threshold (float, optional): If ``None``, use ``goal_cfg`` default.
        success_orn_threshold (float, optional): If ``None``, use ``goal_cfg`` default.

    Returns:
        tuple: ``(success_pos_threshold, success_orn_threshold)`` as floats.
    """
    if success_pos_threshold is None:
        success_pos_threshold = float(goal_cfg.get("success_pos_threshold", 0.1))
    if success_orn_threshold is None:
        success_orn_threshold = float(goal_cfg.get("success_orn_threshold", 0.05))
    return success_pos_threshold, success_orn_threshold


def _generate_feasible_goal(pos_range, orn_range, np_random, ee_pos, max_goal_dist):
    """Sample a goal via :func:`generate_goal`, retrying if it is too far from ``ee_pos``.

    Args:
        pos_range: Passed to :func:`generate_goal`.
        orn_range: Passed to :func:`generate_goal`.
        np_random (numpy.random.RandomState): RNG instance.
        ee_pos (ndarray): Current EE position for distance check, shape ``(3,)``.
        max_goal_dist (float): Desired maximum ``‖goal_pos - ee_pos‖`` (best effort).

    Returns:
        tuple: ``(goal_pos, goal_orn, goal_distance)`` with ``goal_distance`` the final norm.
    """
    goal_pos, goal_orn = generate_goal(pos_range, orn_range, np_random)
    goal_distance = np.linalg.norm(goal_pos - ee_pos)
    for _ in range(20):
        if goal_distance <= max_goal_dist:
            break
        goal_pos, goal_orn = generate_goal(pos_range, orn_range, np_random)
        goal_distance = np.linalg.norm(goal_pos - ee_pos)
    return goal_pos, goal_orn, goal_distance


def test_ee_planner_with_mpsf(
    config_path,
    n_episodes=10,
    max_tracking_error_pos=0.1,
    max_tracking_error_orn=0.2,
    success_pos_threshold=None,
    success_orn_threshold=None,
    gui=False,
    use_ik_solver=False,
    planner_mode="open_loop",
):
    """Test EE planner with MPSF controller.

    Args:
        config_path: Path to configuration file
        n_episodes: Number of test episodes
        max_tracking_error_pos: Maximum allowed position tracking error (m)
        max_tracking_error_orn: Maximum allowed orientation tracking error
        success_pos_threshold: Position threshold for goal success (m)
        success_orn_threshold: Orientation threshold for goal success
        gui: Whether to show GUI
        use_ik_solver: If True, extract base velocities from MPC and pass through IK solver
                      (fair comparison to RL training). If False, use MPC's full output.
        planner_mode: EEPlanner mode: "open_loop" or "closed_loop".

    Returns:
        dict: Keys ``passed``, ``n_episodes``, ``tracking_good_count``, ``goal_reached_count``,
        ``mean_tracking_error_pos``, ``mean_tracking_error_orn``, ``results`` (per-episode list).
    """

    # --- Config ---
    config = parsing.load_config(str(config_path))
    ensure_controller_config(config)
    if gui:
        config["simulation"]["gui"] = True

    ctrl_config = config["controller"]
    setup_mpsf_config(ctrl_config)
    print("MPSF Configuration:")
    print(f"  mpsf_ee_mask: {ctrl_config['mpsf_params']['mpsf_ee_mask']}")
    print(
        f"  mpsf_weight_multiplier: {ctrl_config['mpsf_params']['mpsf_weight_multiplier']}"
    )
    print(f"  EEVel Qk: {ctrl_config['cost_params']['EEVel']['Qk']}")
    print(f"  planner_mode: {planner_mode}")

    sim_config = config["simulation"]

    # Initialize simulation
    timestamp = datetime.datetime.now()
    sim = simulation.BulletSimulation(sim_config, timestamp, cli_args=None)
    robot = sim.robot

    # --- Simulation & controller ---
    controller = MPC.MPC(ctrl_config)
    nu = sim_config["robot"]["dims"]["v"]
    nq = sim_config["robot"]["dims"]["q"]
    ctrl_period = 1.0 / ctrl_config.get("ctrl_rate", 10.0)

    ik_params = None
    if use_ik_solver:
        ik_params = build_ik_params_from_config(config, nu, nq)

    # --- Episode parameters (from config) ---
    goal_cfg = config.get("goal", {})
    pos_range = goal_cfg.get("pos_range", [[-2.0, -2.0, 0.6], [2.0, 2.0, 1.4]])
    orn_range = goal_cfg.get("orn_range", 0)
    max_goal_dist = goal_cfg.get("max_goal_distance", 3.0)
    termination_cfg = config.get("termination", {})
    slack_pos_threshold = float(termination_cfg.get("slack_pos_threshold", 0.1))
    slack_rot_threshold = float(termination_cfg.get("slack_rot_threshold", 0.05))
    ik_fail_thresh = int(termination_cfg.get("ik_fail_thresh", 20))
    success_pos_threshold, success_orn_threshold = _resolve_success_thresholds(
        goal_cfg, success_pos_threshold, success_orn_threshold
    )

    planner_cfg = config.get("planner", {})
    max_lin = float(planner_cfg["max_linear_speed"])
    robot_cfg = config.get("robot", {})
    ee_max_linear_vel = float(robot_cfg.get("ee_linear_vel_limit", 0.5))
    ee_max_angular_vel = float(robot_cfg.get("ee_angular_vel_limit", 0.75))

    np_random = np.random.RandomState(42)

    # --- Episodes ---
    all_results = []
    tracking_errors_pos = []
    tracking_errors_orn = []
    goal_reached_count = 0
    tracking_good_count = 0

    for episode in range(n_episodes):
        print(f"\n--- Episode {episode + 1}/{n_episodes} ---")

        # Reset simulation
        robot.reset_joint_configuration(robot.home)

        # Get initial EE pose for goal generation
        ee_pos, ee_orn = robot.link_pose()

        # Generate feasible goal
        goal_pos, goal_orn, goal_distance = _generate_feasible_goal(
            pos_range, orn_range, np_random, ee_pos, max_goal_dist
        )
        if goal_distance > max_goal_dist:
            print(
                f"  Warning: Goal {goal_distance:.2f}m away (max {max_goal_dist:.2f}m)"
            )

        print(f"Goal position: {goal_pos}")
        print(f"Goal orientation: {goal_orn}")

        ee_planner = EEPlanner(
            goal_pos,
            goal_orn,
            ee_pos,
            ee_orn,
            max_linear_speed=max_lin,
            dt=sim.timestep,
            planner_mode=planner_mode,
        )

        # Reset controller state for new episode
        controller.reset()

        print(
            f"  max_linear_speed: {max_lin:.3f} m/s, Goal distance: {goal_distance:.2f}m"
        )

        # Simulation loop
        t = 0.0
        max_duration = 30.0  # Maximum episode duration
        last_controller_time = -ctrl_period
        u = np.zeros(nu)
        v_bar = None

        episode_tracking_errors_pos = []
        episode_tracking_errors_orn = []
        max_tracking_error_pos_episode = 0.0
        max_tracking_error_orn_episode = 0.0
        nr_kin_failures = 0

        while t <= max_duration:
            robot_states = robot.joint_states(add_noise=False)

            # Desired EE vel from planner (updated every step)
            if planner_mode == "closed_loop":
                ee_pos_now, ee_orn_now = robot.link_pose()
                desired_lin_vel, desired_ang_w = ee_planner.step(ee_pos_now, ee_orn_now)
            else:
                _, ee_orn_now = robot.link_pose()
                desired_lin_vel, desired_ang_w = ee_planner.step()
            clamp = (ee_max_linear_vel, ee_max_angular_vel) if use_ik_solver else None
            desired_ee_vel_world, desired_ee_vel_mpc = (
                mm_math.ee_twist_world_and_mpc_reference(
                    desired_lin_vel,
                    desired_ang_w,
                    ee_orn_now,
                    clamp_limits=clamp,
                )
            )

            # Controller (at control rate): MPC returns u_bar, v_bar
            if t - last_controller_time >= ctrl_period:
                # Create references dict for MPC
                # We don't use TaskManager, so create minimal references
                references = {
                    "base_pose": None,
                    "base_velocity": None,
                    "ee_pose": None,
                    "ee_velocity": None,
                    "desired_velocity": {
                        "ee_velocity": desired_ee_vel_mpc,
                    },
                }

                # Control
                try:
                    v_bar, _ = controller.control(t, robot_states, references)
                    last_controller_time = t
                except Exception as e:
                    print(f"  Controller error at t={t:.3f}: {e}")
                    break

            # Compute velocity command: IK path vs MPC interpolation
            if use_ik_solver:
                mpc_base_vel = np.zeros(3) if v_bar is None else v_bar[1, :3]
                u = solve_ik(robot, desired_ee_vel_world, mpc_base_vel, ik_params)
            else:
                u = v_bar[1, :]

            # Command robot
            robot.command_velocity(u)
            t, _ = sim.step(t)

            # Measure tracking error
            actual_ee_pos, actual_ee_orn = robot.link_pose()
            desired_ee_pos, desired_ee_orn = ee_planner.get_desired_pose()

            pos_error = np.linalg.norm(actual_ee_pos - desired_ee_pos)
            orn_error = mm_math.quat_orientation_error(actual_ee_orn, desired_ee_orn)

            # Track errors
            episode_tracking_errors_pos.append(pos_error)
            episode_tracking_errors_orn.append(orn_error)
            max_tracking_error_pos_episode = max(
                max_tracking_error_pos_episode, pos_error
            )
            max_tracking_error_orn_episode = max(
                max_tracking_error_orn_episode, orn_error
            )

            # Check if goal reached
            goal_pos_error = np.linalg.norm(actual_ee_pos - goal_pos)
            goal_orn_error = mm_math.quat_orientation_error(actual_ee_orn, goal_orn)

            if (
                goal_pos_error <= success_pos_threshold
                and goal_orn_error <= success_orn_threshold
            ):
                print(f"  Goal reached at t={t:.2f}s!")
                print(
                    f"    Final pos error: {goal_pos_error:.4f}m, orn error: {goal_orn_error:.4f}"
                )
                goal_reached_count += 1
                break

            # Training-aligned early termination:
            # count cumulative slack violations; terminate once threshold is reached.
            if pos_error > slack_pos_threshold or orn_error > slack_rot_threshold:
                nr_kin_failures += 1
            if nr_kin_failures >= ik_fail_thresh:
                print(f"  Kinematic failures reached at t={t:.2f}s, terminating")
                print(f"    Pos error: {pos_error:.4f}m, orn error: {orn_error:.4f}")
                print(
                    f"    Slack thresholds: pos={slack_pos_threshold:.4f}, orn={slack_rot_threshold:.4f}"
                )
                print(f"    Failure count: {nr_kin_failures}/{ik_fail_thresh}")
                print(f"    Desired EE vel (world / IK): {desired_ee_vel_world}")
                actual_ee_lin_vel, actual_ee_ang_vel = robot.link_velocity()
                actual_ee_vel = np.concatenate([actual_ee_lin_vel, actual_ee_ang_vel])
                print(f"    Actual EE vel: {actual_ee_vel}")
                break

        # Episode summary
        mean_tracking_error_pos = np.mean(episode_tracking_errors_pos)
        mean_tracking_error_orn = np.mean(episode_tracking_errors_orn)

        tracking_errors_pos.append(mean_tracking_error_pos)
        tracking_errors_orn.append(mean_tracking_error_orn)

        # Check if tracking was good
        tracking_good = (
            max_tracking_error_pos_episode <= max_tracking_error_pos
            and max_tracking_error_orn_episode <= max_tracking_error_orn
        )
        if tracking_good:
            tracking_good_count += 1

        # Final goal error
        final_ee_pos, final_ee_orn = robot.link_pose()
        final_pos_error = np.linalg.norm(final_ee_pos - goal_pos)
        final_orn_error = mm_math.quat_orientation_error(final_ee_orn, goal_orn)
        goal_reached = (
            final_pos_error <= success_pos_threshold
            and final_orn_error <= success_orn_threshold
        )

        print("  Episode completed:")
        print(f"    Duration: {t:.2f}s")
        print(
            f"    Mean tracking error: pos={mean_tracking_error_pos:.4f}m, orn={mean_tracking_error_orn:.4f}"
        )
        print(
            f"    Max tracking error: pos={max_tracking_error_pos_episode:.4f}m, orn={max_tracking_error_orn_episode:.4f}"
        )
        print(
            f"    Final goal error: pos={final_pos_error:.4f}m, orn={final_orn_error:.4f}"
        )
        print(f"    Goal reached: {goal_reached}")
        print(f"    Tracking good: {tracking_good}")

        all_results.append(
            {
                "episode": episode + 1,
                "duration": t,
                "mean_tracking_error_pos": mean_tracking_error_pos,
                "mean_tracking_error_orn": mean_tracking_error_orn,
                "max_tracking_error_pos": max_tracking_error_pos_episode,
                "max_tracking_error_orn": max_tracking_error_orn_episode,
                "final_pos_error": final_pos_error,
                "final_orn_error": final_orn_error,
                "goal_reached": goal_reached,
                "tracking_good": tracking_good,
            }
        )

    # --- Summary ---
    tracking_errors_pos = np.array(tracking_errors_pos)
    tracking_errors_orn = np.array(tracking_errors_orn)

    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Episodes: {n_episodes}")
    print("\nTracking Performance:")
    print(
        f"  Mean tracking error (pos): {np.mean(tracking_errors_pos):.4f} ± {np.std(tracking_errors_pos):.4f} m"
    )
    print(
        f"  Mean tracking error (orn): {np.mean(tracking_errors_orn):.4f} ± {np.std(tracking_errors_orn):.4f}"
    )
    print(f"  Max tracking error (pos): {np.max(tracking_errors_pos):.4f} m")
    print(f"  Max tracking error (orn): {np.max(tracking_errors_orn):.4f}")
    print(
        f"  Tracking good rate: {tracking_good_count}/{n_episodes} ({100*tracking_good_count/n_episodes:.1f}%)"
    )

    print("\nGoal Reaching Performance:")
    print(
        f"  Goal reached: {goal_reached_count}/{n_episodes} ({100*goal_reached_count/n_episodes:.1f}%)"
    )
    final_pos_errors = [r["final_pos_error"] for r in all_results]
    final_orn_errors = [r["final_orn_error"] for r in all_results]
    print(
        f"  Mean final pos error: {np.mean(final_pos_errors):.4f} ± {np.std(final_pos_errors):.4f} m"
    )
    print(
        f"  Mean final orn error: {np.mean(final_orn_errors):.4f} ± {np.std(final_orn_errors):.4f}"
    )

    # Pass/fail criteria
    mean_tracking_pos_ok = np.mean(tracking_errors_pos) <= max_tracking_error_pos
    mean_tracking_orn_ok = np.mean(tracking_errors_orn) <= max_tracking_error_orn
    tracking_rate_ok = (
        tracking_good_count / n_episodes >= 0.8
    )  # At least 80% should have good tracking

    passed = mean_tracking_pos_ok and mean_tracking_orn_ok and tracking_rate_ok

    print(f"\nTest {'PASSED' if passed else 'FAILED'}")
    print(
        f"  Mean tracking error within threshold: {mean_tracking_pos_ok and mean_tracking_orn_ok}"
    )
    print(f"  Tracking good rate >= 80%: {tracking_rate_ok}")

    return {
        "passed": passed,
        "n_episodes": n_episodes,
        "tracking_good_count": tracking_good_count,
        "goal_reached_count": goal_reached_count,
        "mean_tracking_error_pos": np.mean(tracking_errors_pos),
        "mean_tracking_error_orn": np.mean(tracking_errors_orn),
        "results": all_results,
    }


def main():
    """CLI: load config, run :func:`test_ee_planner_with_mpsf`, return process exit code."""
    parser = argparse.ArgumentParser(description="Test EE planner with MPSF controller")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/train_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=10, help="Number of test episodes"
    )
    parser.add_argument(
        "--max-tracking-error-pos",
        type=float,
        default=0.1,
        help="Maximum allowed position tracking error (m)",
    )
    parser.add_argument(
        "--max-tracking-error-orn",
        type=float,
        default=0.2,
        help="Maximum allowed orientation tracking error",
    )
    parser.add_argument("--gui", action="store_true", help="Enable GUI visualization")
    parser.add_argument(
        "--use-ik-solver",
        action="store_true",
        help="Extract base velocities from MPC and pass through IK solver (fair comparison to RL training)",
    )
    parser.add_argument(
        "--closed-loop-planner",
        action="store_true",
        help="Use closed-loop EEPlanner for desired EE commands (default: open-loop)",
    )

    args = parser.parse_args()

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    print(f"Loading configuration from: {config_path}")

    # Run test
    planner_mode = "closed_loop" if args.closed_loop_planner else "open_loop"
    test_ee_planner_with_mpsf(
        config_path=config_path,
        n_episodes=args.n_episodes,
        max_tracking_error_pos=args.max_tracking_error_pos,
        max_tracking_error_orn=args.max_tracking_error_orn,
        gui=args.gui,
        use_ik_solver=args.use_ik_solver,
        planner_mode=planner_mode,
    )


if __name__ == "__main__":
    main()
