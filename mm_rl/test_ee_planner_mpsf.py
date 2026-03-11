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
from mm_simulator import simulation
from mm_utils import math as mm_math
from mm_utils import parsing
from mm_utils.parsing import recursive_dict_update


def _ensure_controller_config(config):
    """Merge MPC and scene config into config if missing (required for MPC)."""
    if "controller" not in config or config.get("controller", {}).get("type") != "MPC":
        path = parsing.parse_ros_path(
            {"package": "mm_run", "path": "config/controller/MPC.yaml"}
        )
        ctrl = parsing.load_config(path)
        config.setdefault("controller", {})
        config["controller"] = recursive_dict_update(
            ctrl.get("controller", {}), config["controller"]
        )
    if "scene" not in config.get("controller", {}):
        path = parsing.parse_ros_path(
            {"package": "mm_run", "path": "config/scene/empty.yaml"}
        )
        scene = parsing.load_config(path)
        config["controller"]["scene"] = scene.get("controller", {}).get(
            "scene",
            {
                "enabled": False,
                "collision_link_names": {"static_obstacles": ["ground"]},
            },
        )


def _setup_mpsf_config(ctrl_config):
    """Set MPSF-specific overrides: base/EE masks, EEVel weights, scene placeholder."""
    ctrl_config["base_mask"] = [0, 0, 0]
    ctrl_config["ee_mask"] = [0, 0, 0, 0, 0, 0]
    ctrl_config.setdefault("mpsf_params", {})
    ctrl_config["mpsf_params"]["mpsf_base_mask"] = [0, 0, 0]
    ctrl_config["mpsf_params"]["mpsf_ee_mask"] = [1, 1, 1, 1, 1, 1]
    ctrl_config["mpsf_params"]["mpsf_weight_multiplier"] = 100.0
    ctrl_config.setdefault("cost_params", {})
    ctrl_config["cost_params"].setdefault("EEVel", {})
    ctrl_config["cost_params"]["EEVel"]["Qk"] = [10.0] * 6
    ctrl_config["cost_params"]["EEVel"]["P"] = [10.0] * 6
    ctrl_config.setdefault(
        "scene",
        {"enabled": False, "collision_link_names": {"static_obstacles": ["ground"]}},
    )


def test_ee_planner_with_mpsf(
    config_path,
    n_episodes=10,
    max_tracking_error_pos=0.1,
    max_tracking_error_orn=0.2,
    success_pos_threshold=0.1,
    success_orn_threshold=0.1,
    gui=False,
    use_ik_solver=False,
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

    Returns:
        dict: Test results
    """

    # --- Config ---
    config = parsing.load_config(str(config_path))
    _ensure_controller_config(config)
    if gui:
        config["simulation"]["gui"] = True

    ctrl_config = config["controller"]
    _setup_mpsf_config(ctrl_config)
    print("MPSF Configuration:")
    print(f"  mpsf_ee_mask: {ctrl_config['mpsf_params']['mpsf_ee_mask']}")
    print(
        f"  mpsf_weight_multiplier: {ctrl_config['mpsf_params']['mpsf_weight_multiplier']}"
    )
    print(f"  EEVel Qk: {ctrl_config['cost_params']['EEVel']['Qk']}")

    sim_config = config["simulation"]

    # Initialize simulation
    timestamp = datetime.datetime.now()
    sim = simulation.BulletSimulation(sim_config, timestamp, cli_args=None)
    robot = sim.robot

    # --- Simulation & controller ---
    controller = MPC.MPC(ctrl_config)
    nu = sim_config["robot"]["dims"]["v"]
    nq = sim_config["robot"]["dims"]["q"]

    ik_params = None
    if use_ik_solver:
        ik_cfg = config.get("ik", {})
        lim = (
            config.get("controller", {})
            .get("robot", {})
            .get("limits", {})
            .get("state", {})
        )
        vel_lo = np.array(lim.get("lower", [0] * nu)[nq:])
        vel_hi = np.array(lim.get("upper", [0] * nu)[nq:])
        ik_params = {
            "nu": nu,
            "use_weighted_regularization": ik_cfg.get(
                "use_weighted_regularization", True
            ),
            "regularization_strength": ik_cfg.get("regularization_strength", 0.1),
            "joint_vel_lower": vel_lo,
            "joint_vel_upper": vel_hi,
        }

    # --- Episode parameters (from config) ---
    goal_cfg = config.get("goal", {})
    pos_range = goal_cfg.get("pos_range", [[-2.0, -2.0, 0.6], [2.0, 2.0, 1.4]])
    orn_range = goal_cfg.get("orn_range", 0)
    max_goal_dist = goal_cfg.get("max_goal_distance", 3.0)

    planner_cfg = config.get("planner", {})
    planner_vel_range = planner_cfg.get("vel_range", [0.2, 0.35])
    planner_slowdown = planner_cfg.get("slowdown_distance", 0.1)

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
        robot_states = robot.joint_states(add_noise=False)

        # Get initial EE pose for goal generation
        ee_pos, ee_orn = robot.link_pose()

        # Generate feasible goal
        goal_pos, goal_orn = generate_goal(pos_range, orn_range, np_random)
        goal_distance = np.linalg.norm(goal_pos - ee_pos)
        for _ in range(20):
            if goal_distance <= max_goal_dist:
                break
            goal_pos, goal_orn = generate_goal(pos_range, orn_range, np_random)
            goal_distance = np.linalg.norm(goal_pos - ee_pos)
        if goal_distance > max_goal_dist:
            print(
                f"  Warning: Goal {goal_distance:.2f}m away (max {max_goal_dist:.2f}m)"
            )

        print(f"Goal position: {goal_pos}")
        print(f"Goal orientation: {goal_orn}")

        ee_planner = EEPlanner(
            goal_pos,
            goal_orn,
            vel_range=tuple(planner_vel_range),
            dt=sim.timestep,
            np_random=np_random,
            slowdown_distance=planner_slowdown,
        )
        ee_planner.reset(ee_pos, ee_orn)

        # Reset controller state for new episode
        controller.reset()

        # Log initial planner velocity
        initial_vel = ee_planner.planner_vel
        print(
            f"  Planner velocity: {initial_vel:.3f} m/s, Goal distance: {goal_distance:.2f}m"
        )

        # Simulation loop
        t = 0.0
        max_duration = 30.0  # Maximum episode duration
        ctrl_period = 1.0 / ctrl_config.get("ctrl_rate", 10.0)
        last_controller_time = -ctrl_period
        u = np.zeros(nu)
        v_bar = None

        episode_tracking_errors_pos = []
        episode_tracking_errors_orn = []
        max_tracking_error_pos_episode = 0.0
        max_tracking_error_orn_episode = 0.0

        while t <= max_duration:
            robot_states = robot.joint_states(add_noise=False)

            # Desired EE vel from planner (updated every step)
            desired_lin_vel, desired_ang_vel = ee_planner.step()
            desired_ee_vel = np.concatenate([desired_lin_vel, desired_ang_vel])
            max_lin_vel = 1.0
            if np.linalg.norm(desired_ee_vel[:3]) > max_lin_vel:
                desired_ee_vel[:3] = (
                    desired_ee_vel[:3]
                    / np.linalg.norm(desired_ee_vel[:3])
                    * max_lin_vel
                )
            max_ang_vel = 2.0
            if np.linalg.norm(desired_ee_vel[3:]) > max_ang_vel:
                desired_ee_vel[3:] = (
                    desired_ee_vel[3:]
                    / np.linalg.norm(desired_ee_vel[3:])
                    * max_ang_vel
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
                        "ee_velocity": desired_ee_vel,  # MPSF will track this
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
                u = solve_ik(robot, desired_ee_vel, mpc_base_vel, ik_params)
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

            # Check if tracking error is too large (early termination)
            # Use a more lenient threshold - allow some deviation
            if (
                pos_error > max_tracking_error_pos * 3
                or orn_error > max_tracking_error_orn * 3
            ):
                print(f"  Tracking error too large at t={t:.2f}s, terminating")
                print(f"    Pos error: {pos_error:.4f}m, orn error: {orn_error:.4f}")
                print(f"    Desired EE vel: {desired_ee_vel}")
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


def solve_ik(robot, desired_ee_vel, base_vel, ik_params):
    """Solve inverse kinematics (same as RL training).

    Args:
        robot: Robot instance
        desired_ee_vel: Desired end-effector velocity (6,)
        base_vel: Base velocity (3,)
        ik_params: IK parameters {nu, use_weighted_regularization, regularization_strength, joint_vel_lower, joint_vel_upper}

    Returns:
        ndarray: Joint velocities (nu,)
    """
    nu = ik_params["nu"]
    jlo = ik_params["joint_vel_lower"]
    jhi = ik_params["joint_vel_upper"]
    # Clamp desired EE vel (same limits as in main loop)
    v = desired_ee_vel.copy()
    nlin = np.linalg.norm(v[:3])
    if nlin > 1.0:
        v[:3] = v[:3] / nlin * 1.0
    nang = np.linalg.norm(v[3:])
    if nang > 2.0:
        v[3:] = v[3:] / nang * 2.0
    # Jacobian and arm residual
    q, _ = robot.joint_states()
    J = robot.jacobian(q)
    base_idx = [0, 1, 2]
    arm_idx = list(range(3, nu))
    v_ee_from_base = J[:, base_idx] @ base_vel
    v_arm = v - v_ee_from_base
    J_arm = J[:, arm_idx]
    if ik_params["use_weighted_regularization"]:
        max_vels = np.maximum(np.abs(jhi[arm_idx]), 1e-6)
        W = np.diag(1.0 / max_vels)
        A = J_arm.T @ J_arm + ik_params["regularization_strength"] * (W.T @ W)
        arm_vel = np.linalg.pinv(A) @ (J_arm.T @ v_arm)
    else:
        damp = 0.01
        Jpinv = J_arm.T @ np.linalg.inv(J_arm @ J_arm.T + damp * np.eye(6))
        arm_vel = Jpinv @ v_arm
    u = np.zeros(nu)
    u[base_idx] = base_vel
    u[arm_idx] = arm_vel
    for i in range(nu):
        u[i] = np.clip(u[i], jlo[i], jhi[i])
    return u


def generate_goal(pos_range, orn_range, np_random):
    """Generate a random goal within the specified range.

    Args:
        pos_range: Position range [min, max]
        orn_range: Orientation range [min, max]
        np_random: Random number generator

    Returns:
        goal_pos: Goal position (3,)
        goal_orn: Goal orientation (4,)
    """
    pos_min = np.array(pos_range[0])
    pos_max = np.array(pos_range[1])
    goal_pos = np_random.uniform(pos_min, pos_max)

    max_angle = float(orn_range)
    axis = np_random.uniform(-1, 1, size=3)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    angle = np_random.uniform(0, max_angle)
    goal_orn = np.array(
        [
            axis[0] * np.sin(angle / 2),
            axis[1] * np.sin(angle / 2),
            axis[2] * np.sin(angle / 2),
            np.cos(angle / 2),
        ]
    )
    return goal_pos, goal_orn


def main():
    """Main entry point."""
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

    args = parser.parse_args()

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    print(f"Loading configuration from: {config_path}")

    # Run test
    results = test_ee_planner_with_mpsf(
        config_path=config_path,
        n_episodes=args.n_episodes,
        max_tracking_error_pos=args.max_tracking_error_pos,
        max_tracking_error_orn=args.max_tracking_error_orn,
        gui=args.gui,
        use_ik_solver=args.use_ik_solver,
    )

    return 0 if results["passed"] else 1


if __name__ == "__main__":
    exit(main())
