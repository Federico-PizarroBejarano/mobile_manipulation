import argparse
import datetime
import logging

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import mm_control.MPC as MPC
from mm_plan.TaskManager import TaskManager
from mm_simulator import simulation
from mm_utils import parsing
from mm_utils.math import (
    compute_base_pose_errors,
    compute_ee_pose_errors,
    compute_velocity_command,
    normalize_mask,
    wrap_pi_scalar,
)
from mm_utils.metrics import MPSFMetricsCollector, extract_robot_states


def parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with config, ctrl_config,
            planner_config, and GUI attributes.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, help="Path to configuration file."
    )
    parser.add_argument(
        "--ctrl_config",
        type=str,
        default="default",
        help="Controller config. Overwrites yaml settings if not 'default'",
    )
    parser.add_argument(
        "--planner_config",
        type=str,
        default="default",
        help="Planner config. Overwrites yaml settings if not 'default'",
    )
    parser.add_argument("--GUI", action="store_true", help="Enable Pybullet GUI")
    return parser.parse_args()


def load_config(args):
    """Load and merge configuration files.

    Args:
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        dict: Merged configuration dictionary.
    """
    config = parsing.load_config(args.config)

    if args.ctrl_config != "default":
        ctrl_config = parsing.load_config(args.ctrl_config)
        config = parsing.recursive_dict_update(config, ctrl_config)

    if args.planner_config != "default":
        planner_config = parsing.load_config(args.planner_config)
        config = parsing.recursive_dict_update(config, planner_config)

    if args.GUI:
        config["simulation"]["gui"] = True

    return config


def setup_experiment(config):
    """Initialize simulator, controller, and task manager.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        tuple: (sim, controller, task_manager, ctrl_config, sim_config).
    """
    # Setup basic logging for controllers and planners
    log_level = config.get("logging", {}).get("log_level", logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    for logger_name in ["Controller", "Planner", "Simulator"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        if not logger.handlers:  # Avoid duplicate handlers
            logger.addHandler(handler)

    sim_config = config["simulation"]
    ctrl_config = config["controller"]
    planner_config = config.get("planner", None)

    # Simulator
    sim = simulation.BulletSimulation(
        config=sim_config, timestamp=datetime.datetime.now(), cli_args=None
    )

    # Controller
    control_class = getattr(MPC, ctrl_config["type"], None)
    if control_class is None:
        raise ValueError(f"Unknown controller type: {ctrl_config['type']}")
    controller = control_class(ctrl_config)

    # Task Manager
    task_manager = TaskManager(planner_config)
    task_manager.activatePlanners()

    print(f"Starting simulation: duration={sim.duration}s, timestep={sim.timestep}s")

    return sim, controller, task_manager, ctrl_config, sim_config


def run_simulation(
    sim,
    controller,
    task_manager,
    ctrl_config,
    sim_config,
    base_goal=None,
    ee_goal=None,
):
    """Run the main simulation loop and collect MPSF metrics.

    Args:
        sim (simulation.BulletSimulation): Simulation environment.
        controller: Controller instance with control(), N, dt, and mask attributes.
        task_manager (TaskManager): Task manager with getReferences() and update().
        ctrl_config (dict): Controller configuration with "cmd_vel_type" key.
        sim_config (dict): Simulation configuration with "robot" -> "dims" -> "v".
        base_goal (np.ndarray, optional): Base goal pose [x, y, yaw]. Defaults to None.
        ee_goal (np.ndarray, optional): EE goal pose [x, y, z, roll, pitch, yaw]. Defaults to None.

    Returns:
        MPSFMetricsCollector: Metrics collector with all collected metrics.
    """
    robot = sim.robot
    u = np.zeros(sim_config["robot"]["dims"]["v"])

    # Controller frequency management (one solve per controller.dt of sim time)
    ctrl_period = float(controller.dt)
    last_controller_time = -ctrl_period  # Initialize to allow first call

    metrics_collector = MPSFMetricsCollector()

    # Initial state
    robot_states = robot.joint_states(add_noise=False)
    states = extract_robot_states(robot, robot_states)

    t = 0.0
    while t <= sim.duration:
        print(f"-------------- {t:.3f}s/{float(sim.duration):.3f}s ------------------")
        robot_states = robot.joint_states(add_noise=False)

        # Only call controller if enough time has passed
        if t - last_controller_time + 1e-6 >= ctrl_period:
            # Get references from TaskManager (only when controller is called)
            references = task_manager.getReferences(
                t, robot_states, controller.N + 1, controller.dt
            )

            desired_base_vel, desired_ee_vel = calculate_desired_velocity(
                base_goal, ee_goal, states, controller
            )

            # Add desired velocities for MPSF (only add if not None)
            if desired_base_vel is not None or desired_ee_vel is not None:
                desired_velocity = {}
                if desired_base_vel is not None:
                    desired_velocity["base_velocity"] = desired_base_vel
                if desired_ee_vel is not None:
                    desired_velocity["ee_velocity"] = desired_ee_vel
                references["desired_velocity"] = desired_velocity

            # Control
            v_bar, u_bar = controller.control(t, robot_states, references)
            last_controller_time = t

        # Compute velocity command using the latest controller output
        u = compute_velocity_command(
            u,
            u_bar,
            v_bar,
            ctrl_config["cmd_vel_type"],
            t - last_controller_time,
            controller.dt,
            simulation_dt=sim.timestep,
        )

        robot.command_velocity(u)
        t, _ = sim.step(t)

        # Extract states
        states = extract_robot_states(robot, robot_states)
        task_manager.update(
            t, states, base_mask=controller.base_mask, ee_mask=controller.ee_mask
        )

        # Normalize masks and compute errors using utility functions
        if base_goal is not None:
            mpsf_base_mask = normalize_mask(controller.mpsf_base_mask, dim=3)
            base_pos_err, base_yaw_err, _, _ = compute_base_pose_errors(
                states["base"]["pose"], base_goal, mpsf_base_mask, 0, 0
            )
            print(
                "EXPERIMENT - ",
                f"Base Pos Error: {base_pos_err:.3f} | ",
                f"Base Yaw Error: {base_yaw_err:.3f}",
            )

        if ee_goal is not None:
            mpsf_ee_mask = normalize_mask(controller.mpsf_ee_mask, dim=6)
            ee_pos_err, ee_ori_err, _, _ = compute_ee_pose_errors(
                states["EE"]["pose"], ee_goal, mpsf_ee_mask, 0, 0
            )
            print(
                "EXPERIMENT - ",
                f"EE Pos Error: {ee_pos_err:.3f} | ",
                f"EE Orientation Error: {ee_ori_err:.3f}",
            )

        # Update metrics
        metrics_collector.update(
            references,
            states,
            u,
            desired_base_vel,
            desired_ee_vel,
            controller,
            robot_states,
            sim.timestep,
        )

    return metrics_collector


def calculate_desired_velocity(base_goal, ee_goal, states, controller):
    """Calculate desired base and EE velocity from goal positions and robot states.

    Uses a deadband approach: maintains constant velocity when far from goal,
    switches to proportional control when close to goal.

    Args:
        base_goal (np.ndarray or None): Goal base position in world frame, shape (3,).
        ee_goal (np.ndarray or None): Goal EE position in world frame, shape (6,).
        states (dict): Current robot states.
        controller (MPCBase): Controller instance with robot.ub_u and robot.lb_u attributes.

    Returns:
        tuple: (desired_base_vel, desired_ee_vel) where each is a (3,) or (6,) array, or None.
    """
    # Velocity thresholds: maintain constant velocity if error > threshold
    base_threshold = [0.3, 0.3, 0.3]  # meters
    ee_threshold = [0.2, 0.2, 0.2, 0.3, 0.3, 0.3]  # meters, radians

    # Maximum velocities (used when far from goal)
    max_base_vel = np.array([1, 1, 1])
    max_ee_vel = np.array([0.8, 0.8, 0.8, 0.5, 0.5, 0.5])

    # Calculate base velocity
    if base_goal is not None:
        base_vel = base_goal - states["base"]["pose"]
        base_vel[-1] = wrap_pi_scalar(base_vel[-1])

        for i in range(3):
            if abs(base_vel[i]) > base_threshold[i]:
                base_vel[i] = np.sign(base_vel[i]) * max_base_vel[i]

        # Clip velocities to constraints
        base_vel = np.clip(
            base_vel, controller.robot.lb_u[:3], controller.robot.ub_u[:3]
        )
    else:
        base_vel = None

    # Calculate EE velocity
    if ee_goal is not None:
        ee_vel = ee_goal - states["EE"]["pose"]

        # Orientation error
        R_goal = Rot.from_euler("xyz", ee_goal[3:]).as_matrix()
        R_curr = Rot.from_euler("xyz", states["EE"]["pose"][3:]).as_matrix()
        R_error = R_goal @ R_curr.T
        rotvec = Rot.from_matrix(R_error).as_rotvec()
        ee_vel[3:] = rotvec

        for i in range(6):
            if abs(ee_vel[i]) > ee_threshold[i]:
                ee_vel[i] = np.sign(ee_vel[i]) * max_ee_vel[i]

        # Clip velocities to constraints
        ee_vel = np.clip(ee_vel, controller.robot.lb_u[3:9], controller.robot.ub_u[3:9])
    else:
        ee_vel = None

    return base_vel, ee_vel


def main():
    """Main entry point for MPSF experiment script."""
    np.set_printoptions(precision=3, suppress=True)

    args = parse_args()
    config = load_config(args)
    sim, controller, task_manager, ctrl_config, sim_config = setup_experiment(config)

    # Extract MPSF goals from config if available
    mpsf_goals = ctrl_config.get("mpsf_params", {})
    base_goal = mpsf_goals.get("mpsf_base_goal")
    base_goal = np.array(base_goal) if base_goal is not None else None
    ee_goal = mpsf_goals.get("mpsf_ee_goal")
    ee_goal = np.array(ee_goal) if ee_goal is not None else None

    metrics_collector = run_simulation(
        sim,
        controller,
        task_manager,
        ctrl_config,
        sim_config,
        base_goal=base_goal,
        ee_goal=ee_goal,
    )
    metrics_collector.print_summary()


if __name__ == "__main__":
    main()
