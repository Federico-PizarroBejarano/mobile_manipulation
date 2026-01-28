import argparse
import datetime
import logging
import time

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as Rot

import mm_control.MPC as MPC
from mm_plan.TaskManager import TaskManager
from mm_simulator import simulation
from mm_utils import parsing
from mm_utils.math import (
    compute_base_pose_errors,
    compute_ee_pose_errors,
    normalize_mask,
    wrap_pi_scalar,
)


def compute_jerkiness(velocities, dt):
    """Compute jerkiness metric as the rate of change of velocity commands.

    Args:
        velocities (np.ndarray): Velocity commands, shape (N, nu).
        dt (float): Time step in seconds.

    Returns:
        float: Jerkiness metric. Returns 0.0 if len(velocities) < 2.
    """
    if len(velocities) < 2:
        return 0.0
    velocity_changes = np.diff(velocities, axis=0)
    jerkiness = np.mean(np.linalg.norm(velocity_changes, axis=1)) / dt
    return jerkiness


def compute_rmse(actual, reference, mask=None):
    """Compute Root Mean Square Error between actual and reference.

    Args:
        actual (np.ndarray): Actual values, shape (N, dim) or (dim,).
        reference (np.ndarray): Reference values, shape (N, dim) or (dim,).
        mask (np.ndarray, optional): Mask to apply, shape (dim,). Defaults to None.

    Returns:
        float: RMSE value. Returns 0.0 if actual or reference is None.
    """
    if actual is None or reference is None:
        return 0.0

    # Handle 1D case
    if actual.ndim == 1:
        actual = actual.reshape(1, -1)
    if reference.ndim == 1:
        reference = reference.reshape(1, -1)

    # Ensure same shape
    min_len = min(len(actual), len(reference))
    actual = actual[:min_len]
    reference = reference[:min_len]

    # Apply mask if provided
    if mask is not None:
        actual = actual * mask
        reference = reference * mask

    errors = actual - reference
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    return rmse


def compute_corrections(desired_vel, actual_vel, mask=None):
    """Compute corrections as norm of difference between desired and actual velocity.

    Args:
        desired_vel (np.ndarray): Desired velocity, shape (dim,) or (N, dim).
        actual_vel (np.ndarray): Actual velocity, shape (dim,) or (N, dim).
        mask (np.ndarray, optional): Mask to apply, shape (dim,). Defaults to None.

    Returns:
        float: L2 norm of corrections. Returns 0.0 if desired_vel or actual_vel is None.
    """
    if desired_vel is None or actual_vel is None:
        return 0.0

    # Handle 1D case
    if desired_vel.ndim == 1:
        desired_vel = desired_vel.reshape(1, -1)
    if actual_vel.ndim == 1:
        actual_vel = actual_vel.reshape(1, -1)

    # Apply mask if provided
    if mask is not None:
        desired_vel = desired_vel * mask
        actual_vel = actual_vel * mask

    corrections = desired_vel - actual_vel
    return np.linalg.norm(corrections)


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


def compute_velocity_command(
    u, u_bar, v_bar, cmd_vel_type, sim_timestep, controller_dt
):
    """Compute velocity command from controller output.

    Args:
        u (np.ndarray): Current velocity command, shape (nu,).
        u_bar (np.ndarray): Control input, shape (N+1, nu) or (nu,).
        v_bar (np.ndarray): Velocity trajectory, shape (N+1, nu).
        cmd_vel_type (str): "integration" or "interpolation".
        sim_timestep (float): Simulation timestep in seconds.
        controller_dt (float): Controller timestep in seconds.

    Returns:
        np.ndarray: Computed velocity command, shape (nu,).
    """
    if cmd_vel_type == "integration":
        return u + u_bar[0] * sim_timestep
    elif cmd_vel_type == "interpolation":
        N = v_bar.shape[0]
        t_v_bar = np.arange(N) * controller_dt
        v_interp = interp1d(
            t_v_bar, v_bar, axis=0, bounds_error=False, fill_value="extrapolate"
        )
        return v_interp(sim_timestep)
    else:
        raise ValueError(f"Unknown cmd_vel_type: {cmd_vel_type}")


def extract_robot_states(robot, robot_states):
    """Extract base and end-effector pose and velocity from robot states.

    Args:
        robot: Robot instance with link_pose() and link_velocity() methods.
        robot_states (tuple): (joint_positions, joint_velocities) from robot.joint_states().

    Returns:
        dict: Dictionary with "base" and "EE" keys, each containing "pose" and
            "velocity" arrays. Base pose/vel shape (3,), EE pose/vel shape (6,).
    """
    ee_curr_pos, ee_cur_orn = robot.link_pose()
    ee_euler = Rot.from_quat(ee_cur_orn).as_euler("xyz")
    ee_pose = np.hstack([ee_curr_pos, ee_euler])

    ee_lin_vel, ee_ang_vel = robot.link_velocity()
    ee_vel = np.hstack([ee_lin_vel, ee_ang_vel])

    base_pose = robot_states[0][:3]
    base_vel = robot_states[1][:3]

    return {
        "base": {"pose": base_pose, "velocity": base_vel},
        "EE": {"pose": ee_pose, "velocity": ee_vel},
    }


def get_constraint_violations(controller, robot_states):
    """Extract constraint violations from measured robot states and controller log.

    Args:
        controller: Controller instance with log, collision_link_names, and robot attributes.
        robot_states (tuple): (q, v) tuple from robot.joint_states(), where q is positions
            and v is velocities.

    Returns:
        dict: Dictionary with constraint names as keys. Values are dicts with:
            - "max": Maximum violation value
            - "violations": Number of timesteps with violation > threshold
            - "max_per_dim" (optional): Per-dimension max violations for state/control
    """
    violations = {}
    violation_threshold = 1e-3  # 1mm or 0.001rad for measured violations
    collision_violation_threshold = 1e-3

    # Collision constraints from log
    for name in controller.collision_link_names:
        constraint_key = "_".join([name, "constraint"])
        constraint_vals = controller.log.get(constraint_key)
        if (
            constraint_vals
            and isinstance(constraint_vals, list)
            and len(constraint_vals) > 0
        ):
            # Each element in constraint_vals is a timestep
            all_values = []
            timesteps_with_violation = 0
            for v in constraint_vals:
                # Convert CasADi objects to numpy arrays
                if hasattr(v, "full"):
                    v = v.full()
                if not isinstance(v, np.ndarray):
                    v = np.array(v)

                v_flat = v.flatten()
                all_values.extend(v_flat)
                if np.any(v_flat > collision_violation_threshold):
                    timesteps_with_violation += 1

            # Max value (closest to boundary, can be negative if satisfied)
            max_value = np.max(all_values) if all_values else 0.0

            violations[name] = {
                "max": max_value,
                "violations": timesteps_with_violation,
            }

    # Measured state constraints - evaluate from actual robot states
    q_meas, v_meas = robot_states
    q_meas_wrapped = q_meas.copy()
    for i in range(2, len(q_meas)):  # Wrap revolute joints (base yaw + arm joints)
        q_meas_wrapped[i] = wrap_pi_scalar(q_meas[i])
    x_meas = np.hstack([q_meas_wrapped, v_meas])

    # Compute violations: positive means violation
    vio_upper = x_meas - controller.robot.ub_x
    vio_lower = controller.robot.lb_x - x_meas
    vio_per_dim = np.maximum(vio_upper, vio_lower)

    max_vio = float(np.max(vio_per_dim)) if vio_per_dim.size else 0.0
    has_violation = max_vio > violation_threshold

    # Print violation details if significant
    if has_violation:
        argmax_vio = int(np.argmax(vio_per_dim))
        nq = len(q_meas)
        dim_name = f"q[{argmax_vio}]" if argmax_vio < nq else f"v[{argmax_vio - nq}]"
        print(
            "EXPERIMENT - "
            f"State bound violation: max={max_vio:.6f} at {dim_name} "
            f"(x={x_meas[argmax_vio]:.6f}, "
            f"lb={controller.robot.lb_x[argmax_vio]:.6f}, "
            f"ub={controller.robot.ub_x[argmax_vio]:.6f})"
        )

    violations["state"] = {
        "max": max_vio,
        "violations": 1 if has_violation else 0,
        "max_per_dim": vio_per_dim,
    }

    # Control constraints - evaluate from u_bar if available (still useful for MPC diagnostics)
    if (
        hasattr(controller, "controlCst")
        and hasattr(controller, "x_bar")
        and hasattr(controller, "u_bar")
    ):
        try:
            nlp_p_map_bar = controller.log.get("ocp_param", [])
            if not nlp_p_map_bar:
                nlp_p_map_bar = [{}] * (controller.N + 1)

            control_vals = controller.evaluate_constraints(
                controller.controlCst, controller.x_bar, controller.u_bar, nlp_p_map_bar
            )

            if control_vals and len(control_vals) > 0:
                # Each element is a timestep
                nu = controller.robot.ssSymMdl["nu"]
                all_values = []
                timesteps_with_violation = 0
                per_dim_max = np.full(
                    nu, -np.inf
                )  # Start with -inf to get max correctly

                for v in control_vals:
                    # Convert CasADi objects to numpy arrays
                    if hasattr(v, "full"):
                        v = v.full()
                    v_arr = np.array(v).flatten()
                    all_values.extend(v_arr)

                    # Check if this timestep has any violation
                    if np.any(v_arr > violation_threshold):
                        timesteps_with_violation += 1

                    upper_values = v_arr[:nu]
                    lower_values = v_arr[nu:]
                    per_dim_max = np.maximum(
                        per_dim_max, np.maximum(upper_values, lower_values)
                    )

                # Max value (closest to boundary, can be negative if satisfied)
                max_value = np.max(all_values) if all_values else 0.0

                violations["control"] = {
                    "max": max_value,
                    "violations": timesteps_with_violation,
                    "max_per_dim": per_dim_max,
                }
        except Exception:
            pass  # Skip if evaluation fails

    return violations


def update_metrics(
    mpsf_metrics,
    references,
    states,
    u,
    u_prev,
    desired_base_vel,
    desired_ee_vel,
    controller,
    robot_states,
    sim_timestep,
):
    """Update MPSF metrics dictionary with current timestep data.

    Args:
        mpsf_metrics (dict): Dictionary with metric lists to append to.
        references (dict): Reference trajectories with optional "base_pose" and
            "ee_pose" keys, shape (N+1, dim).
        states (dict): Current robot states.
        u (np.ndarray): Current velocity command, shape (nu,).
        u_prev (np.ndarray): Previous velocity command, shape (nu,).
        desired_base_vel (np.ndarray or None): Desired base velocity, shape (3,).
        desired_ee_vel (np.ndarray or None): Desired EE velocity, shape (6,).
        controller: Controller instance with mask attributes and log.
        robot_states (tuple): (q, v) tuple from robot.joint_states().
        sim_timestep (float): Simulation timestep in seconds.
    """
    # Corrections
    if desired_base_vel is not None:
        correction = compute_corrections(
            desired_base_vel, u[:3], controller.mpsf_base_mask
        )
        mpsf_metrics["base_corrections"].append(correction)

    if desired_ee_vel is not None:
        correction = compute_corrections(desired_ee_vel, u[3:], controller.mpsf_ee_mask)
        mpsf_metrics["ee_corrections"].append(correction)

    # RMSE
    base_pose_ref = references.get("base_pose")
    if base_pose_ref is not None:
        base_rmse = compute_rmse(
            states["base"]["pose"], base_pose_ref[0], controller.base_mask
        )
        mpsf_metrics["base_rmses"].append(base_rmse)

    ee_pose_ref = references.get("ee_pose")
    if ee_pose_ref is not None:
        ee_rmse = compute_rmse(states["EE"]["pose"], ee_pose_ref[0], controller.ee_mask)
        mpsf_metrics["ee_rmses"].append(ee_rmse)

    # Jerkiness
    jerkiness = compute_jerkiness(np.vstack([u_prev, u]), sim_timestep)
    mpsf_metrics["jerks"].append(jerkiness)

    # Control effort (L2 norm of velocity command)
    control_effort = np.linalg.norm(u)
    mpsf_metrics["control_efforts"].append(control_effort)

    # Constraint violations (using measured states)
    violations = get_constraint_violations(controller, robot_states)
    mpsf_metrics["constraint_violations"].append(violations)


def run_simulation(
    sim,
    controller,
    task_manager,
    ctrl_config,
    sim_config,
):
    """Run the main simulation loop and collect MPSF metrics.

    Args:
        sim (simulation.BulletSimulation): Simulation environment.
        controller: Controller instance with control(), N, dt, and mask attributes.
        task_manager (TaskManager): Task manager with getReferences() and update().
        ctrl_config (dict): Controller configuration with "cmd_vel_type" key.
        sim_config (dict): Simulation configuration with "robot" -> "dims" -> "v".

    Returns:
        dict: Dictionary with keys "base_corrections", "ee_corrections",
            "base_rmses", "ee_rmses", "jerks", "control_efforts",
            "constraint_violations".
    """
    robot = sim.robot
    t = 0.0
    u = np.zeros(sim_config["robot"]["dims"]["v"])
    u_prev = u.copy()

    mpsf_metrics = {
        "base_corrections": [],
        "ee_corrections": [],
        "base_rmses": [],
        "ee_rmses": [],
        "jerks": [],
        "control_efforts": [],
        "constraint_violations": [],
    }

    # Goal position in world frame
    base_goal = np.array([1.5, 0.4, 0])
    ee_goal = np.array([2.5, 0.4, 0.7, 0, 0, 0])

    # Initial state
    robot_states = robot.joint_states(add_noise=False)
    states = extract_robot_states(robot, robot_states)

    while t <= sim.duration:
        print(f"-------------- {t:.3f}/{sim.duration} ------------------")
        robot_states = robot.joint_states(add_noise=False)
        references = task_manager.getReferences(
            t, robot_states, controller.N + 1, controller.dt
        )

        desired_base_vel, desired_ee_vel = calculate_desired_velocity(
            base_goal, ee_goal, states, controller
        )

        # Add desired velocities for MPSF
        if desired_base_vel is not None or desired_ee_vel is not None:
            references["desired_velocity"] = {
                "base_velocity": desired_base_vel,
                "ee_velocity": desired_ee_vel,
            }

        # Control
        v_bar, u_bar = controller.control(t, robot_states, references)

        # Compute velocity command
        u = compute_velocity_command(
            u, u_bar, v_bar, ctrl_config["cmd_vel_type"], sim.timestep, controller.dt
        )

        robot.command_velocity(u)
        t, _ = sim.step(t)

        # Extract states
        states = extract_robot_states(robot, robot_states)
        task_manager.update(
            t, states, base_mask=controller.base_mask, ee_mask=controller.ee_mask
        )

        # Normalize masks and compute errors using utility functions
        mpsf_base_mask = normalize_mask(controller.mpsf_base_mask, dim=3)
        mpsf_ee_mask = normalize_mask(controller.mpsf_ee_mask, dim=6)
        base_pos_err, base_yaw_err, _, _ = compute_base_pose_errors(
            states["base"]["pose"], base_goal, mpsf_base_mask, 0, 0
        )
        ee_pos_err, ee_ori_err, _, _ = compute_ee_pose_errors(
            states["EE"]["pose"], ee_goal, mpsf_ee_mask, 0, 0
        )

        print(
            "EXPERIMENT - ",
            f"Base Pos Error: {base_pos_err:.3f} | ",
            f"Base Yaw Error: {base_yaw_err:.3f}",
        )
        print(
            "EXPERIMENT - ",
            f"EE Pos Error: {ee_pos_err:.3f} | ",
            f"EE Orientation Error: {ee_ori_err:.3f}",
        )

        # Update metrics
        update_metrics(
            mpsf_metrics,
            references,
            states,
            u,
            u_prev,
            desired_base_vel,
            desired_ee_vel,
            controller,
            robot_states,
            sim.timestep,
        )
        u_prev = u.copy()

        time.sleep(sim.timestep)

    return mpsf_metrics


def calculate_desired_velocity(base_goal, ee_goal, states, controller):
    """Calculate desired base and EE velocity from goal positions and robot states.

    Args:
        base_goal (np.ndarray): Goal base position in world frame, shape (3,).
        ee_goal (np.ndarray): Goal EE position in world frame, shape (6,).
        states (dict): Current robot states.
        controller (MPCBase): Controller instance with robot.ub_u and robot.lb_u attributes.

    Returns:
        tuple: (desired_base_vel, desired_ee_vel) where each is a (3,) or (6,) array.
    """

    # Gain constants
    k_base = np.array([0.5, 0.5, 0.5])
    k_ee = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    # Calculate base velocity
    base_vel = base_goal - states["base"]["pose"]
    base_vel[-1] = wrap_pi_scalar(base_vel[-1])
    base_vel = k_base * base_vel

    # Calculate EE velocity
    ee_vel = ee_goal - states["EE"]["pose"]
    R_goal = Rot.from_euler("xyz", ee_goal[3:]).as_matrix()
    R_curr = Rot.from_euler("xyz", states["EE"]["pose"][3:]).as_matrix()
    R_error = R_goal @ R_curr.T
    rotvec = Rot.from_matrix(R_error).as_rotvec()
    ee_vel[3:] = rotvec
    ee_vel = k_ee * ee_vel

    # Clip velocities to constraints
    base_vel = np.clip(base_vel, controller.robot.lb_u[:3], controller.robot.ub_u[:3])
    ee_vel = np.clip(ee_vel, controller.robot.lb_u[3:9], controller.robot.ub_u[3:9])

    return base_vel, ee_vel


def print_metrics(mpsf_metrics):
    """Print formatted summary of MPSF experiment metrics.

    Args:
        mpsf_metrics (dict): Dictionary of collected metrics from run_simulation().
    """
    print("\n" + "=" * 80)
    print("MPSF EXPERIMENT METRICS SUMMARY")
    print("=" * 80)

    # RMSE
    if mpsf_metrics["base_rmses"]:
        print(f"Base Pose RMSE (average): {np.mean(mpsf_metrics['base_rmses']):.4f}")
    else:
        print("Base Pose RMSE: N/A (no base pose references)")

    if mpsf_metrics["ee_rmses"]:
        print(f"EE Pose RMSE (average): {np.mean(mpsf_metrics['ee_rmses']):.4f}")
    else:
        print("EE Pose RMSE: N/A (no EE pose references)")

    # Corrections
    if mpsf_metrics["base_corrections"]:
        print(f"Mean Base Corrections: {np.mean(mpsf_metrics['base_corrections']):.4f}")
        print(
            f"Max Base Correction: {np.max(mpsf_metrics['base_corrections']):.4f} m/s"
        )
    if mpsf_metrics["ee_corrections"]:
        print(f"Mean EE Corrections: {np.mean(mpsf_metrics['ee_corrections']):.4f}")
        print(f"Max EE Correction: {np.max(mpsf_metrics['ee_corrections']):.4f} m/s")

    # Jerkiness
    if mpsf_metrics["jerks"]:
        print(f"Mean Jerkiness: {np.mean(mpsf_metrics['jerks']):.4f} m/s³")

    # Control effort
    if mpsf_metrics["control_efforts"]:
        print(
            f"Mean Control Effort: {np.mean(mpsf_metrics['control_efforts']):.4f} m/s/s"
        )
        print(
            f"Max Control Effort: {np.max(mpsf_metrics['control_efforts']):.4f} m/s/s"
        )

    # Constraint violations
    if any(mpsf_metrics["constraint_violations"]):
        print("\nConstraint Violations Summary:")
        all_names = set()
        for violations in mpsf_metrics["constraint_violations"]:
            all_names.update(violations.keys())

        for name in sorted(all_names):
            # Collect all violations for this constraint across all timesteps
            constraint_data = [
                v.get(name)
                for v in mpsf_metrics["constraint_violations"]
                if name in v and isinstance(v.get(name), dict)
            ]

            if not constraint_data:
                continue

            # Get max across all timesteps
            max_v = max([d.get("max", 0.0) for d in constraint_data])
            # Count how many simulation timesteps had violations
            timesteps_with_violation = sum(
                [1 for d in constraint_data if d.get("violations", 0) > 0]
            )
            total_steps = len(constraint_data)

            # Check if per-dimension data exists
            has_per_dim = any("max_per_dim" in d for d in constraint_data)

            if has_per_dim:
                # Get max per dimension across all timesteps
                per_dim_arrays = [
                    d.get("max_per_dim") for d in constraint_data if "max_per_dim" in d
                ]
                if per_dim_arrays:
                    per_dim_max = np.max(per_dim_arrays, axis=0)
                    per_dim_str = ", ".join([f"{v:.4f}" for v in per_dim_max])
                    print(
                        f"  {name}: max={max_v:.6f}, violations={timesteps_with_violation}/{total_steps}, per_dim_max=[{per_dim_str}]"
                    )
                else:
                    print(
                        f"  {name}: max={max_v:.6f}, violations={timesteps_with_violation}/{total_steps}"
                    )
            else:
                print(
                    f"  {name}: max={max_v:.6f}, violations={timesteps_with_violation}/{total_steps}"
                )

    print("=" * 80 + "\n")


def main():
    """Main entry point for MPSF experiment script."""
    np.set_printoptions(precision=3, suppress=True)

    args = parse_args()
    config = load_config(args)
    sim, controller, task_manager, ctrl_config, sim_config = setup_experiment(config)

    mpsf_metrics = run_simulation(
        sim,
        controller,
        task_manager,
        ctrl_config,
        sim_config,
    )
    print_metrics(mpsf_metrics)


if __name__ == "__main__":
    main()
