import time
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from mm_control.MPCBase import MPCBase
from mm_control.MPCCostFunctions import CostFunctionRegistry
from mm_utils.math import wrap_pi_array


class MPSF(MPCBase):
    """Model Predictive Safety Filter - minimizes difference between desired and actual velocities while maintaining constraints.

    Uses BaseVel and EEVel cost functions (same as MPC) to track velocities in state-space
    (base + EE) rather than accelerations, which is more intuitive for teleoperation.
    """

    def __init__(self, config):
        """Initialize MPSF controller.

        Args:
            config (dict): Configuration dictionary with MPSF parameters.
        """
        super().__init__(config)
        cost_params = config["cost_params"]

        # Create cost functions - use BaseVel and EEVel (same as MPC)
        costs = []

        # Base velocity tracking cost - tracks base velocity in world frame
        costs.append(
            CostFunctionRegistry.create(
                "BaseVel", self.robot, cost_params.get("BaseVel", {}), dimension=3
            )
        )

        # EE velocity tracking cost - tracks EE velocity (6D: linear + angular)
        costs.append(
            CostFunctionRegistry.create(
                "EEVel", self.robot, cost_params.get("EEVel", {})
            )
        )

        # Add collision costs/constraints (same as MPC)
        constraints = []
        for name in self.collision_link_names:
            is_static = (
                name
                in self.model_interface.scene.collision_link_names["static_obstacles"]
            )
            softened = self.params["collision_constraints_softened"][
                "static_obstacles" if is_static else name
            ]
            if softened:
                costs.append(self.collisionSoftCsts[name])
            else:
                constraints.append(self.collisionCsts[name])

        name = self.params["acados"].get("name", "MPSF")
        # No terminal cost for MPSF (num_terminal_cost = 0)
        self.ocp, self.ocp_solver, self.p_struct = self._construct(
            costs, constraints, num_terminal_cost=0, name=name
        )

        self.cost = costs
        self.constraints = constraints + [self.controlCst, self.stateCst]

        # Get objective_horizon and decay_rate (defaults: full horizon, no decay)
        self.objective_horizon = self.params.get("objective_horizon", self.N + 1)
        self.decay_rate = self.params.get("decay_rate", 1.0)

    def control(
        self,
        t: float,
        robot_states: Tuple[NDArray[np.float64], NDArray[np.float64]],
        references: dict,
    ):
        """Compute control input using MPSF.

        Args:
            t (float): Current control time.
            robot_states (tuple): (q, v) generalized coordinates and velocities.
            references (dict): Dictionary with desired velocity reference:
                {
                    "velocity": array of shape (N+1, 9) or (9,) - desired state-space velocities
                                [v_base (3D), v_ee (6D)] where:
                                - v_base: [vx, vy, vyaw] base velocity in world frame
                                - v_ee: [vx, vy, vz, wx, wy, wz] EE velocity in world frame
                }

        Returns:
            tuple: (v_bar, u_bar) where:
                - v_bar: velocity trajectory, shape (N+1, nu)
                - u_bar: control input trajectory, shape (N, nu)
        """
        self.py_logger.debug(f"control time {t}")
        self.curr_control_time = t
        q, v = robot_states
        q[2:9] = wrap_pi_array(q[2:9])
        xo = np.hstack((q, v))

        x_bar_initial, u_bar_initial = self._prepare_warm_start(t, xo)

        # Extract desired velocity from references
        v_des = references.get("velocity")
        if v_des is None:
            raise ValueError("MPSF requires 'velocity' key in references dictionary")

        # Convert to horizon format if needed
        if v_des.ndim == 1:
            # Single desired velocity - repeat for entire horizon
            v_des_bar = np.tile(v_des, (self.N + 1, 1))
        else:
            # Already in horizon format
            if v_des.shape[0] < self.N + 1:
                # Pad with last value if not enough steps
                last_v = v_des[-1:]
                v_des_bar = np.vstack(
                    [v_des, np.tile(last_v, (self.N + 1 - v_des.shape[0], 1))]
                )
            else:
                v_des_bar = v_des[: self.N + 1]

        curr_p_map_bar = self._setup_horizon_parameters(
            v_des_bar, x_bar_initial, u_bar_initial
        )
        self._solve_and_extract(xo, t, curr_p_map_bar, x_bar_initial, u_bar_initial)
        self._update_logging(curr_p_map_bar)

        velocity_traj = self.x_bar[:, self.DoF :].copy()
        return velocity_traj, self.u_bar.copy()

    def _prepare_warm_start(self, t, xo):
        """Prepare warm start trajectories from previous solution or zeros.

        Args:
            t (float): Current control time.
            xo (ndarray): Current state vector.

        Returns:
            tuple: (x_bar_initial, u_bar_initial) initial guess trajectories.
        """
        if self.t_bar is not None:
            self.u_t = interp1d(
                self.t_bar,
                self.u_bar,
                axis=0,
                bounds_error=False,
                fill_value="extrapolate",
            )
            t_bar_new = t + np.arange(self.N) * self.dt
            self.u_bar = self.u_t(t_bar_new)
            self.x_bar = self._predictTrajectories(xo, self.u_bar)
        else:
            self.u_bar = np.zeros_like(self.u_bar)
            self.x_bar = self._predictTrajectories(xo, self.u_bar)

        return self.x_bar.copy(), self.u_bar.copy()

    def _setup_horizon_parameters(self, v_des_bar, x_bar_initial, u_bar_initial):
        """Setup OCP parameters for each horizon step.

        Args:
            v_des_bar (ndarray): Desired velocity trajectory, shape (N+1, 9) where
                                 each row is [v_base (3D), v_ee (6D)].
            x_bar_initial (ndarray): Initial state trajectory guess, shape (N+1, nx).
            u_bar_initial (ndarray): Initial control trajectory guess, shape (N, nu).

        Returns:
            list: List of parameter maps for each horizon step.
        """
        tp1 = time.perf_counter()
        curr_p_map_bar = []

        # Reset time logging
        for key in self.log.keys():
            if "time" in key:
                self.log[key] = 0

        for i in range(self.N + 1):
            curr_p_map = self.p_struct(0)
            self._set_initial_guess(curr_p_map, i, x_bar_initial, u_bar_initial)
            self._set_velocity_tracking_params(curr_p_map, v_des_bar, i)
            self._set_ocp_params(curr_p_map, i)
            curr_p_map_bar.append(curr_p_map)

        tp2 = time.perf_counter()
        self.log["time_ocp_set_params"] = tp2 - tp1
        return curr_p_map_bar

    def _set_initial_guess(self, curr_p_map, i, x_bar_initial, u_bar_initial):
        """Set initial guess for state, control, and multipliers.

        Args:
            curr_p_map (casadi.struct_MX): Current parameter map (not used, but kept for consistency).
            i (int): Horizon step index.
            x_bar_initial (ndarray): Initial state trajectory guess, shape (N+1, nx).
            u_bar_initial (ndarray): Initial control trajectory guess, shape (N, nu).
        """
        t1 = time.perf_counter()
        self.ocp_solver.set(i, "x", x_bar_initial[i])
        if i < self.N:
            self.ocp_solver.set(i, "u", u_bar_initial[i])
        if self.lam_bar is not None:
            self.ocp_solver.set(i, "lam", self.lam_bar[i])
        t2 = time.perf_counter()
        self.log["time_ocp_set_params_set_x"] += t2 - t1

    def _set_velocity_tracking_params(self, curr_p_map, v_des_bar, i):
        """Set BaseVel and EEVel cost function parameters with time-varying weights.

        The weight matrix W at time step i is computed as:
        - W = diag(weights) * (decay_rate ** i) if i < objective_horizon
        - W = 0 (all zeros) if i >= objective_horizon

        Args:
            curr_p_map (casadi.struct_MX): Current parameter map to update.
            v_des_bar (ndarray): Desired velocity trajectory, shape (N+1, 9) where
                                each row is [v_base (3D), v_ee (6D)].
            i (int): Horizon step index (0-indexed).
        """
        t1 = time.perf_counter()
        p_keys = self.p_struct.keys()

        # Split desired velocity into base and EE components
        v_des = v_des_bar[i] if i < len(v_des_bar) else v_des_bar[-1]
        v_base_des = v_des[:3]  # [vx, vy, vyaw]
        v_ee_des = v_des[3:9]  # [vx, vy, vz, wx, wy, wz]

        # Set BaseVel parameters
        p_name_base_r = "r_BaseVel3"
        p_name_base_W = "W_BaseVel3"
        if p_name_base_r in p_keys:
            curr_p_map[p_name_base_r] = v_base_des
            # Get weights and time-varying parameters from config
            base_vel_params = self.params["cost_params"].get("BaseVel", {})
            weight_key = "P" if i == self.N else "Qk"
            weights = base_vel_params.get(weight_key, [1.0, 1.0, 1.0])

            if i <= self.objective_horizon:
                weight_scale = self.decay_rate**i
                if isinstance(weights, (list, np.ndarray)):
                    curr_p_map[p_name_base_W] = weight_scale * np.diag(weights)
                else:
                    curr_p_map[p_name_base_W] = np.eye(3) * weights * weight_scale
            else:
                curr_p_map[p_name_base_W] = np.zeros((3, 3))

        # Set EEVel parameters
        p_name_ee_r = "r_EEVel6"
        p_name_ee_W = "W_EEVel6"
        if p_name_ee_r in p_keys:
            curr_p_map[p_name_ee_r] = v_ee_des
            # Get weights and time-varying parameters from config
            ee_vel_params = self.params["cost_params"].get("EEVel", {})
            weight_key = "P" if i == self.N else "Qk"
            weights = ee_vel_params.get(weight_key, [1.0] * 6)

            if i <= self.objective_horizon:
                weight_scale = self.decay_rate**i
                if isinstance(weights, (list, np.ndarray)):
                    curr_p_map[p_name_ee_W] = weight_scale * np.diag(weights)
                else:
                    curr_p_map[p_name_ee_W] = np.eye(6) * weights * weight_scale
            else:
                curr_p_map[p_name_ee_W] = np.zeros((6, 6))

        t2 = time.perf_counter()
        self.log["time_ocp_set_params_velocity_tracking"] = self.log.get(
            "time_ocp_set_params_velocity_tracking", 0
        ) + (t2 - t1)

    def _set_ocp_params(self, curr_p_map, i):
        """Set OCP parameters for the current horizon step.

        Args:
            curr_p_map (casadi.struct_MX): Current parameter map.
            i (int): Horizon step index.
        """
        t1 = time.perf_counter()
        self.ocp_solver.set(i, "p", curr_p_map.cat.full().flatten())
        t2 = time.perf_counter()
        self.log["time_ocp_set_params_setp"] = self.log.get(
            "time_ocp_set_params_setp", 0
        ) + (t2 - t1)

    def _solve_and_extract(self, xo, t, curr_p_map_bar, x_bar_initial, u_bar_initial):
        """Solve the OCP and extract solution.

        Args:
            xo (ndarray): Current state vector.
            t (float): Current control time.
            curr_p_map_bar (list): List of parameter maps for each horizon step.
            x_bar_initial (ndarray): Initial state trajectory guess, shape (N+1, nx).
            u_bar_initial (ndarray): Initial control trajectory guess, shape (N, nu).
        """
        t1 = time.perf_counter()
        self.ocp_solver.solve_for_x0(xo, fail_on_nonzero_status=False)
        t2 = time.perf_counter()
        self.log["time_ocp_solve"] = t2 - t1

        self.ocp_solver.print_statistics()
        self.log["solver_status"] = self.ocp_solver.status
        if self.ocp.solver_options.nlp_solver_type != "SQP_RTI":
            self.log["step_size"] = np.mean(self.ocp_solver.get_stats("alpha"))
        else:
            self.log["step_size"] = -1
        self.log["sqp_iter"] = self.ocp_solver.get_stats("sqp_iter")
        self.log["qp_iter"] = sum(self.ocp_solver.get_stats("qp_iter"))
        self.log["cost_final"] = self.ocp_solver.get_cost()

        if self.ocp_solver.status != 0:
            x_bar = [self.ocp_solver.get(i, "x") for i in range(self.N)]
            u_bar = [self.ocp_solver.get(i, "u") for i in range(self.N)]
            x_bar.append(self.ocp_solver.get(self.N, "x"))

            self.log["iter_snapshot"] = {
                "t": t,
                "xo": xo,
                "p_map_bar": [p.cat.full().flatten() for p in curr_p_map_bar],
                "x_bar_init": x_bar_initial,
                "u_bar_init": u_bar_initial,
                "x_bar": x_bar,
                "u_bar": u_bar,
            }

            if self.params["acados"]["raise_exception_on_failure"]:
                raise Exception(
                    f"acados acados_ocp_solver returned status {self.ocp_solver.status}"
                )
        else:
            self.log["iter_snapshot"] = None

        # Extract solution
        self.lam_bar = []
        for i in range(self.N):
            self.x_bar[i, :] = self.ocp_solver.get(i, "x")
            self.u_bar[i, :] = self.ocp_solver.get(i, "u")
            self.lam_bar.append(self.ocp_solver.get(i, "lam"))

        self.x_bar[self.N, :] = self.ocp_solver.get(self.N, "x")
        self.lam_bar.append(self.ocp_solver.get(self.N, "lam"))
        self.t_bar = t + np.arange(self.N) * self.dt
        self.v_cmd = self.x_bar[0][self.robot.DoF :].copy()

    def _update_logging(self, curr_p_map_bar):
        """Update logging and visualization data.

        Args:
            curr_p_map_bar (list): List of parameter maps for each horizon step.
        """
        t1 = time.perf_counter()

        self.ee_bar, self.base_bar = self._getEEBaseTrajectories(self.x_bar)

        for name in self.collision_link_names:
            self.log["_".join([name, "constraint"])] = self.evaluate_constraints(
                self.collisionCsts[name], self.x_bar, self.u_bar, curr_p_map_bar
            )

        self.log["ee_pos"] = self.ee_bar.copy()
        self.log["base_pos"] = self.base_bar.copy()
        self.log["ocp_param"] = [p.cat.full().flatten() for p in curr_p_map_bar]
        self.log["x_bar"] = self.x_bar.copy()
        self.log["u_bar"] = self.u_bar.copy()
        t2 = time.perf_counter()
        self.log["time_ocp_overhead"] = t2 - t1

    def _get_log(self):
        """Get log dictionary structure with default keys.

        Returns:
            dict: Log dictionary with default keys initialized to zero or empty.
        """
        log = {
            "cost_final": 0,
            "step_size": 0,
            "sqp_iter": 0,
            "qp_iter": 0,
            "solver_status": 0,
            "time_ocp_set_params": 0,
            "time_ocp_solve": 0,
            "time_ocp_set_params_set_x": 0,
            "time_ocp_set_params_velocity_tracking": 0,
            "time_ocp_set_params_setp": 0,
            "state_constraint": 0,
            "control_constraint": 0,
            "x_bar": 0,
            "u_bar": 0,
            "lam_bar": 0,
            "ee_pos": 0,
            "base_pos": 0,
            "ocp_param": {},
            "iter_snapshot": {},
        }
        for name in self.collision_link_names:
            log["_".join([name, "constraint"])] = 0
            log["_".join([name, "constraint", "gradient"])] = 0

        return log

    def reset(self):
        """Reset controller state and solver."""
        super().reset()
        self.ocp_solver.reset()
