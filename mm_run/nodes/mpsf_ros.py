#!/usr/bin/env python3

import os
import sys

# Add paths for imports when running as ROS node
# Get the directory containing this file (nodes/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add nodes directory to path for mpc_ros import
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
# Add parent directory (mm_run) to path for scripts import
mm_run_dir = os.path.dirname(current_dir)
if mm_run_dir not in sys.path:
    sys.path.insert(0, mm_run_dir)

import numpy as np  # noqa: E402
import rospy  # noqa: E402

# Import the base controller node (same directory)
from mpc_ros import ControllerROSNode  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

# Import MPSF-specific functions
from scripts.mpsf_experiment import calculate_desired_velocity  # noqa: E402


class MPSFControllerROSNode(ControllerROSNode):
    """ROS node for MPSF (Model Predictive Shared Framework) experiments.

    Inherits from ControllerROSNode and adds MPSF-specific functionality:
    - Calculates desired velocities from MPSF goals
    - Adds desired velocities to controller references
    """

    def __init__(self):
        self.base_goal = None
        self.ee_goal = None
        self._mpsf_goals_initialized = False

        super().__init__()

    def _extract_states_for_mpsf(self, robot_states, use_vicon_tool_data):
        """Extract robot states in the format needed for MPSF calculations.

        Args:
            robot_states (tuple): (q, v) tuple from robot interface.
            use_vicon_tool_data (bool): Whether Vicon tool data is available.

        Returns:
            dict: Dictionary with "base" and "EE" keys, each containing "pose" and "velocity".
        """
        if use_vicon_tool_data:
            ee_pos = self.vicon_tool_interface.position
            ee_quat = self.vicon_tool_interface.orientation
            # For Vicon data, velocity is not directly available, set to zeros
            ee_vel = np.zeros(6)
        else:
            ee_pos, ee_quat = self.robot.getEE(robot_states[0])
            # Compute EE velocity using spatial Jacobian if available
            tool_name = self.robot.tool_link_name
            spatial_jac_key = tool_name + "_spatial"
            if spatial_jac_key in self.robot.jacSymMdls:
                J_spatial = self.robot.jacSymMdls[spatial_jac_key](robot_states[0])
                ee_vel = (J_spatial @ robot_states[1]).toarray().flatten()
            else:
                # Fallback: use position Jacobian and pad with zeros for angular velocity
                J_pos = self.robot.jacSymMdls[tool_name](robot_states[0])
                ee_lin_vel = (J_pos @ robot_states[1]).toarray().flatten()
                ee_vel = np.hstack([ee_lin_vel, np.zeros(3)])

        ee_euler = Rot.from_quat(ee_quat).as_euler("xyz")
        ee_pose = np.hstack([ee_pos, ee_euler])

        base_pose = robot_states[0][:3]  # [x, y, yaw] already in world frame
        base_vel = robot_states[1][:3]  # [vx, vy, vyaw]

        return {
            "base": {"pose": base_pose, "velocity": base_vel},
            "EE": {"pose": ee_pose, "velocity": ee_vel},
        }

    def update_references(self, references, robot_states):
        """Update references for MPSF calculations.

        Args:
            references (dict): References dictionary.
            robot_states (tuple): (q, v) tuple from robot interface.
        """
        if not self._mpsf_goals_initialized:
            mpsf_goals = self.ctrl_config.get("mpsf_params", {})
            if mpsf_goals.get("mpsf_base_goal") is not None:
                self.base_goal = np.array(mpsf_goals.get("mpsf_base_goal"))
            if mpsf_goals.get("mpsf_ee_goal") is not None:
                self.ee_goal = np.array(mpsf_goals.get("mpsf_ee_goal"))
            self._mpsf_goals_initialized = True

            if self.base_goal is None and self.ee_goal is None:
                raise ValueError("MPSF goals not found in config")
            else:
                rospy.loginfo(
                    f"MPSF goals - Base: {self.base_goal}, EE: {self.ee_goal}"
                )

        # MPSF-specific: Calculate desired velocities and add to references
        if self.base_goal is not None or self.ee_goal is not None:
            states = self._extract_states_for_mpsf(
                robot_states, self.use_vicon_tool_data
            )
            desired_base_vel, desired_ee_vel = calculate_desired_velocity(
                self.base_goal, self.ee_goal, states, self.controller
            )

            # Add desired velocities to references if not None
            if desired_base_vel is not None or desired_ee_vel is not None:
                desired_velocity = {}
                if desired_base_vel is not None:
                    desired_velocity["base_velocity"] = desired_base_vel
                if desired_ee_vel is not None:
                    desired_velocity["ee_velocity"] = desired_ee_vel
                references["desired_velocity"] = desired_velocity


if __name__ == "__main__":
    rospy.init_node("controller_ros_mpsf")

    node = MPSFControllerROSNode()
    node.run()
