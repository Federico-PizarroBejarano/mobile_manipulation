import argparse
import copy
import logging
import os
import sys
import threading
import time

import numpy as np
import rospy
import tf.transformations as tf
from geometry_msgs.msg import Point, PoseStamped, Quaternion, Transform, Twist
from mobile_manipulation_central import PointToPointTrajectory, bound_array
from mobile_manipulation_central.ros_interface import (
    JoystickButtonInterface,
    MobileManipulatorROSInterface,
    ViconObjectInterface,
)
from nav_msgs.msg import Odometry, Path
from robotiq_3f_gripper_articulated_msgs.msg import (
    Robotiq3FGripperRobotInput,
    Robotiq3FGripperRobotOutput,
)
from scipy.spatial.transform import Rotation as Rot
from sensor_msgs.msg import Joy
from spatialmath.base import r2q, rpy2r
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

import mm_control.MPC as MPC
from mm_control.robot import MobileManipulator3D
from mm_plan.TaskManager import TaskManager
from mm_run.msg import MpcPlan
from mm_utils import parsing
from mm_utils.base_velocity_guard import sanitize_base_velocity
from mm_utils.enums import RefType
from mm_utils.logging import DataLogger
from mm_utils.math import wrap_pi_scalar
from mm_utils.robotiq_gripper import (
    GRIPPER_TOGGLE_BUTTON,
    gripper_position,
    toggle_gripper_open,
)
from mm_utils.teleop_session_logging import (
    TrialBagRecorder,
    append_teleop_sample,
    clear_experiment_timestamp,
    resolve_experiment_timestamp,
    session_root,
)


class ControllerROSNode:
    def __init__(self):
        np.set_printoptions(precision=3, suppress=True)
        # Initialize controller state parameters early to prevent simulation from exiting
        # This must be done before any operations that might take time
        rospy.set_param("/controller_finished", False)
        rospy.set_param("/controller_started", False)

        argv = rospy.myargv(argv=sys.argv)
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-c", "--config", required=True, help="Path to configuration file."
        )
        parser.add_argument(
            "--ctrl_config",
            type=str,
            help="controller config. This overwrites the yaml settings in config if not set to default",
        )
        parser.add_argument(
            "--planner_config",
            type=str,
            help="planner config. This overwrites the yaml settings in config if not set to default",
        )
        parser.add_argument(
            "--logging_sub_folder",
            type=str,
            help="save data in a sub folder of logging directory",
        )
        args = parser.parse_args(argv[1:])

        # load configuration and overwrite with args
        config = parsing.load_config(args.config)
        if args.ctrl_config != "default":
            ctrl_config = parsing.load_config(args.ctrl_config)
            config = parsing.recursive_dict_update(config, ctrl_config)
        if args.planner_config != "default":
            planner_config = parsing.load_config(args.planner_config)
            config = parsing.recursive_dict_update(config, planner_config)

        if args.logging_sub_folder != "default":
            config["logging"]["log_dir"] = os.path.join(
                config["logging"]["log_dir"], args.logging_sub_folder
            )

        self.ctrl_config = config["controller"]
        self.simulation_timestep = float(config["simulation"]["timestep"])
        self.planner_config = config.get("planner", {}).copy()
        print(self.ctrl_config["type"])
        # controller
        control_class = getattr(MPC, self.ctrl_config["type"], None)
        if control_class is None:
            raise ValueError(f"Unknown controller type: {self.ctrl_config['type']}")

        self.controller = control_class(self.ctrl_config)

        self._mpc_loop_hz = float(self.ctrl_config["ctrl_rate"])
        if self._mpc_loop_hz <= 0.0:
            raise ValueError(f"ctrl_rate must be positive, got {self._mpc_loop_hz}")
        # set py logger level
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        self.planner_log = logging.getLogger("Planner")
        self.planner_log.setLevel(config["logging"]["log_level"])
        self.planner_log.addHandler(ch)
        self.controller_log = logging.getLogger("Controller")
        self.controller_log.setLevel(config["logging"]["log_level"])
        self.controller_log.addHandler(ch)

        # init logger
        self.logger = DataLogger(copy.deepcopy(config), name="control")

        self.logger.add("sim_timestep", config["simulation"]["timestep"])
        self.logger.add("duration", config["simulation"]["duration"])
        self.sim_duration = float(config["simulation"]["duration"])

        self.logger.add("nq", self.ctrl_config["robot"]["dims"]["q"])
        self.logger.add("nv", self.ctrl_config["robot"]["dims"]["v"])
        self.logger.add("nx", self.ctrl_config["robot"]["dims"]["x"])
        self.logger.add("nu", self.ctrl_config["robot"]["dims"]["u"])
        self._nu_cmd = int(self.ctrl_config["robot"]["dims"]["u"])

        # Shared timestamp with sim_ros when present. Never overwrite an existing
        # param (avoids split sim/ vs control/ folders).
        self.session_timestamp = resolve_experiment_timestamp(create=True)

        record_bag = rospy.get_param(
            "~record_bag", rospy.get_param("/mm_run/record_bag", False)
        )
        bag_all = rospy.get_param("~bag_all", rospy.get_param("/mm_run/bag_all", False))
        self.bag_recorder = TrialBagRecorder(
            session_root(self.logger.base_directory, self.session_timestamp),
            record_bag=record_bag,
            bag_all=bag_all,
            tool_vicon_name=self.ctrl_config["robot"]["tool_vicon_name"],
            plan_topic=rospy.resolve_name("mpc_plan"),
        )

        # ROS Related
        self.robot_interface = MobileManipulatorROSInterface()
        self.vicon_tool_interface = ViconObjectInterface(
            self.ctrl_config["robot"]["tool_vicon_name"]
        )

        self.start_end_button_interface = JoystickButtonInterface(2)  # square
        # Triangle/Y: raw rising-edge (JoystickButtonInterface re-fires while held).
        self._gripper_btn = 0
        self._gripper_btn_prev = 0
        self._gripper_open = True  # software latch; first press closes if HW is open
        self._gripper_no_sub_warned = False
        self._gripper_status = None
        # Persistent output — open/close only mutates rPRA (avoids re-activation dance).
        self._gripper_cmd = None
        self._gripper_pub = rospy.Publisher(
            "/Robotiq3FGripperRobotOutput",
            Robotiq3FGripperRobotOutput,
            queue_size=1,
        )
        rospy.Subscriber(
            "/bluetooth_teleop/joy", Joy, self._gripper_joy_cb, queue_size=1
        )
        rospy.Subscriber(
            "/Robotiq3FGripperRobotInput",
            Robotiq3FGripperRobotInput,
            self._gripper_status_cb,
            queue_size=1,
        )

        self.teleop_enabled = False

        mi = self.controller.model_interface
        self.self_collision_func = mi.getSignedDistanceSymMdls("self")
        self.ground_collision_func = mi.getSignedDistanceSymMdls("ground")
        self._estop_margin = float(self.ctrl_config["collision_estop_margin"])
        self._viz_enabled = bool(
            self.ctrl_config.get("ros_visualization_enabled", True)
        )
        self._viz_rate = float(self.ctrl_config.get("ros_visualization_rate", 5.0))
        self._use_sim_time = bool(rospy.get_param("/use_sim_time", False))

        self.controller_visualization_pub = rospy.Publisher(
            "controller_visualization", Marker, queue_size=10
        )
        self.controller_visualization_array_pub = rospy.Publisher(
            "controller_visualization_array", MarkerArray, queue_size=10
        )
        self.plan_visualization_pub = rospy.Publisher(
            "plan_visualization", Marker, queue_size=10
        )
        self.pose_plan_visualization_pub = rospy.Publisher(
            "pose_plan_visualization", PoseStamped, queue_size=10
        )

        self.current_plan_visualization_pub = rospy.Publisher(
            "current_plan_visualization", Marker, queue_size=10
        )
        self.controller_ref_pub = rospy.Publisher(
            "controller_reference", Path, queue_size=5
        )

        self.tracking_point_pub = rospy.Publisher(
            "controller_tracking_pt", MultiDOFJointTrajectory, queue_size=5
        )

        # High-rate cmd_vel is published by low_level_cmd_node (see controller.launch).
        self.mpc_plan_pub = rospy.Publisher("mpc_plan", MpcPlan, queue_size=1)

        self.sot_lock = threading.Lock()

        guard_cfg = self.ctrl_config.get("base_velocity_guard", {})
        self._base_vel_guard_enabled = bool(guard_cfg.get("enabled", False))
        self._base_vel_guard_max_disagreement = float(
            guard_cfg.get("max_disagreement", 0.25)
        )
        self._base_vel_guard_stale_s = float(guard_cfg.get("stale_s", 0.2))
        self._odom_twist_body = None
        self._odom_stamp = None
        if self._base_vel_guard_enabled:
            odom_topic = str(guard_cfg.get("odom_topic", "/odometry/filtered"))
            rospy.Subscriber(
                odom_topic,
                Odometry,
                self._on_odometry_for_guard,
                queue_size=1,
            )
            self.controller_log.info(
                "Base velocity guard enabled (topic=%s, max_disagreement=%.2f)",
                odom_topic,
                self._base_vel_guard_max_disagreement,
            )

        rospy.on_shutdown(self.shutdownhook)
        self.ctrl_c = False

    def shutdownhook(self):
        self.ctrl_c = True
        self.robot_interface.brake()
        self.bag_recorder.stop()
        self.logger.save(session_timestamp=self.session_timestamp)
        clear_experiment_timestamp()

    def _sync_session_timestamp(self):
        """Re-read shared stamp (sim may set it after our __init__)."""
        ts = resolve_experiment_timestamp(create=True)
        if ts == self.session_timestamp:
            return
        rospy.loginfo(
            "Adopting experiment timestamp %s (was %s)", ts, self.session_timestamp
        )
        self.session_timestamp = ts
        self.bag_recorder.session_root = session_root(
            self.logger.base_directory, self.session_timestamp
        )

    def _on_odometry_for_guard(self, msg: Odometry):
        tw = msg.twist.twist
        self._odom_twist_body = np.array(
            [tw.linear.x, tw.linear.y, tw.angular.z], dtype=float
        )
        self._odom_stamp = msg.header.stamp

    def _apply_base_velocity_guard(self, q, v):
        if not self._base_vel_guard_enabled or self._odom_twist_body is None:
            return q, v

        if self._odom_stamp is None:
            return q, v

        age = (rospy.Time.now() - self._odom_stamp).to_sec()
        if age > self._base_vel_guard_stale_s:
            return q, v

        v_safe, replaced = sanitize_base_velocity(
            q,
            v,
            self._odom_twist_body,
            self._base_vel_guard_max_disagreement,
        )
        if replaced:
            rospy.logwarn_throttle(
                1.0,
                "Base velocity guard: replaced joint_states base vel "
                "(disagreement > %.2f m/s)",
                self._base_vel_guard_max_disagreement,
            )
        return q, v_safe

    def _publish_mpc_plan(self, t, u_bar, v_bar):
        """Publish MPC plan for an external low-level velocity node."""
        dof = int(self.controller.DoF)
        msg = MpcPlan()
        msg.header.stamp = rospy.Time.from_sec(t)
        msg.mpc_dt = float(self.controller.dt)
        msg.N = int(u_bar.shape[0])
        msg.nu = int(u_bar.shape[1])
        msg.dof = dof
        msg.u_flat = np.asarray(u_bar, dtype=float).ravel(order="C").tolist()
        msg.v_flat = np.asarray(v_bar, dtype=float).ravel(order="C").tolist()
        msg.q_flat = (
            np.asarray(self.controller.x_bar, dtype=float)[:, :dof]
            .ravel(order="C")
            .tolist()
        )
        self.mpc_plan_pub.publish(msg)

    def _publish_brake_plan(self, t):
        """Publish a zero plan so low_level_cmd_node stops replaying old cmds.

        ``robot_interface.brake()`` alone is not enough: the low-level node keeps
        publishing from the last ``MpcPlan`` and overwrites the brake.
        """
        nu = int(self.controller.nu)
        dof = int(self.controller.DoF)
        N = 1
        u_bar = np.zeros((N, nu))
        v_bar = np.zeros((N + 1, nu))
        q = np.asarray(self.robot_interface.q, dtype=float).reshape(-1)[:dof]
        q_bar = np.tile(q, (N + 1, 1))
        msg = MpcPlan()
        msg.header.stamp = rospy.Time.from_sec(t)
        msg.mpc_dt = float(self.controller.dt)
        msg.N = N
        msg.nu = nu
        msg.dof = dof
        msg.u_flat = u_bar.ravel(order="C").tolist()
        msg.v_flat = v_bar.ravel(order="C").tolist()
        msg.q_flat = q_bar.ravel(order="C").tolist()
        self.mpc_plan_pub.publish(msg)

    def _brake(self, t):
        """Stop the robot: zero the plan first, then brake the interface."""
        self._publish_brake_plan(t)
        self.robot_interface.brake()

    def _publish_trajectory_tracking_pt(self, t, robot_states, planner):
        msg = MultiDOFJointTrajectory()
        msg.header.stamp = rospy.Time.now()

        # Return early if no planner
        if planner is None:
            return

        # Get base reference if available
        if planner.has_base_ref:
            p, v = planner.getBaseTrackingPoint(t, robot_states)
            if p is not None:
                msg.joint_names.append("base")
                pt_msg = MultiDOFJointTrajectoryPoint()
                transform = Transform()
                transform.translation.x = p[0]
                transform.translation.y = p[1]
                transform.translation.z = 0.25  # Display height
                quat = tf.quaternion_from_euler(0, 0, p[2])
                transform.rotation.x = quat[0]
                transform.rotation.y = quat[1]
                transform.rotation.z = quat[2]
                transform.rotation.w = quat[3]
                pt_msg.transforms.append(transform)
                if v is not None:
                    velocity = Twist()
                    velocity.linear.x = v[0]
                    velocity.linear.y = v[1]
                    velocity.angular.z = v[2]
                    pt_msg.velocities.append(velocity)
                msg.points.append(pt_msg)

        # Get EE reference if available
        if planner.has_ee_ref:
            p, v = planner.getEETrackingPoint(t, robot_states)
            if p is not None:
                msg.joint_names.append("EE")
                pt_msg = MultiDOFJointTrajectoryPoint()
                transform = Transform()
                transform.translation.x = p[0]
                transform.translation.y = p[1]
                transform.translation.z = p[2]
                quat = tf.quaternion_from_euler(*p[3:])
                transform.rotation.x = quat[0]
                transform.rotation.y = quat[1]
                transform.rotation.z = quat[2]
                transform.rotation.w = quat[3]
                pt_msg.transforms.append(transform)
                if v is not None:
                    velocity = Twist()
                    velocity.linear.x = v[0]
                    velocity.linear.y = v[1]
                    velocity.linear.z = v[2]
                    velocity.angular.x = v[3] if len(v) > 3 else 0
                    velocity.angular.y = v[4] if len(v) > 4 else 0
                    velocity.angular.z = v[5] if len(v) > 5 else 0
                    pt_msg.velocities.append(velocity)
                msg.points.append(pt_msg)

        if len(msg.points) > 0:
            self.tracking_point_pub.publish(msg)

    def _publish_controller_reference(self, ref_pose, ref_velocity):
        # send reference poses as a path
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "world"
        for i in range(len(ref_pose)):
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = "world"
            pose.pose.position.x = ref_pose[i][0]
            pose.pose.position.y = ref_pose[i][1]
            quat = tf.quaternion_from_euler(0, 0, ref_pose[i][2])
            pose.pose.orientation.x = 0
            pose.pose.orientation.y = 0
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            path_msg.poses.append(pose)
        self.controller_ref_pub.publish(path_msg)

    def _make_marker(self, marker_type, id, rgba, scale):
        # make a visualization marker array for the occupancy grid
        m = Marker()
        m.header.frame_id = "world"
        m.header.stamp = rospy.Time.now()
        m.id = id
        m.type = marker_type
        m.action = Marker.ADD

        m.scale.x = scale[0]
        m.scale.y = scale[1]
        m.scale.z = scale[2]
        m.color.r = rgba[0]
        m.color.g = rgba[1]
        m.color.b = rgba[2]
        m.color.a = rgba[3]
        m.lifetime = rospy.Duration.from_sec(1.0 / self._mpc_loop_hz)

        m.pose.orientation.w = 1

        return m

    def _end_mpc_cycle(self, cycle_t0, period, rate):
        """Pad the cycle to ``period``: sim time via ``rate``, else leftover wall time."""
        work_s = time.perf_counter() - cycle_t0
        if self._use_sim_time:
            rate.sleep()
        else:
            leftover = float(period) - work_s
            if leftover > 0.0:
                time.sleep(leftover)
        cycle_s = time.perf_counter() - cycle_t0
        self.controller_log.log(
            20, f"Controller Cycle Work: {work_s:.6f} Period: {cycle_s:.6f}"
        )
        self.logger.append("controller_cycle_work", work_s)
        self.logger.append("controller_cycle_period", cycle_s)

    def _publish_planner_data(self, event):
        self.sot_lock.acquire()
        for pid, planner in enumerate(self.sot.planners):
            color = [0] * 3
            color[pid % 3] = 1

            # Visualize base waypoint/path if available
            if planner.has_base_ref:
                if planner.ref_type == RefType.WAYPOINT:
                    quat = tf.quaternion_from_euler(0, 0, planner.base_target[2])
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = rospy.Time()
                    pose_msg.pose.position = Point(*planner.base_target[:2], 0.25)
                    pose_msg.pose.orientation = Quaternion(*list(quat))
                    self.pose_plan_visualization_pub.publish(pose_msg)
                elif planner.ref_type == RefType.PATH:
                    marker_plan = self._make_marker(
                        Marker.POINTS, pid, rgba=color + [1], scale=[0.1, 0.1, 0.1]
                    )
                    marker_plan.points = [
                        Point(*pt[:2], 0) for pt in planner.base_plan["p"]
                    ]
                    marker_plan.lifetime = rospy.Duration.from_sec(0.1)
                    self.plan_visualization_pub.publish(marker_plan)

            # Visualize EE waypoint/path if available
            if planner.has_ee_ref:
                if planner.ref_type == RefType.WAYPOINT:
                    quat = tf.quaternion_from_euler(*planner.ee_target[3:])
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = rospy.Time()
                    pose_msg.header.frame_id = "world"
                    pose_msg.pose.position = Point(*planner.ee_target[:3])
                    pose_msg.pose.orientation = Quaternion(*list(quat))
                    self.pose_plan_visualization_pub.publish(pose_msg)
                elif planner.ref_type == RefType.PATH:
                    marker_plan = self._make_marker(
                        Marker.POINTS,
                        pid + 100,
                        rgba=color + [1],
                        scale=[0.1, 0.1, 0.1],
                    )
                    marker_plan.points = [Point(*pt[:3]) for pt in planner.ee_plan["p"]]
                    marker_plan.lifetime = rospy.Duration.from_sec(0.1)
                    self.plan_visualization_pub.publish(marker_plan)

        planner = self.sot.getPlanner()
        # Visualize current base waypoint/path if available
        if planner is not None and planner.has_base_ref:
            if planner.ref_type == RefType.WAYPOINT:
                quat = tf.quaternion_from_euler(0, 0, planner.base_target[2])
                pose_msg = PoseStamped()
                pose_msg.header.stamp = rospy.Time()
                pose_msg.pose.position = Point(*planner.base_target[:2], 0.25)
                pose_msg.pose.orientation = Quaternion(*list(quat))
                self.pose_plan_visualization_pub.publish(pose_msg)
            elif planner.ref_type == RefType.PATH:
                marker_plan = self._make_marker(
                    Marker.POINTS,
                    0,
                    rgba=[1, 0, 0, 1],
                    scale=[0.1, 0.1, 0.1],
                )
                marker_plan.points = [
                    Point(*pt[:2], 0) for pt in planner.base_plan["p"]
                ]
                marker_plan.lifetime = rospy.Duration.from_sec(0.1)
                self.current_plan_visualization_pub.publish(marker_plan)

        # Visualize current EE waypoint/path if available
        if planner is not None and planner.has_ee_ref:
            if planner.ref_type == RefType.WAYPOINT:
                quat = tf.quaternion_from_euler(*planner.ee_target[3:])
                pose_msg = PoseStamped()
                pose_msg.header.frame_id = "world"
                pose_msg.header.stamp = rospy.Time()
                pose_msg.pose.position = Point(*planner.ee_target[:3])
                pose_msg.pose.orientation = Quaternion(*list(quat))
                self.pose_plan_visualization_pub.publish(pose_msg)
            elif planner.ref_type == RefType.PATH:
                marker_plan = self._make_marker(
                    Marker.POINTS,
                    100,
                    rgba=[0, 1, 0, 1],
                    scale=[0.1, 0.1, 0.1],
                )
                marker_plan.points = [Point(*pt[:3]) for pt in planner.ee_plan["p"]]
                marker_plan.lifetime = rospy.Duration.from_sec(0.1)
                self.current_plan_visualization_pub.publish(marker_plan)

        self.sot_lock.release()

    def _publish_mpc_data(self, controller):
        # ee prediction
        marker_ee = self._make_marker(
            Marker.POINTS, 0, rgba=[1.0, 1.0, 1.0, 1], scale=[0.1, 0.1, 0.1]
        )
        marker_ee.points = [Point(*pt) for pt in controller.ee_bar]
        self.controller_visualization_pub.publish(marker_ee)

        # base prediction
        marker_base = self._make_marker(
            Marker.POINTS, 1, rgba=[1.0, 1.0, 1.0, 1], scale=[0.1, 0.1, 0.1]
        )
        marker_base.points = [Point(*pt[:2], 0) for pt in controller.base_bar]
        self.controller_visualization_pub.publish(marker_base)

        # ee tracking points
        if len(controller.ree_bar) > 0 and len(controller.ree_bar[0]) == 3:
            marker_ree = self._make_marker(
                Marker.POINTS, 2, rgba=[0.0, 1.0, 1.0, 1], scale=[0.1, 0.1, 0.1]
            )
            marker_ree.points = [Point(*pt[:3]) for pt in controller.ree_bar]
            self.controller_visualization_pub.publish(marker_ree)

        # base tracking points
        marker_rbase = self._make_marker(
            Marker.POINTS, 3, rgba=[0.0, 0.0, 1, 1], scale=[0.1] * 3
        )
        marker_rbase.points = [Point(*pt[:2], 0) for pt in controller.rbase_bar]
        self.controller_visualization_pub.publish(marker_rbase)

    def _joint_state_wait_reason(self):
        """Why robot_interface.ready() is still false, for logs and low-level."""
        published = {name for name, _ in rospy.get_published_topics()}
        missing = []
        if not self.robot_interface.base.ready():
            missing.append("/ridgeback/joint_states")
        if not self.robot_interface.arm.ready():
            missing.append("/ur10/joint_states")
        details = []
        for topic in missing:
            if topic in published:
                details.append(f"{topic} is published but no usable message received")
            else:
                details.append(
                    f"{topic} has no publisher (thing.launch / vicon / UR10?)"
                )
        return "; ".join(details)

    def run(self):
        rate = rospy.Rate(self._mpc_loop_hz)

        print("-----Checking Robot Interface-----")
        robot_wait_t0 = 0.0
        while not self.robot_interface.ready():
            self.robot_interface.brake()
            if time.perf_counter() - robot_wait_t0 > 5.0:
                reason = self._joint_state_wait_reason()
                rospy.set_param("/controller_wait_reason", reason)
                rospy.loginfo("Waiting for joint states: %s", reason)
                robot_wait_t0 = time.perf_counter()
            rate.sleep()

            if rospy.is_shutdown():
                return
        rospy.set_param("/controller_wait_reason", "")
        print("Controller received joint states. Proceed ... ")
        self.home = self.robot_interface.q

        states = (self.robot_interface.q, self.robot_interface.v)
        print(f"robot coord: {self.robot_interface.q}")
        self.sot = TaskManager(
            self.planner_config if self.planner_config.get("tasks") else None
        )

        print("-----Checking Planners----- ")
        for planner in self.sot.planners:
            while not planner.ready():
                self.robot_interface.brake()
                rate.sleep()

                if rospy.is_shutdown():
                    return
            # Print planner targets
            targets = []
            if planner.has_base_ref:
                if planner.ref_type == RefType.WAYPOINT:
                    targets.append(f"base: {planner.base_target}")
                else:
                    targets.append(
                        f"base: path with {len(planner.base_plan['p'])} points"
                    )
            if planner.has_ee_ref:
                if planner.ref_type == RefType.WAYPOINT:
                    targets.append(f"EE: {planner.ee_target}")
                else:
                    targets.append(f"EE: path with {len(planner.ee_plan['p'])} points")
            print(f"planner {planner.name} targets: {', '.join(targets)}")

        print("-----Checking Vicon Tool messages----- ")
        self.use_vicon_tool_data = True
        if not self.vicon_tool_interface.ready():
            self.use_vicon_tool_data = False
            print(
                "Controller did not receive vicon tool "
                + self.ctrl_config["robot"]["tool_vicon_name"]
                + ". Using Robot Model"
            )
            self.robot = MobileManipulator3D(self.ctrl_config)
        else:
            print(
                "Controller received vicon tool "
                + self.ctrl_config["robot"]["tool_vicon_name"]
            )

        print("-----Checking Joy stick messages----- ")
        if self.start_end_button_interface.ready():
            print("Received joystick msg on /bluetooth_teleop/joy.")
        else:
            print("No joystick msg yet (Square/Triangle optional if it appears later).")

        if self._viz_enabled and self._viz_rate > 0.0:
            viz_period = 1.0 / self._viz_rate
            rospy.Timer(rospy.Duration.from_sec(viz_period), self._publish_planner_data)

        self._wait_for_start(rate)
        self._sync_session_timestamp()

        self.bag_recorder.start()
        if self.bag_recorder.last_error is not None:
            rospy.logwarn(
                "rosbag record failed to start: %s", self.bag_recorder.last_error
            )

        self.sot.activatePlanners()
        t = rospy.Time.now().to_sec()
        t0 = t
        self.sot.started = True

        # Signal that controller has started (for simulation timing)
        rospy.set_param("/controller_started", True)
        rospy.set_param("/controller_finished", False)
        self.controller_log.info(
            "Controller started (session %s)", self.session_timestamp
        )

        mpc_period = 1.0 / self._mpc_loop_hz
        previous_cycle_t0 = time.perf_counter()
        while not self.ctrl_c:
            cycle_t0 = time.perf_counter()
            cycle_period = cycle_t0 - previous_cycle_t0
            previous_cycle_t0 = cycle_t0
            t = rospy.Time.now().to_sec()

            # Match experiment.py: one line per control tick (MPC + ROS overhead).
            print(
                f"-------------- {(t - t0):.3f}s/{float(self.sim_duration):.3f}s ------------------"
            )

            # open-loop command
            q_raw, v_raw = self.robot_interface.q, self.robot_interface.v
            q, v = self._apply_base_velocity_guard(q_raw, v_raw)
            robot_states = (q, v)

            # Emergency stop. Braking cannot restore clearance, so this is
            # terminal: continuing would re-trip every cycle forever.
            q = robot_states[0]
            sd_self = float(np.min(self.self_collision_func(q).full()))
            sd_ground = float(np.min(self.ground_collision_func(q).full()))

            if sd_self < self._estop_margin or sd_ground < self._estop_margin:
                self.controller_log.error(
                    "Collision E-stop (self sd_min=%.4f, ground sd_min=%.4f, "
                    "margin=%.4f). Braking, then shutting down.",
                    sd_self,
                    sd_ground,
                    self._estop_margin,
                )
                self._brake(t)
                # Let the zero plan and brake reach the low-level node before exit.
                time.sleep(2.0 * mpc_period)
                raise RuntimeError(
                    f"collision E-stop: self sd_min={sd_self:.4f}, "
                    f"ground sd_min={sd_ground:.4f}, margin={self._estop_margin:.4f}"
                )

            # Get references from TaskManager
            self.sot_lock.acquire()
            references = self.sot.getReferences(
                t - t0, robot_states, self.controller.N + 1, self.controller.dt
            )
            self.sot_lock.release()

            self.update_references(references, robot_states)

            tc1 = time.perf_counter()
            try:
                v_bar, u_bar = self.controller.control(t - t0, robot_states, references)
            except Exception:
                self.controller_log.exception(
                    "MPC solve failed. Braking, then shutting down."
                )
                self._brake(t)
                # Let the zero plan and brake reach the low-level node before exit.
                time.sleep(2.0 * mpc_period)
                raise
            tc2 = time.perf_counter()
            self.controller_log.log(20, f"Controller Run Time: {tc2 - tc1}")

            # if the robot is very close to the goal, stop the robot
            # Check if any active planner is close to finish
            self.sot_lock.acquire()
            planner = self.sot.getPlanner()
            close_to_goal = planner.closeToFinish() if planner is not None else False
            self.sot_lock.release()
            if close_to_goal:
                print("Close to goal. Braking")
                v_bar[:, :3] = 0

            self._publish_mpc_plan(t, u_bar, v_bar)

            if self._viz_enabled:
                self._publish_mpc_data(self.controller)
                self.sot_lock.acquire()
                active_planner = self.sot.getPlanner()
                self.sot_lock.release()
                self._publish_trajectory_tracking_pt(
                    t - t0, robot_states, active_planner
                )

            # Update Task Manager
            # Convert to pose arrays in world frame
            tool_name = self.robot.tool_link_name
            spatial_jac_key = tool_name + "_spatial"
            if spatial_jac_key in self.robot.jacSymMdls:
                J_spatial = self.robot.jacSymMdls[spatial_jac_key](robot_states[0])
                ee_vel = (J_spatial @ robot_states[1]).toarray().flatten()
            else:
                # Fallback: use position Jacobian and pad angular velocity with zeros.
                J_pos = self.robot.jacSymMdls[tool_name](robot_states[0])
                ee_lin_vel = (J_pos @ robot_states[1]).toarray().flatten()
                ee_vel = np.hstack([ee_lin_vel, np.zeros(3)])

            if self.use_vicon_tool_data:
                ee_pos = self.vicon_tool_interface.position
                ee_quat = self.vicon_tool_interface.orientation
            else:
                ee_pos, ee_quat = self.robot.getEE(robot_states[0])

            ee_euler = Rot.from_quat(ee_quat).as_euler("xyz")
            ee_pose = np.hstack([ee_pos, ee_euler])

            base_pose = robot_states[0][:3]  # [x, y, yaw] already in world frame
            base_vel = robot_states[1][:3]  # [vx, vy, vyaw]

            states = {
                "base": {"pose": base_pose, "velocity": base_vel},
                "EE": {"pose": ee_pose, "velocity": ee_vel},
            }

            self.sot_lock.acquire()
            self.sot.update(
                t - t0,
                states,
                base_mask=self.controller.base_mask,
                ee_mask=self.controller.ee_mask,
            )
            # Check if all tasks are finished
            all_finished = (
                self.sot.planner_num > 0
                and self.sot.curr_task_id >= self.sot.planner_num - 1
                and self.sot.planners[self.sot.curr_task_id].finished
            )
            self.sot_lock.release()

            # Signal that controller has finished all tasks
            if all_finished:
                rospy.set_param("/controller_finished", True)

            # Hook / MPSF metrics: MPC preview velocity at first horizon knot (same row
            # as in MpcPlan). low_level_cmd_node may integrate/interpolate between solves.
            u_for_hook = (
                np.asarray(v_bar[0], dtype=float).reshape(-1)[: self._nu_cmd].copy()
            )
            self._after_control_step(
                t - t0,
                robot_states,
                states,
                references,
                u_for_hook,
            )

            # log
            teleop_fields = self._teleop_log_fields()
            append_teleop_sample(
                self.logger,
                ts=t,
                q=robot_states[0],
                v=robot_states[1],
                base_pose=states["base"]["pose"],
                base_vel=states["base"]["velocity"],
                ee_pose=states["EE"]["pose"],
                ee_vel=states["EE"]["velocity"],
                u_cmd=u_for_hook,
                cycle_period=cycle_period,
                **teleop_fields,
            )
            self.log_mpc_info(self.logger, self.controller)
            self.logger.append("controller_run_time", tc2 - tc1)
            r_ew_wd = None
            r_bw_wd = None
            v_ew_wd = None
            v_bw_wd = None
            Q_we_d = None
            ω_ew_wd = None

            # Extract reference data from references dictionary for logging
            if references.get("ee_pose") is not None:
                ee_ref = references["ee_pose"][0]  # Current reference
                r_ew_wd = ee_ref[:3]
                Q_we_d = r2q(rpy2r(ee_ref[3:]), order="xyzs")
                if references.get("ee_velocity") is not None:
                    ee_vel_ref = references["ee_velocity"][0]
                    v_ew_wd = ee_vel_ref[:3]
                    ω_ew_wd = ee_vel_ref[3:]

            if references.get("base_pose") is not None:
                base_ref = references["base_pose"][0]  # Current reference
                r_bw_wd = base_ref
                if references.get("base_velocity") is not None:
                    v_bw_wd = references["base_velocity"][0]
            if r_ew_wd is not None:
                self.logger.append("r_ew_w_ds", r_ew_wd)
            if v_ew_wd is not None:
                self.logger.append("v_ew_w_ds", v_ew_wd)
            if Q_we_d is not None:
                self.logger.append("Q_we_ds", Q_we_d)
            if ω_ew_wd is not None:
                self.logger.append("ω_ew_wds", ω_ew_wd)

            if r_bw_wd is not None:
                if r_bw_wd.shape[0] == 2:
                    self.logger.append("r_bw_w_ds", r_bw_wd)

                elif r_bw_wd.shape[0] == 3:
                    self.logger.append("r_bw_w_ds", r_bw_wd[:2])
                    self.logger.append("yaw_bw_w_ds", r_bw_wd[2])
            if v_bw_wd is not None:
                if v_bw_wd.shape[0] == 2:
                    self.logger.append("v_bw_w_ds", v_bw_wd)
                elif v_bw_wd.shape[0] == 3:
                    self.logger.append("v_bw_w_ds", v_bw_wd[:2])
                    self.logger.append("ω_bw_w_ds", v_bw_wd[2])

            self._poll_gripper_toggle()
            if self.start_end_button_interface.button == 1:
                self.start_end_button_interface.reset_button()
                break

            self._end_mpc_cycle(cycle_t0, mpc_period, rate)

        self.bag_recorder.stop()

    def _wait_for_start(self, rate):
        """Start on Square (controller) and/or Enter when a TTY is available.

        Non-interactive (no TTY): wait for Square if joy is already publishing,
        otherwise start immediately so CI/sim without a pad does not hang.
        """
        self.start_end_button_interface.reset_button()
        enter_pressed = threading.Event()

        if sys.stdin.isatty():
            print("----- Press Square (controller) or Enter to start -----")

            def _wait_enter():
                try:
                    input()
                    enter_pressed.set()
                except EOFError:
                    pass

            threading.Thread(target=_wait_enter, daemon=True).start()
        elif self.start_end_button_interface.ready():
            print("----- Press Square (controller) to start -----")
        else:
            print("----- Non-interactive start: skipping prompt -----")
            return

        while not self.ctrl_c and not rospy.is_shutdown():
            if self.start_end_button_interface.button == 1:
                self.start_end_button_interface.reset_button()
                return
            if enter_pressed.is_set():
                return
            rospy.loginfo_throttle(
                5.0,
                "Waiting for Square (button 2) or Enter to start ...",
            )
            rate.sleep()

        raise rospy.ROSInterruptException("shutdown while waiting to start")

    def _gripper_joy_cb(self, msg):
        if GRIPPER_TOGGLE_BUTTON < len(msg.buttons):
            self._gripper_btn = int(msg.buttons[GRIPPER_TOGGLE_BUTTON])

    def _gripper_status_cb(self, msg):
        self._gripper_status = msg

    def _ensure_gripper_cmd(self):
        """Keep one Robotiq output message; only rPRA changes on later toggles."""
        if self._gripper_cmd is not None:
            return self._gripper_cmd
        cmd = Robotiq3FGripperRobotOutput()
        cmd.rACT = 1
        cmd.rGTO = 1
        cmd.rATR = 0
        cmd.rICF = 0
        cmd.rSPA = 255
        cmd.rFRA = 150
        # Preserve current mode so we do not force a mode-change finger dance.
        if self._gripper_status is not None:
            cmd.rMOD = int(self._gripper_status.gMOD)
            cmd.rPRA = int(self._gripper_status.gPRA)
        else:
            cmd.rMOD = 0
            cmd.rPRA = gripper_position(self._gripper_open)
        self._gripper_cmd = cmd
        return cmd

    def _poll_gripper_toggle(self):
        """Rising-edge Triangle/Y: toggle open/close via rPRA only."""
        pressed = self._gripper_btn == 1
        rising = pressed and not self._gripper_btn_prev
        self._gripper_btn_prev = pressed
        if not rising:
            return

        if self._gripper_pub.get_num_connections() == 0:
            if not self._gripper_no_sub_warned:
                self.controller_log.warning(
                    "Gripper toggle ignored: no subscribers on "
                    "/Robotiq3FGripperRobotOutput (driver down or sim)"
                )
                self._gripper_no_sub_warned = True
            return

        cmd = self._ensure_gripper_cmd()
        self._gripper_open = toggle_gripper_open(self._gripper_open)
        cmd.rPRA = gripper_position(self._gripper_open)
        # Stay active and go-to-position; do not rewrite activation/mode each press.
        cmd.rACT = 1
        cmd.rGTO = 1
        self._gripper_pub.publish(cmd)
        self.controller_log.info(
            "Gripper %s (Triangle/Y)",
            "open" if self._gripper_open else "closed",
        )

    def go_home(self):
        rate = rospy.Rate(125)
        q = self.robot_interface.q
        q[2] = wrap_pi_scalar(q[2])
        print(q)

        trajectory = PointToPointTrajectory.quintic(
            q, self.home, max_vel=0.2, max_acc=1, min_duration=1
        )

        # use P control + feedforward velocity to track the trajectory
        while not rospy.is_shutdown():
            q = self.robot_interface.q
            q[2] = wrap_pi_scalar(q[2])

            t = rospy.Time.now().to_sec()

            dist = np.linalg.norm(self.home - q)

            # we want to both let the trajectory complete and ensure we've
            # converged properly
            if trajectory.done(t) and dist < 1e-2:
                break

            qd, vd, _ = trajectory.sample(t)
            cmd_vel = (qd - q) + vd

            # this shouldn't be needed unless the trajectory is poorly tracked, but
            # we do it just in case for safety
            cmd_vel = bound_array(cmd_vel, lb=-0.2, ub=0.2)

            self.robot_interface.publish_cmd_vel(cmd_vel, bodyframe=False)

            rate.sleep()

        self.robot_interface.brake()

        print(f"Converged to within {dist} of home position.")

    def log_mpc_info(self, logger, controller):
        for key, val in controller.log.items():
            logger.append("_".join(["mpc", key]) + "s", val)

    def update_references(self, references, robot_states):
        """Update the references for the controller.

        Args:
            references (dict): The references to update.
            robot_states (tuple): The robot states.
        """
        pass

    def _teleop_log_fields(self):
        """Return neutral shared logging fields for non-teleop MPC runs."""
        return {
            "joy_axes": np.zeros(6),
            "joy_buttons": np.zeros(16),
            "teleop_mode": "none",
            "teleop_enabled": False,
            "desired_base_vel": np.zeros(3),
            "desired_ee_vel": np.zeros(6),
        }

    def _after_control_step(self, t, robot_states, states, references, u_current):
        """Hook method called after each control step, before logging.

        Child classes can override this to perform additional processing,
        such as metrics collection, after each control iteration.

        Args:
            t (float): Current time relative to experiment start.
            robot_states (tuple): (q, v) tuple from robot interface.
            states (dict): Dictionary with "base" and "EE" keys containing pose and velocity.
            references (dict): Current references dictionary.
            u_current (np.ndarray): MPC preview generalized velocity at the first
                horizon knot, shape ``(nu,)`` (matches ``v_bar[0]``). High-rate
                ``cmd_vel`` on the robot comes from ``low_level_cmd_node`` and may
                differ between MPC solves.
        """
        pass


if __name__ == "__main__":
    rospy.init_node("controller_ros")

    node = ControllerROSNode()
    node.run()
