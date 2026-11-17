import argparse
import datetime
import os
import sys
import time

import numpy as np
import rospy
from mobile_manipulation_central.simulation_ros_interface import (
    SimulatedMobileManipulatorROSInterface,
    SimulatedViconObjectInterface,
)

from mm_simulator import simulation
from mm_utils import parsing
from mm_utils.logging import DataLogger
from mm_utils.sim_lockstep import ControlCycleDoneWaiter, lockstep_enabled
from mm_utils.teleop_session_logging import (
    adopt_shared_results_user,
    clear_experiment_timestamp,
    resolve_experiment_timestamp,
)


def _spin_for_cmd_vel(ros_interface, retries=20, sleep_s=0.001):
    """Allow low_level_cmd_node's /clock callback to publish cmd_vel."""
    for _ in range(int(retries)):
        if ros_interface.cmd_vel is not None:
            return
        time.sleep(float(sleep_s))


def _advance_sim_period(
    *,
    sim,
    robot,
    ros_interface,
    vicon_tool_interface,
    logger,
    t,
    t0,
    period_s,
):
    """Step physics for about ``period_s`` of sim time; publish state+/clock each tick."""
    period_s = float(period_s)
    period_end = float(t) + period_s
    while float(t) < period_end - 1e-12 and not rospy.is_shutdown():
        print(
            f"-------------- {(t - t0):.3f}s/{float(sim.duration):.3f}s ------------------"
        )
        q, v = robot.joint_states()
        ros_interface.publish_feedback(t, q, v)
        ros_interface.publish_time(t)
        _spin_for_cmd_vel(ros_interface)

        cmd_model = np.asarray(ros_interface.cmd_vel, dtype=float).reshape(-1)
        cmd_vel_world = robot.command_velocity(cmd_model, bodyframe=True)
        ee_curr_pos, ee_curr_orn = robot.link_pose()
        vicon_tool_interface.publish_pose(t, ee_curr_pos, ee_curr_orn)

        r_ew_w, Q_we = robot.link_pose()
        v_ew_w, ω_ew_w = robot.link_velocity()
        logger.append("ts", t)
        logger.append("xs", np.hstack((q, v)))
        logger.append("cmd_vels_model", cmd_model)
        logger.append("cmd_vels", cmd_vel_world)
        logger.append("r_ew_ws", r_ew_w)
        logger.append("Q_wes", Q_we)
        logger.append("v_ew_ws", v_ew_w)
        logger.append("ω_ew_ws", ω_ew_w)
        logger.append("r_bw_ws", q[:2])
        logger.append("yaw_bw_ws", q[2])
        logger.append("v_bw_ws", v[:2])
        logger.append("ω_bw_ws", v[2])

        t, _ = sim.step(t)
        if float(t) - float(t0) >= float(sim.duration):
            break
    return t


def main():
    np.set_printoptions(precision=3, suppress=True)
    argv = rospy.myargv(argv=sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, help="Path to configuration file."
    )
    parser.add_argument(
        "--video",
        nargs="?",
        default=None,
        const="",
        help="Record video. Optionally specify prefix for video directory.",
    )
    parser.add_argument(
        "--logging_sub_folder",
        type=str,
        help="save data in a sub folder of logging directory",
    )
    parser.add_argument(
        "--ctrl_config",
        default="default",
        help="Optional overlay YAML (same as controller); merges simulation.duration etc.",
    )
    parser.add_argument(
        "--GUI",
        action="store_true",
        help="Enable PyBullet GUI. This overwrites the yaml settings",
    )
    args = parser.parse_args(argv[1:])

    # load configuration and overwrite with args
    config = parsing.load_config(args.config)
    if args.ctrl_config != "default":
        config = parsing.recursive_dict_update(
            config, parsing.load_config(args.ctrl_config)
        )

    if args.GUI:
        config["simulation"]["gui"] = True

    if args.logging_sub_folder != "default":
        config["logging"]["log_dir"] = os.path.join(
            config["logging"]["log_dir"], args.logging_sub_folder
        )

    sim_config = config["simulation"]
    ctrl_rate = float(config.get("controller", {}).get("ctrl_rate", 20.0))
    if ctrl_rate <= 0.0:
        raise ValueError(f"controller.ctrl_rate must be positive, got {ctrl_rate}")
    ctrl_period = 1.0 / ctrl_rate

    # start the simulation
    timestamp = datetime.datetime.now()
    sim = simulation.BulletSimulation(
        config=sim_config, timestamp=timestamp, cli_args=args
    )
    robot = sim.robot

    # initial time, state, input
    t = 0.0

    # ros interface
    rospy.init_node("sim_ros")
    # Prefer an existing stamp (controller may have set it during Bullet GUI load).
    # Never overwrite — that was splitting sim/ vs control/ into two folders.
    session_timestamp = resolve_experiment_timestamp(create=True)
    rospy.on_shutdown(clear_experiment_timestamp)

    # init logger
    logger = DataLogger(config, name="sim")
    logger.add("sim_timestep", sim.timestep)
    logger.add("duration", sim.duration)

    logger.add("nq", sim_config["robot"]["dims"]["q"])
    logger.add("nv", sim_config["robot"]["dims"]["v"])
    logger.add("nx", sim_config["robot"]["dims"]["x"])
    logger.add("nu", sim_config["robot"]["dims"]["u"])

    # ros interface
    ros_interface = SimulatedMobileManipulatorROSInterface()
    ros_interface.publish_time(t)

    vicon_tool_interface = SimulatedViconObjectInterface(
        sim_config["robot"]["tool_vicon_name"]
    )

    # Subscribe before /controller_started so early cycle-done pulses are not dropped.
    use_lockstep = lockstep_enabled()
    cycle_waiter = ControlCycleDoneWaiter() if use_lockstep else None
    if use_lockstep:
        print(
            f"----- Sim lockstep ON (advance {ctrl_period:.3f}s per control cycle) -----"
        )
    else:
        print("----- Sim lockstep OFF (free-running wall sleep) -----")

    while (
        not ros_interface.ready()
        or not rospy.has_param("/controller_started")
        or not rospy.get_param("/controller_started")
    ):
        q, v = robot.joint_states()
        ros_interface.publish_feedback(t, q, v)
        ros_interface.publish_time(t)
        t += sim.timestep
        time.sleep(sim.timestep)

        if rospy.is_shutdown():
            return

    print("Control commands received. Proceed ... ")
    t0 = t
    while not rospy.is_shutdown():
        if use_lockstep:
            # Poll finished/shutdown with short waits so we do not sit for 180s
            # after the controller stops signaling.
            got_cycle = False
            wait_deadline = time.perf_counter() + 180.0
            while not rospy.is_shutdown():
                if rospy.get_param("/controller_finished", False):
                    print("Controller finished all tasks. Stopping simulation.")
                    got_cycle = None
                    break
                remaining = wait_deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                if cycle_waiter.wait(timeout_s=min(1.0, remaining)):
                    got_cycle = True
                    break
            if got_cycle is None:
                break
            if not got_cycle:
                rospy.logwarn(
                    "sim lockstep: timed out waiting for /mm_run/control_cycle_done"
                )
                break
            t = _advance_sim_period(
                sim=sim,
                robot=robot,
                ros_interface=ros_interface,
                vicon_tool_interface=vicon_tool_interface,
                logger=logger,
                t=t,
                t0=t0,
                period_s=ctrl_period,
            )
        else:
            print(
                f"-------------- {(t - t0):.3f}s/{float(sim.duration):.3f}s ------------------"
            )
            q, v = robot.joint_states()
            ros_interface.publish_feedback(t, q, v)
            ros_interface.publish_time(t)

            # /ridgeback/cmd_vel is body-frame; command_velocity converts it to world.
            cmd_model = np.asarray(ros_interface.cmd_vel, dtype=float).reshape(-1)
            cmd_vel_world = robot.command_velocity(cmd_model, bodyframe=True)
            ee_curr_pos, ee_curr_orn = robot.link_pose()
            vicon_tool_interface.publish_pose(t, ee_curr_pos, ee_curr_orn)

            # log
            r_ew_w, Q_we = robot.link_pose()
            v_ew_w, ω_ew_w = robot.link_velocity()
            logger.append("ts", t)
            logger.append("xs", np.hstack((q, v)))
            logger.append("cmd_vels_model", cmd_model)
            logger.append("cmd_vels", cmd_vel_world)
            logger.append("r_ew_ws", r_ew_w)
            logger.append("Q_wes", Q_we)
            logger.append("v_ew_ws", v_ew_w)
            logger.append("ω_ew_ws", ω_ew_w)

            logger.append("r_bw_ws", q[:2])
            logger.append("yaw_bw_ws", q[2])
            logger.append("v_bw_ws", v[:2])
            logger.append("ω_bw_ws", v[2])

            t, _ = sim.step(t)
            time.sleep(sim.timestep)

        # Check if controller has finished all tasks or if duration exceeded
        controller_finished = rospy.get_param("/controller_finished", False)
        if controller_finished:
            print("Controller finished all tasks. Stopping simulation.")
            break
        if t - t0 >= sim.duration:
            print(
                f"Simulation duration ({float(sim.duration):.3f}s) exceeded. Stopping simulation."
            )
            break

    # Controller may have nested under <log_dir>/<user>/ via trial metadata;
    # adopt the shared ROS param so sim/ is not orphaned at <log_dir>/<ts>/.
    adopt_shared_results_user(logger)
    logger.save(session_timestamp=session_timestamp)


if __name__ == "__main__":
    main()
