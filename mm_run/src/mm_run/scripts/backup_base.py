#!/usr/bin/env python3
"""Drive the Ridgeback backward in the body frame, holding the arm still.

Used between hardware ACT trials so ``home.py`` does not swing the arm through
the high table. Stops after the base has moved ``--distance`` meters in the
world xy plane, or when the motion times out.
"""

from __future__ import annotations

import argparse

import mobile_manipulation_central as mm
import numpy as np
import rospy


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--distance", type=float, default=1.0, help="Meters to back up")
    p.add_argument(
        "--speed",
        type=float,
        default=0.25,
        help="Body-frame reverse speed (m/s)",
    )
    p.add_argument("--rate", type=float, default=50.0, help="Command rate (Hz)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.distance <= 0.0:
        raise SystemExit("--distance must be positive")
    if args.speed <= 0.0:
        raise SystemExit("--speed must be positive")
    if args.rate <= 0.0:
        raise SystemExit("--rate must be positive")

    rospy.init_node("backup_base")
    base = mm.RidgebackROSInterface()
    arm = mm.UR10ROSInterface()
    rate = rospy.Rate(args.rate)
    timeout_s = max(8.0, 4.0 * float(args.distance) / float(args.speed))

    while not rospy.is_shutdown() and not base.ready():
        rate.sleep()
    if rospy.is_shutdown() or not base.ready():
        raise SystemExit("base joint states never arrived")

    origin = np.array(base.q[:2], dtype=float).copy()
    cmd = np.array([-float(args.speed), 0.0, 0.0])
    arm_hold = np.zeros(arm.nv)
    t0 = rospy.Time.now().to_sec()
    dist = 0.0
    while not rospy.is_shutdown():
        dist = float(np.linalg.norm(np.array(base.q[:2], dtype=float) - origin))
        if dist >= float(args.distance):
            break
        if rospy.Time.now().to_sec() - t0 > timeout_s:
            base.brake()
            arm.publish_cmd_vel(arm_hold)
            raise SystemExit(
                "backup timed out after %.1fs (moved %.2fm, wanted %.2fm)"
                % (timeout_s, dist, args.distance)
            )
        base.publish_cmd_vel(cmd, bodyframe=True)
        if arm.ready():
            arm.publish_cmd_vel(arm_hold)
        rate.sleep()

    base.brake()
    if arm.ready():
        arm.publish_cmd_vel(arm_hold)
    print("Backed up %.2fm." % dist)


if __name__ == "__main__":
    main()
