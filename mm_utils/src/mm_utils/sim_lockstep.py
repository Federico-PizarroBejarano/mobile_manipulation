"""Lockstep sim-time sync between ``sim_ros`` and plan publishers.

When ``/use_sim_time`` and ``/mm_run/sim_lockstep`` are true, the simulator
advances physics only after each control cycle finishes. Wall-clock slowness
(GUI, ACT inference, acados) then stretches real time without skipping sim
ticks relative to the controller.
"""

from __future__ import annotations

import threading
import time

LOCKSTEP_PARAM = "/mm_run/sim_lockstep"
CONTROL_CYCLE_DONE_TOPIC = "/mm_run/control_cycle_done"


def lockstep_enabled():
    """True when sim time is on and lockstep has not been disabled."""
    import rospy

    if not bool(rospy.get_param("/use_sim_time", False)):
        return False
    return bool(rospy.get_param(LOCKSTEP_PARAM, True))


class ControlCycleDonePublisher:
    """Publishes a monotonic seq so ``sim_ros`` can wait for each cycle."""

    def __init__(self):
        import rospy
        from std_msgs.msg import UInt64

        self._UInt64 = UInt64
        self._seq = 0
        # Latch so sim still sees the latest pulse if it subscribes after signal.
        self._pub = rospy.Publisher(
            CONTROL_CYCLE_DONE_TOPIC, UInt64, queue_size=1, latch=True
        )
        # Give subscribers a moment to connect (sim may start first).
        time.sleep(0.05)

    def signal(self):

        self._seq += 1
        self._pub.publish(self._UInt64(data=int(self._seq)))
        return self._seq


class ControlCycleDoneWaiter:
    """Sim-side waiter for the next controller cycle-done pulse."""

    def __init__(self):
        import rospy
        from std_msgs.msg import UInt64

        self._lock = threading.Lock()
        self._event = threading.Event()
        self._last_seq = 0
        self._consumed_seq = 0
        self._sub = rospy.Subscriber(
            CONTROL_CYCLE_DONE_TOPIC, UInt64, self._cb, queue_size=10
        )

    def _cb(self, msg):
        seq = int(msg.data)
        with self._lock:
            if seq <= self._last_seq:
                return
            self._last_seq = seq
        self._event.set()

    def wait(self, timeout_s=120.0):
        """Block until a new unconsumed cycle-done arrives.

        Returns False on timeout/shutdown. Safe if the controller signaled
        before this wait started (seq is latched / retained).
        """
        import rospy

        t0 = time.perf_counter()
        while not rospy.is_shutdown():
            with self._lock:
                if self._last_seq > self._consumed_seq:
                    self._consumed_seq = self._last_seq
                    return True
            if timeout_s is not None and (time.perf_counter() - t0) >= float(timeout_s):
                return False
            self._event.wait(timeout=0.05)
            self._event.clear()
        return False


def wait_for_sim_time_advance(t_start, period_s, timeout_s=120.0):
    """Wait until ``rospy.Time.now() >= t_start + period_s`` (sim time)."""
    import rospy

    period_s = float(period_s)
    if period_s <= 0.0:
        return True
    if t_start is None:
        t_start = rospy.Time.now()
    target = t_start + rospy.Duration.from_sec(period_s)
    t0 = time.perf_counter()
    rate = rospy.Rate(500)
    while not rospy.is_shutdown():
        if rospy.Time.now() >= target:
            return True
        if timeout_s is not None and (time.perf_counter() - t0) >= float(timeout_s):
            return False
        try:
            rate.sleep()
        except rospy.ROSInterruptException:
            return False
    return False


def end_control_cycle(rate, period_s, cycle_done_pub=None, *, use_lockstep=None):
    """Finish one control tick: lockstep signal+wait, else ``rate.sleep()``.

    ``cycle_done_pub`` is a ``ControlCycleDonePublisher`` (required when lockstep).
    """
    import rospy

    if use_lockstep is None:
        use_lockstep = lockstep_enabled()
    if use_lockstep:
        if cycle_done_pub is None:
            raise ValueError("cycle_done_pub required when lockstep is enabled")
        t_before = rospy.Time.now()
        cycle_done_pub.signal()
        if not wait_for_sim_time_advance(t_before, period_s):
            rospy.logwarn_throttle(
                5.0,
                "sim lockstep: timed out waiting for /clock to advance by %.3fs",
                float(period_s),
            )
        return
    rate.sleep()
