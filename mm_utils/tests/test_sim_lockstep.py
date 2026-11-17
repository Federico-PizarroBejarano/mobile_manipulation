# mm_utils/tests/test_sim_lockstep.py

import threading
import types

import pytest

from mm_utils.sim_lockstep import (
    CONTROL_CYCLE_DONE_TOPIC,
    LOCKSTEP_PARAM,
    ControlCycleDoneWaiter,
    end_control_cycle,
)


def test_lockstep_constants():
    assert LOCKSTEP_PARAM == "/mm_run/sim_lockstep"
    assert CONTROL_CYCLE_DONE_TOPIC == "/mm_run/control_cycle_done"


def test_end_control_cycle_requires_publisher_when_lockstep():
    class _Rate:
        def sleep(self):
            raise AssertionError("rate.sleep should not run in lockstep")

    with pytest.raises(ValueError, match="cycle_done_pub"):
        end_control_cycle(_Rate(), 0.05, None, use_lockstep=True)


def test_end_control_cycle_rate_sleep_when_not_lockstep():
    class _Rate:
        def __init__(self):
            self.n = 0

        def sleep(self):
            self.n += 1

    rate = _Rate()
    end_control_cycle(rate, 0.05, None, use_lockstep=False)
    assert rate.n == 1


def test_waiter_consumes_already_arrived_seq(monkeypatch):
    """Signal before wait() must not be lost (startup race)."""
    fake = types.ModuleType("rospy")
    fake.is_shutdown = lambda: False
    monkeypatch.setitem(__import__("sys").modules, "rospy", fake)

    waiter = object.__new__(ControlCycleDoneWaiter)
    waiter._lock = threading.Lock()
    waiter._event = threading.Event()
    waiter._last_seq = 1
    waiter._consumed_seq = 0

    assert waiter.wait(timeout_s=0.2) is True
    assert waiter._consumed_seq == 1
    assert waiter.wait(timeout_s=0.05) is False
