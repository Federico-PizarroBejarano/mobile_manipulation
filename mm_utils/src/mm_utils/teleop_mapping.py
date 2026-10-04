"""Joystick index map. Change mappings here only.

Human-readable roles: mm_run/config/teleop/README.md
"""

HARDWARE_DEADMAN_AXIS = 2
ENABLE_TRIGGER_AXIS = 4
ENABLE_TRIGGER_THRESHOLD = 0.5
EE_YAW_AXIS = 2
EE_ROLL_BUTTONS = (12, 11)
EE_PITCH_BUTTONS = (13, 14)
# PS4 Square / Xbox X. Starts a trial; extra presses do not end it.
START_TRIAL_BUTTON = 2
# PS4 Circle / Xbox B. Ends a running trial; does not start one.
END_TRIAL_BUTTON = 1


def trial_button_action(*, waiting, start_latched, end_latched):
    """Square starts a waiting trial. Circle ends a running trial."""
    if waiting and start_latched:
        return "start"
    if not waiting and end_latched:
        return "end"
    return None


def hardware_deadman_held(raw_axes):
    if HARDWARE_DEADMAN_AXIS >= len(raw_axes):
        return False
    return float(raw_axes[HARDWARE_DEADMAN_AXIS]) < 0.0
