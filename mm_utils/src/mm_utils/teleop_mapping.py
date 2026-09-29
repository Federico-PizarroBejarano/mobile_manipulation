"""Joystick index map. Change mappings here only.

Human-readable roles: mm_run/config/teleop/README.md
"""

HARDWARE_DEADMAN_AXIS = 2
ENABLE_TRIGGER_AXIS = 4
ENABLE_TRIGGER_THRESHOLD = 0.5
EE_YAW_AXIS = 2
EE_ROLL_BUTTONS = (12, 11)
EE_PITCH_BUTTONS = (13, 14)


def hardware_deadman_held(raw_axes):
    if HARDWARE_DEADMAN_AXIS >= len(raw_axes):
        return False
    return float(raw_axes[HARDWARE_DEADMAN_AXIS]) < 0.0
