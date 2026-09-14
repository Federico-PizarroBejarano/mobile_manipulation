"""Pure helpers for Robotiq 3F gripper open/close commands (no ROS deps)."""

# Triangle / Y on PS4 / Xbox (same index sim and hardware).
GRIPPER_TOGGLE_BUTTON = 3

# Position command [0, 255] — same encoding as mobile_manipulation_central
# scripts/control/gripper.py
GRIPPER_OPEN_POS = 0
GRIPPER_CLOSE_POS = 255


def toggle_gripper_open(currently_open):
    """Return the next open/closed latch after a toggle press."""
    return not currently_open


def gripper_position(is_open):
    """Robotiq rPRA value for open (True) or closed (False)."""
    return GRIPPER_OPEN_POS if is_open else GRIPPER_CLOSE_POS
