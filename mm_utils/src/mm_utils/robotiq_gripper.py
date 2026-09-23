"""Pure helpers for Robotiq 3F gripper open/close commands (no ROS deps)."""

# Triangle / Y on PS4 / Xbox (same index sim and hardware).
GRIPPER_TOGGLE_BUTTON = 3

# Position command [0, 255] — same encoding as mobile_manipulation_central
# scripts/control/gripper.py
GRIPPER_OPEN_POS = 0
GRIPPER_CLOSE_POS = 255

# gIMC == 3 means activation complete / gripper ready (Robotiq status).
GRIPPER_IMC_READY = 3


def toggle_gripper_open(currently_open):
    """Return the next open/closed latch after a toggle press."""
    return not currently_open


def gripper_position(is_open):
    """Robotiq rPRA value for open (True) or closed (False)."""
    return GRIPPER_OPEN_POS if is_open else GRIPPER_CLOSE_POS


def is_gripper_ready(g_act, g_imc):
    """True when the gripper is activated and initialization is complete."""
    return int(g_act) == 1 and int(g_imc) == GRIPPER_IMC_READY


def seed_gripper_mode_position(g_mod, g_pra, default_open):
    """Seed (rMOD, rPRA) from status fields, or defaults when status is absent.

    Pass g_mod/g_pra as None when no status has been received yet.
    """
    if g_mod is None or g_pra is None:
        return 0, gripper_position(default_open)
    return int(g_mod), int(g_pra)
