from mm_utils.robotiq_gripper import (
    GRIPPER_CLOSE_POS,
    GRIPPER_OPEN_POS,
    GRIPPER_TOGGLE_BUTTON,
    gripper_position,
    toggle_gripper_open,
)


class TestRobotiqGripperHelpers:
    def test_toggle_button_is_triangle(self):
        assert GRIPPER_TOGGLE_BUTTON == 3

    def test_toggle_open_closed(self):
        assert toggle_gripper_open(True) is False
        assert toggle_gripper_open(False) is True

    def test_position_commands(self):
        assert gripper_position(True) == GRIPPER_OPEN_POS == 0
        assert gripper_position(False) == GRIPPER_CLOSE_POS == 255
