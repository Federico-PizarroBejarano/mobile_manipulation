from mm_utils.robotiq_gripper import (
    GRIPPER_CLOSE_POS,
    GRIPPER_OPEN_POS,
    GRIPPER_TOGGLE_BUTTON,
    gripper_position,
    is_gripper_ready,
    seed_gripper_mode_position,
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

    def test_is_gripper_ready(self):
        assert is_gripper_ready(1, 3) is True
        assert is_gripper_ready(0, 3) is False
        assert is_gripper_ready(1, 0) is False
        assert is_gripper_ready(1, 1) is False
        assert is_gripper_ready(1, 2) is False

    def test_seed_from_status(self):
        assert seed_gripper_mode_position(1, 200, default_open=True) == (1, 200)
        assert seed_gripper_mode_position(0, 0, default_open=False) == (0, 0)

    def test_seed_without_status(self):
        assert seed_gripper_mode_position(None, None, default_open=True) == (
            0,
            GRIPPER_OPEN_POS,
        )
        assert seed_gripper_mode_position(None, None, default_open=False) == (
            0,
            GRIPPER_CLOSE_POS,
        )
        assert seed_gripper_mode_position(1, None, default_open=True) == (
            0,
            GRIPPER_OPEN_POS,
        )
