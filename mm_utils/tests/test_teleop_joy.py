import numpy as np

from mm_utils.teleop_joy import (
    FORCE_ZERO_LL_KP_PARAM,
    HARDWARE_DEADMAN_BUTTON,
    STICKS_ACTIVE_PARAM,
    TELEOP_MODE_PARAM,
    VALID_TELEOP_MODES,
    axes_to_base_velocity,
    axes_to_ee_velocity,
    ee_yaw_from_buttons,
    gate_teleop_velocity,
    joint_velocity_command,
    parse_goal_velocity_params,
    parse_teleop_config,
    teleop_enable_held,
)


class TestEeYawFromButtons:
    def test_bumper_indices_sim_default(self):
        buttons = [0] * 6
        buttons[4] = 1
        assert ee_yaw_from_buttons(buttons, 4, 5) == -1.0
        buttons[4] = 0
        buttons[5] = 1
        assert ee_yaw_from_buttons(buttons, 4, 5) == 1.0

    def test_dpad_indices_hardware(self):
        buttons = [0] * 16
        buttons[14] = 1
        assert ee_yaw_from_buttons(buttons, 14, 15) == -1.0
        buttons[14] = 0
        buttons[15] = 1
        assert ee_yaw_from_buttons(buttons, 14, 15) == 1.0

    def test_both_pressed_cancels(self):
        buttons = [0] * 6
        buttons[4] = 1
        buttons[5] = 1
        assert ee_yaw_from_buttons(buttons, 4, 5) == 0.0

    def test_missing_button_index(self):
        assert ee_yaw_from_buttons([0, 0], 4, 5) == 0.0


class TestGateTeleopVelocity:
    def test_enabled_passthrough(self):
        vel = np.array([1.0, 0.5, -0.2])
        np.testing.assert_array_equal(gate_teleop_velocity(vel, True), vel)

    def test_disabled_zeros(self):
        vel = np.array([1.0, 0.5, -0.2])
        np.testing.assert_array_equal(gate_teleop_velocity(vel, False), np.zeros(3))


class TestTeleopEnableHeld:
    def test_none_always_enabled(self):
        assert teleop_enable_held([0, 0, 0], None) is True

    def test_button_held(self):
        buttons = [0] * 6
        buttons[5] = 1
        assert teleop_enable_held(buttons, 5) is True
        assert teleop_enable_held(buttons, 4) is False

    def test_out_of_range_index(self):
        assert teleop_enable_held([0, 1], 5) is False


class TestParseGoalVelocityParams:
    def test_defaults_when_empty(self):
        params = parse_goal_velocity_params({})
        np.testing.assert_array_equal(
            params["max_ee_vel"], [0.15, 0.15, 0.15, 0.3, 0.3, 0.3]
        )
        np.testing.assert_array_equal(params["max_base_vel"], [0.3, 0.3, 0.3])

    def test_nested_goal_velocity_override(self):
        params = parse_goal_velocity_params(
            {"goal_velocity": {"max_ee_vel": [0.1, 0.1, 0.1, 0.2, 0.2, 0.2]}}
        )
        np.testing.assert_array_equal(
            params["max_ee_vel"], [0.1, 0.1, 0.1, 0.2, 0.2, 0.2]
        )
        np.testing.assert_array_equal(params["max_base_vel"], [0.3, 0.3, 0.3])


class TestAxesToBaseVelocity:
    def test_full_deflection(self):
        # joy_axes layout: [left_x, left_y, right_x, right_y, lt, rt]
        axes = np.array([0.5, 1.0, -0.5, 0.0, 0.0, 0.0])
        max_vel = np.array([0.3, 0.4, 0.2])
        out = axes_to_base_velocity(axes, max_vel)
        np.testing.assert_allclose(out, [0.3, 0.2, -0.1])


class TestAxesToEeVelocity:
    def test_full_deflection_with_yaw_button(self):
        axes = np.array([0.5, 1.0, -0.5, 0.5, 0.0, 1.0])  # wy = rt - lt = 1
        buttons = [0] * 16
        buttons[15] = 1  # right yaw
        max_vel = np.array([0.12, 0.12, 0.12, 0.25, 0.25, 0.25])
        out = axes_to_ee_velocity(axes, buttons, max_vel, [14, 15])
        np.testing.assert_allclose(out, [0.12, 0.06, 0.06, -0.125, 0.25, 0.25])


class TestJointVelocityCommand:
    def test_base_mode_copies_twist(self):
        base = np.array([0.1, -0.2, 0.3])
        ee = np.array([9.0, 9.0, 9.0, 9.0, 9.0, 9.0])
        cmd = joint_velocity_command("base", base, ee, nu=9)
        np.testing.assert_allclose(cmd[:3], base)
        np.testing.assert_allclose(cmd[3:], 0.0)

    def test_ee_mode_deferred_to_diff_ik(self):
        # joint_velocity_command stays base-only; EE uses solve_diff_ik elsewhere.
        base = np.array([0.1, -0.2, 0.3])
        ee = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        cmd = joint_velocity_command("ee", base, ee, nu=9)
        np.testing.assert_allclose(cmd, 0.0)


class TestParseTeleopConfig:
    def test_defaults(self):
        cfg = parse_teleop_config({})
        assert cfg["enabled"] is False
        assert cfg["enable_button"] == HARDWARE_DEADMAN_BUTTON
        np.testing.assert_array_equal(cfg["max_base_vel"], [0.3, 0.3, 0.3])
        assert cfg["ee_yaw_buttons"] == [14, 15]

    def test_controller_teleop_section(self):
        cfg = parse_teleop_config(
            {
                "teleop": {
                    "enabled": True,
                    "enable_button": 13,
                    "max_base_vel": [0.5, 0.5, 0.4],
                    "max_ee_vel": [0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
                    "ee_yaw_buttons": [4, 5],
                }
            }
        )
        assert cfg["enabled"] is True
        np.testing.assert_array_equal(cfg["max_base_vel"], [0.5, 0.5, 0.4])
        assert cfg["ee_yaw_buttons"] == [4, 5]

    def test_sticks_active_param_name(self):
        assert STICKS_ACTIVE_PARAM == "/teleop_sticks_active"

    def test_force_zero_ll_kp_param_name(self):
        assert FORCE_ZERO_LL_KP_PARAM == "/mm_run/force_zero_ll_kp"

    def test_teleop_mode_constants(self):
        assert TELEOP_MODE_PARAM == "/mm_run/teleop_mode"
        assert VALID_TELEOP_MODES == ("none", "mpsf", "direct", "rl")
