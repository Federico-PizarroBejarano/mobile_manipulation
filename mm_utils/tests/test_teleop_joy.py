import numpy as np

from mm_utils.teleop_joy import (
    FORCE_ZERO_LL_KP_PARAM,
    STICKS_ACTIVE_PARAM,
    TELEOP_MODE_PARAM,
    VALID_TELEOP_MODES,
    axes_to_base_velocity,
    axes_to_ee_velocity,
    chassis_base_twist_to_world,
    ee_yaw_from_buttons,
    gate_teleop_velocity,
    joint_velocity_command,
    parse_goal_velocity_params,
    parse_teleop_config,
    teleop_ee_twist_for_control,
    teleop_ee_twist_to_world,
    teleop_enable_held,
)
from mm_utils.teleop_mapping import (
    EE_PITCH_BUTTONS,
    EE_ROLL_BUTTONS,
    EE_YAW_AXIS,
    ENABLE_TRIGGER_AXIS,
    HARDWARE_DEADMAN_AXIS,
    hardware_deadman_held,
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
        buttons[12] = 1
        assert ee_yaw_from_buttons(buttons, 12, 11) == -1.0
        buttons[12] = 0
        buttons[11] = 1
        assert ee_yaw_from_buttons(buttons, 12, 11) == 1.0

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
    def test_left_trigger_held(self):
        axes = [0.0] * 6
        axes[ENABLE_TRIGGER_AXIS] = 0.6
        assert teleop_enable_held(axes) is True
        axes[ENABLE_TRIGGER_AXIS] = 0.4
        assert teleop_enable_held(axes) is False

    def test_missing_trigger_slot(self):
        assert teleop_enable_held([0.0, 1.0]) is False

    def test_hardware_deadman_raw_axis(self):
        axes = [0.0] * 6
        axes[HARDWARE_DEADMAN_AXIS] = 1.0
        assert hardware_deadman_held(axes) is False
        axes[HARDWARE_DEADMAN_AXIS] = -0.1
        assert hardware_deadman_held(axes) is True


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


class TestChassisBaseTwistToWorld:
    def test_yaw_zero_passthrough(self):
        v = np.array([0.3, -0.1, 0.2])
        np.testing.assert_allclose(chassis_base_twist_to_world(v, 0.0), v)

    def test_yaw_pi_over_two_forward_becomes_plus_y(self):
        # Chassis +x (forward) -> world +y when yaw = pi/2
        v = np.array([1.0, 0.0, 0.15])
        out = chassis_base_twist_to_world(v, np.pi / 2)
        np.testing.assert_allclose(out, [0.0, 1.0, 0.15], atol=1e-12)


class TestTeleopEeTwistForControl:
    def test_yaw_zero_identity_ee_passthrough(self):
        tw = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6])
        np.testing.assert_allclose(teleop_ee_twist_for_control(tw, 0.0, np.eye(3)), tw)

    def test_yaw_rotates_linear_and_converts_ang_to_body(self):
        # Chassis +x lin -> world +y; chassis ωz stays body ωz when R=I
        tw = np.array([1.0, 0.0, 0.3, 0.0, 0.0, 0.1])
        out = teleop_ee_twist_for_control(tw, np.pi / 2, np.eye(3))
        np.testing.assert_allclose(out, [0.0, 1.0, 0.3, 0.0, 0.0, 0.1], atol=1e-12)

    def test_world_yaw_maps_into_body_when_tool_y_is_up(self):
        # Tool +Y = world up: chassis/world yaw (ωz) becomes body ωy
        tw = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        out = teleop_ee_twist_for_control(tw, 0.0, R)
        np.testing.assert_allclose(out[3:], [0.0, 0.5, 0.0], atol=1e-12)


class TestTeleopEeTwistToWorld:
    def test_identity_passthrough(self):
        tw = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6])
        np.testing.assert_allclose(
            teleop_ee_twist_to_world(tw, 0.0, np.eye(3)), tw, atol=1e-12
        )

    def test_chassis_ang_rotates_with_base_yaw(self):
        # Chassis ωx -> world ωy when base yaw = pi/2
        tw = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0])
        out = teleop_ee_twist_to_world(tw, np.pi / 2, np.eye(3))
        np.testing.assert_allclose(out, [0.0, 0.0, 0.0, 0.0, 0.5, 0.0], atol=1e-12)

    def test_yaw_rate_independent_of_base_yaw(self):
        tw = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.4])
        out0 = teleop_ee_twist_to_world(tw, 0.0, np.eye(3))
        out90 = teleop_ee_twist_to_world(tw, np.pi / 2, np.eye(3))
        np.testing.assert_allclose(out0[3:], out90[3:], atol=1e-12)
        np.testing.assert_allclose(out0[3:], [0.0, 0.0, 0.4], atol=1e-12)


class TestAxesToEeVelocity:
    def test_full_deflection_stick_yaw_and_dpad_angles(self):
        axes = np.zeros(6)
        axes[0] = 0.5
        axes[1] = 1.0
        axes[EE_YAW_AXIS] = -0.5
        axes[3] = 0.5
        buttons = [0] * 16
        buttons[EE_ROLL_BUTTONS[1]] = 1
        buttons[EE_PITCH_BUTTONS[1]] = 1
        max_vel = np.array([0.12, 0.12, 0.12, 0.25, 0.25, 0.25])
        out = axes_to_ee_velocity(axes, buttons, max_vel)
        np.testing.assert_allclose(out, [0.12, 0.06, 0.06, -0.25, 0.25, -0.125])

    def test_dpad_up_is_negative_pitch(self):
        axes = np.zeros(6)
        buttons = [0] * 16
        buttons[13] = 1
        out = axes_to_ee_velocity(axes, buttons, np.ones(6))
        np.testing.assert_allclose(out[4], -1.0)
        np.testing.assert_allclose(out[5], 0.0)

    def test_dpad_left_is_positive_roll(self):
        axes = np.zeros(6)
        buttons = [0] * 16
        buttons[EE_ROLL_BUTTONS[0]] = 1
        out = axes_to_ee_velocity(axes, buttons, np.ones(6))
        np.testing.assert_allclose(out[3], 1.0)

    def test_right_stick_x_is_yaw_not_pitch(self):
        axes = np.zeros(6)
        axes[EE_YAW_AXIS] = 1.0
        out = axes_to_ee_velocity(axes, [0] * 16, np.ones(6))
        np.testing.assert_allclose(out[3], 0.0)
        np.testing.assert_allclose(out[4], 0.0)
        np.testing.assert_allclose(out[5], 1.0)


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
        np.testing.assert_array_equal(cfg["max_base_vel"], [0.3, 0.3, 0.3])
        assert "enable_axis" not in cfg
        assert "ee_yaw_buttons" not in cfg
        assert "ee_pitch_buttons" not in cfg

    def test_controller_teleop_section_ignores_mapping_keys(self):
        cfg = parse_teleop_config(
            {
                "teleop": {
                    "enabled": True,
                    "enable_axis": None,
                    "max_base_vel": [0.5, 0.5, 0.4],
                    "max_ee_vel": [0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
                    "ee_yaw_buttons": [4, 5],
                    "ee_pitch_buttons": [1, 2],
                }
            }
        )
        assert cfg["enabled"] is True
        np.testing.assert_array_equal(cfg["max_base_vel"], [0.5, 0.5, 0.4])
        assert "enable_axis" not in cfg
        assert "ee_yaw_buttons" not in cfg

    def test_sticks_active_param_name(self):
        assert STICKS_ACTIVE_PARAM == "/teleop_sticks_active"

    def test_force_zero_ll_kp_param_name(self):
        assert FORCE_ZERO_LL_KP_PARAM == "/mm_run/force_zero_ll_kp"

    def test_teleop_mode_constants(self):
        assert TELEOP_MODE_PARAM == "/mm_run/teleop_mode"
        assert VALID_TELEOP_MODES == ("none", "mpsf", "direct", "rl")
