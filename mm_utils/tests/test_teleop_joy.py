import numpy as np

from mm_utils.teleop_joy import (
    FORCE_ZERO_LL_KP_PARAM,
    REFERENCE_RAMP_DEFAULTS,
    STICKS_ACTIVE_PARAM,
    TELEOP_MODE_PARAM,
    VALID_TELEOP_MODES,
    TeleopTwistRamp,
    axes_to_base_velocity,
    axes_to_ee_velocity,
    chassis_base_twist_to_world,
    ee_yaw_from_buttons,
    gate_teleop_velocity,
    joint_velocity_command,
    parse_goal_velocity_params,
    parse_teleop_config,
    scale_teleop_control_effort,
    slew_toward,
    teleop_ee_twist_for_control,
    teleop_ee_twist_to_world,
    teleop_enable_held,
)
from mm_utils.teleop_mapping import (
    EE_PITCH_BUTTONS,
    EE_ROLL_BUTTONS,
    EE_YAW_AXIS,
    ENABLE_TRIGGER_AXIS,
    END_TRIAL_BUTTON,
    HARDWARE_DEADMAN_AXIS,
    START_TRIAL_BUTTON,
    hardware_deadman_held,
    trial_button_action,
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
    def test_yaw_zero_passthrough(self):
        tw = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6])
        np.testing.assert_allclose(teleop_ee_twist_for_control(tw, 0.0), tw)

    def test_yaw_rotates_linear_and_keeps_body_angular(self):
        # Chassis +x lin -> world +y; tool-frame ω is unchanged by base yaw.
        tw = np.array([1.0, 0.0, 0.3, 0.4, -0.5, 0.1])
        out = teleop_ee_twist_for_control(tw, np.pi / 2)
        np.testing.assert_allclose(out, [0.0, 1.0, 0.3, 0.4, -0.5, 0.1], atol=1e-12)

    def test_body_angular_independent_of_base_yaw(self):
        tw = np.array([0.0, 0.0, 0.0, 0.2, -0.3, 0.5])
        out0 = teleop_ee_twist_for_control(tw, 0.0)
        out90 = teleop_ee_twist_for_control(tw, np.pi / 2)
        np.testing.assert_allclose(out0[3:], tw[3:], atol=1e-12)
        np.testing.assert_allclose(out90[3:], tw[3:], atol=1e-12)


class TestTeleopEeTwistToWorld:
    def test_identity_passthrough(self):
        tw = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6])
        np.testing.assert_allclose(
            teleop_ee_twist_to_world(tw, 0.0, np.eye(3)), tw, atol=1e-12
        )

    def test_body_ang_maps_with_tool_rotation(self):
        # Tool +Y = world up: body ωy becomes world ωz.
        tw = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0])
        R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        out = teleop_ee_twist_to_world(tw, 0.0, R)
        np.testing.assert_allclose(out[3:], [0.0, 0.0, 0.5], atol=1e-12)

    def test_body_ang_independent_of_base_yaw(self):
        tw = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.4])
        out0 = teleop_ee_twist_to_world(tw, 0.0, np.eye(3))
        out90 = teleop_ee_twist_to_world(tw, np.pi / 2, np.eye(3))
        np.testing.assert_allclose(out0[3:], out90[3:], atol=1e-12)
        np.testing.assert_allclose(out0[3:], [0.5, 0.0, 0.4], atol=1e-12)


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
        np.testing.assert_allclose(out, [0.12, 0.06, 0.06, 0.25, -0.25, -0.125])

    def test_dpad_up_is_positive_pitch(self):
        axes = np.zeros(6)
        buttons = [0] * 16
        buttons[13] = 1
        out = axes_to_ee_velocity(axes, buttons, np.ones(6))
        np.testing.assert_allclose(out[4], 1.0)
        np.testing.assert_allclose(out[5], 0.0)

    def test_dpad_left_is_negative_roll(self):
        axes = np.zeros(6)
        buttons = [0] * 16
        buttons[EE_ROLL_BUTTONS[0]] = 1
        out = axes_to_ee_velocity(axes, buttons, np.ones(6))
        np.testing.assert_allclose(out[3], -1.0)

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
        np.testing.assert_array_equal(
            cfg["reference_ramp"]["base"], REFERENCE_RAMP_DEFAULTS["base"]
        )
        assert "enable_axis" not in cfg
        assert "ee_yaw_buttons" not in cfg

    def test_sticks_active_param_name(self):
        assert STICKS_ACTIVE_PARAM == "/teleop_sticks_active"

    def test_force_zero_ll_kp_param_name(self):
        assert FORCE_ZERO_LL_KP_PARAM == "/mm_run/force_zero_ll_kp"

    def test_teleop_mode_constants(self):
        assert TELEOP_MODE_PARAM == "/mm_run/teleop_mode"
        assert VALID_TELEOP_MODES == ("none", "mpsf", "direct", "rl")

    def test_reference_ramp_defaults_and_override(self):
        cfg = parse_teleop_config({})
        np.testing.assert_array_equal(
            cfg["reference_ramp"]["base"], REFERENCE_RAMP_DEFAULTS["base"]
        )
        np.testing.assert_array_equal(
            cfg["reference_ramp"]["ee"], REFERENCE_RAMP_DEFAULTS["ee"]
        )
        cfg = parse_teleop_config(
            {
                "teleop": {
                    "reference_ramp": {
                        "base": [2.0, 2.0, 1.5],
                        "ee": [0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
                    }
                }
            }
        )
        np.testing.assert_array_equal(cfg["reference_ramp"]["base"], [2.0, 2.0, 1.5])
        np.testing.assert_array_equal(
            cfg["reference_ramp"]["ee"], [0.5, 0.5, 0.5, 1.0, 1.0, 1.0]
        )


class TestTeleopEffortStrictness:
    def _nominal(self):
        return [
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.01, 0.02]),
            np.array([0.05, 0.05, 0.05]),
            np.array([0.01, 0.02]),
            np.array([0.1, 0.1, 0.1]),
        ]

    def test_parse_defaults_and_override(self):
        cfg = parse_teleop_config({})
        assert cfg["ee_teleop_strictness"] == 1.0
        assert cfg["base_teleop_strictness"] == 1.0
        cfg = parse_teleop_config(
            {
                "teleop": {
                    "ee_teleop_strictness": 10.0,
                    "base_teleop_strictness": 0.0,
                }
            }
        )
        assert cfg["ee_teleop_strictness"] == 10.0
        assert cfg["base_teleop_strictness"] == 0.0

    def test_parse_rejects_negative(self):
        try:
            parse_teleop_config({"teleop": {"ee_teleop_strictness": -1.0}})
        except ValueError:
            return
        raise AssertionError("expected ValueError for negative strictness")

    def test_none_and_unity_leave_weights_unchanged(self):
        nominal = self._nominal()
        for mode, ee_s, base_s in (
            (None, 10.0, 10.0),
            ("ee", 1.0, 4.0),
            ("base", 4.0, 1.0),
        ):
            scaled = scale_teleop_control_effort(nominal, mode, ee_s, base_s)
            for got, src in zip(scaled, nominal):
                np.testing.assert_allclose(got, src)
        scaled = scale_teleop_control_effort(nominal, "ee", 10.0, 1.0)
        nominal[5][0] = 999.0
        np.testing.assert_allclose(scaled[5][0], 1.0)

    def test_ee_scales_only_base_effort(self):
        nominal = self._nominal()
        scaled = scale_teleop_control_effort(nominal, "ee", 10.0, 3.0)
        np.testing.assert_allclose(scaled[2], nominal[2])
        np.testing.assert_allclose(scaled[4], nominal[4])
        np.testing.assert_allclose(scaled[3], nominal[3] * 10.0)
        np.testing.assert_allclose(scaled[5], nominal[5] * 10.0)

    def test_base_scales_only_arm_effort(self):
        nominal = self._nominal()
        scaled = scale_teleop_control_effort(nominal, "base", 10.0, 4.0)
        np.testing.assert_allclose(scaled[3], nominal[3])
        np.testing.assert_allclose(scaled[5], nominal[5])
        np.testing.assert_allclose(scaled[2], nominal[2] * 4.0)
        np.testing.assert_allclose(scaled[4], nominal[4] * 4.0)


class TestTrialButtons:
    def test_square_starts_and_circle_ends(self):
        assert START_TRIAL_BUTTON == 2
        assert END_TRIAL_BUTTON == 1
        assert trial_button_action(
            waiting=True, start_latched=True, end_latched=False
        ) == ("start")
        assert trial_button_action(
            waiting=False, start_latched=False, end_latched=True
        ) == ("end")
        assert (
            trial_button_action(waiting=False, start_latched=True, end_latched=False)
            is None
        )
        assert (
            trial_button_action(waiting=True, start_latched=False, end_latched=True)
            is None
        )


class TestTeleopTwistRamp:
    def test_one_step_is_clipped_to_rate_times_dt(self):
        out = slew_toward(np.zeros(3), [0.5, -0.4, 0.2], [1.0, 1.0, 1.0], 0.05)
        np.testing.assert_allclose(out, [0.05, -0.05, 0.05])

    def test_repeated_steps_reach_target(self):
        ramp = TeleopTwistRamp([1.0, 1.0, 1.0], [1.0] * 6)
        target = np.array([0.5, 0.0, -0.25])
        got = np.zeros(3)
        for _ in range(20):
            got = ramp.base(target, 0.05)
        np.testing.assert_allclose(got, target)

    def test_reset_returns_to_zero(self):
        ramp = TeleopTwistRamp([1.0, 1.0, 1.0], [1.5] * 6)
        ramp.base([0.5, 0.0, 0.0], 0.05)
        ramp.ee([0.3, 0.0, 0.0, 0.0, 0.0, 0.2], 0.05)
        ramp.reset()
        # A surviving pre-reset state of 0.05 would advance to 0.10.
        np.testing.assert_allclose(ramp.base([0.5, 0.0, 0.0], 0.05), [0.05, 0.0, 0.0])
        np.testing.assert_allclose(
            ramp.ee([0.3, 0.0, 0.0, 0.0, 0.0, 0.2], 0.05),
            [0.075, 0.0, 0.0, 0.0, 0.0, 0.075],
        )
