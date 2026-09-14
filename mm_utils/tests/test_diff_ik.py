import numpy as np

from mm_utils.diff_ik import build_ik_params_from_config, solve_diff_ik


def _ik_params(nu=9, reg=True, strength=1e-6):
    return {
        "nu": nu,
        "use_weighted_regularization": reg,
        "regularization_strength": strength,
        "joint_vel_lower": -np.ones(nu),
        "joint_vel_upper": np.ones(nu),
        "ee_max_linear_vel": 10.0,
        "ee_max_angular_vel": 10.0,
    }


class TestSolveDiffIk:
    def test_recovers_arm_velocity_with_identity_arm_block(self):
        nu = 9
        # J maps arm joints 3:9 directly onto EE twist; base columns zero.
        J = np.zeros((6, nu))
        J[:, 3:9] = np.eye(6)
        desired = np.array([0.1, -0.2, 0.3, 0.05, -0.05, 0.1])
        u = solve_diff_ik(J, desired, base_vel=np.zeros(3), ik_params=_ik_params(nu))
        np.testing.assert_allclose(u[:3], 0.0, atol=1e-10)
        np.testing.assert_allclose(u[3:], desired, atol=1e-6)

    def test_subtracts_base_induced_ee_velocity(self):
        nu = 9
        J = np.zeros((6, nu))
        J[:, 0] = [1, 0, 0, 0, 0, 0]  # base vx -> EE vx
        J[:, 3:9] = np.eye(6)
        base_vel = np.array([0.2, 0.0, 0.0])
        desired = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        u = solve_diff_ik(J, desired, base_vel=base_vel, ik_params=_ik_params(nu))
        np.testing.assert_allclose(u[:3], base_vel)
        # Arm only needs to supply the residual 0.3 in vx.
        np.testing.assert_allclose(u[3], 0.3, atol=1e-6)
        np.testing.assert_allclose(u[4:], 0.0, atol=1e-6)

    def test_clips_to_joint_limits(self):
        nu = 9
        J = np.zeros((6, nu))
        J[:, 3:9] = np.eye(6)
        params = _ik_params(nu)
        params["joint_vel_upper"][3] = 0.05
        desired = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        u = solve_diff_ik(J, desired, base_vel=np.zeros(3), ik_params=params)
        assert u[3] <= 0.05 + 1e-12


class TestBuildIkParamsFromConfig:
    def test_reads_limits_and_ik_section(self):
        config = {
            "ik": {
                "regularization_strength": 0.2,
                "use_weighted_regularization": False,
            },
            "robot": {"ee_linear_vel_limit": 0.4, "ee_angular_vel_limit": 0.6},
            "controller": {
                "robot": {
                    "limits": {
                        "state": {
                            "lower": [0] * 9 + [-0.7] * 9,
                            "upper": [0] * 9 + [0.7] * 9,
                        }
                    }
                }
            },
        }
        params = build_ik_params_from_config(config, nu=9, nq=9)
        assert params["regularization_strength"] == 0.2
        assert params["use_weighted_regularization"] is False
        assert params["ee_max_linear_vel"] == 0.4
        np.testing.assert_array_equal(params["joint_vel_lower"], [-0.7] * 9)
