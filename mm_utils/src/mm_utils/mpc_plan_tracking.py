"""MPC plan interpolation and split-rate joint-space velocity tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy.interpolate import interp1d

from mm_utils.math import wrap_pi_scalar

CmdVelType = Literal["integration", "interpolation"]


@dataclass
class PlanInterpolators:
    """Interpolators built from one MPC solution."""

    mpc_dt: float
    cmd_vel_type: str
    u_interp: Callable[[float], np.ndarray] | None
    v_interp: Callable[[float], np.ndarray] | None
    q_interp: Callable[[float], np.ndarray] | None
    nu: int
    dof: int


def build_plan_interpolators(
    mpc_dt: float,
    u_bar: np.ndarray,
    v_bar: np.ndarray,
    q_bar: np.ndarray | None,
    cmd_vel_type: str,
) -> PlanInterpolators:
    """Build scipy interpolators over the MPC horizon (matches ``mpc_ros`` time grids).

    Args:
        mpc_dt: MPC discretization ``dt`` (seconds).
        u_bar: Control trajectory, shape ``(N, nu)``.
        v_bar: Velocity trajectory, shape ``(N+1, nu)``.
        q_bar: Joint reference trajectory, shape ``(N+1, dof)``, or None to skip P path.
        cmd_vel_type: ``\"integration\"`` or ``\"interpolation\"``.

    Returns:
        PlanInterpolators: Callable interpolators and dimensions.
    """
    u_bar = np.asarray(u_bar, dtype=float)
    v_bar = np.asarray(v_bar, dtype=float)
    nu = int(u_bar.shape[1])
    n_u = int(u_bar.shape[0])
    n_v = int(v_bar.shape[0])

    u_interp = None
    v_interp = None
    if cmd_vel_type == "integration":
        t_u = np.arange(n_u) * mpc_dt
        u_interp = interp1d(
            t_u,
            u_bar,
            axis=0,
            bounds_error=False,
            fill_value=np.zeros(nu),
        )
    elif cmd_vel_type == "interpolation":
        t_v = np.arange(n_v) * mpc_dt
        v_interp = interp1d(
            t_v,
            v_bar,
            axis=0,
            bounds_error=False,
            fill_value="extrapolate",
        )
    else:
        raise ValueError(f"Unknown cmd_vel_type: {cmd_vel_type}")

    q_interp = None
    dof = 0
    if q_bar is not None and np.asarray(q_bar).size > 0:
        q_bar = np.asarray(q_bar, dtype=float)
        dof = int(q_bar.shape[1])
        t_q = np.arange(q_bar.shape[0]) * mpc_dt
        q_interp = interp1d(
            t_q,
            q_bar,
            axis=0,
            bounds_error=False,
            fill_value="extrapolate",
        )

    return PlanInterpolators(
        mpc_dt=float(mpc_dt),
        cmd_vel_type=cmd_vel_type,
        u_interp=u_interp,
        v_interp=v_interp,
        q_interp=q_interp,
        nu=nu,
        dof=dof,
    )


def _joint_error_to_nu(q_ref: np.ndarray, q_meas: np.ndarray, nu: int) -> np.ndarray:
    """Map joint-space error to length ``nu`` (pad or truncate)."""
    q_ref = np.asarray(q_ref, dtype=float).reshape(-1)
    q_meas = np.asarray(q_meas, dtype=float).reshape(-1)
    dof = min(len(q_ref), len(q_meas))
    err = np.zeros(nu, dtype=float)
    n = min(nu, dof)
    err[:n] = q_ref[:n] - q_meas[:n]
    if n > 2:
        err[2] = wrap_pi_scalar(err[2])
    return err


def low_level_velocity_step(
    u_cmd: np.ndarray,
    t_elapsed: float,
    sim_dt: float,
    interps: PlanInterpolators,
    q_meas: np.ndarray,
    kp: np.ndarray | None,
    lb_u: np.ndarray | None,
    ub_u: np.ndarray | None,
    lpf_alpha: float = 1.0,
    return_diagnostics: bool = False,
):
    """Feedforward from MPC plan plus optional joint P correction, then clamp.

    Integration mode: ``u_next = u_cmd + u(t_elapsed) * sim_dt`` with ``u`` from MPC.
    Interpolation mode: ``u_next = v(t_elapsed + mpc_dt)`` from MPC velocity trajectory.

    Optional: ``u_next += kp * (q_ref - q)`` using ``q_ref`` from ``q_bar`` horizon.
    Then ``u_next = alpha * u_next + (1 - alpha) * u_cmd`` if ``lpf_alpha < 1``.

    Args:
        u_cmd: Current commanded generalized velocity, shape ``(nu,)``.
        t_elapsed: Time since current plan start (seconds).
        sim_dt: Low-level / simulation integration step (seconds).
        interps: Interpolators from ``build_plan_interpolators``.
        q_meas: Measured joint positions.
        kp: Optional length-``nu`` gains; None or all-zero disables P term.
        lb_u: Lower velocity bounds, shape ``(nu,)``, or None.
        ub_u: Upper velocity bounds, shape ``(nu,)``, or None.
        lpf_alpha: Weight on the new sample in ``[0, 1]``. ``1.0`` disables the filter.

    Returns:
        np.ndarray: Updated velocity command, shape ``(nu,)``.
        If ``return_diagnostics`` is True, returns ``(out, dict)`` with keys
        ``v_ff``, ``q_ref`` (or None), ``v_cmd`` (after clamp).
    """
    nu = interps.nu
    if interps.cmd_vel_type == "integration":
        if interps.u_interp is None:
            v_ff = np.asarray(u_cmd, dtype=float).reshape(-1)[:nu].copy()
        else:
            acc = np.asarray(interps.u_interp(t_elapsed), dtype=float).reshape(-1)[:nu]
            v_ff = np.asarray(u_cmd, dtype=float).reshape(-1)[:nu] + acc * float(sim_dt)
    else:
        if interps.v_interp is None:
            v_ff = np.zeros(nu, dtype=float)
        else:
            v_ff = np.asarray(
                interps.v_interp(t_elapsed + interps.mpc_dt), dtype=float
            ).reshape(-1)[:nu]

    v_ff = np.asarray(v_ff, dtype=float).reshape(-1)[:nu].copy()
    out = v_ff.copy()
    q_ref_diag = None
    if interps.q_interp is not None and interps.dof > 0:
        q_ref_diag = (
            np.asarray(interps.q_interp(t_elapsed), dtype=float).reshape(-1).copy()
        )

    if kp is not None and interps.q_interp is not None and interps.dof > 0:
        kp = np.asarray(kp, dtype=float).reshape(-1)
        if kp.size == 1:
            kp = np.full(nu, float(kp[0]))
        else:
            kp = np.resize(kp, nu)
        if np.any(kp != 0.0):
            q_ref = (
                q_ref_diag
                if q_ref_diag is not None
                else np.asarray(interps.q_interp(t_elapsed), dtype=float).reshape(-1)
            )
            err = _joint_error_to_nu(q_ref, q_meas, nu)
            out = out + kp * err

    alpha = float(lpf_alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"lpf_alpha must be in (0, 1], got {alpha}")
    if alpha < 1.0:
        prev = np.asarray(u_cmd, dtype=float).reshape(-1)[:nu]
        out = alpha * out + (1.0 - alpha) * prev

    if lb_u is not None and ub_u is not None:
        lb_u = np.asarray(lb_u, dtype=float).reshape(-1)
        ub_u = np.asarray(ub_u, dtype=float).reshape(-1)
        n = min(len(out), len(lb_u), len(ub_u))
        out[:n] = np.clip(out[:n], lb_u[:n], ub_u[:n])

    if return_diagnostics:
        return out, {"v_ff": v_ff, "q_ref": q_ref_diag, "v_cmd": out.copy()}
    return out


def low_level_velocity_step_simple(
    u_cmd: np.ndarray,
    t_elapsed: float,
    sim_dt: float,
    mpc_dt: float,
    cmd_vel_type: str,
    u_bar: np.ndarray,
    v_bar: np.ndarray,
    q_bar: np.ndarray | None,
    q_meas: np.ndarray,
    kp: np.ndarray | None,
    lb_u: np.ndarray | None,
    ub_u: np.ndarray | None,
    lpf_alpha: float = 1.0,
    return_diagnostics: bool = False,
):
    """Convenience: rebuild interpolators each call."""
    interps = build_plan_interpolators(mpc_dt, u_bar, v_bar, q_bar, cmd_vel_type)
    return low_level_velocity_step(
        u_cmd,
        t_elapsed,
        sim_dt,
        interps,
        q_meas,
        kp,
        lb_u,
        ub_u,
        lpf_alpha=lpf_alpha,
        return_diagnostics=return_diagnostics,
    )
