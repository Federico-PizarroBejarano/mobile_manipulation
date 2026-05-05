from abc import ABC, abstractmethod

import casadi as cs
import numpy as np

from mm_utils.casadi_struct import casadi_sym_struct


class Constraint(ABC):
    def __init__(self, nx, nu, name):
        """MPC constraints base class.

        Args:
            nx (int): State dimension.
            nu (int): Control dimension.
            name (str): Name to identify the constraint.
        """

        self.nx = nx
        self.nu = nu
        self.name = name

        self.x_sym = cs.MX.sym("x", nx)
        self.u_sym = cs.MX.sym("u", nu)

        self.p_dict = None
        self.p_sym = None

        self.slack_enabled = False

        super().__init__()

    @abstractmethod
    def check(self, x, u, p):
        """Check constraint satisfaction.

        Args:
            x (ndarray): State vector.
            u (ndarray): Control input vector.
            p (ndarray): Parameter vector.

        Returns:
            ndarray: Constraint values (should be < 0 for satisfaction).
        """
        pass

    def get_p_dict(self, sym=True):
        """Get parameter dictionary with name suffixes.

        Args:
            sym (bool): If True, return symbolic parameters; if False, return zero matrices.

        Returns:
            dict or None: Parameter dictionary with keys suffixed by constraint name, or None if no parameters.
        """
        if self.p_dict is None:
            return None
        else:
            if sym:
                return {
                    key + f"_{self.name}": val for (key, val) in self.p_dict.items()
                }
            else:
                return {
                    key + f"_{self.name}": cs.DM.zeros(val.shape)
                    for (key, val) in self.p_dict.items()
                }

    def get_p_dict_default(self):
        """Get default parameter dictionary (zero values).

        Returns:
            dict or None: Default parameter dictionary with zero values, or None if no parameters.
        """
        p_dict = self.get_p_dict(False)
        return p_dict


class NonlinearConstraint(Constraint):
    def __init__(self, nx, nu, ng, g_fcn, p_dict, constraint_name):
        """Nonlinear inequality constraint g(x, u, p) < 0.

        Args:
            nx (int): State dimension.
            nu (int): Control dimension.
            ng (int): Constraint dimension.
            g_fcn (casadi.Function): Function g_fcn(x_bar_sym, u_bar_sym, *params_sym).
            p_dict (dict): Parameter dictionary.
            constraint_name (str): Name to identify the constraint.
        """

        super().__init__(nx, nu, constraint_name)
        self.ng = ng

        self.p_dict = p_dict
        self.p_struct = casadi_sym_struct(p_dict)
        self.p_sym = self.p_struct.cat

        self.g_fcn = g_fcn
        if self.g_fcn is not None:
            self.g_eqn = g_fcn(self.x_sym, self.u_sym, self.p_sym)
        else:
            self.g_eqn = None

    def check(self, x, u, p):
        """Check nonlinear constraint satisfaction.

        Args:
            x (ndarray): State vector.
            u (ndarray): Control input vector.
            p (ndarray): Parameter vector.

        Returns:
            ndarray: Constraint values (should be < 0 for satisfaction).
        """
        g = self.g_fcn(x, u, p)

        return g


class SignedDistanceConstraint(NonlinearConstraint):
    def __init__(self, robot_mdl, signed_distance_fcn, d_safe, name="obstacle"):
        """Signed Distance Constraint: -(sd(x_k, param1, param2, ...) - d_safe) < 0.

        Args:
            robot_mdl (MobileManipulator3D): Robot model.
            signed_distance_fcn (casadi.Function): Signed distance model.
            d_safe (float): Safe clearance, scalar, same for all body pairs.
            name (str): Name of this constraint.
        """
        nx = robot_mdl.ssSymMdl["nx"]
        nu = robot_mdl.ssSymMdl["nu"]
        nq = robot_mdl.q_sym.size()[0]
        ng = signed_distance_fcn.size_out(0)[0]
        p_sym = signed_distance_fcn.mx_in()[1:]
        p_name = signed_distance_fcn.name_in()[1:]
        p_dict = {name: sym for (name, sym) in zip(p_name, p_sym)}

        super().__init__(nx, nu, ng, None, p_dict, name)

        self.g_eqn = (
            -signed_distance_fcn(
                self.x_sym[:nq], *[self.p_struct[k] for k in self.p_dict.keys()]
            )
            + d_safe
        )

        self.g_fcn = cs.Function(
            "g_" + self.name, [self.x_sym, self.u_sym, self.p_sym], [self.g_eqn]
        )

        self.g_grad_eqn = cs.jacobian(self.g_eqn, cs.veccat(self.u_sym, self.x_sym))
        self.g_grad_fcn = cs.Function(
            "g_grad", [self.x_sym, self.u_sym, self.p_sym], [self.g_grad_eqn]
        )

        self.slack_enabled = True


class StateBoxConstraints(NonlinearConstraint):
    def __init__(self, robot_mdl, name="state"):
        """State Box Constraint: lb_x < x < ub_x.

        Args:
            robot_mdl (MobileManipulator3D): Robot model.
            name (str): Name of this constraint.
        """
        nx = robot_mdl.ssSymMdl["nx"]
        nu = robot_mdl.ssSymMdl["nu"]
        ng = nx * 2
        p_dict = {}
        super().__init__(nx, nu, ng, None, p_dict, name)

        self.g_eqn = cs.vertcat(
            self.x_sym - robot_mdl.ssSymMdl["ub_x"],
            robot_mdl.ssSymMdl["lb_x"] - self.x_sym,
        )
        self.g_fcn = cs.Function(
            "g_" + self.name, [self.x_sym, self.u_sym, self.p_sym], [self.g_eqn]
        )


class ControlBoxConstraints(NonlinearConstraint):
    def __init__(self, robot_mdl, name="control"):
        """Control Box Constraint: lb_u < u < ub_u.

        Args:
            robot_mdl (MobileManipulator3D): Robot model.
            name (str): Name of this constraint.
        """
        nx = robot_mdl.ssSymMdl["nx"]
        nu = robot_mdl.ssSymMdl["nu"]
        ng = nu * 2
        p_dict = {}
        super().__init__(nx, nu, ng, None, p_dict, name)

        self.g_eqn = cs.vertcat(
            self.u_sym - robot_mdl.ssSymMdl["ub_u"],
            robot_mdl.ssSymMdl["lb_u"] - self.u_sym,
        )
        self.g_fcn = cs.Function(
            "g_" + self.name, [self.x_sym, self.u_sym, self.p_sym], [self.g_eqn]
        )


class AlignedToolConstraint(NonlinearConstraint):
    """Tool-acceleration alignment constraint."""

    _WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=float)
    _GRAVITY_MAG = 9.81

    def __init__(
        self,
        robot_mdl,
        eps_align=1e-3,
        name="aligned",
    ):
        """Enforce ``|cross(z_tool, f_dir)_i| <= eps_align`` and ``dot(z_tool, f_dir) >= 0``.

        Args:
            robot_mdl (MobileManipulator3D): Robot model.
            eps_align (float): Component-wise tolerance for normalized cross terms.
            name (str): Constraint name.
        """
        nx = robot_mdl.ssSymMdl["nx"]
        nu = robot_mdl.ssSymMdl["nu"]
        nq = int(robot_mdl.q_sym.size1())
        ng = 7
        p_dict = {}
        super().__init__(nx, nu, ng, None, p_dict, name)

        tool_name = robot_mdl.tool_link_name
        fk_tool = robot_mdl.kinSymMdls[tool_name]
        _, C_world_tool = fk_tool(self.x_sym[:nq])

        a_tool_world, _, _ = tool_origin_linear_accel_world_expr(
            robot_mdl, self.x_sym, self.u_sym
        )
        g_vec = -float(self._GRAVITY_MAG) * cs.DM(self._WORLD_UP)
        a_eff = a_tool_world - g_vec

        z_tool_world = C_world_tool @ cs.DM([0.0, 0.0, 1.0])
        f_dir = a_eff / (cs.norm_2(a_eff) + float(1e-6))
        cross_vec = cs.cross(z_tool_world, f_dir)
        dot_val = cs.dot(z_tool_world, f_dir)
        tol = float(eps_align)
        self.g_eqn = cs.vertcat(cross_vec - tol, -cross_vec - tol, -dot_val)
        self.g_fcn = cs.Function(
            "g_" + self.name, [self.x_sym, self.u_sym, self.p_sym], [self.g_eqn]
        )
        self.slack_enabled = True


def tool_origin_linear_accel_world_expr(robot_mdl, x_sym, u_sym):
    """Tool linear acceleration from first-order kinematics: ``a_tool = J(q) qdd``.

    Args:
        robot_mdl: ``MobileManipulator3D`` instance.
        x_sym (cs.MX): MPC state, shape ``(nx,)``.
        u_sym (cs.MX): MPC input (generalized acceleration), shape ``(nu,)``.

    Returns:
        tuple[cs.MX, cs.MX, cs.MX]: ``(a_tool_world, J_pos, jdot_qdot)``.
    """
    nq = int(robot_mdl.q_sym.size1())
    q_live = x_sym[:nq]
    qdd = u_sym
    q_alg = cs.MX.sym("q_alg", nq)
    fk_tool = robot_mdl.kinSymMdls[robot_mdl.tool_link_name]
    p_tool, _ = fk_tool(q_alg)
    J_pos = cs.jacobian(p_tool, q_alg)
    jdot_qdot = cs.DM.zeros(3, 1)
    a_tool_world = cs.mtimes(J_pos, qdd)
    a_tool_world = cs.substitute(a_tool_world, q_alg, q_live)
    J_pos = cs.substitute(J_pos, q_alg, q_live)
    return a_tool_world, J_pos, jdot_qdot
