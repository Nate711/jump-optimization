from quadruped_lib import kinematics_config, kinematics, dynamics
import cvxpy as cp
import numpy as np


class StaticTorqueOptimizer:
    def __init__(self, config: kinematics_config.KinematicsConfig, mu, tau_max):
        self.config = config
        self.leg_jac = cp.Parameter((3, 3))
        self.f = cp.Variable(3)
        self.mu = mu
        self.tau_max = tau_max
        self.objective = cp.Maximize(-self.f[2])  # negative sign bc pushing down
        self.constraints = [
            -self.tau_max <= self.leg_jac.T @ self.f,
            self.leg_jac.T @ self.f <= self.tau_max,
            cp.SOC(-self.mu * self.f[2], self.f[0:2]),
        ]  # negative sign bc pushing down
        self.problem = cp.Problem(self.objective, self.constraints)

    def solve_for_configuration(self, foot_location: np.ndarray, leg_index=0):
        """
        Using parameterization of jacobian reduces compute time per config from about 11ms to 2ms
        """
        try:
            joint_angles = kinematics.leg_inverse_kinematics_relative_to_hip(
                foot_location, leg_index, self.config
            )
        except ArithmeticError:
            return np.array((0, 0, 0))
        jac = dynamics.leg_jacobian(joint_angles, self.config, leg_index)
        self.leg_jac.value = jac
        self.problem.solve()
        return self.f.value
