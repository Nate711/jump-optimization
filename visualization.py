import static_torque_optimizer
import plot_util
from quadruped_lib import kinematics, kinematics_config
import numpy as np


class InverseKinematicsCallback:
    def __init__(self, config: kinematics_config.KinematicsConfig):
        self.config = config

    def animated_leg_callback(self, mouse_x, mouse_y):
        (x, z) = (mouse_x, mouse_y)
        y = self.config.abduction_offset_i(0)
        try:
            (abd, hip, knee) = kinematics.leg_inverse_kinematics_relative_to_hip(
                (x, y, z), leg_index=0, config=self.config
            )
        except ArithmeticError:
            return [0, 0, 0], [0, 0, 0]
        knee_position = (
            np.array((-np.sin(hip), -np.cos(hip))) * self.config.upper_link_length
        )
        return ([0, knee_position[0], x], [0, knee_position[1], z])


class FeasibleForcePoly:
    def __init__(
        self,
        config: kinematics_config.KinematicsConfig,
        n_sample_points: int,
        mu,
        tau_max: float,
    ):
        self.config = config
        self.mu = mu
        self.tau_max = tau_max

        self.optimal_fs = np.zeros((n_sample_points, 3))
        sample_angles = np.linspace(0, np.pi * 2, n_sample_points)
        self.sample_directions = np.stack(
            (np.sin(sample_angles), np.zeros(n_sample_points), np.cos(sample_angles)),
            axis=1,
        )
        self.torque_optimizer = static_torque_optimizer.StaticTorqueOptimizer(
            self.config, mu=self.mu, tau_max=self.tau_max
        )

    def force_circle_callback(self, mouse_x, mouse_y):
        (x, z) = (mouse_x, mouse_y)
        y = self.config.abduction_offset_i(0)

        for i in range(self.sample_directions.shape[0]):
            self.optimal_fs[i] = self.torque_optimizer.solve_for_configuration(
                foot_location=(x, y, z),
                leg_index=0,
                direction=self.sample_directions[i],
            )

        return (self.optimal_fs[:, 0] + x, self.optimal_fs[:, 2] + z)
