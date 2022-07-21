from quadruped_lib import kinematics_config, kinematics, dynamics

import numpy as np


class GravityCompensation:
    def __init__(
        self,
        config: kinematics_config.KinematicsConfig,
    ):
        self.config = config

    def gravity_compensation(self, x, z, gravity_force):
        y = self.config.abduction_offset_i(0)
        foot_location = np.array((x, y, z))
        leg_index = 0
        try:
            joint_angles = kinematics.leg_inverse_kinematics_relative_to_hip(
                foot_location, leg_index=leg_index, config=self.config
            )
        except ArithmeticError:
            return np.array((0, 0, 0))
        jac = dynamics.leg_jacobian(joint_angles, self.config, leg_index)
        grav_vector = np.array((0, 0, gravity_force))
        joint_torques = jac.T @ grav_vector
        return joint_torques
