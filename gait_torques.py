import statics
from quadruped_lib import kinematics_config
import pdb
import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from dataclasses import dataclass


@dataclass
class TrotConfig():
    mass: float
    L1: float
    L2: float
    x_velocity: float
    frequency: float
    stance_proportion: float = 0.5
    gravity: float = 9.81

    @property
    def trot_force(self):
        return self.mass * self.stance_proportion * self.gravity

    @property
    def stance_time(self):
        return 1.0 / self.frequency * self.stance_proportion

    @property
    def stride_length(self):
        return self.x_velocity * self.stance_time


def stride_joint_torques(trot_config: TrotConfig):
    DX = 0.001
    DZ = 0.01
    FOOT_Z_MIN = -0.20
    FOOT_Z_MAX = -0.05

    config = kinematics_config.KinematicsConfig(
        abduction_offset=0.0,
        upper_link_length=trot_config.L1,
        lower_link_length=trot_config.L2,
        hip_x_offset=0,
        hip_y_offset=0,
    )
    grav_comp = statics.GravityCompensation(config=config)

    foot_xs = np.arange(-trot_config.stride_length/2.0,
                        trot_config.stride_length/2.0, step=DX)
    foot_zs = np.arange(FOOT_Z_MIN, FOOT_Z_MAX, step=DZ)
    n_z = foot_zs.size
    n_x = foot_xs.size

    leg_torques = np.zeros((n_z, 2, n_x))
    for idx_z, foot_z in enumerate(foot_zs):
        for idx_x, foot_x in enumerate(foot_xs):
            torques = grav_comp.gravity_compensation(
                foot_x,
                foot_z,
                gravity_force=trot_config.trot_force)
            leg_torques[idx_z, :, idx_x] = torques[1:]

    return foot_xs, foot_zs, leg_torques


def compute_motor_heats(joint_torques, stance_proportion):
    motor_heats = np.zeros((2, joint_torques.shape[0]))
    for z_idx in range(joint_torques.shape[0]):
        for motor_idx in range(joint_torques.shape[1]):
            motor_heats[motor_idx, z_idx] = np.mean(
                joint_torques[z_idx, motor_idx, :]**2)**0.5
    return motor_heats * stance_proportion**0.5


def plot_joint_torques(foot_xs, foot_zs, leg_torques, config):
    fig = go.Figure()
    for z_idx, z in enumerate(foot_zs):
        hip = leg_torques[z_idx, 0, :]
        knee = leg_torques[z_idx, 1, :]
        fig.add_trace(go.Scatter(name=f"hip-{z:0.3f}", x=foot_xs, y=hip))
        fig.add_trace(go.Scatter(name=f"knee-{z:0.3f}", x=foot_xs, y=knee))
    fig.update_layout(
        title=f"Joint torque versus foot x position for various stance heights <br><sup>{config}</sup>",
        xaxis_title='x axis position of foot (m)',
        yaxis_title='joint torque (Nm)',
    )
    fig.show()


def plot_heat_generation(foot_zs, motor_heats, config):
    fig = go.Figure()
    fig.add_trace(go.Scatter(name="hip", x=foot_zs, y=motor_heats[0, :]))
    fig.add_trace(go.Scatter(name="knee",
                  x=foot_zs, y=motor_heats[1, :]))
    fig.update_layout(
        title=f"RMS torque versus stance height <br><sup>{config}</sup>",
        xaxis_title='stance height (m)',
        yaxis_title='RMS torque over gait cycle (Nm)',
    )
    fig.show()


if __name__ == "__main__":
    config = TrotConfig(mass=3,
                        L1=0.075,
                        L2=0.1,
                        x_velocity=0.8,
                        frequency=2.0)
    foot_xs, foot_zs, joint_torques = stride_joint_torques(config)
    motor_heats = compute_motor_heats(joint_torques, config.stance_proportion)
    plot_joint_torques(foot_xs, foot_zs, joint_torques, config)
    plot_heat_generation(foot_zs, motor_heats, config)
