import statics
from quadruped_lib import kinematics_config
import pdb
import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp


def stride_joint_torques():
    # MASS #
    GRAVITY_FORCE = 13.8  # N

    # GEOMETRY #
    L1 = 0.10
    L2 = 0.10

    # GAIT #
    X_VELOCITY = 0.8  # m/s
    GAIT_FREQ = 2.0  # Hz
    STANCE_PROPORTION = 0.5  # unitless
    N_X = 100
    N_Z = 16
    FOOT_Z_MIN = -0.20
    FOOT_Z_MAX = -0.05

    stance_time = 1.0 / GAIT_FREQ * STANCE_PROPORTION
    stride_length = X_VELOCITY * stance_time

    config = kinematics_config.KinematicsConfig(
        abduction_offset=0.0,
        upper_link_length=L1,
        lower_link_length=L2,
        hip_x_offset=0,
        hip_y_offset=0,
    )
    grav_comp = statics.GravityCompensation(config=config)

    foot_xs = np.linspace(-stride_length/2.0, stride_length/2.0, N_X)
    foot_zs = np.linspace(FOOT_Z_MIN, FOOT_Z_MAX, N_Z)

    leg_torques = np.zeros((N_Z, 2, N_X))
    for idx_z, foot_z in enumerate(foot_zs):
        for idx_x, foot_x in enumerate(foot_xs):
            torques = grav_comp.gravity_compensation(
                foot_x,
                foot_z,
                gravity_force=GRAVITY_FORCE)
            leg_torques[idx_z, :, idx_x] = torques[1:]

    return foot_xs, foot_zs, leg_torques


def compute_motor_heats(joint_torques):
    motor_heats = np.zeros((2, joint_torques.shape[0]))
    for z_idx in range(joint_torques.shape[0]):
        for motor_idx in range(joint_torques.shape[1]):
            motor_heats[motor_idx, z_idx] = np.mean(
                joint_torques[z_idx, motor_idx, :]**2)**0.5
    return motor_heats


def plot_joint_torques(foot_xs, foot_zs, leg_torques):
    fig = go.Figure()
    for z_idx, z in enumerate(foot_zs):
        hip = leg_torques[z_idx, 0, :]
        knee = leg_torques[z_idx, 1, :]
        fig.add_trace(go.Scatter(name=f"hip-{z:0.3f}", x=foot_xs, y=hip))
        fig.add_trace(go.Scatter(name=f"knee-{z:0.3f}", x=foot_xs, y=knee))
    fig.show()


def plot_heat_generation(foot_zs, motor_heats):
    fig = go.Figure()
    fig.add_trace(go.Scatter(name="hip-power", x=foot_zs, y=motor_heats[0, :]))
    fig.add_trace(go.Scatter(name="knee-power",
                  x=foot_zs, y=motor_heats[1, :]))
    fig.show()


if __name__ == "__main__":
    foot_xs, foot_zs, joint_torques = stride_joint_torques()
    motor_heats = compute_motor_heats(joint_torques)
    plot_joint_torques(foot_xs, foot_zs, joint_torques)
    plot_heat_generation(foot_zs, motor_heats)
