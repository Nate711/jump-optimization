import multiprocessing
from multiprocessing.spawn import freeze_support
import matplotlib
from matplotlib import animation
import quadruped_lib
from quadruped_lib import dynamics, kinematics
from quadruped_lib import kinematics_config

# from jump_config_optimization import static_torque_optimizer
import static_torque_optimizer
import visualization
import plot_util

# from quadruped_lib import dynamics, kinematics
import cvxpy
import cvxpy as cp
import numpy as np
import time
from multiprocessing import Pool, Value
import itertools

import matplotlib.pyplot as plt

from matplotlib import widgets

if __name__ == "__main__":
    # freeze_support() # for multiprocessing?

    MU = 1.0
    # TAU_MAX = 9.0  # 5008 9:1 actuator
    TAU_MAX = 4.0  # GIM4305 10:1 actuator
    # mu=1.0
    # (0.15, 0.15) -> 44.5
    # (0.14, 0.16) -> 48
    # (0.13, 0.17) -> 51.3
    # (0.12, 0.18) -> 52.5
    # (0.11, 0.19) -> 52.4
    L1 = 0.095
    L2 = 0.10
    x_range = (-0.15, 0.05)
    z_range = (-L1 - L2, -0.01)
    N = 50  # number of x options to evaluate
    M = 100  # number of z positions to integrate over

    MAX_FORCE = 225  # 30Gs of acceleration!
    OUTLIER_PERCENTILE = 0.95

    config = kinematics_config.KinematicsConfig(
        abduction_offset=0.0,
        upper_link_length=L1,
        lower_link_length=L2,
        hip_x_offset=0,
        hip_y_offset=0,
    )
    static_torque_opt = static_torque_optimizer.StaticTorqueOptimizer(
        config,
        direction=np.array([0, 0, -1]),
        mu=MU,
        tau_max=TAU_MAX,
    )

    start = time.time()

    launch_xs = np.linspace(*x_range, N)
    eval_zs = np.linspace(*z_range, M)
    dz = eval_zs[1] - eval_zs[0]
    optimal_forces = np.zeros((M, N, 3))

    # multiprocessing method took 10% longer
    # pool = multiprocessing.Pool(multiprocessing.cpu_count())
    # foot_locations = itertools.product(launch_xs, [0.0], eval_zs)
    # optimal_forces_flat = pool.map(static_torque_opt.solve_for_configuration, foot_locations)

    for i, foot_z in enumerate(eval_zs):
        for j, launch_x in enumerate(launch_xs):
            optimal_forces[i, j, :] = static_torque_opt.solve_for_configuration(
                np.array([launch_x, 0, foot_z])
            )

    # remove outliers from the map. probably due to singularities
    # unrealistic that those forces would actually appear, mostly likely absorbed into compliance
    max_reasonable_force = (
        MAX_FORCE if MAX_FORCE else np.quantile(
            abs(optimal_forces), OUTLIER_PERCENTILE)
    )
    optimal_forces = np.clip(
        optimal_forces, -max_reasonable_force, max_reasonable_force
    )
    vertical_delta_KE = np.sum(optimal_forces, axis=0) * dz  # Shape: (N, 3)
    optimal_fz_work = np.max(abs(vertical_delta_KE[:, 2]))
    optimal_x_offset = launch_xs[np.argmax(abs(vertical_delta_KE[:, 2]))]

    end = time.time()
    print("optimization took: ", end - start)

    fig, axs = plt.subplots(ncols=2, figsize=(16, 8))
    img = axs[0].imshow(
        optimal_forces[:, :, 2],
        extent=(*x_range, *z_range),
        aspect="equal",
        origin="lower",
    )

    leg = visualization.InverseKinematicsCallback(
        config,
    )

    feasible_force = visualization.FeasibleForcePoly(
        config,
        n_sample_points=40,
        mu=MU,
        tau_max=0.005,
    )

    gravity_comp_joint_torques = visualization.GravityCompensationCallback(
        config,
        gravity_force=15.0,  # N
    )

    cursor_widget = plot_util.CursorCallbackWidget(
        ax=axs[0],
        move_callbacks=[
            leg.animated_leg_callback,
            feasible_force.force_circle_callback,
            gravity_comp_joint_torques.gravity_compensation_callback,
        ],
        lineprops=[{"color": "k", "linewidth": 10},
                   {"marker": ".", "color": "r"}, {}],
    )

    fig.colorbar(img, ax=axs[0])
    axs[0].set_title(
        "Maximum achievable vertical force as function of leg configuration"
    )

    # plt.figure()
    axs[1].plot(launch_xs, vertical_delta_KE)
    axs[1].legend(["Fx", "Fy", "Fz"])
    axs[1].set_title(
        f"Max fz work of {optimal_fz_work:0.2f} with x offset {optimal_x_offset:0.2f}"
    )
    plt.show()
