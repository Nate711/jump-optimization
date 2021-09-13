import multiprocessing
from multiprocessing.spawn import freeze_support
import quadruped_lib
from quadruped_lib import dynamics, kinematics
from quadruped_lib import kinematics_config

# from jump_config_optimization import static_torque_optimizer
import static_torque_optimizer

# from quadruped_lib import dynamics, kinematics
import cvxpy
import cvxpy as cp
import numpy as np
import time
from multiprocessing import Pool
import itertools


import matplotlib.pyplot as plt

if __name__ == '__main__':
    # freeze_support() # for multiprocessing?

    MU = 0.5
    TAU_MAX = 9.0  # 5008 9:1 actuator
    L1 = 0.10
    L2 = 0.20
    x_range = (-0.15, 0.15)
    z_range = (-L1-L2, -0.01)
    N = 30 # number of z 
    M = 120 # number of x 
    OUTLIER_PERCENTILE=0.95
    length_tolerance = 0.01

    config = kinematics_config.KinematicsConfig(
        abduction_offset=0.0,
        upper_link_length=L1,
        lower_link_length=L2,
        hip_x_offset=0,
        hip_y_offset=0,
    )
    static_torque_opt = static_torque_optimizer.StaticTorqueOptimizer(
        config,
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
            optimal_forces[i, j, :] = static_torque_opt.solve_for_configuration([launch_x, 0, foot_z])

    # remove outliers from the map. probably due to singularities
    # unrealistic that those forces would actually appear, mostly likely absorbed into compliance
    max_reasonable_force = np.quantile(abs(optimal_forces), OUTLIER_PERCENTILE)
    optimal_forces = np.clip(optimal_forces, -max_reasonable_force, max_reasonable_force)
    vertical_delta_KE = np.sum(optimal_forces, axis=0) * dz

    end = time.time()
    print("optimization took: ", end - start)

    fig, axs = plt.subplots(ncols=2, figsize=(16,8))
    img = axs[0].imshow(
        optimal_forces[:, :, 2],
        extent=(*x_range, *z_range),
        aspect="equal",
        origin="lower",
    )
    # plt.pcolormesh(optimal_forces[:,:,2], extent=(*x_range,*z_range[::-1]), aspect='equal')
    fig.colorbar(img, ax=axs[0])

    # plt.figure()
    axs[1].plot(launch_xs, vertical_delta_KE)
    axs[1].legend(["Fx", "Fy", "Fz"])
    plt.show()
