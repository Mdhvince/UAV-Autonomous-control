import warnings
import logging
import configparser
from pathlib import Path
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

from control.quadrotor import Quadrotor
from control.controller import CascadedController
from planning.minimum_snap import MinimumSnap
from simulation_3d import Sim3d

warnings.filterwarnings('ignore')


# def takeoff(quad, controller, state_history, omega_history, frequency):
#     """
#     Takeoff of the quadrotor from z=0 to z=z_des
#     """
#     # desired state at the end of the takeoff
#     des_x = np.array([quad.X[0], 0.0, 0.0])
#     des_y = np.array([quad.X[1], 0.0, 0.0])
#     des_z = np.array([1.0, 0.0, 0.0])
#     des_yaw = quad.X[5]
#
#     steps_to_reach_z = 0
#     # while the quadrotor is not at the desired height +- 0.1
#     while not (des_z[0] - 0.1 <= quad.X[2] <= des_z[0] + 0.1):
#         steps_to_reach_z += 1
#         state_history, omega_history = fly(
#             state_history, omega_history, controller, quad, des_x, des_y, des_z, des_yaw, frequency
#         )
#     return state_history, omega_history, steps_to_reach_z

def fly(state_history, omega_history, controller, quad, des_x, des_y, des_z, des_yaw, frequency):
    R = quad.R()
    F_cmd = controller.altitude(quad, des_z, R)
    bxy_cmd = controller.lateral(quad, des_x, des_y, F_cmd)
    pqr_cmd = controller.reduced_attitude(quad, bxy_cmd, des_yaw, R)
    for _ in range(frequency):
        # flight controller
        moment_cmd = controller.body_rate_controller(quad, pqr_cmd)
        quad.set_propeller_speed(F_cmd, moment_cmd)
        quad.update_state()

    state_history = np.vstack((state_history, quad.X))
    omega_history = np.vstack((omega_history, quad.omega))

    return state_history, omega_history


if __name__ == "__main__":

    logging.basicConfig(
        filename="logs/sim.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

    Desired = namedtuple("Desired", ["x", "y", "z", "x_vel", "y_vel", "z_vel", "x_acc", "y_acc", "z_acc", "yaw"])
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config_file = Path("/home/medhyvinceslas/Documents/programming/quad3d_sim/config.ini")
    config.read(config_file)
    sim_config = config["SIMULATION"]

    frequency = sim_config.getint("frequency")

    T = MinimumSnap(config)
    T.generate_collision_free_trajectory()
    desired_trajectory = T.full_trajectory

    ctrl = CascadedController(config)
    quad = Quadrotor(config, desired_trajectory)
    logging.info(f"Quadrotor initialized at XYZ: {np.round(quad.X[:3], 2)}")

    state_history, omega_history = quad.X, quad.omega
    n_timesteps = desired_trajectory.shape[0]

    # logging.info("Takeoff 🚀...")
    # state_history, omega_history, takeoff_steps = takeoff(quad, ctrl, state_history, omega_history, frequency)
    # logging.info(f"Takeoff completed: Quadrotor at XYZ: {np.round(quad.X[:3], 2)}")


    logging.info("Flying 🚀...")

    for i in range(0, n_timesteps):
        des_x = desired_trajectory[i, [0, 3, 6]]
        des_y = desired_trajectory[i, [1, 4, 7]]
        des_z = desired_trajectory[i, [2, 5, 8]]
        des_yaw = desired_trajectory[i, 9]
        current_segment = desired_trajectory[i, 10]

        # log the current segment only if it is different from the previous one
        if i == 0 or current_segment != desired_trajectory[i-1, 10]:
            logging.info(f"Completing segment n° {int(current_segment)} ...")

        state_history, omega_history = fly(
            state_history, omega_history, ctrl, quad, des_x, des_y, des_z, des_yaw, frequency
        )

    logging.info(f"Flight completed.: Quadrotor at XYZ: {np.round(quad.X[:3], 2)}")

    sim = Sim3d(config, desired_trajectory, state_history, T.obstacle_edges)
    ani = sim.run_sim(frames=n_timesteps, interval=5)
    plt.show()
    # sim.save_sim(ani, "docs/youtube/tracking_perf.mp4")


