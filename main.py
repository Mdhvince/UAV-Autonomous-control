import math
import time
import warnings
import logging
import configparser
from pathlib import Path
from collections import namedtuple, deque

import numpy as np
import matplotlib.pyplot as plt

from control.quadrotor import Quadrotor
from control.controller import CascadedController
from planning.minimum_snap import MinimumSnap
from simulation_3d import Sim3d

warnings.filterwarnings('ignore')

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
        filename="logs/sim.log", filemode="w", format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )

    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config_file = Path("/home/medhyvinceslas/Documents/programming/quad3d_sim/config.ini")
    config.read(config_file)

    cfg = config["DEFAULT"]
    frequency = cfg.getint("frequency")

    ctrl = CascadedController(config)
    quad = Quadrotor(config)
    state_history, omega_history = quad.X, quad.omega

    modes = ["takeoff", "flight", "landing"]
    total_timesteps = 0
    combined_desired_trajectory = np.empty((0, 11))

    for mode in modes:
        logging.info(f"Starting {mode} mode...")

        T = MinimumSnap(config, mode)
        desired_trajectory = T.get_trajectory()

        logging.info("Optimized trajectory successfully generated")

        total_timesteps += desired_trajectory.shape[0]
        combined_desired_trajectory = np.vstack((combined_desired_trajectory, desired_trajectory))

        while True:

            des_x = desired_trajectory[0, [0, 3, 6]]
            des_y = desired_trajectory[0, [1, 4, 7]]
            des_z = desired_trajectory[0, [2, 5, 8]]
            des_yaw = desired_trajectory[0, 9]

            state_history, omega_history = fly(
                state_history, omega_history, ctrl, quad, des_x, des_y, des_z, des_yaw, frequency
            )

            target_has_been_reached = np.linalg.norm(quad.X[:3] - desired_trajectory[0, :3]) < 0.5

            if target_has_been_reached:
                desired_trajectory = np.delete(desired_trajectory, 0, axis=0)  # remove current waypoint from desired
                logging.info(f"Waypoint {round(des_x[0], 1), round(des_y[0], 1), round(des_z[0], 1)} visited.")

            if desired_trajectory.shape[0] == 0:  # if all waypoints have been visited
                break

        logging.info(f"{mode} completed.: Quadrotor at XYZ: {np.round(quad.X[:3], 2)}")


    sim = Sim3d(config, combined_desired_trajectory, state_history)
    ani = sim.run_sim(frames=total_timesteps+10, interval=5)
    plt.show()
    # sim.save_sim(ani, "docs/youtube/test.mp4")



