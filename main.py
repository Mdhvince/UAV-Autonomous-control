import math
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
    final_desired = np.empty((0, 11))

    for mode in modes:
        logging.info(f"Starting {mode} mode...")

        T = MinimumSnap(config, mode)
        T.generate_collision_free_trajectory()
        desired_trajectory = T.full_trajectory

        n_timesteps = desired_trajectory.shape[0]

        visited_wp = []

        for i in range(0, n_timesteps):

            while i not in visited_wp:
                des_x = desired_trajectory[i, [0, 3, 6]]
                des_y = desired_trajectory[i, [1, 4, 7]]
                des_z = desired_trajectory[i, [2, 5, 8]]
                des_yaw = desired_trajectory[i, 9]
                current_segment = desired_trajectory[i, 10]

                state_history, omega_history = fly(
                    state_history, omega_history, ctrl, quad, des_x, des_y, des_z, des_yaw, frequency
                )

                # add current waypoint to visited if quad is within 0.1m (x, y, z) of it
                if np.linalg.norm(quad.X[:3] - desired_trajectory[i, :3]) < 0.3:
                    visited_wp.append(i)
                    logging.info(f"Waypoint {i} visited.")

        logging.info(f"{mode} completed.: Quadrotor at XYZ: {np.round(quad.X[:3], 2)}")

        total_timesteps += n_timesteps
        final_desired = np.vstack((final_desired, desired_trajectory))

    sim = Sim3d(config, final_desired, state_history, T.obstacle_edges)
    ani = sim.run_sim(frames=total_timesteps+10, interval=5)
    plt.show()
    # sim.save_sim(ani, "docs/youtube/tracking_perf.mp4")


