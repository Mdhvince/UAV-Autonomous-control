import warnings
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

def takeoff(quad, controller, state_history, omega_history, des_x, des_y, des_z, des_yaw, frequency):
    """
    Takeoff of the quadrotor from z=0 to z=z_des
    """
    print("Taking off...")
    steps_to_reach_z = 0
    # while the quadrotor is not at the desired height +- 0.1
    while not (des_z[0] - 0.1 <= quad.X[2] <= des_z[0] + 0.1):
        steps_to_reach_z += 1
        state_history, omega_history = fly(
            state_history, omega_history, controller, quad, des_x, des_y, des_z, des_yaw, frequency
        )

    print("Takeoff completed.")
    return state_history, omega_history, steps_to_reach_z


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

    Desired = namedtuple("Desired", ["x", "y", "z", "x_vel", "y_vel", "z_vel", "x_acc", "y_acc", "z_acc", "yaw"])
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config_file = Path("/home/medhyvinceslas/Documents/programming/quad3d_sim/config.ini")
    config.read(config_file)
    sim_config = config["SIMULATION"]

    frequency = sim_config.getint("frequency")

    T = MinimumSnap(config)
    T.generate_collision_free_trajectory()
    r_des = T.full_trajectory

    desired = Desired(
        r_des[:, 0], r_des[:, 1], r_des[:, 2],          # desired position over time
        r_des[:, 3], r_des[:, 4], r_des[:, 5],          # desired velocity over time
        r_des[:, 6], r_des[:, 7], r_des[:, 8],          # desired acceleration over time
        np.zeros(r_des.shape[0])                        # 0 yaw for now
    )

    ctrl = CascadedController(config)
    quad = Quadrotor(config, desired)

    state_history, omega_history = quad.X, quad.omega
    n_waypoints = desired.z.shape[0]

    # takeoff
    des_x = np.array([quad.X[0], 0.0, 0.0])
    des_y = np.array([quad.X[1], 0.0, 0.0])
    des_z = np.array([1.0, 0.0, 0.0])
    des_yaw = quad.X[5]

    state_history, omega_history, takeoff_steps = takeoff(
        quad, ctrl, state_history, omega_history, des_x, des_y, des_z, des_yaw, frequency
    )

    print("Flying...")
    for i in range(0, n_waypoints):
        # flight computer
        des_x = np.array([desired.x[i], desired.x_vel[i], desired.x_acc[i]])
        des_y = np.array([desired.y[i], desired.y_vel[i], desired.y_acc[i]])
        des_z = np.array([desired.z[i], desired.z_vel[i], desired.z_acc[i]])
        des_yaw = desired.yaw[i]
        state_history, omega_history = fly(
            state_history, omega_history, ctrl, quad, des_x, des_y, des_z, des_yaw, frequency
        )

    print("Flight completed.")

    sim = Sim3d(config, takeoff_steps, r_des, state_history, T.obstacle_edges)
    ani = sim.run_sim(frames=n_waypoints, interval=5)
    plt.show()
    # sim.save_sim(ani, "docs/youtube/tracking_perf.mp4")
