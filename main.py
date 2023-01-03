import warnings
import configparser
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

from control.simulation_3d import Sim3d
from control.quadrotor import Quadrotor
from control.controller import TFC
from planning.trajectory import TrajectoryPlanner, getwp

warnings.filterwarnings('ignore')


if __name__ == "__main__":

    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config_file = "/home/medhyvinceslas/Documents/programming/quad3d_sim/config.ini"
    config.read(config_file)

    inner_loop_relative_to_outer_loop = 10
    dt = 0.01
    velocity = 2.0

    waypoints = np.array([[10, 0, 0], [10, 4, 1], [6, 5, 1.5], [7, 8, 1.5], [2, 7, 2], [1, 0, 2]])
    # waypoints = np.array([[0, 0, 10], [0, 10, 10], [10, 10, 10], [10, 0, 10], [5, 5, 20]])
    # waypoints = getwp("angle").T

    planner = TrajectoryPlanner(waypoints, velocity, dt)
    r_des = planner.get_min_snap_trajectory()


    Desired = namedtuple("Desired", ["x", "y", "z", "x_vel", "y_vel", "z_vel", "x_acc", "y_acc", "z_acc", "yaw"])
    desired = Desired(
        r_des[:, 0], r_des[:, 1], r_des[:, 2],     # position
        r_des[:, 3], r_des[:, 4], r_des[:, 5],     # velocity
        r_des[:, 6], r_des[:, 7], r_des[:, 8], 0)  # acc and yaw
    
    quad = Quadrotor(config, desired)
    controller = TFC(config)

    state_history, omega_history = quad.X, quad.omega    
    n_waypoints = desired.z.shape[0]

    for i in range(0, n_waypoints):
        # flight computer
        R = quad.R()
        F_cmd = controller.altitude(quad, desired, R, dt, index=i)
        bxy_cmd = controller.lateral(quad, F_cmd, desired, index=i)
        pqr_cmd = controller.reduced_attitude(quad, bxy_cmd, desired.yaw, R)
        
        for _ in range(inner_loop_relative_to_outer_loop):
            # flight controller
            moment_cmd = controller.body_rate_controller(quad, pqr_cmd)
            quad.set_propeller_speed(F_cmd, moment_cmd)
            quad.update_state(dt/inner_loop_relative_to_outer_loop)


        state_history = np.vstack((state_history, quad.X))
        omega_history = np.vstack((omega_history, quad.omega))



    sim = Sim3d(r_des, state_history)
    _ = sim.run_sim(frames=n_waypoints, interval=5)
    plt.show()

