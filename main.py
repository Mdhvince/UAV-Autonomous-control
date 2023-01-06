import warnings
import configparser
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

from control.simulation_3d import Sim3d
from control.quadrotor import Quadrotor
from control.controller import TFC
from planning.trajectory import TrajectoryPlanner, getwp, collision_free_min_snap

warnings.filterwarnings('ignore')


if __name__ == "__main__":

    Desired = namedtuple("Desired", ["x", "y", "z", "x_vel", "y_vel", "z_vel", "x_acc", "y_acc", "z_acc", "yaw"])
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config_file = "/home/medhyvinceslas/Documents/programming/quad3d_sim/config.ini"
    config.read(config_file)

    FREQ = 10  # inner loop speed relative to outer loop
    dt = 0.01
    velocity = 2.0

    waypoints = np.array([[10, 0, 0], [10, 4, 1], [6, 5, 1.5], [7, 8, 1.5], [2, 7, 2], [1, 0, 2]])
    coord_obstacles = np.array([[8, 6, 1.5, 5, 0], [4, 9, 1.5, 5, 0], [4, 1, 2, 5, 0], [3, 5, 1, 5, 0], [4, 3.5, 2.5, 5, 0], [5, 5, 10, .5, 5]])

    planner, waypoints, r_des, obstacle_edges = collision_free_min_snap(waypoints, coord_obstacles, velocity, dt)

    desired = Desired(
        r_des[:, 0], r_des[:, 1], r_des[:, 2],     # position
        r_des[:, 3], r_des[:, 4], r_des[:, 5],     # velocity
        r_des[:, 6], r_des[:, 7], r_des[:, 8],     # acc
        np.arctan2(r_des[:, 3], r_des[:, 4])*0       # yaw
    )      
    
    quad = Quadrotor(config, desired)
    controller = TFC(config)

    state_history, omega_history = quad.X, quad.omega    
    n_waypoints = desired.z.shape[0]

    for i in range(0, n_waypoints):
        # flight computer
        R = quad.R()
        F_cmd = controller.altitude(quad, desired, R, dt, index=i)
        bxy_cmd = controller.lateral(quad, F_cmd, desired, index=i)
        pqr_cmd = controller.reduced_attitude(quad, bxy_cmd, desired.yaw[i], R)
        
        for _ in range(FREQ):
            # flight controller
            moment_cmd = controller.body_rate_controller(quad, pqr_cmd)
            quad.set_propeller_speed(F_cmd, moment_cmd)
            quad.update_state(dt/FREQ)


        state_history = np.vstack((state_history, quad.X))
        omega_history = np.vstack((omega_history, quad.omega))



    sim = Sim3d(r_des, state_history, obstacle_edges)
    ani = sim.run_sim(frames=n_waypoints, interval=5)
    plt.show()

