import warnings
import configparser
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

from control import utils_plot
from control.quadrotor import Quadrotor
from control.controller import Controller
from planning.trajectory import TrajectoryPlanner

warnings.filterwarnings('ignore')


if __name__ == "__main__":

    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config_file = "/home/medhyvinceslas/Documents/programming/quad3d_sim/config.ini"
    config.read(config_file)

    inner_loop_relative_to_outer_loop = 10
    dt = 0.02
    velocity = 5
    # waypoints = np.array([
    #     [10, 0, 0], [10, 4, 1], [6, 5, 1.5], [7, 8, 1.5], [2, 7, 2], [1, 0, 2]
    # ])
    waypoints = np.array([
        [0, 0, 0], [0, 0, 20]
    ])
    
    tp = TrajectoryPlanner(waypoints, velocity, dt)
    traj = tp.get_min_snap_trajectory()

    Desired = namedtuple("Desired", ["x", "y", "z", "x_vel", "y_vel", "z_vel", "x_acc", "y_acc", "z_acc", "yaw"])
    desired = Desired(
        traj[:, 0], traj[:, 1], traj[:, 2],     # position
        traj[:, 3], traj[:, 4], traj[:, 5],     # velocity
        traj[:, 6], traj[:, 7], traj[:, 8], 0)  # acc and yaw
    
    quad = Quadrotor(config, desired)
    control = Controller(config)

    state_history, omega_history = quad.X, quad.omega    
    n_waypoints = desired.z.shape[0]

    for i in range(0, n_waypoints):

        thrust_cmd = control.altitude(quad, desired, dt, index=i)
        acc_cmd = control.lateral(quad, desired, index=i)
        
        for _ in range(inner_loop_relative_to_outer_loop):
            moment_cmd = control.attitude(quad, thrust_cmd, acc_cmd, 0.0)
            quad.set_propeller_speed(thrust_cmd, moment_cmd)
            quad.update_state(dt/inner_loop_relative_to_outer_loop)

        state_history = np.vstack((state_history, quad.X))
        omega_history = np.vstack((omega_history, quad.omega))





    obstacle_boundary = (10, 10, 10)
    obstacle_coords = np.array([[8, 2, 0], [4, 6, 0], [7, 5, 0], [4, 0, 0]])
    obstacle_shapes = np.array([[1, 1, 5], [2, 1, 5], [1, 1, 5], [1, 4, 5]])
    fig, ax, norm, scalar_map = utils_plot.setup_plot(colormap="turbo")
    ani = utils_plot.run_animation(fig, n_waypoints, 5, ax, obstacle_coords, obstacle_shapes, obstacle_boundary, waypoints, state_history, desired, scalar_map, norm)
    plt.show()
    # utils_plot.save_animation(ani, "docs/figHelix.mp4")
    # utils_plot.plot_results(t, state_history, omega_history, desired)

