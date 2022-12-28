import warnings
import configparser

import numpy as np
import matplotlib.pyplot as plt
import utils_plot


from quadrotor import Quadrotor
from controller import Controller
from trajectory import *

warnings.filterwarnings('ignore')



if __name__ == "__main__":

    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config_file = "/home/medhyvinceslas/Documents/programming/quad3d_sim/config.ini"
    config.read(config_file)
    inner_loop_relative_to_outer_loop = 10
    dt=0.024
    
    obstacle_boundary = (10, 10, 10)
    # x_center, y_center, z_center
    obstacle_coords = np.array([[8, 2, 0], [4, 6, 0], [7, 5, 0], [4, 0, 0]])
    # w, l, h
    obstacle_shapes = np.array([[1, 1, 5], [2, 1, 5], [1, 1, 5], [1, 4, 5]])
    # intermediate WPs
    waypoints = np.array([[10, 0, 0], [9, 4, 1], [6, 5, 1.5], [7, 8, 1.5], [2, 7, 2], [1, 0, 2]])

    
    desired = minimum_jerk_trajectory(waypoints, T=30, speed=1.2, dt=dt)



    quad = Quadrotor(config, desired)
    control = Controller(config)

    state_history, omega_history = quad.X, quad.omega    
    n_waypoints = desired.z.shape[0]

    for i in range(0, n_waypoints):
        # disable acceleration
        # desired.x_acc[i] = 0
        # desired.y_acc[i] = 0
        # desired.z_acc[i] = 0
        # desired.yaw[i] = 0

        thrust_cmd = control.altitude(quad, desired, dt, index=i)
        acc_cmd = control.lateral(quad, desired, index=i)
        
        for _ in range(inner_loop_relative_to_outer_loop):
            moment_cmd = control.attitude(quad, thrust_cmd, acc_cmd, desired.yaw[i])
            quad.set_propeller_speed(thrust_cmd, moment_cmd)
            quad.update_state(dt/inner_loop_relative_to_outer_loop)

        state_history = np.vstack((state_history, quad.X))
        omega_history = np.vstack((omega_history, quad.omega))



    fig, ax, norm, scalar_map = utils_plot.setup_plot(colormap="turbo")
    ani = utils_plot.run_animation(fig, n_waypoints, 5, ax, obstacle_coords, obstacle_shapes, obstacle_boundary, waypoints, state_history, desired, scalar_map, norm)
    plt.show()
    # utils_plot.save_animation(ani, "docs/figHelix.mp4")
    # utils_plot.plot_results(t, state_history, omega_history, desired)

