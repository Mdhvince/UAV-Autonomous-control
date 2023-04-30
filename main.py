import warnings
import configparser
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

from control.simulation_3d import Sim3d
from control.quadrotor import Quadrotor
from control.controller import TFC
from planning.trajectory import MinimumSnap, getwp

warnings.filterwarnings('ignore')
plt.style.use('ggplot')


def draw_controller_response(history, target_history, dim):
    """Draw the controller response given a history of 1d position"""
    plt.plot(history, label="history", color="blue", linewidth=2)
    plt.plot(target_history, "--", label="target", color="red", linewidth=4, alpha=0.5)
    plt.xlabel("Ts")
    plt.ylabel(f"{dim.upper()} [m]")

    leg = plt.legend(loc='upper right', fancybox=True)
    for text in leg.get_texts():
        text.set_color("black")



if __name__ == "__main__":

    Desired = namedtuple("Desired", ["x", "y", "z", "x_vel", "y_vel", "z_vel", "x_acc", "y_acc", "z_acc", "yaw"])
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config_file = "/home/medhyvinceslas/Documents/programming/quad3d_sim/config.ini"
    config.read(config_file)

    FREQ = 10  # inner loop speed relative to outer loop
    dt = 0.01
    velocity = 2.0

    waypoints = np.array([[10., 0.0, 1.0],
                          [10., 4.0, 1.0],
                          [6.0, 5.0, 1.5],
                          [4.0, 7.0, 1.5],
                          [2.0, 7.0, 2.0],
                          [1.0, 0.0, 2.0]])

    coord_obstacles = np.array([[8.0, 6.0, 1.5, 5.0, 0.0],  # x, y, side_length, height, altitude_start
                                [4.0, 9.0, 1.5, 5.0, 0.0],
                                [4.0, 1.0, 2.0, 5.0, 0.0],
                                [3.0, 5.0, 1.0, 5.0, 0.0],
                                [4.0, 3.5, 2.5, 5.0, 0.0], 
                                [5.0, 5.0, 10., 0.5, 5.0]])

    T = MinimumSnap(waypoints, velocity=velocity, dt=dt)
    T.generate_collision_free_trajectory(coord_obstacles)
    r_des = T.full_trajectory


    desired = Desired(
        r_des[:, 0], r_des[:, 1], r_des[:, 2],     # position
        r_des[:, 3], r_des[:, 4], r_des[:, 5],     # velocity
        r_des[:, 6], r_des[:, 7], r_des[:, 8],     # acc
        np.arctan2(r_des[:, 3], r_des[:, 4])*0     # yaw
    )      
    
    controller = TFC(config)
    quad = Quadrotor(config, desired)
    quad.X = np.zeros(12)
    quad.X[0], quad.X[1] = desired.x[0], desired.y[0]

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

    plt.figure(figsize=(20, 10))

    plt.subplot(2, 2, 1)
    draw_controller_response(state_history[:, 0], desired.x, "x")
    plt.subplot(2, 2, 2)
    draw_controller_response(state_history[:, 1], desired.y, "y")
    plt.subplot(2, 2, 3)
    draw_controller_response(state_history[:, 2], desired.z, "z")

    plt.show()


    # sim = Sim3d(r_des, state_history, T.obstacle_edges)
    # ani = sim.run_sim(frames=n_waypoints, interval=5)
    # plt.show()

