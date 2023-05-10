import warnings
import configparser
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

from control.simulation_3d import Sim3d
from control.quadrotor import Quadrotor
from control.controller import TFC
from planning.minimum_snap import MinimumSnap
from planning.trajectory import plot_3d_trajectory_and_obstacle

warnings.filterwarnings('ignore')
plt.style.use('ggplot')



def draw_controller_response(history, target_history, dim):
    """Draw the controller response given a history of 1d position"""
    plt.plot(history, label="history", color="blue", linewidth=2)
    plt.plot(target_history, "--", label="target", color="red", linewidth=4, alpha=0.5)
    plt.xlabel("Ts")
    plt.ylabel(f"{dim.upper()}")

    leg = plt.legend(loc='best', fancybox=True)
    for text in leg.get_texts():
        text.set_color("black")


def draw_all(state_history, desired, T, waypoints):
    fig = plt.figure(figsize=(16, 9))

    plt.subplot(2, 3, 1)
    draw_controller_response(state_history[:, 0], desired.x, "x (m)")

    plt.subplot(2, 3, 2)
    draw_controller_response(state_history[:, 1], desired.y, "y (m)")

    plt.subplot(2, 3, 3)
    draw_controller_response(state_history[:, 2], desired.z, "z (m)")

    plt.subplot(2, 3, 4)
    draw_controller_response(state_history[:, 5], desired.yaw, "yaw (rad)")

    plt.subplot(2, 3, 5)
    vel = np.linalg.norm(state_history[:, 6:9], axis=1)
    stacked = np.vstack((desired.x_vel, desired.y_vel, desired.z_vel)).T
    vel_desired = np.linalg.norm(stacked, axis=1)
    draw_controller_response(vel, vel_desired, "velocity (m/s)")

    ax = plt.subplot(2, 3, 6, projection='3d')
    ax.view_init(23, -40)
    plot_3d_trajectory_and_obstacle(ax, waypoints, T)

    plt.tight_layout()


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
    T.generate_collision_free_trajectory(coord_obstacles=None)
    r_des = T.full_trajectory

    desired = Desired(
        r_des[:, 0], r_des[:, 1], r_des[:, 2],  # desired position over time
        r_des[:, 3], r_des[:, 4], r_des[:, 5],  # desired velocity over time
        r_des[:, 6], r_des[:, 7], r_des[:, 8],  # desired acceleration over time
        np.arctan2(r_des[:, 3], r_des[:, 4])    # desired yaw over time such that the quadrotor faces the next waypoint
    )

    controller = TFC(config)
    quad = Quadrotor(config, desired)
    # initialize the quadrotor at the first desired position and yaw
    quad.X[0:3] = desired.x[0], desired.y[0], desired.z[0]
    quad.X[5] = desired.yaw[0]

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

    draw_all(state_history, desired, T, waypoints)
    # sim = Sim3d(r_des, state_history, T.obstacle_edges)
    # ani = sim.run_sim(frames=n_waypoints, interval=5)

    plt.show()
    # plt.savefig("docs/controller_response.png", dpi=300, bbox_inches='tight', facecolor="white")

