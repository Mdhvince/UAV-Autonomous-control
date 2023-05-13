import logging
import warnings
import configparser
from pathlib import Path
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

from control.quadrotor import Quadrotor
from control.controller import TFC
from planning.minimum_snap import MinimumSnap
from simulation_3d import Sim3d

warnings.filterwarnings('ignore')
plt.style.use('ggplot')



if __name__ == "__main__":
    logdir = Path("logs")
    logdir.mkdir(exist_ok=True)

    logging.basicConfig(
        filename=logdir / "sim.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", filemode="w")

    Desired = namedtuple("Desired", ["x", "y", "z", "x_vel", "y_vel", "z_vel", "x_acc", "y_acc", "z_acc", "yaw"])
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config_file = Path("/home/medhyvinceslas/Documents/programming/quad3d_sim/config.ini")
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

    coord_obstacles = np.array([
        [8.0, 6.0, 1.5, 5.0, 0.0],  # x, y, side_length, height, altitude_start
        [4.0, 9.0, 1.5, 5.0, 0.0],
        [4.0, 1.0, 2.0, 5.0, 0.0],
        [3.0, 5.0, 1.0, 5.0, 0.0],
        [4.0, 3.5, 2.5, 5.0, 0.0],
        # [5.0, 5.0, 10., 0.5, 5.0]
    ])

    T = MinimumSnap(waypoints, velocity=velocity, dt=dt)
    T.generate_collision_free_trajectory(coord_obstacles=coord_obstacles)
    r_des = T.full_trajectory

    desired = Desired(
        r_des[:, 0], r_des[:, 1], r_des[:, 2],  # desired position over time
        r_des[:, 3], r_des[:, 4], r_des[:, 5],  # desired velocity over time
        r_des[:, 6], r_des[:, 7], r_des[:, 8],  # desired acceleration over time

        # create desired yaw that always in the direction of the next waypoint
        np.arctan2(r_des[:, 1], r_des[:, 0]) + np.pi/2  # added pi/2 to make it face the right direction
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

        # state_history = np.vstack((state_history, quad.X))
        # omega_history = np.vstack((omega_history, quad.omega))

        logging.info(
            f" | Position: {quad.X[0]:.2f}, {quad.X[1]:.2f}, {quad.X[2]:.2f}"
            f" | RPY: {quad.X[3]:.2f}, {quad.X[4]:.2f}, {quad.X[5]:.2f}"
            f" | Velocity: {quad.X[6]:.2f}, {quad.X[7]:.2f}, {quad.X[8]:.2f}"
        )

    # sim = Sim3d(r_des, state_history, T.obstacle_edges)
    # ani = sim.run_sim(frames=n_waypoints, interval=5)
    # plt.show()
    # plt.savefig("docs/controller_response.png", dpi=300, bbox_inches='tight', facecolor="white")

