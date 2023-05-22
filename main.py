import warnings
import logging
import configparser
from pathlib import Path

import numpy as np
from mayavi import mlab

from control.quadrotor import Quadrotor
from control.controller import CascadedController
from planning.minimum_snap import MinimumSnap

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

def plot_trajectory(config, rrt, optim, state_history, animate=False, delay=60):
    f = mlab.figure(size=(1920, 1080), bgcolor=(.9, .9, .9))
    # start and goal points (static)
    takeoff_height = eval(config["SIM_TAKEOFF"].get("height"))
    start = np.array([0., 0., takeoff_height])
    goal = np.array(eval(config["SIM_FLIGHT"].get("goal_loc")))
    mlab.points3d(start[0], start[1], start[2], color=(1, 0, 0), scale_factor=0.2, resolution=60)
    mlab.points3d(goal[0], goal[1], goal[2], color=(0, 1, 0), scale_factor=0.2, resolution=60)

    # obstacles
    obstacles = np.array(eval(config["SIM_FLIGHT"].get("coord_obstacles")))[1:, :]  # ignore the floor
    offset = 0.5  # need to offset the obstacle by 0.5 due to mayavi way of plotting cubes

    for obstacle in obstacles:
        xx, yy, zz = np.mgrid[
                     obstacle[0] + offset:obstacle[1]:1,
                     obstacle[2] + offset:obstacle[3]:1,
                     obstacle[4] + offset:obstacle[5]:1
                     ]
        mlab.points3d(xx, yy, zz, color=(.6, .6, .6), scale_factor=1, mode='cube', opacity=0.2)

    # true obstacles (non-enhanced): their width are 0.5 smaller and height 0.5 smaller than the ones in the config file
    rm = 0.25
    offset = offset + rm
    matrix_factor = np.ones(obstacles.shape) * rm
    matrix_factor[:, [1, 3, 5]] = matrix_factor[:, [1, 3, 5]] * -1
    obstacles_true = obstacles + matrix_factor

    # ensure z_min is 0 if z_min=0.25
    obstacles_true[:, 4] = np.where(obstacles_true[:, 4] == 0.25, 0, obstacles_true[:, 4])

    for obstacle in obstacles_true:
        xx, yy, zz = (
            np.mgrid[obstacle[0]+offset:obstacle[1]:1,
                     obstacle[2]+offset:obstacle[3]:1,
                     obstacle[4]+offset-rm:obstacle[5]:1]
        )
        mlab.points3d(xx, yy, zz, color=(1, 0, 0), scale_factor=1, mode='cube', opacity=1)

    # Nodes in the tree
    for node in rrt.all_nodes:
        mlab.points3d(node[0], node[1], node[2], color=(0, 0, 1), scale_factor=.1, resolution=60)

    # Edges in the tree
    for connected_node in rrt.connected_nodes:
        node1, node2 = connected_node
        mlab.plot3d(
            [node1[0], node2[0]], [node1[1], node2[1]], [node1[2], node2[2]], color=(0, 0, 0), tube_radius=0.01)

    # Path found
    path = rrt.get_path()
    mlab.plot3d(path[:, 0], path[:, 1], path[:, 2], color=(0, 1, 1), tube_radius=0.02)

    # Optimal trajectory
    mlab.plot3d(optim[:, 0], optim[:, 1], optim[:, 2], tube_radius=0.02, color=(1, 0, 1))

    if animate:
        # quad position (to animate)
        quad_pos = mlab.points3d(
            state_history[0, 0], state_history[0, 1], state_history[0, 2], color=(0, 1, 1), scale_factor=0.2)

        @mlab.animate(delay=delay)
        def anim():
            for coord in state_history:
                x, y, z = coord[:3]
                quad_pos.mlab_source.set(x=x, y=y, z=z)
                yield

        anim()
    mlab.show()



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

    rrt = None
    min_distance_target = .5  # minimum distance to target to consider it reached
    optim = None

    for mode in modes:
        logging.info(f"Starting {mode} mode...")
        T = MinimumSnap(config, mode)
        desired_trajectory = T.get_trajectory()

        if mode == "flight":
            rrt = T.rrt
            optim = desired_trajectory

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

            target_has_been_reached = np.linalg.norm(quad.X[:3] - desired_trajectory[0, :3]) < min_distance_target

            if target_has_been_reached:
                desired_trajectory = np.delete(desired_trajectory, 0, axis=0)  # remove current waypoint from desired
                logging.info(f"Waypoint {round(des_x[0], 1), round(des_y[0], 1), round(des_z[0], 1)} visited.")

            # if all waypoints have been visited
            if desired_trajectory.shape[0] == 0:
                break

        logging.info(f"{mode} completed.: Quadrotor at XYZ: {np.round(quad.X[:3], 2)}")


    plot_trajectory(config, rrt, optim, state_history, animate=True, delay=30)
