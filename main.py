import warnings
import logging
import configparser
from pathlib import Path

import numpy as np
from mayavi import mlab

from control.quadrotor import Quadrotor
from control.controller import CascadedController
from planning.minimum_snap import MinimumSnap
from planning.rrt import RRTStar

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

def plot_trajectory(rrt, optimal_trajectory, obstacles, state_history, animate=False, draw_nodes=False, draw_obstacles=False, delay=60):
    mlab.figure(size=(1920, 1080), bgcolor=(.3, .3, .3))

    # start and goal points (static)
    start = rrt.start
    goal = rrt.goal

    mlab.points3d(*start, color=(1, 0, 0), scale_factor=0.2, resolution=60)
    mlab.points3d(*goal, color=(0, 1, 0), scale_factor=0.2, resolution=60)

    if draw_obstacles:
        # obstacles
        obstacles = np.array(obstacles)[1:-2, :]  # ignore the floor and ceiling
        offset = 0.5  # need to offset the obstacle by 0.5 due to mayavi way of plotting cubes

        for obstacle in obstacles:
            xx, yy, zz = np.mgrid[obstacle[0] + offset:obstacle[1]:1,
                                  obstacle[2] + offset:obstacle[3]:1,
                                  obstacle[4] + offset:obstacle[5]:1]
            mlab.points3d(xx, yy, zz, color=(.6, .6, .6), scale_factor=1, mode='cube', opacity=0.2)

        # true obstacles (non-enhanced):
        # their width are 0.5 smaller and height 0.5 smaller than the ones in the config file
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

    if draw_nodes:
        for node, parent in rrt.best_tree.items():
            node = np.array(eval(node))
            # plot the nodes and connections between the nodes and their parents
            mlab.points3d(node[0], node[1], node[2], color=(0, 0, 1), scale_factor=.1, opacity=0.1)
            mlab.points3d(parent[0], parent[1], parent[2], color=(0, 0, 1), scale_factor=.1, opacity=0.1)
            mlab.plot3d([node[0], parent[0]], [node[1], parent[1]], [node[2], parent[2]],
                        color=(0, 0, 0),tube_radius=0.01, opacity=0.1)

    # Path found
    path = rrt.best_path
    mlab.plot3d(path[:, 0], path[:, 1], path[:, 2], color=(1, 1, 0), tube_radius=0.02)

    # Optimal trajectory
    mlab.plot3d(optimal_trajectory[:, 0], optimal_trajectory[:, 1], optimal_trajectory[:, 2], tube_radius=0.02, color=(1, 0, 1))

    if animate:
        # quad position (to animate)
        # quad_mesh = mlab.pipeline.open(
        #     '/home/medhyvinceslas/Documents/programming/quad3d_sim/quad_model/quadrotor_base.stl')
        # quad_mesh = mlab.pipeline.surface(quad_mesh, color=(0, 0, 0))

        quad_pos = mlab.points3d(
            state_history[0, 0], state_history[0, 1], state_history[0, 2], color=(1, 1, 0), scale_factor=0.2)

        @mlab.animate(delay=delay)
        def anim():
            for coord in state_history:
                x, y, z = coord[:3]
                quad_pos.mlab_source.set(x=x, y=y, z=z)

                # roll, pitch, yaw = coord[3:6]
                # quad_mesh.actor.actor.rotate_y(roll)
                # quad_mesh.actor.actor.rotate_x(pitch)
                # quad_mesh.actor.actor.rotate_z(yaw)
                # quad_mesh.actor.actor.position = (x, y, z)
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
    cfg_flight = config["SIM_FLIGHT"]
    cfg_rrt = config["RRT"]

    dt = cfg.getfloat("dt")
    frequency = cfg.getint("frequency")

    # FLIGHT
    velocity = cfg_flight.getfloat("velocity")
    obstacles = np.array(eval(cfg_flight.get("coord_obstacles")))
    goal_loc = np.array(eval(cfg_flight.get("goal_loc")))
    start_loc = np.array([0., 0., 1.0])

    # RRT
    space_limits = np.array(eval(cfg_rrt.get("space_limits")))
    max_distance = cfg_rrt.getfloat("max_distance")
    max_iterations = cfg_rrt.getint("max_iterations")

    ctrl = CascadedController(config)
    quad = Quadrotor(config)
    state_history, omega_history = quad.X, quad.omega

    total_timesteps = 0
    combined_desired_trajectory = np.empty((0, 11))
    min_distance_target = .4  # minimum distance to target to consider it reached


    ################## Starting here, things can change over time ##################

    rrt = RRTStar(space_limits, start_loc, goal_loc, max_distance, max_iterations, obstacles)
    rrt.run()
    path = rrt.best_path

    min_snap = MinimumSnap(path, obstacles, velocity, dt)
    desired_trajectory = min_snap.get_trajectory()

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


    plot_trajectory(
        rrt, combined_desired_trajectory, obstacles, state_history,
        animate=False, draw_nodes=False, draw_obstacles=True
    )
