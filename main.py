import time
import warnings
import configparser
from pathlib import Path

import numpy as np
from mayavi import mlab

import utils
from control.quadrotor import Quadrotor
from control.controller import CascadedController
from planning.minimum_snap import MinimumSnap
from planning.plot import RRTPlotter
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
    rrt_plotter = RRTPlotter(rrt, optimal_trajectory, state_history)

    rrt_plotter.plot_start_and_goal()
    rrt_plotter.plot_path()
    rrt_plotter.plot_trajectory()
    rrt_plotter.plot_executed_trajectory()

    if draw_obstacles:
        obstacles = np.array(obstacles)[1:-2, :]  # ignore the floor and ceiling
        rrt_plotter.plot_obstacles(obstacles)

    if draw_nodes:
        rrt_plotter.plot_tree()

    if animate:
        quad_pos = mlab.points3d(
            state_history[0, 0], state_history[0, 1], state_history[0, 2], color=(1, 1, 0), scale_factor=0.2)
        rrt_plotter.animate_point(quad_pos, delay=delay)

    mlab.show()



if __name__ == "__main__":
    cfg, cfg_rrt, cfg_flight, cfg_vehicle, cfg_controller = utils.get_config()

    dt = cfg.getfloat("dt")
    frequency = cfg.getint("frequency")

    # FLIGHT
    velocity = cfg_flight.getfloat("velocity")
    obstacles = np.array(eval(cfg_flight.get("coord_obstacles")))
    min_distance_target = cfg_flight.getfloat("min_dist_target")
    goal_loc = np.array(eval(cfg_flight.get("goal_loc")))
    start_loc = np.array([0., 0., 1.0])

    # RRT
    space_limits = np.array(eval(cfg_rrt.get("space_limits")))
    max_distance = cfg_rrt.getfloat("max_distance")
    max_iterations = cfg_rrt.getint("max_iterations")

    ctrl = CascadedController()
    quad = Quadrotor()
    quad.X[:3] = start_loc
    state_history, omega_history = quad.X, quad.omega

    total_timesteps = 0
    combined_desired_trajectory = np.empty((0, 11))


    ################## Starting here, things can change over time ##################

    rrt = RRTStar(space_limits, start_loc, goal_loc, max_distance, max_iterations, obstacles)
    rrt.run()
    path = rrt.best_path

    min_snap = MinimumSnap(path, obstacles, velocity, dt)
    desired_trajectory = min_snap.get_trajectory()

    total_timesteps += desired_trajectory.shape[0]
    combined_desired_trajectory = np.vstack((combined_desired_trajectory, desired_trajectory))

    start_time = time.time()

    print("Starting flight...\n")
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
            start_time = time.time()


        too_long = time.time() - start_time > 5
        if too_long:
            print("Took too long to reach target. Program will terminate.")

        # if all waypoints have been visited or issue reaching target, then break
        if desired_trajectory.shape[0] == 0 or too_long:
            break


    plot_trajectory(
        rrt, combined_desired_trajectory, obstacles, state_history,
        animate=True, draw_nodes=False, draw_obstacles=True
    )


    #  if target has been reached and no change in the current obstacle configuration has been detected, then we can
    #  continue with the same trajectory. Otherwise, we need to recompute from the current state, the Path (RRT*) and
    #  the trajectory (Minimum Snap). But this should be done in a separate thread, so that the quadrotor can continue
    #  hovering while the new trajectory is being computed.
    #
    #  Other thoughts:
    #  1 - Compute a global trajectory (RRT* + Minimum Snap) from start to goal
    #  While the goal has not been reached:
    #  2 - Compute a local trajectory (RRT* + Minimum Snap) from current position to some point on the global trajectory
