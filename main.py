import time
import warnings

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

def plot(rrt, optimal_trajectory, obstacles, state_history, animate=False, draw_nodes=False, draw_obstacles=False, delay=60):
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

    mlab.orientation_axes()
    mlab.show()


def receding_horizon(
        lt_path, current_position, horizon, max_distance, max_iterations, obstacles, velocity, dt):
    """
    :param lt_path: long-term path
    :param current_position: current position of the quadrotor
    :param horizon: length of the horizon in meters
    :param max_distance: maximum distance of a newly sampled node to the nearest node in the tree
    :param max_iterations: maximum number of iterations to run RRT
    :param obstacles: obstacles in the environment
    :param velocity: velocity of the quadrotor
    :param dt: time step
    :return: new trajectory
    """

    is_last = False
    lt_path = lt_path[:, :3]
    start_index = np.argmin(np.linalg.norm(lt_path - current_position, axis=1))
    lt_path = lt_path[start_index:, :]  # remove the waypoints that have already been reached

    distances = np.linalg.norm(lt_path - current_position, axis=1)

    # Index of the waypoint that is horizon meters far from the current position (last waypoint if not found)
    try:
        index = np.where(distances >= horizon)[0][0]
    except IndexError:
        index = -1
        is_last = True

    goal = lt_path[index, :]

    freedom = .1  # the quad can move freedom cm in any direction to avoid unexpected obstacles (env changes)
    space_limits = np.array([
        [current_position[0] - freedom, current_position[1] - freedom, current_position[2] - freedom],
        [goal[0] + freedom, goal[1] + freedom, goal[2] + freedom]
    ])

    print("Current position: ", np.round(current_position, 2))
    print("Goal: ", np.round(goal, 2))

    rrt = RRTStar(space_limits, current_position, goal, max_distance, max_iterations, obstacles)
    rrt.dynamic_break_at = max_iterations
    rrt.run()
    st_path = rrt.best_path  # short-term path
    min_snap = MinimumSnap(st_path, obstacles, velocity, dt)
    return min_snap.get_trajectory(), is_last





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

    ################## Starting here, things can change over time ##################

    rrt = RRTStar(space_limits, start_loc, goal_loc, max_distance, max_iterations, obstacles)
    rrt.run()
    global_path = rrt.best_path  # long-term path

    min_snap = MinimumSnap(global_path, obstacles, velocity, dt)
    global_trajectory = min_snap.get_trajectory()  # long-term trajectory
    global_trajectory_plot = np.copy(global_trajectory)

    start_time = time.time()

    while True:
        des_x = global_trajectory[0, [0, 3, 6]]
        des_y = global_trajectory[0, [1, 4, 7]]
        des_z = global_trajectory[0, [2, 5, 8]]
        des_yaw = global_trajectory[0, 9]


        state_history, omega_history = fly(
            state_history, omega_history, ctrl, quad, des_x, des_y, des_z, des_yaw, frequency
        )

        target_has_been_reached = np.linalg.norm(quad.X[:3] - global_trajectory[0, :3]) < min_distance_target

        if target_has_been_reached:
            global_trajectory = np.delete(global_trajectory, 0, axis=0)  # remove current waypoint from desired
            start_time = time.time()


        too_long = time.time() - start_time > 5
        if too_long:
            print("Took too long to reach target. Program will terminate.")

        # if all waypoints have been visited or issue reaching target, then break
        if global_trajectory.shape[0] == 0 or too_long:
            break


    plot(rrt, global_trajectory_plot, obstacles, state_history, animate=False, draw_nodes=False, draw_obstacles=True)
