import warnings

import numpy as np

import utils
from quadrotor.quad import Quad
from control.controller import CascadedController
from planning.minimum_snap import MinimumSnap
from planning.plot import RRTPlotter
from planning.rrt import RRTStar

warnings.filterwarnings('ignore')


def fly(state_history, omega_history, controller, quad, des_x, des_y, des_z, des_yaw, frequency):
    R = quad.R()
    F_cmd = controller.altitude(quad, des_z, R, quad.kp_z, quad.kd_z, quad.ki_z)
    bxy_cmd = controller.lateral(quad, des_x, des_y, F_cmd, quad.kp_xy, quad.kd_xy)
    pqr_cmd = controller.reduced_attitude(quad, bxy_cmd, des_yaw, R, quad.kp_roll, quad.kp_pitch, quad.kp_yaw)

    for _ in range(frequency):
        # flight controller
        moment_cmd = controller.body_rate_controller(quad, pqr_cmd, quad.kp_p, quad.kp_q, quad.kp_r)
        quad.set_propeller_speed(F_cmd, moment_cmd)
        quad.update_state()

    state_history = np.vstack((state_history, quad.X))
    omega_history = np.vstack((omega_history, quad.omega))

    return state_history, omega_history


def plot(rrt, optimal_trajectory, obstacles, state_history, draw_nodes=False, draw_obstacles=False):
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

    rrt_plotter.show()


if __name__ == "__main__":
    cfg, cfg_rrt, cfg_flight = utils.get_config()

    g = cfg.getfloat("g")
    dt = cfg.getfloat("dt")
    frequency = cfg.getint("frequency")

    # FLIGHT
    velocity = cfg_flight.getfloat("velocity")
    obstacles = utils.parse_array(cfg_flight, "coord_obstacles")
    min_distance_target = cfg_flight.getfloat("min_dist_target")
    goal_loc = utils.parse_array(cfg_flight, "goal_loc")
    start_loc = np.array([0., 0., 1.0])

    # RRT
    space_limits = utils.parse_array(cfg_rrt, "space_limits")
    max_distance = cfg_rrt.getfloat("max_distance")
    max_iterations = cfg_rrt.getint("max_iterations")

    ctrl = CascadedController(g, dt)
    quad = Quad(g, dt / frequency)
    quad.X[:3] = start_loc
    state_history, omega_history = quad.X, quad.omega

    rrt = RRTStar(space_limits, start_loc, goal_loc, max_distance, max_iterations, obstacles)
    rrt.run()
    global_path = rrt.best_path

    min_snap = MinimumSnap(global_path, obstacles, velocity, dt)
    global_trajectory = min_snap.get_trajectory()

    # the trajectory is time-parameterized at dt per row and each fly() call spans dt,
    # so following it in real time means consuming one row per control cycle
    for target in global_trajectory:
        des_x = target[[0, 3, 6]]
        des_y = target[[1, 4, 7]]
        des_z = target[[2, 5, 8]]
        des_yaw = target[9]

        state_history, omega_history = fly(
            state_history, omega_history, ctrl, quad, des_x, des_y, des_z, des_yaw, frequency
        )

    distance_to_goal = np.linalg.norm(quad.X[:3] - goal_loc)
    goal_has_been_reached = distance_to_goal < min_distance_target
    print(f"Flight finished {distance_to_goal:.2f} m away from the goal "
          f"({'reached' if goal_has_been_reached else 'missed'}).")

    plot(rrt, global_trajectory, obstacles, state_history[1:], draw_nodes=True, draw_obstacles=True)
