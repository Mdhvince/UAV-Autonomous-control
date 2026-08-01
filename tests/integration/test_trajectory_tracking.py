import numpy as np
import pytest

from uav_ac.control.controller import CascadedController
from uav_ac.planning.minimum_snap import MinimumSnap
from uav_ac.quadrotor.quad import Quad


G = 9.81
DT = 0.01
FREQUENCY = 10


def fly_one_cycle(controller: CascadedController, quad: Quad, target: np.ndarray) -> None:
    """
    Run one outer control cycle (attitude/position) with FREQUENCY inner body-rate cycles.
    :param controller: the cascaded controller
    :param quad: the quadrotor being flown
    :param target: one trajectory row [x, y, z, vx, vy, vz, ax, ay, az, yaw, spline_id]
    """
    des_x, des_y, des_z, des_yaw = target[[0, 3, 6]], target[[1, 4, 7]], target[[2, 5, 8]], target[9]
    R = quad.R()
    thrust_cmd = controller.altitude(quad, des_z, R, quad.kp_z, quad.kd_z, quad.ki_z)
    bxy_cmd = controller.lateral(quad, des_x, des_y, thrust_cmd, quad.kp_xy, quad.kd_xy)
    pqr_cmd = controller.reduced_attitude(quad, bxy_cmd, des_yaw, R, quad.kp_roll, quad.kp_pitch, quad.kp_yaw)

    for _ in range(FREQUENCY):
        moment_cmd = controller.body_rate_controller(quad, pqr_cmd, quad.kp_p, quad.kp_q, quad.kp_r)
        quad.set_propeller_speed(thrust_cmd, moment_cmd)
        quad.update_state()


def test_quad_follows_minimum_snap_trajectory_in_real_time():
    # Arrange
    waypoints = np.array([[0., 0., 1.], [2., 1., 1.5], [4., 0., 2.], [5., 2., 2.]])
    trajectory = MinimumSnap(waypoints, None, velocity=2.0, dt=DT).get_trajectory()
    controller = CascadedController(g=G, dt=DT)
    quad = Quad(g=G, dt=DT / FREQUENCY)
    quad.X[:3] = waypoints[0]

    # Act
    tracking_errors = []
    for target in trajectory:
        fly_one_cycle(controller, quad, target)
        tracking_errors.append(np.linalg.norm(quad.X[:3] - target[:3]))

    # Assert
    final_distance_to_goal = np.linalg.norm(quad.X[:3] - waypoints[-1])
    assert final_distance_to_goal < 0.5
    assert np.mean(tracking_errors) < 0.5
