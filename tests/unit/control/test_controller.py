from types import SimpleNamespace

import numpy as np
import pytest

from uav_ac.control.controller import CascadedController
from uav_ac.quadrotor.quad import Quad


G = 9.81
DT = 0.01


@pytest.fixture
def controller():
    return CascadedController(g=G, dt=DT)


@pytest.fixture
def quad():
    return Quad(g=G, dt=DT)


@pytest.mark.parametrize("angle, expected", [
    (0.1, 0.1),
    (2 * np.pi + 0.1, 0.1),
    (-3 * np.pi / 2, np.pi / 2),
    (3 * np.pi / 2, -np.pi / 2),
])
def test_wrap_to_pi_maps_angle_into_range(angle, expected):
    # Arrange
    # Act
    result = CascadedController.wrap_to_pi(angle)

    # Assert
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("angle, expected", [
    (0.1, 0.1),
    (2 * np.pi + 0.1, 0.1),
    (-0.1, 2 * np.pi - 0.1),
])
def test_wrap_to_2pi_maps_angle_into_range(angle, expected):
    # Arrange
    # Act
    result = CascadedController.wrap_to_2pi(angle)

    # Assert
    assert result == pytest.approx(expected)


def test__pd_combines_terms_and_feedforward():
    # Arrange
    kp, kd = 2.0, 3.0
    error, error_dot, feedforward = 1.5, 0.5, 0.25

    # Act
    result = CascadedController._pd(kp, kd, error, error_dot, feedforward)

    # Assert
    assert result == pytest.approx(2.0 * 1.5 + 3.0 * 0.5 + 0.25)


def test__pid_combines_terms_and_feedforward():
    # Arrange
    kp, kd, ki = 2.0, 3.0, 0.5
    error, error_dot, i_error, feedforward = 1.5, 0.5, 4.0, 0.25

    # Act
    result = CascadedController._pid(kp, kd, ki, error, error_dot, i_error, feedforward)

    # Assert
    assert result == pytest.approx(2.0 * 1.5 + 3.0 * 0.5 + 0.5 * 4.0 + 0.25)


def test_altitude_at_setpoint_returns_hover_thrust(controller, quad):
    # Arrange
    des_z = np.array([quad.z, 0.0, 0.0])
    rot_mat = np.eye(3)

    # Act
    result = controller.altitude(quad, des_z, rot_mat, quad.kp_z, quad.kd_z, quad.ki_z)

    # Assert
    assert result == pytest.approx(quad.m * G)


def test_altitude_limits_commanded_descent_rate(quad):
    # Arrange
    controller_excessive = CascadedController(g=G, dt=DT)
    controller_at_limit = CascadedController(g=G, dt=DT)
    rot_mat = np.eye(3)
    # NED: descending means positive z_dot
    des_z_excessive = np.array([quad.z, 100.0, 0.0])
    des_z_at_limit = np.array([quad.z, quad.max_descent_rate, 0.0])

    # Act
    thrust_excessive = controller_excessive.altitude(quad, des_z_excessive, rot_mat, quad.kp_z, quad.kd_z, quad.ki_z)
    thrust_at_limit = controller_at_limit.altitude(quad, des_z_at_limit, rot_mat, quad.kp_z, quad.kd_z, quad.ki_z)

    # Assert
    assert thrust_excessive == pytest.approx(thrust_at_limit)


def test_altitude_limits_commanded_ascent_rate(quad):
    # Arrange
    controller_excessive = CascadedController(g=G, dt=DT)
    controller_at_limit = CascadedController(g=G, dt=DT)
    rot_mat = np.eye(3)
    # NED: climbing means negative z_dot
    des_z_excessive = np.array([quad.z, -100.0, 0.0])
    des_z_at_limit = np.array([quad.z, -quad.max_ascent_rate, 0.0])

    # Act
    thrust_excessive = controller_excessive.altitude(quad, des_z_excessive, rot_mat, quad.kp_z, quad.kd_z, quad.ki_z)
    thrust_at_limit = controller_at_limit.altitude(quad, des_z_at_limit, rot_mat, quad.kp_z, quad.kd_z, quad.ki_z)

    # Assert
    assert thrust_excessive == pytest.approx(thrust_at_limit)


def test_altitude_thrust_stays_within_rotor_bounds(controller, quad):
    # Arrange
    des_z = np.array([quad.z - 100.0, 0.0, 0.0])  # huge error far below current altitude
    rot_mat = np.eye(3)
    thrust_margin = 0.2 * (quad.max_thrust - quad.min_thrust)

    # Act
    result = controller.altitude(quad, des_z, rot_mat, quad.kp_z, quad.kd_z, quad.ki_z)

    # Assert
    assert result <= (quad.max_thrust - thrust_margin) * 4
    assert result >= (quad.min_thrust + thrust_margin) * 4


def test_altitude_integral_error_is_bounded(controller, quad):
    # Arrange
    des_z = np.array([quad.z + 5.0, 0.0, 0.0])  # persistent error
    rot_mat = np.eye(3)

    # Act
    for _ in range(10_000):
        controller.altitude(quad, des_z, rot_mat, quad.kp_z, quad.kd_z, quad.ki_z)

    # Assert
    assert abs(controller.integral_error) <= CascadedController.INTEGRAL_ERROR_LIMIT


def test_lateral_tilt_command_is_saturated(controller, quad):
    # Arrange
    des_x = np.array([100.0, 0.0, 0.0])
    des_y = np.array([-100.0, 0.0, 0.0])
    hover_thrust = quad.m * G

    # Act
    result = controller.lateral(quad, des_x, des_y, hover_thrust, quad.kp_xy, quad.kd_xy)

    # Assert
    assert np.all(np.abs(result) <= quad.max_tilt_angle)


def test_body_rate_controller_moment_is_proportional_to_rate_error(controller, quad):
    # Arrange
    pqr_cmd = np.array([1.0, 0.0, 0.0])

    # Act
    result = controller.body_rate_controller(quad, pqr_cmd, quad.kp_p, quad.kp_q, quad.kp_r)

    # Assert
    assert result == pytest.approx(np.array([quad.i_x * quad.kp_p, 0.0, 0.0]))


def test_body_rate_controller_saturates_moment_norm(controller, quad):
    # Arrange
    pqr_cmd = np.array([100.0, 100.0, 100.0])

    # Act
    result = controller.body_rate_controller(quad, pqr_cmd, quad.kp_p, quad.kp_q, quad.kp_r)

    # Assert
    assert np.linalg.norm(result) == pytest.approx(quad.max_torque)


def test_yaw_controller_takes_shortest_direction(controller):
    # Arrange
    quad_stub = SimpleNamespace(psi=0.1)
    psi_des = -0.1
    kp_yaw = 2.0

    # Act
    result = controller.yaw_controller(quad_stub, psi_des, kp_yaw)

    # Assert
    assert result == pytest.approx(kp_yaw * -0.2)
