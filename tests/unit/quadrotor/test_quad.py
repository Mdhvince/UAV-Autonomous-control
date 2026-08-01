import numpy as np
import pytest

from uav_ac.quadrotor.quad import Quad


G = 9.81
DT = 0.001


@pytest.fixture
def quad():
    return Quad(g=G, dt=DT)


def test_quat_to_rot_identity_quaternion_returns_identity():
    # Arrange
    q = np.array([1.0, 0.0, 0.0, 0.0])

    # Act
    result = Quad.quat_to_rot(q)

    # Assert
    assert result == pytest.approx(np.eye(3))


def test_quat_to_rot_quarter_turn_about_z_maps_x_to_y():
    # Arrange
    half_angle = np.pi / 4
    q = np.array([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)])

    # Act
    result = Quad.quat_to_rot(q) @ np.array([1.0, 0.0, 0.0])

    # Assert
    assert result == pytest.approx(np.array([0.0, 1.0, 0.0]), abs=1e-12)


def test_quat_to_rot_returns_orthonormal_matrix():
    # Arrange
    q = np.array([0.4, -0.3, 0.5, 0.2])

    # Act
    result = Quad.quat_to_rot(q)

    # Assert
    assert result.T @ result == pytest.approx(np.eye(3))
    assert np.linalg.det(result) == pytest.approx(1.0)


def test_euler_angles_extracts_pure_roll(quad):
    # Arrange
    roll = 0.3
    quad.X[3:7] = np.array([np.cos(roll / 2), np.sin(roll / 2), 0.0, 0.0])

    # Act
    result = quad.euler_angles

    # Assert
    assert result == pytest.approx(np.array([roll, 0.0, 0.0]))


def test_euler_angles_extracts_pure_yaw(quad):
    # Arrange
    yaw = 1.2
    quad.X[3:7] = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])

    # Act
    result = quad.euler_angles

    # Assert
    assert result == pytest.approx(np.array([0.0, 0.0, yaw]))


def test_set_propeller_speed_zero_moments_produces_commanded_total_thrust(quad):
    # Arrange
    thrust_cmd = 2.0
    moment_cmd = np.zeros(3)

    # Act
    quad.set_propeller_speed(thrust_cmd, moment_cmd)

    # Assert
    assert quad.f_total == pytest.approx(thrust_cmd)


def test_set_propeller_speed_produces_commanded_roll_moment(quad):
    # Arrange
    thrust_cmd = 4.0
    moment_cmd = np.array([0.2, 0.0, 0.0])

    # Act
    quad.set_propeller_speed(thrust_cmd, moment_cmd)

    # Assert
    assert quad.tau_x == pytest.approx(0.2)
    assert quad.tau_y == pytest.approx(0.0, abs=1e-12)


def test_set_propeller_speed_produces_commanded_yaw_moment(quad):
    # Arrange
    thrust_cmd = 4.0
    moment_cmd = np.array([0.0, 0.0, 0.01])

    # Act
    quad.set_propeller_speed(thrust_cmd, moment_cmd)

    # Assert
    assert quad.tau_z == pytest.approx(0.01)


def test_set_propeller_speed_clips_unreachable_negative_rotor_forces(quad):
    # Arrange
    thrust_cmd = 0.4
    moment_cmd = np.array([0.0, 0.0, 0.5])  # yaw demand far beyond rotor authority

    # Act
    quad.set_propeller_speed(thrust_cmd, moment_cmd)

    # Assert
    forces = np.array([quad.f_1, quad.f_2, quad.f_3, quad.f_4])
    assert np.all(forces >= 0.0)
    assert not np.any(np.isnan(quad.omega))


def test_update_state_hover_thrust_keeps_altitude(quad):
    # Arrange
    quad.X[2] = 1.0
    quad.set_propeller_speed(quad.m * G, np.zeros(3))

    # Act
    for _ in range(1000):
        quad.update_state()

    # Assert
    assert quad.z == pytest.approx(1.0, abs=1e-6)
    assert quad.velocity == pytest.approx(np.zeros(3), abs=1e-6)


def test_update_state_keeps_quaternion_normalized(quad):
    # Arrange
    quad.X[10] = 2.0  # roll rate [rad/s]

    # Act
    for _ in range(100):
        quad.update_state()

    # Assert
    assert np.linalg.norm(quad.quaternion) == pytest.approx(1.0)
