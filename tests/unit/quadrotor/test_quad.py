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
    commanded_forces = quad.kf * quad.omega_command ** 2

    # Assert
    assert np.sum(commanded_forces) == pytest.approx(thrust_cmd)


def test_set_propeller_speed_produces_commanded_roll_moment(quad):
    # Arrange
    thrust_cmd = 4.0
    moment_cmd = np.array([0.2, 0.0, 0.0])

    # Act
    quad.set_propeller_speed(thrust_cmd, moment_cmd)
    forces = quad.kf * quad.omega_command ** 2
    commanded_roll_moment = quad.l * (forces[0] + forces[3] - forces[1] - forces[2])
    commanded_pitch_moment = quad.l * (forces[0] + forces[1] - forces[2] - forces[3])

    # Assert
    assert commanded_roll_moment == pytest.approx(0.2)
    assert commanded_pitch_moment == pytest.approx(0.0, abs=1e-12)


def test_set_propeller_speed_produces_commanded_yaw_moment(quad):
    # Arrange
    thrust_cmd = 4.0
    moment_cmd = np.array([0.0, 0.0, 0.01])

    # Act
    quad.set_propeller_speed(thrust_cmd, moment_cmd)
    forces = quad.kf * quad.omega_command ** 2
    commanded_yaw_moment = quad.km * (-forces[0] + forces[1] - forces[2] + forces[3])

    # Assert
    assert commanded_yaw_moment == pytest.approx(0.01)


def test_set_propeller_speed_scales_moments_to_keep_all_rotors_within_bounds(quad):
    # Arrange
    thrust_cmd = 4.0
    moment_cmd = np.array([0.0, 0.0, 0.5])  # yaw demand far beyond rotor authority

    # Act
    quad.set_propeller_speed(thrust_cmd, moment_cmd)

    # Assert
    commanded_forces = quad.kf * quad.omega_command ** 2
    assert np.all(commanded_forces >= quad.min_thrust)
    assert np.all(commanded_forces <= quad.max_thrust)
    assert np.sum(commanded_forces) == pytest.approx(thrust_cmd)


def test_set_propeller_speed_clips_collective_thrust_to_rotor_limits(quad):
    # Arrange
    thrust_cmd = 100.0
    moment_cmd = np.zeros(3)

    # Act
    quad.set_propeller_speed(thrust_cmd, moment_cmd)

    # Assert
    commanded_forces = quad.kf * quad.omega_command ** 2
    assert commanded_forces == pytest.approx(np.full(4, quad.max_thrust))


def test_set_propeller_speed_applies_first_order_motor_rise_dynamics(quad):
    # Arrange
    thrust_cmd = 4.0
    expected_target = np.sqrt(thrust_cmd / 4 / quad.kf)
    expected_alpha = 1 - np.exp(-quad.dt / quad.motor_rise_time_constant)

    # Act
    quad.set_propeller_speed(thrust_cmd, np.zeros(3))

    # Assert
    assert quad.omega_command == pytest.approx(np.full(4, expected_target))
    assert quad.omega == pytest.approx(np.full(4, expected_alpha * expected_target))


def test_set_propeller_speed_uses_slower_motor_dynamics_when_speed_decreases(quad):
    # Arrange
    initial_speed = np.sqrt(quad.max_thrust / quad.kf)
    target_speed = np.sqrt(quad.min_thrust / quad.kf)
    quad.omega = np.full(4, initial_speed)
    expected_alpha = 1 - np.exp(-quad.dt / quad.motor_fall_time_constant)
    expected_speed = initial_speed + expected_alpha * (target_speed - initial_speed)

    # Act
    quad.set_propeller_speed(0.0, np.zeros(3))

    # Assert
    assert quad.omega == pytest.approx(np.full(4, expected_speed))


def test_controller_gains_are_derived_from_response_parameters(quad):
    # Arrange
    expected_kp_xy = 1 / quad.tau_xy ** 2
    expected_kd_xy = 2 * quad.zeta_xy / quad.tau_xy
    expected_kp_p = 1 / quad.tau_p

    # Act
    gains = np.array([quad.kp_xy, quad.kd_xy, quad.kp_p])

    # Assert
    assert gains == pytest.approx(np.array([expected_kp_xy, expected_kd_xy, expected_kp_p]))


def test_update_state_hover_thrust_keeps_altitude(quad):
    # Arrange
    quad.X[2] = 1.0
    hover_speed = np.sqrt(quad.m * G / 4 / quad.kf)
    quad.omega = np.full(4, hover_speed)

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


def test_update_state_free_fall_drifts_toward_positive_z(quad):
    # Arrange
    quad.X[:3] = np.array([0.0, 0.0, -5.0])  # NED: 5 m above ground
    quad.omega = np.zeros(4)

    # Act
    for _ in range(1000):
        quad.update_state()

    # Assert
    assert quad.z > -5.0  # NED: falling means z increases (toward the ground)


def test_update_state_positive_roll_at_hover_accelerates_toward_positive_y(quad):
    # Arrange
    half_roll = np.radians(10) / 2
    quad.X[3:7] = np.array([np.cos(half_roll), np.sin(half_roll), 0.0, 0.0])
    hover_force_per_rotor = quad.m * G / 4
    quad.omega = np.sqrt(hover_force_per_rotor / quad.kf) * np.ones(4)

    # Act
    for _ in range(1000):
        quad.update_state()

    # Assert
    assert quad.y_vel > 0.0  # NED: roll right pushes the drone to the right (+y)


def test_update_state_positive_pitch_at_hover_accelerates_toward_negative_x(quad):
    # Arrange
    half_pitch = np.radians(10) / 2
    quad.X[3:7] = np.array([np.cos(half_pitch), 0.0, np.sin(half_pitch), 0.0])
    hover_force_per_rotor = quad.m * G / 4
    quad.omega = np.sqrt(hover_force_per_rotor / quad.kf) * np.ones(4)

    # Act
    for _ in range(1000):
        quad.update_state()

    # Assert
    assert quad.x_vel < 0.0  # NED: pitch up tilts the thrust backward (-x)
