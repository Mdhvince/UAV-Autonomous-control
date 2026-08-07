import numpy as np
import pytest

from uav_ac.planning.minimum_snap import MinimumSnap


@pytest.mark.parametrize("indexes, expected", [
    ([1, 3], np.array([[0, 0, 0], [.5, .5, .5], [1, 1, 1], [2, 2, 2], [2.5, 2.5, 2.5], [3, 3, 3]])),
    ([], np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])),
])
def test_insert_midpoints_at_indexes(indexes, expected):
    # Arrange
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])

    # Act
    result = MinimumSnap.insert_midpoints_at_indexes(points, indexes)

    # Assert
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("order, expected", [
    (0, np.array([1., 0., 0., 0., 0., 0., 0., 0.])),
    (1, np.array([0., 1., 0., 0., 0., 0., 0., 0.])),
    (2, np.array([0., 0., 2., 0., 0., 0., 0., 0.])),
    (3, np.array([0., 0., 0., 6., 0., 0., 0., 0.])),
    (4, np.array([0., 0., 0., 0., 24., 0., 0., 0.])),
    (5, np.array([0., 0., 0., 0., 0., 120., 0., 0.])),
    (6, np.array([0., 0., 0., 0., 0., 0., 720., 0.])),
])
def test_polynom_at_t0(order, expected):
    # Arrange
    n_coeffs = 8
    t = 0

    # Act
    result = MinimumSnap.polynom(n_coeffs, order, t)

    # Assert
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("order, expected", [
    (0, np.array([1., 3., 9., 27., 81., 243., 729., 2187.])),
    (1, np.array([0., 1., 6., 27., 108., 405., 1458., 5103.])),
    (2, np.array([0., 0., 2., 18., 108., 540., 2430., 10206.])),
    (3, np.array([0., 0., 0., 6., 72., 540., 3240., 17010.])),
    (4, np.array([0., 0., 0., 0., 24., 360., 3240., 22680.])),
    (5, np.array([0., 0., 0., 0., 0., 120., 2160., 22680.])),
    (6, np.array([0., 0., 0., 0., 0., 0., 720., 15120.])),
])
def test_polynom_at_t3(order, expected):
    # Arrange
    n_coeffs = 8
    t = 3

    # Act
    result = MinimumSnap.polynom(n_coeffs, order, t)

    # Assert
    assert result == pytest.approx(expected)


def test_get_trajectory_passes_through_all_waypoints():
    # Arrange
    waypoints = np.array([[0., 0., 1.], [3., 0., 1.], [3., 3., 1.]])
    minimum_snap = MinimumSnap(waypoints, None, velocity=2.0, dt=0.01)

    # Act
    trajectory = minimum_snap.get_trajectory()

    # Assert
    positions = trajectory[:, :3]
    for waypoint in waypoints:
        distances = np.linalg.norm(positions - waypoint, axis=1)
        assert distances.min() == pytest.approx(0.0, abs=0.05)


def test_get_trajectory_returns_positions_velocities_accelerations_yaw_and_spline_id():
    # Arrange
    waypoints = np.array([[0., 0., 1.], [3., 0., 1.]])
    minimum_snap = MinimumSnap(waypoints, None, velocity=2.0, dt=0.01)

    # Act
    trajectory = minimum_snap.get_trajectory()

    # Assert
    assert trajectory.shape[1] == 11
    assert trajectory[:, 9] == pytest.approx(np.zeros(len(trajectory)))
    assert np.all(trajectory[:, 10] == 0.0)  # single spline


def test_get_trajectory_should_point_yaw_along_horizontal_velocity():
    # Arrange
    waypoints = np.array([[0., 0., -1.], [0., 3., -1.]])
    minimum_snap = MinimumSnap(waypoints, None, velocity=2.0, dt=0.01)

    # Act
    trajectory = minimum_snap.get_trajectory()

    # Assert
    assert trajectory[:, 9] == pytest.approx(np.pi / 2)


def test_calculate_yaws_should_hold_direction_while_horizontal_speed_is_negligible():
    # Arrange
    velocities = np.array([[0., 0., -1.], [0., 2., 0.], [0., 0., 1.]])

    # Act
    yaws = MinimumSnap._calculate_yaws(velocities)

    # Assert
    assert yaws == pytest.approx(np.full(3, np.pi / 2))


def test_calculate_yaws_should_stay_continuous_when_direction_crosses_pi():
    # Arrange
    velocities = np.array([[-1., 0.01, 0.], [-1., -0.01, 0.]])

    # Act
    yaws = MinimumSnap._calculate_yaws(velocities)

    # Assert
    assert abs(yaws[1] - yaws[0]) < 0.1
    assert yaws[1] > np.pi


def test_calculate_yaws_should_default_to_zero_without_horizontal_motion():
    # Arrange
    velocities = np.array([[0., 0., -1.], [0., 0., 0.], [0., 0., 1.]])

    # Act
    yaws = MinimumSnap._calculate_yaws(velocities)

    # Assert
    assert yaws == pytest.approx(np.zeros(3))


def test_get_trajectory_velocity_is_continuous_across_splines():
    # Arrange
    waypoints = np.array([[0., 0., 1.], [3., 0., 1.], [3., 3., 1.]])
    dt = 0.01
    minimum_snap = MinimumSnap(waypoints, None, velocity=2.0, dt=dt)

    # Act
    trajectory = minimum_snap.get_trajectory()

    # Assert
    velocities = trajectory[:, 3:6]
    velocity_jumps = np.linalg.norm(np.diff(velocities, axis=0), axis=1)
    assert velocity_jumps.max() < 0.5  # no discontinuity, even at the spline junction


def test_compute_spline_parameters_should_minimize_integrated_snap():
    # Arrange
    waypoints = np.array([[0., 0., -1.], [2., 1., -1.], [4., -1., -2.], [6., 0., -2.]])
    minimum_snap = MinimumSnap(waypoints, None, velocity=2.0, dt=0.01)

    # Act
    minimum_snap._compute_spline_parameters("lstsq")
    cost_matrix = minimum_snap._create_snap_cost_matrix()
    _, singular_values, right_vectors = np.linalg.svd(minimum_snap.A, full_matrices=True)
    rank = np.sum(singular_values > 1e-10)
    feasible_directions = right_vectors[rank:].T
    projected_gradient = feasible_directions.T @ cost_matrix @ minimum_snap.coeffs

    # Assert
    assert np.linalg.norm(projected_gradient) < 1e-6


def test_get_trajectory_corrects_splines_that_cut_through_an_obstacle():
    # Arrange
    waypoints = np.array([[0., 0., 1.], [3., 0., 1.], [3., 3., 1.]])
    obstacle = np.array([[3.2, 4.0, 0.5, 1.5, 0., 2.]])  # sits in the corner-cut overshoot of the raw spline
    raw_trajectory = MinimumSnap(waypoints, None, velocity=2.0, dt=0.01).get_trajectory()
    assert any(MinimumSnap.is_collision_cuboid(*point, obstacle[0]) for point in raw_trajectory[:, :3])
    minimum_snap = MinimumSnap(waypoints, obstacle, velocity=2.0, dt=0.01)

    # Act
    trajectory = minimum_snap.get_trajectory()

    # Assert
    assert not any(MinimumSnap.is_collision_cuboid(*point, obstacle[0]) for point in trajectory[:, :3])


@pytest.mark.parametrize(
    "x, y, z, cuboid_params,expected", [
        (2, 3, 4, np.array([1, 5, 2, 6, 3, 7]), True),
        (0, 0, 0, np.array([1, 5, 2, 6, 3, 7]), False),
        (1, 6, 3, np.array([1, 5, 2, 6, 3, 7]), True),
        (5, 2, 8, np.array([1, 5, 2, 6, 3, 7]), False)
    ]
)
def test_is_collisionCuboid(x, y, z, cuboid_params, expected):
    # Arrange
    # Act
    result = MinimumSnap.is_collision_cuboid(x, y, z, cuboid_params)

    # Assert
    assert result == expected


def test_generate_time_per_spline_slows_down_first_and_last_splines():
    # Arrange
    waypoints = np.array([[0., 0., -1.], [2., 0., -1.], [4., 0., -1.], [6., 0., -1.]])
    minimum_snap = MinimumSnap(waypoints, None, velocity=2.0, dt=0.01)
    nominal_time = 1.0  # 2 m between waypoints at 2 m/s

    # Act
    minimum_snap._setup()

    # Assert
    expected_boundary_time = nominal_time * MinimumSnap.START_END_TIME_FACTOR
    assert minimum_snap.times[0] == pytest.approx(expected_boundary_time)
    assert minimum_snap.times[1] == pytest.approx(nominal_time)
    assert minimum_snap.times[-1] == pytest.approx(expected_boundary_time)
