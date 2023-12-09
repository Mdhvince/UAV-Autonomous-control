import numpy as np
import pytest

from planning.minimum_snap import MinimumSnap
from tests.fixtures.fixtures import config  # noqa: F401


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


def test_generate_time_per_spline(config):
    # Arrange
    T = MinimumSnap(config)

    T.nb_splines = T.waypoints.shape[0] - 1

    distances = np.array([1.0, 3.0, 2.0])  # computed manually from waypoints fixture
    expected = distances / T.velocity

    # Act
    T._generate_time_per_spline()

    # Assert
    assert T.times == pytest.approx(expected)


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
