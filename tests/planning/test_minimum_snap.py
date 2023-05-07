import numpy as np
import pytest

from planning.minimum_snap import MinimumSnap


def test_insert_midpoints_at_indexes():
    # Arrange
    points = np.array([
        [0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]
    ])
    indexes = [1, 3]
    expected = np.array([
        [0, 0, 0], [.5, .5, .5], [1, 1, 1], [2, 2, 2], [2.5, 2.5, 2.5], [3, 3, 3]
    ])

    # Act
    result = MinimumSnap.insert_midpoints_at_indexes(points, indexes)

    # Assert
    assert result == pytest.approx(expected)


def test_insert_midpoints_at_indexes_with_empty_indexes():
    # Arrange
    points = np.array([
        [0, 0, 0], [1, 1, 1], [2, 2, 2]
    ])
    indexes = []
    expected = points

    # Act
    result = MinimumSnap.insert_midpoints_at_indexes(points, indexes)

    # Assert
    assert result == pytest.approx(expected)


def test_is_collision():
    # Arrange
    points = np.array([1, 1, 1])
    obstacles = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    expected = True

    # Act
    result = MinimumSnap.is_collision(points, obstacles)

    # Assert
    assert result == expected


def test_is_collision_is_false():
    # Arrange
    points = np.array([.5, .5, .5])
    obstacles = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    expected = False

    # Act
    result = MinimumSnap.is_collision(points, obstacles)

    # Assert
    assert result == expected


def test_is_collision_with_empty_obstacles():
    # Arrange
    points = np.array([.5, .5, .5])
    obstacles = np.array([])
    expected = False

    # Act
    result = MinimumSnap.is_collision(points, obstacles)

    # Assert
    assert result == expected

