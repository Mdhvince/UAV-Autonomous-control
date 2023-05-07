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


