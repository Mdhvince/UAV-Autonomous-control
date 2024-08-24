import numpy as np
import pytest

from uav_ac.planning.rrt import RRTStar


def test_path_cost():
    # Arrange
    path = np.array([[1, 1, 1], [3, 3, 9], [11, 5, 5], [1, 1, 1]])

    # Act
    result = RRTStar.path_cost(path)

    # Assert
    assert np.isclose(result, 29.1, atol=0.1)


def test__generate_random_node_in_limits(rrt_object):
    # Act
    result = rrt_object._generate_random_node()

    # Assert
    assert np.all(result >= rrt_object.space_limits_lw)
    assert np.all(result <= rrt_object.space_limits_up)


@pytest.mark.parametrize("node, expected", [
    ([1, 1, 1], [1, 1, 1]),
    ([3, 3, 9], [3, 3, 9]),
    ([11, 5, 5], [11, 5, 5]),
    ([1, 1, 1], [1, 1, 1])
])
def test__find_nearest_node(rrt_object, node, expected):
    # Arrange
    rrt_object.all_nodes = np.array([[1, 1, 1], [3, 3, 9], [11, 5, 5], [1, 1, 1]])

    # Act
    result = rrt_object._find_nearest_node(node)

    # Assert
    assert np.all(result == expected)


def test__adapt_random_node_position_if_too_far_from_nearest_node(rrt_object):
    # Arrange
    new_node = np.array([1, 1, 1])
    nearest_node = np.array([3, 3, 9])
    rrt_object.step_size = 2  # max distance
    expected = np.array([2.53, 2.53, 7.11])

    # Act
    result = rrt_object._adapt_random_node_position(new_node, nearest_node)

    # Assert
    assert np.all(result == expected)


def test__find_valid_neighbors_with_no_obstacles(rrt_object):
    # Arrange
    new_node = np.array([1, 1, 1])
    rrt_object.all_nodes = np.array([[1, 1, 1], [2, 2, 2], [11, 5, 5], [1, 1, 1]])
    rrt_object.neighborhood_radius = 3
    expected = np.array([[1, 1, 1], [2, 2, 2], [1, 1, 1]])

    # Act
    result = rrt_object._find_valid_neighbors(new_node)

    # Assert
    assert np.all(result == expected)


def test__find_best_neighbor(rrt_object):
    # Arrange
    rrt_object.start = np.array([0, 0, 0])
    neighbors = np.array([[18, 1, 3], [2, 5, 20], [13, 1, 1]])
    expected = np.array([13, 1, 1])

    # Act
    result = rrt_object._find_best_neighbor(neighbors)

    # Assert
    assert np.all(result == expected)


