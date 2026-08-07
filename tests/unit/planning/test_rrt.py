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


def test_simplify_path_should_remove_redundant_waypoints_when_direct_connection_is_clear(rrt_object):
    # Arrange
    rrt_object.obstacles = None
    path = np.array([[0., 0., 0.], [2., 0., 0.], [4., 0., 0.], [6., 0., 0.]])

    # Act
    result = rrt_object.simplify_path(path)

    # Assert
    assert result == pytest.approx(np.array([[0., 0., 0.], [6., 0., 0.]]))


def test_simplify_path_should_preserve_collision_free_detour_around_obstacle(rrt_object):
    # Arrange
    rrt_object.obstacles = np.array([[5., 7., -1., 1., -1., 1.]])
    path = np.array([[0., 0., 0.], [4., 2., 0.], [8., 2., 0.], [12., 0., 0.]])

    # Act
    result = rrt_object.simplify_path(path)

    # Assert
    assert len(result) > 2
    assert all(rrt_object._is_valid_connection(start, end) for start, end in zip(result[:-1], result[1:]))


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


def test__cost_to_come_of_start_is_zero(rrt_object):
    # Arrange
    rrt_object.start = np.array([0., 0., 0.])

    # Act
    result = rrt_object._cost_to_come(rrt_object.start)

    # Assert
    assert result == pytest.approx(0.0)


def test__cost_to_come_sums_edge_lengths_to_start(rrt_object):
    # Arrange
    rrt_object.start = np.array([0., 0., 0.])
    node_a = np.array([0., 0., 2.])
    node_b = np.array([0., 2., 2.])
    rrt_object.tree = {
        "[0.0, 0.0, 2.0]": rrt_object.start,
        "[0.0, 2.0, 2.0]": node_a,
    }

    # Act
    result = rrt_object._cost_to_come(node_b)

    # Assert
    assert result == pytest.approx(4.0)


def test__find_best_neighbor_prefers_lowest_cost_through_tree(rrt_object):
    # Arrange
    rrt_object.start = np.array([0., 0., 0.])
    near = np.array([0., 0., 2.])  # cost-to-come: 2.0
    detour = np.array([0., 3., 0.])
    close_but_costly = np.array([0., 0., 1.])  # cost-to-come through detour: 3 + sqrt(10)
    rrt_object.tree = {
        "[0.0, 0.0, 2.0]": rrt_object.start,
        "[0.0, 3.0, 0.0]": rrt_object.start,
        "[0.0, 0.0, 1.0]": detour,
    }
    neighbors = [close_but_costly, near]
    new_node = np.array([0., 0., 3.])

    # Act
    result = rrt_object._find_best_neighbor(neighbors, new_node)

    # Assert
    assert np.all(result == near)


def test__rewire_safely_rewires_every_improving_neighbor(rrt_object):
    # Arrange
    rrt_object.start = np.array([0., 0., 0.])
    detour = np.array([0., 0., 5.])
    neighbor_1 = np.array([1., 0., 0.])
    neighbor_2 = np.array([0., 1., 0.])
    new_node = np.array([0., 0., 1.])
    rrt_object.tree = {
        "[0.0, 0.0, 5.0]": rrt_object.start,
        "[1.0, 0.0, 0.0]": detour,
        "[0.0, 1.0, 0.0]": detour,
        "[0.0, 0.0, 1.0]": rrt_object.start,
    }

    # Act
    has_rewired = rrt_object._rewire_safely([neighbor_1, neighbor_2], new_node)

    # Assert
    assert has_rewired
    assert np.all(rrt_object.tree["[1.0, 0.0, 0.0]"] == new_node)
    assert np.all(rrt_object.tree["[0.0, 1.0, 0.0]"] == new_node)


def test__rewire_safely_never_rewires_the_start_node(rrt_object):
    # Arrange
    rrt_object.start = np.array([0., 0., 0.])
    new_node = np.array([0., 0., 1.])
    rrt_object.tree = {"[0.0, 0.0, 1.0]": rrt_object.start}

    # Act
    has_rewired = rrt_object._rewire_safely([rrt_object.start], new_node)

    # Assert
    assert not has_rewired
    assert "[0.0, 0.0, 0.0]" not in rrt_object.tree


def test__update_tree_keeps_cheaper_existing_parent(rrt_object):
    # Arrange
    rrt_object.start = np.array([0., 0., 0.])
    node = np.array([0., 0., 2.])
    worse_parent = np.array([0., 3., 0.])
    rrt_object.tree = {
        "[0.0, 0.0, 2.0]": rrt_object.start,
        "[0.0, 3.0, 0.0]": rrt_object.start,
    }

    # Act
    rrt_object._update_tree(worse_parent, node)

    # Assert
    assert np.all(rrt_object.tree["[0.0, 0.0, 2.0]"] == rrt_object.start)


def test__is_valid_connection_detects_thin_obstacle_between_samples(rrt_object):
    # Arrange
    rrt_object.obstacles = np.array([[4.999, 5.001, -10., 10., -10., 10.]])
    node1 = np.array([0., 0., 0.])
    node2 = np.array([10., 0., 0.])

    # Act
    result = rrt_object._is_valid_connection(node1, node2)

    # Assert
    assert not result


def test__is_valid_connection_accepts_segment_missing_obstacle(rrt_object):
    # Arrange
    rrt_object.obstacles = np.array([[4., 6., 1., 2., -10., 10.]])
    node1 = np.array([0., 0., 0.])
    node2 = np.array([10., 0., 0.])

    # Act
    result = rrt_object._is_valid_connection(node1, node2)

    # Assert
    assert result


