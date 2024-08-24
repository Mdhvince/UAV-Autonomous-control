import numpy as np

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