import pytest
import numpy as np


@pytest.fixture
def waypoints():
    waypoints = np.array([[0.0, 0.0, 1.0],
                          [1.0, 0.0, 1.0],
                          [4.0, 0.0, 1.0],
                          [6.0, 0.0, 1.0]])
    return waypoints


@pytest.fixture
def coord_obstacles():
    coord_obstacles = np.array([[8.0, 6.0, 1.5, 5.0, 0.0],
                                [4.0, 9.0, 1.5, 5.0, 0.0],
                                [4.0, 1.0, 2.0, 5.0, 0.0],
                                [3.0, 5.0, 1.0, 5.0, 0.0],
                                [4.0, 3.5, 2.5, 5.0, 0.0],
                                [5.0, 5.0, 10., 0.5, 5.0]])
    return coord_obstacles
