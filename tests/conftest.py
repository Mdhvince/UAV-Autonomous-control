import numpy as np
import pytest

from uav_ac.planning.rrt import RRTStar


@pytest.fixture
def rrt_object():
    return RRTStar(space_limits=np.array([[0, 0, 0], [10, 10, 10]]),
                   start=np.array([0, 0, 0]),
                   goal=np.array([8, 8, 8]),
                   max_distance=2,
                   max_iterations=1)
