import configparser

import numpy as np
import pytest

from uav_ac import utils


def test_get_config_returns_all_sections():
    # Arrange
    # Act
    cfg, cfg_rrt, cfg_flight = utils.get_config()

    # Assert
    assert cfg.getfloat("g") == pytest.approx(9.81)
    assert cfg_rrt.getint("max_iterations") > 0
    assert cfg_flight.getfloat("velocity") > 0


def test_large_world_should_provide_long_route_and_many_visible_obstacles():
    # Arrange
    _, cfg_rrt, cfg_flight = utils.get_config()
    space_limits = utils.parse_array(cfg_rrt, "space_limits")
    start = utils.parse_array(cfg_flight, "start_loc")
    goal = utils.parse_array(cfg_flight, "goal_loc")
    obstacles = utils.parse_array(cfg_flight, "coord_obstacles")
    visible_obstacle_count = cfg_flight.getint("visible_obstacle_count")

    # Act
    world_size = space_limits[1] - space_limits[0]
    route_length = np.linalg.norm(goal - start)

    # Assert
    assert world_size[0] >= 80.0
    assert world_size[1] >= 60.0
    assert route_length >= 90.0
    assert visible_obstacle_count >= 18
    assert len(obstacles) > visible_obstacle_count
    assert cfg_rrt.getint("random_seed") >= 0


def test_large_world_should_use_city_blocks_that_cannot_be_overflown():
    # Arrange
    _, cfg_rrt, cfg_flight = utils.get_config()
    space_limits = utils.parse_array(cfg_rrt, "space_limits")
    start = utils.parse_array(cfg_flight, "start_loc")
    goal = utils.parse_array(cfg_flight, "goal_loc")
    obstacles = utils.parse_array(cfg_flight, "coord_obstacles")
    visible_obstacle_count = cfg_flight.getint("visible_obstacle_count")

    # Act
    buildings = obstacles[:visible_obstacle_count]
    direct_route_is_blocked = any(
        _segment_intersects_cuboid(start, goal, building) for building in buildings
    )

    # Assert
    assert direct_route_is_blocked
    assert np.all(buildings[:, 4] <= space_limits[0, 2])
    assert np.all(buildings[:, 5] >= 0.0)
    assert np.all((buildings[:, 1] - buildings[:, 0]) >= 5.0)
    assert np.all((buildings[:, 3] - buildings[:, 2]) >= 5.0)


@pytest.mark.parametrize("location_key", ["start_loc", "goal_loc"])
def test_large_world_should_keep_route_endpoints_inside_planning_limits(location_key):
    # Arrange
    _, cfg_rrt, cfg_flight = utils.get_config()
    space_limits = utils.parse_array(cfg_rrt, "space_limits")
    location = utils.parse_array(cfg_flight, location_key)

    # Act
    is_inside = np.all(location >= space_limits[0]) and np.all(location <= space_limits[1])

    # Assert
    assert is_inside


def test_parse_array_reads_list_literal_as_numpy_array():
    # Arrange
    config = configparser.ConfigParser()
    config.read_string("[SECTION]\npoints = [[1.0, 2.0], [3.0, 4.0]]\n")
    section = config["SECTION"]

    # Act
    result = utils.parse_array(section, "points")

    # Assert
    assert result == pytest.approx(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_parse_array_rejects_arbitrary_code():
    # Arrange
    config = configparser.ConfigParser()
    config.read_string("[SECTION]\npoints = __import__('os').getcwd()\n")
    section = config["SECTION"]

    # Act
    # Assert
    with pytest.raises(ValueError):
        utils.parse_array(section, "points")


def _segment_intersects_cuboid(start, end, cuboid):
    direction = end - start
    minimum_time = 0.0
    maximum_time = 1.0

    for axis in range(3):
        low, high = cuboid[2 * axis:2 * axis + 2]
        if abs(direction[axis]) < 1e-12:
            if start[axis] < low or start[axis] > high:
                return False
            continue

        times = sorted(((low - start[axis]) / direction[axis], (high - start[axis]) / direction[axis]))
        minimum_time = max(minimum_time, times[0])
        maximum_time = min(maximum_time, times[1])
        if minimum_time > maximum_time:
            return False

    return True
