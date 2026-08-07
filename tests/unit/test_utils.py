import configparser

import numpy as np
import pytest

from uav_ac import utils
from uav_ac.simulation.mujoco_sim import MujocoSimulation


def test_get_config_returns_all_sections():
    # Arrange
    # Act
    cfg, cfg_flight = utils.get_config()

    # Assert
    assert cfg.getint("frequency") > 0
    assert cfg_flight.getfloat("velocity") > 0


def test_laboratory_course_should_provide_compact_multi_challenge_route():
    # Arrange
    simulation = MujocoSimulation()

    # Act
    world_size = simulation.space_limits[1] - simulation.space_limits[0]
    segment_lengths = np.linalg.norm(np.diff(simulation.mission_waypoints, axis=0), axis=1)

    # Assert
    assert world_size[0] == pytest.approx(24.0)
    assert world_size[1] == pytest.approx(14.0)
    assert np.sum(segment_lengths) > 25.0
    assert len(simulation.mission_waypoints) == 9


def test_laboratory_course_should_use_obstacles_at_multiple_heights():
    # Arrange
    simulation = MujocoSimulation()

    # Act
    obstacle_lower_altitudes = -simulation.obstacles[:, 5]
    obstacle_upper_altitudes = -simulation.obstacles[:, 4]

    # Assert
    assert np.any(obstacle_lower_altitudes > 2.0)
    assert np.any(np.isclose(obstacle_lower_altitudes, 0.0))
    assert np.any(obstacle_upper_altitudes < 3.0)


@pytest.mark.parametrize("waypoint_index", range(8))
def test_laboratory_course_should_keep_mission_waypoints_inside_planning_limits(waypoint_index):
    # Arrange
    simulation = MujocoSimulation()
    location = simulation.mission_waypoints[waypoint_index]

    # Act
    is_inside = (
        np.all(location >= simulation.space_limits[0])
        and np.all(location <= simulation.space_limits[1])
    )

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
