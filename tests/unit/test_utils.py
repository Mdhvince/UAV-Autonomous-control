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
