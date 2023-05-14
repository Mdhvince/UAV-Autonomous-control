import configparser
from pathlib import Path

import pytest


@pytest.fixture
def config():
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config_file = Path("/home/medhyvinceslas/Documents/programming/quad3d_sim/tests/conf_test.ini")
    config.read(config_file)
    return config
