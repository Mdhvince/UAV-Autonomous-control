import ast
import configparser
from pathlib import Path

import numpy as np


def get_config():
    """
    :return: config object (default, flight)
    """
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config_file = Path(Path(__file__).parent, "config.ini")
    config.read(config_file)

    cfg = config["DEFAULT"]
    cfg_flight = config["SIM_FLIGHT"]

    return cfg, cfg_flight


def parse_array(section: configparser.SectionProxy, key: str) -> np.ndarray:
    """
    :param section: config section holding the entry
    :param key: name of the entry containing a Python list literal
    :return: the parsed entry as a numpy array
    """
    return np.array(ast.literal_eval(section.get(key)))
