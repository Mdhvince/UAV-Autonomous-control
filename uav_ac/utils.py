import configparser
from pathlib import Path


def get_config():
    """
    :return: config object (default, rrt, flight)
    """
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config_file = Path(Path(__file__).parent, "config.ini")
    config.read(config_file)

    cfg = config["DEFAULT"]
    cfg_rrt = config["RRT"]
    cfg_flight = config["SIM_FLIGHT"]

    return cfg, cfg_rrt, cfg_flight
