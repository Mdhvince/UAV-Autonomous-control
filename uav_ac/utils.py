import configparser
from pathlib import Path


def get_config():
    """
    :return: config object (default, rrt, flight, vehicle, controller)
    """
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config_file = Path(Path(__file__).parent, "config.ini")
    config.read(config_file)

    cfg = config["DEFAULT"]
    cfg_rrt = config["RRT"]
    cfg_flight = config["SIM_FLIGHT"]
    cfg_vehicle = config["VEHICLE"]
    cfg_controller = config["CONTROLLER"]

    return cfg, cfg_rrt, cfg_flight, cfg_vehicle, cfg_controller
