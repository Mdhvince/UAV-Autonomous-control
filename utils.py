import configparser
from pathlib import Path, PosixPath

from mayavi import mlab


def get_config():
    """
    :return: config object (default, rrt, flight, vehicle, controller)
    """
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config_file = Path("/home/medhyvinceslas/Documents/programming/quad3d_sim/config.ini")
    config.read(config_file)

    cfg = config["DEFAULT"]
    cfg_rrt = config["RRT"]
    cfg_flight = config["SIM_FLIGHT"]
    cfg_vehicle = config["VEHICLE"]
    cfg_controller = config["CONTROLLER"]

    return cfg, cfg_rrt, cfg_flight, cfg_vehicle, cfg_controller


def read_stl_file_mayavi(file_path, color=(0, 0, 0)):
    """
    :param file_path: path to the stl file
    :param color: color of the mesh
    :return: mlab mesh
    """
    if isinstance(file_path, PosixPath):
        file_path = str(file_path)


    mesh = mlab.pipeline.open(file_path)
    mesh = mlab.pipeline.surface(mesh, color=color)
    return mesh
