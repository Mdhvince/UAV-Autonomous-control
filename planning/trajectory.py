import warnings
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize

from planning.minimum_snap import MinimumSnap

warnings.filterwarnings('ignore')
plt.style.use('ggplot')


def get_path(total_time=20, dt=0.01):
    Desired = namedtuple(
        "Desired", ["x", "y", "z", "x_vel", "y_vel", "z_vel", "x_acc", "y_acc", "z_acc", "yaw"])

    t = np.linspace(0.0, total_time, int(total_time / dt))

    omega_x = 0.8
    omega_y = 0.4
    omega_z = 0.4

    a_x = 1.0
    a_y = 1.0
    a_z = 1.0

    x = a_x * np.sin(omega_x * t)
    x_vel = a_x * omega_x * np.cos(omega_x * t)
    x_acc = -a_x * omega_x ** 2 * np.sin(omega_x * t)

    y = a_y * np.cos(omega_y * t) + 2
    y_vel = -a_y * omega_y * np.sin(omega_y * t)
    y_acc = -a_y * omega_y ** 2 * np.cos(omega_y * t)

    z = a_z * np.cos(omega_z * t) + 2
    z_vel = -a_z * omega_z * np.sin(omega_z * t)
    z_acc = - a_z * omega_z ** 2 * np.cos(omega_z * t)

    yaw = np.arctan2(y_vel, x_vel)

    desired_trajectory = Desired(x, y, z, x_vel, y_vel, z_vel, x_acc, y_acc, z_acc, yaw)

    return t, dt, desired_trajectory


def get_path_helix(total_time=20, r=1, height=10, dt=0.01):
    Desired = namedtuple(
        "Desired", ["x", "y", "z", "x_vel", "y_vel", "z_vel", "x_acc", "y_acc", "z_acc", "yaw"])

    t = np.linspace(0.0, total_time, int(total_time / dt))

    omega_x = 0.8
    omega_y = 0.4

    a_x = r  # radius of helix
    a_y = r  # radius of helix
    a_z = height / total_time  # vertical movement of helix per unit time

    x = a_x * np.sin(omega_x * t)
    x_vel = a_x * omega_x * np.cos(omega_x * t)
    x_acc = -a_x * omega_x ** 2 * np.sin(omega_x * t)

    y = a_y * np.cos(omega_y * t)
    y_vel = -a_y * omega_y * np.sin(omega_y * t)
    y_acc = -a_y * omega_y ** 2 * np.cos(omega_y * t)

    z = a_z * t  # z moves linearly with time
    z_vel = np.full(len(t), a_z)  # z_vel is constant
    z_acc = np.full(len(t), 0)  # z_acc is zero

    yaw = np.arctan2(y_vel, x_vel)

    desired_trajectory = Desired(x, y, z, x_vel, y_vel, z_vel, x_acc, y_acc, z_acc, yaw)

    return t, dt, desired_trajectory


def getwp(form, a=None, phi=None):
    if form == 'angle' and a is None and phi is None:
        w = np.array([[0, 0, 0],
                      [0, 0, 2],
                      [0, 4, 2],
                      [0, 0, 1.1]]).T
    elif form == 'helix' and a is None and phi is None:
        r = 2
        h_max = 20
        t = np.pi * np.arange(0, h_max + 0.4, 0.4)
        x = r * np.cos(t)
        y = r * np.sin(t)
        z = t / np.pi
        w = np.array([x, y, z])
    elif form == 'maneuvre' and a is not None and phi is not None:
        w = np.array([[0, 0, 2],
                      [0, a, 2],
                      [a * np.sin(phi), a * (1 - np.cos(phi)), 2]]).T
    return w


def plot_3d_trajectory_and_obstacle(ax, waypoints, trajectory_obj):
    """
    This function plots the trajectory and the obstacle in 3D. And apply color on the path based on the velocity
    """
    trajectory = trajectory_obj.full_trajectory

    # filter-out some rows to reduce the number of points to plot
    n = 2
    trajectory = trajectory[::n]

    # map color to velocity
    vel = np.linalg.norm(trajectory[:, 3:6], axis=1)
    max_vel = np.max(vel)
    norm = Normalize(vmin=0, vmax=max_vel)
    scalar_map = get_cmap("jet")
    sm = ScalarMappable(cmap=scalar_map, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, location="bottom", shrink=0.5)
    cbar.set_label('Velocity (m/s)')
    colors = scalar_map(norm(vel))

    # plot min snap trajectory
    # for i in range(len(trajectory)):
    #     label = "Minimum snap trajectory" if i == 0 else None
    #     if i > 0:
    #         ax.plot(
    #             [trajectory[i - 1, 0], trajectory[i, 0]],
    #             [trajectory[i - 1, 1], trajectory[i, 1]],
    #             [trajectory[i - 1, 2], trajectory[i, 2]],
    #             color=colors[i], alpha=.2, linewidth=5, label=label)


    # # plot waypoints
    # for i in range(len(waypoints)):
    #     x, y, z = waypoints[i]
    #     ax.plot(x, y, z, marker=".", markersize=10, color="black", label="Waypoints" if i == 0 else None)

    if len(trajectory_obj.obstacle_edges) > 0:
        # plot obstacles edges
        for edges in trajectory_obj.obstacle_edges:
            for edge in edges:
                x, y, z = zip(*edge)
                ax.plot(x, y, z, color="red", alpha=.2)

    # labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')



if __name__ == "__main__":

    waypoints = np.array([
        [10., 0.0, 1.0],
        [10., 4.0, 1.0],
        [6.0, 5.0, 1.5],
        [4.0, 7.0, 1.5],
        [2.0, 7.0, 2.0],
        [1.0, 0.0, 2.0]
    ])

    coord_obstacles = np.array([
        [8.0, 6.0, 1.5, 5.0, 0.0],  # x, y, side_length, height, altitude_start
        [4.0, 9.0, 1.5, 5.0, 0.0],
        [4.0, 1.0, 2.0, 5.0, 0.0],
        [3.0, 5.0, 1.0, 5.0, 0.0],
        [4.0, 3.5, 2.5, 5.0, 0.0],
        # [5.0, 5.0, 10., 0.5, 5.0]
    ])

    T = MinimumSnap(waypoints, velocity=1.0, dt=0.02)
    T.generate_collision_free_trajectory(coord_obstacles=coord_obstacles)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(90, -90)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    plot_3d_trajectory_and_obstacle(ax, waypoints, T)
    plt.show()
