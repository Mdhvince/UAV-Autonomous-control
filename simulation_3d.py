import copy
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
from stl import mesh

warnings.filterwarnings('ignore')
plt.style.use('ggplot')


def timer(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start} seconds to run")
    return wrapper


class Sim3d:

    def __init__(self, des_trajectory, state_history, obstacles_edges=None, stl_filepath="", scale=1, colormap="jet"):
        """
        This class aims to simulate the quadrotor trajectory in 3D
        """
        self.quad_pos_history = state_history
        self.obstacles_edges = obstacles_edges
        self.desired = des_trajectory

        self.stl_filepath = stl_filepath
        self.scale = scale
        self.quad_model = self._init_quadrotor_model()

        self._reduce_data(factor=2)
        self._setup_plot(colormap)


    def run_sim(self, frames, interval, *args):
        ani = animation.FuncAnimation(self.fig, self.animate, frames=frames, interval=interval, fargs=(args))
        return ani

    def animate(self, index):
        self.update_plot(index)

    def update_plot(self, index):
        self.ax.clear()

        current_position = self.quad_pos_history[index, :3]
        current_orientation = self.quad_pos_history[index, 3:6]
        current_orientation[0] = -current_orientation[0]
        current_orientation[1] = -current_orientation[1]
        current_velocity = np.linalg.norm(self.quad_pos_history[index, 6:9])

        self.ax_2d.plot(index, current_velocity, 'ro', markersize=2, alpha=.5)
        self.ax_2d.set_xlim(0, len(self.quad_pos_history))
        self.ax_2d.set_ylim(0, 5)
        self.ax_2d.set_xlabel("Timestep")
        self.ax_2d.set_ylabel("Velocity (m/s)")
        plt.tight_layout()

        # if self.obstacles_edges is not None:
        #     self.draw_obstacles()
        self.draw_quad(current_position, current_orientation)
        self.draw_trajectory()

        self.ax.set_xlim(0, 11)
        self.ax.set_ylim(0, 11)
        self.ax.set_zlim(0, 11)
        self.ax.xaxis.line.set_color('black')
        self.ax.yaxis.line.set_color('black')
        self.ax.zaxis.line.set_color('black')

        # add text on the figure showing the current position and orientation in degrees
        self.ax.text2D(0.05, 0.95, f"XYZ: {np.round(current_position, 2)}", transform=self.ax.transAxes)
        self.ax.text2D(0.05, 0.90, f"RPY: {np.round(np.rad2deg(current_orientation), 2)}", transform=self.ax.transAxes)

    def draw_quad(self, current_position, current_orientation):
        # plot the quadrotor according to its current position
        self.quad_model = self._init_quadrotor_model()  # reset the quadrotor model to avoid deepcopy
        self.quad_model.x += current_position[0]
        self.quad_model.y += current_position[1]
        self.quad_model.z += current_position[2]

        # plot the quadrotor according to its current orientation
        self.quad_model.rotate([1, 0, 0], current_orientation[0], point=current_position)
        self.quad_model.rotate([0, 1, 0], current_orientation[1], point=current_position)
        self.quad_model.rotate([0, 0, 1], current_orientation[2], point=current_position)

        self.ax.add_collection3d(mplot3d.art3d.Poly3DCollection(self.quad_model.vectors, alpha=.2, color="gray"))


    def draw_trajectory(self):
        self.ax.plot(self.desired[:, 0], self.desired[:, 1], self.desired[:, 2], color="k", alpha=1, linewidth=2)


    def draw_obstacles(self):
        for edges in self.obstacles_edges:
            for edge in edges:
                x, y, z = zip(*edge)
                self.ax.plot(x, y, z, color="red", alpha=.2)


    def draw_axis(self, rot_mat, current_position, onboard_axis=False):
        # Axis Body frame
        px, py, pz = current_position
        if onboard_axis:
            scale = 0.3
            roll_axis = (rot_mat[:, 0] * scale) + current_position
            pitch_axis = (rot_mat[:, 1] * scale) + current_position
            yaw_axis = -(rot_mat[:, 2] * scale) + current_position
            self.ax.plot([px, roll_axis[0]], [py, roll_axis[1]], [pz, roll_axis[2]], 'r', lw=3)
            self.ax.plot([px, pitch_axis[0]], [py, pitch_axis[1]], [pz, pitch_axis[2]], 'g', lw=3)
            self.ax.plot([px, yaw_axis[0]], [py, yaw_axis[1]], [pz, yaw_axis[2]], 'b', lw=3)
        else:
            # Define the starting and ending points of the axes
            scale = 1.5
            roll_axis = rot_mat[:, 0] * scale
            pitch_axis = rot_mat[:, 1] * scale
            yaw_axis = rot_mat[:, 2] * scale

            x0, y0, z0 = [3, -3, -3]
            self.ax.quiver(x0, y0, z0, roll_axis[0], roll_axis[1], roll_axis[2], color='r', lw=2)
            self.ax.quiver(x0, y0, z0, pitch_axis[0], pitch_axis[1], pitch_axis[2], color='g', lw=2)
            self.ax.quiver(x0, y0, z0, yaw_axis[0], yaw_axis[1], yaw_axis[2], color='b', lw=2)

    def _reduce_data(self, factor=2):
        """
        Reduces the number of points in the desired trajectory by a factor of "factor"
        """
        self.desired = self.desired[::factor]

    def _setup_plot(self, colormap):
        """
        Sets up the figure and the colormap
        """
        # init figure
        self.fig = plt.figure(figsize=(16, 9))
        self.ax = self.fig.add_subplot(121, projection='3d')
        self.ax_2d = self.fig.add_subplot(122)

        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax.view_init(15, -133)

        # set colormap proportional to velocity
        # self.colormap = colormap
        # vel = np.linalg.norm(self.desired[:, 3:6], axis=1)
        # max_vel = np.max(vel)
        # self.norm = Normalize(vmin=0, vmax=max_vel)
        # self.scalar_map = get_cmap(self.colormap)
        # sm = ScalarMappable(cmap=self.scalar_map, norm=self.norm)
        # sm.set_array([])
        # self.cbar = plt.colorbar(sm, shrink=0.5)
        # self.cbar.set_label('Velocity (m/s)')
        # self.colors = self.scalar_map(self.norm(vel))

    def _init_quadrotor_model(self):
        """
        Loads the quadrotor model from an STL file and scales it by a factor of "scale"
        """
        quad_model = mesh.Mesh.from_file(self.stl_filepath)
        quad_model.x *= self.scale
        quad_model.y *= self.scale
        quad_model.z *= self.scale
        return quad_model


    @staticmethod
    def quad_pos(current_position, rot_mat, L, H=.05):
        pos = current_position.reshape(-1, 1)  # Convert current position to column vector

        # Create homogeneous transformation from body to world
        wHb = np.hstack((rot_mat, pos))
        wHb = np.vstack((wHb, [0, 0, 0, 1]))

        # Points in body frame
        quad_body_frame = np.array([[L, 0, 0, 1],
                                    [0, L, 0, 1],
                                    [-L, 0, 0, 1],
                                    [0, -L, 0, 1],
                                    [0, 0, 0, 1],
                                    [0, 0, H, 1]]).T

        quad_world_frame = np.dot(wHb, quad_body_frame)  # Transform points to world frame
        quad_wf = quad_world_frame[:3, :]  # Points of the quadrotor in world frame

        return quad_wf

    @staticmethod
    def euler2Rot(phi, theta, psi):

        r_x = np.array([[1, 0, 0],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi), np.cos(phi)]])

        r_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]])

        r_z = np.array([[np.cos(psi), -np.sin(psi), 0],
                        [np.sin(psi), np.cos(psi), 0],
                        [0, 0, 1]])
        r_yx = np.matmul(r_y, r_x)
        rot_mat = np.matmul(r_z, r_yx)
        return rot_mat

    @staticmethod
    def save_sim(ani, filepath):
        writer = animation.FFMpegFileWriter(fps=60)
        ani.save(filepath, writer=writer)





if __name__ == "__main__":
    pass
