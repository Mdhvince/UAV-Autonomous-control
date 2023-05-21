import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
from stl import mesh

warnings.filterwarnings('ignore')
plt.style.use('seaborn-paper')


def timer(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start} seconds to run")
    return wrapper


class Sim3d:
    def __init__(self, config, des_trajectory, state_history):
        """
        This class aims to simulate the quadrotor trajectory in 3D.
        :param config: ConfigParser object
        :param des_trajectory: Desired trajectory of the quadrotor
        :param state_history: State history of the quadrotor
        """
        self.cfg = config["DEFAULT"]
        self.flight_wp = np.array(eval(config["SIM_FLIGHT"].get("waypoints")))

        self.show_obstacles = self.cfg.getboolean("show_obstacles")
        self.track_mode = self.cfg.getboolean("track_mode")
        self.show_stats = self.cfg.getboolean("show_stats")
        # stats cannot be shown in track mode
        assert not (self.track_mode and self.show_stats), "Cannot show stats in track mode"

        self.quad_pos_history = state_history
        self.desired = des_trajectory
        self.des_raw = des_trajectory
        self.n_time_steps = self.quad_pos_history.shape[0]

        self.quad_model = self._init_quadrotor_model()
        self._reduce_data(factor=2)
        self._setup_plot()

        try:
            self.coord_obstacles = np.array(eval(config["SIM_FLIGHT"].get("coord_obstacles")))
        except TypeError:
            self.coord_obstacles = None

        self.xy_max_limit = 11  # plot limits on x and y axis

    def run_sim(self, frames, interval, *args):
        ani = animation.FuncAnimation(self.fig, self.animate, frames=frames, interval=interval, fargs=(args))
        return ani

    def animate(self, index):
        self.update_plot(index)

    def update_plot(self, index):
        self.ax.clear()

        current_position = self.quad_pos_history[index, :3]
        current_orientation = self.quad_pos_history[index, 3:6]

        self.draw_quad(current_position, current_orientation)
        self.draw_desired_trajectory()
        self.draw_executed_trajectory(index)
        self.draw_flight_waypoints()

        if self.show_stats:
            current_velocity_wo_takeoff = np.linalg.norm(self.quad_pos_history[index, 6:9])
            current_position_wo_takeoff = self.quad_pos_history[index, :3]
            self.draw_stats(current_position_wo_takeoff, current_velocity_wo_takeoff, index)

        if self.show_obstacles and self.coord_obstacles is not None:
            self.draw_obstacles()

        self.format_axis()


    def format_axis(self):
        self.ax.w_xaxis.set_pane_color((1, 1, 1, 0))
        self.ax.w_yaxis.set_pane_color((1, 1, 1, 0))          # white
        self.ax.w_zaxis.set_pane_color((.25, .25, .25, 0.8))  # set the floor color to #404854

        # add more grids resolution
        self.ax.set_xticks(np.arange(0, self.xy_max_limit, 1))
        self.ax.set_yticks(np.arange(0, self.xy_max_limit, 1))

        # labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.ax.set_xlim(0, self.xy_max_limit)
        self.ax.set_ylim(0, self.xy_max_limit)
        self.ax.set_zlim(0, self.xy_max_limit)

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

    def draw_flight_waypoints(self):
        self.ax.plot(self.flight_wp[:, 0], self.flight_wp[:, 1], self.flight_wp[:, 2], 'ro', markersize=8, alpha=.2)

    def draw_executed_trajectory(self, index):
        """
        draw the executed path of the quadrotor from the beginning until the current position
        """
        self.ax.plot(self.quad_pos_history[:index, 0], self.quad_pos_history[:index, 1],
                     self.quad_pos_history[:index, 2], color="r", alpha=.7, linewidth=1)

    def draw_desired_trajectory(self):
        self.ax.plot(self.desired[:, 0], self.desired[:, 1], self.desired[:, 2], color="g", alpha=.3, linewidth=2)

    def draw_obstacles(self):
        x, y, z = np.indices((self.xy_max_limit, self.xy_max_limit, 5))  # space limits where cuboids can be placed

        for obs in self.coord_obstacles:
            x_min, x_max = obs[:2]
            y_min, y_max = obs[2:4]
            z_min, z_max = obs[4:]

            # represent the cuboid as a binary array
            cube = np.logical_and.reduce((
                x_min <= x, x < x_max,  # x is between x_min and x_max
                y_min <= y, y < y_max,  # y_min <= y <= y_max
                z_min <= z, z < z_max  # z_min <= z <= z_max
            ))

            colors = np.empty(cube.shape, dtype=object)
            colors[cube] = '#9881F3'
            self.ax.voxels(cube, facecolors=colors, alpha=.5)

    def draw_stats(self, current_position, current_velocity, index):
        if index % 10 == 0 and index != 0:
            # position error over time
            self.ax_pos.plot(index, self.des_raw[index, 0] - current_position[0], 'ro', markersize=2, alpha=.5)
            self.ax_pos.plot(index, self.des_raw[index, 1] - current_position[1], 'go', markersize=2, alpha=.5)
            self.ax_pos.plot(index, self.des_raw[index, 2] - current_position[2], 'bo', markersize=2, alpha=.5)

            # velocity over time in orange
            self.ax_vel.plot(index, current_velocity, 'o', color="orange", markersize=2, alpha=1.)
            # desired velocity over time
            self.ax_vel.plot(index, np.linalg.norm(self.des_raw[index, 3:6]), 'ko', markersize=2, alpha=.5)

            if index == 10:
                self.ax_pos.legend(["x error", "y error", "z error"], loc="best")
                self.ax_pos.set_ylabel("Position error (m)")
                self.ax_pos.set_ylim(-0.2, 0.2)
                self.ax_pos.set_xlim(0, self.n_time_steps)

                self.ax_vel.legend(["current velocity", "desired velocity"], loc="best")
                self.ax_vel.set_xlabel("Timestep")
                self.ax_vel.set_ylabel("Velocity (m/s)")
                self.ax_vel.set_ylim(0, 5)
                self.ax_vel.set_xlim(0, self.n_time_steps)

                plt.tight_layout()

    def _reduce_data(self, factor=2):
        """
        Reduces the number of points in the desired trajectory by a factor of "factor"
        """
        self.desired = self.desired[::factor]

    def _setup_plot(self):
        """
        Sets up the figure and the colormap
        """
        self.fig = plt.figure(figsize=(32, 18))
        if self.track_mode:
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            if self.show_stats:
                self.ax = self.fig.add_subplot(121, projection='3d')
                self.ax_pos = self.fig.add_subplot(222)
                self.ax_vel = self.fig.add_subplot(224)
            else:
                self.ax = self.fig.add_subplot(111, projection='3d')

        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        elev_azim = eval(self.cfg.get("elev_azim"))
        self.ax.view_init(elev=elev_azim[0], azim=elev_azim[1])

    def _init_quadrotor_model(self):
        """
        Loads the quadrotor model from an STL file and scales it by a factor of "scale"
        """
        scale = self.cfg.getfloat("scale")
        quad_model = mesh.Mesh.from_file(self.cfg.get("stl_filepath"))
        quad_model.x *= scale
        quad_model.y *= scale
        quad_model.z *= scale
        return quad_model


    @staticmethod
    def save_sim(ani, filepath):
        writer = animation.FFMpegFileWriter(fps=60)
        ani.save(filepath, writer=writer)








if __name__ == "__main__":
    fig = plt.figure(figsize=(32, 18))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = np.indices((30, 30, 30))  # space limits where cuboids can be placed

    x_min, x_max = 0, 3
    y_min, y_max = 3, 5
    z_min, z_max = 0, 10

    # represent the cuboid as a binary array
    cube = np.logical_and.reduce((
        x_min <= x, x < x_max,  # x is between x_min and x_max
        y_min <= y, y < y_max,  # y_min <= y <= y_max
        z_min <= z, z < z_max   # z_min <= z <= z_max
    ))

    colors = np.empty(cube.shape, dtype=object)
    colors[cube] = 'gray'
    ax.voxels(cube, facecolors=colors, alpha=.7)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.show()
