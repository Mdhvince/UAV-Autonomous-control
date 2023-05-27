from mayavi import mlab
import numpy as np

from planning.rrt import RRTStar


class RRTPlotter:
    def __init__(self, rrt: RRTStar, optimal_trajectory, state_history):
        mlab.figure(size=(1920, 1080), bgcolor=(.3, .3, .3))
        self.rrt = rrt
        self.optimal_trajectory = optimal_trajectory
        self.state_history = state_history


    def animate_mesh(self, mesh, rpy=(True, True, True), delay=60):
        r, p, y = rpy

        @mlab.animate(delay=delay)
        def anim():
            for coord in self.state_history:
                x, y, z = coord[:3]

                if r or p or y:
                    roll, pitch, yaw = coord[3:6]

                    if r: mesh.actor.actor.rotate_y(roll)
                    if p: mesh.actor.actor.rotate_x(pitch)
                    if y: mesh.actor.actor.rotate_z(yaw)

                mesh.actor.actor.position = (x, y, z)
                yield
        anim()


    def animate_point(self, moving_object, delay=60):
        @mlab.animate(delay=delay)
        def anim():
            for coord in self.state_history:
                x, y, z = coord[:3]
                moving_object.mlab_source.set(x=x, y=y, z=z)
                yield
        anim()


    def plot_executed_trajectory(self):
        mlab.plot3d(
            self.state_history[:, 0],
            self.state_history[:, 1],
            self.state_history[:, 2], color=(0, 0, 1), tube_radius=0.04, opacity=0.3)


    def plot_start_and_goal(self, color_start=(1, 0, 0), color_goal=(0, 1, 0)):
        start = self.rrt.start
        goal = self.rrt.goal
        mlab.points3d(*start, color=color_start, scale_factor=0.2, resolution=60)
        mlab.points3d(*goal, color=color_goal, scale_factor=0.2, resolution=60)


    @staticmethod
    def plot_obstacles(obstacles, color_obs=(.6, .6, .6), color_true_obs=(.9, 0, 0), true_obstacles_size_factor=0.5):
        offset = 0.5  # need to offset the obstacle by 0.5 due to mayavi way of plotting cubes

        for obstacle in obstacles:
            xx, yy, zz = np.mgrid[obstacle[0] + offset:obstacle[1]:1,
                                  obstacle[2] + offset:obstacle[3]:1,
                                  obstacle[4] + offset:obstacle[5]:1]
            mlab.points3d(xx, yy, zz, color=color_obs, scale_factor=1, mode='cube', opacity=0.2)

        # true obstacles (non-enhanced):
        rm = true_obstacles_size_factor / 2  # remove factor/2 from each side and factor from height
        offset = offset + rm
        matrix_factor = np.ones(obstacles.shape) * rm if rm > 0 else np.ones(obstacles.shape)

        matrix_factor[:, [1, 3, 5]] = matrix_factor[:, [1, 3, 5]] * -1
        obstacles_true = obstacles + matrix_factor

        # ensure z_min is 0 if z_min=rm
        obstacles_true[:, 4] = np.where(obstacles_true[:, 4] == rm, 0, obstacles_true[:, 4])

        for obstacle in obstacles_true:
            xx, yy, zz = (
                np.mgrid[obstacle[0] + offset:obstacle[1]:1,
                         obstacle[2] + offset:obstacle[3]:1,
                         obstacle[4] + offset - rm:obstacle[5]:1]
            )
            mlab.points3d(xx, yy, zz, color=color_true_obs, scale_factor=1, mode='cube', opacity=1)


    def plot_tree(self, color_node=(0, 0, 1), color_edges=(0, 0, 0)):
        """
        Plot the nodes and connections between the nodes and their parents
        """
        for node, parent in self.rrt.best_tree.items():
            node = np.array(eval(node))
            mlab.points3d(node[0], node[1], node[2], color=color_node, scale_factor=.1, opacity=0.1)
            mlab.points3d(parent[0], parent[1], parent[2], color=color_node, scale_factor=.1, opacity=0.1)
            mlab.plot3d([node[0], parent[0]], [node[1], parent[1]], [node[2], parent[2]],
                        color=color_edges, tube_radius=0.01, opacity=0.1)


    def plot_path(self, color_path=(1, 1, 0)):
        path = self.rrt.best_path
        mlab.plot3d(path[:, 0], path[:, 1], path[:, 2], color=color_path, tube_radius=0.02)


    def plot_trajectory(self, color_traj=(0, 1, 1)):
        mlab.plot3d(
            self.optimal_trajectory[:, 0],
            self.optimal_trajectory[:, 1], self.optimal_trajectory[:, 2], tube_radius=0.02, color=color_traj)


if __name__ == "__main__":
    pass