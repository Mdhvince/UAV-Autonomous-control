import numpy as np
import plotly.graph_objects as go


class RRTPlotter:
    def __init__(self, rrt, optimal_trajectory, state_history):
        self.fig = go.Figure()
        self.rrt = rrt
        self.optimal_trajectory = optimal_trajectory
        self.state_history = state_history

    def plot_executed_trajectory(self):
        self.fig.add_trace(go.Scatter3d(
            x=self.state_history[:, 0],
            y=self.state_history[:, 1],
            z=self.state_history[:, 2],
            mode='lines',
            line=dict(color='blue', width=4),
            opacity=0.3
        ))

    def plot_start_and_goal(self, color_start='red', color_goal='green'):
        start = self.rrt.start
        goal = self.rrt.goal
        self.fig.add_trace(go.Scatter3d(
            x=[start[0]], y=[start[1]], z=[start[2]],
            mode='markers',
            marker=dict(size=5, color=color_start)
        ))
        self.fig.add_trace(go.Scatter3d(
            x=[goal[0]], y=[goal[1]], z=[goal[2]],
            mode='markers',
            marker=dict(size=5, color=color_goal)
        ))

    def plot_obstacles(self, obstacles, color_obs='gray', color_true_obs='red', true_obstacles_size_factor=0.5):
        offset = 0.5
        for obstacle in obstacles:
            xx, yy, zz = np.mgrid[obstacle[0] + offset:obstacle[1]:1,
                                  obstacle[2] + offset:obstacle[3]:1,
                                  obstacle[4] + offset:obstacle[5]:1]

            self.fig.add_trace(go.Scatter3d(
                x=xx.flatten(), y=yy.flatten(), z=zz.flatten(),
                mode='markers',
                marker=dict(size=2, color=color_obs, opacity=0.2)
            ))

        rm = true_obstacles_size_factor / 2
        offset += rm
        matrix_factor = np.ones(obstacles.shape) * rm if rm > 0 else np.ones(obstacles.shape)
        matrix_factor[:, [1, 3, 5]] *= -1
        obstacles_true = obstacles + matrix_factor
        obstacles_true[:, 4] = np.where(obstacles_true[:, 4] == rm, 0, obstacles_true[:, 4])

        for obstacle in obstacles_true:
            xx, yy, zz = np.mgrid[obstacle[0] + offset:obstacle[1]:1,
                                  obstacle[2] + offset:obstacle[3]:1,
                                  obstacle[4] + offset - rm:obstacle[5]:1]

            self.fig.add_trace(go.Scatter3d(
                x=xx.flatten(), y=yy.flatten(), z=zz.flatten(),
                mode='markers',
                marker=dict(size=2, color=color_true_obs, opacity=1)
            ))

    def plot_tree(self, color_node='blue', color_edges='black'):
        for node, parent in self.rrt.best_tree.items():
            node = np.array(eval(node))
            parent = np.array(parent)
            self.fig.add_trace(go.Scatter3d(
                x=[node[0], parent[0]], y=[node[1], parent[1]], z=[node[2], parent[2]],
                mode='lines',
                line=dict(color=color_edges, width=1),
                opacity=0.1
            ))
            self.fig.add_trace(go.Scatter3d(
                x=[node[0]], y=[node[1]], z=[node[2]],
                mode='markers',
                marker=dict(size=2, color=color_node, opacity=0.1)
            ))
            self.fig.add_trace(go.Scatter3d(
                x=[parent[0]], y=[parent[1]], z=[parent[2]],
                mode='markers',
                marker=dict(size=2, color=color_node, opacity=0.1)
            ))

    def plot_path(self, color_path='yellow'):
        path = self.rrt.best_path
        self.fig.add_trace(go.Scatter3d(
            x=path[:, 0], y=path[:, 1], z=path[:, 2],
            mode='lines',
            line=dict(color=color_path, width=2)
        ))

    def plot_trajectory(self, color_traj='cyan'):
        self.fig.add_trace(go.Scatter3d(
            x=self.optimal_trajectory[:, 0],
            y=self.optimal_trajectory[:, 1],
            z=self.optimal_trajectory[:, 2],
            mode='lines',
            line=dict(color=color_traj, width=2)
        ))

    def show(self):
        self.fig.show()


if __name__ == "__main__":
    pass
