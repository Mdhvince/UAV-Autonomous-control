import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.animation as animation

plt.style.use('dark_background')


def plot_results(t, state_history, omega_history, desired):
    plt.style.use('seaborn-paper')
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    plot_trajectory(ax, state_history, desired)

    ax = fig.add_subplot(2, 2, 2)
    plot_yaw_angle(t, desired, state_history)

    ax = fig.add_subplot(2, 2, 3)
    plot_props_speed(t, omega_history)

    ax = fig.add_subplot(2, 2, 4)
    plot_position_error(t, state_history, desired)
    plt.show()

def plot_trajectory(ax, drone_state_history, desired):
    ax.plot(desired.x, desired.y, desired.z,linestyle='-',marker='.',color='red')
    ax.plot(drone_state_history[:,0],
            drone_state_history[:,1],
            drone_state_history[:,2],
            linestyle='-',color='blue')

    plt.title('Flight path')
    ax.set_xlabel('$x$ [$m$]')
    ax.set_ylabel('$y$ [$m$]')
    ax.set_zlabel('$z$ [$m$]')
    plt.legend(['Planned path','Executed path'])

def plot_yaw_angle(t, desired, drone_state_history):
    plt.plot(t,desired.yaw,marker='.')
    plt.plot(t,drone_state_history[:-1,5])
    plt.title('Yaw angle')
    plt.xlabel('$t$ [$s$]')
    plt.ylabel('$\psi$ [$rad$]')
    plt.legend(['Planned yaw','Executed yaw'])

def plot_props_speed(t, omega_history):
    plt.plot(t, -omega_history[:-1,0],color='blue')
    plt.plot(t, omega_history[:-1,1],color='red')
    plt.plot(t, -omega_history[:-1,2],color='green')
    plt.plot(t, omega_history[:-1,3],color='black')
    plt.title('Angular velocities')
    plt.xlabel('$t$ [$s$]')
    plt.ylabel('$\omega$ [$rad/s$]')
    plt.legend(['P 1','P 2','P 3','P 4' ])

def plot_position_error(t, drone_state_history, desired):
    err_x = np.sqrt((desired.x-drone_state_history[:-1,0])**2)
    err_y = np.sqrt((desired.y-drone_state_history[:-1,1])**2)
    err_z = np.sqrt((desired.z-drone_state_history[:-1,2])**2)

    plt.plot(t, err_x)
    plt.plot(t, err_y)
    plt.plot(t, err_z)
    plt.title('Error in flight position')
    plt.xlabel('$t$ [$s$]')
    plt.ylabel('$e$ [$m$]')
    plt.ylim(0, .02)
    plt.legend(['x', 'y', 'z'])


class Sim3d():

    def __init__(self, desired_trajectory, quad_pos_history, obstacles_edges=None, colormap="jet"):
        
        self.desired = desired_trajectory
        # only keep some of the rows
        n = 2
        for _ in range(3):
            mask = np.ones(self.desired.shape[0], dtype=bool)
            mask[::n] = False
            self.desired = self.desired[mask]

        self.quad_pos_history = quad_pos_history
        self.obstacles_edges = obstacles_edges
        self.colormap = colormap

        self.fig = plt.figure(figsize=(20,20))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        vel = np.linalg.norm(self.desired[:, 3:6], axis=1)
        max_vel = np.max(vel)

        self.norm = Normalize(vmin=0, vmax=max_vel)
        self.scalar_map = get_cmap(self.colormap)
        sm = ScalarMappable(cmap=self.scalar_map, norm=self.norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, location="bottom", shrink=0.5)
        cbar.set_label('Velocity (m/s)')
        self.colors = self.scalar_map(self.norm(vel))

    def run_sim(self, frames, interval, *args):
        ani = animation.FuncAnimation(self.fig, self.animate, frames=frames, interval=interval, fargs=(args))
        return ani

    def animate(self, index):
        self.vehicle_3d_pos(index)

    def vehicle_3d_pos(self, index):
        self.ax.clear()

        phi, theta, psi = -self.quad_pos_history[index, 3:6]  # add negative for mirroring effect of matplotlib
        rot_mat = Sim3d.euler2Rot(phi, theta, psi)
        current_position = self.quad_pos_history[index, :3]
        
        if self.obstacles_edges is not None:
            self.draw_obstacles()
        self.draw_quad(current_position, rot_mat)
        self.draw_trajectory()
        self.draw_axis(rot_mat, current_position)
        
        
        self.ax.w_zaxis.pane.set_color('#2D3047')
        self.ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.grid(False)
        self.ax.legend(facecolor="gray", bbox_to_anchor=(1, 1), loc='upper left')
        self.ax.set_xlim(0, 11)
        self.ax.set_ylim(0, 11)
        self.ax.set_zlim(0, 11)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.xaxis.line.set_color('black')
        self.ax.yaxis.line.set_color('black')
        self.ax.zaxis.line.set_color('black')

    def draw_quad(self, current_position, rot_mat):
        quad_wf = self.quad_pos(current_position, rot_mat, L=.7, H=0.005)
        x, y, z = quad_wf[0, :], quad_wf[1, :], quad_wf[2, :]

        self.ax.scatter(x, y, z, alpha=.2, color="red")

        COM, FL, FR, RL, RR = 5, 0, 1, 3, 2
        # COM to front left / rigth / rear-left / rear-right
        self.ax.plot([x[COM], x[FL]], [y[COM], y[FL]], [z[COM], z[FL]], '-', color='y', label="Front")
        self.ax.plot([x[FR], x[COM]], [y[FR], y[COM]], [z[FR], z[COM]], '-', color='y')
        self.ax.plot([x[COM], x[RL]], [y[COM], y[RL]], [z[COM], z[RL]], '-', color='w', label="Rear")
        self.ax.plot([x[COM], x[RR]], [y[COM], y[RR]], [z[COM], z[RR]], '-', color='w')
        # contour of the quad
        # ax.plot([x[FL], x[FR], x[RR], x[RL], x[FL]],
        #         [y[FL], y[FR], y[RR], y[RL], y[FL]],
        #         [z[FL], z[FR], z[RR], z[RL], z[FL]], '-')
        # shadow
        px, py, pz = current_position
        self.ax.plot(px, py, 0, color='black', alpha=0.5, markersize=5-pz, marker='o')

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

    def draw_trajectory(self):
    
        self.ax.plot(self.desired[:, 0], self.desired[:, 1], self.desired[:, 2],
                marker='.', alpha=.2, markersize=20)

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
        quad_wf = quad_world_frame[:3, :]                # Points of the quadrotor in world frame

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
                        [0,0,1]])
        r_yx = np.matmul(r_y, r_x)
        rot_mat = np.matmul(r_z, r_yx)
        return rot_mat

    @staticmethod
    def save_sim(ani, filepath):
        writer=animation.FFMpegFileWriter(fps=60)
        ani.save(filepath, writer=writer)

    def min_snap_plots():
        """
        - independently plot X, Y, Z
        - magnitude of Vel, Acc, Jerk, Snap over steps
        - final 3d trajectory
        """
        pass

    def draw_obstacles(self):
        for edges in self.obstacles_edges:
            for edge in edges:
                x, y, z = zip(*edge)
                self.ax.plot(x, y, z, color="red", alpha=.2)

        


if __name__ == "__main__":
    pass
    

    