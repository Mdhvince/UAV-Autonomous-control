import math
import warnings
import configparser

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize

from quadrotor import Quadrotor
from controller import Controller
from trajectory import get_path, get_path_random, get_path_straight, get_path_helix

warnings.filterwarnings('ignore')


def plot_trajectory(ax, drone_state_history, desired):
    ax.plot(desired.x, desired.y, desired.z,linestyle='-',marker='.',color='red')
    ax.plot(drone_state_history[:,0],
            drone_state_history[:,1],
            drone_state_history[:,2],
            linestyle='-',color='blue')

    plt.title('Flight path').set_fontsize(20)
    ax.set_xlabel('$x$ [$m$]').set_fontsize(20)
    ax.set_ylabel('$y$ [$m$]').set_fontsize(20)
    ax.set_zlabel('$z$ [$m$]').set_fontsize(20)
    plt.legend(['Planned path','Executed path'],fontsize = 14)
    plt.show()

def plot_yaw_angle(t, desired, drone_state_history):
    plt.plot(t,desired.yaw,marker='.')
    plt.plot(t,drone_state_history[:-1,5])
    plt.title('Yaw angle').set_fontsize(20)
    plt.xlabel('$t$ [$s$]').set_fontsize(20)
    plt.ylabel('$\psi$ [$rad$]').set_fontsize(20)
    plt.legend(['Planned yaw','Executed yaw'],fontsize = 14)
    plt.show()

def plot_props_speed(t, omega_history):
    plt.plot(t, -omega_history[:-1,0],color='blue')
    plt.plot(t, omega_history[:-1,1],color='red')
    plt.plot(t, -omega_history[:-1,2],color='green')
    plt.plot(t, omega_history[:-1,3],color='black')
    plt.title('Angular velocities').set_fontsize(20)
    plt.xlabel('$t$ [$s$]').set_fontsize(20)
    plt.ylabel('$\omega$ [$rad/s$]').set_fontsize(20)
    plt.legend(['P 1','P 2','P 3','P 4' ],fontsize = 14)
    plt.grid()
    plt.show()

def plot_position_error(t, drone_state_history, desired):
    err_x = np.sqrt((desired.x-drone_state_history[:-1,0])**2)
    err_y = np.sqrt((desired.y-drone_state_history[:-1,1])**2)
    err_z = np.sqrt((desired.z-drone_state_history[:-1,2])**2)

    plt.plot(t, err_x)
    plt.plot(t, err_y)
    plt.plot(t, err_z)
    plt.title('Error in flight position').set_fontsize(20)
    plt.xlabel('$t$ [$s$]').set_fontsize(20)
    plt.ylabel('$e$ [$m$]').set_fontsize(20)
    plt.ylim(0.1, 1)
    plt.legend(['x', 'y', 'z'],fontsize = 14)
    plt.show()

def quad_pos(current_position, rot, L, H=0.05) -> np.ndarray:
    pos = current_position.reshape(-1, 1)  # Convert current position to column vector

    # Create homogeneous transformation from body to world
    wHb = np.hstack((rot, pos))
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

def plot3d_quad(ax, quad, drone_state_history, desired, scalar_map, norm):
    rot_mat = quad.R()
    current_position = np.array([quad.x, quad.y, quad.z])
    quad_plot_coordinates = quad_pos(current_position, rot_mat, L=.7, H=0.005)
    a = quad_plot_coordinates[0, :]
    b = quad_plot_coordinates[1, :]
    c = quad_plot_coordinates[2, :]

    # Text
    roll, pitch, yaw = math.degrees(quad.phi), math.degrees(quad.theta), math.degrees(quad.psi)
    vel = np.linalg.norm([quad.x_vel, quad.y_vel, quad.z_vel])
    text = (
        f"""
        Velocity: {round(vel, 2)} m/s\n
        R: {round(roll, 2)} deg        P: {round(pitch, 2)} deg        Y: {round(yaw, 2)} deg\n
        """
    )
    ax.text2D(0.1, 0.98, text, transform=ax.transAxes, size=12)

    # quad
    ax.plot(a[[0, 2]], b[[0, 2]], c[[0, 2]], 'b-', lw=5, alpha=.2)
    ax.plot(a[[1, 3]], b[[1, 3]], c[[1, 3]], 'b-', lw=5, alpha=.2)
    ax.scatter(a, b, c, alpha=.1)

    # trajectory
    color = scalar_map(norm(vel))

    ax.plot(desired.x, desired.y, desired.z, marker='.',color='red', alpha=.2, markersize=1)
    ax.plot(drone_state_history[-80:, 0],
            drone_state_history[-80:, 1],
            drone_state_history[-80:, 2],
            marker='o',color=color, alpha=1, markersize=6, fillstyle='none')

    # axis
    scale = 0.5
    px, py, pz = current_position
    roll_axis = (rot_mat[:, 0] * scale) + current_position
    pitch_axis = (rot_mat[:, 1] * scale) + current_position
    yaw_axis = -(rot_mat[:, 2] * scale) + current_position

    ax.plot([px, roll_axis[0]], [py, roll_axis[1]], [pz, roll_axis[2]], 'r', lw=3)
    ax.plot([px, pitch_axis[0]], [py, pitch_axis[1]], [pz, pitch_axis[2]], 'g', lw=3)
    ax.plot([px, yaw_axis[0]], [py, yaw_axis[1]], [pz, yaw_axis[2]], 'b', lw=3)
    
def setup_plot(colormap="coolwarm"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    norm = Normalize(vmin=0, vmax=1)
    scalar_map = get_cmap(colormap)
    sm = ScalarMappable(cmap=scalar_map, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Velocity (m/s)')
    return ax, norm, scalar_map

if __name__ == "__main__":

    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config_file = "/home/medhyvinceslas/Documents/programming/quad3d_sim/config.ini"
    config.read(config_file)
    inner_loop_relative_to_outer_loop = 10
    
    ax, norm, scalar_map = setup_plot(colormap="turbo")
    # t, dt, desired = get_path_helix(total_time=20, r=1.5, height=3, dt=0.01)
    t, dt, desired = get_path(total_time=20)

    quad = Quadrotor(config, desired)
    control = Controller(config)
    
    drone_state_history, omega_history = quad.X, quad.omega    
    n_waypoints = desired.z.shape[0]
    
    for i in range(0, 2000):
        i = n_waypoints-1 if i > n_waypoints-1 else i  # stay at the last waypoint if arrived

        thrust_cmd = control.altitude(quad, desired, dt, index=i)
        acc_cmd = control.lateral(quad, desired, index=i)
        
        for _ in range(inner_loop_relative_to_outer_loop):
            
            moment_cmd = control.attitude(quad, thrust_cmd, acc_cmd, desired.yaw[i])
            quad.set_propeller_speed(thrust_cmd, moment_cmd)
            _ = quad.advance_state(dt/inner_loop_relative_to_outer_loop)
       

        drone_state_history = np.vstack((drone_state_history, quad.X))
        omega_history = np.vstack((omega_history, quad.omega))
        ax.clear()
        plot3d_quad(ax, quad, drone_state_history, desired, scalar_map, norm)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 3.5)
        ax.set_zlim(0, 3.5)
        plt.pause(.0001)


    plot_trajectory(ax, drone_state_history, desired)
    plot_yaw_angle(t, desired, drone_state_history)
    plot_props_speed(t, omega_history)
    plot_position_error(t, drone_state_history, desired)
    plt.show()

