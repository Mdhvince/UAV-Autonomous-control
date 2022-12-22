import configparser

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from quadrotor import Quadrotor
from controller import Controller
from trajectory import get_path, get_path_random, get_path_straight


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

def plot_angular_velocities(t, omega_history):
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

if __name__ == "__main__":

    config = configparser.ConfigParser()
    config_file = "/home/medhyvinceslas/Documents/programming/quad3d_sim/config.ini"
    config.read(config_file)
    inner_loop_relative_to_outer_loop = 10
    
    quad = Quadrotor(config)
    control_system = Controller(config)
    t, dt, desired = get_path(20)

    
    
    quad.X = np.array([
        desired.x[0], desired.y[0], desired.z[0],
        0.0, 0.0, 0.0,
        desired.x_vel[0], desired.y_vel[0], desired.z_vel[0],
        0.0, 0.0, 0.0
    ])
    
    drone_state_history = quad.X
    omega_history = quad.omega
    accelerations = quad.linear_acceleration()
    accelerations_history= accelerations
    angular_vel_history = quad.get_euler_derivatives()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n_waypoints = desired.z.shape[0]
    R = []


    for i in range(0, n_waypoints):
        rot_mat = quad.R()

        thrust_cmd = control_system.altitude_controller(
                desired.z[i], desired.z_vel[i], desired.z_acc[i], rot_mat, quad, dt)
        
        # reserve some thrust margin for angle control
        thrust_margin = 0.1 * (quad.max_thrust - quad.min_thrust)
        thrust_cmd = np.clip(thrust_cmd, (quad.min_thrust + thrust_margin) * 4, (quad.max_thrust-thrust_margin) * 4)

        acc_cmd = control_system.lateral_controller(
            desired.x[i], desired.x_vel[i], desired.x_acc[i],
            desired.y[i], desired.y_vel[i], desired.y_acc[i], quad)
        
        for _ in range(inner_loop_relative_to_outer_loop):
            rot_mat = quad.R()
            pq_cmd = control_system.roll_pitch_controller(acc_cmd, thrust_cmd, rot_mat, quad)
            r_cmd = control_system.yaw_controller(desired.yaw[i], quad)
            pqr_cmd = np.append(pq_cmd, r_cmd)
            moment_cmd = control_system.body_rate_controller(pqr_cmd, quad)

            quad.set_propeller_angular_velocities(thrust_cmd, moment_cmd)
            _ = quad.advance_state(dt/inner_loop_relative_to_outer_loop)
       
        current_position = np.array([quad.x, quad.y, quad.z])
        quad_plot_coordinates = quad_pos(current_position, rot_mat, L=.7, H=0.005)
        a = quad_plot_coordinates[0, :]
        b = quad_plot_coordinates[1, :]
        c = quad_plot_coordinates[2, :]

        ax.clear()
        vel = np.linalg.norm([quad.x_vel, quad.y_vel, quad.z_vel])
        text = f"Velocity: {round(vel, 2)} m/s"
        ax.text2D(0.1, 1, text, transform=ax.transAxes, size=15)
        ax.plot(desired.x, desired.y, desired.z, marker='.',color='red', alpha=.2, markersize=1)
        ax.plot(a[[0, 2]], b[[0, 2]], c[[0, 2]], 'b-', lw=5, alpha=.4)
        ax.plot(a[[1, 3]], b[[1, 3]], c[[1, 3]], 'b-', lw=5, alpha=.4)
        ax.scatter(a, b, c)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 3.5)
        ax.set_zlim(0, 3.5)

        plt.pause(.0001)

     
        drone_state_history = np.vstack((drone_state_history, quad.X))
        omega_history = np.vstack((omega_history, quad.omega))
        accelerations = quad.linear_acceleration()
        accelerations_history= np.vstack((accelerations_history, accelerations))
        angular_vel_history = np.vstack((angular_vel_history, quad.get_euler_derivatives()))




    plot_trajectory(ax, drone_state_history, desired)
    plot_yaw_angle(t, desired, drone_state_history)
    plot_angular_velocities(t, omega_history)
    plot_position_error(t, drone_state_history, desired)
    plt.show()

