import configparser
import roblib
import numpy as np
import matplotlib.pyplot as plt

from quadrotor import Quadrotor
from controller import Controller
from trajectory import get_path, get_path_random, get_path_straight


def animate_trajectory(ax, quad, R, desired_traj):
    ax.clear()    
    ax.plot(desired_traj.x, desired_traj.y, desired_traj.z, linestyle='-', marker='.',color='red', alpha=0.5, markersize=2)
    st = np.copy(quad.X).reshape(-1, 1)
    roblib.draw_quadrotor3D(ax, st, np.array([[0, 0, 0, 0]]).T, l=.4)
    ax.set_xlabel('$x$ [$m$]').set_fontsize(12)
    ax.set_ylabel('$y$ [$m$]').set_fontsize(12)
    ax.set_zlabel('$z$ [$m$]').set_fontsize(12)
    plt.legend(['Roll-axis', 'Pitch-axis', 'Yaw-axis', 'Planned path'],fontsize=10)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 3.5)
    ax.set_zlim(0, 3.5)
    plt.pause(.001)

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
    err= np.sqrt((desired.x-drone_state_history[:-1,0])**2 
             +(desired.y-drone_state_history[:-1,1])**2 
             +(desired.z-drone_state_history[:-1,2])**2)

    plt.plot(t,err)
    plt.title('Error in flight position').set_fontsize(20)
    plt.xlabel('$t$ [$s$]').set_fontsize(20)
    plt.ylabel('$e$ [$m$]').set_fontsize(20)
    plt.show()


if __name__ == "__main__":

    config = configparser.ConfigParser()
    config_file = "/home/medhyvinceslas/Documents/programming/quad3d_sim/config.ini"
    config.read(config_file)
    
    inner_loop_relative_to_outer_loop = 10
    t, dt, desired = get_path()

    quad = Quadrotor(config)
    control_system = Controller(config)
    
    
    # declaring the initial state
    quad.X = np.array([
        desired.x[0],
        desired.y[0],
        desired.z[0],
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

        # animate_trajectory(ax, quad, rot_mat, desired)
    
        drone_state_history = np.vstack((drone_state_history, quad.X))
        omega_history = np.vstack((omega_history, quad.omega))
        accelerations = quad.linear_acceleration()
        accelerations_history= np.vstack((accelerations_history, accelerations))
        angular_vel_history = np.vstack((angular_vel_history, quad.get_euler_derivatives()))

    plot_trajectory(ax, drone_state_history, desired)
    plot_yaw_angle(t, desired, drone_state_history)
    plot_angular_velocities(t, omega_history)
    plot_position_error(t, drone_state_history, desired)

