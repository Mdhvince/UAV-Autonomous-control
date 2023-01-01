import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.animation as animation

plt.style.use('seaborn-paper')


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

def quad_pos(current_position, rot, L, H=.05) -> np.ndarray:
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

def plot3d_quad(ax, obstacle_coords, obstacle_shapes, obstacle_boundary, waypoints, drone_state_history, desired, scalar_map, norm, index):
    ax.clear()
    phi, theta, psi = -drone_state_history[index, 3:6]  # add negative for mirroring effect of matplotlib
    roll, pitch, yaw = math.degrees(phi), math.degrees(theta), math.degrees(psi)
    vel = np.linalg.norm(drone_state_history[index, 6:9])
    text = (
        f"""
        Velocity   {round(vel, 2)} m/s

        Roll       {round(roll, 2)}°
        Pitch      {round(pitch, 2)}°
        Yaw        {round(yaw, 2)}°
        """
    )
    row, col = 0, 0
    ax.text2D(col, row, text, transform=ax.transAxes, size=12)

    # Quad 
    rot_mat = euler2Rot(phi, theta, psi)

    current_position = drone_state_history[index, :3]
    quad_wf = quad_pos(current_position, rot_mat, L=.7, H=0.005)
    x, y, z = quad_wf[0, :], quad_wf[1, :], quad_wf[2, :]

    color = scalar_map(norm(vel))
    ax.scatter(x, y, z, color=color, alpha=1)

    COM, FL, FR, RL, RR = 5, 0, 1, 3, 2

    # COM to front left / rigth / rear-left / rear-right
    ax.plot([x[COM], x[FL]], [y[COM], y[FL]], [z[COM], z[FL]], '-', color='y', label="Front")
    ax.plot([x[FR], x[COM]], [y[FR], y[COM]], [z[FR], z[COM]], '-', color='y')
    ax.plot([x[COM], x[RL]], [y[COM], y[RL]], [z[COM], z[RL]], '-', color='w', label="Rear")
    ax.plot([x[COM], x[RR]], [y[COM], y[RR]], [z[COM], z[RR]], '-', color='w')

    # contour of the quad
    # ax.plot([x[FL], x[FR], x[RR], x[RL], x[FL]],
    #         [y[FL], y[FR], y[RR], y[RL], y[FL]],
    #         [z[FL], z[FR], z[RR], z[RL], z[FL]], '-')
    
    # shadow
    px, py, pz = current_position
    ax.plot(px, py, 0, color='black', alpha=0.5, markersize=5-pz, marker='o')
    
    # Trajectory 
    ax.plot(desired.x,
            desired.y,
            desired.z, marker='.',color='red', alpha=.2, markersize=1, label="Trajectory")

    # Axis Body frame 
    onboard_axis = False
    
    if onboard_axis:
        scale = 0.3
        roll_axis = (rot_mat[:, 0] * scale) + current_position
        pitch_axis = (rot_mat[:, 1] * scale) + current_position
        yaw_axis = (rot_mat[:, 2] * scale) + current_position
        ax.plot([px, roll_axis[0]], [py, roll_axis[1]], [pz, roll_axis[2]], 'r', lw=3)
        ax.plot([px, pitch_axis[0]], [py, pitch_axis[1]], [pz, pitch_axis[2]], 'g', lw=3)
        ax.plot([px, yaw_axis[0]], [py, yaw_axis[1]], [pz, yaw_axis[2]], 'b', lw=3)
    else:
        # Define the starting and ending points of the axes
        scale = 1.5
        roll_axis = rot_mat[:, 0] * scale
        pitch_axis = rot_mat[:, 1] * scale
        yaw_axis = rot_mat[:, 2] * scale

        x0, y0, z0 = [3, -3, -3]
        ax.quiver(x0, y0, z0, roll_axis[0], roll_axis[1], roll_axis[2], color='r', lw=2)
        ax.quiver(x0, y0, z0, pitch_axis[0], pitch_axis[1], pitch_axis[2], color='g', lw=2)
        ax.quiver(x0, y0, z0, yaw_axis[0], yaw_axis[1], yaw_axis[2], color='b', lw=2)
    
    # # plot intermediate waypoints and obstacles
    # for i in range(len(obstacle_coords)):
    #     obstacle = get_obstacle(obstacle_coords[i], obstacle_shapes[i], obstacle_boundary)
    #     ax.voxels(obstacle)
    
    n_wp = len(waypoints)
    for i in range(n_wp):
        x, y, z = waypoints[i]
        ax.plot(x, y, z, alpha=.5, marker=".",  markersize=40)


    ax.w_zaxis.pane.set_color('#2D3047')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    ax.legend(facecolor="gray", bbox_to_anchor=(1, 1), loc='upper left')
    # ax.set_xlim(0, 11)
    # ax.set_ylim(0, 11)
    # ax.set_zlim(0, 11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.line.set_color('black')
    ax.yaxis.line.set_color('black')
    ax.zaxis.line.set_color('black')

def animate(index, ax, obstacle_coords, obstacle_shapes, obstacle_boundary, waypoints, drone_state_history, desired, scalar_map, norm):
        plot3d_quad(ax, obstacle_coords, obstacle_shapes, obstacle_boundary, waypoints, drone_state_history, desired, scalar_map, norm, index)

def run_animation(fig, frames, interval, *args):
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, fargs=(args))
    return ani

def setup_plot(colormap="coolwarm"):
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # ax.view_init(elev=0, azim=45)

    # fig.set_facecolor('#2D3047')
    norm = Normalize(vmin=0, vmax=1)
    scalar_map = get_cmap(colormap)
    sm = ScalarMappable(cmap=scalar_map, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, location="bottom", shrink=0.5)
    cbar.set_label('Velocity (m/s)')

    return fig, ax, norm, scalar_map

def save_animation(ani, filepath):
    writer=animation.FFMpegFileWriter(fps=60)
    ani.save(filepath, writer=writer)

def get_obstacle(coordinates, shapes, limits):
    x_size, y_size, z_size = limits
    x, y, z = coordinates
    width, length, height = shapes

    voxel = np.zeros((x_size, y_size, z_size))
    obstacle = np.ones((width, length, height))
    voxel[x:x+width, y:y+length, z:z+height] = obstacle
    return voxel

def plot_intermediate_waypoints(ax, waypoints, n_waypoints, index):
    label = None
    color = "black"
    x, y, z = waypoints
    if index == 0:
        label = "start"; color = "blue"
    elif index == 1:
        label = "waypoints"
    elif index == n_waypoints-1:
        label = "goal"; color = "red"

    ax.plot(x, y, z, alpha=.5, marker=".",  markersize=20, color=color, label=label)

def calculate_time_between_waypoints(wp1, wp2, speed=1.0):
  x1, y1, z1 = wp1
  x2, y2, z2 = wp2
  
  distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
  
  # Calculate the time required to travel between the two waypoints
  time = distance / speed
  return time

if __name__ == "__main__":
    pass
    

    