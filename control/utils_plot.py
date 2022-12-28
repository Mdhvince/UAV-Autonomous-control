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

def plot3d_quad(ax, drone_state_history, desired, scalar_map, norm, index):
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

        x_start, y_start, z_start = [3, -3, -3]
        ax.quiver(x_start, y_start, z_start, roll_axis[0], roll_axis[1], roll_axis[2], color='r', lw=2)
        ax.quiver(x_start, y_start, z_start, pitch_axis[0], pitch_axis[1], pitch_axis[2], color='g', lw=2)
        ax.quiver(x_start, y_start, z_start, yaw_axis[0], yaw_axis[1], yaw_axis[2], color='b', lw=2)

    ax.w_zaxis.pane.set_color('#2D3047')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    ax.legend(facecolor="gray", bbox_to_anchor=(1, 1), loc='upper left')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.line.set_color('black')
    ax.yaxis.line.set_color('black')
    ax.zaxis.line.set_color('black')

def animate(index, ax, drone_state_history, desired, scalar_map, norm):
        plot3d_quad(ax, drone_state_history, desired, scalar_map, norm, index)

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

def create_obstacle(ax, coordinates, shapes, limits):
    # Define the dimensions of the voxel plot OVERALL size
    x_size, y_size, z_size = limits
    x, y, z = coordinates
    width, length, height = shapes

    # Create an empty 3D array of voxels
    voxel_plot = np.zeros((x_size, y_size, z_size))

    # Create an array of ones to represent the voxels inside the obstacle
    obstacle = np.ones((width, length, height))

    # Set the value of the voxels inside the obstacle to 1
    voxel_plot[x:x+width, y:y+length, z:z+height] = obstacle
    ax.voxels(voxel_plot)

def create_waypoints(ax, waypoints, n_waypoints):
    label = None
    color = "black"
    x, y, z = waypoints
    if i == 0:
        label = "start"; color = "blue"
    elif i == 1:
        label = "waypoints"
    elif i == n_waypoints-1:
        label = "goal"; color = "red"

    ax.plot(x, y, z, alpha=.5, marker=".",  markersize=20, color=color, label=label)

if __name__ == "__main__":
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    limits = (10, 10, 10)

    coordinates = np.array([
        [8, 2, 0],
        [4, 6, 0],
        [7, 5, 0],
        [4, 0, 0]
    ])
    shapes = np.array([
        [1, 1, 5],
        [2, 1, 5],
        [1, 1, 5],
        [1, 4, 5]
    ])
    waypoints = np.array([
        [10, 0, 0],
        [9, 4, 1],
        [6, 5, 1.5],
        [7, 8, 1.5],
        [2, 7, 2],
        [1, 0, 2]
    ])
    
    for i in range(len(coordinates)):
        create_obstacle(ax, coordinates[i], shapes[i], limits)
    
    n_wp = len(waypoints)
    for i in range(n_wp):
        create_waypoints(ax, waypoints[i], n_wp)
    
    # these waypoints are ok, but the quad controller is a 4th order system, so we need a
    # differentiable path that connect the waypoints otherwise the controller wont't be able to
    # minimize errors.
    # So we need to specify trajectory that can be differentiated at least 4 times.
    # And this motivates the use of Minimum Snap trajectory

    # Here is what we need for Minimum Jerk trajectory first:
    # 1. Design a trajectory x(t) such that :
    #   - x(0) = pos_a and x(T) = pos_b
    #   - x(0) = vel_a = 0 and x(T) = vel_b = 0
    #   - x(0) = acc_a = 0 and x(T) = acc_b = 0

    # in plain word: Generate a trajectory that start with pos_a, vel_a, acc_a and ends with pos_b,
    # vel_b, acc_b. (6 constraints in total)

    # so the trajectory at time t will look like this:
    # x(t) = c5*t^5 + c4*t^4 + c3*t^3 + c2*t^2 + c1*t^1 + c0*t^0  => to respect position constraint

    # what we are intersted in is to find the coefficient c0, c1, c2, c3, c4, c5 that satisfy all
    # the constraints (boundary conditions) mentionned above
    # note: that if I have another constraint to respect, I will have to find one more coeff c6*t^6.

    # each of the conditions gives an equation, so we can represent them in a matrix.
    # we can write the equation in terms of unknown constant and boundary conditions. Solving for
    # these constants (coeffs) are a linear problem

    # To respect the position constraint: 
    # x(t) = c5*t^5 + c4*t^4 + c3*t^3 + c2*t^2 + c1*t^1 + c0*t^0

    # So we must have 
    #   x(0) = c0 = a
    #   x(T) = c5*(T^5) + c4*(T^4) + c3*(T^3) + c2*(T^2) + c1*(T^1) + c0*(T^0) = b

    # in matrix form, at t=0 we must have:
    #               |c5|
    #               |c4|
    # [0 0 0 0 0 1] |c3| = a
    #               |c2|
    #               |c1|
    #               |c0|

    # in matrix form, at t=T we must have:
    #                           |c5|
    #                           |c4|
    # [T^5 T^4 T^3 T^2 T^1 T^0] |c3| = b
    #                           |c2|
    #                           |c1|
    #                           |c0|

    # to find the equation for the velocity, we just have to defferentiate the position equation
    # x_dot(t) = 5*c5*t^4 + 4*c4*t^3 + 3*c3*t^2 + 2*c2*t^1 + c1 + 0
    # x_dot(0) = c1 = vel_a
    # x_dot(T) = 5*c5*(T^4) + 4*c4*(T^3) + 3*c3*(T^2) + 2*c2*(T^1) + c1 + 0

    # in matrix form, at t=0 we must have
    #               |c5|
    #               |c4|
    # [0 0 0 0 1 0] |c3| = vel_a
    #               |c2|
    #               |c1|
    #               |c0|

    # in matrix form, at t=T we must have
    #                               |c5|
    #                               |c4|
    # [5T^4 4T^3 3T^2 2T^1 T^0 0]   |c3| = vel_b
    #                               |c2|
    #                               |c1|
    #                               |c0|

    # same for accelerations ... we differentiate and we compute
    # x_dot_dot = 20*c5*t^3 + 12*c4*t^2 + 6*c3*t^2 + 2*c2*t^0 + 0 + 0 

    # all of the 6 constraint can be written as a 6x6 matrix in order to find the coefficient

    positions = []
    T = 20
    num_steps = 1000

    for i in range(waypoints.shape[0] - 1):
        A = np.array([
            [0, 0, 0, 0, 0, 1],                      # POSITION AT T=0 CONSTRAINT
            [T**5, T**4, T**3, T**2, T, 1],          # POSITION AT T=T CONSTRAINT
            [0, 0, 0, 0, 1, 0],                      # VELOCITY AT T=0 CONSTRAINT
            [5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],     # VELOCITY AT T=T CONSTRAINT
            [0, 0, 0, 2, 0, 0],                      # ACCELERATION AT T=0 CONSTRAINT
            [20*T**3, 12*T**2, 6*T, 2, 0, 0]         # ACCELERATION AT T=T CONSTRAINT
        ])
        x_start, y_start, z_start = waypoints[i][0:3]
        x_end, y_end, z_end = waypoints[i+1][0:3]
        
        conditions = np.array([[x_start, y_start, z_start],   # POSITION X AT T=0 CONSTRAINT
                            [x_end, y_end, z_end],     # POSITION X AT T=T CONSTRAINT
                            [0.0, 0.0, 0.0],           # VELOCITY X AT T=0 CONSTRAINT
                            [0.0, 0.0, 0.0],           # VELOCITY X AT T=T CONSTRAINT
                            [0.0, 0.0, 0.0],           # ACCELERATION X AT T=0 CONSTRAINT
                            [0.0, 0.0, 0.0]])          # ACCELERATION X AT T=T CONSTRAINT
        
        # now we have a problem in the for Ax = conditions where x are the unknown coefficents we are
        # looking for. So we can use the inverse of A to solve this:
        # x = A^1 @ conditions
        
        # assuming det(A) is not 0
        COEFFS = np.linalg.inv(A) @ conditions

        # now we have the coeffs for the current start and current end, let find all the poses in between
        time_steps = np.linspace(0, T, num_steps)
        for n, T in enumerate(time_steps):
            # so the minimum jerk position at time T is:
            position = COEFFS[0] * T**5 + COEFFS[1] * T**4 + COEFFS[2] * T**3 + COEFFS[3] * T**2 + COEFFS[4] * T**1 + COEFFS[5]
            # for the velocity and acceleration, we differenciate
            velocity = 5 * COEFFS[0] * T**4 + 4 * COEFFS[1] * T**3 + 3 * COEFFS[2] * T**2 + COEFFS[3] * T + COEFFS[4]
            acceleration = 4*5 * COEFFS[0] * T**3 + 3*4 * COEFFS[1] * T**2 + 2*3 * COEFFS[2] * T + COEFFS[3]

            positions.append(position)


    dt = T/num_steps
    print(dt)

    positions = np.array(positions)

    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
    ax.legend(facecolor="gray")
    plt.show()
