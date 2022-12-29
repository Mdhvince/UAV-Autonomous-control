import random
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt

# Toy trajectories
def get_path(total_time=20, dt=0.01):
    Desired = namedtuple(
            "Desired", ["x", "y", "z", "x_vel", "y_vel", "z_vel", "x_acc", "y_acc", "z_acc", "yaw"])
    
    t = np.linspace(0.0, total_time, int(total_time/dt))

    omega_x = 0.8
    omega_y = 0.4
    omega_z = 0.4

    a_x = 1.0 
    a_y = 1.0
    a_z = 1.0

    x = a_x * np.sin(omega_x * t) 
    x_vel = a_x * omega_x * np.cos(omega_x * t)
    x_acc = -a_x * omega_x**2 * np.sin(omega_x * t)

    y = a_y * np.cos(omega_y * t) + 2
    y_vel = -a_y * omega_y * np.sin(omega_y * t)
    y_acc = -a_y * omega_y**2 * np.cos(omega_y * t)

    z = a_z * np.cos(omega_z * t) + 2
    z_vel = -a_z * omega_z * np.sin(omega_z * t)
    z_acc = - a_z * omega_z**2 * np.cos(omega_z * t)

    yaw = np.arctan2(y_vel, x_vel)

    desired_trajectory = Desired(x, y, z, x_vel, y_vel, z_vel, x_acc, y_acc, z_acc, yaw)

    return t, dt, desired_trajectory

def get_path_helix(total_time=20, r=1, height=10, dt=0.01):
    Desired = namedtuple(
            "Desired", ["x", "y", "z", "x_vel", "y_vel", "z_vel", "x_acc", "y_acc", "z_acc", "yaw"])
    
    t = np.linspace(0.0, total_time, int(total_time/dt))

    omega_x = 0.8
    omega_y = 0.4
    omega_z = 0.4

    a_x = r  # radius of helix
    a_y = r  # radius of helix
    a_z = height / total_time  # vertical movement of helix per unit time

    x = a_x * np.sin(omega_x * t) 
    x_vel = a_x * omega_x * np.cos(omega_x * t)
    x_acc = -a_x * omega_x**2 * np.sin(omega_x * t)

    y = a_y * np.cos(omega_y * t)
    y_vel = -a_y * omega_y * np.sin(omega_y * t)
    y_acc = -a_y * omega_y**2 * np.cos(omega_y * t)

    z = a_z * t  # z moves linearly with time
    z_vel = np.full(len(t), a_z)  # z_vel is constant
    z_acc = np.full(len(t), 0)  # z_acc is zero

    yaw = np.arctan2(y_vel,x_vel)

    desired_trajectory = Desired(x, y, z, x_vel, y_vel, z_vel, x_acc, y_acc, z_acc, yaw)

    return t, dt, desired_trajectory

# Optimal trajectories
def get_time_between_segments(waypoints, speed):
    times = []
    for i in range(waypoints.shape[0] - 1):
        # the time required to travel between each pair of waypoints
        distance = np.linalg.norm(waypoints[i+1] - waypoints[i])
        time = distance / speed
        times.append(time)

    return times

def get_polynomial_matrix(T, mode):
    if mode == "jerk":
        A = np.array([
            [0, 0, 0, 0, 0, 1],                                 # POSITION AT T=0 CONSTRAINT
            [T**5, T**4, T**3, T**2, T, 1],                     # POSITION AT T=T CONSTRAINT
            [0, 0, 0, 0, 1, 0],                                 # VELOCITY AT T=0 CONSTRAINT
            [5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],                # VELOCITY AT T=T CONSTRAINT
            [0, 0, 0, 2, 0, 0],                                 # ACCELERATION AT T=0 CONSTRAINT
            [20*T**3, 12*T**2, 6*T, 2, 0, 0]                    # ACCELERATION AT T=T CONSTRAINT
        ])
    else: # snap
        A = np.array([
            [0, 0, 0, 0, 0, 0, 0, 1],
            [T**7, T**6, T**5, T**4, T**3, T**2, T, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [7*T**6, 6*T**5, 5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],
            [0, 0, 0, 0, 0, 2, 0, 0],
            [42*T**5, 30*T**4, 20*T**3, 12*T**2, 6*T, 2, 0, 0],
            [0, 0, 0, 0, 6, 0, 0, 0],                                   # JERK AT T=0 CONSTRAINT
            [210*T**4, 120*T**3, 60*T**2, 24*T, 6, 0, 0, 0]             # JERK AT T=T CONSTRAINT
        ])
    
    return A

def get_boundary_conditions_matrix(mode, start_conditions, end_conditions):
    if mode == "jerk":
        x0, y0, z0, vx0, vy0, vz0, ax0, ay0, az0 = start_conditions
        xT, yT, zT, vxT, vyT, vzT, axT, ayT, azT = end_conditions
        conditions = np.array([[x0, y0, z0],                  # POSITION X AT T=0 CONSTRAINT
                               [xT, yT, zT],                  # POSITION X AT T=T CONSTRAINT
                                [vx0, vy0, vz0],               # VELOCITY X AT T=0 CONSTRAINT
                                [vxT, vyT, vzT],               # VELOCITY X AT T=T CONSTRAINT
                                [ax0, ay0, az0],               # ACCELERATION X AT T=0 CONSTRAINT
                                [axT, ayT, azT]])              # ACCELERATION X AT T=T CONSTRAINT
    else:
        x0, y0, z0, vx0, vy0, vz0, ax0, ay0, az0, jx0, jy0, jz0 = start_conditions
        xT, yT, zT, vxT, vyT, vzT, axT, ayT, azT, jxT, jyT, jzT = end_conditions
        conditions = np.array([[x0, y0, z0],                      
                                [xT, yT, zT],                      
                                [vx0, vy0, vz0],                   
                                [vxT, vyT, vzT],                  
                                [ax0, ay0, az0],              
                                [axT, ayT, azT],
                                [jx0, jy0, jz0],               # START WITH THIS JERK         
                                [jxT, jyT, jzT]])              # END WITH THIS JERK
    return conditions


def optimal_trajectory(waypoints, speed=1.2, speed_at_wp=.1, dt=.1, mode="jerk"):
    """
    Inputs:
        - waypoints: 2D array of start, intermediate and goal 3d waypoints
        - speed: desired speed between a segment of WP
        - dt: time steps
    """
    if mode != "jerk" and mode != "snap":
        raise ValueError(f"Invalid mode argument, expected 'jerk' or 'snap', got {mode}")
    
    positions, velocities, accelerations, jerks, snaps = [], [], [], [], []
    Desired = namedtuple(
            "Desired", ["x", "y", "z", "x_vel", "y_vel", "z_vel", "x_acc", "y_acc", "z_acc", "yaw"])
    

    times = get_time_between_segments(waypoints, speed)   # 1D array of time to complete a particular segement
    S = np.array([0] + np.cumsum(times).tolist())         # 1D array of time it takes to reach a waypoint from the very first waypoint to the current waypoint
    
    for i in range(waypoints.shape[0] - 1):
        Ti = times[i]   # time to complete the segment
        Si = S[i]       # time to reach the waypoint cummulated from the very first wp

        T = times[i]

        # if t > S[-1]:
        #     t = S[-1] - 0.01
        # scale = (t - Si) / Ti

        # T = scale
        # raise



        # Get boundary conditions
        is_first_seg = (i == 0)
        is_last_seg = (i+1 == waypoints.shape[0] - 1)


        x0, y0, z0 = waypoints[i]
        xT, yT, zT = waypoints[i+1]
        vx0, vy0, vz0 = [0.0, 0.0, 0.0] if is_first_seg else [speed_at_wp, speed_at_wp, 0.0]
        vxT, vyT, vzT = [0.0, 0.0, 0.0] if is_last_seg else [speed_at_wp, speed_at_wp, 0.0]
        
        ax0, ay0, az0, axT, ayT, azT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        jx0, jy0, jz0, jxT, jyT, jzT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        start_conditions = [x0, y0, z0, vx0, vy0, vz0, ax0, ay0, az0]
        end_conditions = [xT, yT, zT, vxT, vyT, vzT, axT, ayT, azT]

        if mode == "snap":
            start_conditions.extend([jx0, jy0, jz0])
            end_conditions.extend([jxT, jyT, jzT])

        b = get_boundary_conditions_matrix(mode, start_conditions, end_conditions)
        A = get_polynomial_matrix(T, mode)
        
        # now we have a problem in the for Ac = conditions where c are the unknown coefficents
        # (spline parameters) we are looking for. So we can use the inverse of A to solve this:
        # c = A^1 @ conditions, assuming det(A) is not 0
        
        c = np.linalg.solve(A, b)  # spline parameters


        # now we have the coeffs for the current start and current end, let find all the poses in between
        if mode == "jerk":
            c5, c4, c3, c2, c1, c0 = c[0], c[1], c[2], c[3], c[4], c[5]
            for t in np.arange(0.0, T, dt):
                position = c5 * t**5 + c4 * t**4 + c3 * t**3 + c2 * t**2 + c1 * t**1 + c0
                velocity = 5 * c5 * t**4 + 4 * c4 * t**3 + 3 * c3 * t**2 + c2 * t + c1       # dp/dt
                acceleration = 20 * c5 * t**3 + 12 * c4 * t**2 + 6 * c3 * t + c2             # dv/dt
                jerk = 60 * c5 * t**2 + 24 * c4 * t + 6 * c3                                 # da/dt

                positions.append(position)
                velocities.append(velocity)
                accelerations.append(acceleration)
                jerks.append(jerk)
        else:
            c7, c6, c5, c4, c3, c2, c1, c0 = c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]
            for t in np.arange(0.0, T, dt):
                position = c7 * t**7 + c6 * t**6 + c5 * t**5 + c4 * t**4 + c3 * t**3 + c2 * t**2 + c1 * t**1 + c0
                velocity = 7 * c7 * t**6 + 6 * c6 * t**5 + 5 * c5 * t**4 + 4 * c4 * t**3 + 3 * c3 * t**2 + 2 * c2 * t + c1
                acceleration = 42 * c7 * t**5 + 30 * c6 * t**4 + 20 * c5 * t**3 + 12 * c4 * t**2 + 6 * c3 * t + 2 * c2
                jerk = 210 * c7  *t**4 + 120 * c6 * t**3 + 60 * c5 * t**2 + 24 * c4 * t + 6  *c3
                snap = 840 * c7 * t**3 + 360 * c6 * t**2 + 180 * c5 * t + 24 * c4

                positions.append(position)
                velocities.append(velocity)
                accelerations.append(acceleration)
                jerks.append(jerk)

    traj = np.hstack((positions, velocities, accelerations))
    yaw = np.full(len(traj), 0.0)
    desired_trajectory = Desired(
            traj[:, 0], traj[:, 1], traj[:, 2],
            traj[:, 3], traj[:, 4], traj[:, 5],
            traj[:, 6], traj[:, 7], traj[:, 8], yaw)
    
    return desired_trajectory, jerks

# WP
def getwp(form, a=None, phi=None):
    if form == 'angle' and a is None and phi is None:
        w = np.array([[0, 0, 0],
                      [0, 0, 2],
                      [0, 4, 2],
                      [0, 0, 1.1]]).T
    elif form == 'helix' and a is None and phi is None:
        r = 5
        h_max = 16.5
        t = np.pi * np.arange(0, h_max + 0.4, 0.4)
        x = r * np.cos(t)
        y = r * np.sin(t)
        z = t / np.pi
        w = np.array([x, y, z])
    elif form == 'maneuvre' and a is not None and phi is not None:
        w = np.array([[0, 0, 2],
                      [0, a, 2],
                      [a * np.sin(phi), a * (1 - np.cos(phi)), 2]]).T
    return w


if __name__ == "__main__":
    
    print(getwp("helix").T)
  
    
