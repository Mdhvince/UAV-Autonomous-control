import random
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt


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

def minimum_jerk_trajectory(waypoints, T=30, speed=1.2, dt=0.1):
    """
    Inputs:
        - waypoints: 2D array of start, intermediate and goal 3d waypoints
        - T: Total time of the trajectory
        - speed: desired speed between a segment of WP
        - dt: time steps
    """
    positions = []
    velocities = []
    accelerations = []
    times = []

    Desired = namedtuple(
            "Desired", ["x", "y", "z", "x_vel", "y_vel", "z_vel", "x_acc", "y_acc", "z_acc", "yaw"])

    for i in range(waypoints.shape[0] - 1):
        # the time required to travel between each pair of waypoints
        distance = np.linalg.norm(waypoints[i+1] - waypoints[i])
        time = distance / speed
        times.append(time)

    for i in range(waypoints.shape[0] - 1):
        T = times[i]                               # Total time for the motion between the waypoints

        A = np.array([
            [0, 0, 0, 0, 0, 1],                    # POSITION AT T=0 CONSTRAINT
            [T**5, T**4, T**3, T**2, T, 1],        # POSITION AT T=T CONSTRAINT
            [0, 0, 0, 0, 1, 0],                    # VELOCITY AT T=0 CONSTRAINT
            [5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],   # VELOCITY AT T=T CONSTRAINT
            [0, 0, 0, 2, 0, 0],                    # ACCELERATION AT T=0 CONSTRAINT
            [20*T**3, 12*T**2, 6*T, 2, 0, 0]       # ACCELERATION AT T=T CONSTRAINT
        ])

        x0, y0, z0 = waypoints[i]                  # initial positions
        xT, yT, zT = waypoints[i+1]                # end positions

        # conditions for velocities: 
        # if start: intial velocity is 0 and end velocity is the speed of travel (constant)
        # for any other intermadiate waypoints: start vel = end vel = speed*0.1
        # for the final waypoint, I set velocity to 0        
        is_first_wp = (i == 0)
        is_last_wp = (i+1 == waypoints.shape[0] - 1)

        vx0, vy0, vz0 = [0.0, 0.0, 0.0] if is_first_wp else [speed*0.1, speed*0.1, speed*0.1]
        vxT, vyT, vzT = [0.0, 0.0, 0.0] if is_last_wp else [speed*0.1, speed*0.1, speed*0.1]

        
        conditions = np.array([[x0, y0, z0],   # POSITION X AT T=0 CONSTRAINT
                            [xT, yT, zT],     # POSITION X AT T=T CONSTRAINT
                            [vx0, vy0, vz0],           # VELOCITY X AT T=0 CONSTRAINT
                            [vxT, vyT, vzT],           # VELOCITY X AT T=T CONSTRAINT
                            [0.0, 0.0, 0.0],           # ACCELERATION X AT T=0 CONSTRAINT
                            [0.0, 0.0, 0.0]])          # ACCELERATION X AT T=T CONSTRAINT
        
        # now we have a problem in the for Ax = conditions where x are the unknown coefficents we are
        # looking for. So we can use the inverse of A to solve this:
        # x = A^1 @ conditions
        
        # assuming det(A) is not 0
        COEFFS = np.linalg.solve(A, conditions)

        # now we have the coeffs for the current start and current end, let find all the poses in between
        for t in range(int(T/dt)):
            t = t*dt
            # so the minimum jerk position is:
            position = COEFFS[0] * t**5 + COEFFS[1] * t**4 + COEFFS[2] * t**3 + COEFFS[3] * t**2 + COEFFS[4] * t**1 + COEFFS[5]
            # for the velocity and acceleration, we differenciate
            velocity = 5 * COEFFS[0] * t**4 + 4 * COEFFS[1] * t**3 + 3 * COEFFS[2] * t**2 + COEFFS[3] * t + COEFFS[4]
            acceleration = 4*5 * COEFFS[0] * t**3 + 3*4 * COEFFS[1] * t**2 + 2*3 * COEFFS[2] * t + COEFFS[3]

            positions.append(position)
            velocities.append(velocity)
            accelerations.append(acceleration)

    traj = np.hstack((positions, velocities, accelerations))
    yaw = np.full(len(traj), 0.0)
    desired_trajectory = Desired(
            traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3], traj[:, 4], traj[:, 5], traj[:, 6], traj[:, 7], traj[:, 8], yaw)
    return desired_trajectory


if __name__ == "__main__":
    
    pass
  
    
