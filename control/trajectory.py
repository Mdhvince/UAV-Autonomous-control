import random
import warnings
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')

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

def optimal_trajectory(waypoints, speed=1.2, dt=.1, mode="jerk", dim=0):
    positions, velocities, accelerations, jerks, snaps = [], [], [], [], []
    
    waypoints = waypoints[:, dim]                                     # solving only for 1 dimension
    times = get_time_between_segments(waypoints, speed)               # 1D array of time to complete a particular segement
    S = np.array([0] + np.cumsum(times).tolist())                     # 1D array of time it takes to reach a waypoint from the very first waypoint to the current waypoint
    
    nb_splines = waypoints.shape[0] - 1
    ord = 7 if mode == "snap" else 5
    nb_constrain = ord + 1

    A = np.zeros((nb_constrain*nb_splines, nb_constrain*nb_splines))
    b = np.zeros((1, nb_constrain*nb_splines))

    print(f"\nGenerating a {ord}th order polynomial trajectory for {nb_splines} splines segments and {nb_constrain} per spline")
    print(f"Matrix A: {A.shape} , Matrix b : {b.shape}\n")

    for i in range(waypoints.shape[0] - 1):
        Ti = times[i]   # time to complete the segment
        Si = S[i]       # time to reach the waypoint cummulated from the very first wp
        T = times[i]

        # Get boundary conditions
        is_first_seg = (i == 0)
        is_last_seg = (i+1 == waypoints.shape[0] - 1)
        speed_at_wp = np.linalg.norm(waypoints[i] - waypoints[i+1]) / T

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
    return traj

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

def polynom(n, k, t):
    T = np.zeros(n)
    D = np.zeros(n)

    for i in range(n):
        D[i] = i 
        T[i] = 1

    # Derivative: (if k=0, this is not executed)
    for j in range(k):
        for i in range(n):
            
            T[i] = T[i] * D[i]
            if D[i] > 0:
                D[i] = D[i] - 1

    # put t value
    for i in range(n):
        T[i] = T[i] * t**D[i]

    return T.T


if __name__ == "__main__":

    
    waypoints = np.array([
        [10, 0, 0], [10, 4, 1], [6, 5, 1.5], [7, 8, 1.5], [2, 7, 2], [1, 0, 2],
        [10, 0, 0], [10, 4, 1], [6, 5, 1.5], [7, 8, 1.5], [2, 7, 2], [1, 0, 2]
    ])
    speed = 1.0
    # waypoints = getwp("helix").T
    times = get_time_between_segments(waypoints, speed=speed)
    S = np.array([0] + np.cumsum(times).tolist())

    dt = 0.02
    time0 = 0.0
    k0 = 0  # position
    k1 = 1  # 1st derivative (velocity)
    k2 = 2  # 2nd derivative and so on...
    k3 = 3
    k4 = 4
    k5 = 5
    k6 = 6

    nb_splines = waypoints.shape[0] - 1
    ord = 7
    nb_constrain = ord + 1

    A = np.zeros((nb_constrain*nb_splines, nb_constrain*nb_splines))
    b = np.zeros((nb_constrain*nb_splines, 3))

    positions = []
    velocities = []
    accelerations = []
    jerks = []

    
    # nb_constrain on position at t=0 - FOR ALL SEGMENTS
    row = 0
    poly = polynom(n=nb_constrain, k=k0, t=time0)                                                      # fill A with polynomial for position at time 0
    for i in range(nb_splines):
        wp0 = waypoints[i]
        A[row, i*nb_constrain:nb_constrain*(i+1)] = poly                                               # i=0 -> 0:8   i=1 -> 8:16   i=2 -> 16:24  ...
        b[row, :] = wp0 # required position at every start of a segment

        # print(f"{len(A[row, i*nb_constrain:nb_constrain*(i+1)].flatten())} values of A about to be replaced")
        # print(f"Filling A with values {poly} at row {row}, from col {i*nb_constrain} : {nb_constrain*(i+1)} for k={k0} and t={time0}")
        # print(f"Filling b with values {wp0} at row {row}\n")

        row += 1
    
    # nb_constrain on position at t=T - FOR ALL SEGMENTS                                                     
    for i in range(nb_splines):
        wpT = waypoints[i+1]
        timeT = times[i]
        poly = polynom(n=nb_constrain, k=k0, t=timeT)                                                  # fill A with polynomial for position at time t (from the 4th row)
        A[row, i*nb_constrain:nb_constrain*(i+1)] = poly                                               # i=0 -> 0:8   i=1 -> 8:16   i=2 -> 16:24  ...
        b[row, :] = wpT  # required position at every end of a segment

        row += 1

    # CONSTRAIN FOR THE VERY FIRST SEGMENT at t=0 (vel=0, acc=0, jerk=0) : 3 in total
    for i, k in enumerate([k1, k2, k3]):
        poly = polynom(n=nb_constrain, k=k, t=time0)

        A[row, 0:nb_constrain] = poly
        row += 1
    

    # CONSTRAIN FOR THE VERY LAST SEGMENT at t=T (vel=0, acc=0, jerk=0) : 3 in total
    for i, k in enumerate([k1, k2, k3]):
        poly = polynom(n=nb_constrain, k=k, t=times[-1])
        A[row, (nb_splines-1)*nb_constrain:nb_constrain*nb_splines] = poly

        row += 1
    
    # CONSTRAIN FOR INTERMEDIARY SEGMENTS ONLY SO -> n_constrain - 1 constrains
    # so we will have (n_constrain - 1) for velocity at time T / same for acceleration/jerk/snap/snap_dot/snap_ddot
    # for velocity we want to fill the row with the value of velT - vel0, and in matrix b we will have a value of 0
    # in order to satisfy the fact that we want velT = vel0, same for 2nd derivative, 3rd, 4th, 5th, 6th.
    for s in range(1, nb_splines):  # loop over intermediate segments
        timeT = times[s]
        for k in [k1, k2, k3, k4, k5, k6]:
            poly0 = -1 * polynom(n=nb_constrain, k=k, t=time0)
            polyT = polynom(n=nb_constrain, k=k, t=timeT)
            poly = np.hstack((polyT, poly0))
            A[row, (s-1)*nb_constrain:nb_constrain*(s+1)] = poly     # here we fill the 16 values (hstack) on the same row from 0 to 16
            row += 1
    
    

    c = np.linalg.solve(A, b)  # spline parameters
    # c, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)


    c0, c1, c2, c3, c4, c5, c6, c7 = c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]
    for it in range(nb_splines):
        timeT = times[it]
        
        for t in np.arange(0.0, timeT, dt):
            # position = polynom(8, k=0, t=t) @ c[0:8]
            position = polynom(8, k=0, t=t) @ c[it*nb_constrain:nb_constrain*(it+1)]


            velocity = polynom(8, k=1, t=t) @ c[it*nb_constrain:nb_constrain*(it+1)]
            acceleration = polynom(8, k=2, t=t) @ c[it*nb_constrain:nb_constrain*(it+1)]
            jerk = polynom(8, k=3, t=t) @ c[it*nb_constrain:nb_constrain*(it+1)]
            snap = polynom(8, k=4, t=t) @ c[it*nb_constrain:nb_constrain*(it+1)]

            # position = c7 * t**7 + c6 * t**6 + c5 * t**5 + c4 * t**4 + c3 * t**3 + c2 * t**2 + c1 * t**1 + c0
            # velocity = 7 * c7 * t**6 + 6 * c6 * t**5 + 5 * c5 * t**4 + 4 * c4 * t**3 + 3 * c3 * t**2 + 2 * c2 * t + c1
            # acceleration = 42 * c7 * t**5 + 30 * c6 * t**4 + 20 * c5 * t**3 + 12 * c4 * t**2 + 6 * c3 * t + 2 * c2
            # jerk = 210 * c7  *t**4 + 120 * c6 * t**3 + 60 * c5 * t**2 + 24 * c4 * t + 6  *c3
            # snap = 840 * c7 * t**3 + 360 * c6 * t**2 + 180 * c5 * t + 24 * c4

            positions.append(position)
            velocities.append(velocity)
            accelerations.append(acceleration)
            jerks.append(jerk)
    
    
    traj = np.hstack((positions, velocities, accelerations))


    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    n_wp = len(waypoints)
    for i in range(n_wp):
        x, y, z = waypoints[i]
        ax.plot(x, y, z, alpha=.5, marker=".",  markersize=20)

    ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], marker='s', alpha=.5, s=2)
    plt.show()





  
    # TESTS FOR 4 SPLINES SEGMENTS
    # expected_pos0 = np.array([1., 0., 0., 0., 0., 0., 0., 0.])
    # expected_posT = np.array([1., timeT, timeT**2, timeT**3, timeT**4, timeT**5, timeT**6, timeT**7])
    # expected_vel0_first_seg = np.array([0., 1., 0., 0., 0., 0., 0., 0.])
    # expected_acc0_first_seg = np.array([0., 0., 2., 0., 0., 0., 0., 0.])
    # expected_jerk0_first_seg = np.array([0., 0., 0., 6., 0., 0., 0., 0.])

    # expected_velT = np.array([0., 1., 2*timeT, 3*timeT**2, 4*timeT**3, 5*timeT**4, 6*timeT**5, 7*timeT**6])
    # expected_accT = np.array([0., 0., 2., 6*timeT, 12*timeT**2, 20*timeT**3, 30*timeT**4, 42*timeT**5])
    # expected_jerkT = np.array([0., 0., 0., 6., 24*timeT, 60*timeT**2, 120*timeT**3, 210*timeT**4])

    # np.testing.assert_array_equal(A[0, :8], expected_pos0)
    # np.testing.assert_array_equal(A[1, 8:16], expected_pos0)
    # np.testing.assert_array_equal(A[2, 16:24], expected_pos0)
    # np.testing.assert_array_equal(A[3, 24:32], expected_pos0)

    # np.testing.assert_array_equal(A[4, :8], expected_posT)
    # np.testing.assert_array_equal(A[5, 8:16], expected_posT)
    # np.testing.assert_array_equal(A[6, 16:24], expected_posT)
    # np.testing.assert_array_equal(A[7, 24:32], expected_posT)

    # np.testing.assert_array_equal(A[8, :8], expected_vel0_first_seg)
    # np.testing.assert_array_equal(A[9, 8:16], expected_acc0_first_seg)
    # np.testing.assert_array_equal(A[10, 16:24], expected_jerk0_first_seg)

    # np.testing.assert_array_equal(A[11, :8], expected_velT)
    # np.testing.assert_array_equal(A[12, 8:16], expected_accT)
    # np.testing.assert_array_equal(A[13, 16:24], expected_jerkT)
