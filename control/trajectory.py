import random
import warnings
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


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


class TrajectoryPlanner():
    def __init__(self, speed=1.0, dt=0.02):
        self.speed = speed   
        self.dt = dt                  # speed to maintain
        self.times = []               # will hold the time between segment based on the speed
        self.waypoints = None
        self.nb_splines = None
        self.nb_constraint = 8
        self.positions = []
        self.velocities = []
        self.accelerations = []
        self.jerks = []
        self.snap = []
        self.A = None
        self.b = None
        self.coeffs = None
        self.full_trajectory = None
    
    def get_min_snap_trajectory(self, method="lstsq"):
        self._compute_spline_parameters(method)

        NB_C = self.nb_constraint

        for it in range(self.nb_splines):
            timeT = self.times[it]
            
            for t in np.arange(0.0, timeT, self.dt):
                position = self.polynom(8, k=0, t=t) @ self.coeffs[it*NB_C : NB_C*(it+1)]
                velocity = self.polynom(8, k=1, t=t) @ self.coeffs[it*NB_C : NB_C*(it+1)]
                acceleration = self.polynom(8, k=2, t=t) @ self.coeffs[it*NB_C : NB_C*(it+1)]
                # jerk = self.polynom(8, k=3, t=t) @ self.coeffs[it*NB_C : NB_C*(it+1)]
                # snap = self.polynom(8, k=4, t=t) @ self.coeffs[it*NB_C : NB_C*(it+1)]

                self.positions.append(position)
                self.velocities.append(velocity)
                self.accelerations.append(acceleration)
                # self.jerks.append(jerk)
                # self.snap.append(snap)
        
        self.full_trajectory = np.hstack((self.positions, self.velocities, self.accelerations))
        return self.full_trajectory
    
    def _compute_spline_parameters(self, method):
        self._create_polynom_matrices()
        if method == "lstsq":
            self.coeffs, residuals, rank, s = np.linalg.lstsq(self.A, self.b, rcond=None)
        else:
            self.coeffs = np.linalg.solve(self.A, self.b)
            
    
    def _create_polynom_matrices(self):
        """Populate matrices A and b with the constraints/boundary conditions"""

        self._setup()
        N_BC = self.nb_constraint
        N_SPLINES = self.nb_splines
        row_counter = 0

        # m constraints on position at t=0 - FOR ALL START OF SEGMENTS, m=nb of splines
        
        poly = TrajectoryPlanner.polynom(n=N_BC, k=0, t=0)
        for i in range(N_SPLINES):
            wp0 = self.waypoints[i]
            self.A[row_counter, i*N_BC : N_BC*(i+1)] = poly     # i=0 -> 0:8   i=1 -> 8:16   i=2 -> 16:24  ...
            self.b[row_counter, :] = wp0                                        # required position at every start of a segment
            row_counter += 1
        
        # self.nb_constraint on position at t=T - FOR ALL END OF SEGMENTS                                                     
        for i in range(N_SPLINES):
            wpT = self.waypoints[i+1]
            timeT = self.times[i]
            poly = TrajectoryPlanner.polynom(n=N_BC, k=0, t=timeT)                                                  # fill A with polynomial for position at time t (from the 4th row)
            self.A[row_counter, i*N_BC:N_BC*(i+1)] = poly                                               # i=0 -> 0:8   i=1 -> 8:16   i=2 -> 16:24  ...
            self.b[row_counter, :] = wpT  # required position at every end of a segment
            row_counter += 1

        # CONSTRAIN FOR THE VERY FIRST SEGMENT at t=0 (vel=0, acc=0, jerk=0) : 3 in total
        for k in [1, 2, 3]:
            poly = TrajectoryPlanner.polynom(n=N_BC, k=k, t=0)
            self.A[row_counter, 0:N_BC] = poly  # for only the first one so from 0:self.nb_constraint
            row_counter += 1
        
        # CONSTRAIN FOR THE VERY LAST SEGMENT at t=T (vel=0, acc=0, jerk=0) : 3 in total
        for k in [1, 2, 3]:
            poly = TrajectoryPlanner.polynom(n=N_BC, k=k, t=self.times[-1])
            self.A[row_counter, (N_SPLINES-1)*N_BC:N_BC*N_SPLINES] = poly  # only for the last one
            row_counter += 1

        
        # CONSTRAIN FOR INTERMEDIARY SEGMENTS ONLY SO -> n_constrain - 1 constrains
        # so we will have (n_constrain - 1) for velocity at time T / same for acceleration/jerk/snap/snap_dot/snap_ddot
        # for velocity we want to fill the row with the value of velT - vel0, and in matrix b we will have a value of 0
        # in order to satisfy the fact that we want velT = vel0, same for 2nd derivative, 3rd, 4th, 5th, 6th.
        for s in range(1, N_SPLINES):  # loop over intermediate segments
            timeT = self.times[s]
            for k in [1, 2, 3, 4, 5, 6]:
                poly0 = -1 * TrajectoryPlanner.polynom(n=N_BC, k=k, t=0)
                polyT = TrajectoryPlanner.polynom(n=N_BC, k=k, t=timeT)
                poly = np.hstack((polyT, poly0))
                self.A[row_counter, (s-1)*N_BC:N_BC*(s+1)] = poly     # here we fill the 16 values (hstack) on the same row from 0 to 16
                row_counter += 1
    
    @staticmethod
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

    def _setup(self):
        self._generate_waypoints()
        self._generate_time_per_spline()
        self._init_matrices()

    def _init_matrices(self):
        self.A = np.zeros((self.nb_constraint*self.nb_splines, self.nb_constraint*self.nb_splines))
        self.b = np.zeros((self.nb_constraint*self.nb_splines, 3))

    def _generate_time_per_spline(self):
        for i in range(self.waypoints.shape[0] - 1):
            # the time required to travel between each pair of waypoints
            distance = np.linalg.norm(self.waypoints[i+1] - self.waypoints[i])
            time = distance / self.speed
            self.times.append(time)

    def _generate_waypoints(self):
        self.waypoints = np.array([  # while waiting for the algorithm to generate them
            [10, 0, 0], [10, 4, 1], [6, 5, 1.5], [7, 8, 1.5], [2, 7, 2], [1, 0, 2],
        ])
        self.nb_splines = self.waypoints.shape[0] - 1
    

if __name__ == "__main__":

    
    tp = TrajectoryPlanner()
    traj = tp.get_min_snap_trajectory()


    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    n_wp = len(tp.waypoints)
    for i in range(n_wp):
        x, y, z = tp.waypoints[i]
        ax.plot(x, y, z, alpha=.5, marker=".",  markersize=20)

    ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], marker='s', alpha=.5, s=2)

    plt.show()

