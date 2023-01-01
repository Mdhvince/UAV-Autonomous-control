import random
import warnings
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize

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
        r = 2
        h_max = 20
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
    def __init__(self, waypoints, velocity=1.0, dt=0.02):
        self.dt = dt 
        self.waypoints = waypoints
        self.velocity = velocity       # mean velocity of travel         
        self.times = []                # will hold the time between segment based on the velocity
        self.nb_splines = None
        self.nb_constraint = 8
        self.positions = []
        self.velocities = []
        self.accelerations = []
        self.jerks = []
        self.snap = []
        self.full_trajectory = None
        self.row_counter = 0
        self.constraint_poly = None    # A
        self.constraint_values = None  # b
        self.coeffs = None             # c
    
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
            self.coeffs, residuals, rank, s = np.linalg.lstsq(self.constraint_poly, self.constraint_values, rcond=None)
        else:
            self.coeffs = np.linalg.solve(self.constraint_poly, self.constraint_values)
            
    def _create_polynom_matrices(self):
        """Populate matrices A and b with the constraints/boundary conditions"""
        self._setup()        
        self._generate_position_constraints()
        self._generate_start_and_goal_constraints()
        self._generate_continuity_constraints()
    
    def _generate_continuity_constraints(self):
        """
        This function populate the A and b matrices with constraints on intermediate splines in order to ensure
        continuity, hence smoothness.

        - Constraints up to the 6th derivative at t=0 should be the same at t=T. For example no change of velocity
        between the end of a spline (polyT) and the start of the nex spline (poly0)

        We have 1 constraint for each derivatives(6).
        """

        N_BC = self.nb_constraint
        N_SPLINES = self.nb_splines

        for s in range(1, N_SPLINES):
            timeT = self.times[s]
            for k in [1, 2, 3, 4, 5, 6]:
                poly0 = -1 * TrajectoryPlanner.polynom(n=N_BC, k=k, t=0)
                polyT = TrajectoryPlanner.polynom(n=N_BC, k=k, t=timeT)
                poly = np.hstack((polyT, poly0))
                self.constraint_poly[self.row_counter, (s-1)*N_BC:N_BC*(s+1)] = poly
                self.row_counter += 1

    def _generate_start_and_goal_constraints(self):
        """
        This function populate the A and b matrices with constraints on the starting and ending splines.

        - Starting spline constraint: Velocity/Acceleration/Jerk should be 0
        - Ending spline constraint: Velocity/Acceleration/Jerk should be 0

        We have 1 constraint for each derivatives(3) and for 2 splines. So 3 constraints per splines. In total
        we have 6 constraints.
        """

        N_BC = self.nb_constraint
        N_SPLINES = self.nb_splines

        # CONSTRAINTS FOR THE VERY FIRST SEGMENT at t=0
        for k in [1, 2, 3]:
            poly = TrajectoryPlanner.polynom(n=N_BC, k=k, t=0)
            self.constraint_poly[self.row_counter, 0:N_BC] = poly                                                          # for only the first one so from 0:self.nb_constraint
            self.row_counter += 1
        
        # CONSTRAINTS FOR THE VERY LAST SEGMENT at t=T
        for k in [1, 2, 3]:
            poly = TrajectoryPlanner.polynom(n=N_BC, k=k, t=self.times[-1])
            self.constraint_poly[self.row_counter, (N_SPLINES-1)*N_BC:N_BC*N_SPLINES] = poly                               # only for the last one
            self.row_counter += 1

    def _generate_position_constraints(self):
        """
        This function populate the A and b matrices with constraints on positions.

        - The first position constraint is on every start of splines : every start of splines should
        be at a particular waypoint (Last waypoint is excluded since it is not a start of spline)
        - The second position constraint is on every end of splines : every end of splines should
        be at a particular waypoint (First waypoint is excluded since it is not an end of spline)

        If the number of splines is denoted by m, we have m constraints at t=0 and m constraints at t=T.
        So 2m constraints for position
        """

        N_BC = self.nb_constraint
        N_SPLINES = self.nb_splines

        # at t=0 - FOR ALL START OF SEGMENTS
        poly = TrajectoryPlanner.polynom(n=N_BC, k=0, t=0)
        for i in range(N_SPLINES):
            wp0 = self.waypoints[i]
            self.constraint_poly[self.row_counter, i*N_BC : N_BC*(i+1)] = poly
            self.constraint_values[self.row_counter, :] = wp0
            self.row_counter += 1
        
        # at t=T - FOR ALL END OF SEGMENTS                                                     
        for i in range(N_SPLINES):
            wpT = self.waypoints[i+1]
            timeT = self.times[i]
            poly = TrajectoryPlanner.polynom(n=N_BC, k=0, t=timeT)
            self.constraint_poly[self.row_counter, i*N_BC:N_BC*(i+1)] = poly
            self.constraint_values[self.row_counter, :] = wpT
            self.row_counter += 1

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
        self.constraint_poly = np.zeros((self.nb_constraint*self.nb_splines, self.nb_constraint*self.nb_splines))
        self.constraint_values = np.zeros((self.nb_constraint*self.nb_splines, 3))

    def _generate_time_per_spline(self):
        for i in range(self.nb_splines):
            # the time required to travel between each pair of waypoints
            distance = np.linalg.norm(self.waypoints[i+1] - self.waypoints[i])
            time = distance / self.velocity
            self.times.append(time)

    def _generate_waypoints(self):
        # while waiting for the algorithm to generate them
        self.nb_splines = self.waypoints.shape[0] - 1
    

if __name__ == "__main__":

    waypoints = np.array([
        [10, 0, 0], [10, 4, 1], [6, 3, 2], [7, 8, 1.5], [2, 7, 2], [1, 0, 2]
    ])
    # waypoints = getwp("angle").T

    # waypoints = np.array([
    #     [0, 0, 0], [0, 0, 20]
    # ])

    tp = TrajectoryPlanner(waypoints, velocity=1.0)
    traj = tp.get_min_snap_trajectory("inv")


    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    n_wp = len(waypoints)
    for i in range(n_wp):
        x, y, z = waypoints[i]
        ax.plot(x, y, z, alpha=.5, marker=".",  markersize=20)


    # only plot some of the rows
    n = 2
    mask = np.ones(traj.shape[0], dtype=bool)
    mask[::n] = False
    traj = traj[mask]
    
    # map color to velocity
    vel = np.linalg.norm(traj[:, 3:6], axis=1)
    max_vel = np.max(vel)
    norm = Normalize(vmin=0, vmax=max_vel)
    scalar_map = get_cmap("jet")
    sm = ScalarMappable(cmap=scalar_map, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, location="bottom", shrink=0.5)
    cbar.set_label('Velocity (m/s)')
    colors = scalar_map(norm(vel))


    # plot min snap
    for i in range(len(colors)):
        label = "Minimum snap trajectory" if i == 0 else None
        ax.plot(traj[i, 0], traj[i, 1], traj[i, 2],
                marker='.', alpha=.2, markersize=20, color=colors[i], label=label)

    # plot normal
    x, y, z = waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]
    ax.plot(x, y, z, alpha=.5, color="black", label="Naive trajectory")

    ax.legend()
    plt.show()

