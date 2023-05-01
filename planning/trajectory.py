import random
import warnings
from collections import namedtuple

import numpy as np
import matplotlib.path

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize

warnings.filterwarnings('ignore')
plt.style.use('ggplot')


def get_path(total_time=20, dt=0.01):
    Desired = namedtuple(
        "Desired", ["x", "y", "z", "x_vel", "y_vel", "z_vel", "x_acc", "y_acc", "z_acc", "yaw"])

    t = np.linspace(0.0, total_time, int(total_time / dt))

    omega_x = 0.8
    omega_y = 0.4
    omega_z = 0.4

    a_x = 1.0
    a_y = 1.0
    a_z = 1.0

    x = a_x * np.sin(omega_x * t)
    x_vel = a_x * omega_x * np.cos(omega_x * t)
    x_acc = -a_x * omega_x ** 2 * np.sin(omega_x * t)

    y = a_y * np.cos(omega_y * t) + 2
    y_vel = -a_y * omega_y * np.sin(omega_y * t)
    y_acc = -a_y * omega_y ** 2 * np.cos(omega_y * t)

    z = a_z * np.cos(omega_z * t) + 2
    z_vel = -a_z * omega_z * np.sin(omega_z * t)
    z_acc = - a_z * omega_z ** 2 * np.cos(omega_z * t)

    yaw = np.arctan2(y_vel, x_vel)

    desired_trajectory = Desired(x, y, z, x_vel, y_vel, z_vel, x_acc, y_acc, z_acc, yaw)

    return t, dt, desired_trajectory


def get_path_helix(total_time=20, r=1, height=10, dt=0.01):
    Desired = namedtuple(
        "Desired", ["x", "y", "z", "x_vel", "y_vel", "z_vel", "x_acc", "y_acc", "z_acc", "yaw"])

    t = np.linspace(0.0, total_time, int(total_time / dt))

    omega_x = 0.8
    omega_y = 0.4
    omega_z = 0.4

    a_x = r  # radius of helix
    a_y = r  # radius of helix
    a_z = height / total_time  # vertical movement of helix per unit time

    x = a_x * np.sin(omega_x * t)
    x_vel = a_x * omega_x * np.cos(omega_x * t)
    x_acc = -a_x * omega_x ** 2 * np.sin(omega_x * t)

    y = a_y * np.cos(omega_y * t)
    y_vel = -a_y * omega_y * np.sin(omega_y * t)
    y_acc = -a_y * omega_y ** 2 * np.cos(omega_y * t)

    z = a_z * t  # z moves linearly with time
    z_vel = np.full(len(t), a_z)  # z_vel is constant
    z_acc = np.full(len(t), 0)  # z_acc is zero

    yaw = np.arctan2(y_vel, x_vel)

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


def insert_midpoints_at_indexes(points, indexes):
    result = []
    i = 0
    while i < len(points):
        if i in indexes:
            p1 = points[i - 1]
            p2 = points[i]
            midpoint = (p1 + p2) / 2
            result.extend([midpoint])
        result.append(points[i])
        i += 1
    return np.array(result)


def is_collision(point, vertices):
    x, y, z = point
    xs, ys, zs = zip(*vertices)
    return min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys) and min(zs) <= z <= max(zs)


class MinimumSnap:
    def __init__(self, waypoints, velocity=1.0, dt=0.02):
        self.obstacle_edges = None
        self.dt = dt
        self.waypoints = waypoints
        self.velocity = velocity                # mean velocity of travel
        self.times = []                         # will hold the time between segment based on the velocity
        self.spline_id = []                     # identify on which spline the newly generated point belongs to
        self.nb_splines = None                  # number of splines in the trajectory
        self.n_boundary_conditions = 8          # number of boundary conditions per spline to respect minimum snap

        self.positions = []                     # will hold the desired positions of the trajectory
        self.velocities = []                    # will hold the desired velocities of the trajectory
        self.accelerations = []                 # will hold the desired accelerations of the trajectory
        self.jerks = []                         # will hold the desired jerks of the trajectory
        self.snap = []                          # will hold the desired snap of the trajectory

        self.full_trajectory = None             # will hold the full trajectory
        self.row_counter = 0                    # keep track of the current row being filled in the A matrix
        self.A = None
        self.b = None
        self.coeffs = None                      # will hold the coefficients of the trajectory

    def reset(self):
        self.times = []
        self.spline_id = []
        self.nb_splines = None
        self.positions = []
        self.velocities = []
        self.accelerations = []
        self.jerks = []
        self.snap = []
        self.full_trajectory = None
        self.row_counter = 0
        self.A = None
        self.b = None
        self.coeffs = None

    def generate_collision_free_trajectory(self, coord_obstacles):
        self.obstacle_edges = []

        # create a collision free minimal snap path
        for coord in coord_obstacles:
            self.reset()
            O = Obstacle(center=coord[:2], side_length=coord[2], height=coord[3], altitude_start=coord[4])
            self.obstacle_edges.append(O.edges)

            # Generate a minimum snap trajectory and check if there is collision with the current obstacle
            traj = self.generate_trajectory()

            # create mid-point in splines that goes through an obstacle
            id_spline_to_correct = set([1])
            while len(id_spline_to_correct) > 0:

                id_spline_to_correct = set([])
                for n, point in enumerate(traj[:, :3]):
                    if is_collision(point, O.vertices):
                        spline_id = traj[n, -1]
                        id_spline_to_correct.add(spline_id + 1)

                if len(id_spline_to_correct) > 0:
                    self.reset()
                    new_waypoints = insert_midpoints_at_indexes(self.waypoints, id_spline_to_correct)
                    self.waypoints = new_waypoints
                    traj = self.generate_trajectory()

    def generate_trajectory(self, method="lstsq"):
        self._compute_spline_parameters(method)

        NB_C = self.n_boundary_conditions

        for it in range(self.nb_splines):
            timeT = self.times[it]

            for t in np.arange(0.0, timeT, self.dt):
                position = self.polynom(8, k=0, t=t) @ self.coeffs[it * NB_C: NB_C * (it + 1)]
                velocity = self.polynom(8, k=1, t=t) @ self.coeffs[it * NB_C: NB_C * (it + 1)]
                acceleration = self.polynom(8, k=2, t=t) @ self.coeffs[it * NB_C: NB_C * (it + 1)]
                # jerk = self.polynom(8, k=3, t=t) @ self.coeffs[it*NB_C : NB_C*(it+1)]
                # snap = self.polynom(8, k=4, t=t) @ self.coeffs[it*NB_C : NB_C*(it+1)]

                self.positions.append(position)
                self.velocities.append(velocity)
                self.accelerations.append(acceleration)
                self.spline_id.append(np.array([it, it]))
                # self.jerks.append(jerk)
                # self.snap.append(snap)

        self.full_trajectory = np.hstack((self.positions, self.velocities, self.accelerations, self.spline_id))
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
        self._generate_position_constraints()
        self._generate_start_and_goal_constraints()
        self._generate_continuity_constraints()

    def _generate_continuity_constraints(self):
        """
        This function populate the A and b matrices with constraints on intermediate splines in order to ensure
        continuity, hence smoothness.

        - Constraints up to the 6th derivative at t=0 should be the same at t=T. For example no change of velocity
        between the end of a spline (polyT) and the start of the next spline (poly0)

        We have 1 constraint for each derivatives(6).
        """

        N_BC = self.n_boundary_conditions
        N_SPLINES = self.nb_splines

        for s in range(1, N_SPLINES):
            timeT = self.times[s - 1]
            for k in [1, 2, 3, 4, 5, 6]:
                poly0 = -1 * MinimumSnap.polynom(n=N_BC, k=k, t=0)
                polyT = MinimumSnap.polynom(n=N_BC, k=k, t=timeT)
                poly = np.hstack((polyT, poly0))  # (end of seg) - (start of seg) must be 0. so no change of velocity/acc/jerk/snap...
                self.A[self.row_counter, (s - 1) * N_BC:N_BC * (s + 1)] = poly
                self.row_counter += 1

    def _generate_start_and_goal_constraints(self):
        """
        This function populate the A and b matrices with constraints on the starting and ending splines.

        - Starting spline constraint: Velocity/Acceleration/Jerk should be 0
        - Ending spline constraint: Velocity/Acceleration/Jerk should be 0

        We have 1 constraint for each derivative(3) and for 2 splines. So 3 constraints per splines. In total,
        we have 6 constraints.
        """

        N_BC = self.n_boundary_conditions
        N_SPLINES = self.nb_splines

        # CONSTRAINTS FOR THE VERY FIRST SEGMENT at t=0
        for k in [1, 2, 3]:
            poly = MinimumSnap.polynom(n=N_BC, k=k, t=0)
            self.A[self.row_counter, 0:N_BC] = poly
            self.row_counter += 1

        # CONSTRAINTS FOR THE VERY LAST SEGMENT at t=T
        for k in [1, 2, 3]:
            poly = MinimumSnap.polynom(n=N_BC, k=k, t=self.times[-1])
            self.A[self.row_counter, (N_SPLINES - 1) * N_BC:N_BC * N_SPLINES] = poly
            self.row_counter += 1

    def _generate_position_constraints(self):
        """
        This function populate the A and b matrices with constraints on positions.

        - The first position constraint is on every start of splines : every start of splines should
        be at a particular waypoint (Last waypoint is excluded since it is not a start of spline)
        - The second position constraint is on every end of splines : every end of splines should
        be at a particular waypoint (First waypoint is excluded since it is not an end of spline)

        If the number of splines is denoted by m, we have m constraints at t=0 (start of spline) and m constraints at
        t=T (emd of spline). So 2m constraints for position
        """

        N_BC = self.n_boundary_conditions  
        N_SPLINES = self.nb_splines

        # at t=0 - FOR ALL START OF SEGMENTS
        poly = MinimumSnap.polynom(n=N_BC, k=0, t=0)
        for i in range(N_SPLINES):
            wp0 = self.waypoints[i]
            self.A[self.row_counter, i * N_BC: N_BC * (i + 1)] = poly
            self.b[self.row_counter, :] = wp0
            self.row_counter += 1

        # at t=T - FOR ALL END OF SEGMENTS                                                     
        for i in range(N_SPLINES):
            wpT = self.waypoints[i + 1]
            timeT = self.times[i]
            poly = MinimumSnap.polynom(n=N_BC, k=0, t=timeT)
            self.A[self.row_counter, i * N_BC:N_BC * (i + 1)] = poly
            self.b[self.row_counter, :] = wpT
            self.row_counter += 1

    @staticmethod
    def polynom(n, k, t):
        """
        This function returns a polynom of degree n and order k evaluated at t.
        :param n: degree of the polynom or number of boundary conditions
        :param k: order of the polynom (k=1: velocity; k=2: acceleration; k=3: jerk; k=4: snap)
        :param t: time at which the polynom is evaluated
        :return: the polynom evaluated at t
        """
        T = np.zeros(n)  # T is the polynom
        D = np.zeros(n)  # D is the derivative of the polynom

        # Initialisation
        for i in range(n):
            D[i] = i
            T[i] = 1

        # compute derivative
        for j in range(k):
            for i in range(n):

                T[i] = T[i] * D[i]
                if D[i] > 0:
                    D[i] = D[i] - 1

        # compute polynom
        for i in range(n):
            T[i] = T[i] * t ** D[i]

        return T.T

    def _setup(self):
        self._generate_waypoints()
        self._generate_time_per_spline()
        self._init_matrices()

    def _init_matrices(self):
        self.A = np.zeros((self.n_boundary_conditions * self.nb_splines, self.n_boundary_conditions * self.nb_splines))
        self.b = np.zeros((self.n_boundary_conditions * self.nb_splines, len(self.waypoints[0])))

        print("A shape: ", self.A.shape)

    def _generate_time_per_spline(self):
        """
        This function computes the time required to travel between each pair of waypoints.
        """
        for i in range(self.nb_splines):
            distance = np.linalg.norm(self.waypoints[i + 1] - self.waypoints[i])
            time = distance / self.velocity
            self.times.append(time)

    def _generate_waypoints(self):
        # while waiting for the algorithm to generate them
        self.nb_splines = self.waypoints.shape[0] - 1


class Obstacle:
    def __init__(self, center, side_length, height, altitude_start=0):

        self.vertices = [
            (center[0] - side_length / 2, center[1] - side_length / 2, altitude_start + height),
            (center[0] + side_length / 2, center[1] - side_length / 2, altitude_start + height),
            (center[0] + side_length / 2, center[1] + side_length / 2, altitude_start + height),
            (center[0] - side_length / 2, center[1] + side_length / 2, altitude_start + height),
            (center[0] - side_length / 2, center[1] - side_length / 2, altitude_start),
            (center[0] + side_length / 2, center[1] - side_length / 2, altitude_start),
            (center[0] + side_length / 2, center[1] + side_length / 2, altitude_start),
            (center[0] - side_length / 2, center[1] + side_length / 2, altitude_start)
        ]

        # self.edges = []
        # for i, vertex1 in enumerate(self.vertices):
        #     for vertex2 in self.vertices[i+1:]:
        #         self.edges.append((vertex1, vertex2))

        self.edges = [
            (self.vertices[0], self.vertices[1]),
            (self.vertices[1], self.vertices[2]),
            (self.vertices[2], self.vertices[3]),
            (self.vertices[3], self.vertices[0]),
            (self.vertices[4], self.vertices[5]),
            (self.vertices[5], self.vertices[6]),
            (self.vertices[6], self.vertices[7]),
            (self.vertices[7], self.vertices[4]),
            (self.vertices[0], self.vertices[4]),
            (self.vertices[1], self.vertices[5]),
            (self.vertices[2], self.vertices[6]),
            (self.vertices[3], self.vertices[7])
        ]


if __name__ == "__main__":

    waypoints = np.array([[10., 0.0, 0.0],
                          [10., 4.0, 1.0],
                          [6.0, 5.0, 1.5],
                          [7.0, 8.0, 1.5],
                          [2.0, 7.0, 2.0],
                          [1.0, 0.0, 2.0], ])

    coord_obstacles = np.array([[8.0, 6.0, 1.5, 5.0, 0.0],  # x, y, side_length, height, altitude_start
                                [4.0, 9.0, 1.5, 5.0, 0.0],
                                [4.0, 1.0, 2.0, 5.0, 0.0],
                                [3.0, 5.0, 1.0, 5.0, 0.0],
                                [4.0, 3.5, 2.5, 5.0, 0.0],
                                [5.0, 5.0, 10., 0.5, 5.0]])

    T = MinimumSnap(waypoints, velocity=1.0, dt=0.02)
    T.generate_collision_free_trajectory(coord_obstacles)
    traj = T.full_trajectory

    ############################################################## PLOTTING ##############################################################

    # filter-out some rows for plotting
    n = 2
    for _ in range(2):
        mask = np.ones(traj.shape[0], dtype=bool)
        mask[::n] = False
        traj = traj[mask]

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(90, -90)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

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
    for i in range(len(traj)):
        label = "Minimum snap trajectory" if i == 0 else None
        ax.plot(traj[i, 0], traj[i, 1], traj[i, 2], marker='.', alpha=.2, markersize=20, color=colors[i], label=label)

    # plot waypoints
    for i in range(len(waypoints)):
        x, y, z = waypoints[i]
        ax.plot(x, y, z, marker=".", markersize=20, alpha=.2)

    # plot obstacles
    for edges in T.obstacle_edges:
        for edge in edges:
            x, y, z = zip(*edge)
            ax.plot(x, y, z, color="red", alpha=.2)

    ax.legend()
    ax.grid(False)
    plt.show()
