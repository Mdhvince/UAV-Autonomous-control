import numpy as np

from planning.obstacle import Obstacle


class MinimumSnap:
    def __init__(self, waypoints, velocity=1.0, dt=0.02):
        self.obstacle_edges = None
        self.dt = dt
        self.waypoints = waypoints
        self.velocity = velocity                # mean velocity of travel
        self.times = []                         # will hold the time between segment based on the velocity
        self.spline_id = []                     # identify on which spline the newly generated point belongs to
        self.nb_splines = None                  # number of splines in the trajectory
        self.nb_coeffs = 8          # number of boundary conditions per spline to respect minimum snap

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


    def generate_collision_free_trajectory(self, coord_obstacles=None):
        """
        Generate a collision free trajectory. The trajectory is generated in two steps:
        1. Generate a minimum snap trajectory
        2. Correct the trajectory to avoid collision with obstacles:
        - if the trajectory goes through an obstacle, create a mid-point in the spline that goes through the obstacle

        :param coord_obstacles: list of coordinates of the obstacles
        """
        self.obstacle_edges = []

        if coord_obstacles is not None:
            # create a collision free minimal snap path
            for coord in coord_obstacles:
                self.reset()
                Obs = Obstacle(center=coord[:2], side_length=coord[2], height=coord[3], altitude_start=coord[4])
                self.obstacle_edges.append(Obs.edges)

                # Generate a minimum snap trajectory and check if there is collision with the current obstacle
                traj = self.generate_trajectory()

                # create mid-point in splines that goes through an obstacle
                id_spline_to_correct = {1}
                while len(id_spline_to_correct) > 0:

                    id_spline_to_correct = set([])
                    for n, point in enumerate(traj[:, :3]):
                        if MinimumSnap.is_collision(point, Obs.vertices):
                            spline_id = traj[n, -1]
                            id_spline_to_correct.add(spline_id + 1)

                    if len(id_spline_to_correct) > 0:
                        self.reset()
                        new_waypoints = MinimumSnap.insert_midpoints_at_indexes(self.waypoints, id_spline_to_correct)
                        self.waypoints = new_waypoints
                        traj = self.generate_trajectory()
        else:
            _ = self.generate_trajectory()


    def generate_trajectory(self, method="lstsq"):
        self._compute_spline_parameters(method)

        for it in range(self.nb_splines):
            timeT = self.times[it]

            for t in np.arange(0.0, timeT, self.dt):
                position = self.polynom(8, k=0, t=t) @ self.coeffs[it * self.nb_coeffs: self.nb_coeffs * (it + 1)]
                velocity = self.polynom(8, k=1, t=t) @ self.coeffs[it * self.nb_coeffs: self.nb_coeffs * (it + 1)]
                acceleration = self.polynom(8, k=2, t=t) @ self.coeffs[it * self.nb_coeffs: self.nb_coeffs * (it + 1)]
                # jerk = self.polynom(8, k=3, t=t) @ self.coeffs[it*self.nb_coeffs : self.nb_coeffs*(it+1)]
                # snap = self.polynom(8, k=4, t=t) @ self.coeffs[it*self.nb_coeffs : self.nb_coeffs*(it+1)]

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

        N_SPLINES = self.nb_splines

        for s in range(1, N_SPLINES):
            timeT = self.times[s - 1]
            for k in [1, 2, 3, 4, 5, 6]:
                poly0 = -1 * MinimumSnap.polynom(n=self.nb_coeffs, k=k, t=0)
                polyT = MinimumSnap.polynom(n=self.nb_coeffs, k=k, t=timeT)
                poly = np.hstack((polyT, poly0))  # (end of seg) - (start of seg) must be 0. so no change of vel/acc/...
                self.A[self.row_counter, (s - 1) * self.nb_coeffs:self.nb_coeffs * (s + 1)] = poly
                self.row_counter += 1


    def _generate_start_and_goal_constraints(self):
        """
        This function populate the A and b matrices with constraints on the starting and ending splines.

        - Starting spline constraint: Velocity/Acceleration/Jerk should be 0
        - Ending spline constraint: Velocity/Acceleration/Jerk should be 0

        We have 1 constraint for each derivative(3) and for 2 splines. So 3 constraints per splines. In total,
        we have 6 constraints.
        """

        N_SPLINES = self.nb_splines

        # CONSTRAINTS FOR THE VERY FIRST SEGMENT at t=0
        for k in [1, 2, 3]:
            poly = MinimumSnap.polynom(n=self.nb_coeffs, k=k, t=0)
            self.A[self.row_counter, 0:self.nb_coeffs] = poly
            self.row_counter += 1

        # CONSTRAINTS FOR THE VERY LAST SEGMENT at t=T
        for k in [1, 2, 3]:
            poly = MinimumSnap.polynom(n=self.nb_coeffs, k=k, t=self.times[-1])
            self.A[self.row_counter, (N_SPLINES - 1) * self.nb_coeffs:self.nb_coeffs * N_SPLINES] = poly
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

        N_SPLINES = self.nb_splines

        # at t=0 - FOR ALL START OF SEGMENTS
        poly = MinimumSnap.polynom(n=self.nb_coeffs, k=0, t=0)
        for i in range(N_SPLINES):
            wp0 = self.waypoints[i]
            self.A[self.row_counter, i * self.nb_coeffs: self.nb_coeffs * (i + 1)] = poly
            self.b[self.row_counter, :] = wp0
            self.row_counter += 1

        # at t=T - FOR ALL END OF SEGMENTS                                                     
        for i in range(N_SPLINES):
            wpT = self.waypoints[i + 1]
            timeT = self.times[i]
            poly = MinimumSnap.polynom(n=self.nb_coeffs, k=0, t=timeT)
            self.A[self.row_counter, i * self.nb_coeffs:self.nb_coeffs * (i + 1)] = poly
            self.b[self.row_counter, :] = wpT
            self.row_counter += 1


    @staticmethod
    def polynom(n, k, t):
        """
        This function returns a polynom of degree n and order k evaluated at t.
        :param n: degree of the polynom or number of boundary conditions or number of unknown coefficients
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
        self.A = np.zeros((self.nb_coeffs * self.nb_splines, self.nb_coeffs * self.nb_splines))
        self.b = np.zeros((self.nb_coeffs * self.nb_splines, len(self.waypoints[0])))


    def _generate_time_per_spline(self):
        """
        This function computes the time required to travel between each pair of waypoints given the velocity.
        """
        for i in range(self.nb_splines):
            distance = np.linalg.norm(self.waypoints[i + 1] - self.waypoints[i])
            time = distance / self.velocity
            self.times.append(time)


    def _generate_waypoints(self):
        # while waiting for the algorithm to generate them
        self.nb_splines = self.waypoints.shape[0] - 1


    @staticmethod
    def is_collision(point, vertices):
        """
        :param point: row numpy vector representing (x, y, z) coordinates of a point
        :param vertices: list of (x, y, z) coordinates of the vertices of obstacles cubes
        """
        x, y, z = point
        try:
            xs, ys, zs = zip(*vertices)
        except ValueError:
            return False

        return min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys) and min(zs) <= z <= max(zs)


    @staticmethod
    def insert_midpoints_at_indexes(points, indexes):
        """
        :param points: 2D numpy array of shape (n, 3) where n is the number of points and 3 is the dimension (x, y, z)
        :param indexes: list of indexes where to insert the midpoints (between which points we want to insert midpoints)
        the index is the index of the last point of the segment. So if we want to insert a midpoint at "index" we need
        to insert it between points[index-1] and points[index].
        :return: a 2D numpy array of shape (n + len(indexes), 3) where n is the number of points and 3 is the dimension
        """
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


if __name__ == "__main__":
    pass
