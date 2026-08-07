import numpy as np


class Quad:
    """
    Quadrotor state, control parameters and rotor actuation in NED/FRD.

    MuJoCo owns the rigid-body dynamics and synchronizes the state vector.
    """

    def __init__(
            self,
            g: float,
            dt: float,
            mass: float,
            inertia: np.ndarray,
            arm_length: float,
            force_coefficient: float,
            drag_to_thrust: float,
            thrust_limits: np.ndarray,
            motor_time_constants: np.ndarray,
            flight_limits: np.ndarray,
    ):
        """
        :param g: gravity acceleration
        :param dt: MuJoCo and motor update time step
        :param mass: vehicle mass read from the MuJoCo body
        :param inertia: diagonal body inertia read from MuJoCo
        :param arm_length: roll and pitch lever arm of each rotor
        :param force_coefficient: rotor speed-to-thrust coefficient
        :param drag_to_thrust: rotor reaction torque-to-thrust ratio
        :param thrust_limits: minimum and maximum thrust per rotor
        :param motor_time_constants: rise and fall time constants
        :param flight_limits: ascent, descent, horizontal speed, acceleration and tilt limits
        """
        self.g = g
        self.dt = dt

        self.l = float(arm_length)
        self.m = float(mass)
        self.kf = float(force_coefficient)
        self.kappa = float(drag_to_thrust)
        self.i_x, self.i_y, self.i_z = np.asarray(inertia, dtype=float)
        self.min_thrust, self.max_thrust = np.asarray(thrust_limits, dtype=float)
        (
            self.max_ascent_rate,
            self.max_descent_rate,
            self.max_speed_xy,
            self.max_horiz_accel,
            self.max_tilt_angle,
        ) = np.asarray(flight_limits, dtype=float)

        # Controller response parameters
        self.tau_xy = 0.25
        self.zeta_xy = 0.875
        self.tau_altitude = 0.2
        self.zeta_altitude = 0.8
        self.tau_roll = 0.07
        self.tau_pitch = 0.07
        self.tau_yaw = 0.25
        self.tau_p = 0.008
        self.tau_q = 0.008
        self.tau_r = 0.09

        self.kp_xy, self.kd_xy = Quad.second_order_gains(self.tau_xy, self.zeta_xy)
        self.kp_z, self.kd_z = Quad.second_order_gains(self.tau_altitude, self.zeta_altitude)
        self.ki_z = 0.1
        self.kp_roll = 1 / self.tau_roll
        self.kp_pitch = 1 / self.tau_pitch
        self.kp_yaw = 1 / self.tau_yaw
        self.kp_p = 1 / self.tau_p
        self.kp_q = 1 / self.tau_q
        self.kp_r = 1 / self.tau_r

        # State synchronized from MuJoCo (position, quaternion, velocity, angular velocity body)
        # x = [x, y, z, q0, q1, q2, q3, x_dot, y_dot, z_dot, p, q, r]
        # quaternion q = [q0, q1, q2, q3] where q0 is the scalar part
        self.X = np.zeros(13)
        # Initialize quaternion to identity (no rotation)
        self.X[3] = 1.0  # q0 = 1, representing identity rotation

        # Propeller speed and first-order motor response
        self.motor_rise_time_constant, self.motor_fall_time_constant = np.asarray(
            motor_time_constants, dtype=float)
        self.omega = np.array([0.0, 0.0, 0.0, 0.0])
        self.omega_command = np.array([0.0, 0.0, 0.0, 0.0])

    def set_propeller_speed(self, thrust_cmd: float, moment_cmd: np.ndarray):
        """
        Convert a collective thrust and body moment command into individual propeller speeds.
        :param thrust_cmd: desired collective thrust [N]
        :param moment_cmd: desired moments about the body axes [tau_x, tau_y, tau_z] [N m]
        """
        rotor_forces = self._allocate_rotor_forces(thrust_cmd, moment_cmd)
        self.omega_command = np.sqrt(rotor_forces / self.kf)

        time_constants = np.where(
            self.omega_command > self.omega,
            self.motor_rise_time_constant,
            self.motor_fall_time_constant,
        )
        response = 1 - np.exp(-self.dt / time_constants)
        self.omega += response * (self.omega_command - self.omega)

    def _allocate_rotor_forces(self, thrust_cmd: float, moment_cmd: np.ndarray) -> np.ndarray:
        """Preserve feasible collective thrust while scaling moments to rotor limits."""
        c_bar = np.clip(thrust_cmd, self.min_thrust * 4, self.max_thrust * 4)
        p_bar = moment_cmd[0] / self.l
        q_bar = moment_cmd[1] / self.l
        r_bar = -moment_cmd[2] / self.kappa

        moment_forces = Quad.propeller_coeffs() @ np.array([p_bar, q_bar, r_bar, 0.0]) / 4
        collective_force = c_bar / 4
        scale_limits = np.ones(4)
        positive = moment_forces > 0
        negative = moment_forces < 0
        scale_limits[positive] = (self.max_thrust - collective_force) / moment_forces[positive]
        scale_limits[negative] = (self.min_thrust - collective_force) / moment_forces[negative]
        moment_scale = np.clip(np.min(scale_limits), 0.0, 1.0)

        rotor_forces = collective_force + moment_scale * moment_forces
        return np.clip(rotor_forces, self.min_thrust, self.max_thrust)

    @staticmethod
    def second_order_gains(time_constant: float, damping_ratio: float) -> tuple[float, float]:
        """Derive proportional and derivative gains from response parameters."""
        return 1 / time_constant ** 2, 2 * damping_ratio / time_constant

    def R(self):
        """Rotation matrix from quaternion"""
        return Quad.quat_to_rot(self.quaternion)

    @staticmethod
    def quat_to_rot(q: np.ndarray) -> np.ndarray:
        """
        Converts a quaternion to a rotation matrix.
        :param q: quaternion as numpy array [q0, q1, q2, q3] where q0 is the scalar part
        :return: 3x3 rotation matrix
        """
        # Normalize quaternion
        q = q / np.sqrt(np.sum(q ** 2))

        # Build skew-symmetric matrix to express 3D cross product as a matrix multiplication
        skew_symmetric_matrix = np.zeros((3, 3))
        skew_symmetric_matrix[0, 1] = -q[3]
        skew_symmetric_matrix[0, 2] = q[2]
        skew_symmetric_matrix[1, 2] = -q[1]
        skew_symmetric_matrix[1, 0] = q[3]
        skew_symmetric_matrix[2, 0] = -q[2]
        skew_symmetric_matrix[2, 1] = q[1]

        # Compute rotation matrix
        R = np.eye(3) + 2 * skew_symmetric_matrix @ skew_symmetric_matrix + 2 * q[0] * skew_symmetric_matrix

        return R

    @staticmethod
    def propeller_coeffs() -> np.ndarray:
        """
        Mixing matrix mapping [p_bar, q_bar, r_bar, c_bar] to the four rotor forces (after division by 4),
        consistent with the moment definitions tau_x, tau_y, tau_z below.
        """
        return np.array([[1, 1, 1, 1],  # front left
                         [-1, 1, -1, 1],  # front right
                         [-1, -1, 1, 1],  # rear left
                         [1, -1, -1, 1]])  # rear right

    @property
    def x(self):
        return self.X[0]

    @property
    def y(self):
        return self.X[1]

    @property
    def z(self):
        return self.X[2]

    @property
    def position(self):
        return np.array([self.x, self.y, self.z])

    @property
    def quaternion(self):
        """Returns quaternion [q0, q1, q2, q3] where q0 is the scalar part"""
        return self.X[3:7]

    @property
    def phi(self):
        """Roll angle extracted from quaternion"""
        q = self.quaternion
        # Roll (phi) = atan2(2(q0*q1 + q2*q3), 1 - 2(q1^2 + q2^2))
        return np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 
                         1 - 2 * (q[1]**2 + q[2]**2))

    @property
    def theta(self):
        """Pitch angle extracted from quaternion"""
        q = self.quaternion
        # Pitch (theta) = asin(2(q0*q2 - q3*q1))
        sin_theta = 2 * (q[0] * q[2] - q[3] * q[1])
        # Clamp to avoid numerical issues with arcsin
        sin_theta = np.clip(sin_theta, -1.0, 1.0)
        return np.arcsin(sin_theta)

    @property
    def psi(self):
        """Yaw angle extracted from quaternion"""
        q = self.quaternion
        # Yaw (psi) = atan2(2(q0*q3 + q1*q2), 1 - 2(q2^2 + q3^2))
        return np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 
                         1 - 2 * (q[2]**2 + q[3]**2))

    @property
    def euler_angles(self):
        return np.array([self.phi, self.theta, self.psi])

    @property
    def x_vel(self):
        return self.X[7]

    @property
    def y_vel(self):
        return self.X[8]

    @property
    def z_vel(self):
        return self.X[9]

    @property
    def velocity(self):
        return np.array([self.x_vel, self.y_vel, self.z_vel])

    # body rates [rad / s] (in body frame)
    @property
    def p(self):
        return self.X[10]

    @property
    def q(self):
        return self.X[11]

    @property
    def r(self):
        return self.X[12]

    @property
    def body_angular_velocity(self):
        return np.array([self.p, self.q, self.r])
