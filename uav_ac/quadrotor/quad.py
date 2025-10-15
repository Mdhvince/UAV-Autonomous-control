import math
import numpy as np


class Quad:
    def __init__(self, cfg):
        self.g = cfg.getfloat("g")
        self.dt = cfg.getfloat("dt") / cfg.getint("frequency")

        # Vehicle physical/geometrical properties (hardcoded)
        distance_rotor_to_rotor = 0.17  # [m]
        self.l = distance_rotor_to_rotor / math.sqrt(2)  # distance from center to rotor
        self.m = 0.5  # [kg]
        self.kf = 1.0
        self.km = 1.0
        self.i_x = 0.0023  # [kg m2]
        self.i_y = 0.0023
        self.i_z = 0.0046
        self.kappa = 0.016  # drag/thrust ratio
        self.max_thrust = 4.5  # N
        self.min_thrust = 0.1
        self.max_torque = 1.0  # Nm
        self.max_ascent_rate = 3
        self.max_descent_rate = 2
        self.max_speed_xy = 3
        self.max_horiz_accel = 12
        self.max_tilt_angle = 0.7

        # Controller gains (hardcoded)
        self.kp_xy = 30
        self.kd_xy = 12
        self.kp_z = 11
        self.kd_z = 9
        self.ki_z = 0.1
        self.kp_roll = 10
        self.kp_pitch = 10
        self.kp_yaw = 2
        self.kp_p = 95
        self.kp_q = 95
        self.kp_r = 6

        # State (position, quaternion, velocity, angular velocity body)
        # x = [x, y, z, q0, q1, q2, q3, x_dot, y_dot, z_dot, p, q, r]
        # quaternion q = [q0, q1, q2, q3] where q0 is the scalar part
        self.X = np.zeros(13)
        # Initialize quaternion to identity (no rotation)
        self.X[3] = 1.0  # q0 = 1, representing identity rotation

        # Derivative of state (velocity, quaternion derivative, acceleration, angular acceleration body)
        # x_dot = [x_dot, y_dot, z_dot, q0_dot, q1_dot, q2_dot, q3_dot, x_ddot, y_ddot, z_ddot, p_dot, q_dot, r_dot]
        self.dX = np.zeros(13)

        # Propeller speed
        self.omega = np.array([0.0, 0.0, 0.0, 0.0])

    def update_state(self):
        """
        Simulate evolution of the vehicle state over time. Not needed when running on real drone,
        since we can get the values from the sensors.
        """
        self.dX[:3] = self.velocity
        self.update_quaternion_derivatives()
        self.update_body_angular_acceleration()
        self.update_acceleration()
        self.X = self.X + self.dX * self.dt  # integrate using euler method
        # Normalize quaternion to prevent drift
        self.X[3:7] = self.X[3:7] / np.linalg.norm(self.X[3:7])

    def set_propeller_speed(self, thrust_cmd, moment_cmd):
        c_bar = thrust_cmd
        p_bar = moment_cmd[0] / self.l
        q_bar = moment_cmd[1] / self.l
        r_bar = -moment_cmd[2] / self.kappa

        u_bar = np.array([p_bar, q_bar, r_bar, c_bar])

        self.omega = Quad.propeller_coeffs() @ u_bar / 4

    def update_acceleration(self):  # used for state update
        """
        Convert the thrust body frame to world frame , divide by the mass and add the gravity
        in order to have the linear acceleration x_acc, y_acc, z_acc in the world frame
        """
        R = self.R()
        G = np.array([0, 0, self.g]).T
        F = np.array([0, 0, -self.f_total]).T

        # linear accelerations along x, y, z
        self.dX[7:10] = G + np.matmul(R, F) / self.m

    def update_body_angular_acceleration(self):  # used for state update
        """Angular acceleration in the body frame"""
        p_dot = (self.tau_x - self.r * self.q * (self.i_z - self.i_y)) / self.i_x
        q_dot = (self.tau_y - self.r * self.p * (self.i_x - self.i_z)) / self.i_y
        r_dot = (self.tau_z - self.q * self.p * (self.i_y - self.i_x)) / self.i_z

        self.dX[-3:] = np.array([p_dot, q_dot, r_dot])

    def update_quaternion_derivatives(self):  # used for state update
        """Compute quaternion derivative from body angular velocities"""
        # Get body angular velocities
        p, q, r = self.p, self.q, self.r
        
        # Build the omega matrix: 4x4 matrix for quaternion derivative computation
        # q_dot = 0.5 * Omega * q
        omega_matrix = np.array([
            [0, -p, -q, -r],
            [p,  0,  r, -q],
            [q, -r,  0,  p],
            [r,  q, -p,  0]
        ])
        
        # Compute quaternion derivative
        q_current = self.quaternion
        q_dot = 0.5 * np.matmul(omega_matrix, q_current)
        
        # Store in derivative vector
        self.dX[3:7] = q_dot

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
    def propeller_coeffs():
        return np.array([[1, 1, 1, 1],  # front left
                         [1, -1, 1, -1],  # front right
                         [1, 1, -1, -1],  # rear left
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

    # forces from the four propellers [N]
    @property
    def f_1(self):
        f = self.kf * self.omega[0] ** 2
        return f

    @property
    def f_2(self):
        f = self.kf * self.omega[1] ** 2
        return f

    @property
    def f_3(self):
        f = self.kf * self.omega[2] ** 2
        return f

    @property
    def f_4(self):
        f = self.kf * self.omega[3] ** 2
        return f

    # collective force
    @property
    def f_total(self):
        """Actual Thrust. Different from the Desired thrust in (thrust_cmd)"""
        f_t = self.f_1 + self.f_2 + self.f_3 + self.f_4
        return f_t

    # reactive moments [N * m]
    @property
    def tau_1(self):
        tau = -self.km * self.omega[0] ** 2
        return tau

    @property
    def tau_2(self):
        tau = self.km * self.omega[1] ** 2
        return tau

    @property
    def tau_3(self):
        tau = -self.km * self.omega[2] ** 2
        return tau

    @property
    def tau_4(self):
        tau = self.km * self.omega[3] ** 2
        return tau

    # moments about axes [N * m]
    @property
    def tau_x(self):
        tau = self.l * (self.f_1 + self.f_4 - self.f_2 - self.f_3)
        return tau

    @property
    def tau_y(self):
        tau = self.l * (self.f_1 + self.f_2 - self.f_3 - self.f_4)
        return tau

    @property
    def tau_z(self):
        tau = self.tau_1 + self.tau_2 + self.tau_3 + self.tau_4
        return tau


if __name__ == "__main__":
    pass
