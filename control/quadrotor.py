import math
import numpy as np


class Quadrotor():
    def __init__(self, config, des):
        self.g = config["DEFAULT"].getfloat("g")
        quad_params = config["VEHICLE"]

        L = quad_params.getfloat("distance_rotor_to_rotor")
        self.l = L / math.sqrt(2)                                                      # distance from center to rotor
        self.m = quad_params.getfloat("mass")

        self.kf = quad_params.getfloat("kf")
        self.km = quad_params.getfloat("km")

        self.i_x = quad_params.getfloat("Ix")
        self.i_y = quad_params.getfloat("Iy")
        self.i_z = quad_params.getfloat("Iz")

        self.max_thrust = quad_params.getfloat("max_thrust")
        self.min_thrust = quad_params.getfloat("min_thrust")
        self.max_torque = quad_params.getfloat("max_torque")

        self.kappa = quad_params.getfloat("kappa")                                      # drag-thrust ratio

        self.max_ascent_rate = quad_params.getfloat("max_ascent_rate")
        self.max_descent_rate = quad_params.getfloat("max_descent_rate")
        self.max_speed_xy = quad_params.getfloat("max_speed_xy")
        self.max_horiz_accel = quad_params.getfloat("max_horiz_accel")
        self.max_tilt_angle = quad_params.getfloat("max_tilt_angle")


        # State (position, euler angles world, velocity, angular velocity body)
        # x = [x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r]
        self.X = np.zeros(12)

        # Derivative of state (velocity, euler ang vel, acceleration, angular acceleration body)
        # x_dot = [x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot, x_ddot, y_ddot, z_ddot, p_dot, q_dot, r_dot]
        self.dX = np.zeros(12)

        # Propeller speed
        self.omega = np.array([0.0, 0.0, 0.0, 0.0])

        # initialize the (x, y, yaw) state with the desired state
        self.X[0], self.X[1], self.X[5] = des.x[0], des.y[0], des.yaw[0]

    def update_state(self, dt):
        """
        Simulate evolution of the vehicle state over time. Not needed when running on real drone,
        since we can get the values from the sensors.
        """
        self.dX[:3] = self.velocity
        self.get_euler_derivatives()
        self.body_angular_acceleration()
        self.linear_acceleration()
        self.X = self.X + self.dX * dt  # integrate using euler method
    
    def set_propeller_speed(self, thrust_cmd, moment_cmd):
        c_bar = thrust_cmd
        p_bar = moment_cmd[0] / self.l
        q_bar = moment_cmd[1] / self.l
        r_bar = -moment_cmd[2] / self.kappa

        u_bar = np.array([p_bar, q_bar, r_bar, c_bar])

        self.omega = Quadrotor.propeller_coeffs() @ u_bar / 4
        
 
    def linear_acceleration(self):  # used for state update
        """
        Convert the thrust body frame to world frame , divide by the mass and add the gravity
        in order to have the linear acceleration x_acc, y_acc, z_acc in the world frame
        """
        R = self.R()
        G = np.array([0, 0, self.g]).T
        F = np.array([0, 0, -self.f_total]).T

        # linear accelerations along x, y, z
        self.dX[6:9] = G + np.matmul(R, F) / self.m
    
    def body_angular_acceleration(self):  # used for state update
        """Angular aceeleration in the body frame"""
        p_dot = (self.tau_x - self.r * self.q * (self.i_z - self.i_y)) / self.i_x
        q_dot = (self.tau_y - self.r * self.p * (self.i_x - self.i_z)) / self.i_y
        r_dot = (self.tau_z - self.q * self.p * (self.i_y - self.i_x)) / self.i_z
        
        self.dX[-3:] = np.array([p_dot, q_dot, r_dot])
    
    def get_euler_derivatives(self):  # used for state update
        """Angular velocity in the world frame"""

        euler_rot_mat = np.array([
                [1, math.sin(self.phi) * math.tan(self.theta), math.cos(self.phi) * math.tan(self.theta)],
                [0, math.cos(self.phi), -math.sin(self.phi)],
                [0, math.sin(self.phi) / math.cos(self.theta), math.cos(self.phi) / math.cos(self.theta)]
            ])

        pqr = self.body_angular_velocity.T

        # Angular velocities in world frame: phi_dot, theta_dot, psi_dot
        self.dX[3:6] = np.matmul(euler_rot_mat, pqr)
        
    
    def R(self):
        """ZYX"""
        r_x = np.array([[1, 0, 0],
                        [0, np.cos(self.phi), -np.sin(self.phi)],
                        [0, np.sin(self.phi), np.cos(self.phi)]])

        r_y = np.array([[np.cos(self.theta), 0, np.sin(self.theta)],
                        [0, 1, 0],
                        [-np.sin(self.theta), 0, np.cos(self.theta)]])

        r_z = np.array([[np.cos(self.psi), -np.sin(self.psi), 0],
                        [np.sin(self.psi), np.cos(self.psi), 0],
                        [0,0,1]])

        r_yx = np.matmul(r_y, r_x)
        return np.matmul(r_z, r_yx)
    

    @staticmethod
    def propeller_coeffs():
        return np.array([[1, 1, 1, 1],     # front left
                         [1, -1, 1, -1],   # front right
                         [1, 1, -1, -1],   # rear left
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
    def phi(self):
        return self.X[3]

    @property
    def theta(self):
        return self.X[4]

    @property
    def psi(self):
        return self.X[5]
    
    @property
    def euler_angles(self):
        return np.array([self.phi, self.theta, self.psi])
    
    @property
    def x_vel(self):
        return self.X[6]
    
    @property
    def y_vel(self):
        return self.X[7]
    
    @property
    def z_vel(self):
        return self.X[8]
    
    @property
    def velocity(self):
        return np.array([self.x_vel, self.y_vel, self.z_vel])

    # body rates [rad / s] (in body frame)
    @property 
    def p(self):
        return self.X[9]

    @property
    def q(self):
        return self.X[10]

    @property 
    def r(self):
        return self.X[11]
    
    @property
    def body_angular_velocity(self):
        return np.array([self.p, self.q, self.r])

    
    # forces from the four propellers [N]
    @property
    def f_1(self):
        f = self.kf*self.omega[0]**2
        return f

    @property 
    def f_2(self):
        f = self.kf*self.omega[1]**2
        return f

    @property 
    def f_3(self):
        f = self.kf*self.omega[2]**2
        return f

    @property 
    def f_4(self):
        f = self.kf*self.omega[3]**2
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
        tau = -self.km * self.omega[0]**2
        return tau
        
    @property
    def tau_2(self):
        tau = self.km * self.omega[1]**2
        return tau

    @property
    def tau_3(self):
        tau = -self.km * self.omega[2]**2
        return tau

    @property
    def tau_4(self):
        tau = self.km * self.omega[3]**2
        return tau

    # moments about axes [N * m]
    @property
    def tau_x(self):
        tau = self.l*(self.f_1 + self.f_4 - self.f_2 - self.f_3)
        return tau

    @property
    def tau_y(self):
        tau = self.l*(self.f_1 + self.f_2 - self.f_3 - self.f_4)
        return tau

    @property
    def tau_z(self):
        tau = self.tau_1 + self.tau_2 + self.tau_3 + self.tau_4
        return tau
    

if __name__ == "__main__":
    pass