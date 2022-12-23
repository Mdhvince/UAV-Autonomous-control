import math
import numpy as np


class Quadrotor():
    
    def __init__(self, config, desired):

        self.g = config["DEFAULT"].getfloat("g")
        quad_params = config["VEHICLE"]
        L = quad_params.getfloat("distance_rotor_to_rotor")
        self.kf = quad_params.getfloat("kf")
        self.km = quad_params.getfloat("km")
        self.m = quad_params.getfloat("mass")
        self.i_x = quad_params.getfloat("Ix")
        self.i_y = quad_params.getfloat("Iy")
        self.i_z = quad_params.getfloat("Iz")
        self.max_thrust = quad_params.getfloat("max_thrust")
        self.min_thrust = quad_params.getfloat("min_thrust")
        self.max_torque = quad_params.getfloat("max_torque")
        self.kappa = quad_params.getfloat("kappa")

        self.max_ascent_rate = quad_params.getfloat("max_ascent_rate")
        self.max_descent_rate = quad_params.getfloat("max_descent_rate")
        self.max_speed_xy = quad_params.getfloat("max_speed_xy")
        self.max_horiz_accel = quad_params.getfloat("max_horiz_accel")
        self.max_tilt_angle = quad_params.getfloat("max_tilt_angle")

        self.l = L / math.sqrt(2)

        # state: x, y, z, Φ, θ, ψ, x_vel, y_vel, z_vel, p, q, r
        self.X = np.array([
            desired.x[0], desired.y[0], desired.z[0],
            0.0, 0.0, 0.0,
            desired.x_vel[0], desired.y_vel[0], desired.z_vel[0],
            0.0, 0.0, 0.0
        ])

        # propeller speed
        self.omega = np.array([0.0, 0.0, 0.0, 0.0])


    def advance_state(self, dt):
        """
        Simulate evolution of the vehicle state over time. Not needed when running on real drone,
        since we can get the values from the sensors.
        """
        euler_dot_lab = self.get_euler_derivatives()
        body_frame_angle_dot = self.body_angular_acceleration()
        accelerations = self.linear_acceleration()

        X_dot = np.array([self.X[6],               # x velocity
                        self.X[7],                 # y velocity
                        self.X[8],                 # z velocity
                        euler_dot_lab[0],          # phi velocity (world frame)
                        euler_dot_lab[1],          # theta velocity (world frame)
                        euler_dot_lab[2],          # psi velocity (world frame)
                        accelerations[0],          # x acceleration
                        accelerations[1],          # y acceleration
                        accelerations[2],          # z acceleration
                        body_frame_angle_dot[0],   # p acceleration (p rate velocity)
                        body_frame_angle_dot[1],   # q acceleration (q rate velocity)
                        body_frame_angle_dot[2]])  # r acceleration (r rate velocity)

        self.X = self.X + X_dot * dt
        return self.X

    def set_propeller_speed(self, thrust_cmd, moment_cmd):
        c_bar = thrust_cmd
        p_bar = moment_cmd[0] / self.l
        q_bar = moment_cmd[1] / self.l
        r_bar = -moment_cmd[2] / self.kappa
        
        U = np.array([p_bar, q_bar, r_bar, c_bar])
        self.omega = Quadrotor.propeller_coeffs() @ U / 4

    
    def linear_acceleration(self):  # used for state update
        """
        Convert the thrust body frame to world frame , divide by the mass and add the gravity
        in order to have the linear acceleration x_acc, y_acc, z_acc in the world frame
        """
        R = self.R()
        g = np.array([0, 0, self.g]).T
        c = np.array([0, 0, -self.f_total]).T
        return g + np.matmul(R, c) / self.m
    
    def body_angular_acceleration(self):  # used for state update
        """Angular aceeleration in the body frame"""
        p_dot = (self.tau_x - self.r * self.q * (self.i_z - self.i_y)) / self.i_x
        q_dot = (self.tau_y - self.r * self.p * (self.i_x - self.i_z)) / self.i_y
        r_dot = (self.tau_z - self.q * self.p * (self.i_y - self.i_x)) / self.i_z
        
        return np.array([p_dot,q_dot,r_dot])
    
    def get_euler_derivatives(self):  # used for state update
        """Angular velocity in the world frame"""
        euler_rot_mat = np.array([
                [1, math.sin(self.phi) * math.tan(self.theta), math.cos(self.phi) * math.tan(self.theta)],
                [0, math.cos(self.phi), -math.sin(self.phi)],
                [0, math.sin(self.phi) / math.cos(self.theta), math.cos(self.phi) / math.cos(self.theta)]
            ])

        pqr = self.body_angular_velocity.T

        # Angular velocities in world frame
        phi_theta_psi_dot = np.matmul(euler_rot_mat, pqr)
        
        return phi_theta_psi_dot

    
    def R(self):
        """XYZ"""
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