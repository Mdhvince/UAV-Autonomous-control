import math
import numpy as np
import matplotlib.pyplot as plt


# We want to control the drone in the world frame BUT we get some sensore measurement from the IMU that are in the body frame.
# And our controls (especially the moments that we command) have a more intuitive interpretation in the body frame.

G = 9.81

class Quadrotor():
    
    def __init__(self):
        
        L = 0.566  # rotor to rotor distance
        self.kf = 1.0
        self.km = 1.0
        self.m = .5
        self.l = L / (2*math.sqrt(2)) # perpendicular distance to axes
        self.i_x = 0.1
        self.i_y = 0.1
        self.i_z = 0.2
        
        # x, y, z, Φ, θ, ψ, x_vel, y_vel, z_vel, p, q, r
        self.X=np.array([.0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0])
        self.omega = np.array([0.0, 0.0, 0.0, 0.0])
        
    
    def set_propeller_angular_velocities(self, c, u_bar_p, u_bar_q, u_bar_r):
        
        c = np.clip(c, -1.5*G, 2*G)

        u_bar_p = max(-1, u_bar_p)
        u_bar_q = max(-1, u_bar_q)
        u_bar_r = max(-1, u_bar_r)

        c_bar = -c * self.m / self.kf
        p_bar = self.i_x * u_bar_p / (self.kf * self.l)
        q_bar = self.i_y * u_bar_q / (self.kf * self.l)
        r_bar = self.i_z * u_bar_r /  self.km

        omega_4_sq = (c_bar + p_bar - q_bar - r_bar) / 4
        omega_3_sq = (c_bar - q_bar) / 2 - omega_4_sq
        omega_2_sq = (c_bar - p_bar) / 2 - omega_3_sq
        omega_1_sq =  c_bar - omega_2_sq - omega_3_sq - omega_4_sq
                
        self.omega[0] = -np.sqrt(omega_1_sq)
        self.omega[1] =  np.sqrt(omega_2_sq)
        self.omega[2] = -np.sqrt(omega_3_sq)
        self.omega[3] =  np.sqrt(omega_4_sq)
    

    def linear_acceleration(self):
        """
        Convert the thrust body frame to world frame , divide by the mass and add the gravity
        in order to have the linear acceleration x_acc, y_acc, z_acc in the world frame
        """
        R = self.R()
        g = np.array([0, 0, G]).T
        c = np.array([0, 0, -self.f_total]).T
        return g + np.matmul(R, c) / self.m
    

    def get_omega_dot(self):
        """Angular aceeleration in the body frame"""
        p_dot = (self.tau_x - self.r * self.q * (self.i_z - self.i_y)) / self.i_x
        q_dot = (self.tau_y - self.r * self.p * (self.i_x - self.i_z)) / self.i_y
        r_dot = (self.tau_z - self.q * self.p * (self.i_y - self.i_x)) / self.i_z

        return np.array([p_dot,q_dot,r_dot])
    

    def get_euler_derivatives(self):
        """Angular velocity in the world frame"""
        euler_rot_mat = np.array([
                [1, math.sin(self.phi) * math.tan(self.theta), math.cos(self.phi) * math.tan(self.theta)],
                [0, math.cos(self.phi), -math.sin(self.phi)],
                [0, math.sin(self.phi) / math.cos(self.theta), math.cos(self.phi) / math.cos(self.theta)]
            ])

        # Turn rates in the body frame
        pqr = np.array([self.p, self.q, self.r]).T

        # Rotational velocities in world frame
        phi_theta_psi_dot = np.matmul(euler_rot_mat, pqr)
        
        return phi_theta_psi_dot
    

    def advance_state(self, dt):
    
        euler_dot_lab = self.get_euler_derivatives()
        body_frame_angle_dot = self.get_omega_dot()
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
    

    def R(self):
        """
        To transform between body frame accelerations and world frame accelerations
        """
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
    

    @property
    def phi(self):
        return self.X[3]

    @property
    def theta(self):
        return self.X[4]

    @property
    def psi(self):
        return self.X[5]

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


class Controller():
    
    def __init__(
            self, kp_z=1.0, kd_z=1.0, kp_x=1.0, kd_x=1.0, kp_y=1.0, kd_y=1.0,
            kp_roll=1.0, kp_pitch=1.0, kp_yaw=1.0,
            kp_p=1.0,  kp_q=1.0, kp_r=1.0):
        
        self.kp_z = kp_z
        self.kd_z = kd_z
        self.kp_x = kp_x
        self.kd_x = kd_x
        self.kp_y = kp_y
        self.kd_y = kd_y
        self.kp_roll = kp_roll
        self.kp_pitch = kp_pitch
        self.kp_yaw = kp_yaw
        self.kp_p = kp_p
        self.kp_q = kp_q
        self.kp_r = kp_r
            

    # outer loop controller - Position Controller
    def altitude_controller(
            self, z_target, z_dot_target, z_dot_dot_target, z_actual, z_dot_actual, rot_mat):

        """
        Get the vertical linear acceleration u_1_bar(body) then convert it into the thrust c (body)

        Input:
            World frame variables:
                - z_target; z_dot_target; z_dot_dot_target  => From a trajectory planner

            Body frame variables:
                - z_actual, z_dot_actual, rot_mat
        
        Output:
            Body frame variables:
                - thrust along the z-body-axis: c
        """

        # compute the desired acceleration in the body frame
        error = z_target - z_actual
        error_dot = z_dot_target - z_dot_actual

        u_1_bar = Controller._pd(self.kp_z, self.kd_z, error, error_dot, z_dot_dot_target)
        u_1_bar = np.clip(u_1_bar, -1, 8)

        b_z = rot_mat[2, 2]
        c = (u_1_bar - G) / b_z

        return c

    def lateral_controller(
            self, x_target, x_dot_target, x_dot_dot_target, x_actual, x_dot_actual, y_target,
            y_dot_target, y_dot_dot_target, y_actual, y_dot_actual, c):
        
        """
        The lateral controller will use a PD controller to command target values for elements of
        the drone's rotation matrix. The drone generates lateral acceleration by changing the body
        orientation which results in non-zero thrust in the desired direction.

        Input:
            World frame variables:
                - x_target, x_dot_target, x_dot_dot_target  => From a trajectory planner
                - y_target, y_dot_target, y_dot_dot_target  => From a trajectory planner

            Body frame variables:
                - x_actual, x_dot_actual
                - y_actual, y_dot_actual
                - c
        
        Output:
            Body frame variables:
                b_x_c and b_y_c are the desired rotation matrix elements used to control
                the orientation of the drone and generate lateral acceleration.
        """

        x_err = x_target - x_actual
        x_err_dot = x_dot_target - x_dot_actual
        y_err = y_target - y_actual
        y_err_dot = y_dot_target - y_dot_actual

        # desired x, y accelerations in the body frame
        x_dot_dot_command = Controller._pd(self.kp_x, self.kd_x, x_err, x_err_dot, x_dot_dot_target)
        y_dot_dot_command = Controller._pd(self.kp_y, self.kd_y, y_err, y_err_dot, y_dot_dot_target)

        # by dividing by c we can control the orientation independently of the thrust, this allow
        # more precise control
        b_x_c = x_dot_dot_command / c
        b_y_c = y_dot_dot_command / c

        # # quadrotor limits
        # deg2rad = lambda x: x * np.pi / 180
        # min_angle, max_angle = deg2rad(-45), deg2rad(45)
        # b_x_c = np.clip(b_x_c, min_angle, max_angle)
        # b_y_c = np.clip(b_x_c, min_angle, deg2rad(45))
        return b_x_c, b_y_c
    

    # inner loop controller
    def attitude_controller(self, b_x_c_target, b_y_c_target, psi_target, psi_actual, p_actual, q_actual, r_actual, rot_mat):
        """
        The attitude controller consists of the roll-pitch controller, yaw controller, and body rate controller.
        """
        p_c, q_c = self._roll_pitch_controller(b_x_c_target, b_y_c_target, rot_mat)
        r_c = self._yaw_controller(psi_target, psi_actual)
        u_bar_p, u_bar_q, u_bar_r = self._body_rate_controller(p_c, q_c, r_c, p_actual, q_actual, r_actual)
        return u_bar_p, u_bar_q, u_bar_r

    def _roll_pitch_controller(self, b_x_c_target, b_y_c_target, rot_mat):
        """
        The roll-pitch controller is a P controller responsible for commanding the roll and pitch
        rates (angular velocities) p_c and q_c in the body frame.

        Input:

            Body frame variables:
                - b_x_c_target, b_y_c_target: desired rot matrix element
                that describe the desired orientation
                - rot_mat
        
        Output:
            Body frame variables:
                - p_c and q_c: desired angular velocities
        """
        
        p = lambda kp, error: kp * error
        b_x, b_y = rot_mat[0,2], rot_mat[1,2]
        
        # Desired angular velocities component of the rotation matrix
        b_x_commanded_dot = p(self.kp_roll, error=b_x_c_target - b_x)
        b_y_commanded_dot = p(self.kp_pitch, error=b_y_c_target - b_y)

        # transform the desired angular velocities b_x_commanded_dot and y (body frame)
        # to the roll and pitch rates p_c and q_c in the (body frame)
        rot_mat1 = np.array([
                [rot_mat[1, 0], -rot_mat[0, 0]],
                [rot_mat[1, 1], -rot_mat[0, 1]]
            ]) / rot_mat[2, 2]

        rot_rate = np.matmul(rot_mat1, np.array([b_x_commanded_dot, b_y_commanded_dot]).T)
        p_c = rot_rate[0]
        q_c = rot_rate[1]

        return p_c, q_c
    
    def _body_rate_controller(self, p_c, q_c, r_c, p_actual, q_actual, r_actual):
        """
        The commanded roll, pitch, and yaw are collected by the body rate controller, and they are
        translated into the desired angular accelerations along the axis in the body frame.

        Input:
            Body frame variables:
                - p_c and q_c: desired angular velocities
                - p_actual, q_actual, r_actual: actual angular velocities
        
        Output:
            Body frame variables:
                - u_bar_p, u_bar_q, u_bar_r: desired angular acceleration in the body frame
        
        """
        p = lambda kp, error: kp * error
        
        # desired angular acceleration in the body frame
        u_bar_p = p(self.kp_p, error=p_c - p_actual)
        u_bar_q = p(self.kp_q, error=q_c - q_actual)
        u_bar_r = p(self.kp_r, error=r_c - r_actual)

        return u_bar_p, u_bar_q, u_bar_r
    
    def _yaw_controller(self, psi_target, psi_actual):
        """
        The commanded roll, pitch, and yaw are collected by the body rate controller, and they are
        translated into the desired angular accelerations along the axis in the body frame.

        Input:
            World frame variables:
                - psi_target and psi_actual: desired and actual yaw angles in the world frame
        
        Output:
            Body frame variables:
                - r_c: desired angular velocity in the body frame
        
        """
        p = lambda kp, error: kp * error
        r_c = p(self.kp_yaw, error=psi_target - psi_actual)
        return r_c
    

    @staticmethod
    def _pd(kp, kd, error, error_dot, target):
            # Proportional and differential control terms
            p_term = kp * error
            d_term = kd * error_dot
            # Control command (with feed-forward term)
            return p_term + d_term + target


def get_path():
    total_time = 20.0
    dt = 0.01
    t = np.linspace(0.0, total_time, int(total_time/dt))

    omega_x = 0.8
    omega_y = 0.4
    omega_z = 0.4

    a_x = 1.0 
    a_y = 1.0
    a_z = 1.0

    x_path         = a_x * np.sin(omega_x * t) 
    x_dot_path     = a_x * omega_x * np.cos(omega_x * t)
    x_dot_dot_path = -a_x * omega_x**2 * np.sin(omega_x * t)

    y_path         = a_y * np.cos(omega_y * t)
    y_dot_path     = -a_y * omega_y * np.sin(omega_y * t)
    y_dot_dot_path = -a_y * omega_y**2 * np.cos(omega_y * t)

    z_path         = a_z * np.cos(omega_z * t)
    z_dot_path     = -a_z * omega_z * np.sin(omega_z * t)
    z_dot_dot_path = - a_z * omega_z**2 * np.cos(omega_z * t)

    psi_path = np.arctan2(y_dot_path,x_dot_path)
    return dt, x_path, y_path, z_path, x_dot_path, y_dot_path, z_dot_path, x_dot_dot_path, y_dot_dot_path, z_dot_dot_path, psi_path


if __name__ == "__main__":
    #  ------------ TRAJECTORY PLANNER ------------ #
    (
        dt, 
        x_path, y_path, z_path,
        x_dot_path, y_dot_path, z_dot_path,
        x_dot_dot_path, y_dot_dot_path, z_dot_dot_path,
        psi_path
    ) = get_path()


    #  ------------ FLIGHT ------------ #

    # how fast the inner loop (Attitude controller) performs calculations 
    # relative to the outer loops (altitude and position controllers).
    inner_loop_relative_to_outer_loop = 10

    quad = Quadrotor()

    control_system = Controller(
        kp_z=4.0, kd_z=1.5, kp_x=6.0, kd_x=4.0, kp_y=6.0, kd_y=4.0,
        kp_roll=8.0, kp_pitch=8.0, kp_yaw=4.5,
        kp_p=20.0, kp_q=20.0, kp_r=5.0)
    
    # declaring the initial state of the drone with zero
    # height and zero velocity 
    quad.X = np.array([
        x_path[0], y_path[0], z_path[0],
        0.0, 0.0, psi_path[0],
        x_dot_path[0], y_dot_path[0], z_dot_path[0],
        0.0, 0.0, 0.0
    ])
    
    # arrays for recording the state history, 
    # propeller angular velocities and linear accelerations
    drone_state_history = quad.X
    omega_history = quad.omega
    accelerations = quad.linear_acceleration()
    accelerations_history= accelerations
    angular_vel_history = quad.get_euler_derivatives()


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    n_waypoints = z_path.shape[0]

    for i in range(0, n_waypoints):

        rot_mat = quad.R()

        c = control_system.altitude_controller(
                z_path[i], z_dot_path[i], z_dot_dot_path[i], quad.X[2], quad.X[8], rot_mat)
        
        b_x_c, b_y_c = control_system.lateral_controller(
            x_path[i], x_dot_path[i], x_dot_dot_path[i], quad.X[0], quad.X[6],
            y_path[i], y_dot_path[i], y_dot_dot_path[i], quad.X[1], quad.X[7], c)
        
        for _ in range(inner_loop_relative_to_outer_loop):
            rot_mat = quad.R()
            
            # get angular velocities in the world frame (phi_dot, theta_dot, psi_dot)
            angular_vel = quad.get_euler_derivatives()
            
            u_bar_p, u_bar_q, u_bar_r = control_system.attitude_controller(
                b_x_c, b_y_c, psi_path[i], quad.psi, quad.X[9], quad.X[10], quad.X[11], rot_mat)

            quad.set_propeller_angular_velocities(c, u_bar_p, u_bar_q, u_bar_r)
            quad_state = quad.advance_state(dt/inner_loop_relative_to_outer_loop)
        

        # generating a history of the state history, propeller angular velocities and linear accelerations
        drone_state_history = np.vstack((drone_state_history, quad_state))
        
        omega_history = np.vstack((omega_history, quad.omega))
        accelerations = quad.linear_acceleration()
        accelerations_history= np.vstack((accelerations_history, accelerations))
        angular_vel_history = np.vstack((angular_vel_history, quad.get_euler_derivatives()))

        origin = np.array([quad.X[0], quad.X[1], quad.X[2]])
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])

        # Transform the body-fixed axes into the world frame using the rotation matrix
        x_axis_world = np.dot(rot_mat, x_axis)
        y_axis_world = np.dot(rot_mat, y_axis)
        z_axis_world = np.dot(rot_mat, z_axis)

        # Plot the x, y, and z axes in the world frame
        ax.clear()
        ax.plot3D([origin[0], x_axis_world[0]], [origin[1], x_axis_world[1]], [origin[2], x_axis_world[2]], 'r')
        ax.plot3D([origin[0], y_axis_world[0]], [origin[1], y_axis_world[1]], [origin[2], y_axis_world[2]], 'g')
        ax.plot3D([origin[0], z_axis_world[0]], [origin[1], z_axis_world[1]], [origin[2], z_axis_world[2]], 'b')

        ax.plot(x_path, y_path, z_path, linestyle='-', marker='.',color='red')

        ax.set_xlabel('$x$ [$m$]').set_fontsize(12)
        ax.set_ylabel('$y$ [$m$]').set_fontsize(12)
        ax.set_zlabel('$z$ [$m$]').set_fontsize(12)
        plt.legend(['Executed path'],fontsize=10)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)

        plt.pause(.001)

    plt.show()    

    # ax.plot(x_path, y_path, z_path,linestyle='-',marker='.',color='red')
    # ax.plot(drone_state_history[:,0],
    #         drone_state_history[:,1],
    #         drone_state_history[:,2],
    #         linestyle='-',color='blue')

    # plt.title('Flight path').set_fontsize(20)
    # ax.set_xlabel('$x$ [$m$]').set_fontsize(20)
    # ax.set_ylabel('$y$ [$m$]').set_fontsize(20)
    # ax.set_zlabel('$z$ [$m$]').set_fontsize(20)
    # plt.legend(['Planned path','Executed path'],fontsize = 14)

    # ax.set_xlim(-2, 2)
    # ax.set_ylim(-2, 2)
    # ax.set_zlim(-2, 2)

    # plt.show()



    




