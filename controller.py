import numpy as np

# We want to control the drone in the world frame BUT we get some sensore measurement from the IMU that are in the body frame.
# And our controls (especially the moments that we command) have a more intuitive interpretation in the body frame.

class Controller():
    
    def __init__(self, config):
        self.g = config["DEFAULT"].getfloat("g")

        controller = config["CONTROLLER"]

        self.kp_z = controller.getfloat("kp_z")
        self.kd_z = controller.getfloat("kd_z")
        self.kp_x = controller.getfloat("kp_x")
        self.kd_x = controller.getfloat("kd_x")
        self.kp_y = controller.getfloat("kp_y")
        self.kd_y = controller.getfloat("kd_y")
        self.ki_z = controller.getfloat("ki_z")
        self.kp_roll = controller.getfloat("kp_roll")
        self.kp_pitch = controller.getfloat("kp_pitch")
        self.kp_yaw = controller.getfloat("kp_yaw")
        self.kp_p = controller.getfloat("kp_p")
        self.kp_q = controller.getfloat("kp_q")
        self.kp_r = controller.getfloat("kp_r")

        self.integral_error = 0
    
    def altitude_controller(self, z_target, z_dot_target, z_dot_dot_target, rot_mat, quad, dt):
        """
        Inputs:
            - Desired Z pos, vel and acc from the trajectory planner [m], [m/s], [m/s2]
            - rot_mat: Actual vehicle orientation in the body frame
            - quad: vehicle object
            - dt: time step of measurement [s]
        Output:
            - Desired thrust [N]
        """
        error = z_target - quad.z
        error_dot = z_dot_target - quad.z_vel
        self.integral_error += error * dt;

        u_1_bar = Controller._pid(
                self.kp_z, self.kd_z, self.ki_z, error, error_dot, self.integral_error,  z_dot_dot_target)

        b_z = rot_mat[2, 2]
        acc = (u_1_bar - self.g) / b_z
        acc = np.clip(acc, -quad.max_ascent_rate/dt, quad.max_descent_rate/dt)

        thrust_cmd = -quad.m * acc

        return thrust_cmd
    
    def lateral_controller(
            self, x_target, x_dot_target, x_dot_dot_target, y_target,
            y_dot_target, y_dot_dot_target, quad):
        """
        Inputs:
            - Desired XY pos, vel and acc from the trajectory planner [m], [m/s], [m/s2]
            - quad: vehicle object
        Output:
            - Desired horizontal acceleration [m/s2]
        """
        pos_actual = np.array([quad.x, quad.y])
        pos_target = np.array([x_target, y_target])
        vel_actual = np.array([quad.x_vel, quad.y_vel])
        vel_target = np.array([x_dot_target, y_dot_target])
        ff_acc = np.array([x_dot_dot_target, y_dot_dot_target])

        vel_mag = np.linalg.norm(vel_target)  # scalar representing the target speed [m/s]

        if(vel_mag > quad.max_speed_xy):
            vel_target = (vel_target / vel_mag) * quad.max_speed_xy
        vel_err = vel_target - vel_actual

        kp_xy = np.array([self.kp_x, self.kp_y])
        kd_xy = np.array([self.kd_x, self.kd_y])
        pos_err = pos_target - pos_actual
        acc_cmd = Controller._pd(kp_xy, kd_xy, pos_err, vel_err, ff_acc)

        acc_mag = np.linalg.norm(acc_cmd)
        if acc_mag > quad.max_horiz_accel:
            acc_cmd = (acc_cmd / acc_mag) * quad.max_horiz_accel
    
        return acc_cmd
    
    def roll_pitch_controller(self, acc_cmd, thrust_cmd, rot_mat, quad):
        
        pq_cmd = np.array([0.0, 0.0])

        if thrust_cmd > 0:
            c = -thrust_cmd / quad.m
            
            bx_c_target, by_y_target = np.clip(acc_cmd/c, -quad.max_tilt_angle, quad.max_tilt_angle)
        
            kps = np.array([self.kp_roll, self.kp_pitch])
            b_xy = np.array([rot_mat[0, 2], rot_mat[1,2]])
            bxy_cmd = np.array([bx_c_target, by_y_target])
            errors = bxy_cmd - b_xy
        
            b_xy_cmd_dot = kps * errors  # Desired angular velocities component of the rotation matrix

            # transform the desired angular velocities b_xy_cmd_dot (body frame)
            # to the roll and pitch rates p_c and q_c in the (body frame)
            rot_mat1 = np.array([
                    [rot_mat[1, 0], -rot_mat[0, 0]],
                    [rot_mat[1, 1], -rot_mat[0, 1]]
                ]) / rot_mat[2, 2]
            
            pq_cmd = np.matmul(rot_mat1, b_xy_cmd_dot.T)  # rotation rate [p_c, q_c]

        return pq_cmd
    
    def body_rate_controller(self, pqr_cmd, quad):

        MOI = np.array([quad.i_x, quad.i_y, quad.i_z])  # moment of inertia

        kp_pqr = np.array([self.kp_p, self.kp_q, self.kp_r])
        pqr_actual = np.array([quad.p, quad.q, quad.r])
        error = pqr_cmd - pqr_actual

        moment_cmd = kp_pqr * error
        moment_cmd = MOI * moment_cmd

        return moment_cmd
    
    def yaw_controller(self, psi_target, quad):
        psi_target = Controller.wrap_to_2pi(psi_target)
        yaw_err = Controller.wrap_to_pi(psi_target - quad.psi)
        r_c = self.kp_yaw * yaw_err
        return r_c
    
            

    @staticmethod
    def wrap_to_pi(angle):
        """maps an angle to the range (-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def wrap_to_2pi(angle):
        """maps an angle to the range [0, 2*pi)"""
        if angle > 0:
            return np.fmod(angle, 2 * np.pi)
        else:
            return np.fmod(angle, -2 * np.pi)
    
    @staticmethod
    def _pd(kp, kd, error, error_dot, target):
        p_term = kp * error
        d_term = kd * error_dot
        return p_term + d_term + target
    
    @staticmethod
    def _pid(kp, kd, ki, error, error_dot, i_error,  target):
        p_term = kp * error
        d_term = kd * error_dot
        i_term = ki * i_error
        return p_term + i_term + d_term + target
    

if __name__ == "__main__":
    pass