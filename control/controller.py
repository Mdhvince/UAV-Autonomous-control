
import numpy as np


class CascadedController:
    """
    Cascaded controller from the paper: https://www.dynsyslab.org/wp-content/papercite-data/pdf/lupashin-mech14.pdf
    """
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


    def altitude(self, quad, desired, rot_mat, dt, index):
        """
        Output:
            - c: collective thrust
        """
        
        error = desired.z[index] - quad.z
        error_dot = desired.z_vel[index] - quad.z_vel
        self.integral_error += error * dt;

        acc_z = CascadedController._pid(self.kp_z, self.kd_z, self.ki_z, error, error_dot, self.integral_error, desired.z_acc[index]) - self.g

        # Project the acceleration along the z-vector of the body (Bz)
        b_z = rot_mat[2, 2]
        acc_z = acc_z / b_z
        acc_z = np.clip(acc_z, -quad.max_ascent_rate/dt, quad.max_descent_rate/dt)

        c = -quad.m * acc_z

        # reserve some thrust margin for angle control
        thrust_margin = 0.2 * (quad.max_thrust - quad.min_thrust)
        c = np.clip(c, (quad.min_thrust + thrust_margin) * 4, (quad.max_thrust-thrust_margin) * 4)

        return c
    
    def lateral(self, quad, thrust_cmd, desired, index):
        """
        Output:
            - Commanded rotation matrix [bx_c, by_c]
        """
        x_des, y_des = desired.x[index], desired.y[index]
        x_dot_des, y_dot_des = desired.x_vel[index], desired.y_vel[index]
        x_dot_dot_des, y_dot_dot_des = desired.x_acc[index], desired.y_acc[index]
        
        pos_des = np.array([x_des, y_des])
        vel_des = np.array([x_dot_des, y_dot_des])
        ff_acc = np.array([x_dot_dot_des, y_dot_dot_des])

        # Scaling down the magnitude velocity vector
        vel_mag = np.linalg.norm(vel_des)
        vel_is_too_high = vel_mag > quad.max_speed_xy
        vel_des = (vel_des / vel_mag) * quad.max_speed_xy if vel_is_too_high else vel_des
        
        # Get required acceleration
        vel_err = vel_des - quad.velocity[:2]
        pos_err = pos_des - quad.position[:2]
        acc_cmd = CascadedController._pd(self.lateral_Pgain, self.lateral_Dgain, pos_err, vel_err, ff_acc)

        # Scaling down the magnitude acceleration vector
        acc_mag = np.linalg.norm(acc_cmd)
        acc_is_too_high = acc_mag > quad.max_horiz_accel
        acc_cmd = (acc_cmd / acc_mag) * quad.max_horiz_accel if acc_is_too_high else acc_cmd

        # Scale acc_cmd to a value appropriate for the current thrust being applied
        acc_z = -thrust_cmd / quad.m
        scaled_acc_cmd = acc_cmd / acc_z
        bxy_cmd = np.clip(scaled_acc_cmd, -quad.max_tilt_angle, quad.max_tilt_angle)
        return bxy_cmd
    
    def reduced_attitude(self, quad, bxy_cmd, psi_des, rot_mat):
        pq_c = self.roll_pitch_controller(bxy_cmd, rot_mat)
        r_c = self.yaw_controller(quad, psi_des)
        pqr_cmd = np.append(pq_c, r_c)
        return pqr_cmd

    def body_rate_controller(self, quad, pqr_cmd):
        I = np.array([quad.i_x, quad.i_y, quad.i_z])  # moment of inertia
        kp_pqr = np.array([self.kp_p, self.kp_q, self.kp_r])
        pqr_actual = np.array([quad.p, quad.q, quad.r])

        moment_cmd = I * kp_pqr * (pqr_cmd - pqr_actual)

        moment_mag = np.linalg.norm(moment_cmd)
        if moment_mag > quad.max_torque:
            moment_cmd = moment_cmd * quad.max_torque / moment_mag
        return moment_cmd    
    
    def roll_pitch_controller(self, bxy_cmd, rot_mat):

        b_xy = np.array([rot_mat[0, 2], rot_mat[1, 2]])
        errors = bxy_cmd - b_xy

        # Desired angular velocities component of the rotation matrix
        b_xy_cmd_dot = self.roll_pitch_Pgain * errors

        # transform the desired angular velocities component of R: b_xy_cmd_dot (body frame)
        # to the roll and pitch rates p_c and q_c in the (body frame)
        rot_mat1 = np.array([[rot_mat[1, 0], -rot_mat[0, 0]],
                             [rot_mat[1, 1], -rot_mat[0, 1]]]) / rot_mat[2, 2]
        
        pq_cmd = np.matmul(rot_mat1, b_xy_cmd_dot.T)

        return pq_cmd
    
    def yaw_controller(self, quad, psi_des):
        psi_des = CascadedController.wrap_to_2pi(psi_des)
        yaw_err = CascadedController.wrap_to_pi(psi_des - quad.psi)
        r_c = self.kp_yaw * yaw_err
        return r_c
     
    @property
    def lateral_Pgain(self):
        return np.array([self.kp_x, self.kp_y])
    
    @property
    def lateral_Dgain(self):
        return np.array([self.kd_x, self.kd_y])
    
    @property
    def roll_pitch_Pgain(self):
        return np.array([self.kp_roll, self.kp_pitch])

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
    def _pd(kp, kd, error, error_dot, des):
        p_term = kp * error
        d_term = kd * error_dot
        return p_term + d_term + des
    
    @staticmethod
    def _pid(kp, kd, ki, error, error_dot, i_error,  des):
        p_term = kp * error
        d_term = kd * error_dot
        i_term = ki * i_error
        return p_term + i_term + d_term + des
    

if __name__ == "__main__":
    pass