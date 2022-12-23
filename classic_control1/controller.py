import math

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
    

    ######### POSITION #########
    def altitude(self, quad, desired, dt, index):
        """
        """
        z_des = desired.z[index]
        z_dot_des = desired.z_vel[index]
        z_dot_dot_des = desired.z_acc[index]

        error = z_des - quad.z
        error_dot = z_dot_des - quad.z_vel
        self.integral_error += error * dt;

        u_1_bar = Controller._pid(
                self.kp_z, self.kd_z, self.ki_z, error, error_dot, self.integral_error,  z_dot_dot_des)

        # Project the acceleration along the z-vector of the body (Bz)
        rot_mat = quad.R()
        b_z = rot_mat[2, 2]
        acc = (u_1_bar - self.g) / b_z
        acc = np.clip(acc, -quad.max_ascent_rate/dt, quad.max_descent_rate/dt)

        thrust_cmd = -quad.m * acc
        thrust_cmd = np.clip(thrust_cmd, quad.min_thrust, quad.max_thrust)

        # reserve some thrust margin for angle control
        thrust_margin = 0.1 * (quad.max_thrust - quad.min_thrust)
        thrust_cmd = np.clip(
            thrust_cmd, (quad.min_thrust + thrust_margin) * 4, (quad.max_thrust-thrust_margin) * 4)

        return thrust_cmd
    
    def lateral(self, quad, desired, index):
        """
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
        acc_cmd = Controller._pd(self.lateral_Pgain, self.lateral_Dgain, pos_err, vel_err, ff_acc)

        # Scaling down the magnitude acceleration vector
        acc_mag = np.linalg.norm(acc_cmd)
        acc_is_too_high = acc_mag > quad.max_horiz_accel
        acc_cmd = (acc_cmd / acc_mag) * quad.max_horiz_accel if acc_is_too_high else acc_cmd
            
        return acc_cmd
    

    ######### ATTITUDE #########
    def attitude(self, quad, thrust_cmd, acc_cmd, desired_yaw):
        rot_mat = quad.R()
        pq_cmd = self.roll_pitch_controller(acc_cmd, thrust_cmd, rot_mat, quad)
        r_cmd = self.yaw_controller(desired_yaw, quad)
        pqr_cmd = np.append(pq_cmd, r_cmd)
        moment_cmd = self.body_rate_controller(pqr_cmd, quad)
        return moment_cmd
        
    def roll_pitch_controller(self, acc_cmd, thrust_cmd, rot_mat, quad):

        # Scale acc_cmd to a value appropriate for the current thrust being applied
        c = -thrust_cmd / quad.m
        scaled_acc_cmd = acc_cmd / c

        bxy_cmd = np.clip(scaled_acc_cmd, -quad.max_tilt_angle, quad.max_tilt_angle)
        b_xy = np.array([rot_mat[0, 2], rot_mat[1, 2]])
        errors = bxy_cmd - b_xy

        # Desired angular velocities component of the rotation matrix
        b_xy_cmd_dot = self.roll_pitch_Pgain * errors

        # transform the desired angular velocities component of R: b_xy_cmd_dot (body frame)
        # to the roll and pitch rates p_c and q_c in the (body frame)
        rot_mat1 = np.array([
                [rot_mat[1, 0], -rot_mat[0, 0]],
                [rot_mat[1, 1], -rot_mat[0, 1]]
            ]) / rot_mat[2, 2]
        
        pq_cmd = np.matmul(rot_mat1, b_xy_cmd_dot.T)  # rotation rate [p_c, q_c]

        return pq_cmd
    
    def yaw_controller(self, psi_des, quad):
        psi_des = Controller.wrap_to_2pi(psi_des)
        yaw_err = Controller.wrap_to_pi(psi_des - quad.psi)
        r_c = self.kp_yaw * yaw_err
        return r_c
    
    def body_rate_controller(self, pqr_cmd, quad):
        MOI = np.array([quad.i_x, quad.i_y, quad.i_z])  # moment of inertia
        kp_pqr = np.array([self.kp_p, self.kp_q, self.kp_r])
        pqr_actual = np.array([quad.p, quad.q, quad.r])

        moment_cmd = MOI * kp_pqr * (pqr_cmd - pqr_actual)
        return moment_cmd
    
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