import configparser
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import utils
from control.quadrotor import Quadrotor


class CascadedController:
    """
    Cascaded controller from the paper: https://www.dynsyslab.org/wp-content/papercite-data/pdf/lupashin-mech14.pdf
    """
    def __init__(self):
        cfg, _, _, _, cfg_controller = utils.get_config()

        self.g = cfg.getfloat("g")
        self.dt = cfg.getfloat("dt")

        self.kp_z = cfg_controller.getfloat("kp_z")
        self.kd_z = cfg_controller.getfloat("kd_z")

        self.kp_x = cfg_controller.getfloat("kp_xy")
        self.kd_x = cfg_controller.getfloat("kd_xy")

        self.kp_y = cfg_controller.getfloat("kp_xy")
        self.kd_y = cfg_controller.getfloat("kd_xy")

        self.ki_z = cfg_controller.getfloat("ki_z")
        self.kp_roll = cfg_controller.getfloat("kp_roll")
        self.kp_pitch = cfg_controller.getfloat("kp_pitch")
        self.kp_yaw = cfg_controller.getfloat("kp_yaw")
        self.kp_p = cfg_controller.getfloat("kp_p")
        self.kp_q = cfg_controller.getfloat("kp_q")
        self.kp_r = cfg_controller.getfloat("kp_r")

        self.integral_error = 0


    def altitude(self, quad, des_z, rot_mat):
        """
        Compute the desired thrust command.
        :param quad: The quadrotor object
        :param des_z: 1D array of desired z position, velocity and acceleration [z, z_dot, z_ddot]
        :param rot_mat: The rotation matrix of the quadrotor
        """
        
        error = des_z[0] - quad.z
        error_dot = des_z[1] - quad.z_vel
        self.integral_error += error * self.dt

        acc_z = CascadedController._pid(
            self.kp_z, self.kd_z, self.ki_z, error, error_dot, self.integral_error, des_z[2]) - self.g

        # Project the acceleration along the z-vector of the body (Bz)
        b_z = rot_mat[2, 2]
        acc_z = acc_z / b_z
        acc_z = np.clip(acc_z, -quad.max_ascent_rate/self.dt, quad.max_descent_rate/self.dt)

        c = -quad.m * acc_z

        # reserve some thrust margin for angle control
        thrust_margin = 0.2 * (quad.max_thrust - quad.min_thrust)
        c = np.clip(c, (quad.min_thrust + thrust_margin) * 4, (quad.max_thrust-thrust_margin) * 4)

        return c
    
    def lateral(self, quad, des_x, des_y, thrust_cmd):
        """
        Compute the desired roll and pitch angles.
        :param quad: The quadrotor object
        :param des_x: 1D array of desired x position, velocity and acceleration [x, x_dot, x_ddot]
        :param des_y: 1D array of desired y position, velocity and acceleration [y, y_dot, y_ddot]
        :param thrust_cmd: The thrust command coming from the altitude controller
        :param index: The current index of the trajectory
        """
        x_des, y_des = des_x[0], des_y[0]
        x_dot_des, y_dot_des = des_x[1], des_y[1]
        x_dot_dot_des, y_dot_dot_des = des_x[2], des_y[2]
        
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
        """
        Put together the roll and pitch and yaw rate commands.
        :param quad: The quadrotor object
        :param bxy_cmd: The roll and pitch commands coming from the lateral controller
        :param psi_des: The desired yaw angle
        :param rot_mat: The rotation matrix of the quadrotor
        """
        pq_c = self.roll_pitch_controller(bxy_cmd, rot_mat)
        r_c = self.yaw_controller(quad, psi_des)
        pqr_cmd = np.append(pq_c, r_c)
        return pqr_cmd

    def body_rate_controller(self, quad, pqr_cmd):
        """
        Compute the desired moments.
        :param quad: The quadrotor object
        :param pqr_cmd: The desired roll, pitch and yaw rates coming from the reduced attitude controller
        """
        I = np.array([quad.i_x, quad.i_y, quad.i_z])  # moment of inertia
        kp_pqr = np.array([self.kp_p, self.kp_q, self.kp_r])
        pqr_actual = np.array([quad.p, quad.q, quad.r])

        moment_cmd = I * kp_pqr * (pqr_cmd - pqr_actual)

        moment_mag = np.linalg.norm(moment_cmd)
        if moment_mag > quad.max_torque:
            moment_cmd = moment_cmd * quad.max_torque / moment_mag
        return moment_cmd    
    
    def roll_pitch_controller(self, bxy_cmd, rot_mat):
        """
        Compute the desired roll and pitch rates.
        :param bxy_cmd: The desired acceleration in the body frame coming from the lateral controller
        :param rot_mat: The rotation matrix of the quadrotor
        """
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
        """
        Compute the desired yaw rate.
        :param quad: The quadrotor object
        :param psi_des: The desired yaw angle
        """
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

    @staticmethod
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
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config_file = Path("/config.ini")
    config.read(config_file)
    cfg = config["DEFAULT"]
    frequency = cfg.getint("frequency")

    ctrl = CascadedController(config)
    quad = Quadrotor(config)
    state_history, omega_history = quad.X, quad.omega

    # Test the controller response for a desired position and velocity in the z direction
    des_x = np.array([5, 2, 0])
    des_y = np.array([12, 2, 0])
    des_z = np.array([10, 2, 0])
    des_yaw = 0

    for _ in range(1500):
        R = quad.R()
        F_cmd = ctrl.altitude(quad, des_z, R)
        bxy_cmd = ctrl.lateral(quad, des_x, des_y, F_cmd)
        pqr_cmd = ctrl.reduced_attitude(quad, bxy_cmd, des_yaw, R)

        for _ in range(frequency):
            # flight controller
            moment_cmd = ctrl.body_rate_controller(quad, pqr_cmd)
            quad.set_propeller_speed(F_cmd, moment_cmd)
            quad.update_state()

        state_history = np.vstack((state_history, quad.X))
        omega_history = np.vstack((omega_history, quad.omega))


    # plot controller response for the x, y, z direction
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(state_history[:, 0], label="x")
    plt.plot(des_x[0] * np.ones_like(state_history[:, 0]), label="desired x")
    # plt.ylim([-1, 15])

    plt.subplot(3, 1, 2)
    plt.plot(state_history[:, 1], label="y")
    plt.plot(des_y[0] * np.ones_like(state_history[:, 1]), label="desired y")
    # plt.ylim([-1, 15])

    plt.subplot(3, 1, 3)
    plt.plot(state_history[:, 2], label="z")
    plt.plot(des_z[0] * np.ones_like(state_history[:, 2]), label="desired z")
    # plt.ylim([-1, 15])


    plt.show()


