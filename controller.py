import numpy as np


G = 9.81


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
            
    
    def altitude_controller(
            self, z_target, z_dot_target, z_dot_dot_target, rot_mat, quad):

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
        error = z_target - quad.z
        error_dot = z_dot_target - quad.z_vel

        u_1_bar = Controller._pd(self.kp_z, self.kd_z, error, error_dot, z_dot_dot_target)

        b_z = rot_mat[2, 2]
        c = (u_1_bar - G) / b_z
        return c

    def lateral_controller(
            self, x_target, x_dot_target, x_dot_dot_target, y_target,
            y_dot_target, y_dot_dot_target, c, quad):
        
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

        kp_xy = np.array([self.kp_x, self.kp_y])
        kd_xy = np.array([self.kd_x, self.kd_y])
         
        pos_actual = np.array([quad.x, quad.y])
        pos_target = np.array([x_target, y_target])
        vel_actual = np.array([quad.x_vel, quad.y_vel])
        vel_target = np.array([x_dot_target, y_dot_target])
        ff_acc = np.array([x_dot_dot_target, y_dot_dot_target])

        pos_err = pos_target - pos_actual
        vel_err = vel_target - vel_actual
        acc_cmd = kp_xy * pos_err + kd_xy * vel_err + ff_acc

        # by dividing by c we can control the orientation independently of the thrust, this allow
        # more precise control
        bxy_cmd = acc_cmd / c

        return bxy_cmd
    

    # inner loop controller
    def attitude_controller(self, bxy_cmd, psi_target, rot_mat, quad):
        """
        The attitude controller consists of the roll-pitch controller, yaw controller, and body rate controller.
        """
        pq_cmd = self._roll_pitch_controller(bxy_cmd, rot_mat)
        r_c = self._yaw_controller(psi_target, quad)
        pqr_cmd = np.append(pq_cmd, r_c)

        ubar_pqr = self._body_rate_controller(pqr_cmd, quad)
        return ubar_pqr

    def _roll_pitch_controller(self, bxy_cmd, rot_mat):
        """
        The roll-pitch controller is a P controller responsible for commanding the roll and pitch
        rates (angular velocities) p_c and q_c in the body frame.

        Input:

            Body frame variables:
                - bxy_cmd: desired rot matrix element
                that describe the desired orientation
                - rot_mat
        
        Output:
            Body frame variables:
                - p_c and q_c: desired angular velocities
        """
        
        kps = np.array([self.kp_roll, self.kp_pitch])
        b_xy = np.array([rot_mat[0, 2], rot_mat[1,2]])
        errors = bxy_cmd - b_xy
        
        b_xy_cmd_dot = kps * errors  # Desired angular velocities component of the rotation matrix

        # transform the desired angular velocities b_xy_cmd_dot (body frame)
        # to the roll and pitch rates p_c and q_c in the (body frame)
        rot_mat1 = np.array([
                [rot_mat[1, 0], -rot_mat[0, 0]],
                [rot_mat[1, 1], -rot_mat[0, 1]]
            ]) / rot_mat[2, 2]

        pq_cmd = np.matmul(rot_mat1, b_xy_cmd_dot.T)  # ratation rate [p_c, q_c]

        return pq_cmd
    
    def _body_rate_controller(self, pqr_cmd, quad):
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

        kp_pqr = np.array([self.kp_p, self.kp_q, self.kp_r])
        pqr_actual = np.array([quad.p, quad.q, quad.r])
        ubar_pqr = kp_pqr * (pqr_cmd - pqr_actual)  # desired angular acceleration in the body frame
        return ubar_pqr
    
    def _yaw_controller(self, psi_target, quad):
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

        r_c = self.kp_yaw * (psi_target - quad.psi)
        return r_c
    

    @staticmethod
    def _pd(kp, kd, error, error_dot, target):
        # Proportional and differential control terms
        p_term = kp * error
        d_term = kd * error_dot
        # Control command (with feed-forward term)
        return p_term + d_term + target
    

if __name__ == "__main__":
    pass