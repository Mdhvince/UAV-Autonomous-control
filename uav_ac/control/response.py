import numpy as np
import matplotlib.pyplot as plt


def quaternion_to_euler(quaternion):
    """
    Extract roll, pitch, yaw from quaternion [q0, q1, q2, q3] where q0 is scalar part.
    :param quaternion: numpy array [q0, q1, q2, q3]
    :return: tuple (roll, pitch, yaw) in radians
    """
    q = quaternion
    
    # Roll (phi) = atan2(2(q0*q1 + q2*q3), 1 - 2(q1^2 + q2^2))
    roll = np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 
                      1 - 2 * (q[1]**2 + q[2]**2))
    
    # Pitch (theta) = asin(2(q0*q2 - q3*q1))
    pitch = np.arcsin(2 * (q[0] * q[2] - q[3] * q[1]))
    
    # Yaw (psi) = atan2(2(q0*q3 + q1*q2), 1 - 2(q2^2 + q3^2))
    yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 
                     1 - 2 * (q[2]**2 + q[3]**2))
    
    return roll, pitch, yaw


def plot_controller_response(state_history, desired_history, dt):
    """
    Plot controller performance with 6 subplots showing absolute error over time.
    
    :param state_history: numpy array of shape (N, 13) containing [x, y, z, q0, q1, q2, q3, x_dot, y_dot, z_dot, p, q, r]
    :param desired_history: numpy array of shape (N, 10) containing [x, y, z, x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot, yaw]
    :param dt: time step in seconds
    """
    
    n_steps = state_history.shape[0]
    time = np.arange(n_steps) * dt
    
    # Extract actual values
    actual_x = state_history[:, 0]
    actual_y = state_history[:, 1]
    actual_z = state_history[:, 2]
    
    # Extract desired values
    desired_x = desired_history[:, 0]
    desired_y = desired_history[:, 1]
    desired_z = desired_history[:, 2]
    desired_yaw = desired_history[:, 9]
    
    # Extract euler angles from quaternions
    actual_roll = np.zeros(n_steps)
    actual_pitch = np.zeros(n_steps)
    actual_yaw = np.zeros(n_steps)
    
    for i in range(n_steps):
        quaternion = state_history[i, 3:7]
        actual_roll[i], actual_pitch[i], actual_yaw[i] = quaternion_to_euler(quaternion)
    
    # For desired roll and pitch, we need to compute them from the desired accelerations
    # In the cascaded controller, roll and pitch are computed from lateral accelerations
    # For simplicity, we'll set desired roll and pitch to 0 (hover condition)
    desired_roll = np.zeros(n_steps)
    desired_pitch = np.zeros(n_steps)
    
    # Compute errors (positive and negative)
    # Convert position errors from meters to centimeters
    error_x = (desired_x - actual_x) * 100
    error_y = (desired_y - actual_y) * 100
    error_z = (desired_z - actual_z) * 100
    # Convert angle errors from radians to degrees
    error_roll = (desired_roll - actual_roll) * 180 / np.pi
    error_pitch = (desired_pitch - actual_pitch) * 180 / np.pi
    error_yaw = (desired_yaw - actual_yaw) * 180 / np.pi
    
    # Create figure with 6 subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Controller Performance: Error Over Time', fontsize=16, fontweight='bold')
    
    # Plot x position error
    axes[0, 0].plot(time, error_x, 'r-', linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('X Position Error (cm)')
    axes[0, 0].set_title('X Position Error')
    axes[0, 0].set_ylim([-30, 30])
    axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot y position error
    axes[0, 1].plot(time, error_y, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Y Position Error (cm)')
    axes[0, 1].set_title('Y Position Error')
    axes[0, 1].set_ylim([-30, 30])
    axes[0, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot z position error
    axes[0, 2].plot(time, error_z, 'r-', linewidth=2)
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Z Position Error (cm)')
    axes[0, 2].set_title('Z Position Error')
    axes[0, 2].set_ylim([-30, 30])
    axes[0, 2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot roll error
    axes[1, 0].plot(time, error_roll, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Roll Error (deg)')
    axes[1, 0].set_title('Roll Angle Error')
    axes[1, 0].set_ylim([-10, 10])
    axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot pitch error
    axes[1, 1].plot(time, error_pitch, 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Pitch Error (deg)')
    axes[1, 1].set_title('Pitch Angle Error')
    axes[1, 1].set_ylim([-10, 10])
    axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot yaw error
    axes[1, 2].plot(time, error_yaw, 'r-', linewidth=2)
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Yaw Error (deg)')
    axes[1, 2].set_title('Yaw Angle Error')
    axes[1, 2].set_ylim([-10, 10])
    axes[1, 2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    pass
