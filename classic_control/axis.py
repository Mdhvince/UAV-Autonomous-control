import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def Rot(phi, theta, psi):
    """
    To transform between body frame accelerations and world frame accelerations
    """
    r_x = np.array([[1, 0, 0],
                    [0, np.cos(phi), -np.sin(phi)],
                    [0, np.sin(phi), np.cos(phi)]])

    r_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])

    r_z = np.array([[np.cos(psi), -np.sin(psi), 0],
                    [np.sin(psi), np.cos(psi), 0],
                    [0,0,1]])

    r_yx = np.matmul(r_y, r_x)
    return np.matmul(r_z, r_yx)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

deg2rad = lambda x: x * np.pi / 180

# Define the origin of the body-fixed frame
axis_length = 1
origin = axis_length * np.array([0, 0, 0])  # Origin of the body-fixed frame in the world frame
x_axis = axis_length * np.array([1, 0, 0])  # X-axis of the body-fixed frame
y_axis = axis_length * np.array([0, 1, 0])  # Y-axis of the body-fixed frame
z_axis = axis_length * np.array([0, 0, 1])  # Z-axis of the body-fixed frame


# for i in range(360):
    
roll, pitch, yaw = deg2rad(0), deg2rad(0), deg2rad(0)
R = Rot(roll, pitch, yaw)

# Transform the body-fixed axes into the world frame using the rotation matrix
x_axis_world = np.dot(R, x_axis)
y_axis_world = np.dot(R, y_axis)
z_axis_world = np.dot(R, z_axis)

# Plot the x, y, and z axes in the world frame
ax.clear()
ax.plot3D([origin[0], x_axis_world[0]], [origin[1], x_axis_world[1]], [origin[2], x_axis_world[2]], 'r')
ax.plot3D([origin[0], y_axis_world[0]], [origin[1], y_axis_world[1]], [origin[2], y_axis_world[2]], 'g')
ax.plot3D([origin[0], z_axis_world[0]], [origin[1], z_axis_world[1]], [origin[2], z_axis_world[2]], 'b')


lim = 10
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# plt.pause(.001)

plt.show()
