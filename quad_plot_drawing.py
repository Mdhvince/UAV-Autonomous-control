import roblib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


"""
Quadrotor Dynamics - 3D Simulation
"""

# I = inertial matrix
# b = drag coeff
# d = delta
I = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 20]])
m, g, b, d, l = 10, 9.81, 2, 1, 1

path_x = []
path_y = []
path_z = []
def draw_quad(state, angles, lim=30):

    ax.clear()
    ax.set_xlim3d(-lim, lim)
    ax.set_ylim3d(-lim, lim)
    ax.set_zlim3d(0, lim)

    x, y, z = state[0:3]

    path_x.append(-x); path_y.append(y); path_z.append(abs(z))
    ax.scatter(path_x, path_y, path_z)
    roblib.draw_quadrotor3D(ax, state, angles, 5*l)


def B_mat():
    return np.array([
            [b, b, b, b],
            [-b*l, 0, b*l, 0],
            [0, -b*l, 0, b*l],
            [-d, d, -d, d],
        ]) 


def f(state, W):
    state = state.flatten()
    _, _, _, φ, θ, ψ = list(state[0:6])
    vr = state[6:9].reshape(3, 1)  # velocities body frame
    wr = state[9:12].reshape(3, 1)  # angular velocities
    W2 = W * abs(W)
    τ = B_mat() @ W2.flatten()
    E = roblib.eulermat(φ, θ, ψ)  # to change between body frame and world frame

    dvr = -roblib.adjoint(wr) @ vr \
            + np.linalg.inv(E) @ np.array([[0], [0], [g]]) \
                + np.array([[0], [0], [-τ[0]/m]])
    
    dp = E @ vr
    dangles = roblib.eulerderivative(φ, θ, ψ) @ wr
    dwr = np.linalg.inv(I) @ (-roblib.adjoint(wr) @ I @ wr + τ[1:4].reshape(3, 1))
    dX = np.vstack((dp, dangles, dvr, dwr))
    return dX


def controller(state):
    """
    Return the angular velocities of each motors
    """
    state = state.flatten()
    x, y, z, φ, θ, ψ = list(state[0:6])
    wr = state[9:12].reshape(3, 1)
    vr = state[6:9].reshape(3, 1)
    E = roblib.eulermat(φ, θ, ψ)
    dp = E @ vr

    
    xy_desired = np.array([[0], [0]])
    z_desired = -10
    vel_x_desired = 20
    

    # thrust desired (tanh is used to scale between -1 and +1)
    kp = 300
    kd = 60
    error_z = z - z_desired  # remember z pointing downward
    thrust_desired = kp * np.tanh(error_z) + kd * vr[2]  # tau desired (τ0 desired)
    
    # roll desired
    error_direction = roblib.angle(xy_desired) - roblib.angle(dp)
    φ_desired = 0.5 * np.tanh(10 * roblib.sawtooth(error_direction))

    # pitch desired
    error_vel_x = vel_x_desired - vr[0]
    θ_desired = -.3 * np.tanh(error_vel_x)

    # yaw desired
    ψ_desired = roblib.angle(dp)

    # disired angular velocity in the body frame
    kwrd = 5
    omega_desired_bf = kwrd * np.linalg.inv(roblib.eulerderivative(φ, θ, ψ)) @ np.array([
                [roblib.sawtooth(φ_desired - φ)], [roblib.sawtooth(θ_desired - θ)], [roblib.sawtooth(ψ_desired - ψ)]
            ])
    
    error_w = omega_desired_bf - wr
    res = []
    for i in error_w:
        res.append(i[0])            

    error_w = np.array(res).reshape(3, 1)

    kw = 100
    torques_desired = I @ (kw * error_w + roblib.adjoint(wr) @ I @ wr)  # (τ1, τ2, τ3 desired)    

    W2 = np.linalg.inv(B_mat()) @ np.vstack(([thrust_desired], torques_desired))

    w = np.sqrt(np.abs(W2)) * np.sign(W2)
    return w






if __name__ == "__main__":

    fig = plt.figure()
    ax = plt.axes(projection ="3d")

    state = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T
    blade_angles = np.array([[0, 0, 0, 0]]).T
    dt = .01

    for t in np.arange(0, 10, dt):

        angular_velocity = controller(state)

        state = state + dt * f(state, angular_velocity)
        blade_angles = blade_angles + dt * 50 * angular_velocity  # for drawing
        draw_quad(state, blade_angles)
        plt.pause(.001)
    
    plt.pause(1)
