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

def draw_quad(state, angles, lim=30):
    ax.clear()
    ax.set_xlim3d(-lim, lim)
    ax.set_ylim3d(-lim, lim)
    ax.set_zlim3d(0, lim)
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
    _, _, _, φ, θ, ψ = list(state[0:6])
    wr = state[9:12].reshape(3, 1)
    # vr = state[6:9].reshape(3, 1)
    # E = roblib.eulermat(φ, θ, ψ)
    # dp = E @ vr

    # tau desired, roll, pitch, yaw desired
    τd0 = 200
    φd = 0
    θd = 0
    ψd = 0

    kwrd = 5
    wrd = kwrd * np.linalg.inv(roblib.eulerderivative(φ, θ, ψ)) \
            @ np.array([
                [roblib.sawtooth(φd - φ)], [roblib.sawtooth(θd - θ)], [roblib.sawtooth(ψd - ψ)]
            ])

    error_w = wrd - wr
    kw = 200
    τd13 = I @ ((kw * error_w) + roblib.adjoint(wr) @ I @ wr)

    W2 = np.linalg.inv(B_mat()) @ np.vstack(([τd0], τd13))
    w = np.sqrt(np.abs(W2)) * np.sign(W2)
    return w






if __name__ == "__main__":
    
    # rot speed: w1, w2, w3, w4 
    # positions: x(forward), y, z(down) 
    # derivative are velovities x_d, y_d, z_d (body frame)
    # euler angles: phi, theta, psi (roll, pitch, yaw)
    # derivative: phi_d, theta_d, yaw_d (body frame)
    # state vector = [x, y, z, phi, theta, psi, x_d, y_d, z_d, phi_d, theta_d, yaw_d]
    # robot_speed_world_frame = rot_mat . robot_speed_body_frame

    fig = plt.figure()
    ax = plt.axes(projection ="3d")

    state = np.array([[0, 0, -5, 1, 0, 0, 10, 10, 0, 0, 0, 0]]).T
    blade_angles = np.array([[0, 0, 0, 0]]).T
    dt = .01

    for t in np.arange(0, 3, dt):

        angular_velocity = controller(state)

        state = state + dt * f(state, angular_velocity)
        blade_angles = blade_angles + dt * 50 * angular_velocity  # for drawing
        draw_quad(state, blade_angles)
        plt.pause(.001)
    
    plt.pause(1)
