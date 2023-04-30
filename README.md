## 3D UAV simulation and autonomous control for path tracking

### Controller response
![Controller response](docs/controller_response.png "")

### Result on the Helix path
https://user-images.githubusercontent.com/17160701/209675444-38a1941d-f8d2-4477-b8ec-09222fa9c32d.mp4

### 1. Control  

![Controller Arch](docs/Controller%20Arch.png "")  

source: A platform for aerial robotics research and demonstration: The Flying Machine Arena  
Sergei Lupashin, Markus Hehn, Mark W. Mueller, Angela P. Schoellig, Michael Sherback, Raffaello D’Andrea

### 2. Trajectory Planner with minimum snap

Minimum Snap trajectory has been implemented for this project. Here are one a result for an arbritary trajectory, we can find the comparison between a naive trajectory (just connecting waypoints with a straight line) and the optimal one (Minimum snap)

![Min Snap](docs/min_snap.png "")

#### Without collision check
![Min Snap](docs/no_collision_check.png "")

#### With collision check (upper right waypoint is mandadory)
![Min Snap](docs/collision_check.png "")


https://user-images.githubusercontent.com/17160701/211073314-78f419ca-e53b-4a88-a63e-565018963ce7.mp4


# Minimum snap trajectory

Finding a path (waypoints) to a goal location is one of many important steps for autonomous robot. But this path must 
satisfy some conditions such that a controller can handle it. Some of these conditions can be summarize as follow:

- The path must be feasible
- The path must be collision free
- The path must be differentiable
- The path must be smooth enough
  
The thing is that a system as a quadrotor can be represented (in this case) as a 4th order system, so a path for this kind of system must be differentiable at least 4 times:
Let's denote k as the order of derivative:

- k=1 : velocity
- k=2 : acceleration
- k=3 : jerk
- k=4 : snap

So a good trajectory for this system can be thought as minimum snap trajectory, hence a trajectory that minimize the snap criterion. So we need to find the optimal path  

$$x^{*}(t) = argmin_{x(t)} = \int_{0}^{T} \mathcal{L}(x^{....}, x^{...}, \ddot{x}, \dot{x}, x, t) dt =  \int_{0}^{T} x^{....2} dt$$

where $$\mathcal{L}$$ is the Lagrangian of the system and $$x^{....}$$ is the snap of the trajectory.

We can solve the Euler-Lagrange equation to ensure $x^{(6)}=0$ and get a trajectory of the form:  
$$x(t) = c_{7}t^7 + c_{6}t^6 + c_{5}t^5 + c_{4}t^4 + c_{3}t^3 + c_{2}t^2 + c_{1}t + c_{0}$$  

Differentiating this equation gives the velocity/acceleration/jerk/snap constraints and so on...   
$$\dot{x}(t) = 7c_{7}t^6 +6 c_{6}t^5 + 5c_{5}t^4 + 4c_{4}t^3 + 3c_{3}t^2 + 2c_{2}t + c_{1}$$

what we are interested in is to find the coefficient `c0, c1, c2, c3, c4, c5` that satisfy all the constraints (boundary conditions) mentioned above.
_note: If I have another constraint to respect, I will have to find one more coefficients._

Each of the conditions gives an equation, so we can represent them in a **Matrix** $A$. We can write the equation in terms of unknown constant and boundary conditions. Solving for
these constants (coeffs) are a linear problem.

To respect the position constraint: 
$$x(t) = c_{7}t^7 + c_{6}t^6 + c_{5}t^5 + c_{4}t^4 + c_{3}t^3 + c_{2}t^2 + c_{1}t + c_{0}$$  

So we have: 
$$x(0) = c_{0} = a$$  
$$x(T) = c_{7}T^7 + c_{6}T^6 + c_{5}T^5 + c_{4}T^4 + c_{3}T^3 + c_{2}T^2 + c_{1}T + c_{0} = b$$

in matrix form, at t=0 we must have:  

$$\begin{bmatrix} 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \end{bmatrix} \cdot  \begin{bmatrix}
                                                                        c_{7} \\
                                                                        c_{6} \\
                                                                        c_{5} \\
                                                                        c_{4} \\
                                                                        c_{3} \\
                                                                        c_{2} \\
                                                                        c_{1} \\
                                                                        c_{0}
                                                                        \end{bmatrix} = a$$

in matrix form, at t=T we must have:  

$$\begin{bmatrix} T^7 & T^6 & T^5 & T^4 & T^3 & T^2 & T^1 & T^0 \end{bmatrix} \cdot  \begin{bmatrix}
                                                                                        c_{7} \\
                                                                                        c_{6} \\
                                                                                        c_{5} \\
                                                                                        c_{4} \\
                                                                                        c_{3} \\
                                                                                        c_{2} \\
                                                                                        c_{1} \\
                                                                                        c_{0}
                                                                                        \end{bmatrix} = b$$

To find the equation for the velocity, we just have to differentiate the position equation
$$\dot{x}(t) = 7c_{7}t^6 +6 c_{6}t^5 + 5c_{5}t^4 + 4c_{4}t^3 + 3c_{3}t^2 + 2c_{2}t + c_{1}$$
$$\dot{x}(0) = c_{1} = v_{a}$$
$$\dot{x}(T) = 7c_{7}T^6 +6 c_{6}T^5 + 5c_{5}T^4 + 4c_{4}T^3 + 3c_{3}T^2 + 2c_{2}T + c_{1} = v_{b}$$

in matrix form, at t=0 we must have:  

$$\begin{bmatrix} 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \end{bmatrix} \cdot  \begin{bmatrix}
                                                                        c_{7} \\
                                                                        c_{6} \\
                                                                        c_{5} \\
                                                                        c_{4} \\
                                                                        c_{3} \\
                                                                        c_{2} \\
                                                                        c_{1} \\
                                                                        c_{0}
                                                                        \end{bmatrix} = v_{a}$$

in matrix form, at t=T we must have:  

$$\begin{bmatrix} 7T^6 & 6T^5 & 5T^4 & 4T^3 & 3T^2 & 2T^1 & T^0 & 0 \end{bmatrix} \cdot  \begin{bmatrix}
                                                                                        c_{7} \\
                                                                                        c_{6} \\
                                                                                        c_{5} \\
                                                                                        c_{4} \\
                                                                                        c_{3} \\
                                                                                        c_{2} \\
                                                                                        c_{1} \\
                                                                                        c_{0}
                                                                                        \end{bmatrix} = v_{b}$$


same for accelerations ... we differentiate and we compute:  
$$\ddot{x}(t) = 42c_{7}t^5 + 30c_{6}t^4 + 20c_{5}t^3 + 12c_{4}t^2 + 6c_{3}t + 2c_{2}$$

$$\ddot{x}(0) = 2c_{2} = a_{a}$$
$$\ddot{x}(T) = 42c_{7}T^5 + 30c_{6}T^4 + 20c_{5}T^3 + 12c_{4}T^2 + 6c_{3}T + 2c_{2} = a_{b}$$

in matrix form, at t=0 we must have:  
$$\begin{bmatrix} 0 & 0 & 0 & 0 & 0 & 2 & 0 & 0 \end{bmatrix} \cdot  \begin{bmatrix}
                                                                        c_{7} \\
                                                                        c_{6} \\
                                                                        c_{5} \\
                                                                        c_{4} \\
                                                                        c_{3} \\
                                                                        c_{2} \\
                                                                        c_{1} \\
                                                                        c_{0}
                                                                        \end{bmatrix} = a_{a}$$

in matrix form, at t=T we must have:  
$$\begin{bmatrix} 42T^5 & 30T^4 & 20T^3 & 12T^2 & 6T^1 & 2T^0 & 0 & 0 \end{bmatrix} \cdot  \begin{bmatrix}
                                                                                        c_{7} \\
                                                                                        c_{6} \\
                                                                                        c_{5} \\
                                                                                        c_{4} \\
                                                                                        c_{3} \\
                                                                                        c_{2} \\
                                                                                        c_{1} \\
                                                                                        c_{0}
                                                                                        \end{bmatrix} = a_{b}$$


same for jerk ... we differentiate and we compute:  
$$\dddot{x}(t) = 210c_{7}t^4 + 120c_{6}t^3 + 60c_{5}t^2 + 24c_{4}t + 6c_{3}$$

$$\dddot{x}(0) = 6c_{3} = j_{a}$$
$$\dddot{x}(T) = 210c_{7}T^4 + 120c_{6}T^3 + 60c_{5}T^2 + 24c_{4}T + 6c_{3} = j_{b}$$

in matrix form, at t=0 we must have:  
$$\begin{bmatrix} 0 & 0 & 0 & 0 & 6 & 0 & 0 & 0 \end{bmatrix} \cdot  \begin{bmatrix}
                                                                        c_{7} \\
                                                                        c_{6} \\
                                                                        c_{5} \\
                                                                        c_{4} \\
                                                                        c_{3} \\
                                                                        c_{2} \\
                                                                        c_{1} \\
                                                                        c_{0}
                                                                        \end{bmatrix} = j_{a}$$

nmatrix form, at t=T we must have:  
$$\begin{bmatrix} 210T^4 & 120T^3 & 60T^2 & 24T^1 & 6T^0 & 0 & 0 & 0 \end{bmatrix} \cdot  \begin{bmatrix}
                                                                                        c_{7} \\
                                                                                        c_{6} \\
                                                                                        c_{5} \\
                                                                                        c_{4} \\
                                                                                        c_{3} \\
                                                                                        c_{2} \\
                                                                                        c_{1} \\
                                                                                        c_{0}
                                                                                        \end{bmatrix} = j_{b}$$

same for snap ... we differentiate and we compute:  

$$\ddddot{x}(t) = 840c_{7}t^3 + 360c_{6}t^2 + 120c_{5}t + 24c_{4}$$
$$\ddddot{x}(0) = 24c_{4} = s_{a}$$
$$\ddddot{x}(T) = 840c_{7}T^3 + 360c_{6}T^2 + 120c_{5}T + 24c_{4} = s_{b}$$

in matrix form, at t=0 we must have:  

$$\begin{bmatrix} 0 & 0 & 0 & 24 & 0 & 0 & 0 & 0 \end{bmatrix} \cdot  \begin{bmatrix}
                                                                        c_{7} \\
                                                                        c_{6} \\
                                                                        c_{5} \\
                                                                        c_{4} \\
                                                                        c_{3} \\
                                                                        c_{2} \\
                                                                        c_{1} \\
                                                                        c_{0}
                                                                        \end{bmatrix} = s_{a}$$

in matrix form, at t=T we must have:  

$$\begin{bmatrix} 840T^3 & 360T^2 & 120T^1 & 24T^0 & 0 & 0 & 0 & 0 \end{bmatrix} \cdot  \begin{bmatrix}
                                                                                        c_{7} \\
                                                                                        c_{6} \\
                                                                                        c_{5} \\
                                                                                        c_{4} \\
                                                                                        c_{3} \\
                                                                                        c_{2} \\
                                                                                        c_{1} \\
                                                                                        c_{0}
                                                                                        \end{bmatrix} = s_{b}$$

All the 8 constraints can be written as a 8x8 matrix in order to find the coefficients of the polynomial (coefficients of the trajectory).
The full matrix is the following:

$$A = \begin{bmatrix}
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
        T^7 & T^6 & T^5 & T^4 & T^3 & T^2 & T & 1 \\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
        7T^6 & 6T^5 & 5T^4 & 4T^3 & 3T^2 & 2T & 1 & 0 \\
        0 & 0 & 0 & 0 & 0 & 2 & 0 & 0 \\
        42T^5 & 30T^4 & 20T^3 & 12T^2 & 6T & 2 & 0 & 0 \\
        0 & 0 & 0 & 0 & 6 & 0 & 0 & 0 \\
        210T^4 & 120T^3 & 60T^2 & 24T & 6 & 0 & 0 & 0 \\
        0 & 0 & 0 & 24 & 0 & 0 & 0 & 0 \\
        840T^3 & 360T^2 & 120T & 24 & 0 & 0 & 0 & 0
        \end{bmatrix}$$
