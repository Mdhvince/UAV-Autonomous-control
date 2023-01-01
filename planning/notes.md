# Minimum snap trajectory

Finding a path (waypoints) to a goal location is one of many important steps for autonomous robot. But this path must satisfy some conditions such that a controller can handle it. Some of these conditions can be summurize as follow:

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

<center> $x^{*}(t) = argmin_{x(t)} = \int_{0}^{T} \mathcal{L}(x^{....}, x^{...}, \ddot{x}, \dot{x}, x, t) dt =  \int_{0}^{T} x^{....2} dt$ </center>  
<br>
<br>
We can solve the Euler-Lagrange equation to ensure $x^{(6)}=0$ and get a trajectory of the form:  
<center> $x(t) = c_{7}t^7 + c_{6}t^6 + c_{5}t^5 + c_{4}t^4 + c_{3}t^3 + c_{2}t^2 + c_{1}t + c_{0}$ </center>   
<br>
<br>
Differentiating this equation gives the velocity/acceleration/jerk/snap and so on... constraints  
<center> $\dot{x}(t) = 7c_{7}t^6 +6 c_{6}t^5 + 5c_{5}t^4 + 4c_{4}t^3 + 3c_{3}t^2 + 2c_{2}t + c_{1}$ </center>
<br>
<br>





<!-- what we are intersted in is to find the coefficient c0, c1, c2, c3, c4, c5 that satisfy all
the constraints (boundary conditions) mentionned above
note: that if I have another constraint to respect, I will have to find one more coeff c6*t^6.

each of the conditions gives an equation, so we can represent them in a matrix.
we can write the equation in terms of unknown constant and boundary conditions. Solving for
these constants (coeffs) are a linear problem

To respect the position constraint: 
x(t) = c5*t^5 + c4*t^4 + c3*t^3 + c2*t^2 + c1*t^1 + c0*t^0

So we must have 
  x(0) = c0 = a
  x(T) = c5*(T^5) + c4*(T^4) + c3*(T^3) + c2*(T^2) + c1*(T^1) + c0*(T^0) = b

in matrix form, at t=0 we must have:
              |c5|
              |c4|
[0 0 0 0 0 1] |c3| = a
              |c2|
              |c1|
              |c0|

in matrix form, at t=T we must have:
                          |c5|
                          |c4|
[T^5 T^4 T^3 T^2 T^1 T^0] |c3| = b
                          |c2|
                          |c1|
                          |c0|

to find the equation for the velocity, we just have to defferentiate the position equation
x_dot(t) = 5*c5*t^4 + 4*c4*t^3 + 3*c3*t^2 + 2*c2*t^1 + c1 + 0
x_dot(0) = c1 = vel_a
x_dot(T) = 5*c5*(T^4) + 4*c4*(T^3) + 3*c3*(T^2) + 2*c2*(T^1) + c1 + 0

in matrix form, at t=0 we must have
              |c5|
              |c4|
[0 0 0 0 1 0] |c3| = vel_a
              |c2|
              |c1|
              |c0|

in matrix form, at t=T we must have
                              |c5|
                              |c4|
[5T^4 4T^3 3T^2 2T^1 T^0 0]   |c3| = vel_b
                              |c2|
                              |c1|
                              |c0|

same for accelerations ... we differentiate and we compute
x_dot_dot = 20*c5*t^3 + 12*c4*t^2 + 6*c3*t + 2*c2*t^0 + 0 + 0 

all of the 6 constraint can be written as a 6x6 matrix in order to find the coefficient




###################################################################################################



POS at time 0 and time T : x(t) = c7*t^7 + c6*t^6 + c5*t^5 + c4*t^4 + c3*t^3 + c2*t^2 + c1*t^1 + c0*t^0

0, 0, 0, 0, 0, 0, 0, 1

T**7, T**6, T**5, T**4, T**3, T**2, T, 1

---------------
Vel at time 0 and time T : xd(t) = 7*c7*t^6 + 6*c6*t^5 + 5*c5*t^4 + 4*c4*t^3 + 3*c3*t^2 + 2*c2*t^1 + c1 + 0 

0, 0, 0, 0, 0, 0, 1, 0

7*T**6, 6*T**5, 5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0

---------------
Acc at time 0 and time T : xdd(t) = 6*7*c7*t^5 + 5*6*c6*t^4 + 20*c5*t^3 + 12*c4*t^2 + 6*c3*t + 2*c2 + 0 + 0
 
0, 0, 0, 0, 0, 2, 0, 0

42*T**5, 30*T**4, 20*T**3, 12*T**2, 6*T, 2, 0, 0

---------------
Jerk at time 0 and time T : xddd(t) = 210*c7*t^4 + 120*c6*t^3 + 60*c5*t^2 + 24*c4*t + 6*c3 + 0 + 0 + 0

0, 0, 0, 0, 6, 0, 0, 0

210*T**4, 120*T**3, 60*T**2, 24*T, 6, 0, 0, 0


####################################################
All together

[0, 0, 0, 0, 0, 0, 0, 1],
[T**7, T**6, T**5, T**4, T**3, T**2, T, 1],
[0, 0, 0, 0, 0, 0, 1, 0],
[7*T**6, 6*T**5, 5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],
[0, 0, 0, 0, 0, 2, 0, 0],
[42*T**5, 30*T**4, 20*T**3, 12*T**2, 6*T, 2, 0, 0],
[0, 0, 0, 0, 6, 0, 0, 0],
[210*T**4, 120*T**3, 60*T**2, 24*T, 6, 0, 0, 0] -->