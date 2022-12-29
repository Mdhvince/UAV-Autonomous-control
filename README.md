## 3D UAV simulation and autonomous control for path tracking
### Result on the Helix path
https://user-images.githubusercontent.com/17160701/209675444-38a1941d-f8d2-4477-b8ec-09222fa9c32d.mp4

### Result on the Figure 8 path
![Result Fig 8](docs/results_fig8_classic_control.png "")


### 1. Classic autonomous control  

![Controller Arch](docs/Controller%20Arch.png "")  

The algorithm is composed of  

- Outer loop:
    - Altitude controller
    - Lateral controller  
---
- Inner Loop x10 faster:
    - Roll-Pitch controller
    - Yaw controller
    - Body Rate controller  
---
- UAV Equation of motion
- State update using euler integration method

### To do
- [x] Simulation of 3D UAV Dynamics
    - [x] UAV Equation of motion
    - [x] State update
    - [x] 3D Visual
- [ ] Perception
- [ ] Estimation / Localization / Mapping
- [ ] Planning
    - [x] Minimum jerk trajectory
    - [x] Minimum snap trajectory
    - [ ] Coliision avoidance
- [x] Motion Control
    - [x] Altitude controller
    - [x] Lateral controller
    - [x] Roll/Pitch controller
    - [x] Yaw controller
    - [x] Body Rate controller