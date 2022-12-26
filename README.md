## 3D UAV simulation and autonomous control for path tracking
### Result on the Helix path
https://user-images.githubusercontent.com/17160701/209576543-1019a9b3-0973-407e-a4a9-37cd440d0987.mp4

### Result on the Figure 8 path
https://user-images.githubusercontent.com/17160701/209576536-a56c9950-f2d9-4526-a8a3-7f5d370b8c37.mp4
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
- [ ] State Estimation
- [x] Control - PID
    - [x] Altitude controller
    - [x] Lateral controller
    - [x] Roll/Pitch controller
    - [x] Yaw controller
    - [x] Body Rate controller
- [ ] Estimation/Mapping
- [ ] Planning
- [ ] RL for either Control/Planning/Estimation
- [ ] C++ or Rust code for inner loop controller
