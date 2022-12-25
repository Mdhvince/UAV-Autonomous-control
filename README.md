## 3D UAV simulation and autonomous control for path tracking

### 1. Classic autonomous control  
  
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

### Result on the Figure 8 path

https://user-images.githubusercontent.com/17160701/209466834-ef1d1a3a-e8b7-46ba-816e-88d1c8a753c6.mp4

![Result Fig 8](docs/results_fig8_classic_control.png "")

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
- [ ] Mapping
- [ ] Planning
- [ ] RL Based Control
- [ ] C++ or Rust code for inner loop controller
