## 3D UAV simulation and autonomous control for path tracking

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

### Result on the Figure 8 path
https://user-images.githubusercontent.com/17160701/209479162-a6c44616-ee63-4361-8d10-558682a14795.mp4

### Result on the Helix path
https://user-images.githubusercontent.com/17160701/209479519-78ff2b20-476a-42fe-bb60-d8cf54d24e68.mp4


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
