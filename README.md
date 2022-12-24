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
https://user-images.githubusercontent.com/17160701/209437807-9458b60c-aa87-4924-9fe2-076876c0d746.mp4

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
- [ ] C++ for inner loop controller
