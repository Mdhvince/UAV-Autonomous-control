## 3D UAV simulation and autonomous control for path tracking

### Controller respose
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