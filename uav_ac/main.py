import numpy as np

from uav_ac import utils
from uav_ac.control.controller import CascadedController
from uav_ac.planning.minimum_snap import MinimumSnap
from uav_ac.quadrotor.quad import Quad
from uav_ac.simulation.mujoco_sim import MujocoSimulation


class TrajectoryController:
    """Drive the existing cascaded controller from a time-parameterized trajectory."""

    def __init__(
            self,
            controller: CascadedController,
            quad: Quad,
            trajectory: np.ndarray,
            inner_loop_frequency: int,
    ):
        self.controller = controller
        self.quad = quad
        self.trajectory = trajectory
        self.inner_loop_frequency = inner_loop_frequency
        self.trajectory_index = 0
        self.inner_step = 0
        self.thrust_cmd = 0.0
        self.pqr_cmd = np.zeros(3)

    def reset(self) -> None:
        """Restart trajectory tracking from its initial state."""
        self.controller.reset()
        self.trajectory_index = 0
        self.inner_step = 0
        self.thrust_cmd = 0.0
        self.pqr_cmd.fill(0.0)

    def step(self) -> None:
        """Execute one inner body-rate control cycle."""
        if self.inner_step % self.inner_loop_frequency == 0:
            self._update_outer_loop()

        moment_cmd = self.controller.body_rate_controller(
            self.quad, self.pqr_cmd, self.quad.kp_p, self.quad.kp_q, self.quad.kp_r)
        self.quad.set_propeller_speed(self.thrust_cmd, moment_cmd)
        self.inner_step += 1

    def _update_outer_loop(self) -> None:
        target = self.trajectory[self.trajectory_index]
        rotation = self.quad.R()
        thrust_cmd = self.controller.altitude(
            self.quad, target[[2, 5, 8]], rotation,
            self.quad.kp_z, self.quad.kd_z, self.quad.ki_z)
        bxy_cmd = self.controller.lateral(
            self.quad, target[[0, 3, 6]], target[[1, 4, 7]], thrust_cmd,
            self.quad.kp_xy, self.quad.kd_xy)

        self.thrust_cmd = thrust_cmd
        self.pqr_cmd = self.controller.reduced_attitude(
            self.quad, bxy_cmd, target[9], rotation,
            self.quad.kp_roll, self.quad.kp_pitch, self.quad.kp_yaw)
        self.trajectory_index = min(self.trajectory_index + 1, len(self.trajectory) - 1)


def _trajectory_after_takeoff(
        trajectory: np.ndarray,
        takeoff_waypoint: np.ndarray,
) -> np.ndarray:
    """Return the trajectory from the sample nearest the takeoff waypoint."""
    distances = np.linalg.norm(trajectory[:, :3] - takeoff_waypoint, axis=1)
    return trajectory[np.argmin(distances):]


def _generate_mission_trajectory(
        waypoints: np.ndarray,
        obstacles: np.ndarray,
        velocity: float,
        dt: float,
) -> np.ndarray:
    """Generate an isolated vertical takeoff followed by the laboratory course."""
    takeoff_trajectory = MinimumSnap(
        waypoints[:2], obstacles, velocity, dt).get_trajectory()
    course_trajectory = MinimumSnap(
        waypoints[1:], obstacles, velocity, dt).get_trajectory()
    return np.vstack((takeoff_trajectory, course_trajectory))


def main() -> None:
    cfg, cfg_flight = utils.get_config()

    frequency = cfg.getint("frequency")

    velocity = cfg_flight.getfloat("velocity")
    min_distance_target = cfg_flight.getfloat("min_dist_target")

    simulation = MujocoSimulation()
    quad = simulation.quad
    trajectory_dt = quad.dt * frequency
    ctrl = CascadedController(quad.g, trajectory_dt)

    global_trajectory = _generate_mission_trajectory(
        simulation.mission_waypoints,
        simulation.obstacles,
        velocity,
        trajectory_dt,
    )
    visible_trajectory = _trajectory_after_takeoff(
        global_trajectory, simulation.mission_waypoints[1])
    simulation.set_trajectory_visualization(visible_trajectory[:, :3])

    trajectory_controller = TrajectoryController(ctrl, quad, global_trajectory, frequency)

    print("Trajectory ready. Press Backspace in the MuJoCo viewer to replay the flight.")
    simulation.run_interactive(trajectory_controller.step, trajectory_controller.reset)

    distance_to_goal = np.linalg.norm(quad.position - simulation.goal_position)
    goal_has_been_reached = distance_to_goal < min_distance_target
    print(f"Flight finished {distance_to_goal:.2f} m away from the goal "
          f"({'reached' if goal_has_been_reached else 'missed'}).")
    if simulation.collision_detected:
        print("At least one collision occurred during the flight.")


if __name__ == "__main__":
    main()
