import numpy as np

from uav_ac.control.controller import CascadedController
from uav_ac.main import TrajectoryController, _generate_mission_trajectory
from uav_ac.simulation.mujoco_sim import MujocoSimulation


FREQUENCY = 10


def test_mujoco_simulation_should_follow_minimum_snap_trajectory_without_collision():
    # Arrange
    simulation = MujocoSimulation()
    quad = simulation.quad
    trajectory_dt = quad.dt * FREQUENCY
    trajectory = _generate_mission_trajectory(
        simulation.mission_waypoints,
        simulation.obstacles,
        velocity=2.0,
        dt=trajectory_dt,
    )
    controller = CascadedController(g=quad.g, dt=trajectory_dt)
    trajectory_controller = TrajectoryController(controller, quad, trajectory, FREQUENCY)

    # Act
    tracking_errors = []
    for target in trajectory:
        for _ in range(FREQUENCY):
            trajectory_controller.step()
            simulation.step()
        tracking_errors.append(np.linalg.norm(quad.position - target[:3]))

    # Assert
    assert np.linalg.norm(quad.position - simulation.goal_position) < 0.5
    assert np.mean(tracking_errors) < 0.5
    assert simulation.collision_detected is False
    assert simulation.model.geom("actual_trajectory_segment000").rgba[3] > 0.0