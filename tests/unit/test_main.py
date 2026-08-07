import numpy as np
import pytest

from uav_ac.control.controller import CascadedController
from uav_ac.main import (
    TrajectoryController,
    _generate_mission_trajectory,
    _trajectory_after_takeoff,
)
from uav_ac.simulation.mujoco_sim import MujocoSimulation


def test_trajectory_controller_reset_should_restart_trajectory_and_clear_control_state():
    # Arrange
    simulation = MujocoSimulation()
    controller = CascadedController(simulation.quad.g, simulation.quad.dt * 10)
    trajectory = np.zeros((2, 10))
    trajectory_controller = TrajectoryController(
        controller, simulation.quad, trajectory, inner_loop_frequency=10)
    trajectory_controller.trajectory_index = 1
    trajectory_controller.inner_step = 12
    trajectory_controller.thrust_cmd = 4.0
    trajectory_controller.pqr_cmd[:] = 1.0
    controller.integral_error = 2.0

    # Act
    trajectory_controller.reset()

    # Assert
    assert trajectory_controller.trajectory_index == 0
    assert trajectory_controller.inner_step == 0
    assert trajectory_controller.thrust_cmd == pytest.approx(0.0)
    assert trajectory_controller.pqr_cmd == pytest.approx(np.zeros(3))
    assert controller.integral_error == pytest.approx(0.0)


def test_trajectory_after_takeoff_should_hide_vertical_departure_segment():
    # Arrange
    trajectory = np.zeros((5, 10))
    trajectory[:, :3] = np.array([
        [1.0, 7.0, -0.021],
        [1.0, 7.0, -0.7],
        [1.0, 7.0, -1.3],
        [2.0, 7.0, -1.3],
        [3.0, 6.0, -1.5],
    ])
    takeoff_waypoint = np.array([1.0, 7.0, -1.3])

    # Act
    visible_trajectory = _trajectory_after_takeoff(trajectory, takeoff_waypoint)

    # Assert
    assert visible_trajectory[:, :3] == pytest.approx(trajectory[2:, :3])


def test_generate_mission_trajectory_should_keep_takeoff_vertical_and_above_ground():
    # Arrange
    simulation = MujocoSimulation()

    # Act
    trajectory = _generate_mission_trajectory(
        simulation.mission_waypoints,
        simulation.obstacles,
        velocity=2.0,
        dt=simulation.quad.dt * 10,
    )
    takeoff_index = np.argmin(np.linalg.norm(
        trajectory[:, :3] - simulation.mission_waypoints[1], axis=1))
    takeoff_positions = trajectory[:takeoff_index + 1, :3]

    # Assert
    assert takeoff_positions[:, :2] == pytest.approx(
        np.repeat(simulation.start_position[np.newaxis, :2], len(takeoff_positions), axis=0))
    assert np.all(trajectory[:, 2] <= 0.0)