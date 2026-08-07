from unittest.mock import patch

import mujoco
import numpy as np
import pytest

from uav_ac.simulation.mujoco_sim import DEFAULT_SCENE_PATH, MujocoSimulation, mujoco_to_ned_state


START_POSITION = np.array([1.0, 7.0, -0.021])


@pytest.fixture
def simulation():
    return MujocoSimulation()


@pytest.fixture
def quad(simulation):
    return simulation.quad


def test_mujoco_simulation_should_load_vehicle_and_mission_from_scene():
    # Arrange
    # Act
    simulation = MujocoSimulation()

    # Assert
    assert simulation.quad.m == pytest.approx(
        simulation.model.body_mass[simulation.model.body("quadrotor").id])
    assert simulation.start_position == pytest.approx(START_POSITION)
    assert simulation.goal_position == pytest.approx(np.array([23.0, 7.0, -2.0]))
    assert simulation.space_limits == pytest.approx(
        np.array([[0.0, 0.0, -6.0], [24.0, 14.0, 0.0]]))
    assert len(simulation.obstacles) >= 3


def test_mujoco_simulation_should_load_ordered_mandatory_waypoints_from_scene(simulation):
    # Arrange
    expected_waypoints = np.array([
        [1.0, 7.0, -0.021],
        [1.0, 7.0, -1.3],
        [4.0, 7.0, -1.3],
        [7.5, 4.0, -3.0],
        [11.0, 7.0, -3.5],
        [14.0, 10.0, -2.5],
        [17.0, 10.0, -3.2],
        [20.5, 7.0, -1.4],
        [23.0, 7.0, -2.0],
    ])

    # Act
    waypoints = simulation.mission_waypoints

    # Assert
    assert waypoints == pytest.approx(expected_waypoints)
    assert waypoints[1, :2] == pytest.approx(waypoints[0, :2])
    assert waypoints[1, 2] < waypoints[0, 2]


def test_mujoco_to_ned_state_should_convert_enu_and_flu_frames():
    # Arrange
    position = np.array([1.0, -2.0, 3.0])
    quaternion = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    velocity = np.array([4.0, -5.0, 6.0, 0.1, -0.2, 0.3])

    # Act
    state = mujoco_to_ned_state(position, quaternion, velocity)

    # Assert
    assert state[:3] == pytest.approx(np.array([1.0, 2.0, -3.0]))
    assert state[3:7] == pytest.approx(np.array([np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)]))
    assert state[7:10] == pytest.approx(np.array([4.0, 5.0, -6.0]))
    assert state[10:13] == pytest.approx(np.array([0.1, 0.2, -0.3]))


def test_mujoco_simulation_should_initialize_quad_at_ned_start_position(simulation, quad):
    # Arrange
    expected_position = START_POSITION

    # Act
    actual_position = quad.position

    # Assert
    assert actual_position == pytest.approx(expected_position)
    assert simulation.has_collision is False


def test_mujoco_scene_should_start_drone_immediately_above_ground(simulation):
    # Arrange
    body = simulation.model.geom("body")

    # Act
    body_bottom = simulation.data.geom_xpos[body.id, 2] - body.size[2]

    # Assert
    assert 0.0 < body_bottom < 0.005


def test_mujoco_scene_should_render_checkered_ground_and_hide_boundaries(simulation):
    # Arrange
    ground = simulation.model.geom("ground")
    boundary_names = [
        "world_boundary_north",
        "world_boundary_south",
        "world_boundary_west",
        "world_boundary_east",
        "world_boundary_ceiling",
    ]

    # Act
    ground_material_id = simulation.model.geom_matid[ground.id]
    ground_texture_ids = simulation.model.mat_texid[ground_material_id]
    boundary_opacities = [simulation.model.geom(name).rgba[3] for name in boundary_names]

    # Assert
    assert np.any(ground_texture_ids >= 0)
    assert simulation.model.mat_texrepeat[ground_material_id] == pytest.approx(
        np.array([8.0, 4.6667]))
    assert boundary_opacities == pytest.approx(np.zeros(len(boundary_names)))


def test_mujoco_simulation_should_ignore_ground_contact_during_takeoff(simulation):
    # Arrange
    # Act
    for _ in range(50):
        simulation.step()

    # Assert
    assert simulation.has_collision is True
    assert simulation.collision_detected is False


def test_mujoco_simulation_should_detect_ground_contact_after_takeoff(simulation):
    # Arrange
    simulation.data.qpos[2] = 0.2
    mujoco.mj_forward(simulation.model, simulation.data)
    simulation.step()
    simulation.data.qpos[2] = 0.019
    simulation.data.qvel[:] = 0.0
    mujoco.mj_forward(simulation.model, simulation.data)

    # Act
    simulation.step()

    # Assert
    assert simulation.collision_detected is True


def test_mujoco_simulation_should_apply_gravity_when_rotors_are_stopped(simulation, quad):
    # Arrange
    initial_z = quad.z

    # Act
    for _ in range(20):
        simulation.step()

    # Assert
    assert quad.z > initial_z
    assert quad.z_vel > 0.0


def test_mujoco_simulation_should_hover_with_weight_balanced_by_rotors(simulation, quad):
    # Arrange
    hover_speed = np.sqrt(quad.m * quad.g / (4 * quad.kf))
    quad.omega[:] = hover_speed

    # Act
    for _ in range(100):
        simulation.step()

    # Assert
    assert quad.position == pytest.approx(START_POSITION, abs=1e-6)
    assert quad.velocity == pytest.approx(np.zeros(3), abs=1e-6)


def test_run_interactive_should_use_managed_viewer_without_duplicate_control_steps(simulation):
    # Arrange
    control_times = []

    def record_control_time():
        control_times.append(simulation.data.time)

    def run_viewer(model, data):
        control_callback = mujoco.get_mjcb_control()
        control_callback(model, data)
        control_callback(model, data)
        data.time += model.opt.timestep
        control_callback(model, data)

    # Act
    with patch("mujoco.viewer.launch", side_effect=run_viewer):
        with patch("mujoco.viewer.launch_passive", side_effect=AssertionError("passive viewer used")):
            simulation.run_interactive(record_control_time)

    # Assert
    assert control_times == pytest.approx([0.0, simulation.model.opt.timestep])
    assert mujoco.get_mjcb_control() is None


def test_run_interactive_should_reset_runtime_when_viewer_rewinds_time(simulation):
    # Arrange
    control_times = []
    reset_times = []

    def record_control_time():
        control_times.append(simulation.data.time)

    def record_reset_time():
        reset_times.append(simulation.data.time)

    def run_viewer(model, data):
        control_callback = mujoco.get_mjcb_control()
        control_callback(model, data)
        data.time += model.opt.timestep
        simulation.quad.omega[:] = 2.0
        simulation.quad.omega_command[:] = 3.0
        simulation._collision_detected = True
        simulation._actual_trajectory_positions.append(np.ones(3))
        simulation.model.geom("actual_trajectory_segment000").rgba[3] = 0.9
        control_callback(model, data)
        mujoco.mj_resetData(model, data)
        control_callback(model, data)

    # Act
    with patch("mujoco.viewer.launch", side_effect=run_viewer):
        simulation.run_interactive(record_control_time, record_reset_time)

    # Assert
    assert control_times == pytest.approx([0.0, simulation.model.opt.timestep, 0.0])
    assert reset_times == pytest.approx([0.0])
    assert simulation.quad.omega == pytest.approx(np.zeros(4))
    assert simulation.quad.omega_command == pytest.approx(np.zeros(4))
    assert simulation.collision_detected is False
    assert simulation._actual_trajectory_positions == []
    assert simulation.model.geom("actual_trajectory_segment000").rgba[3] == pytest.approx(0.0)


def test_mujoco_simulation_should_extract_planning_obstacles_from_scene(simulation):
    # Arrange
    # Act
    first_obstacle = simulation.obstacles[0]
    obstacle_geom = simulation.model.geom("obstacle_00")

    # Assert
    assert first_obstacle == pytest.approx(np.array([3.7, 4.3, 4.0, 10.0, -3.4, -2.8]))
    assert obstacle_geom.contype != 0
    assert simulation.model.mat_rgba[obstacle_geom.matid, 3] == pytest.approx(0.65)


def test_set_trajectory_visualization_should_render_ned_path_as_non_colliding_segments(simulation):
    # Arrange
    positions = np.array([
        [2.0, 2.0, -2.0],
        [3.0, 4.0, -3.0],
        [5.0, 5.0, -4.0],
    ])

    # Act
    simulation.set_trajectory_visualization(positions)
    first_segment = simulation.model.geom("trajectory_segment000")
    first_segment_id = first_segment.id
    first_segment_center = simulation.data.geom_xpos[first_segment_id]
    first_segment_axis = simulation.data.geom_xmat[first_segment_id].reshape(3, 3)[:, 2]
    first_segment_half_length = first_segment.size[1]
    rendered_endpoints = np.array([
        first_segment_center - first_segment_axis * first_segment_half_length,
        first_segment_center + first_segment_axis * first_segment_half_length,
    ])
    expected_endpoints = np.array([[2.0, -2.0, 2.0], [3.0, -4.0, 3.0]])

    # Assert
    assert rendered_endpoints == pytest.approx(expected_endpoints)
    assert first_segment.size[0] == pytest.approx(0.025)
    assert simulation.model.geom_rgba[first_segment_id, 3] == pytest.approx(0.35)
    assert simulation.model.geom_rbound[first_segment_id] == pytest.approx(
        first_segment.size[0] + first_segment.size[1])
    assert first_segment.contype == 0
    assert first_segment.conaffinity == 0


def test_actual_trajectory_visualization_should_hide_takeoff_and_use_distinct_style(simulation):
    # Arrange
    simulation.set_trajectory_visualization(np.array([
        [1.0, 7.0, -1.3],
        [2.0, 7.0, -1.3],
    ]))
    simulation._record_actual_trajectory()
    actual_segment = simulation.model.geom("actual_trajectory_segment000")

    simulation.data.qpos[:3] = np.array([1.0, -7.0, 1.3])
    mujoco.mj_forward(simulation.model, simulation.data)
    simulation._sync_quad_state()
    simulation._record_actual_trajectory()

    simulation.data.time += 0.05
    simulation.data.qpos[:3] = np.array([2.0, -7.0, 1.3])
    mujoco.mj_forward(simulation.model, simulation.data)
    simulation._sync_quad_state()

    # Act
    simulation._record_actual_trajectory()

    # Assert
    planned_segment = simulation.model.geom("trajectory_segment000")
    assert actual_segment.rgba[3] > planned_segment.rgba[3]
    assert actual_segment.rgba[:3] != pytest.approx(planned_segment.rgba[:3])
    assert simulation.model.geom("actual_trajectory_segment001").rgba[3] == pytest.approx(0.0)


def test_set_trajectory_visualization_should_reject_invalid_positions(simulation):
    # Arrange
    invalid_positions = np.zeros((3, 2))

    # Act
    def set_invalid_trajectory():
        simulation.set_trajectory_visualization(invalid_positions)

    # Assert
    with pytest.raises(ValueError, match="trajectory positions"):
        set_invalid_trajectory()


def test_set_trajectory_visualization_should_preserve_endpoint_when_path_exceeds_capacity(simulation):
    # Arrange
    positions = np.column_stack((
        np.linspace(1.0, 23.0, 500),
        np.full(500, 7.0),
        np.linspace(-1.5, -2.0, 500),
    ))

    # Act
    simulation.set_trajectory_visualization(positions)
    last_segment = simulation.model.geom("trajectory_segment199")
    last_segment_id = last_segment.id
    last_segment_center = simulation.data.geom_xpos[last_segment_id]
    last_segment_axis = simulation.data.geom_xmat[last_segment_id].reshape(3, 3)[:, 2]
    rendered_endpoint = last_segment_center + last_segment_axis * last_segment.size[1]

    # Assert
    assert rendered_endpoint == pytest.approx(np.array([23.0, -7.0, 2.0]))
    assert simulation.model.geom_rgba[last_segment_id, 3] == pytest.approx(0.35)


def test_mujoco_scene_should_define_under_through_and_over_challenges(simulation):
    # Arrange
    under_bar = simulation.model.geom("obstacle_00")
    low_wall = simulation.model.geom("obstacle_01")
    first_ring_center = simulation.model.site("ring_center_00")
    second_ring_center = simulation.model.site("ring_center_01")

    # Act
    under_waypoint_altitude = -simulation.mission_waypoints[2, 2]
    over_waypoint_altitude = -simulation.mission_waypoints[4, 2]
    under_bar_lower_surface = under_bar.pos[2] - under_bar.size[2]
    low_wall_upper_surface = low_wall.pos[2] + low_wall.size[2]

    # Assert
    assert under_waypoint_altitude < under_bar_lower_surface
    assert over_waypoint_altitude > low_wall_upper_surface
    assert simulation.mission_waypoints[3] == pytest.approx(
        np.array([first_ring_center.pos[0], -first_ring_center.pos[1], -first_ring_center.pos[2]]))
    assert simulation.mission_waypoints[6] == pytest.approx(
        np.array([second_ring_center.pos[0], -second_ring_center.pos[1], -second_ring_center.pos[2]]))
    assert simulation.model.geom("ring_00_segment_00").type == mujoco.mjtGeom.mjGEOM_CAPSULE


def test_mujoco_scene_should_keep_clearance_above_planning_volume(simulation):
    # Arrange
    ceiling = simulation.model.geom("world_boundary_ceiling")

    # Act
    ceiling_lower_surface = ceiling.pos[2] - ceiling.size[2]
    maximum_planned_altitude = -simulation.space_limits[0, 2]

    # Assert
    assert ceiling_lower_surface >= maximum_planned_altitude + 0.1


def test_mujoco_simulation_should_reject_scene_without_required_vehicle(tmp_path):
    # Arrange
    invalid_scene = tmp_path / "invalid.xml"
    invalid_scene.write_text("<mujoco><worldbody/></mujoco>")

    # Act
    def create_simulation():
        return MujocoSimulation(invalid_scene)

    # Assert
    with pytest.raises(ValueError, match="quadrotor"):
        create_simulation()


def test_mujoco_simulation_should_reject_non_consecutive_mandatory_waypoints(tmp_path):
    # Arrange
    invalid_scene = tmp_path / "invalid_waypoints.xml"
    scene_content = DEFAULT_SCENE_PATH.read_text().replace(
        'name="waypoint_01"',
        'name="waypoint_07"',
    )
    invalid_scene.write_text(scene_content)

    # Act
    def create_simulation():
        return MujocoSimulation(invalid_scene)

    # Assert
    with pytest.raises(ValueError, match="consecutively numbered"):
        create_simulation()


def test_mujoco_simulation_should_frame_laboratory_course_without_including_ground_plane(simulation):
    # Arrange
    # Act
    scene_extent = simulation.model.stat.extent

    # Assert
    assert scene_extent < 30.0