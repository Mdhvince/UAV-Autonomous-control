import numpy as np
import pytest

from uav_ac.quadrotor.quad import Quad
from uav_ac.visualization.flight_dashboard import FlightDashboard


G = 9.81
DT = 0.01
N_STEPS = 12
FRAME_STEP = 4
DYNAMIC_TRACE_COUNT = 10  # 7 drone traces + 3 time cursors


@pytest.fixture
def quad():
    return Quad(g=G, dt=DT)


@pytest.fixture
def dashboard(quad):
    state_history = np.zeros((N_STEPS, 13))
    state_history[:, 0] = np.linspace(0.0, 1.0, N_STEPS)
    state_history[:, 3] = 1.0  # identity quaternion
    omega_history = np.ones((N_STEPS, 4))
    reference_trajectory = np.zeros((N_STEPS, 10))
    obstacles = np.array([[2.0, 3.0, 2.0, 3.0, -2.0, 0.0]])
    planned_path = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
    return FlightDashboard(quad, state_history, omega_history, reference_trajectory,
                           DT, frame_step=FRAME_STEP, drone_scale=1.0,
                           obstacles=obstacles, planned_path=planned_path)


def test_init_rejects_mismatched_history_lengths(quad):
    # Arrange
    state_history = np.zeros((N_STEPS, 13))
    state_history[:, 3] = 1.0
    omega_history = np.ones((N_STEPS - 1, 4))
    reference_trajectory = np.zeros((N_STEPS, 10))

    # Act / Assert
    with pytest.raises(ValueError):
        FlightDashboard(quad, state_history, omega_history, reference_trajectory, DT)


def test_rotor_positions_body_match_quad_torque_conventions(dashboard, quad):
    # Arrange
    a = quad.l

    # Act
    result = dashboard._rotor_positions_body()

    # Assert
    expected = np.array([[a, -a, 0.0], [a, a, 0.0], [-a, a, 0.0], [-a, -a, 0.0]])
    assert result == pytest.approx(expected)


def test_blade_angles_spin_in_opposite_directions_for_adjacent_rotors(dashboard):
    # Arrange
    last_index = N_STEPS - 1

    # Act
    angles = dashboard._blade_angles[last_index]

    # Assert
    expected_magnitude = N_STEPS * DT  # integral of a unit speed over the whole history
    assert angles == pytest.approx([expected_magnitude, -expected_magnitude,
                                    expected_magnitude, -expected_magnitude])


def test_to_world_applies_rotation_then_translation():
    # Arrange
    quarter_turn_about_z = np.array([[0.0, -1.0, 0.0],
                                     [1.0, 0.0, 0.0],
                                     [0.0, 0.0, 1.0]])
    points_body = np.array([[1.0, 0.0, 0.0]])
    position = np.array([10.0, 20.0, 30.0])

    # Act
    result = FlightDashboard._to_world(points_body, quarter_turn_about_z, position)

    # Assert
    assert result[0] == pytest.approx([10.0, 21.0, 30.0])


def test_euler_angles_deg_quarter_turn_about_z_gives_ninety_yaw():
    # Arrange
    half_angle = np.pi / 4
    q = np.array([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)])

    # Act
    roll, pitch, yaw = FlightDashboard._euler_angles_deg(q)

    # Assert
    assert roll == pytest.approx(0.0)
    assert pitch == pytest.approx(0.0)
    assert yaw == pytest.approx(90.0)


def test_euler_angles_deg_vectorizes_over_quaternion_history():
    # Arrange
    half_angle = np.pi / 4
    quaternions = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)],
    ])

    # Act
    result = FlightDashboard._euler_angles_deg(quaternions)

    # Assert
    assert result.shape == (2, 3)
    assert result[0] == pytest.approx([0.0, 0.0, 0.0])
    assert result[1] == pytest.approx([0.0, 0.0, 90.0])


def test_figure_renders_one_frame_per_frame_step(dashboard):
    # Arrange
    expected_frame_count = int(np.ceil(N_STEPS / FRAME_STEP))

    # Act
    fig = dashboard.figure()

    # Assert
    assert len(fig.frames) == expected_frame_count


def test_figure_frames_update_only_the_dynamic_traces(dashboard):
    # Arrange / Act
    fig = dashboard.figure()

    # Assert
    dynamic_trace_ids = list(range(len(fig.data) - DYNAMIC_TRACE_COUNT, len(fig.data)))
    for frame in fig.frames:
        assert len(frame.data) == DYNAMIC_TRACE_COUNT
        assert list(frame.traces) == dynamic_trace_ids


def test_figure_contains_obstacle_meshes_and_planned_path(dashboard):
    # Arrange / Act
    fig = dashboard.figure()

    # Assert
    trace_types = [trace.type for trace in fig.data]
    trace_names = [trace.name for trace in fig.data]
    assert "mesh3d" in trace_types
    assert "RRT path" in trace_names


def test_figure_reverses_y_and_z_display_axes_for_ned(dashboard):
    # Arrange / Act
    fig = dashboard.figure()

    # Assert
    # Reversing both axes is a proper rotation of the view: NED data shows altitude
    # upward without mirroring the attitude
    assert fig.layout.scene.yaxis.autorange == "reversed"
    assert fig.layout.scene.zaxis.autorange == "reversed"


def test_figure_hides_scene_grid_and_background(dashboard):
    # Arrange / Act
    fig = dashboard.figure()

    # Assert
    for axis in (fig.layout.scene.xaxis, fig.layout.scene.yaxis, fig.layout.scene.zaxis):
        assert axis.showbackground is False
        assert axis.showgrid is False
        assert axis.showticklabels is False


def test_merge_segments_separates_segments_with_none():
    # Arrange
    segments = [np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                np.array([[7.0, 8.0, 9.0]])]

    # Act
    xs, ys, zs = FlightDashboard._merge_segments(segments)

    # Assert
    assert xs == [1.0, 4.0, None, 7.0, None]
    assert ys == [2.0, 5.0, None, 8.0, None]
    assert zs == [3.0, 6.0, None, 9.0, None]
