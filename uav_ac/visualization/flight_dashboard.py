import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class FlightDashboard:
    """
    Single-view cockpit dashboard of the simulated quadrotor flight, rendered with plotly.

    Left: interactive 3D animation on a clean background (no grid) showing the drone
    frame placed and oriented from the simulated attitude quaternion, the four
    propellers spinning in their physical direction at the recorded speeds, the body
    axes, the executed trajectory against the minimum snap reference, and optionally
    the obstacles and the RRT path.

    Right: the in-flight controller response as four time series synchronized with
    the animation through a moving time cursor: position tracking per axis versus
    the reference, attitude angles, propeller speeds, and position tracking error.

    The animation is played with the Play/Pause buttons or scrubbed with the time
    slider, and the title displays time, roll/pitch/yaw and propeller speeds. With
    follow_drone=True the camera tracks the drone during playback (chase view); the
    view can still be rotated freely while paused.
    """

    # Spin direction of each rotor about the body z axis. Derived from the reactive
    # torque signs in Quad (tau_1, tau_3 < 0 and tau_2, tau_4 > 0): the reaction
    # torque on the body opposes the rotor spin direction.
    ROTOR_SPIN_DIRECTIONS = np.array([1.0, -1.0, 1.0, -1.0])

    REFERENCE_TRAJECTORY_COLOR = "#0072B2"
    EXECUTED_TRAJECTORY_COLOR = "#D55E00"
    PROPELLER_POSITIVE_SPIN_COLOR = "#009E73"
    PROPELLER_NEGATIVE_SPIN_COLOR = "#8E5AC8"
    PROPELLER_POSITIVE_SPIN_LIGHT_COLOR = "#5BC8A7"
    PROPELLER_NEGATIVE_SPIN_LIGHT_COLOR = "#BFA1E8"
    STRUCTURE_COLOR = "#333333"
    OBSTACLE_COLOR = "#9AA0A6"
    PLANNED_PATH_COLOR = "#8C8C8C"
    TIME_CURSOR_COLOR = "#7F7F7F"
    BODY_AXIS_COLORS = ("#D62728", "#2CA02C", "#1F77B4")  # x, y, z (robotics RGB convention)

    RESPONSE_SUBPLOT_COUNT = 4

    # Chase camera offset from the drone, in normalized scene coordinates
    CHASE_CAMERA_OFFSET = np.array([0.65, 0.65, 0.4])

    def __init__(self, quad, state_history: np.ndarray, omega_history: np.ndarray,
                 reference_trajectory: np.ndarray, dt: float,
                 frame_step: int = 5, drone_scale: float = 3.0,
                 obstacles: np.ndarray = None, planned_path: np.ndarray = None,
                 follow_drone: bool = False):
        """
        :param quad: simulated Quad instance, source of the geometry (arm length) and
                     of the quaternion to rotation matrix conversion
        :param state_history: (N, 13) state rows [x, y, z, q0, q1, q2, q3, x_dot,
                              y_dot, z_dot, p, q, r], one row every dt seconds
        :param omega_history: (N, 4) propeller speeds [rad/s], aligned with state_history
        :param reference_trajectory: (M, >=3) reference trajectory, first three columns are x, y, z
        :param dt: time elapsed between two consecutive history rows [s]
        :param frame_step: number of history rows between two rendered frames
        :param drone_scale: visual magnification of the drone geometry, for visibility in a
                            large scene; 1.0 renders the true size (position, attitude and
                            propeller speeds are untouched)
        :param obstacles: optional (K, 6) cuboids [xmin, xmax, ymin, ymax, zmin, zmax] to render
        :param planned_path: optional (P, 3) waypoints of the global path (e.g. RRT) to render
        :param follow_drone: when True the camera tracks the drone during playback (chase view)
        """
        if len(state_history) != len(omega_history):
            raise ValueError(
                f"state_history ({len(state_history)} rows) and omega_history "
                f"({len(omega_history)} rows) must have the same length"
            )

        self._positions = state_history[:, :3]
        self._quaternions = state_history[:, 3:7]
        self._rotations = np.array([quad.quat_to_rot(q) for q in self._quaternions])
        self._omega_history = omega_history
        self._reference = reference_trajectory[:, :3]
        self._dt = dt
        self._frame_step = frame_step
        self._obstacles = obstacles
        self._planned_path = planned_path

        self._arm_half_length = quad.l * drone_scale
        self._propeller_radius = 0.75 * quad.l * drone_scale
        self._body_axis_length = 2.5 * quad.l * drone_scale

        # Blade angle is the integral of the propeller speed, signed by spin direction
        self._blade_angles = np.cumsum(omega_history * dt, axis=0) * self.ROTOR_SPIN_DIRECTIONS
        self._frame_indices = np.arange(0, len(state_history), frame_step)

        self._times = np.arange(len(state_history)) * dt
        self._reference_times = np.arange(len(reference_trajectory)) * dt
        self._euler_history_deg = self._euler_angles_deg(self._quaternions)
        common_length = min(len(self._positions), len(self._reference))
        self._tracking_errors = np.linalg.norm(
            self._positions[:common_length] - self._reference[:common_length], axis=1)
        self._response_y_ranges = self._compute_response_y_ranges()

        self._follow_drone = follow_drone
        self._scene_ranges, self._scene_aspect_ratio = self._compute_scene_ranges()
        self._fig = None

    def figure(self) -> go.Figure:
        """
        :return: the dashboard figure, built on first call then cached
        """
        if self._fig is None:
            self._fig = self._build_figure()
        return self._fig

    def show(self):
        """Open the dashboard in the browser, paused on the first frame."""
        self.figure().show(auto_play=False)

    def save(self, filename: str):
        """
        :param filename: path of the standalone html file to write
        """
        self.figure().write_html(filename, auto_play=False)

    def _build_figure(self) -> go.Figure:
        fig = make_subplots(
            rows=4, cols=2,
            column_widths=[0.62, 0.38],
            specs=[[{"type": "scene", "rowspan": 4}, {"type": "xy"}],
                   [None, {"type": "xy"}],
                   [None, {"type": "xy"}],
                   [None, {"type": "xy"}]],
            subplot_titles=("", "Position [m] (NED)", "Attitude [deg]",
                            "Propeller speed [rad/s]", "Tracking error [m]"),
            horizontal_spacing=0.05,
            vertical_spacing=0.08,
            shared_xaxes=True,
        )

        for trace in self._static_scene_traces():
            fig.add_trace(trace, row=1, col=1)
        for trace, row in self._response_traces():
            fig.add_trace(trace, row=row, col=2)

        first_index = int(self._frame_indices[0])
        dynamic_traces = self._dynamic_traces(first_index)
        dynamic_trace_ids = list(range(len(fig.data), len(fig.data) + len(dynamic_traces)))
        fig.add_traces(
            [trace for trace, _, _ in dynamic_traces],
            rows=[row for _, row, _ in dynamic_traces],
            cols=[col for _, _, col in dynamic_traces],
        )

        fig.frames = [
            go.Frame(
                data=[trace for trace, _, _ in self._dynamic_traces(int(index))],
                traces=dynamic_trace_ids,
                name=self._frame_name(int(index)),
                layout=self._frame_layout(int(index)),
            )
            for index in self._frame_indices
        ]

        # Hide grid, background panes and ticks of the 3D scene but keep the axes
        # objects alive: the y and z display reversals below must stay in effect.
        hidden_axis = dict(showbackground=False, showgrid=False, zeroline=False,
                           showticklabels=False, showspikes=False, title=dict(text=""))
        x_range, y_range, z_range = self._scene_ranges
        scene = dict(
            # explicit ranges (instead of autorange) so the chase camera can map the
            # drone position to normalized scene coordinates deterministically
            aspectmode="manual",
            aspectratio=dict(x=self._scene_aspect_ratio[0],
                             y=self._scene_aspect_ratio[1],
                             z=self._scene_aspect_ratio[2]),
            # Data is in NED (z down). Reversing BOTH y and z display axes (descending
            # ranges) is a proper rotation (not a mirror), so the scene shows altitude
            # upward while keeping a physically correct attitude.
            xaxis=dict(range=list(x_range), **hidden_axis),
            yaxis=dict(range=list(y_range), **hidden_axis),
            zaxis=dict(range=list(z_range), **hidden_axis),
        )
        if self._follow_drone:
            scene["camera"] = self._chase_camera(first_index)

        fig.update_layout(
            title=dict(text=self._frame_title(first_index), font=dict(size=13)),
            scene=scene,
            legend=dict(x=0.0, y=0.99, xanchor="left", font=dict(size=11)),
            updatemenus=[self._play_controls()],
            sliders=[self._time_slider()],
            height=820,
            plot_bgcolor="white",
        )
        # response subplots: light grid on white, in line with the clean 3D scene
        fig.update_xaxes(showgrid=True, gridcolor="#E6E6E6", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="#E6E6E6", zeroline=False)
        fig.update_xaxes(title_text="t [s]", row=self.RESPONSE_SUBPLOT_COUNT, col=2)
        return fig

    def _frame_layout(self, index: int) -> go.Layout:
        layout = go.Layout(title=dict(text=self._frame_title(index)))
        if self._follow_drone:
            layout.scene = dict(camera=self._chase_camera(index))
        return layout

    def _compute_scene_ranges(self) -> tuple:
        """
        Fixed display ranges and aspect ratio of the 3D scene, covering every plotted
        element. The y and z ranges are descending on purpose (NED display rotation).
        :return: ((x_range, y_range, z_range), aspect_ratio) with spans-proportional ratio
        """
        points = [self._positions, self._reference]
        if self._planned_path is not None:
            points.append(self._planned_path)
        if self._obstacles is not None:
            corners_low = self._obstacles[:, [0, 2, 4]]
            corners_high = self._obstacles[:, [1, 3, 5]]
            points.extend([corners_low, corners_high])
        stacked = np.vstack(points)

        low = stacked.min(axis=0)
        high = stacked.max(axis=0)
        # a flat axis (e.g. constant altitude) would make the camera normalization
        # divide by zero: give it a minimal 1 m extent
        is_flat_axis = (high - low) < 1e-9
        low = np.where(is_flat_axis, low - 0.5, low)
        high = np.where(is_flat_axis, high + 0.5, high)
        margin = 0.05 * (high - low)
        low, high = low - margin, high + margin

        ranges = ((low[0], high[0]), (high[1], low[1]), (high[2], low[2]))
        spans = high - low
        aspect_ratio = spans / spans.max()
        return ranges, aspect_ratio

    def _normalized_scene_position(self, position: np.ndarray) -> np.ndarray:
        """Map a NED point to the normalized scene coordinates used by the camera."""
        normalized = np.zeros(3)
        for axis_id in range(3):
            range_start, range_end = self._scene_ranges[axis_id]
            fraction = (position[axis_id] - range_start) / (range_end - range_start)
            normalized[axis_id] = (fraction - 0.5) * self._scene_aspect_ratio[axis_id]
        return normalized

    def _chase_camera(self, index: int) -> dict:
        center = self._normalized_scene_position(self._positions[index])
        eye = center + self.CHASE_CAMERA_OFFSET
        return dict(
            center=dict(x=center[0], y=center[1], z=center[2]),
            eye=dict(x=eye[0], y=eye[1], z=eye[2]),
            up=dict(x=0, y=0, z=1),
        )

    def _static_scene_traces(self) -> list:
        traces = [
            go.Scatter3d(
                x=self._reference[:, 0], y=self._reference[:, 1], z=self._reference[:, 2],
                mode="lines",
                line=dict(color=self.REFERENCE_TRAJECTORY_COLOR, width=3, dash="dash"),
                name="Reference trajectory (minimum snap)",
            ),
            go.Scatter3d(
                x=self._positions[:, 0], y=self._positions[:, 1], z=self._positions[:, 2],
                mode="lines",
                line=dict(color=self.EXECUTED_TRAJECTORY_COLOR, width=2),
                opacity=0.35,
                name="Executed trajectory",
            ),
            go.Scatter3d(
                x=[self._positions[0, 0]], y=[self._positions[0, 1]], z=[self._positions[0, 2]],
                mode="markers",
                marker=dict(size=5, color="red"),
                name="Start",
            ),
            go.Scatter3d(
                x=[self._reference[-1, 0]], y=[self._reference[-1, 1]], z=[self._reference[-1, 2]],
                mode="markers",
                marker=dict(size=5, color="green"),
                name="Goal",
            ),
        ]
        if self._planned_path is not None:
            traces.append(go.Scatter3d(
                x=self._planned_path[:, 0], y=self._planned_path[:, 1], z=self._planned_path[:, 2],
                mode="lines+markers",
                line=dict(color=self.PLANNED_PATH_COLOR, width=2, dash="dot"),
                marker=dict(size=3, color=self.PLANNED_PATH_COLOR),
                name="RRT path",
            ))
        if self._obstacles is not None:
            for obstacle_id, bounds in enumerate(self._obstacles):
                traces.append(self._cuboid_mesh_trace(
                    bounds, self.OBSTACLE_COLOR, opacity=0.1,
                    name="Obstacles", showlegend=obstacle_id == 0,
                ))
        return traces

    def _response_traces(self) -> list:
        """Time series of the in-flight controller response, as (trace, subplot row) pairs."""
        traces = []
        axis_labels = ("x", "y", "z")
        for axis_id, label in enumerate(axis_labels):
            color = self.BODY_AXIS_COLORS[axis_id]
            traces.append((go.Scatter(
                x=self._reference_times, y=self._reference[:, axis_id],
                mode="lines", line=dict(color=color, width=1, dash="dash"),
                name=f"{label} reference", showlegend=False,
            ), 1))
            traces.append((go.Scatter(
                x=self._times, y=self._positions[:, axis_id],
                mode="lines", line=dict(color=color, width=1.5),
                name=f"{label} actual", showlegend=False,
            ), 1))

        for angle_id, label in enumerate(("roll", "pitch", "yaw")):
            traces.append((go.Scatter(
                x=self._times, y=self._euler_history_deg[:, angle_id],
                mode="lines", line=dict(color=self.BODY_AXIS_COLORS[angle_id], width=1.5),
                name=label, showlegend=False,
            ), 2))

        propeller_colors = (self.PROPELLER_POSITIVE_SPIN_COLOR, self.PROPELLER_NEGATIVE_SPIN_COLOR,
                            self.PROPELLER_POSITIVE_SPIN_LIGHT_COLOR, self.PROPELLER_NEGATIVE_SPIN_LIGHT_COLOR)
        for rotor_id in range(4):
            traces.append((go.Scatter(
                x=self._times, y=self._omega_history[:, rotor_id],
                mode="lines", line=dict(color=propeller_colors[rotor_id], width=1.5),
                name=f"omega {rotor_id + 1}", showlegend=False,
            ), 3))

        traces.append((go.Scatter(
            x=self._times[:len(self._tracking_errors)], y=self._tracking_errors,
            mode="lines", line=dict(color=self.EXECUTED_TRAJECTORY_COLOR, width=1.5),
            name="tracking error", showlegend=False,
        ), 4))
        return traces

    def _compute_response_y_ranges(self) -> list:
        """Vertical extent of each response subplot, used to draw the time cursors."""
        ranges = []
        for values in (
            np.concatenate((self._positions.ravel(), self._reference.ravel())),
            self._euler_history_deg.ravel(),
            self._omega_history.ravel(),
            self._tracking_errors,
        ):
            low, high = float(np.min(values)), float(np.max(values))
            margin = 0.05 * (high - low) if high > low else 1.0
            ranges.append((low - margin, high + margin))
        return ranges

    def _dynamic_traces(self, index: int) -> list:
        """All animated traces for one time step, as (trace, row, col) tuples."""
        rotation = self._rotations[index]
        position = self._positions[index]
        rotors_world = self._to_world(self._rotor_positions_body(), rotation, position)

        traces = [
            (self._trail_trace(index), 1, 1),
            (self._structure_trace(rotors_world, position), 1, 1),
            (self._propeller_trace(index, rotation, position, rotor_ids=(0, 2),
                                   color=self.PROPELLER_POSITIVE_SPIN_COLOR,
                                   name="Propellers 1 and 3 (+z_b spin)"), 1, 1),
            (self._propeller_trace(index, rotation, position, rotor_ids=(1, 3),
                                   color=self.PROPELLER_NEGATIVE_SPIN_COLOR,
                                   name="Propellers 2 and 4 (-z_b spin)"), 1, 1),
        ]
        traces.extend((trace, 1, 1) for trace in self._body_axes_traces(rotation, position))
        traces.extend(
            (self._time_cursor_trace(index, subplot_id), subplot_id + 1, 2)
            for subplot_id in range(self.RESPONSE_SUBPLOT_COUNT)
        )
        return traces

    def _time_cursor_trace(self, index: int, subplot_id: int) -> go.Scatter:
        t = index * self._dt
        low, high = self._response_y_ranges[subplot_id]
        return go.Scatter(
            x=[t, t], y=[low, high],
            mode="lines",
            line=dict(color=self.TIME_CURSOR_COLOR, width=1),
            name="t",
            showlegend=False,
            hoverinfo="skip",
        )

    def _rotor_positions_body(self) -> np.ndarray:
        """
        Rotor centers in body frame, ordered rotor 1 to 4, consistent with the moments
        in Quad: tau_x = l (f1 + f4 - f2 - f3) and tau_y = l (f1 + f2 - f3 - f4).
        """
        a = self._arm_half_length
        return np.array([
            [a, -a, 0.0],
            [a, a, 0.0],
            [-a, a, 0.0],
            [-a, -a, 0.0],
        ])

    def _trail_trace(self, index: int) -> go.Scatter3d:
        trail = self._positions[:index + 1:self._frame_step]
        return go.Scatter3d(
            x=trail[:, 0], y=trail[:, 1], z=trail[:, 2],
            mode="lines",
            line=dict(color=self.EXECUTED_TRAJECTORY_COLOR, width=5),
            name="Executed so far",
            showlegend=False,
        )

    def _structure_trace(self, rotors_world: np.ndarray, position: np.ndarray) -> go.Scatter3d:
        xs, ys, zs = self._merge_segments([
            np.array([rotors_world[0], position, rotors_world[2]]),
            np.array([rotors_world[1], position, rotors_world[3]]),
        ])
        return go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines+markers",
            line=dict(color=self.STRUCTURE_COLOR, width=6),
            marker=dict(size=3, color=self.STRUCTURE_COLOR),
            name="Drone frame",
            showlegend=False,
        )

    def _propeller_trace(self, index: int, rotation: np.ndarray, position: np.ndarray,
                         rotor_ids: tuple, color: str, name: str) -> go.Scatter3d:
        rotors_body = self._rotor_positions_body()
        circle_angles = np.linspace(0.0, 2.0 * np.pi, 24)
        segments = []
        for rotor_id in rotor_ids:
            center = rotors_body[rotor_id]
            circle = center + self._propeller_radius * np.column_stack(
                (np.cos(circle_angles), np.sin(circle_angles), np.zeros_like(circle_angles)))
            blade_angle = self._blade_angles[index, rotor_id]
            blade_direction = np.array([np.cos(blade_angle), np.sin(blade_angle), 0.0])
            blade = np.array([center - self._propeller_radius * blade_direction,
                              center + self._propeller_radius * blade_direction])
            segments.append(self._to_world(circle, rotation, position))
            segments.append(self._to_world(blade, rotation, position))
        xs, ys, zs = self._merge_segments(segments)
        return go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color=color, width=4),
            name=name,
        )

    def _body_axes_traces(self, rotation: np.ndarray, position: np.ndarray) -> list:
        traces = []
        for axis_id, (label, color) in enumerate(zip(("x_b", "y_b", "z_b"), self.BODY_AXIS_COLORS)):
            tip = position + rotation[:, axis_id] * self._body_axis_length
            traces.append(go.Scatter3d(
                x=[position[0], tip[0]], y=[position[1], tip[1]], z=[position[2], tip[2]],
                mode="lines+text",
                text=["", label],
                textposition="top center",
                textfont=dict(color=color, size=12),
                line=dict(color=color, width=5),
                name=f"Body axis {label}",
                showlegend=False,
            ))
        return traces

    def _play_controls(self) -> dict:
        frame_duration_ms = self._dt * self._frame_step * 1000.0
        return dict(
            type="buttons",
            direction="left",
            x=0.0, y=0, xanchor="left", yanchor="top",
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=frame_duration_ms, redraw=True),
                                      transition=dict(duration=0), fromcurrent=True)]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False),
                                        transition=dict(duration=0))]),
            ],
        )

    def _time_slider(self) -> dict:
        steps = [
            dict(
                method="animate",
                args=[[self._frame_name(int(index))],
                      dict(mode="immediate", frame=dict(duration=0, redraw=True),
                           transition=dict(duration=0))],
                label=f"{index * self._dt:.1f}",
            )
            for index in self._frame_indices
        ]
        # x offset keeps the current value label clear of the Play/Pause buttons
        return dict(steps=steps, x=0.15, len=0.85, currentvalue=dict(prefix="t [s] = "))

    def _frame_name(self, index: int) -> str:
        return f"{index * self._dt:.2f}"

    def _frame_title(self, index: int) -> str:
        t = index * self._dt
        phi, theta, psi = self._euler_history_deg[index]
        w = self._omega_history[index]
        return (f"t = {t:6.2f} s | roll = {phi:+6.1f} deg, pitch = {theta:+6.1f} deg, "
                f"yaw = {psi:+6.1f} deg | prop speed [rad/s] = "
                f"[{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}, {w[3]:.2f}]")

    @staticmethod
    def _euler_angles_deg(q: np.ndarray) -> np.ndarray:
        """
        Roll, pitch, yaw [deg] from quaternions, same convention as Quad.phi/theta/psi.
        :param q: quaternion array of shape (..., 4) with the scalar part first
        :return: angles array of shape (..., 3)
        """
        q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        phi = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2))
        theta = np.arcsin(np.clip(2 * (q0 * q2 - q3 * q1), -1.0, 1.0))
        psi = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))
        return np.degrees(np.stack((phi, theta, psi), axis=-1))

    @staticmethod
    def _cuboid_mesh_trace(bounds: np.ndarray, color: str, opacity: float,
                           name: str, showlegend: bool) -> go.Mesh3d:
        """
        :param bounds: cuboid bounds [x_min, x_max, y_min, y_max, z_min, z_max]
        """
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        return go.Mesh3d(
            x=[x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_min],
            y=[y_min, y_min, y_max, y_max, y_min, y_min, y_max, y_max],
            z=[z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max],
            i=[0, 0, 4, 4, 0, 0, 3, 3, 0, 0, 1, 1],
            j=[1, 2, 6, 7, 4, 5, 2, 6, 3, 7, 5, 6],
            k=[2, 3, 5, 6, 5, 1, 6, 7, 7, 4, 6, 2],
            color=color,
            opacity=opacity,
            flatshading=True,
            name=name,
            showlegend=showlegend,
            hoverinfo="skip",
        )

    @staticmethod
    def _to_world(points_body: np.ndarray, rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
        return points_body @ rotation.T + position

    @staticmethod
    def _merge_segments(segments: list) -> tuple:
        """Concatenate line segments into single x, y, z lists separated by None gaps."""
        xs, ys, zs = [], [], []
        for segment in segments:
            xs.extend([*segment[:, 0], None])
            ys.extend([*segment[:, 1], None])
            zs.extend([*segment[:, 2], None])
        return xs, ys, zs


if __name__ == "__main__":
    pass
