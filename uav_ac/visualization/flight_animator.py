import numpy as np
import plotly.graph_objects as go


class FlightAnimator:
    """
    Interactive 3D animation of the simulated quadrotor flight, rendered with plotly.

    At every rendered time step the figure shows:
    - the drone frame (arms and rotor hubs) placed and oriented from the simulated
      position and attitude quaternion,
    - the four propellers, each spinning in its physical direction at the speed
      recorded in the propeller speed history,
    - the body axes attached to the drone center,
    - the executed trajectory trail against the minimum snap reference trajectory.

    The animation is played with the Play/Pause buttons or scrubbed with the time
    slider, and the title displays time, roll/pitch/yaw and propeller speeds.
    """

    # Spin direction of each rotor about the body z axis. Derived from the reactive
    # torque signs in Quad (tau_1, tau_3 < 0 and tau_2, tau_4 > 0): the reaction
    # torque on the body opposes the rotor spin direction.
    ROTOR_SPIN_DIRECTIONS = np.array([1.0, -1.0, 1.0, -1.0])

    REFERENCE_TRAJECTORY_COLOR = "#0072B2"
    EXECUTED_TRAJECTORY_COLOR = "#D55E00"
    PROPELLER_POSITIVE_SPIN_COLOR = "#009E73"
    PROPELLER_NEGATIVE_SPIN_COLOR = "#8E5AC8"
    STRUCTURE_COLOR = "#333333"
    BODY_AXIS_COLORS = ("#D62728", "#2CA02C", "#1F77B4")  # x, y, z (robotics RGB convention)

    def __init__(self, quad, state_history: np.ndarray, omega_history: np.ndarray,
                 reference_trajectory: np.ndarray, dt: float,
                 frame_step: int = 5, drone_scale: float = 6.0):
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
                            large scene (position, attitude and propeller speeds are untouched)
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

        self._arm_half_length = quad.l * drone_scale
        self._propeller_radius = 0.75 * quad.l * drone_scale
        self._body_axis_length = 2.5 * quad.l * drone_scale

        # Blade angle is the integral of the propeller speed, signed by spin direction
        self._blade_angles = np.cumsum(omega_history * dt, axis=0) * self.ROTOR_SPIN_DIRECTIONS
        self._frame_indices = np.arange(0, len(state_history), frame_step)
        self._fig = None

    def figure(self) -> go.Figure:
        """
        :return: the animated figure, built on first call then cached
        """
        if self._fig is None:
            self._fig = self._build_figure()
        return self._fig

    def show(self):
        """Open the animation in the browser, paused on the first frame."""
        self.figure().show(auto_play=False)

    def save(self, filename: str):
        """
        :param filename: path of the standalone html file to write
        """
        self.figure().write_html(filename, auto_play=False)

    def _build_figure(self) -> go.Figure:
        static_traces = self._static_traces()
        first_index = int(self._frame_indices[0])
        dynamic_traces = self._drone_traces(first_index)
        dynamic_trace_ids = list(range(len(static_traces), len(static_traces) + len(dynamic_traces)))

        frames = [
            go.Frame(
                data=self._drone_traces(int(index)),
                traces=dynamic_trace_ids,
                name=self._frame_name(int(index)),
                layout=go.Layout(title=dict(text=self._frame_title(int(index)))),
            )
            for index in self._frame_indices
        ]

        fig = go.Figure(data=static_traces + dynamic_traces, frames=frames)
        fig.update_layout(
            title=dict(text=self._frame_title(first_index), font=dict(size=13)),
            # Data is in NED (z down). Reversing BOTH y and z display axes is a proper
            # rotation (not a mirror), so the scene shows altitude upward while keeping
            # true NED coordinates on the axes and a physically correct attitude.
            scene=dict(
                aspectmode="data",
                xaxis_title="x [m]",
                yaxis=dict(autorange="reversed", title=dict(text="y [m]")),
                zaxis=dict(autorange="reversed", title=dict(text="z [m] (NED, down)")),
            ),
            legend=dict(x=0.99, y=0.99, xanchor="right"),
            updatemenus=[self._play_controls()],
            sliders=[self._time_slider()],
        )
        return fig

    def _static_traces(self) -> list:
        return [
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

    def _drone_traces(self, index: int) -> list:
        rotation = self._rotations[index]
        position = self._positions[index]
        rotors_world = self._to_world(self._rotor_positions_body(), rotation, position)

        traces = [
            self._trail_trace(index),
            self._structure_trace(rotors_world, position),
            self._propeller_trace(index, rotation, position, rotor_ids=(0, 2),
                                  color=self.PROPELLER_POSITIVE_SPIN_COLOR, name="Propellers 1 and 3 (+z_b spin)"),
            self._propeller_trace(index, rotation, position, rotor_ids=(1, 3),
                                  color=self.PROPELLER_NEGATIVE_SPIN_COLOR, name="Propellers 2 and 4 (-z_b spin)"),
        ]
        traces.extend(self._body_axes_traces(rotation, position))
        return traces

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
        phi, theta, psi = self._euler_angles_deg(self._quaternions[index])
        w = self._omega_history[index]
        return (f"t = {t:6.2f} s | roll = {phi:+6.1f} deg, pitch = {theta:+6.1f} deg, "
                f"yaw = {psi:+6.1f} deg | prop speed [rad/s] = "
                f"[{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}, {w[3]:.2f}]")

    @staticmethod
    def _euler_angles_deg(q: np.ndarray) -> np.ndarray:
        """Roll, pitch, yaw [deg] from a quaternion, same convention as Quad.phi/theta/psi."""
        phi = np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
        theta = np.arcsin(np.clip(2 * (q[0] * q[2] - q[3] * q[1]), -1.0, 1.0))
        psi = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        return np.degrees([phi, theta, psi])

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
