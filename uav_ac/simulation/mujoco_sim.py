from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np

from uav_ac.quadrotor.quad import Quad


ENU_TO_NED = np.diag([1.0, -1.0, -1.0])
DEFAULT_SCENE_PATH = Path(__file__).parent / "models" / "lab_course.xml"
TRAJECTORY_SEGMENT_COUNT = 200
TRAJECTORY_COLOR = np.array([1.0, 0.25, 0.05, 0.35])
ACTUAL_TRAJECTORY_SEGMENT_COUNT = 200
ACTUAL_TRAJECTORY_COLOR = np.array([0.1, 0.4, 1.0, 0.9])
ACTUAL_TRAJECTORY_SAMPLE_INTERVAL = 0.05
TAKEOFF_HEIGHT = 0.1


def mujoco_to_ned_state(
        position: np.ndarray,
        quaternion: np.ndarray,
        velocity: np.ndarray,
) -> np.ndarray:
    """
    Convert MuJoCo ENU/FLU free-joint state to the NED/FRD convention.

    :param position: world position in ENU
    :param quaternion: FLU-to-ENU quaternion in scalar-first order
    :param velocity: world linear velocity followed by FLU angular velocity
    :return: controller state vector
    """
    position = _vector(position, 3, "position")
    quaternion = _vector(quaternion, 4, "quaternion")
    velocity = _vector(velocity, 6, "velocity")
    quaternion_norm = np.linalg.norm(quaternion)
    if quaternion_norm == 0:
        raise ValueError("MuJoCo quaternion cannot be zero")

    state = np.empty(13)
    state[:3] = ENU_TO_NED @ position
    state[3:7] = quaternion / quaternion_norm * np.array([1.0, 1.0, -1.0, -1.0])
    state[7:10] = ENU_TO_NED @ velocity[:3]
    state[10:13] = ENU_TO_NED @ velocity[3:]
    return state


class MujocoSimulation:
    """Simulate the quadrotor rigid-body dynamics and collisions with MuJoCo."""

    def __init__(self, model_path: str | Path = DEFAULT_SCENE_PATH):
        """
        Load the vehicle, mission and obstacle geometry from a MuJoCo scene.

        :param model_path: MJCF scene containing the required quadrotor and mission data
        """
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self._body_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
        self._body_geom_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "body")
        self._ground_geom_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self._rotor_site_ids = np.array([_named_id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, f"rotor_{index}") for index in range(4)])
        self._rotor_spin_directions = self.model.site_user[self._rotor_site_ids, 0].copy()
        self._trajectory_segment_ids = np.array([_named_id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"trajectory_segment{index:03d}",
        ) for index in range(TRAJECTORY_SEGMENT_COUNT)])
        self._actual_trajectory_segment_ids = np.array([_named_id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"actual_trajectory_segment{index:03d}",
        ) for index in range(ACTUAL_TRAJECTORY_SEGMENT_COUNT)])
        self.quad = _create_quad(self.model, self._body_id, self._rotor_site_ids)
        self._collision_detected = False
        self._has_taken_off = False
        self._actual_trajectory_positions = []
        self._next_actual_trajectory_sample_time = 0.0

        mujoco.mj_forward(self.model, self.data)
        self._sync_quad_state()
        self.start_position = self.quad.position.copy()
        goal_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal")
        self.goal_position = ENU_TO_NED @ self.data.site_xpos[goal_id]
        self.mission_waypoints = _extract_mission_waypoints(
            self.model, self.data, self.start_position, self.goal_position)
        self.space_limits = _numeric(self.model, "planning_bounds", 6).reshape(2, 3)
        self.obstacles = _extract_obstacles(self.model, self.data)
        if self.has_collision:
            raise ValueError("quadrotor starts in collision")

    @property
    def has_collision(self) -> bool:
        """Return whether the quadrotor currently touches world geometry."""
        return self.data.ncon > 0

    @property
    def collision_detected(self) -> bool:
        """Return whether any collision has occurred since initialization."""
        return self._collision_detected

    def set_trajectory_visualization(self, positions: np.ndarray) -> None:
        """Display a sampled NED trajectory using non-colliding MuJoCo capsules."""
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
            raise ValueError("trajectory positions must have shape (n, 3) with n >= 2")
        if not np.all(np.isfinite(positions)):
            raise ValueError("trajectory positions must be finite")

        self._set_trajectory_segments(
            positions, self._trajectory_segment_ids, TRAJECTORY_COLOR)
        mujoco.mj_forward(self.model, self.data)

    def _set_trajectory_segments(
            self,
            positions: np.ndarray,
            segment_ids: np.ndarray,
            color: np.ndarray,
    ) -> None:
        max_points = len(segment_ids) + 1
        if len(positions) > max_points:
            sample_indices = np.linspace(0, len(positions) - 1, max_points).round().astype(int)
            positions = positions[sample_indices]

        positions_enu = positions @ ENU_TO_NED
        self.model.geom_rgba[segment_ids, 3] = 0.0
        segment_index = 0
        for start, end in zip(positions_enu[:-1], positions_enu[1:]):
            direction = end - start
            length = np.linalg.norm(direction)
            if length == 0:
                continue

            geom_id = segment_ids[segment_index]
            self.model.geom_pos[geom_id] = (start + end) / 2
            self.model.geom_size[geom_id, 1] = length / 2
            self.model.geom_rbound[geom_id] = self.model.geom_size[geom_id, 0] + length / 2
            mujoco.mju_quatZ2Vec(self.model.geom_quat[geom_id], direction)
            self.model.geom_sameframe[geom_id] = mujoco.mjtSameFrame.mjSAMEFRAME_NONE
            self.model.geom_rgba[geom_id] = color
            segment_index += 1

    def step(self) -> np.ndarray:
        """Advance MuJoCo by one inner-loop time step using current rotor speeds."""
        self._apply_rotor_forces()
        mujoco.mj_step(self.model, self.data)
        self._sync_quad_state()
        self._record_collisions()
        self._record_actual_trajectory()
        return self.quad.X.copy()

    def run_interactive(
            self,
            control_step: Callable[[], None],
            reset_control: Callable[[], None] | None = None,
    ) -> None:
        """
        Run the native MuJoCo viewer while the supplied controller drives the rotors.

        :param control_step: one inner-loop update of the existing flight controller
        :param reset_control: reset the controller when the viewer resets the simulation
        """
        from mujoco import viewer

        last_control_time = None

        def apply_control(model, data):
            nonlocal last_control_time
            if last_control_time is not None and data.time < last_control_time:
                self._reset_runtime_state()
                if reset_control is not None:
                    reset_control()
                last_control_time = None

            self._sync_quad_state()
            self._record_actual_trajectory()
            if last_control_time is None or data.time > last_control_time:
                control_step()
                last_control_time = data.time
            self._apply_rotor_forces()
            self._record_collisions()

        mujoco.set_mjcb_control(apply_control)
        try:
            viewer.launch(self.model, self.data)
        finally:
            mujoco.set_mjcb_control(None)

    def _reset_runtime_state(self) -> None:
        self.quad.omega.fill(0.0)
        self.quad.omega_command.fill(0.0)
        self.data.qfrc_applied.fill(0.0)
        self._collision_detected = False
        self._has_taken_off = False
        self._actual_trajectory_positions.clear()
        self._next_actual_trajectory_sample_time = 0.0
        self.model.geom_rgba[self._actual_trajectory_segment_ids, 3] = 0.0
        self._sync_quad_state()

    def _record_actual_trajectory(self) -> None:
        takeoff_waypoint = self.mission_waypoints[1]
        if self.quad.z > takeoff_waypoint[2]:
            return
        if self.data.time < self._next_actual_trajectory_sample_time:
            return

        self._actual_trajectory_positions.append(self.quad.position.copy())
        self._next_actual_trajectory_sample_time = (
            self.data.time + ACTUAL_TRAJECTORY_SAMPLE_INTERVAL)
        if len(self._actual_trajectory_positions) < 2:
            return

        self._set_trajectory_segments(
            np.asarray(self._actual_trajectory_positions),
            self._actual_trajectory_segment_ids,
            ACTUAL_TRAJECTORY_COLOR,
        )

    def _record_collisions(self) -> None:
        if self.data.xpos[self._body_id, 2] >= TAKEOFF_HEIGHT:
            self._has_taken_off = True

        takeoff_contact = {self._ground_geom_id, self._body_geom_id}
        for contact in self.data.contact:
            contact_geometries = {contact.geom1, contact.geom2}
            if not self._has_taken_off and contact_geometries == takeoff_contact:
                continue
            self._collision_detected = True
            return

    def _apply_rotor_forces(self) -> None:
        self.data.qfrc_applied[:] = 0.0
        body_rotation = self.data.xmat[self._body_id].reshape(3, 3)
        rotor_forces = self.quad.kf * self.quad.omega ** 2

        for index, force in enumerate(rotor_forces):
            force_world = body_rotation @ np.array([0.0, 0.0, force])
            torque_body = np.array([
                0.0, 0.0, self._rotor_spin_directions[index] * self.quad.kappa * force
            ])
            torque_world = body_rotation @ torque_body
            mujoco.mj_applyFT(
                self.model,
                self.data,
                force_world,
                torque_world,
                self.data.site_xpos[self._rotor_site_ids[index]],
                self._body_id,
                self.data.qfrc_applied,
            )

    def _sync_quad_state(self) -> None:
        self.quad.X = mujoco_to_ned_state(
            self.data.qpos[:3], self.data.qpos[3:7], self.data.qvel[:6])


def _create_quad(model: mujoco.MjModel, body_id: int, rotor_site_ids: np.ndarray) -> Quad:
    rotor_positions = model.site_pos[rotor_site_ids]
    arm_lengths = np.abs(rotor_positions[:, :2])
    if not np.allclose(arm_lengths, arm_lengths[0, 0]):
        raise ValueError("MuJoCo rotor sites must use a symmetric X configuration")

    gravity = np.linalg.norm(model.opt.gravity)
    if gravity == 0:
        raise ValueError("MuJoCo gravity must be non-zero")

    return Quad(
        g=gravity,
        dt=model.opt.timestep,
        mass=model.body_mass[body_id],
        inertia=model.body_inertia[body_id],
        arm_length=arm_lengths[0, 0],
        force_coefficient=_numeric(model, "rotor_force_coefficient", 1)[0],
        drag_to_thrust=_numeric(model, "rotor_drag_to_thrust", 1)[0],
        thrust_limits=_numeric(model, "rotor_thrust_limits", 2),
        motor_time_constants=_numeric(model, "motor_time_constants", 2),
        flight_limits=_numeric(model, "flight_limits", 5),
    )


def _extract_obstacles(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    obstacles = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name is None or not name.startswith("obstacle_"):
            continue
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError(f"MuJoCo planning obstacle '{name}' must be an axis-aligned box")
        if not np.allclose(data.geom_xmat[geom_id].reshape(3, 3), np.eye(3)):
            raise ValueError(f"MuJoCo planning obstacle '{name}' must be axis-aligned")

        center_ned = ENU_TO_NED @ data.geom_xpos[geom_id]
        half_size = model.geom_size[geom_id]
        obstacles.append(np.array([
            center_ned[0] - half_size[0], center_ned[0] + half_size[0],
            center_ned[1] - half_size[1], center_ned[1] + half_size[1],
            center_ned[2] - half_size[2], center_ned[2] + half_size[2],
        ]))
    return np.asarray(obstacles, dtype=float).reshape(-1, 6)


def _extract_mission_waypoints(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        start_position: np.ndarray,
        goal_position: np.ndarray,
) -> np.ndarray:
    waypoint_ids = []
    for site_id in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        if name is not None and name.startswith("waypoint_"):
            waypoint_ids.append((name, site_id))

    waypoint_ids.sort()
    expected_names = [f"waypoint_{index:02d}" for index in range(len(waypoint_ids))]
    if [name for name, _ in waypoint_ids] != expected_names:
        raise ValueError("MuJoCo mission waypoints must be consecutively numbered from waypoint_00")
    if not waypoint_ids:
        raise ValueError("MuJoCo scene must define at least one mandatory waypoint")

    mandatory_waypoints = np.array([
        ENU_TO_NED @ data.site_xpos[site_id] for _, site_id in waypoint_ids
    ])
    return np.vstack((start_position, mandatory_waypoints, goal_position))


def _numeric(model: mujoco.MjModel, name: str, expected_size: int) -> np.ndarray:
    numeric_id = _named_id(model, mujoco.mjtObj.mjOBJ_NUMERIC, name)
    size = model.numeric_size[numeric_id]
    if size != expected_size:
        raise ValueError(f"MuJoCo numeric '{name}' must contain {expected_size} values")
    address = model.numeric_adr[numeric_id]
    return model.numeric_data[address:address + size].copy()


def _named_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo scene is missing required element '{name}'")
    return object_id


def _vector(values: np.ndarray, size: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"MuJoCo {name} must contain {size} finite values")
    return vector