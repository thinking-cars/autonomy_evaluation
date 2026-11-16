# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Trajectories of the ego vehicle that the trajectory deviation evaluation compares, planned or driven.

Each input of the evaluation, ``evaluated`` and ``reference``, provides a
trajectory from the source configured for it, which also determines the type of
its messages:

- ``planned``: the planned trajectory of a ``perception_msgs/EgoData``,
  ``trajectory_planned``, each state at the stamp of its header.
- ``driven``: the states of the ego vehicle that ``perception_msgs/EgoData``
  messages report at the following samples of the same sequence, each at the
  stamp of its sample.
- ``trajectory``: a ``trajectory_planning_msgs/Trajectory``, each sample point
  at the stamp of the header plus its time ``T``.

A pose is that of the geometric center, i.e. the position of a state moved by
the ``translation_to_geometric_center`` of its reference point, and the
velocity is the longitudinal velocity of the state, ``None`` if not set. A
``Trajectory`` names no reference point: its sample points are taken to
describe the origin of ``base_link``, e.g. the rear axle, and are moved by
``base_link_to_geometric_center``, the translation from there to the geometric
center, whatever frame the trajectory is given in. Its
heading is that of its sample points, or for a ``REFERENCE`` trajectory, which
has none, that of the direction of travel.

All trajectories are compared in the configured ``fixed_frame``, else in the frame of
the ``reference``; a message in another frame is transformed with the transform
tf2 holds at its stamp, so the frame compared in must be fixed, e.g. ``map``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from autonomy_evaluation.utils.PerceptionMsgsUtils import PerceptionMsgsUtils
from autonomy_evaluation.utils.TfUtils import IDENTITY, PlanarTransform, TfUtils
from perception_msgs.msg import EgoData, ObjectState
from perception_msgs_utils.state_getters import get_vel_lon, get_x, get_y, get_yaw
from perception_msgs_utils.state_index import index_vel_lon
from trajectory_planning_msgs.msg import Trajectory as TrajectoryMessage
from trajectory_planning_msgs_utils.state_index import has_theta, index_t, index_theta, index_v, index_x, index_y
from trajectory_planning_msgs_utils.utils import get_state_dim

#: Source of a trajectory planned by the system under test, the ``trajectory_planned`` of its messages.
PLANNED = "planned"

#: Source of a trajectory actually driven, the states of the following messages of an input.
DRIVEN = "driven"

#: Source of a trajectory planned by the system under test, given as ``trajectory_planning_msgs/Trajectory``.
TRAJECTORY = "trajectory"

#: Sources a trajectory can be taken from.
SOURCES: Tuple[str, ...] = (PLANNED, DRIVEN, TRAJECTORY)

# Distance in meters below which two consecutive sample points are taken to coincide, giving no direction of travel
_MIN_STEP = 1e-6

#: A pose with velocity as ``(x, y, yaw, velocity)``, the velocity ``None`` if not set.
Pose = Tuple[float, float, float, Optional[float]]


@dataclass(frozen=True)
class Trajectory:
    """Poses and velocities of the ego vehicle over time.

    Attributes:
        t: Stamps in seconds, strictly increasing.
        x, y: Positions of the geometric center in meters.
        yaw: Yaw angles in radians, unwrapped so that they can be interpolated.
        velocity: Longitudinal velocities in m/s, ``None`` if not set at every pose.
    """

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    velocity: Optional[np.ndarray]

    @classmethod
    def from_poses(cls, stamps: Sequence[float], poses: Sequence[Pose]) -> Optional["Trajectory"]:
        """Build a trajectory from poses at stamps.

        Returns:
            The trajectory, ``None`` for fewer than two poses or stamps that do
            not strictly increase.
        """
        t = np.asarray(stamps, dtype=float)
        if len(t) < 2 or np.any(np.diff(t) <= 0.0):
            return None
        x, y, yaw, velocity = zip(*poses)
        return cls(
            t=t,
            x=np.asarray(x, dtype=float),
            y=np.asarray(y, dtype=float),
            yaw=np.unwrap(np.asarray(yaw, dtype=float)),
            velocity=None if None in velocity else np.asarray(velocity, dtype=float),
        )

    def interpolated(self, times: np.ndarray) -> "Trajectory":
        """Interpolate the trajectory linearly at stamps within its time span."""
        return Trajectory(
            t=times,
            x=np.interp(times, self.t, self.x),
            y=np.interp(times, self.t, self.y),
            yaw=np.interp(times, self.t, self.yaw),
            velocity=None if self.velocity is None else np.interp(times, self.t, self.velocity),
        )


def message_type(source: str) -> Any:
    """Return the type of the messages a trajectory is taken from: ``Trajectory`` for ``trajectory``, else ``EgoData``."""
    return TrajectoryMessage if source == TRAJECTORY else EgoData


def state_pose(obj: Any) -> Pose:
    """Read the pose and the longitudinal velocity of an ``ObjectState`` or a message holding one, e.g. ``EgoData``."""
    state = obj if isinstance(obj, ObjectState) else obj.state
    yaw = get_yaw(state)
    x, y = PerceptionMsgsUtils.geometric_center(state, get_x(state), get_y(state), yaw)
    return x, y, yaw, PerceptionMsgsUtils.entry(state, get_vel_lon, index_vel_lon)


def planned_trajectory(ego_data: Any) -> Optional[Trajectory]:
    """Read the planned trajectory of an ``EgoData`` message.

    Returns:
        The trajectory, ``None`` for fewer than two planned states, or states
        whose stamps do not strictly increase, e.g. because they are not set.
    """
    states = ego_data.trajectory_planned
    stamps = [_seconds(state.header.stamp) for state in states]
    return Trajectory.from_poses(stamps, [state_pose(state) for state in states])


def planned_frame(ego_data: Any) -> str:
    """Return the frame of the planned trajectory of an ``EgoData`` message: that of its states, else of the message."""
    states = ego_data.trajectory_planned
    return (states[0].header.frame_id if states else "") or ego_data.header.frame_id


def message_trajectory(message: Any, to_geometric_center: Sequence[float]) -> Optional[Trajectory]:
    """Read a ``trajectory_planning_msgs/Trajectory`` as the trajectory of the geometric center, in its frame.

    Args:
        message: The trajectory message of any type, ``REFERENCE``,
            ``DRIVABLE`` or ``DRIVABLERWS``.
        to_geometric_center: Translation ``[x, y]`` in meters from the point
            the sample points describe to the geometric center of the vehicle,
            along and across its heading.

    Returns:
        The trajectory, each sample point at the stamp of the header plus its
        time ``T``; ``None`` for fewer than two sample points or times that do
        not strictly increase.

    Raises:
        ValueError: The type of the trajectory is unknown or its states do not
            form whole sample points.
    """
    dimension = get_state_dim(message.type_id)
    states = np.asarray(message.states, dtype=float)
    if states.size % dimension:
        raise ValueError(f"Trajectory of type {message.type_id} holds {states.size} states, no multiple of {dimension}")
    points = states.reshape(-1, dimension)
    type_id = message.type_id
    x, y, velocity = points[:, index_x(type_id)], points[:, index_y(type_id)], points[:, index_v(type_id)]
    yaw = points[:, index_theta(type_id)] if has_theta(type_id) else _heading_of_travel(x, y, velocity)
    dx, dy = to_geometric_center
    cos, sin = np.cos(yaw), np.sin(yaw)
    stamps = _seconds(message.header.stamp) + points[:, index_t(type_id)]
    poses = zip(x + cos * dx - sin * dy, y + sin * dx + cos * dy, yaw, velocity)
    return Trajectory.from_poses(stamps, [(float(px), float(py), float(pyaw), float(pv)) for px, py, pyaw, pv in poses])


def _heading_of_travel(x: np.ndarray, y: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Derive the heading at sample points from the direction of travel to the next one.

    Reversing, i.e. at a negative velocity, the heading is opposite to the
    direction of travel. Where the vehicle does not move, it keeps the heading
    of the last movement, before the first one that of the first; without
    movement at all, it heads along the x axis of the frame, e.g. forward in
    the frame of the vehicle.
    """
    if len(x) < 2:
        return np.zeros(len(x))
    dx, dy = np.diff(x), np.diff(y)
    moving = np.hypot(dx, dy) > _MIN_STEP
    if not moving.any():
        return np.zeros(len(x))
    headings = np.arctan2(dy, dx) + np.where(velocity[:-1] < 0.0, np.pi, 0.0)
    # index of the last moving segment up to each segment, before the first one that first one
    last_moving = np.maximum.accumulate(np.where(moving, np.arange(len(moving)), -1))
    last_moving[last_moving < 0] = int(np.argmax(moving))
    segment_headings = headings[last_moving]
    return np.append(segment_headings, segment_headings[-1])


def transformed_pose(pose: Pose, transform: PlanarTransform) -> Pose:
    """Apply a planar transform to a pose; the longitudinal velocity is unchanged."""
    x, y, yaw, velocity = pose
    dx, dy, dyaw = transform
    rotated_x, rotated_y = PerceptionMsgsUtils.rotated(x, y, dyaw)
    return rotated_x + dx, rotated_y + dy, yaw + dyaw, velocity


def transformed_trajectory(trajectory: Optional[Trajectory], transform: PlanarTransform) -> Optional[Trajectory]:
    """Apply a planar transform to a trajectory."""
    if trajectory is None or transform == IDENTITY:
        return trajectory
    x, y = TfUtils.transform_points(trajectory.x, trajectory.y, transform)
    return Trajectory(t=trajectory.t, x=x, y=y, yaw=trajectory.yaw + transform[2], velocity=trajectory.velocity)


def driven_trajectory(stamps: List[float], poses: List[Pose], start: int, horizon: float) -> Optional[Trajectory]:
    """Build the trajectory driven from one sample of a sequence over a horizon.

    Args:
        stamps: Stamps of the samples of the sequence.
        poses: Poses of the ego vehicle at the samples.
        start: Index of the sample the trajectory starts at.
        horizon: Seconds the trajectory covers at most, up to the end of the sequence.

    Returns:
        The trajectory through the poses from the starting sample on, including
        the first pose beyond the horizon so that it can be interpolated up to
        the horizon; ``None`` if no later sample follows.
    """
    end = start + 1
    while end < len(stamps) and stamps[end - 1] < stamps[start] + horizon:
        end += 1
    return Trajectory.from_poses(stamps[start:end], poses[start:end])


def _seconds(stamp: Any) -> float:
    """Convert a ``builtin_interfaces/Time`` into seconds."""
    return stamp.sec + stamp.nanosec * 1e-9
