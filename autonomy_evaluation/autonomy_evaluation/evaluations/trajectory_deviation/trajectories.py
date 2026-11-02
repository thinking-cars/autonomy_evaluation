# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Trajectories of the ego vehicle that the trajectory deviation evaluation compares, planned or driven.

Both inputs of the evaluation, ``evaluated`` and ``reference``, are
``perception_msgs/EgoData``, of which each provides a trajectory from the
source configured for it:

- ``planned``: the planned trajectory of the message, ``trajectory_planned``,
  each state at the stamp of its header.
- ``driven``: the states of the ego vehicle that the input reports at the
  following samples of the same sequence, each at the stamp of its sample.

A pose is that of the geometric center, i.e. the position of a state moved by
the ``translation_to_geometric_center`` of its reference point, and the
velocity is the longitudinal velocity of the state, ``None`` if not set. All
trajectories are compared in the frame of the ``reference``; an evaluated
message in another frame is transformed with the transform tf2 holds at its
stamp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from autonomy_evaluation.utils.PerceptionMsgsUtils import PerceptionMsgsUtils
from perception_msgs.msg import ObjectState
from perception_msgs_utils.state_getters import get_vel_lon, get_x, get_y, get_yaw
from perception_msgs_utils.state_index import index_vel_lon
from rclpy.time import Time
from tf2_ros import TransformException

#: Source of a trajectory planned by the system under test, the ``trajectory_planned`` of its messages.
PLANNED = "planned"

#: Source of a trajectory actually driven, the states of the following messages of an input.
DRIVEN = "driven"

#: Sources a trajectory can be taken from.
SOURCES: Tuple[str, ...] = (PLANNED, DRIVEN)

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


def frame_transform(tf_buffer: Any, source: str, target: str, stamp: Any) -> Tuple[float, float, float]:
    """Look up the planar transform from one frame into another.

    Args:
        tf_buffer: Transforms between the frames, ``None`` if none is available.
        source, target: Frames; an empty frame is taken as the other one.
        stamp: ``builtin_interfaces/Time`` to look the transform up at.

    Returns:
        ``(dx, dy, dyaw)``: the translation and the rotation about the vertical
        axis that take poses in ``source`` to ``target``, the identity for the
        same frame.

    Raises:
        ValueError: No transform between the frames is available at the stamp.
    """
    if not source or not target or source == target:
        return 0.0, 0.0, 0.0
    error = f"Trajectories in frame '{source}' cannot be transformed into frame '{target}' of the reference"
    if tf_buffer is None:
        raise ValueError(f"{error}, as no transforms are available")
    try:
        transform = tf_buffer.lookup_transform(target, source, Time.from_msg(stamp)).transform
    except TransformException as exception:
        raise ValueError(f"{error}: {exception}") from exception
    rotation = transform.rotation
    dyaw = math.atan2(2.0 * (rotation.w * rotation.z + rotation.x * rotation.y), 1.0 - 2.0 * (rotation.y**2 + rotation.z**2))
    return transform.translation.x, transform.translation.y, dyaw


def transformed_pose(pose: Pose, transform: Tuple[float, float, float]) -> Pose:
    """Apply a planar transform to a pose; the longitudinal velocity is unchanged."""
    x, y, yaw, velocity = pose
    dx, dy, dyaw = transform
    rotated_x, rotated_y = PerceptionMsgsUtils.rotated(x, y, dyaw)
    return rotated_x + dx, rotated_y + dy, yaw + dyaw, velocity


def transformed_trajectory(trajectory: Optional[Trajectory], transform: Tuple[float, float, float]) -> Optional[Trajectory]:
    """Apply a planar transform to a trajectory."""
    if trajectory is None or transform == (0.0, 0.0, 0.0):
        return trajectory
    dx, dy, dyaw = transform
    cos, sin = math.cos(dyaw), math.sin(dyaw)
    return Trajectory(
        t=trajectory.t,
        x=cos * trajectory.x - sin * trajectory.y + dx,
        y=sin * trajectory.x + cos * trajectory.y + dy,
        yaw=trajectory.yaw + dyaw,
        velocity=trajectory.velocity,
    )


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
