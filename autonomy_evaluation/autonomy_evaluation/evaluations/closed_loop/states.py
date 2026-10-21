# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""States of the ego vehicle and the other road users, which all metrics of a closed-loop evaluation share.

The ego vehicle is read from the ``ego_data`` input (``perception_msgs/EgoData``),
the other road users from the ``objects`` input (``perception_msgs/ObjectList``).
Both are reduced to oriented rectangles on the ground plane:

- Positions are those of the geometric centers: the reference point of a state
  is moved by its ``translation_to_geometric_center``, rotated by the yaw.
- The ego vehicle has the ``length`` and ``width`` of the ``EgoData`` message,
  an object the extent of its state; an object whose motion model has no extent
  is a point.
- Velocities are given along the heading (longitudinal) and across it
  (lateral), as ``perception_msgs`` state models define them, and are taken as
  velocities over ground. An entry that ``perception_msgs`` marks as not set,
  i.e. ``CONTINUOUS_STATE_COVARIANCE_INVALID`` on the diagonal of the state
  covariance, is unavailable. A state without covariance has all its entries set.
- Objects given in another frame than the ego vehicle are transformed into the
  frame of the ego vehicle with tf2.

Everything derived from a sample is derived once and shared among the metrics
through :meth:`~autonomy_evaluation.evaluations.Metric.Sample.memo`.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import rclpy.logging
import tf2_perception_msgs  # noqa: F401, registers the tf2 transforms of perception_msgs/ObjectList and EgoData
from autonomy_evaluation.evaluations.Metric import Sample
from perception_msgs_utils.checks import InvalidStateCovarianceSizeError
from perception_msgs_utils.constants import CONTINUOUS_STATE_COVARIANCE_INVALID
from perception_msgs_utils.convenience_state_getters import get_continuous_state_covariance_at
from perception_msgs_utils.state_getters import (
    get_acc_lat,
    get_acc_lon,
    get_length,
    get_vel_lat,
    get_vel_lon,
    get_width,
    get_x,
    get_y,
    get_yaw,
    get_yaw_rate,
)
from perception_msgs_utils.state_index import (
    index_acc_lat,
    index_acc_lon,
    index_vel_lat,
    index_vel_lon,
    index_yaw_rate,
    UnknownStateEntryError,
)
from tf2_ros import TransformException

_LOGGER = rclpy.logging.get_logger("closed_loop")


@dataclass(frozen=True)
class EgoState:
    """State of the ego vehicle at a sample.

    Attributes:
        frame_id: Frame of the position and the yaw.
        x, y: BEV position of the geometric center in meters.
        yaw: Yaw angle in radians.
        length, width: Extent of the ego vehicle in meters.
        vel_lon, vel_lat: Velocity along and across the heading in m/s,
            ``None`` if not set.
        acc_lon, acc_lat: Acceleration along and across the heading in m/s²,
            ``None`` if not set.
        yaw_rate: Yaw rate in rad/s, ``None`` if not set.
    """

    frame_id: str
    x: float
    y: float
    yaw: float
    length: float
    width: float
    vel_lon: Optional[float]
    vel_lat: Optional[float]
    acc_lon: Optional[float]
    acc_lat: Optional[float]
    yaw_rate: Optional[float]

    @property
    def rectangle(self) -> Tuple[float, float, float, float, float]:
        """Footprint as ``(x, y, yaw, length, width)``."""
        return self.x, self.y, self.yaw, self.length, self.width

    @property
    def velocity(self) -> Optional[Tuple[float, float]]:
        """Velocity in the frame of the position as ``(vx, vy)``, ``None`` if not set."""
        if self.vel_lon is None:
            return None
        return _rotated(self.vel_lon, self.vel_lat or 0.0, self.yaw)

    @property
    def speed(self) -> Optional[float]:
        """Speed in m/s, ``None`` if the velocity is not set."""
        return None if self.vel_lon is None else math.hypot(self.vel_lon, self.vel_lat or 0.0)


@dataclass(frozen=True)
class ObjectStates:
    """States of the road users other than the ego vehicle at a sample, one array entry per road user.

    Attributes:
        ids: Lifetime IDs of the road users.
        x, y: BEV positions of the geometric centers in meters, in the frame of the ego vehicle.
        yaw: Yaw angles in radians.
        length, width: Extents in meters, zero for a road user without extent.
        vx, vy: Velocities over ground in m/s, zero for a road user whose velocity is not set.
        objects: The ``perception_msgs/Object`` messages of the road users, in the frame of the ego vehicle.
        header: Header of the object list, in the frame of the ego vehicle.
    """

    ids: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    length: np.ndarray
    width: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    objects: List[Any]
    header: Any

    @property
    def rectangles(self) -> Tuple[np.ndarray, ...]:
        """Footprints as ``(x, y, yaw, length, width)`` arrays."""
        return self.x, self.y, self.yaw, self.length, self.width

    def select(self, mask: np.ndarray) -> List[Any]:
        """Return the ``perception_msgs/Object`` messages of the road users selected by a boolean mask."""
        return [obj for obj, selected in zip(self.objects, mask) if selected]


def ego_state(sample: Sample) -> EgoState:
    """Read the state of the ego vehicle from the ``ego_data`` of a sample.

    Args:
        sample: Sample holding the ``ego_data`` message.

    Returns:
        The state of the ego vehicle.
    """

    def read() -> EgoState:
        ego_data = sample["ego_data"]
        yaw = get_yaw(ego_data)
        x, y = _geometric_center(ego_data.state, get_x(ego_data), get_y(ego_data), yaw)
        return EgoState(
            frame_id=ego_data.header.frame_id,
            x=x,
            y=y,
            yaw=yaw,
            length=float(ego_data.length),
            width=float(ego_data.width),
            vel_lon=_entry(ego_data, get_vel_lon, index_vel_lon),
            vel_lat=_entry(ego_data, get_vel_lat, index_vel_lat),
            acc_lon=_entry(ego_data, get_acc_lon, index_acc_lon),
            acc_lat=_entry(ego_data, get_acc_lat, index_acc_lat),
            yaw_rate=_entry(ego_data, get_yaw_rate, index_yaw_rate),
        )

    return sample.memo("ego_state", read)


def object_states(sample: Sample) -> ObjectStates:
    """Read the states of the other road users from the ``objects`` of a sample, in the frame of the ego vehicle.

    The road user of ID ``ego_object_id`` (evaluation configuration) is the ego
    vehicle itself, which some simulations list among the objects, and is left
    out; ``-1`` keeps all road users.

    Args:
        sample: Sample holding the ``ego_data`` and ``objects`` messages.

    Returns:
        The states of the road users.

    Raises:
        ValueError: The objects cannot be transformed into the frame of the ego vehicle.
    """

    def read() -> ObjectStates:
        object_list = _in_frame(sample["objects"], ego_state(sample).frame_id, sample.tf_buffer)
        ego_object_id = sample.config.get("ego_object_id", -1)
        objects = [obj for obj in object_list.objects if ego_object_id < 0 or obj.id != ego_object_id]
        fields = {name: np.zeros(len(objects)) for name in ("x", "y", "yaw", "length", "width", "vx", "vy")}
        for index, obj in enumerate(objects):
            yaw = get_yaw(obj)
            fields["x"][index], fields["y"][index] = _geometric_center(obj.state, get_x(obj), get_y(obj), yaw)
            fields["yaw"][index] = yaw
            fields["length"][index] = _extent(obj, get_length)
            fields["width"][index] = _extent(obj, get_width)
            vel_lon = _entry(obj, get_vel_lon, index_vel_lon) or 0.0
            vel_lat = _entry(obj, get_vel_lat, index_vel_lat) or 0.0
            fields["vx"][index], fields["vy"][index] = _rotated(vel_lon, vel_lat, yaw)
        return ObjectStates(
            ids=np.array([obj.id for obj in objects], dtype=np.uint64),
            objects=objects,
            header=object_list.header,
            **fields,
        )

    return sample.memo("object_states", read)


def to_ego_frame(ego: EgoState, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Express BEV positions in the frame of the ego vehicle: origin at its geometric center, x axis along its heading.

    Args:
        ego: State of the ego vehicle.
        x, y: BEV positions in the frame of the ego vehicle's position.

    Returns:
        The positions as ``(x, y)`` in the frame of the ego vehicle.
    """
    dx, dy = np.asarray(x) - ego.x, np.asarray(y) - ego.y
    cos, sin = math.cos(ego.yaw), math.sin(ego.yaw)
    return cos * dx + sin * dy, -sin * dx + cos * dy


def _rotated(along: float, across: float, yaw: float) -> Tuple[float, float]:
    """Rotate a vector given along and across a heading into the frame the heading is given in."""
    return along * math.cos(yaw) - across * math.sin(yaw), along * math.sin(yaw) + across * math.cos(yaw)


def _geometric_center(state: Any, x: float, y: float, yaw: float) -> Tuple[float, float]:
    """Move the BEV position of the reference point of a state to the geometric center."""
    translation = state.reference_point.translation_to_geometric_center
    dx, dy = _rotated(translation.x, translation.y, yaw)
    return x + dx, y + dy


def _entry(obj: Any, getter: Callable[[Any], float], index: Callable[[int], int]) -> Optional[float]:
    """Read a state entry, ``None`` if the motion model has no such entry or ``perception_msgs`` marks it as not set."""
    try:
        value = float(getter(obj))
    except UnknownStateEntryError:
        return None
    try:
        entry = index(obj.state.model_id)
        if get_continuous_state_covariance_at(obj, entry, entry) == CONTINUOUS_STATE_COVARIANCE_INVALID:
            return None
    except InvalidStateCovarianceSizeError:
        pass  # a state without covariance has all its entries set
    return value


def _extent(obj: Any, getter: Callable[[Any], float]) -> float:
    """Read the length or width of an object, zero if its motion model has no extent."""
    try:
        return float(getter(obj))
    except UnknownStateEntryError:
        return 0.0


def _in_frame(object_list: Any, frame_id: str, tf_buffer: Any) -> Any:
    """Transform an object list into a frame, so that its objects can be compared with the ego vehicle.

    An object list in another frame is transformed with the transform
    ``tf_buffer`` holds at its stamp, which is warned about once. An empty
    frame cannot be checked, so it is assumed to be the frame of the ego
    vehicle.

    Raises:
        ValueError: No transform into the frame is available at the stamp of the object list.
    """
    source_frame = object_list.header.frame_id
    if not source_frame or not frame_id or source_frame == frame_id:
        return object_list
    _LOGGER.warning(f"Objects in frame '{source_frame}' are transformed into frame '{frame_id}' of the ego vehicle", once=True)
    error = f"Objects in frame '{source_frame}' cannot be transformed into frame '{frame_id}' of the ego vehicle"
    if tf_buffer is None:
        raise ValueError(f"{error}, as no transforms are available")
    try:
        # the tf2 transform of perception_msgs/ObjectList modifies the message it transforms
        return tf_buffer.transform(copy.deepcopy(object_list), frame_id)
    except TransformException as exception:
        raise ValueError(f"{error}: {exception}") from exception
