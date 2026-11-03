# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Builders of the messages a closed-loop evaluation reads, as a simulation or a dataset publishes them.

The ego vehicle is a ``perception_msgs/EgoData`` of the EGO state model, the other road users are
``perception_msgs/Object`` entries of the HEXAMOTION state model. State entries that are given are
marked as set in the state covariance, all others stay marked as not set, as ``perception_msgs_utils``
initializes them.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

from geometry_msgs.msg import TransformStamped
from perception_msgs.msg import EGO, EgoData, HEXAMOTION, Object, ObjectList, ObjectState
from perception_msgs_utils import initialize_state, set_continuous_state_covariance_to_unknown_at
from route_planning_msgs.msg import LaneElement, Route, RouteElement
from tf2_ros import Buffer

# A car of 4 m length and 2 m width
LENGTH, WIDTH = 4.0, 2.0


def _set(state, entries) -> None:
    """Set the given ``(index, value)`` entries of a state, marking them as set in its covariance."""
    for index, value in entries:
        if value is not None:
            state.continuous_state[index] = value
            set_continuous_state_covariance_to_unknown_at(state, index, index)


def _stamped(message, stamp: float, frame_id: str):
    """Stamp a message and give it a frame."""
    message.header.stamp.sec, nanoseconds = divmod(round(stamp * 1e9), 1_000_000_000)
    message.header.stamp.nanosec = nanoseconds
    message.header.frame_id = frame_id
    return message


def ego_data(
    x: float = 0.0,
    y: float = 0.0,
    yaw: float = 0.0,
    vel_lon: Optional[float] = None,
    vel_lat: Optional[float] = None,
    acc_lon: Optional[float] = None,
    acc_lat: Optional[float] = None,
    yaw_rate: Optional[float] = None,
    length: float = LENGTH,
    width: float = WIDTH,
    stamp: float = 0.0,
    frame_id: str = "map",
    translation_to_geometric_center: Tuple[float, float] = (0.0, 0.0),
    trajectory_planned: Sequence[ObjectState] = (),
) -> EgoData:
    """Build the ``EgoData`` of an ego vehicle, its position given at its reference point."""
    message = EgoData(length=length, width=width, trajectory_planned=list(trajectory_planned))
    initialize_state(message, EGO.MODEL_ID)
    _set(
        message.state,
        [
            (EGO.X, x),
            (EGO.Y, y),
            (EGO.YAW, yaw),
            (EGO.VEL_LON, vel_lon),
            (EGO.VEL_LAT, vel_lat),
            (EGO.ACC_LON, acc_lon),
            (EGO.ACC_LAT, acc_lat),
            (EGO.YAW_RATE, yaw_rate),
        ],
    )
    message.state.reference_point.translation_to_geometric_center.x = translation_to_geometric_center[0]
    message.state.reference_point.translation_to_geometric_center.y = translation_to_geometric_center[1]
    return _stamped(message, stamp, frame_id)


def planned_state(
    stamp: float, x: float, y: float = 0.0, yaw: float = 0.0, vel_lon: Optional[float] = None, frame_id: str = "map"
) -> ObjectState:
    """Build a state of a planned trajectory of the ego vehicle, stamped with the time it is planned for."""
    state = ObjectState()
    initialize_state(state, EGO.MODEL_ID)
    _set(state, [(EGO.X, x), (EGO.Y, y), (EGO.YAW, yaw), (EGO.VEL_LON, vel_lon)])
    return _stamped(state, stamp, frame_id)


def road_user(
    object_id: int,
    x: float,
    y: float,
    yaw: float = 0.0,
    vel_lon: Optional[float] = 0.0,
    vel_lat: Optional[float] = 0.0,
    length: float = LENGTH,
    width: float = WIDTH,
) -> Object:
    """Build an ``Object`` of a road user, its position given at its geometric center."""
    obj = Object(id=object_id, existence_probability=1.0)
    initialize_state(obj, HEXAMOTION.MODEL_ID)
    _set(
        obj.state,
        [
            (HEXAMOTION.X, x),
            (HEXAMOTION.Y, y),
            (HEXAMOTION.YAW, yaw),
            (HEXAMOTION.VEL_LON, vel_lon),
            (HEXAMOTION.VEL_LAT, vel_lat),
            (HEXAMOTION.LENGTH, length),
            (HEXAMOTION.WIDTH, width),
        ],
    )
    return obj


def object_list(objects: Sequence[Object] = (), stamp: float = 0.0, frame_id: str = "map") -> ObjectList:
    """Wrap road users in an ``ObjectList``."""
    return _stamped(ObjectList(objects=list(objects)), stamp, frame_id)


def route(
    points: Sequence[Tuple[float, float]],
    destination: Optional[Tuple[float, float]] = None,
    stamp: float = 0.0,
    frame_id: str = "map",
    lane_offsets: Sequence[float] = (0.0,),
    suggested_lane: int = 0,
) -> Route:
    """Build a route through the given points, towards a destination, by default the last point.

    Each point is a route element whose lanes lie at the given lateral offsets in y, of which the
    suggested lane is the given one.
    """
    message = Route()
    for x, y in points:
        lanes = []
        for offset in lane_offsets:
            lane = LaneElement()
            lane.reference_pose.position.x, lane.reference_pose.position.y = float(x), float(y + offset)
            lanes.append(lane)
        message.route_elements.append(RouteElement(lane_elements=lanes, suggested_lane_idx=suggested_lane))
    message.destination.x, message.destination.y = (float(value) for value in (destination or points[-1]))
    message.starting_route_element_idx = 0
    message.destination_route_element_idx = len(points) - 1
    return _stamped(message, stamp, frame_id)


def static_tf_buffer(parent: str, child: str, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> Buffer:
    """Build a tf2 buffer holding the static transform of a child frame in a parent frame."""
    transform = TransformStamped()
    transform.header.frame_id = parent
    transform.child_frame_id = child
    transform.transform.translation.x = x
    transform.transform.translation.y = y
    transform.transform.rotation.z = math.sin(yaw / 2.0)
    transform.transform.rotation.w = math.cos(yaw / 2.0)
    tf_buffer = Buffer()
    tf_buffer.set_transform_static(transform, "test")
    return tf_buffer
