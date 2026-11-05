# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Progress metrics of the driving behavior evaluation: whether the ego vehicle gets ahead.

- ``distance``: distance the ego vehicle drives, its mean speed and how long it
  stands still, e.g. to tell an ego vehicle that drives safely from one that
  never starts.
- ``route``: how far the ego vehicle gets along the route planned for it,
  towards its destination.

Like the other metrics over time, the distance follows the positions of the
ego vehicle over each contiguous sequence of samples, and each sample counts
for the sampling period of its sequence.

The route is read from the optional ``route`` input
(``route_planning_msgs/Route``), of which every sample is passed the latest
message, as a route is planned independently of the samples: the path through
the reference poses of the suggested lanes of its route elements, each at its
arc length ``s`` from the start of the route, and its ``destination``, which
lies within the published route if it has a ``destination_route_element_idx``.
Without route message, the ``route_planned`` points of the ``ego_data`` are the
route and the last of them its destination. A route in
another frame than the ego vehicle is transformed with the transform tf2 holds
at the stamp of the route.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from autonomy_evaluation.evaluations.driving_behavior.ComfortMetrics import ego_inputs
from autonomy_evaluation.evaluations.driving_behavior.states import ego_state
from autonomy_evaluation.evaluations.Metric import Aggregation, CONTEXT, Input, Metric, Sample
from autonomy_evaluation.evaluations.sampling import sampling_period
from autonomy_evaluation.utils.PlanarUtils import PlanarUtils
from autonomy_evaluation.utils.TfUtils import TfUtils
from route_planning_msgs.msg import Route


class DistanceDriven(Metric):
    """Distance the ego vehicle drives, its mean speed, and how long it stands still.

    The distance is the length of the path through the positions of the ego
    vehicle over each contiguous sequence of samples, the mean speed that
    distance over the duration of the sequences.

    Configuration:
        ``standstill_speed``: speed in m/s below which the ego vehicle stands still.
    """

    NAME = "distance"

    def inputs(self) -> Dict[str, Input]:
        """Read the ego vehicle."""
        return ego_inputs()

    def default_config(self) -> Dict[str, Any]:
        """Take the ego vehicle as standing still below 0.1 m/s, like collisions do."""
        return {"standstill_speed": 0.1}

    def compute_sample(self, sample: Sample) -> Tuple[Optional[float], float, float, Optional[float]]:
        """Record the position and the speed of the ego vehicle at a sample.

        Returns:
            ``(stamp, x, y, speed)``, the speed ``None`` without velocity.
        """
        ego = ego_state(sample)
        return sample.stamp, ego.x, ego.y, ego.speed

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Report the distance driven, the mean speed and the time standing still.

        Returns:
            ``{"_value_", "mean_speed", "standstill_time", "standstill_share"}``:
            the distance driven in meters; the mean speed in m/s, ``None`` if no
            time passes; the time the ego vehicle stands still in seconds; and
            the share of the samples with velocity at which it stands still,
            ``None`` without such samples.
        """
        standstill_speed = self.config["standstill_speed"]
        distance, duration, standstill_time = 0.0, 0.0, 0.0
        num_with_speed, num_standing = 0, 0
        for sequence in aggregation.sequences():
            records = aggregation.records(self.name, sequence)
            period = sampling_period(sequence) or 0.0
            for (_, x, y, _), (_, next_x, next_y, _) in zip(records, records[1:]):
                distance += math.hypot(next_x - x, next_y - y)
            if records[0][0] is not None and records[-1][0] is not None:
                duration += records[-1][0] - records[0][0]
            for _, _, _, speed in records:
                if speed is not None:
                    num_with_speed += 1
                    if speed < standstill_speed:
                        num_standing += 1
                        standstill_time += period
        return {
            "_value_": distance,
            "mean_speed": distance / duration if duration > 0.0 else None,
            "standstill_time": standstill_time,
            "standstill_share": num_standing / num_with_speed if num_with_speed else None,
        }


class RouteProgress(Metric):
    """How far the ego vehicle gets along the route planned for it, towards its destination.

    A route planner may publish only a window of its route, e.g. trimmed behind
    the ego vehicle and ending ahead of it before the destination, as long as
    the arc length ``s`` of each route element counts from the start of the
    whole route. At each sample, the ego vehicle is projected onto the path of
    the route and placed at the arc length interpolated between the route
    elements; the destination is placed likewise while it lies within the
    window. The samples of a scene with the same destination form one route, a
    new destination farther than ``goal_tolerance`` from the previous one a new
    route. The progress along a route is the arc length the ego vehicle gains
    from its first to its last sample; its completion is that progress over the
    distance between the ego vehicle at the first sample and the destination,
    and the destination is reached once the ego vehicle comes within
    ``goal_tolerance`` of it along the route. Both need the destination to lie
    within the window at some sample of the route.

    Configuration:
        ``goal_tolerance``: distance in meters along the route within which the
        destination counts as reached, and by which the destination may move
        while it is still the same.
    """

    NAME = "route"

    def __init__(self) -> None:
        """Start without route path, which is derived once per route message."""
        super().__init__()
        self._route: Optional[Tuple[Any, Optional[RoutePath]]] = None

    def inputs(self) -> Dict[str, Input]:
        """Read the ego vehicle and, optionally, the latest route planned for it."""
        return {**ego_inputs(), "route": Input(Route, CONTEXT, optional=True, latest=True)}

    def default_config(self) -> Dict[str, Any]:
        """Take the destination as reached within 3 m along the route."""
        return {"goal_tolerance": 3.0}

    def compute_sample(self, sample: Sample) -> Optional[Tuple[float, Optional[float], float, float, float]]:
        """Place the ego vehicle and the destination along the route at a sample.

        Returns:
            ``(s_ego, s_destination, lateral, destination_x, destination_y)``:
            the arc length of the ego vehicle and of the destination along the
            route in meters, the latter ``None`` while the destination lies
            beyond the published route; the distance of the ego vehicle from the
            path of the route; and the destination; ``None`` without route.

        Raises:
            ValueError: The route cannot be transformed into the frame of the ego vehicle.
        """
        path = self._path(sample)
        if path is None:
            return None
        ego = ego_state(sample)
        s_ego, lateral = path.locate(ego.x, ego.y)
        return s_ego, path.s_destination, abs(lateral), path.destination[0], path.destination[1]

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Report how far the ego vehicle gets along its routes.

        Returns:
            ``{"_value_", "distance", "remaining_distance", "goals_reached",
            "num_routes", "max_lateral_distance", "coverage"}``: the completion
            of the routes whose destination is known, averaged over them,
            ``None`` without such a route; the progress along all routes and the
            distance remaining to the known destinations at the last samples of
            their routes in meters, summed over the routes; the number of
            routes whose destination the ego vehicle reached, and of all routes;
            the largest distance of the ego vehicle from the path of its route;
            and the share of the samples with route, ``None`` without samples.
        """
        tolerance = self.config["goal_tolerance"]
        records = aggregation.records(self.name)
        completions, distance, remaining, goals_reached = [], 0.0, 0.0, 0
        routes = self._routes(aggregation, tolerance)
        for route in routes:
            first, last = route[0][0], route[-1][0]
            distance += last - first
            destinations = [s_destination for _, s_destination, _ in route if s_destination is not None]
            if not destinations:
                continue
            s_destination = destinations[-1]
            to_go = max(s_destination - first, 0.0)
            left = max(s_destination - last, 0.0)
            completions.append(1.0 if to_go <= tolerance else min(max((to_go - left) / to_go, 0.0), 1.0))
            remaining += left
            goals_reached += min(s_destination - s_ego for s_ego, _, _ in route) <= tolerance
        laterals = [lateral for route in routes for _, _, lateral in route]
        return {
            "_value_": float(np.mean(completions)) if completions else None,
            "distance": distance,
            "remaining_distance": remaining,
            "goals_reached": goals_reached,
            "num_routes": len(routes),
            "max_lateral_distance": max(laterals, default=None),
            "coverage": sum(record is not None for record in records) / len(records) if records else None,
        }

    def _routes(self, aggregation: Aggregation, tolerance: float) -> List[List[Tuple[float, Optional[float], float]]]:
        """Split the records of the samples into routes: per scene, by their destination.

        Returns:
            The ``(s_ego, s_destination, lateral)`` of the samples of each route, in the order of the samples.
        """
        routes: List[List[Tuple[float, Optional[float], float]]] = []
        current: Dict[Any, Tuple[Tuple[float, float], List[Tuple[float, Optional[float], float]]]] = {}
        for entry, record in zip(aggregation.entries, aggregation.records(self.name)):
            if record is None:
                continue
            s_ego, s_destination, lateral, x, y = record
            scene = entry.get("scene_id")
            if scene not in current or math.hypot(x - current[scene][0][0], y - current[scene][0][1]) > tolerance:
                current[scene] = ((x, y), [])
                routes.append(current[scene][1])
            current[scene][1].append((s_ego, s_destination, lateral))
        return routes

    def _path(self, sample: Sample) -> Optional["RoutePath"]:
        """Return the route of a sample in the frame of the ego vehicle, derived once per route message."""
        route = sample.messages.get("route")
        if route is not None and self._route is not None and self._route[0] is route:
            return self._route[1]
        ego_data = sample["ego_data"]
        if route is not None and route.route_elements:
            points, arc_lengths, destination_index = _route_points(route)
            destination, header = (route.destination.x, route.destination.y), route.header
        elif ego_data.route_planned:
            points = [(point.x, point.y) for point in ego_data.route_planned]
            arc_lengths, destination_index = None, len(points) - 1
            destination, header = points[-1], ego_data.header
        else:
            return None
        path = None
        if len(points) >= 2:
            transform = TfUtils.planar_transform(
                sample.tf_buffer, header.frame_id, ego_state(sample).frame_id, header.stamp, "The route"
            )
            x, y = TfUtils.transform_points(*np.asarray(points, dtype=float).T, transform)
            destination_x, destination_y = TfUtils.transform_points(
                np.array([destination[0]]), np.array([destination[1]]), transform
            )
            path = RoutePath(
                np.column_stack([x, y]), arc_lengths, (float(destination_x[0]), float(destination_y[0])), destination_index
            )
        if route is not None:
            self._route = (route, path)
        return path


class RoutePath:
    """The path of a route, with the arc length of its points along the whole route, and its destination.

    Attributes:
        points: Points of the path, an array of shape ``(n, 2)``.
        arc_lengths: Arc length of each point along the whole route in meters.
        destination: Destination ``(x, y)``.
        s_destination: Arc length of the destination along the whole route,
            ``None`` if the destination lies beyond the path.
    """

    def __init__(
        self,
        points: np.ndarray,
        arc_lengths: Optional[np.ndarray],
        destination: Tuple[float, float],
        destination_index: Optional[int],
    ) -> None:
        """Hold a path, placing its points by their distances along it unless their arc lengths are given."""
        self.points = points
        self._lengths = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
        self.arc_lengths = self._lengths if arc_lengths is None else np.asarray(arc_lengths, dtype=float)
        self.destination = destination
        self.s_destination = None if destination_index is None else self.locate(*destination)[0]

    def locate(self, x: float, y: float) -> Tuple[float, float]:
        """Place a point along the route.

        Returns:
            ``(s, lateral)``: the arc length along the whole route of the
            closest point of the path, interpolated between the arc lengths of
            the points of the path, and the lateral offset from the path.
        """
        along, lateral, _ = PlanarUtils.polyline_coordinates(self.points, x, y)
        return float(np.interp(along, self._lengths, self.arc_lengths)), lateral


def _route_points(route: Route) -> Tuple[List[Tuple[float, float]], np.ndarray, Optional[int]]:
    """Collect the reference positions and arc lengths of the suggested lanes of a published route.

    Returns:
        ``(points, arc_lengths, destination_index)``: the reference position
        and the arc length of each route element with a lane, consecutive
        duplicates left out, and the index of the point of the route element
        behind the destination, ``None`` if the route ends before the destination.
    """
    elements = route.route_elements
    points: List[Tuple[float, float]] = []
    arc_lengths: List[float] = []
    destination_index = None
    for index, element in enumerate(elements):
        if not element.lane_elements:
            continue
        lane = element.lane_elements[min(element.suggested_lane_idx, len(element.lane_elements) - 1)]
        point = (lane.reference_pose.position.x, lane.reference_pose.position.y)
        if points and point == points[-1]:
            continue
        points.append(point)
        arc_lengths.append(element.s)
        if index == route.destination_route_element_idx:
            destination_index = len(points) - 1
    return points, np.asarray(arc_lengths, dtype=float), destination_index
