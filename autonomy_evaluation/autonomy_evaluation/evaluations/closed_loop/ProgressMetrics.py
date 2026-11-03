# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Progress metrics of a closed-loop evaluation: whether the ego vehicle gets ahead.

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
the reference poses of the suggested lanes of its route elements, from the
starting route element to the one behind the destination, and its
``destination``. Without route message, the ``route_planned`` points of the
``ego_data`` are the route and the last of them its destination. A route in
another frame than the ego vehicle is transformed with the transform tf2 holds
at the stamp of the route.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from autonomy_evaluation.evaluations.closed_loop.ComfortMetrics import ego_inputs
from autonomy_evaluation.evaluations.closed_loop.states import ego_state
from autonomy_evaluation.evaluations.Metric import Aggregation, CONTEXT, Input, Metric, Sample
from autonomy_evaluation.evaluations.sampling import sampling_period
from autonomy_evaluation.utils.ClosedLoopUtils import ClosedLoopUtils
from autonomy_evaluation.utils.TfUtils import TfUtils
from route_planning_msgs.msg import Route

#: A route as the points of its path, an array of shape ``(n, 2)``, and its destination ``(x, y)``.
RoutePath = Tuple[np.ndarray, Tuple[float, float]]


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

    At each sample, the position of the ego vehicle and the destination are
    projected onto the path of the route; the distance between both
    projections along the path is the distance remaining to the destination.
    The samples of a scene with the same destination form one route, a new
    destination farther than ``goal_tolerance`` from the previous one a new
    route. The completion of a route is the share of the distance remaining at
    its first sample that the ego vehicle has driven by its last sample, and it
    reaches its destination once the remaining distance falls to
    ``goal_tolerance``.

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

    def compute_sample(self, sample: Sample) -> Optional[Tuple[float, float, float, float]]:
        """Measure the distance the ego vehicle has left to its destination at a sample.

        Returns:
            ``(remaining, lateral, destination_x, destination_y)``: the distance
            remaining along the route in meters, the distance of the ego vehicle
            from the path of the route, and the destination; ``None`` without route.

        Raises:
            ValueError: The route cannot be transformed into the frame of the ego vehicle.
        """
        path = self._path(sample)
        if path is None:
            return None
        points, destination = path
        ego = ego_state(sample)
        s_ego, lateral, _ = ClosedLoopUtils.polyline_coordinates(points, ego.x, ego.y)
        s_destination, _, _ = ClosedLoopUtils.polyline_coordinates(points, *destination)
        return max(s_destination - s_ego, 0.0), abs(lateral), destination[0], destination[1]

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Report how far the ego vehicle gets along its routes.

        Returns:
            ``{"_value_", "distance", "remaining_distance", "goals_reached",
            "num_routes", "max_lateral_distance", "coverage"}``: the completion
            of the routes, averaged over the routes, ``None`` without route; the
            distance progressed along the routes and the distance remaining at
            their last samples in meters, summed over the routes; the number of
            routes whose destination the ego vehicle reached, and of all routes;
            the largest distance of the ego vehicle from the path of its route;
            and the share of the samples with route, ``None`` without samples.
        """
        tolerance = self.config["goal_tolerance"]
        routes = self._routes(aggregation, tolerance)
        records = aggregation.records(self.name)
        completions, distance, remaining, goals_reached = [], 0.0, 0.0, 0
        for route in routes:
            first, last = route[0][0], route[-1][0]
            completions.append(1.0 if first <= tolerance else min(max((first - last) / first, 0.0), 1.0))
            distance += first - last
            remaining += last
            goals_reached += min(left for left, _ in route) <= tolerance
        laterals = [lateral for route in routes for _, lateral in route]
        return {
            "_value_": float(np.mean(completions)) if completions else None,
            "distance": distance,
            "remaining_distance": remaining,
            "goals_reached": goals_reached,
            "num_routes": len(routes),
            "max_lateral_distance": max(laterals, default=None),
            "coverage": sum(record is not None for record in records) / len(records) if records else None,
        }

    def _routes(self, aggregation: Aggregation, tolerance: float) -> List[List[Tuple[float, float]]]:
        """Split the records of the samples into routes: per scene, by their destination.

        Returns:
            The ``(remaining, lateral)`` of the samples of each route, in the order of the samples.
        """
        routes: List[List[Tuple[float, float]]] = []
        current: Dict[Any, Tuple[Tuple[float, float], List[Tuple[float, float]]]] = {}
        for entry, record in zip(aggregation.entries, aggregation.records(self.name)):
            if record is None:
                continue
            remaining, lateral, x, y = record
            scene = entry.get("scene_id")
            if scene not in current or math.hypot(x - current[scene][0][0], y - current[scene][0][1]) > tolerance:
                current[scene] = ((x, y), [])
                routes.append(current[scene][1])
            current[scene][1].append((remaining, lateral))
        return routes

    def _path(self, sample: Sample) -> Optional[RoutePath]:
        """Return the route of a sample in the frame of the ego vehicle, derived once per route message."""
        route = sample.messages.get("route")
        if route is not None and self._route is not None and self._route[0] is route:
            return self._route[1]
        ego_data = sample["ego_data"]
        if route is not None and route.route_elements:
            points, destination = _route_points(route), (route.destination.x, route.destination.y)
            header = route.header
        elif ego_data.route_planned:
            points = [(point.x, point.y) for point in ego_data.route_planned]
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
            path = np.column_stack([x, y]), (float(destination_x[0]), float(destination_y[0]))
        if route is not None:
            self._route = (route, path)
        return path


def _route_points(route: Route) -> List[Tuple[float, float]]:
    """Collect the reference positions of the suggested lanes of a route, from its start to behind its destination."""
    elements = route.route_elements
    start = route.starting_route_element_idx if route.starting_route_element_idx < len(elements) else 0
    end = route.destination_route_element_idx
    if not start < end < len(elements):
        end = len(elements) - 1
    points = []
    for element in elements[start : end + 1]:
        if element.lane_elements:
            lane = element.lane_elements[min(element.suggested_lane_idx, len(element.lane_elements) - 1)]
            points.append((lane.reference_pose.position.x, lane.reference_pose.position.y))
    return points
