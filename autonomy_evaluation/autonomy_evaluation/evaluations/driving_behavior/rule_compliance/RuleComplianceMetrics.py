# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Metrics of the rule compliance evaluation: whether the ego vehicle drives as the traffic rules allow.

All metrics read the evaluated ``ego_data`` of the ego vehicle and check it
against the lanes of the Lanelet2 map of the current scene, the optional static
``map`` input, under the traffic rules of vehicles in Germany, the default of
Lanelet2:

- ``drivable_area``: whether the footprint of the ego vehicle stays within the
  lanelets vehicles may drive in, i.e. not on sidewalks or off the road.
- ``driving_direction``: whether the ego vehicle drives in the direction of
  travel of its lane, e.g. not against a one-way lane.
- ``speed_limit``: whether the ego vehicle keeps the speed limit of its lane.

Each metric reports the share of the samples at which the ego vehicle violates
its rule, of the samples at which the rule could be checked, and how long and
how often it does so. The rules can only be checked with the map, and only
while the ego vehicle drives in its lanelets; the share of the samples at which
a rule could be checked is reported as its ``coverage``. Like the other metrics
over time, the time of a violation counts each violating sample for the
sampling period of its sequence, and a violation over consecutive samples of a
sequence is one violation. Nothing is judged as passed or failed; limits on the
reported values can be configured for the evaluation.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

from autonomy_evaluation.evaluations.driving_behavior.states import ego_state, lanelet_map, map_input
from autonomy_evaluation.evaluations.Metric import Aggregation, EVALUATED, Input, Metric, Sample
from autonomy_evaluation.evaluations.sampling import sampling_period
from autonomy_evaluation.utils.PlanarUtils import PlanarUtils
from perception_msgs.msg import EgoData


class _RuleComplianceMetric(Metric):
    """Metric of whether the ego vehicle complies with a traffic rule, checked against the lanes of the map."""

    def inputs(self) -> Dict[str, Input]:
        """Read the ego vehicle and, optionally, the map of the current scene."""
        return {"ego_data": Input(EgoData, EVALUATED), "map": map_input()}

    def _violations(self, aggregation: Aggregation, violated: Callable[[Any], bool]) -> Dict[str, Any]:
        """Report how often and how long the rule is violated over the samples.

        Args:
            aggregation: Samples to aggregate, whose records are ``None`` where
                the rule could not be checked.
            violated: Decides whether the record of a checked sample violates the rule.

        Returns:
            ``{"_value_", "time", "num_violations", "coverage"}``: the share of
            the checked samples that violate the rule, ``None`` without any
            checked sample; the time of the violations in seconds; the number of
            violations, consecutive violating samples counting once; and the
            share of the samples at which the rule could be checked, ``None``
            without samples.
        """
        num_samples, num_checked, num_violating, num_violations, time = 0, 0, 0, 0, 0.0
        for sequence in aggregation.sequences():
            period = sampling_period(sequence) or 0.0
            previous = False
            for record in aggregation.records(self.name, sequence):
                num_samples += 1
                current = record is not None and violated(record)
                num_checked += record is not None
                if current:
                    num_violating += 1
                    time += period
                    num_violations += not previous
                previous = current
        return {
            "_value_": num_violating / num_checked if num_checked else None,
            "time": time,
            "num_violations": num_violations,
            "coverage": num_checked / num_samples if num_samples else None,
        }


class DrivableArea(_RuleComplianceMetric):
    """Whether the footprint of the ego vehicle stays within the lanelets vehicles may drive in.

    At each sample, the distance of every corner of the footprint to the
    lanelets the traffic rules allow vehicles to drive in is measured, zero for
    a corner within one of them. The ego vehicle leaves the drivable area if a
    corner lies further outside than the tolerance, e.g. on a sidewalk or off
    the road. Lanelet2 areas, e.g. parking lots, are no drivable area.

    Configuration:
        ``tolerance``: distance in meters a corner may lie outside the drivable
        area without violating it, for the inaccuracy of maps; nuPlan tolerates
        0.3 m.
    """

    NAME = "drivable_area"

    def default_config(self) -> Dict[str, Any]:
        """Tolerate corners up to 0.3 m outside the drivable area, like nuPlan."""
        return {"tolerance": 0.3}

    def compute_sample(self, sample: Sample) -> Optional[float]:
        """Measure how far the footprint of the ego vehicle reaches out of the drivable area at a sample.

        Returns:
            The largest distance of a corner of the footprint to the drivable
            area in meters, zero within it; ``None`` without map or drivable
            lanelets.
        """
        current_map = lanelet_map(sample)
        if current_map is None:
            return None
        distances = [
            current_map.distance_to_drivable_area(float(x), float(y))
            for x, y in PlanarUtils.rectangle_corners(*ego_state(sample).rectangle)
        ]
        if None in distances:
            return None
        return max(distances)

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Report how often and how far the ego vehicle leaves the drivable area.

        Returns:
            The violations of the drivable area (see the module docstring) and
            ``max_distance``, the largest distance of a corner of the footprint
            to the drivable area in meters, ``None`` without any checked sample.
        """
        tolerance = self.config["tolerance"]
        result = self._violations(aggregation, lambda distance: distance > tolerance)
        distances = [distance for distance in aggregation.records(self.name) if distance is not None]
        result["max_distance"] = max(distances, default=None)
        return result


class DrivingDirection(_RuleComplianceMetric):
    """Whether the ego vehicle drives in the direction of travel of its lane.

    At each sample at which the ego vehicle moves, it drives against the
    direction of travel if its direction of motion, i.e. its heading or, when
    reversing, the opposite of it, deviates by 90° or more from every direction
    the traffic rules allow driving the lanelets containing its center in, e.g.
    against a one-way lane. A lanelet vehicles may drive in both directions is
    never driven against. The distance driven against the direction of travel
    follows the positions of the ego vehicle from each such sample to the next
    sample of its sequence.

    Configuration:
        ``min_speed``: speed in m/s below which the ego vehicle stands still,
        and therefore drives in no direction.
    """

    NAME = "driving_direction"

    def default_config(self) -> Dict[str, Any]:
        """Take the ego vehicle as standing still below 0.1 m/s, like collisions do."""
        return {"min_speed": 0.1}

    def compute_sample(self, sample: Sample) -> Optional[Tuple[float, float, bool]]:
        """Decide whether the ego vehicle drives against the direction of travel at a sample.

        Returns:
            ``(x, y, against)``: the position of the ego vehicle and whether it
            drives against the direction of travel, never while standing still;
            ``None`` without map or outside the drivable lanelets. An ego vehicle
            without velocity is taken as driving forward.
        """
        current_map = lanelet_map(sample)
        if current_map is None:
            return None
        ego = ego_state(sample)
        reversing = ego.vel_lon is not None and ego.vel_lon < 0.0
        against = current_map.against_direction(ego.x, ego.y, ego.yaw + (math.pi if reversing else 0.0))
        if against is None:
            return None
        moving = ego.speed is None or ego.speed >= self.config["min_speed"]
        return ego.x, ego.y, against and moving

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Report how often, how long and how far the ego vehicle drives against the direction of travel.

        Returns:
            The violations of the direction of travel (see the module docstring)
            and ``distance``, the distance driven against it in meters.
        """
        distance = 0.0
        for sequence in aggregation.sequences():
            records = aggregation.records(self.name, sequence)
            for record, following in zip(records[:-1], records[1:]):
                if record is not None and following is not None and record[2]:
                    distance += math.hypot(following[0] - record[0], following[1] - record[1])
        result = self._violations(aggregation, lambda record: record[2])
        result["distance"] = distance
        return result


class SpeedLimit(_RuleComplianceMetric):
    """Whether the ego vehicle keeps the speed limit of its lane.

    At each sample, the speed limit is that of the lanelet the ego vehicle
    drives along (see :meth:`~autonomy_evaluation.utils.Lanelet2Map.Lanelet2Map.lane_of`),
    as the traffic rules of vehicles in Germany assign it: from a speed limit
    regulatory element or the ``speed_limit`` attribute of the lanelet, else by
    its ``location``, e.g. 50 km/h in urban areas. The ego vehicle exceeds the
    limit if its speed over ground exceeds it by more than the tolerance.

    Configuration:
        ``tolerance``: speed in m/s by which the speed limit may be exceeded
        without violating it.
    """

    NAME = "speed_limit"

    def default_config(self) -> Dict[str, Any]:
        """Exceed no speed limit, without tolerance."""
        return {"tolerance": 0.0}

    def compute_sample(self, sample: Sample) -> Optional[Tuple[float, float]]:
        """Compare the speed of the ego vehicle with the speed limit of its lane at a sample.

        Returns:
            ``(speed, limit)`` in m/s; ``None`` without map, velocity of the ego
            vehicle or a lanelet it drives along.
        """
        current_map = lanelet_map(sample)
        ego = ego_state(sample)
        if current_map is None or ego.speed is None:
            return None
        lane = current_map.lane_of(ego.x, ego.y, ego.yaw)
        if lane is None:
            return None
        return ego.speed, current_map.speed_limit(lane)

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Report how often, how long and by how much the ego vehicle exceeds the speed limit.

        Returns:
            The violations of the speed limit (see the module docstring) and
            ``max_excess``, the largest speed above the speed limit in m/s, zero
            if it is never exceeded, ``None`` without any checked sample.
        """
        tolerance = self.config["tolerance"]
        result = self._violations(aggregation, lambda record: record[0] > record[1] + tolerance)
        excesses = [max(speed - limit, 0.0) for speed, limit in filter(None, aggregation.records(self.name))]
        result["max_excess"] = max(excesses, default=None)
        return result
