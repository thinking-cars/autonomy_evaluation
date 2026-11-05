# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for the rule compliance evaluation of the driving behavior: drivable area, driving direction and speed limit.

An ego vehicle of 4 m length and 2 m width drives on a straight road of two one-way lanes along the
x axis with the urban speed limit of 50 km/h, or on a two-way road limited to 30 km/h with a sidewalk
along its right edge at y < -1.75 m, sampled at 10 Hz.
"""

from __future__ import annotations

import math

import pytest
from autonomy_evaluation.evaluations.driving_behavior.rule_compliance.RuleCompliance import RuleCompliance
from tests.evaluations.driving_behavior.messages import ego_data
from tests.utils.lanelet2_maps import road_with_sidewalk, two_lane_road

_PERIOD = 0.1


def _evaluate(egos, lanelet2_map, metrics=None, config=None) -> dict:
    """Record one sample per set of ``ego_data`` keyword arguments, 0.1 s apart, with the map, and aggregate them."""
    evaluation = RuleCompliance()
    evaluation.configure({**({"metrics": metrics} if metrics else {}), **(config or {})})
    for index, ego in enumerate(egos):
        stamp = index * _PERIOD
        evaluation.record_sample(sample_id=str(index), stamp=stamp, ego_data=ego_data(stamp=stamp, **ego), map=lanelet2_map)
    return evaluation.finalize()["metrics"]


class TestDrivableArea:
    """Tests detecting the ego vehicle leaving the lanelets vehicles may drive in."""

    def test_driving_onto_the_sidewalk(self):
        """Reaching 0.75 m onto the sidewalk for three samples is one violation of 0.3 s."""
        road, _ = road_with_sidewalk()
        egos = [{"x": 10.0 + index, "y": -1.5 if 5 <= index < 8 else 0.0} for index in range(10)]

        drivable_area = _evaluate(egos, road)["drivable_area"]

        assert drivable_area["_value_"] == pytest.approx(0.3)
        assert drivable_area["time"] == pytest.approx(0.3)
        assert drivable_area["num_violations"] == 1
        assert drivable_area["max_distance"] == pytest.approx(0.75)
        assert drivable_area["coverage"] == 1.0

    def test_small_overlaps_are_tolerated(self):
        """A corner 0.15 m outside the drivable area is within the tolerance of inaccurate maps."""
        road, _ = road_with_sidewalk()

        drivable_area = _evaluate([{"x": 10.0, "y": -0.9}], road)["drivable_area"]

        assert drivable_area["_value_"] == 0.0
        assert drivable_area["max_distance"] == pytest.approx(0.15)

    def test_leaving_the_road(self):
        """Beside the road, the distance is measured to the closest lane."""
        road, _ = two_lane_road()

        drivable_area = _evaluate([{"x": 50.0, "y": -5.0}], road)["drivable_area"]

        assert drivable_area["_value_"] == 1.0
        assert drivable_area["max_distance"] == pytest.approx(5.0 + 1.0 - 1.75)

    def test_cannot_be_checked_without_map(self):
        """Without map, nothing is checked."""
        drivable_area = _evaluate([{"x": 10.0, "y": -1.5}], None)["drivable_area"]

        assert drivable_area == {"_value_": None, "time": 0.0, "num_violations": 0, "coverage": 0.0, "max_distance": None}


class TestDrivingDirection:
    """Tests detecting the ego vehicle driving against the direction of travel."""

    def test_driving_against_a_one_way_lane(self):
        """Driving backwards along the right lane at 10 m/s for 0.5 s covers 4 m between the five samples."""
        road, _ = two_lane_road()
        egos = [{"x": 50.0 - index, "yaw": math.pi, "vel_lon": 10.0} for index in range(5)]

        driving_direction = _evaluate(egos, road)["driving_direction"]

        assert driving_direction["_value_"] == 1.0
        assert driving_direction["time"] == pytest.approx(0.5)
        assert driving_direction["num_violations"] == 1
        assert driving_direction["distance"] == pytest.approx(4.0)

    def test_reversing_against_a_one_way_lane(self):
        """Reversing moves against the heading, so reversing along a one-way lane goes against its direction."""
        road, _ = two_lane_road()
        egos = [{"x": 50.0 - 0.2 * index, "vel_lon": -2.0} for index in range(3)]

        driving_direction = _evaluate(egos, road)["driving_direction"]

        assert driving_direction["_value_"] == 1.0
        assert driving_direction["distance"] == pytest.approx(0.4)

    def test_standing_against_the_direction_is_no_violation(self):
        """A vehicle standing still drives in no direction."""
        road, _ = two_lane_road()

        driving_direction = _evaluate([{"x": 50.0, "yaw": math.pi, "vel_lon": 0.0}], road)["driving_direction"]

        assert driving_direction["_value_"] == 0.0

    def test_two_way_road_may_be_driven_in_both_directions(self):
        """A lanelet that may be driven in both directions is never driven against."""
        road, _ = road_with_sidewalk()
        egos = [{"x": 50.0 - index, "yaw": math.pi, "vel_lon": 10.0} for index in range(3)]

        assert _evaluate(egos, road)["driving_direction"]["_value_"] == 0.0

    def test_cannot_be_checked_off_the_road(self):
        """Off the road, there is no direction of travel."""
        road, _ = two_lane_road()

        driving_direction = _evaluate([{"x": 50.0, "y": -10.0, "vel_lon": 10.0}], road)["driving_direction"]

        assert driving_direction["_value_"] is None
        assert driving_direction["coverage"] == 0.0


class TestSpeedLimit:
    """Tests detecting the ego vehicle exceeding the speed limit of its lane."""

    def test_exceeding_the_urban_speed_limit(self):
        """Driving 15 m/s for three samples exceeds the 50 km/h of urban roads once, for 0.3 s."""
        road, _ = two_lane_road()
        egos = [{"x": 10.0 + index, "vel_lon": 15.0 if index < 3 else 10.0} for index in range(5)]

        speed_limit = _evaluate(egos, road)["speed_limit"]

        assert speed_limit["_value_"] == pytest.approx(0.6)
        assert speed_limit["time"] == pytest.approx(0.3)
        assert speed_limit["num_violations"] == 1
        assert speed_limit["max_excess"] == pytest.approx(15.0 - 50.0 / 3.6)

    def test_speed_limit_set_for_the_lanelet(self):
        """A lanelet limited to 30 km/h is exceeded at 10 m/s."""
        road, _ = road_with_sidewalk()

        speed_limit = _evaluate([{"x": 10.0, "vel_lon": 10.0}], road)["speed_limit"]

        assert speed_limit["_value_"] == 1.0
        assert speed_limit["max_excess"] == pytest.approx(10.0 - 30.0 / 3.6)

    def test_tolerance(self):
        """Exceeding the limit by less than the tolerance is no violation, but still reported as excess."""
        road, _ = two_lane_road()

        speed_limit = _evaluate([{"x": 10.0, "vel_lon": 15.0}], road, config={"speed_limit": {"tolerance": 2.0}})["speed_limit"]

        assert speed_limit["_value_"] == 0.0
        assert speed_limit["max_excess"] == pytest.approx(15.0 - 50.0 / 3.6)

    def test_cannot_be_checked_without_velocity(self):
        """Without velocity of the ego vehicle, its speed is unknown."""
        road, _ = two_lane_road()

        speed_limit = _evaluate([{"x": 10.0}], road)["speed_limit"]

        assert speed_limit["_value_"] is None
        assert speed_limit["max_excess"] is None


class TestRuleComplianceSelection:
    """Tests selecting the rule compliance metrics."""

    def test_reads_the_ego_vehicle_and_the_optional_map_only(self):
        """The rules are checked against the map, without other road users."""
        evaluation = RuleCompliance()

        inputs = evaluation.inputs()

        assert {name: declared.role for name, declared in inputs.items()} == {"ego_data": "evaluated", "map": "context"}
        assert inputs["map"].optional
