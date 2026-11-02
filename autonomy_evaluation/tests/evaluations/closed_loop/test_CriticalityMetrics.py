# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for the criticality metrics of the closed-loop evaluation: collision, time to collision and time headway.

Cars of 4 m length and 2 m width drive scenarios simple enough to compute the expected values by
hand, e.g. the ego vehicle approaching a standing car at 10 m/s or following a car around a bend.
"""

from __future__ import annotations

import math

import pytest
from autonomy_evaluation.evaluations.closed_loop.ClosedLoop import ClosedLoop
from tests.evaluations.closed_loop.messages import ego_data, object_list, road_user
from tests.utils.lanelet2_maps import bend, junction, two_lane_road

_PERIOD = 0.1


def _evaluate(samples, metrics=None, config=None, scene_ids=None, lanelet2_map=None) -> dict:
    """Record ``(stamp, ego_data keyword arguments, road users)`` samples, with the map if given, and aggregate them."""
    evaluation = ClosedLoop()
    evaluation.configure({"metrics": metrics or ["criticality"], **(config or {})})
    for index, (stamp, ego, road_users) in enumerate(samples):
        evaluation.record_sample(
            sample_id=str(index),
            scene_id=scene_ids[index] if scene_ids else None,
            stamp=stamp,
            ego_data=ego_data(stamp=stamp, **ego),
            objects=object_list(road_users, stamp=stamp),
            map=lanelet2_map,
        )
    return evaluation.finalize()["metrics"]["criticality"]


def _collision(ego_x, ego_speed, other_x, other_y=0.0, other_yaw=0.0, other_speed=0.0) -> list:
    """Two samples, 1 s apart, of an ego vehicle on the x axis approaching another road user and overlapping it in the second."""
    ego = {"yaw": 0.0, "vel_lon": ego_speed}
    other = {"yaw": other_yaw, "vel_lon": other_speed}
    other_before = (other_x - other_speed * math.cos(other_yaw), other_y - other_speed * math.sin(other_yaw))
    return [
        (0.0, {"x": ego_x - ego_speed, **ego}, [road_user(1, *other_before, **other)]),
        (1.0, {"x": ego_x, **ego}, [road_user(1, other_x, other_y, **other)]),
    ]


class TestCollision:
    """Tests detecting and classifying the collisions of the ego vehicle."""

    @pytest.mark.parametrize(
        "samples, collision_type, at_fault",
        [
            # driving into a standing car
            (_collision(ego_x=7.0, ego_speed=10.0, other_x=10.0), "stopped_object", "at_fault"),
            # driving into a slower car ahead
            (_collision(ego_x=7.0, ego_speed=10.0, other_x=10.0, other_speed=5.0), "active_front", "at_fault"),
            # hit from behind while waiting
            (_collision(ego_x=0.0, ego_speed=0.0, other_x=-3.0, other_speed=5.0), "stopped_ego", "not_at_fault"),
            # hit from behind by a faster car
            (_collision(ego_x=0.0, ego_speed=5.0, other_x=-3.0, other_speed=10.0), "active_rear", "not_at_fault"),
            # hit at the left side by a car crossing from the left
            (
                _collision(ego_x=0.0, ego_speed=10.0, other_x=0.0, other_y=2.5, other_yaw=-math.pi / 2.0, other_speed=5.0),
                "active_lateral",
                "undetermined",
            ),
        ],
    )
    def test_classifies_collisions_like_nuplan(self, samples, collision_type, at_fault):
        """Whether the ego vehicle is at fault follows from who stood still and which side was hit."""
        collision = _evaluate(samples, metrics=["criticality.collision"])["collision"]

        assert collision["_value_"] == 1
        assert collision["types"][collision_type] == 1
        assert collision[at_fault] == 1

    def test_overlap_over_consecutive_samples_is_one_collision(self):
        """A collision lasts as long as the road users overlap."""
        overlapping = [(index * _PERIOD, {"vel_lon": 1.0}, [road_user(1, 3.0, 0.0)]) for index in range(3)]

        assert _evaluate(overlapping)["collision"]["_value_"] == 1

    def test_overlapping_again_is_another_collision(self):
        """Road users that separate and overlap again collide twice."""
        positions = [3.0, 10.0, 3.0]
        samples = [(index * _PERIOD, {"vel_lon": 1.0}, [road_user(1, x, 0.0)]) for index, x in enumerate(positions)]

        assert _evaluate(samples)["collision"]["_value_"] == 2

    def test_collision_continuing_into_a_new_scene_counts_again(self):
        """Scenes are evaluated on their own, so an overlap at the start of a scene is a collision of that scene."""
        samples = [(index * _PERIOD, {"vel_lon": 1.0}, [road_user(1, 3.0, 0.0)]) for index in range(2)]

        assert _evaluate(samples, scene_ids=["a", "b"])["collision"]["_value_"] == 2

    def test_reports_the_relative_speed_at_impact(self):
        """The severity of a collision grows with the relative speed of the road users."""
        collision = _evaluate(_collision(ego_x=7.0, ego_speed=10.0, other_x=10.0, other_speed=4.0))["collision"]

        assert collision["max_relative_speed"] == pytest.approx(6.0)

    def test_no_collision(self):
        """Road users that keep their distance never collide."""
        collision = _evaluate([(0.0, {"vel_lon": 10.0}, [road_user(1, 20.0, 0.0)])])["collision"]

        assert collision == {
            "_value_": 0,
            "at_fault": 0,
            "not_at_fault": 0,
            "undetermined": 0,
            "types": {"active_front": 0, "active_rear": 0, "active_lateral": 0, "stopped_ego": 0, "stopped_object": 0},
            "max_relative_speed": None,
        }

    def test_ego_vehicle_among_the_objects_is_no_collision(self):
        """A simulation listing the ego vehicle among the objects does not make it collide with itself."""
        samples = [(0.0, {"vel_lon": 10.0}, [road_user(0, 0.0, 0.0, vel_lon=10.0)])]

        assert _evaluate(samples)["collision"]["_value_"] == 1
        assert _evaluate(samples, config={"ego_object_id": 0})["collision"]["_value_"] == 0


def _standing_car_ahead(gap: float, num_samples: int = 10, ego_speed: float = 10.0) -> list:
    """Samples of the ego vehicle approaching a standing car ahead, the gap between them kept constant."""
    return [(index * _PERIOD, {"vel_lon": ego_speed}, [road_user(1, gap + 4.0, 0.0)]) for index in range(num_samples)]


class TestTimeToCollision:
    """Tests the time to collision of the ego vehicle and its exposure to small ones."""

    def test_time_to_collision_with_a_standing_car(self):
        """A standing car 20 m ahead is reached at 10 m/s after 2 s."""
        assert _evaluate(_standing_car_ahead(gap=20.0))["ttc"]["_value_"] == pytest.approx(2.0, abs=1e-3)

    def test_time_exposed_and_integrated(self):
        """A TTC of 2 s for 1 s is exposed below 3 s for 1 s, integrating to (3 s - 2 s) · 1 s."""
        ttc = _evaluate(_standing_car_ahead(gap=20.0))["ttc"]

        assert ttc["share"] == 1.0
        assert ttc["time_exposed"] == pytest.approx(1.0)
        assert ttc["time_integrated"] == pytest.approx(1.0, abs=1e-2)

    def test_exposure_threshold_is_configurable(self):
        """Below a TTC* of 1.5 s, a TTC of 2 s is no exposure."""
        ttc = _evaluate(_standing_car_ahead(gap=20.0), config={"criticality": {"ttc": {"exposure_threshold": 1.5}}})["ttc"]

        assert ttc["time_exposed"] == 0.0
        assert ttc["time_integrated"] == 0.0

    def test_no_time_to_collision_beyond_the_horizon(self):
        """A car reached after 4.6 s has no TTC within a horizon of 3 s."""
        samples = _standing_car_ahead(gap=46.0)

        assert _evaluate(samples)["ttc"]["_value_"] == pytest.approx(4.6, abs=1e-3)
        ttc = _evaluate(samples, config={"criticality": {"ttc": {"horizon": 3.0}}})["ttc"]
        assert ttc["_value_"] is None
        assert ttc["share"] == 0.0

    def test_ego_vehicle_without_velocity_has_no_time_to_collision(self):
        """Without its velocity, the motion of the ego vehicle cannot be extrapolated."""
        ttc = _evaluate([(0.0, {}, [road_user(1, 10.0, 0.0)])])["ttc"]

        assert ttc == {"_value_": None, "share": None, "time_exposed": 0.0, "time_integrated": 0.0}


class TestTimeHeadway:
    """Tests the time headway of the ego vehicle to the road user it follows."""

    def test_time_headway_to_the_car_ahead(self):
        """A car 40 m ahead of the front of the ego vehicle driving 20 m/s is 2 s ahead.

        A car in the neighboring lane and a car behind are not followed.
        """
        road_users = [road_user(1, 44.0, 0.0, vel_lon=20.0), road_user(2, 10.0, 3.5), road_user(3, -10.0, 0.0)]

        thw = _evaluate([(0.0, {"vel_lon": 20.0, "yaw_rate": 0.0}, road_users)])["thw"]

        assert thw == {"_value_": pytest.approx(2.0), "mean": pytest.approx(2.0), "share": 1.0, "sources": ["corridor"]}

    def test_follows_the_car_ahead_around_a_bend(self):
        """In a bend of 100 m radius, the car 50 m ahead along the bend lies 12 m to the left, but is followed."""
        bend_angle = 0.5
        ahead = road_user(1, 100.0 * math.sin(bend_angle), 100.0 * (1.0 - math.cos(bend_angle)), yaw=bend_angle)

        straight = _evaluate([(0.0, {"vel_lon": 20.0}, [ahead])])["thw"]
        bend = _evaluate([(0.0, {"vel_lon": 20.0, "yaw_rate": 0.2}, [ahead])])["thw"]

        assert straight["_value_"] is None
        assert bend["_value_"] == pytest.approx((50.0 - 2.0 - 2.0) / 20.0)

    def test_follows_the_closest_car_ahead(self):
        """Of two cars ahead in the lane, the closer one is followed."""
        road_users = [road_user(1, 64.0, 0.0), road_user(2, 24.0, 0.0)]

        assert _evaluate([(0.0, {"vel_lon": 10.0}, road_users)])["thw"]["_value_"] == pytest.approx(2.0)

    def test_standing_ego_vehicle_has_no_time_headway(self):
        """An ego vehicle that hardly moves would take forever to reach any car ahead."""
        thw = _evaluate([(0.0, {"vel_lon": 0.2}, [road_user(1, 10.0, 0.0)])])["thw"]

        assert thw == {"_value_": None, "mean": None, "share": 0.0, "sources": []}

    def test_mean_and_share_over_the_samples_following_a_car(self):
        """The mean is taken over the samples in which the ego vehicle follows a car, the share over all."""
        samples = [
            (0.0, {"vel_lon": 10.0}, [road_user(1, 14.0, 0.0)]),
            (_PERIOD, {"vel_lon": 10.0}, [road_user(1, 34.0, 0.0)]),
            (2 * _PERIOD, {"vel_lon": 10.0}, []),
        ]

        thw = _evaluate(samples)["thw"]

        assert thw["_value_"] == pytest.approx(1.0)
        assert thw["mean"] == pytest.approx(2.0)
        assert thw["share"] == pytest.approx(2.0 / 3.0)


def _drive(duration: float, road_users, period: float = _PERIOD, ego_speed: float = 10.0) -> list:
    """Samples of the ego vehicle driving along the x axis from x = -20 m, with road users given as functions of time."""
    return [
        (
            index * period,
            {"x": -20.0 + ego_speed * index * period, "vel_lon": ego_speed},
            [user(index * period) for user in road_users],
        )
        for index in range(int(round(duration / period)) + 1)
    ]


def _crossing_car(t: float) -> object:
    """A car crossing the path of the ego vehicle at x = 0 northwards at 10 m/s, reaching y = -20 m at 3 s."""
    return road_user(1, 0.0, -50.0 + 10.0 * t, yaw=math.pi / 2.0, vel_lon=10.0)


class TestPostEncroachmentTime:
    """Tests the post-encroachment time of the ego vehicle and the other road users passing the same spots."""

    def test_crossing_behind_the_ego_vehicle(self):
        """A car crossing 2.45 s after the ego vehicle left the intersection, as computed for the occupied cells by hand.

        The ego vehicle passes x = 0.75 m with its rear at 2.275 s, the crossing car reaches y = -0.75 m with its front at
        4.725 s, the closest pair of leaving and entering among the shared cells of 0.5 m. Both times are estimated to
        within half the time step of 0.1 s, so the PET is resolved to within the time step.
        """
        pet = _evaluate(_drive(7.0, [_crossing_car]))["pet"]

        assert pet["_value_"] == pytest.approx(2.45, abs=0.1)
        assert pet["num_encounters"] == 1
        assert pet["ego_first"] == 1
        assert pet["types"]["crossing"]["num_encounters"] == 1

    def test_following_a_car(self):
        """The ego vehicle reaches the spot the rear of a car 16 m ahead left after 1.6 s at 10 m/s."""

        def lead(t: float) -> object:
            return road_user(2, 0.0 + 10.0 * t, 0.0, vel_lon=10.0)

        pet = _evaluate(_drive(5.0, [lead]))["pet"]

        assert pet["_value_"] == pytest.approx(1.6, abs=0.1)
        assert pet["ego_first"] == 0
        assert pet["types"]["rear_end"] == {"_value_": pet["_value_"], "num_encounters": 1}

    def test_car_in_the_neighboring_lane_is_no_encounter(self):
        """Road users in parallel lanes never pass the same spot."""

        def neighbor(t: float) -> object:
            return road_user(3, -10.0 + 10.0 * t, 3.5, vel_lon=10.0)

        pet = _evaluate(_drive(5.0, [neighbor]))["pet"]

        assert pet["_value_"] is None
        assert pet["num_encounters"] == 0

    def test_spots_passed_long_after_each_other_are_no_encounter(self):
        """A car 146 m ahead passes the same spots 14.6 s before the ego vehicle, beyond the maximum PET of 10 s."""

        def far_ahead(t: float) -> object:
            return road_user(4, 130.0 + 10.0 * t, 0.0, vel_lon=10.0)

        samples = _drive(20.0, [far_ahead], period=1.0)

        assert _evaluate(samples)["pet"]["num_encounters"] == 0
        assert _evaluate(samples, config={"criticality": {"pet": {"max_pet": 20.0}}})["pet"]["_value_"] == pytest.approx(
            14.6, abs=0.1
        )

    def test_road_users_of_different_scenes_never_encounter(self):
        """The crossing car enters the intersection in a scene of its own, which is a sequence of its own."""
        samples = _drive(7.0, [_crossing_car])
        scene_ids = ["a" if stamp < 3.5 else "b" for stamp, _, _ in samples]

        assert _evaluate(samples, scene_ids=scene_ids)["pet"]["num_encounters"] == 0


class TestVisualization:
    """Tests publishing the road users behind the criticality of a sample for RViz."""

    def test_publishes_the_colliding_approaching_and_followed_road_users(self):
        """Criticality metrics with a per-sample outcome publish the road users behind it, in the frame of the ego vehicle."""
        evaluation = ClosedLoop()
        messages = {
            "ego_data": ego_data(vel_lon=10.0, yaw_rate=0.0),
            "objects": object_list(
                [
                    road_user(1, 3.0, 0.0),  # overlaps the front of the ego vehicle
                    road_user(2, 24.0, 3.5, yaw=math.pi, vel_lon=10.0),  # oncoming in the neighboring lane
                    road_user(3, 30.0, -15.0, yaw=math.pi / 2.0, vel_lon=5.0),  # crossing the path when the ego vehicle passes
                ]
            ),
        }
        evaluation.record_sample(sample_id="0", stamp=0.0, **messages)

        visualization = evaluation.visualize_sample(sample_id="0", **messages)

        assert sorted(evaluation.visualization_outputs()) == ["approaching_objects", "colliding_objects", "lead_vehicle"]
        assert [obj.id for obj in visualization["colliding_objects"].objects] == [1]
        assert [obj.id for obj in visualization["approaching_objects"].objects] == [1, 3]
        assert [obj.id for obj in visualization["lead_vehicle"].objects] == [1]
        assert visualization["lead_vehicle"].header.frame_id == "map"

    def test_publishes_the_outputs_of_the_computed_metrics_only(self):
        """Comfort metrics have no per-sample outcome to publish."""
        evaluation = ClosedLoop()
        evaluation.configure({"metrics": ["comfort", "criticality.thw"]})

        assert list(evaluation.visualization_outputs()) == ["lead_vehicle"]


class TestTimeHeadwayAlongLanes:
    """Tests finding the road user the ego vehicle follows along the lanes of the Lanelet2 map."""

    def test_follows_the_car_ahead_in_the_following_lanelet(self):
        """The lane continues into the following lanelet; a closer car in the neighboring lane is not followed."""
        road, _ = two_lane_road()
        road_users = [road_user(1, 134.0, 0.0, vel_lon=20.0), road_user(2, 40.0, 3.5, vel_lon=20.0)]

        thw = _evaluate([(0.0, {"x": 10.0, "vel_lon": 20.0}, road_users)], lanelet2_map=road)["thw"]

        assert thw["_value_"] == pytest.approx((124.0 - 2.0 - 2.0) / 20.0)
        assert thw["sources"] == ["lanes"]

    def test_follows_the_car_ahead_around_a_bend_without_yaw_rate(self):
        """The lane bends, so the car 50 m ahead along it is followed, although the ego vehicle sets no yaw rate."""
        road, _ = bend(radius=100.0, angle=1.0)
        ahead = road_user(1, 100.0 * math.sin(0.5), 100.0 * (1.0 - math.cos(0.5)), yaw=0.5)

        with_map = _evaluate([(0.0, {"vel_lon": 20.0}, [ahead])], lanelet2_map=road)["thw"]
        without_map = _evaluate([(0.0, {"vel_lon": 20.0}, [ahead])])["thw"]

        assert with_map["_value_"] == pytest.approx((50.0 - 2.0 - 2.0) / 20.0, abs=0.01)
        assert without_map["_value_"] is None

    def test_follows_the_closest_car_on_any_branch_of_a_junction(self):
        """Which branch the ego vehicle takes is unknown, so the closest car on any of them is followed.

        The car on the branch is 50 m + 25 m ahead along the lanes, the car straight on 110 m.
        """
        road, _ = junction()
        on_branch = road_user(2, 100.0 + 25.0 / math.sqrt(2.0), 25.0 / math.sqrt(2.0), yaw=math.pi / 4.0)
        road_users = [road_user(1, 160.0, 0.0), on_branch]

        thw = _evaluate([(0.0, {"x": 50.0, "vel_lon": 10.0}, road_users)], lanelet2_map=road)["thw"]

        assert thw["_value_"] == pytest.approx((75.0 - 2.0 - 2.0) / 10.0, abs=0.01)

    def test_falls_back_to_the_corridor_off_the_lanes(self):
        """An ego vehicle driving along no lanelet, e.g. on a parking lot, follows the corridor ahead."""
        road, _ = two_lane_road()

        thw = _evaluate([(0.0, {"x": 10.0, "y": -20.0, "vel_lon": 10.0}, [road_user(1, 34.0, -20.0)])], lanelet2_map=road)["thw"]

        assert thw["_value_"] == pytest.approx(2.0)
        assert thw["sources"] == ["corridor"]

    def test_map_in_another_frame_is_not_used(self):
        """Without transform between map and ego vehicle, the lanes cannot be related to the ego vehicle."""
        road, _ = two_lane_road()
        road.frame_id = "utm"

        thw = _evaluate([(0.0, {"x": 10.0, "vel_lon": 20.0}, [road_user(1, 134.0, 0.0)])], lanelet2_map=road)["thw"]

        assert thw["sources"] == ["corridor"]


class TestLateralCollisionFault:
    """Tests deciding who is at fault for a lateral collision by the lanes the ego vehicle covers."""

    @staticmethod
    def _lateral_collision(ego_y: float, lanelet2_map) -> dict:
        """A car overlapping the left side of the ego vehicle, which drives along the road at 10 m/s."""
        samples = [
            (0.0, {"x": 50.0, "y": ego_y, "vel_lon": 10.0}, [road_user(1, 50.0, ego_y + 5.0, vel_lon=10.0)]),
            (1.0, {"x": 60.0, "y": ego_y, "vel_lon": 10.0}, [road_user(1, 60.0, ego_y + 1.8, vel_lon=10.0)]),
        ]
        return _evaluate(samples, metrics=["criticality.collision"], lanelet2_map=lanelet2_map)["collision"]

    def test_ego_vehicle_in_its_lane_is_not_at_fault(self):
        """A car changing into the lane of the ego vehicle causes the collision."""
        road, _ = two_lane_road()

        collision = self._lateral_collision(ego_y=0.0, lanelet2_map=road)

        assert collision["types"]["active_lateral"] == 1
        assert collision["not_at_fault"] == 1

    def test_ego_vehicle_covering_two_lanes_is_at_fault(self):
        """An ego vehicle straddling the lane boundary causes the collision."""
        road, _ = two_lane_road()

        assert self._lateral_collision(ego_y=1.0, lanelet2_map=road)["at_fault"] == 1

    def test_ego_vehicle_leaving_the_lanes_is_at_fault(self):
        """An ego vehicle off the road causes the collision."""
        road, _ = two_lane_road()

        assert self._lateral_collision(ego_y=-3.0, lanelet2_map=road)["at_fault"] == 1

    def test_fault_is_undetermined_without_map(self):
        """Without lanes, it is unknown which road user left its lane."""
        assert self._lateral_collision(ego_y=0.0, lanelet2_map=None)["undetermined"] == 1


class TestCriticalitySelection:
    """Tests selecting the criticality metrics."""

    def test_criticality_reads_the_objects_as_context(self):
        """The other road users are needed to evaluate the ego vehicle, but not evaluated themselves."""
        evaluation = ClosedLoop()
        evaluation.configure({"metrics": ["criticality"]})

        inputs = evaluation.inputs()

        assert {name: declared.role for name, declared in inputs.items()} == {
            "ego_data": "evaluated",
            "objects": "context",
            "map": "context",
        }
        # the map is no topic, but context the node passes with every sample, and the metrics fall back to heuristics without
        assert inputs["map"].static and inputs["map"].optional

    def test_all_groups_are_selected_by_default(self):
        """By default, the closed-loop evaluation computes its criticality, rule compliance, progress and comfort."""
        assert ClosedLoop().config["metrics"] == ["criticality", "rule_compliance", "progress", "comfort"]
