# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for the progress evaluation of the driving behavior: the distance driven and the progress along the route.

The ego vehicle drives along the x axis, sampled at 10 Hz, at speeds simple enough to compute the
distance and the time standing still by hand, along a route of 100 m along the x axis.
"""

from __future__ import annotations

import pytest
from autonomy_evaluation.evaluations.driving_behavior.progress.Progress import Progress
from geometry_msgs.msg import Point
from tests.evaluations.driving_behavior.messages import ego_data, route, static_tf_buffer

_PERIOD = 0.1


def _evaluate(egos, scene_ids=None, config=None, routes=None, tf_buffer=None) -> dict:
    """Record one sample per set of ``ego_data`` keyword arguments, 0.1 s apart, with the given routes, and aggregate them."""
    evaluation = Progress()
    evaluation.configure(config or {})
    evaluation.tf_buffer = tf_buffer
    for index, ego in enumerate(egos):
        stamp = index * _PERIOD
        evaluation.record_sample(
            sample_id=str(index),
            scene_id=scene_ids[index] if scene_ids else None,
            stamp=stamp,
            ego_data=ego_data(stamp=stamp, **ego),
            route=routes[index] if routes else None,
        )
    return evaluation.finalize()["metrics"]


# a route of 100 m along the x axis, through a point every 10 m
_ROUTE = route([(10.0 * index, 0.0) for index in range(11)])


class TestDistanceDriven:
    """Tests the distance the ego vehicle drives, its mean speed and the time it stands still."""

    def test_driving_and_waiting(self):
        """Driving 10 m/s for 1 s covers 10 m, then waiting for five samples stands still for 0.5 s."""
        egos = [{"x": 10.0 * _PERIOD * index, "vel_lon": 10.0} for index in range(11)]
        egos += [{"x": 10.0, "vel_lon": 0.0} for _ in range(5)]

        distance = _evaluate(egos)["distance"]

        assert distance["_value_"] == pytest.approx(10.0)
        assert distance["mean_speed"] == pytest.approx(10.0 / 1.5)
        assert distance["standstill_time"] == pytest.approx(0.5)
        assert distance["standstill_share"] == pytest.approx(5 / 16)

    def test_distance_does_not_run_between_scenes(self):
        """The ego vehicle of the next scene starts elsewhere, which is no distance driven."""
        egos = [{"x": 0.0}, {"x": 1.0}, {"x": 100.0}, {"x": 101.0}]

        distance = _evaluate(egos, scene_ids=["a", "a", "b", "b"])["distance"]

        assert distance["_value_"] == pytest.approx(2.0)
        assert distance["standstill_share"] is None

    def test_teleport_is_not_driven(self):
        """A simulation teleporting the ego vehicle to the start of its scenario does not drive it there."""
        egos = [{"x": 0.0}, {"x": 1.0}, {"x": 351.0}, {"x": 352.0}]

        assert _evaluate(egos)["distance"]["_value_"] == pytest.approx(2.0)

    def test_single_sample_has_no_mean_speed(self):
        """Without time passing, there is no mean speed."""
        assert _evaluate([{"x": 0.0, "vel_lon": 5.0}])["distance"]["mean_speed"] is None


class TestRouteProgress:
    """Tests how far the ego vehicle gets along the route planned for it."""

    def test_driving_half_of_the_route(self):
        """Driving from the start of a 100 m route to its middle completes half of it."""
        egos = [{"x": 5.0 * index} for index in range(11)]

        progress = _evaluate(egos, routes=[_ROUTE] * len(egos))["route"]

        assert progress["_value_"] == pytest.approx(0.5)
        assert progress["distance"] == pytest.approx(50.0)
        assert progress["remaining_distance"] == pytest.approx(50.0)
        assert (progress["goals_reached"], progress["num_routes"], progress["coverage"]) == (0, 1, 1.0)

    def test_reaching_the_destination(self):
        """The destination counts as reached within the goal tolerance of 3 m along the route."""
        egos = [{"x": 0.0}, {"x": 50.0}, {"x": 98.0}]

        progress = _evaluate(egos, routes=[_ROUTE] * len(egos))["route"]

        assert progress["goals_reached"] == 1
        assert progress["_value_"] == pytest.approx(0.98)

    def test_new_destination_starts_a_new_route(self):
        """A destination that moves farther than the goal tolerance is a new route, each completed on its own."""
        to_middle = route([(10.0 * index, 0.0) for index in range(11)], destination=(60.0, 0.0))
        egos = [{"x": 0.0}, {"x": 50.0}, {"x": 50.0}, {"x": 60.0}]

        progress = _evaluate(egos, routes=[_ROUTE, _ROUTE, to_middle, to_middle])["route"]

        assert progress["num_routes"] == 2
        assert progress["_value_"] == pytest.approx((0.5 + 1.0) / 2.0)
        assert progress["goals_reached"] == 1

    def test_route_window_moving_along_with_the_ego_vehicle(self):
        """A planner publishing 50 m of a 150 m route around the ego vehicle, the destination only once it is near.

        The arc lengths count from the start of the whole route, so the ego vehicle progresses along the route
        across the windows, and the destination at 150 m is known from the last window on.
        """

        def window(start: float, destination_index=None):
            return route(
                [(start + 10.0 * index, 0.0) for index in range(6)],
                destination=(150.0, 0.0),
                start_s=start,
                destination_index=destination_index,
            )

        egos = [{"x": 0.0}, {"x": 60.0}, {"x": 120.0}]
        routes = [window(0.0), window(50.0), window(100.0, destination_index=5)]

        progress = _evaluate(egos, routes=routes)["route"]

        assert progress["distance"] == pytest.approx(120.0)
        assert progress["remaining_distance"] == pytest.approx(30.0)
        assert progress["_value_"] == pytest.approx(120.0 / 150.0)
        assert progress["num_routes"] == 1

    def test_completion_needs_the_destination_within_the_route(self):
        """While the destination lies beyond every published window, only the progress is known."""
        far = route([(10.0 * index, 0.0) for index in range(6)], destination=(500.0, 0.0), destination_index=None)

        progress = _evaluate([{"x": 0.0}, {"x": 30.0}], routes=[far] * 2)["route"]

        assert progress["distance"] == pytest.approx(30.0)
        assert progress["_value_"] is None
        assert (progress["goals_reached"], progress["num_routes"]) == (0, 1)

    def test_lateral_distance_to_the_suggested_lane(self):
        """The path of the route follows the suggested lane, here the left one of two lanes 3.5 m apart."""
        two_lanes = route([(10.0 * index, 0.0) for index in range(11)], lane_offsets=(0.0, 3.5), suggested_lane=1)

        progress = _evaluate([{"x": 0.0, "y": 0.0}, {"x": 10.0, "y": 3.5}], routes=[two_lanes] * 2)["route"]

        assert progress["max_lateral_distance"] == pytest.approx(3.5)

    def test_route_in_another_frame_is_transformed(self):
        """A route in 'odom', whose origin lies 20 m along x in 'map', ends at 100 m in the frame of the ego vehicle."""
        in_odom = route([(10.0 * index - 20.0, 0.0) for index in range(11)], frame_id="odom")

        progress = _evaluate([{"x": 0.0}, {"x": 20.0}], routes=[in_odom] * 2, tf_buffer=static_tf_buffer("map", "odom", x=20.0))[
            "route"
        ]

        # untransformed, the route would end at 80 m and leave 60 m
        assert progress["remaining_distance"] == pytest.approx(80.0)

    def test_route_planned_with_the_ego_data_without_route_message(self):
        """Without route message, the route planned in the ego data is followed towards its last point."""
        planned = [Point(x=10.0 * index) for index in range(11)]
        egos = [{"x": 0.0}, {"x": 25.0}]
        evaluation = Progress()
        evaluation.configure({"metrics": ["route"]})
        for index, ego in enumerate(egos):
            message = ego_data(stamp=index * _PERIOD, **ego)
            message.route_planned = planned
            evaluation.record_sample(sample_id=str(index), stamp=index * _PERIOD, ego_data=message, route=None)

        assert evaluation.finalize()["metrics"]["route"]["_value_"] == pytest.approx(0.25)

    def test_without_route_nothing_is_measured(self):
        """Without route, there is no progress along one."""
        progress = _evaluate([{"x": 0.0}, {"x": 5.0}])["route"]

        assert progress["_value_"] is None
        assert (progress["num_routes"], progress["coverage"]) == (0, 0.0)

    def test_reads_the_latest_route_without_waiting_for_it(self):
        """The route is planned independently of the samples, so each sample is passed the latest one."""
        evaluation = Progress()
        evaluation.configure({"metrics": ["route"]})

        declared = evaluation.inputs()["route"]

        assert (declared.latest, declared.optional, declared.role) == (True, True, "context")
