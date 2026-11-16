# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for querying the lanes of a Lanelet2 map: which lane a road user drives in and what lies ahead of it."""

from __future__ import annotations

import math

import pytest
from autonomy_evaluation.utils.Lanelet2Map import Lanelet2Map
from tests.utils.lanelet2_maps import bend, junction, ORIGIN, osm_contents, road_with_sidewalk, two_lane_road


class TestLoading:
    """Tests loading a map from the OSM XML a map server holds in its parameters."""

    def test_loads_a_map_written_as_osm(self):
        """A map projected around its origin is loaded back in the frame of the map, with its lanes connected."""
        written, _ = two_lane_road()

        loaded = Lanelet2Map.from_osm(osm_contents(written), *ORIGIN, frame_id="map")

        assert loaded.num_lanelets == 3
        assert loaded.frame_id == "map"
        lane = loaded.lane_of(150.0, 0.0, 0.0)
        assert loaded.lane_coordinates(lane, 150.0, 0.2)[0:2] == pytest.approx((50.0, 0.2), abs=1e-3)
        assert len(loaded.paths_ahead(loaded.lane_of(50.0, 0.0, 0.0), 100.0)[0]) == 2

    @pytest.mark.parametrize("contents", ["", "  ", '<?xml version="1.0"?><osm version="0.6" generator="blank"></osm>'])
    def test_map_without_lanelets_is_no_map(self, contents):
        """A blank map, as a dataset without map publishes it, is no map."""
        assert Lanelet2Map.from_osm(contents, *ORIGIN) is None


class TestLanes:
    """Tests finding the lane of a road user and its coordinates along the lane."""

    def setup_method(self):
        """Build a straight road of two lanes."""
        self.map, self.lanelets = two_lane_road()

    def test_lanelets_containing_a_point(self):
        """A point in the right lane lies in its lanelet only."""
        assert [lanelet.id for lanelet in self.map.lanelets_at(50.0, 0.5)] == [self.lanelets["right"].id]
        assert self.map.lanelets_at(50.0, -5.0) == []

    def test_lane_follows_the_heading(self):
        """A road user driving along the lane is in it, one driving against it is in no lane."""
        assert self.map.lane_of(50.0, 3.5, 0.2).id == self.lanelets["left"].id
        assert self.map.lane_of(50.0, 3.5, math.pi) is None

    def test_coordinates_along_the_lane(self):
        """Arc length, lateral offset to the left and heading follow the centerline of the lanelet."""
        s, d, heading = self.map.lane_coordinates(self.lanelets["right"], 40.0, 0.5)

        assert (s, d, heading) == pytest.approx((40.0, 0.5, 0.0))
        assert self.map.lane_coordinates(self.lanelets["right"], 40.0, -0.5)[1] == pytest.approx(-0.5)

    def test_length_of_a_lanelet(self):
        """The length of a lanelet is the length of its centerline."""
        assert self.map.length(self.lanelets["right"]) == pytest.approx(100.0)

    def test_consecutive_lanelets_are_one_lane(self):
        """Lanelets following each other form one lane, neighboring lanelets do not."""
        assert self.map.connected(self.lanelets["right"], self.lanelets["right_next"])
        assert self.map.connected(self.lanelets["right_next"], self.lanelets["right"])
        assert not self.map.connected(self.lanelets["right"], self.lanelets["left"])


class TestPathsAhead:
    """Tests listing the lanelets a road user may follow."""

    def test_path_continues_into_the_following_lanelet(self):
        """A lane that continues is followed up to the requested distance."""
        road, lanelets = two_lane_road()

        paths = road.paths_ahead(lanelets["right"], 150.0)

        assert [[lanelet.id for lanelet in path] for path in paths] == [[lanelets["right"].id, lanelets["right_next"].id]]

    def test_path_ends_with_the_lane(self):
        """A lane that does not continue ends the path, however far it is requested."""
        road, lanelets = two_lane_road()

        assert [len(path) for path in road.paths_ahead(lanelets["left"], 500.0)] == [1]

    def test_paths_branch_at_a_junction(self):
        """A lane continuing straight and turning off yields a path for each."""
        road, lanelets = junction()

        paths = road.paths_ahead(lanelets["approach"], 150.0)

        assert sorted(path[-1].id for path in paths) == sorted([lanelets["straight"].id, lanelets["turn"].id])

    def test_coordinates_along_a_bend(self):
        """The arc length along a bend follows its centerline, not the straight line."""
        road, lanelets = bend(radius=100.0, angle=1.0)
        x, y = 100.0 * math.sin(0.5), 100.0 * (1.0 - math.cos(0.5))

        s, d, heading = road.lane_coordinates(lanelets["bend"], x, y)

        assert s == pytest.approx(50.0, abs=0.1)
        assert d == pytest.approx(0.0, abs=0.05)
        assert heading == pytest.approx(0.5, abs=0.02)


class TestTrafficRules:
    """Tests what the traffic rules of vehicles allow: where, in which direction and how fast to drive."""

    def setup_method(self):
        """Build a straight road of two one-way lanes and a two-way road with a sidewalk."""
        self.road, self.road_lanelets = two_lane_road()
        self.two_way, self.two_way_lanelets = road_with_sidewalk()

    def test_sidewalks_are_not_drivable(self):
        """Vehicles may drive in road lanelets, but not on sidewalks."""
        assert self.two_way.drivable(self.two_way_lanelets["road"])
        assert not self.two_way.drivable(self.two_way_lanelets["sidewalk"])

    def test_distance_to_the_drivable_area(self):
        """A point on the road is within the drivable area, one on the sidewalk or beside it at its distance."""
        assert self.two_way.distance_to_drivable_area(50.0, 1.0) == 0.0
        assert self.two_way.distance_to_drivable_area(50.0, -2.75) == pytest.approx(1.0)
        assert self.two_way.distance_to_drivable_area(50.0, 4.75) == pytest.approx(3.0)

    def test_driving_against_a_one_way_lane(self):
        """Driving against the direction of a one-way lane goes against the direction of travel."""
        assert self.road.against_direction(50.0, 0.0, 0.1) is False
        assert self.road.against_direction(50.0, 0.0, math.pi) is True
        assert self.road.against_direction(50.0, -5.0, math.pi) is None

    def test_a_two_way_road_may_be_driven_in_both_directions(self):
        """A lanelet that may be driven in both directions is never driven against."""
        assert self.two_way.against_direction(50.0, 0.0, math.pi) is False
        assert self.two_way.lane_of(50.0, 0.0, math.pi).inverted()

    def test_speed_limit_in_meters_per_second(self):
        """The speed limit is the default of urban roads, or the one set for the lanelet."""
        assert self.road.speed_limit(self.road_lanelets["right"]) == pytest.approx(50.0 / 3.6)
        assert self.two_way.speed_limit(self.two_way_lanelets["road"]) == pytest.approx(30.0 / 3.6)
