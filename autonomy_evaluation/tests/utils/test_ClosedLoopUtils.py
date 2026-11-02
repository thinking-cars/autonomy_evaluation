# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for the geometry and kinematics helpers of the closed-loop evaluations.

The expected values are derived by hand from scenarios simple enough to solve analytically, e.g.
two vehicles driving towards each other at constant speed, or two vehicles crossing an intersection.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from autonomy_evaluation.utils.ClosedLoopUtils import ClosedLoopUtils

# A car of 4 m length and 2 m width
_LENGTH, _WIDTH = 4.0, 2.0


def _objects(*objects) -> tuple:
    """Stack ``(x, y, yaw, length, width, vx, vy)`` tuples of objects into one array per field."""
    return tuple(np.array(field, dtype=float) for field in zip(*objects))


class TestWrapAngle:
    """Tests wrapping angles to [-pi, pi)."""

    @pytest.mark.parametrize("angle, wrapped", [(0.0, 0.0), (1.5 * math.pi, -0.5 * math.pi), (math.pi, -math.pi), (-3.0, -3.0)])
    def test_wraps_angles(self, angle, wrapped):
        """Angles outside of [-pi, pi) are shifted by full turns."""
        assert ClosedLoopUtils.wrap_angle(angle) == pytest.approx(wrapped)


class TestFootprints:
    """Tests the corners and the overlap of oriented rectangles."""

    def test_corners_of_an_axis_aligned_rectangle(self):
        """The corners run counter-clockwise from front left."""
        corners = ClosedLoopUtils.rectangle_corners(0.0, 0.0, 0.0, _LENGTH, _WIDTH)

        assert corners.tolist() == [[2.0, 1.0], [-2.0, 1.0], [-2.0, -1.0], [2.0, -1.0]]

    def test_corners_follow_the_yaw(self):
        """A rectangle turned to the left has its front left corner at its rear left before turning."""
        corners = ClosedLoopUtils.rectangle_corners(1.0, 0.0, math.pi / 2.0, _LENGTH, _WIDTH)

        assert corners[0] == pytest.approx([0.0, 2.0])

    @pytest.mark.parametrize(
        "other, overlap",
        [
            ((3.0, 0.0, 0.0, _LENGTH, _WIDTH), True),  # rear of a car ahead overlaps the front
            ((4.1, 0.0, 0.0, _LENGTH, _WIDTH), False),  # 10 cm gap
            ((4.0, 0.0, 0.0, _LENGTH, _WIDTH), True),  # touching
            ((0.0, 2.5, math.pi / 2.0, _LENGTH, _WIDTH), True),  # crossing car reaching into the side
            ((3.3, 1.8, math.pi / 4.0, 1.0, 1.0), False),  # diamond next to a corner, inside the bounding box
        ],
    )
    def test_overlap_by_the_separating_axis_theorem(self, other, overlap):
        """Rectangles overlap unless an edge normal of one of them separates their projections."""
        ego = (0.0, 0.0, 0.0, _LENGTH, _WIDTH)

        assert bool(ClosedLoopUtils.rectangles_overlap(ego, other)) is overlap

    def test_overlap_is_vectorized(self):
        """The overlap of one rectangle with many is checked at once."""
        ego = (0.0, 0.0, 0.0, _LENGTH, _WIDTH)
        others = (np.array([3.0, 10.0, -3.5]), np.zeros(3), np.zeros(3), np.full(3, _LENGTH), np.full(3, _WIDTH))

        assert ClosedLoopUtils.rectangles_overlap(ego, others).tolist() == [True, False, True]

    @pytest.mark.parametrize(
        "other, side",
        [
            ((3.5, 0.0, 0.0, _LENGTH, _WIDTH), "front"),
            ((-3.5, 0.3, 0.0, _LENGTH, _WIDTH), "rear"),
            ((0.5, 1.8, 0.0, _LENGTH, _WIDTH), "left"),
            ((-0.5, -2.5, math.pi / 2.0, _LENGTH, _WIDTH), "right"),
            ((10.0, 0.0, 0.0, _LENGTH, _WIDTH), None),
        ],
    )
    def test_impact_side_is_the_side_closest_to_the_overlap(self, other, side):
        """The hit side of the ego vehicle is found from the centroid of the overlap."""
        assert ClosedLoopUtils.impact_side((0.0, 0.0, 0.0, _LENGTH, _WIDTH), other) == side


class TestTimeToCollision:
    """Tests the time until two road users moving at constant velocity overlap."""

    _EGO = (0.0, 0.0, 0.0, _LENGTH, _WIDTH, 10.0, 0.0)

    def test_head_on(self):
        """Cars 26 m apart driving towards each other at 10 m/s each collide after 1.3 s."""
        ttc = ClosedLoopUtils.time_to_collision(self._EGO, _objects((30.0, 0.0, math.pi, _LENGTH, _WIDTH, -10.0, 0.0)), 5.0, 0.1)

        assert ttc[0] == pytest.approx(1.3, abs=1e-3)

    def test_crossing(self):
        """A car crossing in front of a standing car reaches it once its front passes the side of the other one."""
        standing = (0.0, 0.0, 0.0, _LENGTH, _WIDTH, 0.0, 0.0)
        crossing = (0.0, 20.0, -math.pi / 2.0, _LENGTH, _WIDTH, 0.0, -10.0)

        # the center of the crossing car has to come as close as 1 m (half width) + 2 m (half length), i.e. travel 17 m
        assert ClosedLoopUtils.time_to_collision(standing, _objects(crossing), 5.0, 0.1)[0] == pytest.approx(1.7, abs=1e-3)

    def test_following_beyond_the_horizon(self):
        """A slower car 46 m ahead is reached after 4.6 s, beyond a horizon of 3 s but within one of 5 s."""
        lead = _objects((50.0, 0.0, 0.0, _LENGTH, _WIDTH, 0.0, 0.0))
        ego = (0.0, 0.0, 0.0, _LENGTH, _WIDTH, 10.0, 0.0)

        assert math.isnan(ClosedLoopUtils.time_to_collision(ego, lead, 3.0, 0.1)[0])
        assert ClosedLoopUtils.time_to_collision(ego, lead, 5.0, 0.1)[0] == pytest.approx(4.6, abs=1e-3)

    def test_diverging_objects_never_collide(self):
        """A car driving away faster than the ego vehicle is never reached."""
        ttc = ClosedLoopUtils.time_to_collision(self._EGO, _objects((10.0, 0.0, 0.0, _LENGTH, _WIDTH, 15.0, 0.0)), 10.0, 0.1)

        assert math.isnan(ttc[0])

    def test_overlapping_objects_collided_already(self):
        """An object overlapping the ego vehicle has a time to collision of zero."""
        ttc = ClosedLoopUtils.time_to_collision(self._EGO, _objects((1.0, 0.5, 0.0, _LENGTH, _WIDTH, 0.0, 0.0)), 5.0, 0.1)

        assert ttc[0] == 0.0

    def test_passing_objects_never_collide(self):
        """Oncoming traffic in the neighboring lane passes without overlapping."""
        oncoming = _objects((30.0, 3.5, math.pi, _LENGTH, _WIDTH, -10.0, 0.0))

        assert math.isnan(ClosedLoopUtils.time_to_collision(self._EGO, oncoming, 5.0, 0.1)[0])

    def test_one_time_to_collision_per_object(self):
        """The objects are evaluated at once, unreachable ones skipped."""
        objects = _objects(
            (30.0, 0.0, math.pi, _LENGTH, _WIDTH, -10.0, 0.0),
            (1000.0, 0.0, 0.0, _LENGTH, _WIDTH, 0.0, 0.0),
            (1.0, 0.0, 0.0, _LENGTH, _WIDTH, 0.0, 0.0),
        )

        ttc = ClosedLoopUtils.time_to_collision(self._EGO, objects, 5.0, 0.1)

        assert ttc[0] == pytest.approx(1.3, abs=1e-3)
        assert math.isnan(ttc[1])
        assert ttc[2] == 0.0

    def test_no_objects(self):
        """Without objects, there is no time to collision."""
        empty = tuple(np.empty(0) for _ in range(7))

        assert ClosedLoopUtils.time_to_collision(self._EGO, empty, 5.0, 0.1).size == 0


class TestPathCoordinates:
    """Tests expressing points along a path of constant curvature."""

    def test_straight_path_keeps_the_coordinates(self):
        """Along a straight path, the arc length is x and the lateral offset y."""
        s, d = ClosedLoopUtils.path_coordinates(np.array([5.0, -2.0]), np.array([1.0, 3.0]), 0.0)

        assert s.tolist() == [5.0, -2.0]
        assert d.tolist() == [1.0, 3.0]

    @pytest.mark.parametrize(
        "x, y, curvature, s, d",
        [
            (10.0, 10.0, 0.1, 5.0 * math.pi, 0.0),  # a quarter of a left turn of 10 m radius
            (0.0, 20.0, 0.1, 10.0 * math.pi, 0.0),  # half of it
            (0.0, 5.0, 0.1, 0.0, 5.0),  # inside the turn, left of its start
            (10.0, -10.0, -0.1, 5.0 * math.pi, 0.0),  # a quarter of a right turn
            (10.0, 0.0, 0.1, 10.0 * math.pi / 4.0, 10.0 * (1.0 - math.sqrt(2.0))),  # outside of the left turn
        ],
    )
    def test_curved_path(self, x, y, curvature, s, d):
        """Along a turn, the arc length follows the circle and the lateral offset is measured from it."""
        path_s, path_d = ClosedLoopUtils.path_coordinates(np.array([x]), np.array([y]), curvature)

        assert path_s[0] == pytest.approx(s)
        assert path_d[0] == pytest.approx(d, abs=1e-12)


class TestTimeDerivative:
    """Tests differentiating non-uniformly sampled signals."""

    def test_linear_signal_has_a_constant_derivative(self):
        """The derivative of a linear signal is its slope, at irregular samples too."""
        times = np.array([0.0, 0.1, 0.3, 0.4, 1.0])

        assert ClosedLoopUtils.time_derivative(times, 2.0 * times + 1.0) == pytest.approx(np.full(5, 2.0))

    def test_quadratic_signal_is_differentiated_exactly_inside(self):
        """Central differences are exact for a quadratic signal at uniform samples."""
        times = np.linspace(0.0, 1.0, 11)

        assert ClosedLoopUtils.time_derivative(times, times**2)[1:-1] == pytest.approx(2.0 * times[1:-1])

    def test_single_sample_has_no_derivative(self):
        """A single sample has no slope."""
        assert np.isnan(ClosedLoopUtils.time_derivative(np.array([0.0]), np.array([1.0]))).all()


class TestResamplePoses:
    """Tests resampling trajectories at a fixed time step."""

    def test_interpolates_positions_linearly(self):
        """Positions between samples lie on the straight line between them."""
        times, x, y, _ = ClosedLoopUtils.resample_poses(
            np.array([0.0, 1.0]), np.array([0.0, 10.0]), np.zeros(2), np.zeros(2), 0.25
        )

        assert times.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
        assert x.tolist() == [0.0, 2.5, 5.0, 7.5, 10.0]

    def test_interpolates_yaw_along_the_shorter_way(self):
        """A turn from 3 rad to -3 rad passes pi, not zero."""
        *_, yaw = ClosedLoopUtils.resample_poses(np.array([0.0, 1.0]), np.zeros(2), np.zeros(2), np.array([3.0, -3.0]), 0.5)

        assert abs(yaw[1]) == pytest.approx(math.pi)


class TestOccupancy:
    """Tests the cells a road user occupies and the post-encroachment time of two road users."""

    def test_standing_road_user_occupies_the_cells_under_its_footprint(self):
        """A 1 m square centered at the origin covers the four cells of 0.5 m around it."""
        occupancy = ClosedLoopUtils.occupancy_intervals(
            np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]), 1.0, 1.0, 0.5, 0.1
        )

        assert sorted(map(tuple, occupancy["cell"].tolist())) == [(-1, -1), (-1, 0), (0, -1), (0, 0)]
        # a single pose estimates the occupancy to within half a time step around it
        assert occupancy["start"] == pytest.approx([-0.05] * 4)
        assert occupancy["end"] == pytest.approx([0.05] * 4)

    def test_moving_road_user_occupies_a_cell_while_passing_it(self):
        """A 1 m long road user at 1 m/s covers the cell centered at x = 5.25 m while its center is within 0.5 m of it.

        It enters at 4.75 s and leaves at 5.75 s, which the occupancy, resampled every 0.3 s, estimates to within 0.15 s.
        """
        occupancy = ClosedLoopUtils.occupancy_intervals(
            np.array([0.0, 10.0]), np.array([0.0, 10.0]), np.zeros(2), np.zeros(2), 1.0, 0.6, 0.5, 0.3
        )

        index = next(i for i, cell in enumerate(occupancy["cell"].tolist()) if cell == [10, 0])
        assert occupancy["start"][index] == pytest.approx(4.75, abs=0.15)
        assert occupancy["end"][index] == pytest.approx(5.75, abs=0.15)

    def test_passing_a_cell_twice_yields_two_intervals(self):
        """A road user returning to a cell occupies it in two separate intervals."""
        occupancy = ClosedLoopUtils.occupancy_intervals(
            np.array([0.0, 5.0, 10.0]), np.array([0.0, 5.0, 0.0]), np.zeros(3), np.zeros(3), 1.0, 0.6, 0.5, 0.25
        )

        cells = [tuple(cell) for cell in occupancy["cell"].tolist()]
        assert cells.count((4, 0)) == 2

    @staticmethod
    def _crossing(start_time: float, along_x: bool) -> dict:
        """Occupancy of a car crossing the origin at 10 m/s from 20 m before to 20 m behind it, starting at the given time."""
        times = np.array([start_time, start_time + 4.0])
        travel = np.array([-20.0, 20.0])
        x, y = (travel, np.zeros(2)) if along_x else (np.zeros(2), travel)
        yaw = np.zeros(2) if along_x else np.full(2, math.pi / 2.0)
        return ClosedLoopUtils.occupancy_intervals(times, x, y, yaw, _LENGTH, _WIDTH, 0.5, 0.05)

    def test_post_encroachment_time_of_an_intersection(self):
        """A car crossing behind the ego vehicle passes the spot the ego vehicle left last after 2.45 s.

        The ego vehicle passes x = 0.75 m with its rear at 2.275 s, the crossing car reaches y = -0.75 m with its front
        at 4.725 s, which is the closest pair of leaving and entering times among the shared cells.
        """
        ego, crossing = self._crossing(0.0, along_x=True), self._crossing(3.0, along_x=False)

        pet = ClosedLoopUtils.post_encroachment_time(ego, crossing)

        assert pet["pet"] == pytest.approx(2.45, abs=0.1)
        assert pet["first_enters"] is True
        assert pet["angle"] == pytest.approx(math.pi / 2.0)

    def test_post_encroachment_time_is_symmetric(self):
        """Swapping the road users keeps the time and swaps who entered first."""
        ego, crossing = self._crossing(0.0, along_x=True), self._crossing(3.0, along_x=False)

        forward, backward = ClosedLoopUtils.post_encroachment_time(ego, crossing), ClosedLoopUtils.post_encroachment_time(
            crossing, ego
        )

        assert backward["pet"] == pytest.approx(forward["pet"])
        assert backward["first_enters"] is False

    def test_simultaneous_occupancy_has_no_post_encroachment_time(self):
        """Road users occupying a cell at the same time collide, which is a PET of zero."""
        ego, crossing = self._crossing(0.0, along_x=True), self._crossing(0.0, along_x=False)

        assert ClosedLoopUtils.post_encroachment_time(ego, crossing)["pet"] == 0.0

    def test_road_users_sharing_no_cell_have_no_post_encroachment_time(self):
        """Parallel lanes are never encroached."""
        ego = self._crossing(0.0, along_x=True)
        parallel = ClosedLoopUtils.occupancy_intervals(
            np.array([0.0, 4.0]), np.array([-20.0, 20.0]), np.full(2, 3.5), np.zeros(2), _LENGTH, _WIDTH, 0.5, 0.05
        )

        assert ClosedLoopUtils.post_encroachment_time(ego, parallel) is None
