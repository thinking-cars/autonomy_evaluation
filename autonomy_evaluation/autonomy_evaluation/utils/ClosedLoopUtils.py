# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Utility helpers for closed-loop evaluations: geometry and kinematics of road users on the ground plane.

Road users are oriented rectangles on the ground plane, given by the BEV
position of their geometric center, their yaw, length and width. All helpers
are vectorized with numpy over arrays of road users or poses:

- **Footprints** (``rectangle_corners``, ``rectangles_overlap``,
  ``impact_side``): corners of oriented rectangles, their overlap by the
  separating axis theorem, and the side of a rectangle another one hits.
- **Time to collision** (``time_to_collision``): time until two road users
  moving at constant velocity overlap.
- **Path coordinates** (``path_coordinates``): arc length along and lateral
  offset from a path of constant curvature, e.g. the path ahead of the ego
  vehicle.
- **Occupancy** (``occupancy_intervals``, ``post_encroachment_time``): the
  time intervals a road user occupies the cells of a grid on the ground plane,
  and the post-encroachment time of two road users passing the same cells.
- **Time derivatives** (``time_derivative``, ``resample_poses``,
  ``wrap_angle``): finite differences and linear resampling of non-uniformly
  sampled signals, and angles wrapped to ``[-pi, pi)``.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon

# Iterations bisecting the time step in which two road users start to overlap, refining the time to collision to
# 1 / 2**10 of the time step
_TTC_BISECTION_ITERATIONS = 10

# Factor by which two consecutive occupancies of a cell may be further apart than the time step to belong to the
# same interval, absorbing the rounding of the resampled times
_OCCUPANCY_GAP_FACTOR = 1.5


class ClosedLoopUtils:
    """Static helpers shared by the closed-loop evaluations.

    Groups stateless geometry and kinematics of road users, each an oriented
    rectangle on the ground plane given by ``x``, ``y`` (BEV position of its
    geometric center), ``yaw``, ``length`` and ``width``.
    """

    @staticmethod
    def wrap_angle(angle: np.ndarray) -> np.ndarray:
        """Wrap angles to ``[-pi, pi)``.

        Args:
            angle: Angles in radians.

        Returns:
            The angles wrapped to ``[-pi, pi)``.
        """
        return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def rectangle_corners(x, y, yaw, length, width) -> np.ndarray:
        """Compute the corners of oriented rectangles.

        Args:
            x, y: BEV positions of the geometric centers, broadcastable arrays.
            yaw: Yaw angles in radians.
            length, width: Extents along and across the yaw.

        Returns:
            Corners of shape ``(..., 4, 2)``, counter-clockwise from front left.
        """
        x, y, yaw, length, width = np.broadcast_arrays(*(np.asarray(value, dtype=float) for value in (x, y, yaw, length, width)))
        cos, sin = np.cos(yaw), np.sin(yaw)
        # corners in the frame of the rectangle, counter-clockwise from front left
        local = np.array([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]])
        u = local[:, 0] * length[..., None]
        v = local[:, 1] * width[..., None]
        corners_x = x[..., None] + u * cos[..., None] - v * sin[..., None]
        corners_y = y[..., None] + u * sin[..., None] + v * cos[..., None]
        return np.stack([corners_x, corners_y], axis=-1)

    @staticmethod
    def rectangles_overlap(first: Tuple, second: Tuple) -> np.ndarray:
        """Check whether oriented rectangles overlap, by the separating axis theorem.

        Two convex polygons are disjoint if and only if their projections onto
        one of their edge normals are disjoint. Rectangles that touch overlap.

        Args:
            first: ``(x, y, yaw, length, width)`` of the first rectangles,
                broadcastable arrays.
            second: ``(x, y, yaw, length, width)`` of the second rectangles,
                broadcastable with the first.

        Returns:
            Whether each pair of rectangles overlaps.
        """
        corners_first = ClosedLoopUtils.rectangle_corners(*first)
        corners_second = ClosedLoopUtils.rectangle_corners(*second)
        corners_first, corners_second = np.broadcast_arrays(corners_first, corners_second)
        separated = np.zeros(corners_first.shape[:-2], dtype=bool)
        for yaw in (np.asarray(first[2], dtype=float), np.asarray(second[2], dtype=float)):
            for angle in (yaw, yaw + np.pi / 2.0):
                axis = np.stack(np.broadcast_arrays(np.cos(angle), np.sin(angle)), axis=-1)[..., None, :]
                projection_first = np.sum(corners_first * axis, axis=-1)
                projection_second = np.sum(corners_second * axis, axis=-1)
                separated |= (projection_first.max(axis=-1) < projection_second.min(axis=-1)) | (
                    projection_second.max(axis=-1) < projection_first.min(axis=-1)
                )
        return ~separated

    @staticmethod
    def impact_side(rectangle: Tuple, other: Tuple) -> Optional[str]:
        """Determine the side of a rectangle that an overlapping rectangle hits.

        The side is the one closest to the centroid of the overlap, measured
        relative to the extent of the rectangle, so that a hit on a corner is
        attributed to the side it reaches further into.

        Args:
            rectangle: ``(x, y, yaw, length, width)`` of the hit rectangle.
            other: ``(x, y, yaw, length, width)`` of the hitting rectangle.

        Returns:
            ``"front"``, ``"rear"``, ``"left"`` or ``"right"`` of the hit
            rectangle; ``None`` if the rectangles do not overlap by an area.
        """
        polygons = [Polygon(ClosedLoopUtils.rectangle_corners(*shape)) for shape in (rectangle, other)]
        overlap = polygons[0].intersection(polygons[1])
        if overlap.is_empty or overlap.area <= 0.0:
            return None
        x, y, yaw, length, width = rectangle
        dx, dy = overlap.centroid.x - x, overlap.centroid.y - y
        u = np.cos(yaw) * dx + np.sin(yaw) * dy
        v = -np.sin(yaw) * dx + np.cos(yaw) * dy
        if abs(u) / (length / 2.0) >= abs(v) / (width / 2.0):
            return "front" if u >= 0.0 else "rear"
        return "left" if v >= 0.0 else "right"

    @staticmethod
    def time_to_collision(
        ego: Tuple[float, ...],
        objects: Tuple[np.ndarray, ...],
        horizon: float,
        time_step: float,
    ) -> np.ndarray:
        """Compute the time until the ego vehicle and each object overlap if both keep their velocity.

        Both rectangles are moved at their constant velocity and keep their
        yaw. The time step in which they start to overlap is found by stepping
        through the horizon and refined by bisection. Objects that cannot reach
        the ego vehicle within the horizon, as their distance exceeds what
        their relative velocity covers, are skipped.

        Args:
            ego: ``(x, y, yaw, length, width, vx, vy)`` of the ego vehicle, with
                its velocity in the frame of its position.
            objects: ``(x, y, yaw, length, width, vx, vy)`` of the objects, each
                an array of the same length.
            horizon: Seconds the motion is extrapolated for.
            time_step: Seconds between the extrapolated positions.

        Returns:
            The time to collision of each object in seconds: ``0`` for an object
            that overlaps the ego vehicle already, ``nan`` for one that does not
            overlap it within the horizon.
        """
        ego_x, ego_y, ego_yaw, ego_length, ego_width, ego_vx, ego_vy = (float(value) for value in ego)
        x, y, yaw, length, width, vx, vy = (np.asarray(value, dtype=float) for value in objects)
        ttc = np.full(x.shape, np.nan)
        if x.size == 0:
            return ttc

        # relative motion of the objects as seen from the ego vehicle, which stands still in its own frame
        relative_vx, relative_vy = vx - ego_vx, vy - ego_vy
        radius = 0.5 * np.hypot(length, width) + 0.5 * np.hypot(ego_length, ego_width)
        distance = np.hypot(x - ego_x, y - ego_y)
        reachable = distance - radius <= np.hypot(relative_vx, relative_vy) * horizon
        if not reachable.any():
            return ttc

        def overlap(index: np.ndarray, times: np.ndarray) -> np.ndarray:
            """Check the overlap of the objects of the given indices at the given times (broadcast)."""
            ego_shape = (ego_x, ego_y, ego_yaw, ego_length, ego_width)
            shape = (
                x[index] + relative_vx[index] * times,
                y[index] + relative_vy[index] * times,
                yaw[index],
                length[index],
                width[index],
            )
            return ClosedLoopUtils.rectangles_overlap(ego_shape, shape)

        index = np.flatnonzero(reachable)
        times = np.arange(0.0, horizon + 0.5 * time_step, time_step)
        overlaps = overlap(index[:, None], times[None, :])
        hit = overlaps.any(axis=1)
        first = np.argmax(overlaps, axis=1)

        # an object overlapping already has a time to collision of 0; the others are refined within the step before
        # their first overlap, in which they start to overlap
        ttc[index[hit & (first == 0)]] = 0.0
        refine = hit & (first > 0)
        if refine.any():
            refined = index[refine]
            lower, upper = times[first[refine] - 1], times[first[refine]]
            for _ in range(_TTC_BISECTION_ITERATIONS):
                middle = 0.5 * (lower + upper)
                overlapping = overlap(refined, middle)
                upper = np.where(overlapping, middle, upper)
                lower = np.where(overlapping, lower, middle)
            ttc[refined] = upper
        return ttc

    @staticmethod
    def path_coordinates(x: np.ndarray, y: np.ndarray, curvature: float) -> Tuple[np.ndarray, np.ndarray]:
        """Express points in coordinates along a path of constant curvature.

        The path starts at the origin heading along the x axis, e.g. the path
        ahead of the ego vehicle in its own frame, and turns left for a positive
        curvature. A curvature of zero is a straight path along the x axis.

        Args:
            x, y: Points in the frame the path starts in.
            curvature: Curvature of the path in 1/m.

        Returns:
            ``(s, d)``: the arc length of the closest point of the path in
            meters, negative behind the origin and within half a circle of it,
            and the lateral offset from the path in meters, positive to the left.
        """
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if abs(curvature) < 1e-9:
            return x, y
        # angle the point is passed at on the circle of the path, measured from the center of the circle
        s = np.arctan2(curvature * x, 1.0 - curvature * y) / curvature
        d = (1.0 - np.hypot(curvature * x, 1.0 - curvature * y)) / curvature
        return s, d

    @staticmethod
    def polyline_coordinates(points: np.ndarray, x: float, y: float) -> Tuple[float, float, float]:
        """Express a point in coordinates along a polyline, e.g. the centerline of a lane or a route.

        Args:
            points: Points of the polyline as array of shape ``(n, 2)``, ``n >= 2``.
            x, y: The point.

        Returns:
            ``(s, d, heading)``: the arc length of the closest point of the
            polyline, the lateral offset from it, positive to the left, and the
            heading of the polyline there.
        """
        start, segment = points[:-1], np.diff(points, axis=0)
        lengths = np.hypot(segment[:, 0], segment[:, 1])
        offset = np.array([x, y]) - start
        fraction = np.clip(np.sum(offset * segment, axis=1) / np.maximum(lengths**2, 1e-12), 0.0, 1.0)
        closest = start + fraction[:, None] * segment
        index = int(np.argmin(np.hypot(closest[:, 0] - x, closest[:, 1] - y)))
        heading = math.atan2(segment[index, 1], segment[index, 0])
        cross = segment[index, 0] * (y - start[index, 1]) - segment[index, 1] * (x - start[index, 0])
        distance = math.hypot(closest[index, 0] - x, closest[index, 1] - y)
        s = float(np.sum(lengths[:index]) + fraction[index] * lengths[index])
        return s, math.copysign(distance, cross), heading

    @staticmethod
    def time_derivative(times: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Differentiate a non-uniformly sampled signal by finite differences.

        Interior samples use second-order central differences, the first and
        last sample one-sided differences, as ``numpy.gradient`` computes them.

        Args:
            times: Strictly increasing sample times in seconds.
            values: Signal values at the sample times.

        Returns:
            The time derivative at each sample; ``nan`` for a signal of fewer
            than two samples.
        """
        times, values = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
        if times.size < 2:
            return np.full(times.shape, np.nan)
        return np.gradient(values, times)

    @staticmethod
    def resample_poses(
        times: np.ndarray, x: np.ndarray, y: np.ndarray, yaw: np.ndarray, time_step: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Resample a trajectory at a fixed time step by linear interpolation.

        Positions are interpolated linearly, yaw angles along the shorter way.

        Args:
            times: Strictly increasing sample times of the trajectory in seconds.
            x, y: BEV positions at the sample times.
            yaw: Yaw angles at the sample times in radians.
            time_step: Seconds between the resampled poses.

        Returns:
            ``(times, x, y, yaw)`` of the resampled poses, from the first sample
            time up to the last one.
        """
        times = np.asarray(times, dtype=float)
        resampled = np.arange(times[0], times[-1] + 0.5 * time_step, time_step) if times.size > 1 else times
        resampled = np.minimum(resampled, times[-1])
        unwrapped = np.unwrap(np.asarray(yaw, dtype=float))
        return (
            resampled,
            np.interp(resampled, times, x),
            np.interp(resampled, times, y),
            ClosedLoopUtils.wrap_angle(np.interp(resampled, times, unwrapped)),
        )

    @staticmethod
    def occupancy_intervals(
        times: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        yaw: np.ndarray,
        length: float,
        width: float,
        cell_size: float,
        time_step: float,
    ) -> Dict[str, np.ndarray]:
        """Compute the time intervals a road user occupies the cells of a grid on the ground plane.

        The trajectory is resampled at ``time_step``, so that a fast road user
        sampled at a low rate does not jump over cells. A cell is occupied while
        its center lies within the footprint of the road user. Consecutive
        occupancies of a cell form one interval, which is extended by half a time
        step on either side: the road user actually enters a cell some time
        between the last resampled pose outside and the first one inside, so
        that the extended interval estimates its entry and exit without bias, to
        within half a time step.

        Args:
            times: Strictly increasing sample times of the trajectory in seconds.
            x, y: BEV positions of the geometric center at the sample times.
            yaw: Yaw angles at the sample times in radians.
            length, width: Extent of the road user.
            cell_size: Edge length of the square grid cells in meters.
            time_step: Seconds between the resampled poses.

        Returns:
            ``{"cell", "start", "end", "yaw"}``, arrays of one entry per interval:
            the cell as ``(column, row)`` index pair of shape ``(n, 2)``, the
            estimated times the road user enters and leaves the cell, and its
            yaw at the first pose inside the cell.
        """
        times, x, y, yaw = ClosedLoopUtils.resample_poses(times, x, y, yaw, time_step)
        # cells whose center may lie within the footprint, relative to the cell of the center of the road user
        reach = int(np.ceil(0.5 * np.hypot(length, width) / cell_size)) + 1
        offsets = np.arange(-reach, reach + 1)
        offset_columns, offset_rows = (offset.ravel() for offset in np.meshgrid(offsets, offsets, indexing="ij"))
        columns = np.floor(x / cell_size).astype(np.int64)[:, None] + offset_columns[None, :]
        rows = np.floor(y / cell_size).astype(np.int64)[:, None] + offset_rows[None, :]
        dx = (columns + 0.5) * cell_size - x[:, None]
        dy = (rows + 0.5) * cell_size - y[:, None]
        cos, sin = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        inside = (np.abs(cos * dx + sin * dy) <= 0.5 * length) & (np.abs(-sin * dx + cos * dy) <= 0.5 * width)

        pose, candidate = np.nonzero(inside)
        if pose.size == 0:
            return {"cell": np.empty((0, 2), dtype=np.int64), "start": np.empty(0), "end": np.empty(0), "yaw": np.empty(0)}
        cell_columns, cell_rows = columns[pose, candidate], rows[pose, candidate]
        order = np.lexsort((pose, cell_rows, cell_columns))
        pose, cell_columns, cell_rows = pose[order], cell_columns[order], cell_rows[order]

        # an interval starts at a new cell or after a gap in the occupancy of the same cell
        same_cell = (cell_columns[1:] == cell_columns[:-1]) & (cell_rows[1:] == cell_rows[:-1])
        consecutive = times[pose[1:]] - times[pose[:-1]] <= _OCCUPANCY_GAP_FACTOR * time_step
        starts = np.flatnonzero(np.concatenate([[True], ~(same_cell & consecutive)]))
        ends = np.concatenate([starts[1:], [pose.size]]) - 1
        return {
            "cell": np.stack([cell_columns[starts], cell_rows[starts]], axis=1),
            "start": times[pose[starts]] - 0.5 * time_step,
            "end": times[pose[ends]] + 0.5 * time_step,
            "yaw": yaw[pose[starts]],
        }

    @staticmethod
    def post_encroachment_time(first: Dict[str, np.ndarray], second: Dict[str, np.ndarray]) -> Optional[Dict[str, float]]:
        """Compute the post-encroachment time of two road users from the cells they occupy.

        The post-encroachment time (PET) is the time between the first road
        user leaving a spot and the second one entering it. It is computed for
        every cell both occupy and minimized over the cells; a cell both occupy
        at the same time, as far as the time step of the occupancy resolves it,
        has a PET of zero.

        Args:
            first: Occupancy intervals of the first road user, see
                :meth:`occupancy_intervals`.
            second: Occupancy intervals of the second road user.

        Returns:
            ``{"pet", "first_enters", "time", "angle"}`` of the cell of the
            smallest PET: the PET in seconds, whether the first road user
            entered the cell first, the time the second road user entered it,
            and the angle between the headings of both road users at the cell in
            radians in ``[0, pi]``; ``None`` if they share no cell.
        """
        if first["cell"].size == 0 or second["cell"].size == 0:
            return None
        first_keys = _cell_keys(first["cell"])
        second_keys = _cell_keys(second["cell"])
        if not np.isin(first_keys, second_keys).any():
            return None

        # pair every interval of the first road user with every interval of the second one in the same cell
        second_order = np.argsort(second_keys, kind="stable")
        sorted_second_keys = second_keys[second_order]
        lower = np.searchsorted(sorted_second_keys, first_keys, side="left")
        counts = np.searchsorted(sorted_second_keys, first_keys, side="right") - lower
        pairs_first = np.repeat(np.arange(first_keys.size), counts)
        offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        pairs_second = second_order[np.repeat(lower, counts) + offsets]

        first_start, first_end = first["start"][pairs_first], first["end"][pairs_first]
        second_start, second_end = second["start"][pairs_second], second["end"][pairs_second]
        pet = np.where(
            first_end < second_start,
            second_start - first_end,
            np.where(second_end < first_start, first_start - second_end, 0.0),
        )
        best = int(np.argmin(pet))
        angle = abs(float(ClosedLoopUtils.wrap_angle(first["yaw"][pairs_first[best]] - second["yaw"][pairs_second[best]])))
        return {
            "pet": float(pet[best]),
            "first_enters": bool(first_start[best] <= second_start[best]),
            "time": float(max(first_start[best], second_start[best])),
            "angle": angle,
        }


def _cell_keys(cells: np.ndarray) -> np.ndarray:
    """Combine the ``(column, row)`` indices of grid cells into one integer key per cell, ordered like the pairs."""
    return cells[:, 0].astype(np.int64) * (1 << 32) + (cells[:, 1].astype(np.int64) + (1 << 31))
