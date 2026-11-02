# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Lanelet2 map of the current scene, which metrics query for the lanes road users drive in.

The map is distributed like ``lanelet2_map_server`` distributes it, and like
``autonomy_datasets`` publishes the map of each scene: as the ROS parameters
``map_contents`` (the map as OSM XML), ``origin_lat`` and ``origin_lon`` (the
origin of its UTM projection) and ``map_frame_id`` (the frame of the projected
map) of a map server node. :meth:`Lanelet2Map.from_osm` loads such a map, which
the node passes to the metrics with every sample.

Queries treat road users as points or rectangles on the ground plane in the
frame of the map:

- ``lanelets_at``, ``lane_of``: the lanelets containing a point, and the one a
  road user drives along.
- ``lane_coordinates``: arc length along and lateral offset from the centerline
  of a lanelet, and its heading there.
- ``paths_ahead``: the sequences of lanelets a road user may follow.
- ``connected``: whether two lanelets are the same lane, i.e. one follows the
  other.

The routing graph and the traffic rules are those of vehicles in Germany, the
default of Lanelet2. ``lanelet2`` is only imported once a map is loaded, so that
evaluations without map do not depend on it.
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Maximum number of paths ahead that are followed through branching lanes
_MAX_PATHS = 32


class Lanelet2Map:
    """Lanelet2 map with the routing graph and traffic rules of vehicles.

    Attributes:
        lanelet_map: The ``lanelet2.core.LaneletMap``, in the frame of the map.
        frame_id: Frame of the map.
        traffic_rules: Traffic rules of vehicles.
        routing_graph: Routing graph of vehicles under the traffic rules.
    """

    def __init__(self, lanelet_map: Any, frame_id: str = "map") -> None:
        """Build the routing graph of a loaded map under the traffic rules of vehicles."""
        import lanelet2

        self.lanelet_map = lanelet_map
        self.frame_id = frame_id
        self.traffic_rules = lanelet2.traffic_rules.create(
            lanelet2.traffic_rules.Locations.Germany, lanelet2.traffic_rules.Participants.Vehicle
        )
        self.routing_graph = lanelet2.routing.RoutingGraph(lanelet_map, self.traffic_rules)
        # centerlines of the lanelets by ID and orientation, as arrays of their points, computed on first use
        self._centerlines: Dict[Tuple[int, bool], np.ndarray] = {}

    @classmethod
    def from_osm(cls, contents: str, origin_lat: float, origin_lon: float, frame_id: str = "map") -> Optional["Lanelet2Map"]:
        """Load a map from its OSM XML contents, projected around its origin.

        Args:
            contents: The map as OSM XML, as the ``map_contents`` parameter holds it.
            origin_lat, origin_lon: WGS84 origin of the UTM projection of the map.
            frame_id: Frame of the projected map.

        Returns:
            The map, ``None`` for a map without lanelets, e.g. the blank map a
            dataset without map publishes.

        Raises:
            ValueError: The contents cannot be loaded as Lanelet2 map.
        """
        import lanelet2
        from lanelet2.io import Origin
        from lanelet2.projection import UtmProjector

        if not contents.strip():
            return None
        handle, path = tempfile.mkstemp(suffix=".osm")
        try:
            with os.fdopen(handle, "w") as osm_file:
                osm_file.write(contents)
            lanelet_map, errors = lanelet2.io.loadRobust(path, UtmProjector(Origin(origin_lat, origin_lon)))
        except RuntimeError as exception:
            raise ValueError(f"Cannot load the Lanelet2 map: {exception}") from exception
        finally:
            os.remove(path)
        if len(lanelet_map.laneletLayer) == 0:
            return None
        return cls(lanelet_map, frame_id)

    @property
    def num_lanelets(self) -> int:
        """Number of lanelets of the map."""
        return len(self.lanelet_map.laneletLayer)

    def lanelets_at(self, x: float, y: float) -> List[Any]:
        """Return the lanelets containing a point.

        Args:
            x, y: Point in the frame of the map.

        Returns:
            The lanelets whose area contains the point, overlapping lanelets e.g.
            of an intersection all of them.
        """
        from lanelet2.core import BasicPoint2d
        from lanelet2.geometry import findWithin2d

        return [
            lanelet
            for distance, lanelet in findWithin2d(self.lanelet_map.laneletLayer, BasicPoint2d(x, y), 0.0)
            if distance == 0.0
        ]

    def lane_of(self, x: float, y: float, yaw: float) -> Optional[Any]:
        """Return the lanelet a road user at a point drives along.

        Args:
            x, y: Position of the road user in the frame of the map.
            yaw: Heading of the road user.

        Returns:
            Of the lanelets containing the position, the one whose direction is
            closest to the heading, if it deviates by less than 90°; ``None`` if
            no lanelet contains the position or the road user drives against all.
        """
        best, best_deviation = None, math.pi / 2.0
        for lanelet in self.lanelets_at(x, y):
            _, _, heading = self.lane_coordinates(lanelet, x, y)
            deviation = abs(math.remainder(yaw - heading, 2.0 * math.pi))
            if deviation < best_deviation:
                best, best_deviation = lanelet, deviation
        return best

    def lane_coordinates(self, lanelet: Any, x: float, y: float) -> Tuple[float, float, float]:
        """Express a point in coordinates along the centerline of a lanelet.

        Args:
            lanelet: The lanelet.
            x, y: Point in the frame of the map.

        Returns:
            ``(s, d, heading)``: the arc length of the closest point of the
            centerline, the lateral offset from it, positive to the left, and
            the heading of the centerline there.
        """
        points = self._centerline(lanelet)
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

    def length(self, lanelet: Any) -> float:
        """Return the length of the centerline of a lanelet."""
        points = self._centerline(lanelet)
        return float(np.sum(np.hypot(*np.diff(points, axis=0).T)))

    def connected(self, first: Any, second: Any) -> bool:
        """Decide whether two lanelets belong to the same lane, i.e. are the same or one directly follows the other."""
        if first.id == second.id:
            return True
        return any(lanelet.id == second.id for lanelet in self.routing_graph.following(first)) or any(
            lanelet.id == first.id for lanelet in self.routing_graph.following(second)
        )

    def paths_ahead(self, lanelet: Any, distance: float) -> List[List[Any]]:
        """List the sequences of lanelets a vehicle in a lanelet may follow up to a distance.

        Args:
            lanelet: Lanelet the paths start in.
            distance: Length the paths cover at least, unless a lane ends before.

        Returns:
            The paths, each starting with the given lanelet; at most a fixed
            number of paths through branching lanes.
        """
        paths: List[List[Any]] = []
        pending = [([lanelet], self.length(lanelet))]
        while pending and len(paths) < _MAX_PATHS:
            path, covered = pending.pop()
            following = self.routing_graph.following(path[-1]) if covered < distance else []
            following = [successor for successor in following if all(successor.id != visited.id for visited in path)]
            if not following:
                paths.append(path)
            for successor in following:
                pending.append((path + [successor], covered + self.length(successor)))
        return paths

    def _centerline(self, lanelet: Any) -> np.ndarray:
        """Return the points of the centerline of a lanelet as array of shape ``(n, 2)``.

        Cached by ID and orientation, as a lanelet that may be driven in both
        directions is also used inverted.
        """
        key = (lanelet.id, lanelet.inverted())
        if key not in self._centerlines:
            self._centerlines[key] = np.array([(point.x, point.y) for point in lanelet.centerline])
        return self._centerlines[key]
