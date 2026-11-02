# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Builders of small Lanelet2 maps with known geometry, e.g. a straight two-lane road.

Lanes are 3.5 m wide one-way roads, separated by dashed lines that allow changing lanes, as a dataset
or a simulation would provide them as Lanelet2 maps.
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import Dict, List, Sequence, Tuple

import lanelet2
from autonomy_evaluation.utils.Lanelet2Map import Lanelet2Map
from lanelet2.core import getId, Lanelet, LaneletMap, LineString3d, Point3d
from lanelet2.io import Origin
from lanelet2.projection import UtmProjector

LANE_WIDTH = 3.5

# Origin of the projection the maps are written with, in Aachen
ORIGIN = (50.78, 6.06)


def _line(points: Sequence[Point3d]) -> LineString3d:
    """Build a dashed lane boundary through the given points."""
    line = LineString3d(getId(), list(points))
    line.attributes["type"] = "line_thin"
    line.attributes["subtype"] = "dashed"
    return line


def _points(coordinates: Sequence[Tuple[float, float]]) -> List[Point3d]:
    """Build points of the given coordinates."""
    return [Point3d(getId(), x, y, 0.0) for x, y in coordinates]


def _lanelet(left: LineString3d, right: LineString3d) -> Lanelet:
    """Build a one-way urban road lanelet between the given boundaries, driven with the left boundary on its left."""
    lanelet = Lanelet(getId(), left, right)
    for key, value in {"type": "lanelet", "subtype": "road", "location": "urban", "one_way": "yes"}.items():
        lanelet.attributes[key] = value
    return lanelet


def two_lane_road() -> Tuple[Lanelet2Map, Dict[str, Lanelet]]:
    """Build a straight road along the x axis with two lanes in the direction of the x axis.

    The right lane is centered on the x axis and consists of the lanelets ``right`` (0 to 100 m) and
    ``right_next`` (100 to 200 m), the left lane ``left`` is centered 3.5 m to the left and spans 0 to 200 m.

    Returns:
        The map and its lanelets by name.
    """
    half = LANE_WIDTH / 2.0
    right_boundary = _points([(0.0, -half), (100.0, -half), (200.0, -half)])
    middle_boundary = _points([(0.0, half), (100.0, half), (200.0, half)])
    lanelets = {
        "right": _lanelet(_line(middle_boundary[:2]), _line(right_boundary[:2])),
        "right_next": _lanelet(_line(middle_boundary[1:]), _line(right_boundary[1:])),
        "left": _lanelet(_line(_points([(0.0, 3.0 * half), (200.0, 3.0 * half)])), _line(middle_boundary)),
    }
    return _map(lanelets), lanelets


def junction() -> Tuple[Lanelet2Map, Dict[str, Lanelet]]:
    """Build a lane along the x axis up to 100 m that continues straight and branches off to the left at 45°.

    The branch ``turn`` runs from (100, 0) to (150, 50) along its centerline.

    Returns:
        The map and its lanelets ``approach``, ``straight`` and ``turn`` by name.
    """
    half = LANE_WIDTH / 2.0
    offset = half / math.sqrt(2.0)
    left, right = _points([(0.0, half), (100.0, half)]), _points([(0.0, -half), (100.0, -half)])
    lanelets = {
        "approach": _lanelet(_line(left), _line(right)),
        "straight": _lanelet(_line([left[1], *_points([(200.0, half)])]), _line([right[1], *_points([(200.0, -half)])])),
        "turn": _lanelet(
            _line([left[1], *_points([(150.0 - offset, 50.0 + offset)])]),
            _line([right[1], *_points([(150.0 + offset, 50.0 - offset)])]),
        ),
    }
    return _map(lanelets), lanelets


def bend(radius: float = 100.0, angle: float = 1.0, num_points: int = 50) -> Tuple[Lanelet2Map, Dict[str, Lanelet]]:
    """Build a lane turning left on a circle of the given radius around (0, radius), starting at the origin along the x axis.

    Returns:
        The map and its lanelet ``bend`` by name.
    """

    def arc(arc_radius: float) -> List[Point3d]:
        angles = [angle * index / (num_points - 1) for index in range(num_points)]
        return _points([(arc_radius * math.sin(a), radius - arc_radius * math.cos(a)) for a in angles])

    lanelets = {"bend": _lanelet(_line(arc(radius - LANE_WIDTH / 2.0)), _line(arc(radius + LANE_WIDTH / 2.0)))}
    return _map(lanelets), lanelets


def osm_contents(lanelet2_map: Lanelet2Map) -> str:
    """Write a map as OSM XML, projected around :data:`ORIGIN`, as a map server holds it in its parameters."""
    handle, path = tempfile.mkstemp(suffix=".osm")
    os.close(handle)
    try:
        lanelet2.io.write(path, lanelet2_map.lanelet_map, UtmProjector(Origin(*ORIGIN)))
        with open(path) as osm_file:
            return osm_file.read()
    finally:
        os.remove(path)


def _map(lanelets: Dict[str, Lanelet]) -> Lanelet2Map:
    """Collect lanelets in a map."""
    lanelet_map = LaneletMap()
    for lanelet in lanelets.values():
        lanelet_map.add(lanelet)
    return Lanelet2Map(lanelet_map)
