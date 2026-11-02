# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Criticality metrics of a closed-loop evaluation: how critically the ego vehicle interacts with other road users.

All metrics read the evaluated ``ego_data`` of the ego vehicle and the
``objects`` around it as context, each an oriented rectangle on the ground
plane (see :mod:`~autonomy_evaluation.evaluations.closed_loop.states`). All
road users are considered, whatever their class:

- ``collision``: collisions of the ego vehicle, classified by the collision
  types of nuPlan to decide whether the ego vehicle is at fault.
- ``ttc``: time to collision, assuming that the ego vehicle and every road user
  keep their velocity, with the time exposed to and the time integrated over
  small times to collision.
- ``thw``: time headway to the road user the ego vehicle follows, found in a
  corridor along the path ahead of the ego vehicle.
- ``pet``: post-encroachment time, i.e. the time between the ego vehicle and
  another road user passing the same spot, classified by the angle of their
  paths like the conflict types of the Surrogate Safety Assessment Model (SSAM).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from autonomy_evaluation.evaluations.closed_loop.states import ego_state, object_states, to_ego_frame
from autonomy_evaluation.evaluations.Metric import Aggregation, CONTEXT, EVALUATED, Input, Metric, Sample
from autonomy_evaluation.evaluations.sampling import sampling_period
from autonomy_evaluation.utils.ClosedLoopUtils import ClosedLoopUtils
from perception_msgs.msg import EgoData, ObjectList

# Collision types, following nuPlan, by whether the ego vehicle is at fault; None if that cannot be decided without
# the lanes the road users drive in
COLLISION_TYPES: Dict[str, Optional[bool]] = {
    "active_front": True,
    "active_rear": False,
    "active_lateral": None,
    "stopped_ego": False,
    "stopped_object": True,
}

# Conflict types of encounters, following SSAM, by the largest angle in radians between the paths of the road users
CONFLICT_TYPES: Dict[str, float] = {
    "rear_end": math.radians(30.0),
    "lane_change": math.radians(85.0),
    "crossing": math.pi,
}


def criticality_inputs() -> Dict[str, Input]:
    """Declare the inputs of the criticality metrics: the evaluated ``ego_data`` and the ``objects`` as context."""
    return {"ego_data": Input(EgoData, EVALUATED), "objects": Input(ObjectList, CONTEXT)}


class _CriticalityMetric(Metric):
    """Metric of the criticality of the ego vehicle towards the other road users."""

    def inputs(self) -> Dict[str, Input]:
        """Read the ego vehicle and the other road users."""
        return criticality_inputs()


class Collision(_CriticalityMetric):
    """Collisions of the ego vehicle, i.e. overlaps of its footprint with that of another road user.

    An overlap with the same road user over consecutive samples is one
    collision. Each collision is classified at its first sample like the
    collision types of nuPlan:

    - ``stopped_ego``: the ego vehicle stands still, so it is not at fault.
    - ``stopped_object``: the other road user stands still, so the ego vehicle
      is at fault.
    - ``active_front``: the other road user hits the front of the ego vehicle,
      which is at fault.
    - ``active_rear``: the other road user hits the rear of the ego vehicle,
      which is not at fault.
    - ``active_lateral``: the other road user hits a side of the ego vehicle;
      whether the ego vehicle is at fault depends on which of them left its
      lane, which is undetermined without the lanes.

    Configuration:
        ``standstill_speed``: speed in m/s below which a road user stands still.
    """

    NAME = "collision"

    def default_config(self) -> Dict[str, Any]:
        """Take road users slower than 0.1 m/s as standing still."""
        return {"standstill_speed": 0.1}

    def compute_sample(self, sample: Sample) -> List[Dict[str, Any]]:
        """Find the road users overlapping the ego vehicle at a sample.

        Returns:
            One ``{"id", "side", "ego_speed", "object_speed",
            "relative_speed"}`` entry per overlapping road user: its ID, the side
            of the ego vehicle it overlaps (``None`` if they only touch), the
            speeds of both in m/s, ``None`` for the ego vehicle without
            velocity, and the magnitude of their relative velocity.
        """
        return _collisions(sample)

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Count the collisions by their types.

        Returns:
            ``{"_value_", "at_fault", "not_at_fault", "undetermined", "types",
            "max_relative_speed"}``: the number of collisions, of those the ego
            vehicle is at fault for, is not at fault for and of those undetermined,
            the number of collisions per type, and the largest relative speed of
            the colliding road users at the first sample of a collision in m/s,
            ``None`` without collision or relative speed.
        """
        types = {collision_type: 0 for collision_type in COLLISION_TYPES}
        relative_speeds: List[float] = []
        for sequence in aggregation.sequences():
            colliding: set = set()
            for record in aggregation.records(self.name, sequence):
                for collision in record:
                    if collision["id"] not in colliding:
                        types[self._collision_type(collision)] += 1
                        if collision["relative_speed"] is not None:
                            relative_speeds.append(collision["relative_speed"])
                colliding = {collision["id"] for collision in record}
        return {
            "_value_": sum(types.values()),
            "at_fault": sum(count for collision_type, count in types.items() if COLLISION_TYPES[collision_type] is True),
            "not_at_fault": sum(count for collision_type, count in types.items() if COLLISION_TYPES[collision_type] is False),
            "undetermined": sum(count for collision_type, count in types.items() if COLLISION_TYPES[collision_type] is None),
            "types": types,
            "max_relative_speed": max(relative_speeds, default=None),
        }

    def visualization_outputs(self) -> Dict[str, Any]:
        """Publish the road users colliding with the ego vehicle."""
        return {"colliding_objects": ObjectList}

    def visualize_sample(self, sample: Sample) -> Dict[str, Any]:
        """Select the road users overlapping the ego vehicle at a sample, in the frame of the ego vehicle."""
        objects = object_states(sample)
        colliding = np.isin(objects.ids, [collision["id"] for collision in _collisions(sample)])
        return {"colliding_objects": ObjectList(header=objects.header, objects=objects.select(colliding))}

    def _collision_type(self, collision: Dict[str, Any]) -> str:
        """Classify a collision at its first sample."""
        standstill_speed = self.config["standstill_speed"]
        if collision["ego_speed"] is not None and collision["ego_speed"] < standstill_speed:
            return "stopped_ego"
        if collision["object_speed"] < standstill_speed:
            return "stopped_object"
        if collision["side"] == "front":
            return "active_front"
        if collision["side"] == "rear":
            return "active_rear"
        return "active_lateral"


class TimeToCollision(_CriticalityMetric):
    """Time to collision (TTC): time until the ego vehicle and a road user collide if both keep their velocity.

    At each sample, the TTC of the ego vehicle is the smallest TTC to any road
    user within the horizon. Over the samples, the time exposed (TET) is the
    time the TTC of the ego vehicle stays below ``exposure_threshold``, and the
    time integrated (TIT) integrates by how much it does so, following
    Minderhoud and Bovy (2001) for the ego vehicle. Each sample counts for the
    sampling period of its sequence. Samples without velocity of the ego
    vehicle have no TTC.

    Configuration:
        ``horizon``: seconds the motion is extrapolated for; longer TTCs are none.
        ``time_step``: seconds between the extrapolated positions, which are
        refined by bisection.
        ``exposure_threshold``: TTC in seconds below which the ego vehicle is
        exposed, the TTC* parameter of TET and TIT.
    """

    NAME = "ttc"

    def default_config(self) -> Dict[str, Any]:
        """Extrapolate for 5 s at 0.1 s, counting exposure below a TTC of 3 s."""
        return {"horizon": 5.0, "time_step": 0.1, "exposure_threshold": 3.0}

    def compute_sample(self, sample: Sample) -> Optional[Dict[str, Any]]:
        """Compute the time to collision of the ego vehicle at a sample.

        Returns:
            ``{"ttc", "object_id"}``: the smallest TTC in seconds and the ID of
            its road user, both ``None`` if no road user collides within the
            horizon; ``None`` without velocity of the ego vehicle.
        """
        ttc = time_to_collision(sample, self.config["horizon"], self.config["time_step"])
        if ttc is None:
            return None
        if np.isnan(ttc).all():
            return {"ttc": None, "object_id": None}
        closest = int(np.nanargmin(ttc))
        return {"ttc": float(ttc[closest]), "object_id": int(object_states(sample).ids[closest])}

    def visualization_outputs(self) -> Dict[str, Any]:
        """Publish the road users the ego vehicle collides with within the horizon."""
        return {"approaching_objects": ObjectList}

    def visualize_sample(self, sample: Sample) -> Dict[str, Any]:
        """Select the road users with a time to collision within the horizon at a sample, in the frame of the ego vehicle."""
        objects = object_states(sample)
        ttc = time_to_collision(sample, self.config["horizon"], self.config["time_step"])
        approaching = np.zeros(len(objects.objects), dtype=bool) if ttc is None else ~np.isnan(ttc)
        return {"approaching_objects": ObjectList(header=objects.header, objects=objects.select(approaching))}

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Report the smallest time to collision and the exposure to small ones.

        Returns:
            ``{"_value_", "share", "time_exposed", "time_integrated"}``: the
            smallest TTC over all samples in seconds, ``None`` if no road user
            ever collides within the horizon; the share of the samples with
            velocity of the ego vehicle that have a TTC within the horizon,
            ``None`` without any such sample; the time exposed in seconds and
            the time integrated in s².
        """
        threshold = self.config["exposure_threshold"]
        ttcs: List[float] = []
        num_available, time_exposed, time_integrated = 0, 0.0, 0.0
        for sequence in aggregation.sequences():
            period = sampling_period(sequence) or 0.0
            for record in aggregation.records(self.name, sequence):
                if record is None:
                    continue
                num_available += 1
                if record["ttc"] is not None:
                    ttcs.append(record["ttc"])
                    if record["ttc"] < threshold:
                        time_exposed += period
                        time_integrated += (threshold - record["ttc"]) * period
        return {
            "_value_": min(ttcs, default=None),
            "share": len(ttcs) / num_available if num_available else None,
            "time_exposed": time_exposed,
            "time_integrated": time_integrated,
        }


class TimeHeadway(_CriticalityMetric):
    """Time headway (THW): time the ego vehicle needs to reach the rear of the road user it follows at its current speed.

    The road user the ego vehicle follows is the closest one ahead whose
    geometric center lies in a corridor along the path ahead of the ego
    vehicle. The path continues the current yaw rate of the ego vehicle as a
    circle, straight if its yaw rate is not set. The gap is the distance along
    the path from the front of the ego vehicle to the rear of the road user,
    zero if they overlap.

    Configuration:
        ``corridor_width``: width in meters of the corridor along the path,
        e.g. the width of a lane.
        ``min_speed``: speed in m/s of the ego vehicle below which it has no
        time headway, as it hardly moves.
    """

    NAME = "thw"

    def default_config(self) -> Dict[str, Any]:
        """Search the lead vehicle in a corridor of a lane width of 3.5 m, for an ego vehicle moving at 0.5 m/s at least."""
        return {"corridor_width": 3.5, "min_speed": 0.5}

    def compute_sample(self, sample: Sample) -> Optional[Dict[str, Any]]:
        """Find the road user the ego vehicle follows at a sample and the time headway to it.

        Returns:
            ``{"thw", "gap", "object_id"}``: the time headway in seconds, the
            gap in meters and the ID of the followed road user; ``None`` if the
            ego vehicle follows no road user or moves slower than ``min_speed``.
        """
        lead = lead_vehicle(sample, self.config["corridor_width"], self.config["min_speed"])
        if lead is None:
            return None
        index, gap = lead
        return {"thw": gap / ego_state(sample).vel_lon, "gap": gap, "object_id": int(object_states(sample).ids[index])}

    def visualization_outputs(self) -> Dict[str, Any]:
        """Publish the road user the ego vehicle follows."""
        return {"lead_vehicle": ObjectList}

    def visualize_sample(self, sample: Sample) -> Dict[str, Any]:
        """Select the road user the ego vehicle follows at a sample, if any, in the frame of the ego vehicle."""
        objects = object_states(sample)
        lead = lead_vehicle(sample, self.config["corridor_width"], self.config["min_speed"])
        followed = np.arange(len(objects.objects)) == (-1 if lead is None else lead[0])
        return {"lead_vehicle": ObjectList(header=objects.header, objects=objects.select(followed))}

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Report the smallest and the mean time headway.

        Returns:
            ``{"_value_", "mean", "share"}``: the smallest and the mean time
            headway over the samples in which the ego vehicle follows a road
            user in seconds, ``None`` if it never does, and the share of the
            samples in which it follows one.
        """
        thws = [record["thw"] for record in aggregation.records(self.name) if record is not None]
        return {
            "_value_": min(thws, default=None),
            "mean": float(np.mean(thws)) if thws else None,
            "share": len(thws) / len(aggregation.entries) if aggregation.entries else None,
        }


class PostEncroachmentTime(_CriticalityMetric):
    """Post-encroachment time (PET): time between the ego vehicle and another road user passing the same spot.

    The trajectories of the ego vehicle and of each road user over a sequence of
    samples, resampled at ``time_step``, are rasterized into the cells of a
    ground grid of ``cell_size``. For every cell both pass, the PET is the time
    between the first road user leaving the cell and the second one entering it,
    zero if both occupy it at the same time. The PET of an encounter is the
    smallest PET over the cells, and the road users encounter each other if it
    is at most ``max_pet``. Each road user encounters the ego vehicle at most
    once per sequence, at its smallest PET. Its resolution is ``time_step``.

    Encounters are classified by the angle between the headings of both road
    users at the cell of their PET, like the conflict types of SSAM:
    ``rear_end`` below 30°, e.g. following a road user, ``lane_change`` up to
    85°, and ``crossing`` beyond, which includes oncoming road users.

    Configuration:
        ``cell_size``: edge length in meters of the cells of the ground grid.
        ``time_step``: seconds between the resampled poses of the trajectories.
        ``max_pet``: PET in seconds up to which two road users encounter each
        other; road users passing the same spot further apart in time do not.
    """

    NAME = "pet"

    def default_config(self) -> Dict[str, Any]:
        """Rasterize at 0.5 m and resample at 0.1 s, counting encounters within 10 s."""
        return {"cell_size": 0.5, "time_step": 0.1, "max_pet": 10.0}

    def compute_sample(self, sample: Sample) -> Dict[str, Any]:
        """Record the footprints of the ego vehicle and the other road users at a sample.

        Returns:
            ``{"stamp", "ego", "ids", "rectangles"}``: the stamp of the sample,
            the footprint of the ego vehicle as ``(x, y, yaw, length, width)``,
            and the IDs and footprints of the other road users as arrays.
        """
        objects = object_states(sample)
        return {"stamp": sample.stamp, "ego": ego_state(sample).rectangle, "ids": objects.ids, "rectangles": objects.rectangles}

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Find the encounters of the ego vehicle with the other road users and report their PETs.

        Returns:
            ``{"_value_", "num_encounters", "ego_first", "types"}``: the smallest
            PET of all encounters in seconds, ``None`` without encounter; the
            number of encounters and of those in which the ego vehicle passed the
            spot first; and ``{"_value_", "num_encounters"}`` per conflict type.
        """
        encounters: List[Dict[str, Any]] = []
        for sequence in aggregation.sequences():
            records = aggregation.records(self.name, sequence)
            if len(records) > 1 and all(record["stamp"] is not None for record in records):
                encounters += self._encounters(records)
        types = {
            conflict_type: [encounter["pet"] for encounter in encounters if encounter["type"] == conflict_type]
            for conflict_type in CONFLICT_TYPES
        }
        return {
            "_value_": min((encounter["pet"] for encounter in encounters), default=None),
            "num_encounters": len(encounters),
            "ego_first": sum(encounter["ego_first"] for encounter in encounters),
            "types": {
                conflict_type: {"_value_": min(pets, default=None), "num_encounters": len(pets)}
                for conflict_type, pets in types.items()
            },
        }

    def _encounters(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find the encounters of the ego vehicle with the other road users over the records of one sequence."""
        cell_size, time_step, max_pet = self.config["cell_size"], self.config["time_step"], self.config["max_pet"]
        times = np.array([record["stamp"] for record in records])
        ego = np.array([record["ego"] for record in records])
        ego_occupancy = ClosedLoopUtils.occupancy_intervals(
            times, ego[:, 0], ego[:, 1], ego[:, 2], ego[:, 3].max(), ego[:, 4].max(), cell_size, time_step
        )
        # road users that never come close to the area the ego vehicle passes cannot share a cell with it
        ego_radius = 0.5 * float(np.hypot(ego[:, 3], ego[:, 4]).max())
        area_min, area_max = ego[:, :2].min(axis=0) - ego_radius, ego[:, :2].max(axis=0) + ego_radius

        encounters = []
        for object_id, trajectory in _trajectories(records).items():
            radius = 0.5 * float(np.hypot(trajectory[:, 4], trajectory[:, 5]).max())
            if (trajectory[:, 1:3].max(axis=0) + radius < area_min).any() or (
                trajectory[:, 1:3].min(axis=0) - radius > area_max
            ).any():
                continue
            occupancy = ClosedLoopUtils.occupancy_intervals(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                trajectory[:, 3],
                trajectory[:, 4].max(),
                trajectory[:, 5].max(),
                cell_size,
                time_step,
            )
            encounter = ClosedLoopUtils.post_encroachment_time(ego_occupancy, occupancy)
            if encounter is None or encounter["pet"] > max_pet:
                continue
            encounters.append(
                {
                    "object_id": object_id,
                    "pet": encounter["pet"],
                    "ego_first": encounter["first_enters"],
                    "type": next(name for name, angle in CONFLICT_TYPES.items() if encounter["angle"] <= angle),
                }
            )
        return encounters


def time_to_collision(sample: Sample, horizon: float, time_step: float) -> Optional[np.ndarray]:
    """Compute the time to collision of the ego vehicle with each road user of a sample.

    Args:
        sample: Sample holding the ``ego_data`` and ``objects`` messages.
        horizon: Seconds the motion is extrapolated for.
        time_step: Seconds between the extrapolated positions.

    Returns:
        The time to collision of each road user in seconds, ``nan`` beyond the
        horizon, see :meth:`ClosedLoopUtils.time_to_collision`; ``None`` without
        velocity of the ego vehicle.
    """

    def compute() -> Optional[np.ndarray]:
        ego = ego_state(sample)
        if ego.velocity is None:
            return None
        objects = object_states(sample)
        return ClosedLoopUtils.time_to_collision(
            (*ego.rectangle, *ego.velocity), (*objects.rectangles, objects.vx, objects.vy), horizon, time_step
        )

    return sample.memo(("ttc", horizon, time_step), compute)


def lead_vehicle(sample: Sample, corridor_width: float, min_speed: float) -> Optional[Tuple[int, float]]:
    """Find the road user the ego vehicle of a sample follows, see :class:`TimeHeadway`.

    Args:
        sample: Sample holding the ``ego_data`` and ``objects`` messages.
        corridor_width: Width in meters of the corridor along the path ahead.
        min_speed: Speed in m/s of the ego vehicle below which it follows no one.

    Returns:
        ``(index, gap)``: the index of the followed road user among the
        objects and the gap to it in meters; ``None`` if the ego vehicle
        follows no road user or moves slower than ``min_speed``.
    """

    def find() -> Optional[Tuple[int, float]]:
        ego = ego_state(sample)
        if ego.vel_lon is None or ego.vel_lon < min_speed:
            return None
        objects = object_states(sample)
        curvature = ego.yaw_rate / ego.vel_lon if ego.yaw_rate is not None else 0.0
        s, d = ClosedLoopUtils.path_coordinates(*to_ego_frame(ego, objects.x, objects.y), curvature)
        # extent of each road user along the path, whose heading at the road user turned by the curvature
        heading = ClosedLoopUtils.wrap_angle(objects.yaw - ego.yaw - curvature * s)
        half_extent = np.abs(0.5 * objects.length * np.cos(heading)) + np.abs(0.5 * objects.width * np.sin(heading))
        gap = np.maximum(s - 0.5 * ego.length - half_extent, 0.0)
        candidates = np.flatnonzero((s > 0.0) & (np.abs(d) <= 0.5 * corridor_width))
        if candidates.size == 0:
            return None
        index = int(candidates[np.argmin(gap[candidates])])
        return index, float(gap[index])

    return sample.memo(("lead_vehicle", corridor_width, min_speed), find)


def _collisions(sample: Sample) -> List[Dict[str, Any]]:
    """Find the road users overlapping the ego vehicle of a sample, see :meth:`Collision.compute_sample`."""

    def find() -> List[Dict[str, Any]]:
        ego, objects = ego_state(sample), object_states(sample)
        overlapping = np.flatnonzero(ClosedLoopUtils.rectangles_overlap(ego.rectangle, objects.rectangles))
        collisions = []
        for index in overlapping:
            rectangle = tuple(float(values[index]) for values in objects.rectangles)
            velocity = (float(objects.vx[index]), float(objects.vy[index]))
            collisions.append(
                {
                    "id": int(objects.ids[index]),
                    "side": ClosedLoopUtils.impact_side(ego.rectangle, rectangle),
                    "ego_speed": ego.speed,
                    "object_speed": math.hypot(*velocity),
                    "relative_speed": (
                        None if ego.velocity is None else math.hypot(velocity[0] - ego.velocity[0], velocity[1] - ego.velocity[1])
                    ),
                }
            )
        return collisions

    return sample.memo("collisions", find)


def _trajectories(records: List[Dict[str, Any]]) -> Dict[int, np.ndarray]:
    """Gather the trajectory of each road user over the records of the PET of one sequence.

    Returns:
        Road user ID to an array of its ``(stamp, x, y, yaw, length, width)``
        at the samples it is present in, in the order of the samples.
    """
    trajectories: Dict[int, List[Tuple[float, ...]]] = {}
    for record in records:
        x, y, yaw, length, width = record["rectangles"]
        for index, object_id in enumerate(record["ids"].tolist()):
            trajectories.setdefault(object_id, []).append(
                (record["stamp"], x[index], y[index], yaw[index], length[index], width[index])
            )
    return {object_id: np.array(trajectory) for object_id, trajectory in trajectories.items()}
