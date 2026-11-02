# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Metrics of the deviation of an evaluated trajectory from a reference trajectory.

At each sample, the trajectory of the ``evaluated`` input and that of the
``reference`` input (see
:mod:`~autonomy_evaluation.evaluations.trajectory_deviation.trajectories`) are
compared over the time they share within the ``horizon`` after the sample:
both are interpolated every ``time_step`` from the stamp of the sample on, and
at each of these times the evaluated pose is compared with the reference pose.
The internal ``deviations`` metric derives these comparisons once:

- ``displacement``: distance between the positions.
- ``longitudinal``, ``lateral``: the displacement along and across the heading
  of the reference, positive if the evaluated position is ahead and to the left.
- ``heading``: angle between the headings.
- ``velocity``: difference of the longitudinal velocities.

Each metric is averaged over the times of a sample and then over the compared
samples, e.g. the average displacement error (ADE). The ``consistency`` of the
planned trajectories of the evaluated input compares each planned trajectory
with the one planned at the previous sample instead.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from autonomy_evaluation.evaluations.Metric import Aggregation, EVALUATED, Input, Metric, REFERENCE, Sample
from autonomy_evaluation.evaluations.trajectory_deviation.trajectories import (
    driven_trajectory,
    frame_transform,
    PLANNED,
    planned_frame,
    planned_trajectory,
    state_pose,
    Trajectory,
    transformed_pose,
    transformed_trajectory,
)
from autonomy_evaluation.utils.ClosedLoopUtils import ClosedLoopUtils
from perception_msgs.msg import EgoData

# Names of the compared inputs
_EVALUATED = "evaluated"
_REFERENCE = "reference"

# Tolerance in seconds of the times a trajectory is compared at
_EPSILON = 1e-9


def deviation_inputs() -> Dict[str, Input]:
    """Declare the compared inputs: the ``evaluated`` and the ``reference`` ``EgoData``."""
    return {_EVALUATED: Input(EgoData, EVALUATED), _REFERENCE: Input(EgoData, REFERENCE)}


class Deviations(Metric):
    """Internal metric comparing the evaluated with the reference trajectory at each sample.

    Its result is no metric, but the comparisons the deviation metrics share:
    per sample, the times compared at and the deviations at these times.
    """

    NAME = "deviations"
    INTERNAL = True

    def inputs(self) -> Dict[str, Input]:
        """Read the evaluated and the reference ego vehicle."""
        return deviation_inputs()

    def compute_sample(self, sample: Sample) -> Dict[str, Any]:
        """Record what both trajectories need of a sample, in the frame of the reference.

        Returns:
            ``{"stamp", "evaluated", "reference"}``: the stamp of the sample, and
            per input its planned trajectory if that is its source, else the
            pose of the ego vehicle, from which the driven trajectory is built.

        Raises:
            ValueError: The evaluated input cannot be transformed into the frame of the reference.
        """
        frame_id = sample[_REFERENCE].header.frame_id
        record: Dict[str, Any] = {"stamp": sample.stamp}
        for name in (_EVALUATED, _REFERENCE):
            message = sample[name]
            if sample.config[f"{name}_source"] == PLANNED:
                transform = frame_transform(sample.tf_buffer, planned_frame(message), frame_id, message.header.stamp)
                record[name] = transformed_trajectory(planned_trajectory(message), transform)
            else:
                transform = frame_transform(sample.tf_buffer, message.header.frame_id, frame_id, message.header.stamp)
                record[name] = transformed_pose(state_pose(message), transform)
        return record

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Compare both trajectories at each sample, building driven trajectories per contiguous sequence.

        Returns:
            ``{"samples": [deviation]}``, one entry per sample, ``None`` if the
            trajectories share no time within the horizon, e.g. without planned
            trajectory or at the last sample of a sequence for a driven one;
            else a dictionary of arrays over the compared times: ``tau``, the
            seconds after the sample, ``displacement``, ``longitudinal``,
            ``lateral``, ``heading`` and ``velocity``, the latter ``None`` if a
            trajectory has no velocities.
        """
        config = aggregation.config
        deviations: List[Optional[Dict[str, Any]]] = []
        for sequence in aggregation.sequences():
            records = aggregation.records(self.name, sequence)
            evaluated = _trajectories(records, _EVALUATED, config[f"{_EVALUATED}_source"], config["horizon"])
            reference = _trajectories(records, _REFERENCE, config[f"{_REFERENCE}_source"], config["horizon"])
            for record, evaluated_trajectory, reference_trajectory in zip(records, evaluated, reference):
                deviations.append(
                    _deviation(
                        evaluated_trajectory, reference_trajectory, record["stamp"], config["horizon"], config["time_step"]
                    )
                )
        return {"samples": deviations}


def _trajectories(records: List[Dict[str, Any]], name: str, source: str, horizon: float) -> List[Optional[Trajectory]]:
    """Return the trajectory of an input at each sample of a sequence, planned or driven over the horizon."""
    if source == PLANNED:
        return [record[name] for record in records]
    if any(record["stamp"] is None for record in records):
        return [None] * len(records)
    stamps, poses = [record["stamp"] for record in records], [record[name] for record in records]
    return [driven_trajectory(stamps, poses, index, horizon) for index in range(len(records))]


def _compared_times(
    first: Trajectory, second: Trajectory, stamp: float, horizon: float, time_step: float
) -> Optional[np.ndarray]:
    """Return the seconds after a sample at which two trajectories are compared: every time step they share within the horizon."""
    start = max(stamp, first.t[0], second.t[0])
    end = min(stamp + horizon, first.t[-1], second.t[-1])
    first_step = math.ceil((start - stamp) / time_step - _EPSILON)
    last_step = math.floor((end - stamp) / time_step + _EPSILON)
    if last_step < first_step:
        return None
    return np.arange(first_step, last_step + 1) * time_step


def _deviation(
    evaluated: Optional[Trajectory], reference: Optional[Trajectory], stamp: Optional[float], horizon: float, time_step: float
) -> Optional[Dict[str, Any]]:
    """Compare the evaluated with the reference trajectory of a sample, see :meth:`Deviations.aggregate`."""
    if evaluated is None or reference is None or stamp is None:
        return None
    tau = _compared_times(evaluated, reference, stamp, horizon, time_step)
    if tau is None:
        return None
    first, second = evaluated.interpolated(stamp + tau), reference.interpolated(stamp + tau)
    dx, dy = first.x - second.x, first.y - second.y
    cos, sin = np.cos(second.yaw), np.sin(second.yaw)
    return {
        "tau": tau,
        "displacement": np.hypot(dx, dy),
        "longitudinal": cos * dx + sin * dy,
        "lateral": -sin * dx + cos * dy,
        "heading": ClosedLoopUtils.wrap_angle(first.yaw - second.yaw),
        "velocity": None if first.velocity is None or second.velocity is None else first.velocity - second.velocity,
    }


class _DeviationMetric(Metric):
    """Metric of the deviation of the evaluated from the reference trajectory, building on their comparisons."""

    def inputs(self) -> Dict[str, Input]:
        """Read the evaluated and the reference ego vehicle."""
        return deviation_inputs()

    def requires(self) -> Tuple[str, ...]:
        """Build on the comparisons of both trajectories."""
        return (Deviations.NAME,)

    @staticmethod
    def _compared(aggregation: Aggregation, quantity: str = "displacement") -> Tuple[List[Dict[str, Any]], Optional[float]]:
        """Return the comparisons of the samples at which a quantity was compared, and the share of these samples."""
        samples = aggregation.results[Deviations.NAME]["samples"]
        compared = [deviation for deviation in samples if deviation is not None and deviation[quantity] is not None]
        return compared, len(compared) / len(samples) if samples else None

    @staticmethod
    def _errors(compared: List[Dict[str, Any]], quantity: str) -> Dict[str, Any]:
        """Report a signed deviation: its mean absolute value, its mean, its largest absolute value and its root mean square."""
        if not compared:
            return {"_value_": None, "mean": None, "max_abs": None, "rms": None}
        values = np.concatenate([deviation[quantity] for deviation in compared])
        return {
            "_value_": float(np.mean([np.mean(np.abs(deviation[quantity])) for deviation in compared])),
            "mean": float(np.mean([np.mean(deviation[quantity]) for deviation in compared])),
            "max_abs": float(np.max(np.abs(values))),
            "rms": float(np.sqrt(np.mean(np.square(values)))),
        }


class Displacement(_DeviationMetric):
    """Displacement of the evaluated from the reference position.

    Configuration:
        ``horizons``: seconds after the sample at which the displacement is
        reported on its own, e.g. after 1 s and 3 s.
        ``miss_threshold``: displacement in meters beyond which the evaluated
        trajectory misses the reference.
    """

    NAME = "displacement"

    def default_config(self) -> Dict[str, Any]:
        """Report the displacement after 1 s and 3 s, and count a miss beyond 2 m, like nuPlan."""
        return {"horizons": [1.0, 3.0], "miss_threshold": 2.0}

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Report the average, final and largest displacement, the displacement at the horizons and the miss rate.

        Returns:
            ``{"_value_", "final", "max", "at_<horizon>s", "miss_rate",
            "coverage"}``: the average displacement error (ADE) in meters, i.e.
            the displacement averaged over the compared times of a sample and
            then over the samples; the final displacement error (FDE), at the
            last compared time of a sample, averaged over the samples; the
            largest displacement; per horizon, e.g. ``at_1s``, the displacement
            at that time averaged over the samples compared up to it; the share
            of the compared samples whose largest displacement exceeds
            ``miss_threshold``; and the share of the samples that were
            compared. All but ``coverage`` are ``None`` without a compared sample.
        """
        compared, coverage = self._compared(aggregation)
        result: Dict[str, Any] = {"_value_": None, "final": None, "max": None}
        for horizon in self.config["horizons"]:
            at_horizon = [
                float(np.interp(horizon, deviation["tau"], deviation["displacement"]))
                for deviation in compared
                if deviation["tau"][0] - _EPSILON <= horizon <= deviation["tau"][-1] + _EPSILON
            ]
            result[f"at_{horizon:g}s"] = float(np.mean(at_horizon)) if at_horizon else None
        result.update({"miss_rate": None, "coverage": coverage})
        if compared:
            result["_value_"] = float(np.mean([np.mean(deviation["displacement"]) for deviation in compared]))
            result["final"] = float(np.mean([deviation["displacement"][-1] for deviation in compared]))
            result["max"] = float(max(np.max(deviation["displacement"]) for deviation in compared))
            threshold = self.config["miss_threshold"]
            result["miss_rate"] = float(np.mean([np.max(deviation["displacement"]) > threshold for deviation in compared]))
        return result


class Longitudinal(_DeviationMetric):
    """Displacement along the heading of the reference, positive while the evaluated position is ahead."""

    NAME = "longitudinal"

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Report the longitudinal deviation.

        Returns:
            ``{"_value_", "mean", "max_abs", "rms"}``: the absolute
            longitudinal deviation in meters, averaged over the compared times
            of a sample and then over the samples; the same of the signed
            deviation, i.e. whether the evaluated position lags behind
            (negative) or runs ahead; the largest absolute and the root mean
            square deviation over all compared times; ``None`` without a
            compared sample.
        """
        return self._errors(self._compared(aggregation)[0], "longitudinal")


class Lateral(_DeviationMetric):
    """Displacement across the heading of the reference, positive while the evaluated position is to its left."""

    NAME = "lateral"

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Report the lateral deviation.

        Returns:
            ``{"_value_", "mean", "max_abs", "rms"}``: like ``longitudinal``,
            across the heading of the reference, the signed mean positive to
            its left.
        """
        return self._errors(self._compared(aggregation)[0], "lateral")


class Heading(_DeviationMetric):
    """Angle between the evaluated and the reference heading."""

    NAME = "heading"

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Report the heading deviation.

        Returns:
            ``{"_value_", "final", "max_abs"}``: the absolute heading deviation
            in radians averaged over the compared times of a sample and then
            over the samples, the same at the last compared time of a sample,
            and the largest absolute deviation; ``None`` without a compared sample.
        """
        compared, _ = self._compared(aggregation)
        if not compared:
            return {"_value_": None, "final": None, "max_abs": None}
        return {
            "_value_": float(np.mean([np.mean(np.abs(deviation["heading"])) for deviation in compared])),
            "final": float(np.mean([abs(deviation["heading"][-1]) for deviation in compared])),
            "max_abs": float(max(np.max(np.abs(deviation["heading"])) for deviation in compared)),
        }


class Velocity(_DeviationMetric):
    """Difference of the evaluated and the reference longitudinal velocity."""

    NAME = "velocity"

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Report the velocity deviation.

        Returns:
            ``{"_value_", "mean", "max_abs", "rms"}``: like ``longitudinal``, of
            the velocity in m/s, the signed mean positive while the evaluated
            velocity is higher; ``None`` without a sample at which both
            trajectories have velocities.
        """
        return self._errors(self._compared(aggregation, "velocity")[0], "velocity")


class Consistency(Metric):
    """Consistency of the planned trajectories of the evaluated input over consecutive samples.

    Each planned trajectory is compared with the one planned at the previous
    sample of the same sequence, over the time both share within the horizon
    after the later sample, like the deviation metrics compare the evaluated
    with the reference trajectory; consecutive planned trajectories are taken
    to be in the same frame. A planner that keeps to its plans deviates little,
    one that jumps between plans much.
    """

    NAME = "consistency"

    def inputs(self) -> Dict[str, Input]:
        """Read the evaluated ego vehicle only."""
        return {_EVALUATED: Input(EgoData, EVALUATED)}

    def compute_sample(self, sample: Sample) -> Tuple[Optional[float], Optional[Trajectory]]:
        """Record the stamp of a sample and the trajectory the evaluated input planned at it."""
        return sample.stamp, planned_trajectory(sample[_EVALUATED])

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Report how far consecutive planned trajectories deviate from each other.

        Returns:
            ``{"_value_", "max", "coverage"}``: the displacement between
            consecutive planned trajectories in meters, averaged over their
            compared times and then over the pairs of consecutive samples; the
            largest displacement; and the share of the pairs of consecutive
            samples that were compared; ``None`` without such pairs.
        """
        horizon, time_step = aggregation.config["horizon"], aggregation.config["time_step"]
        displacements: List[np.ndarray] = []
        num_pairs = 0
        for sequence in aggregation.sequences():
            records = aggregation.records(self.name, sequence)
            for (_, previous), (stamp, current) in zip(records, records[1:]):
                num_pairs += 1
                deviation = _deviation(current, previous, stamp, horizon, time_step)
                if deviation is not None:
                    displacements.append(deviation["displacement"])
        return {
            "_value_": float(np.mean([np.mean(values) for values in displacements])) if displacements else None,
            "max": float(max(np.max(values) for values in displacements)) if displacements else None,
            "coverage": len(displacements) / num_pairs if num_pairs else None,
        }
