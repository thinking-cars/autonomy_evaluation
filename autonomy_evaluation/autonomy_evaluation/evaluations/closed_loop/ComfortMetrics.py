# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Comfort metrics of a closed-loop evaluation: how smoothly the ego vehicle drives.

All metrics read the ``ego_data`` of the ego vehicle only and build on the
motion of the ego vehicle over each contiguous sequence of samples, which the
internal ``ego_motion`` metric derives once:

- ``acceleration``: longitudinal and lateral acceleration.
- ``jerk``: time derivative of the longitudinal and lateral acceleration.
- ``yaw_rate``: rate of turn.

A quantity the ego state does not set is derived from the samples by finite
differences: the yaw rate from the yaw, the accelerations from the velocities
in the rotating frame of the vehicle, i.e. ``a_lon = d(v_lon)/dt - v_lat·ω``
and ``a_lat = d(v_lat)/dt + v_lon·ω``. The jerk is always derived. Derived
quantities depend on the sampling rate: at a low rate, e.g. the 2 Hz key
frames of a dataset, peaks are smoothed out.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from autonomy_evaluation.evaluations.closed_loop.states import ego_state
from autonomy_evaluation.evaluations.Metric import Aggregation, EVALUATED, Input, Metric, Sample
from autonomy_evaluation.utils.ClosedLoopUtils import ClosedLoopUtils
from perception_msgs.msg import EgoData

# Source of a quantity read from the ego state, and of one derived from the samples by finite differences
_STATE = "state"
_DERIVED = "derived"


def ego_inputs() -> Dict[str, Input]:
    """Declare the input of the metrics of the ego vehicle: its evaluated ``ego_data``."""
    return {"ego_data": Input(EgoData, EVALUATED)}


class EgoMotion(Metric):
    """Internal metric deriving the motion of the ego vehicle over each contiguous sequence of samples.

    Its result is no metric, but the data the comfort metrics share: per
    sequence, the sample times and the yaw rate, accelerations and jerks of the
    ego vehicle, each with its source.
    """

    NAME = "ego_motion"
    INTERNAL = True

    def inputs(self) -> Dict[str, Input]:
        """Read the ego vehicle."""
        return ego_inputs()

    def compute_sample(self, sample: Sample) -> Tuple[Optional[float], ...]:
        """Record the kinematic state of the ego vehicle at a sample.

        Returns:
            ``(stamp, yaw, vel_lon, vel_lat, acc_lon, acc_lat, yaw_rate)``, each
            ``None`` if not set.
        """
        ego = ego_state(sample)
        return sample.stamp, ego.yaw, ego.vel_lon, ego.vel_lat, ego.acc_lon, ego.acc_lat, ego.yaw_rate

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Derive the motion of the ego vehicle for each contiguous sequence of samples.

        Returns:
            ``{"sequences": [motion]}``, each motion a dictionary of arrays over
            the samples of a sequence: ``times``, ``yaw_rate``, ``acc_lon``,
            ``acc_lat``, ``jerk_lon`` and ``jerk_lat``, each ``None`` if it can
            neither be read nor derived, and the source (``"state"`` or
            ``"derived"``) of the yaw rate and the accelerations under
            ``sources``.
        """
        return {"sequences": [self._motion(aggregation.records(self.name, sequence)) for sequence in aggregation.sequences()]}

    @staticmethod
    def _motion(records: List[Tuple[Optional[float], ...]]) -> Dict[str, Any]:
        """Derive the motion of the ego vehicle over the records of one sequence."""
        stamps, yaw, vel_lon, vel_lat, acc_lon, acc_lat, yaw_rate = (list(values) for values in zip(*records))
        differentiable = len(records) > 1 and None not in stamps
        times = np.array(stamps, dtype=float) if None not in stamps else None

        def derivative(values: Optional[np.ndarray]) -> Optional[np.ndarray]:
            return ClosedLoopUtils.time_derivative(times, values) if differentiable and values is not None else None

        def state(values: List[Optional[float]]) -> Optional[np.ndarray]:
            return np.array(values, dtype=float) if None not in values else None

        sources: Dict[str, Optional[str]] = {}
        motion: Dict[str, Any] = {"times": times}
        motion["yaw_rate"], sources["yaw_rate"] = _read_or_derive(state(yaw_rate), lambda: derivative(np.unwrap(yaw)))
        omega = motion["yaw_rate"]
        lon, lat = state(vel_lon), state(vel_lat)
        if lon is not None and lat is None:
            lat = np.zeros_like(lon)
        motion["acc_lon"], sources["acc_lon"] = _read_or_derive(
            state(acc_lon), lambda: None if lon is None or omega is None else _plus(derivative(lon), -lat * omega)
        )
        motion["acc_lat"], sources["acc_lat"] = _read_or_derive(
            state(acc_lat), lambda: None if lon is None or omega is None else _plus(derivative(lat), lon * omega)
        )
        motion["jerk_lon"] = derivative(motion["acc_lon"])
        motion["jerk_lat"] = derivative(motion["acc_lat"])
        motion["sources"] = sources
        return motion


def _read_or_derive(read: Optional[np.ndarray], derive) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Take a quantity from the ego state if it is set at every sample, else derive it from the samples."""
    if read is not None:
        return read, _STATE
    derived = derive()
    return derived, None if derived is None else _DERIVED


def _plus(derivative: Optional[np.ndarray], term: np.ndarray) -> Optional[np.ndarray]:
    """Add a term to a derivative, ``None`` if the derivative is unavailable."""
    return None if derivative is None else derivative + term


class _ComfortMetric(Metric):
    """Metric of the comfort of the ego vehicle, building on its motion."""

    def inputs(self) -> Dict[str, Input]:
        """Read the ego vehicle."""
        return ego_inputs()

    def requires(self) -> Tuple[str, ...]:
        """Build on the motion of the ego vehicle."""
        return (EgoMotion.NAME,)

    @staticmethod
    def _concatenated(aggregation: Aggregation, quantity: str) -> np.ndarray:
        """Concatenate a quantity over all sequences in which it is available."""
        values = [motion[quantity] for motion in aggregation.results[EgoMotion.NAME]["sequences"] if motion[quantity] is not None]
        return np.concatenate(values) if values else np.empty(0)

    @staticmethod
    def _sources(aggregation: Aggregation, *quantities: str) -> List[str]:
        """List the sources the given quantities were taken from, over all sequences."""
        sources = {
            motion["sources"][quantity]
            for motion in aggregation.results[EgoMotion.NAME]["sequences"]
            for quantity in quantities
            if motion["sources"].get(quantity)
        }
        return sorted(sources)

    @staticmethod
    def _peak_magnitude(aggregation: Aggregation, longitudinal: str, lateral: str) -> Optional[float]:
        """Return the largest magnitude of a vector quantity over all samples, of the components available per sequence."""
        peaks = []
        for motion in aggregation.results[EgoMotion.NAME]["sequences"]:
            components = [motion[quantity] for quantity in (longitudinal, lateral) if motion[quantity] is not None]
            if components:
                peaks.append(float(np.max(np.sqrt(sum(np.square(component) for component in components)))))
        return max(peaks, default=None)


class Acceleration(_ComfortMetric):
    """Longitudinal and lateral acceleration of the ego vehicle."""

    NAME = "acceleration"

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Report the extremes and the root mean square of the accelerations.

        Returns:
            ``{"_value_", "longitudinal": {"max", "min", "rms", "sources"},
            "lateral": {"max_abs", "rms", "sources"}}``: the largest magnitude of
            the acceleration in m/s², the strongest longitudinal acceleration,
            the strongest deceleration (negative) and the root mean square of the
            longitudinal acceleration, the largest absolute and the root mean
            square lateral acceleration, each with whether it was read from the
            ego state or derived; ``None`` where no acceleration is available.
        """
        lon, lat = self._concatenated(aggregation, "acc_lon"), self._concatenated(aggregation, "acc_lat")
        return {
            "_value_": self._peak_magnitude(aggregation, "acc_lon", "acc_lat"),
            "longitudinal": {
                "max": _max(lon),
                "min": _min(lon),
                "rms": _rms(lon),
                "sources": self._sources(aggregation, "acc_lon"),
            },
            "lateral": {"max_abs": _max(np.abs(lat)), "rms": _rms(lat), "sources": self._sources(aggregation, "acc_lat")},
        }


class Jerk(_ComfortMetric):
    """Longitudinal and lateral jerk of the ego vehicle, the time derivatives of its accelerations."""

    NAME = "jerk"

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Report the extremes and the root mean square of the jerks.

        Returns:
            ``{"_value_", "longitudinal": {"max_abs", "rms"}, "lateral":
            {"max_abs", "rms"}}``: the largest magnitude of the jerk in m/s³,
            and the largest absolute and the root mean square jerk along and
            across the heading; ``None`` where no jerk is available, e.g. for
            sequences of single samples.
        """
        lon, lat = self._concatenated(aggregation, "jerk_lon"), self._concatenated(aggregation, "jerk_lat")
        return {
            "_value_": self._peak_magnitude(aggregation, "jerk_lon", "jerk_lat"),
            "longitudinal": {"max_abs": _max(np.abs(lon)), "rms": _rms(lon)},
            "lateral": {"max_abs": _max(np.abs(lat)), "rms": _rms(lat)},
        }


class YawRate(_ComfortMetric):
    """Yaw rate of the ego vehicle."""

    NAME = "yaw_rate"

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Report the extreme and the root mean square of the yaw rate.

        Returns:
            ``{"_value_", "rms", "sources"}``: the largest absolute yaw rate in
            rad/s, its root mean square, and whether it was read from the ego
            state or derived; ``None`` where no yaw rate is available.
        """
        yaw_rate = self._concatenated(aggregation, "yaw_rate")
        return {"_value_": _max(np.abs(yaw_rate)), "rms": _rms(yaw_rate), "sources": self._sources(aggregation, "yaw_rate")}


def _max(values: np.ndarray) -> Optional[float]:
    """Return the maximum of the values, ``None`` without values."""
    return float(np.max(values)) if values.size else None


def _min(values: np.ndarray) -> Optional[float]:
    """Return the minimum of the values, ``None`` without values."""
    return float(np.min(values)) if values.size else None


def _rms(values: np.ndarray) -> Optional[float]:
    """Return the root mean square of the values, ``None`` without values."""
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else None
