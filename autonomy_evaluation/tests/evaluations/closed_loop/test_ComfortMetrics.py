# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for the comfort metrics of the closed-loop evaluation: acceleration, jerk and yaw rate.

The ego vehicle drives analytically known motions, e.g. accelerating uniformly or driving a circle,
sampled at 10 Hz, so that read and derived quantities can be checked against their exact values.
"""

from __future__ import annotations

import math

import pytest
from autonomy_evaluation.evaluations.closed_loop.ClosedLoop import ClosedLoop
from tests.evaluations.closed_loop.messages import ego_data

_PERIOD = 0.1


def _evaluate(states, scene_ids=None, metrics=None) -> dict:
    """Record the ego states of consecutive samples, ``(stamp, ego_data keyword arguments)``, and aggregate them."""
    evaluation = ClosedLoop()
    evaluation.configure({"metrics": metrics or ["comfort"]})
    for index, (stamp, state) in enumerate(states):
        scene_id = scene_ids[index] if scene_ids else None
        evaluation.record_sample(sample_id=str(index), scene_id=scene_id, stamp=stamp, ego_data=ego_data(stamp=stamp, **state))
    return evaluation.finalize()


def _uniform_acceleration(acceleration: float, with_state: bool, num_samples: int = 20) -> list:
    """States of an ego vehicle accelerating uniformly from 10 m/s, with or without the acceleration in its state."""
    return [
        (
            index * _PERIOD,
            {"vel_lon": 10.0 + acceleration * index * _PERIOD, "acc_lon": acceleration if with_state else None, "yaw_rate": 0.0},
        )
        for index in range(num_samples)
    ]


class TestAcceleration:
    """Tests the longitudinal and lateral acceleration of the ego vehicle."""

    def test_reads_the_acceleration_of_the_ego_state(self):
        """An acceleration set in the ego state is taken as it is."""
        acceleration = _evaluate(_uniform_acceleration(-3.0, with_state=True))["metrics"]["comfort"]["acceleration"]

        assert acceleration["longitudinal"] == {"max": -3.0, "min": -3.0, "rms": 3.0, "sources": ["state"]}
        # the lateral acceleration is not set, so it is derived: v_lat stays 0 and the yaw rate is 0
        assert acceleration["lateral"] == {"max_abs": 0.0, "rms": 0.0, "sources": ["derived"]}

    def test_derives_the_acceleration_from_the_velocity(self):
        """Without acceleration in the ego state, it is the time derivative of the velocity."""
        acceleration = _evaluate(_uniform_acceleration(2.0, with_state=False))["metrics"]["comfort"]["acceleration"]

        assert acceleration["longitudinal"]["max"] == pytest.approx(2.0)
        assert acceleration["longitudinal"]["min"] == pytest.approx(2.0)
        assert acceleration["longitudinal"]["sources"] == ["derived"]

    def test_lateral_acceleration_on_a_circle_is_centripetal(self):
        """Driving a circle at 10 m/s with a yaw rate of 0.2 rad/s accelerates by 2 m/s² towards its center."""
        states = [
            (index * _PERIOD, {"yaw": 0.2 * index * _PERIOD, "vel_lon": 10.0, "vel_lat": 0.0, "yaw_rate": 0.2})
            for index in range(20)
        ]

        acceleration = _evaluate(states)["metrics"]["comfort"]["acceleration"]

        assert acceleration["lateral"]["max_abs"] == pytest.approx(2.0)
        assert acceleration["longitudinal"]["max"] == pytest.approx(0.0, abs=1e-9)
        assert acceleration["_value_"] == pytest.approx(2.0)

    def test_peak_magnitude_combines_both_directions(self):
        """Braking while turning accelerates by the magnitude of both components."""
        states = [(index * _PERIOD, {"vel_lon": 10.0, "acc_lon": -3.0, "acc_lat": 4.0}) for index in range(5)]

        assert _evaluate(states)["metrics"]["comfort"]["acceleration"]["_value_"] == pytest.approx(5.0)

    def test_acceleration_is_not_derived_across_scenes(self):
        """The velocity jumps between two scenes, which is no acceleration, as the scenes are no sequence."""
        states = _uniform_acceleration(0.0, with_state=False, num_samples=5) + [
            ((index + 5) * _PERIOD, {"vel_lon": 30.0, "yaw_rate": 0.0}) for index in range(5)
        ]

        acceleration = _evaluate(states, scene_ids=["a"] * 5 + ["b"] * 5)["metrics"]["comfort"]["acceleration"]

        assert acceleration["longitudinal"]["max"] == pytest.approx(0.0)

    def test_acceleration_without_velocity_is_unavailable(self):
        """Without velocity and acceleration, nothing can be derived."""
        acceleration = _evaluate([(index * _PERIOD, {}) for index in range(5)])["metrics"]["comfort"]["acceleration"]

        assert acceleration == {
            "_value_": None,
            "longitudinal": {"max": None, "min": None, "rms": None, "sources": []},
            "lateral": {"max_abs": None, "rms": None, "sources": []},
        }


class TestJerk:
    """Tests the jerk of the ego vehicle."""

    def test_jerk_is_the_derivative_of_the_acceleration(self):
        """An acceleration rising by 1 m/s² per second is a jerk of 1 m/s³."""
        states = [(index * _PERIOD, {"vel_lon": 10.0, "acc_lon": index * _PERIOD, "acc_lat": 0.0}) for index in range(20)]

        jerk = _evaluate(states)["metrics"]["comfort"]["jerk"]

        assert jerk["longitudinal"]["max_abs"] == pytest.approx(1.0)
        assert jerk["lateral"]["max_abs"] == pytest.approx(0.0)
        assert jerk["_value_"] == pytest.approx(1.0)

    def test_single_samples_have_no_jerk(self):
        """Samples of different scenes are no sequence, so no derivative is known."""
        states = [(index * _PERIOD, {"vel_lon": 10.0, "acc_lon": 1.0}) for index in range(3)]

        jerk = _evaluate(states, scene_ids=["a", "b", "c"])["metrics"]["comfort"]["jerk"]

        assert jerk["_value_"] is None


class TestYawRate:
    """Tests the yaw rate of the ego vehicle."""

    def test_reads_the_yaw_rate_of_the_ego_state(self):
        """A yaw rate set in the ego state is taken as it is."""
        states = [(index * _PERIOD, {"vel_lon": 10.0, "yaw_rate": -0.3}) for index in range(5)]

        yaw_rate = _evaluate(states)["metrics"]["comfort"]["yaw_rate"]

        assert yaw_rate == {"_value_": 0.3, "rms": pytest.approx(0.3), "sources": ["state"]}

    def test_derives_the_yaw_rate_from_the_yaw_across_pi(self):
        """Without yaw rate in the ego state, it is the time derivative of the unwrapped yaw."""
        states = [(index * _PERIOD, {"yaw": math.remainder(3.0 + 0.5 * index * _PERIOD, 2.0 * math.pi)}) for index in range(10)]

        yaw_rate = _evaluate(states)["metrics"]["comfort"]["yaw_rate"]

        assert yaw_rate["_value_"] == pytest.approx(0.5)
        assert yaw_rate["sources"] == ["derived"]


class TestComfortSelection:
    """Tests selecting the comfort metrics."""

    def test_comfort_metrics_read_the_ego_vehicle_only(self):
        """The comfort of the ego vehicle needs no other road users."""
        evaluation = ClosedLoop()
        evaluation.configure({"metrics": ["comfort"]})

        assert list(evaluation.inputs()) == ["ego_data"]
        assert evaluation.inputs()["ego_data"].role == "evaluated"

    def test_single_comfort_metric(self):
        """A single comfort metric is selected as 'comfort.<metric>', building on the internal motion of the ego vehicle."""
        evaluation = ClosedLoop()
        evaluation.configure({"metrics": ["comfort.jerk"]})

        assert list(evaluation.computed_metrics()) == ["ego_motion", "comfort.jerk"]
