# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for the trajectory deviation evaluation: controller, planner against ground truth, and two planners.

The ego vehicle drives along the x axis at 10 m/s, sampled at 10 Hz. At each sample, a planner plans
a trajectory from the current position on, every 0.1 s, deviating from the driven trajectory in a
known way, e.g. 0.5 m to the left or 2 m/s too fast, so that the expected deviations follow by hand.
"""

from __future__ import annotations

import math

import pytest
from autonomy_evaluation.evaluations.trajectory_deviation.TrajectoryDeviation import TrajectoryDeviation
from tests.evaluations.closed_loop.messages import ego_data, planned_state, static_tf_buffer

_PERIOD = 0.1
_SPEED = 10.0


def _plan(stamp: float, duration: float = 5.0, offset: float = 0.0, speed: float = _SPEED, yaw: float = 0.0, frame_id="map"):
    """Plan to drive on from the position at a stamp at a speed, offset to the left and heading as given."""
    steps = round(duration / _PERIOD)
    return [
        planned_state(stamp + step * _PERIOD, _SPEED * stamp + speed * step * _PERIOD, offset, yaw, speed, frame_id)
        for step in range(steps + 1)
    ]


def _ego(stamp: float, plan=(), frame_id: str = "map", x_offset: float = 0.0):
    """The ego vehicle driving along the x axis at a stamp, with the trajectory planned at it."""
    return ego_data(x=_SPEED * stamp + x_offset, vel_lon=_SPEED, stamp=stamp, frame_id=frame_id, trajectory_planned=plan)


def _evaluate(samples, config=None, tf_buffer=None) -> dict:
    """Record ``(stamp, evaluated, reference)`` samples and aggregate them."""
    evaluation = TrajectoryDeviation()
    evaluation.configure(config or {})
    evaluation.tf_buffer = tf_buffer
    for index, (stamp, evaluated, reference) in enumerate(samples):
        evaluation.record_sample(sample_id=str(index), stamp=stamp, evaluated=evaluated, reference=reference)
    return evaluation.finalize()["metrics"]


class TestController:
    """Tests how well the driven trajectory follows the planned one, both read from the same topic."""

    def test_driving_to_the_right_of_the_plan(self):
        """Plans 0.5 m to the left of the driven path leave the driven trajectory 0.5 m to the right of them."""
        samples = []
        for index in range(21):
            ego = _ego(index * _PERIOD, plan=_plan(index * _PERIOD, offset=0.5))
            samples.append((index * _PERIOD, ego, ego))

        metrics = _evaluate(samples, {"evaluated_source": "driven", "reference_source": "planned", "horizon": 1.0})

        assert metrics["_value_"] == pytest.approx(0.5)
        assert metrics["lateral"]["_value_"] == pytest.approx(0.5)
        assert metrics["lateral"]["mean"] == pytest.approx(-0.5)
        assert metrics["longitudinal"]["max_abs"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["heading"]["_value_"] == pytest.approx(0.0, abs=1e-9)
        # the last sample has no driven trajectory ahead of it
        assert metrics["displacement"]["coverage"] == pytest.approx(20 / 21)


class TestPlannerAgainstGroundTruth:
    """Tests a planned trajectory against the trajectory driven by the ground truth, the default sources."""

    def setup_method(self):
        """Plan 12 m/s at each of 31 samples over 3 s, while the ground truth drives 10 m/s."""
        self.samples = []
        for index in range(31):
            stamp = index * _PERIOD
            self.samples.append((stamp, _ego(stamp, plan=_plan(stamp, speed=12.0)), _ego(stamp)))

    def test_planning_too_fast(self):
        """The planned position runs ahead by 2 m per second, which only the first sample compares up to 3 s."""
        metrics = _evaluate(self.samples)

        assert metrics["displacement"]["at_1s"] == pytest.approx(2.0)
        assert metrics["displacement"]["at_3s"] == pytest.approx(6.0)
        assert metrics["displacement"]["max"] == pytest.approx(6.0)
        assert metrics["longitudinal"]["mean"] == pytest.approx(metrics["displacement"]["_value_"])
        assert metrics["velocity"]["_value_"] == pytest.approx(2.0)
        assert metrics["velocity"]["mean"] == pytest.approx(2.0)

    def test_misses_beyond_the_threshold(self):
        """Samples followed by at least 1.1 s of ground truth deviate by more than 2.1 m: 20 of the 30 compared."""
        displacement = _evaluate(self.samples, {"displacement": {"miss_threshold": 2.1}})["displacement"]

        assert displacement["miss_rate"] == pytest.approx(20 / 30)

    def test_horizons_are_configurable(self):
        """The displacement is reported at the configured horizons."""
        displacement = _evaluate(self.samples, {"displacement": {"horizons": [0.5]}})["displacement"]

        assert displacement["at_0.5s"] == pytest.approx(1.0)
        assert "at_1s" not in displacement


class TestTwoPlanners:
    """Tests two planned trajectories against each other."""

    _PLANNED = {"evaluated_source": "planned", "reference_source": "planned"}

    def test_heading_and_offset_between_two_plans(self):
        """Plans heading 0.1 rad off and 1 m to the left of the reference plans."""
        samples = [
            (stamp, _ego(stamp, plan=_plan(stamp, offset=1.0, yaw=0.1)), _ego(stamp, plan=_plan(stamp)))
            for stamp in (0.0, 0.1, 0.2)
        ]

        metrics = _evaluate(samples, self._PLANNED)

        assert metrics["heading"]["_value_"] == pytest.approx(0.1)
        assert metrics["lateral"]["mean"] == pytest.approx(1.0)
        assert metrics["displacement"]["coverage"] == 1.0

    def test_plans_in_another_frame_are_transformed_into_the_frame_of_the_reference(self):
        """The evaluated planner plans in 'odom', whose origin lies 10 m along x in 'map'."""
        tf_buffer = static_tf_buffer("map", "odom", x=10.0)
        evaluated = _ego(0.0, plan=_plan(0.0, frame_id="odom"), frame_id="odom", x_offset=-10.0)
        evaluated.trajectory_planned = [
            planned_state(
                state.header.stamp.sec + state.header.stamp.nanosec * 1e-9, _SPEED * step * _PERIOD - 10.0, frame_id="odom"
            )
            for step, state in enumerate(evaluated.trajectory_planned)
        ]

        metrics = _evaluate([(0.0, evaluated, _ego(0.0, plan=_plan(0.0)))], self._PLANNED, tf_buffer)

        assert metrics["displacement"]["max"] == pytest.approx(0.0, abs=1e-9)

    def test_without_transform_the_sample_cannot_be_compared(self):
        """Plans in another frame without transform cannot be related to the reference."""
        evaluated = _ego(0.0, plan=_plan(0.0, frame_id="odom"), frame_id="odom")

        with pytest.raises(ValueError, match="cannot be transformed"):
            _evaluate([(0.0, evaluated, _ego(0.0, plan=_plan(0.0)))], self._PLANNED)

    def test_plans_without_stamps_are_not_compared(self):
        """A planned trajectory whose states are not stamped cannot be related to time."""
        unstamped = _ego(0.0, plan=[planned_state(0.0, step * 1.0) for step in range(10)])

        assert _evaluate([(0.0, unstamped, _ego(0.0, plan=_plan(0.0)))], self._PLANNED)["displacement"]["coverage"] == 0.0

    def test_samples_without_plan_are_not_compared(self):
        """Without planned trajectory, there is nothing to compare."""
        metrics = _evaluate([(0.0, _ego(0.0), _ego(0.0, plan=_plan(0.0)))], self._PLANNED)

        assert metrics["displacement"]["_value_"] is None
        assert metrics["displacement"]["coverage"] == 0.0
        assert metrics["velocity"]["_value_"] is None


class TestConsistency:
    """Tests the consistency of consecutive plans of the evaluated input."""

    def test_plans_jumping_between_two_paths(self):
        """Plans alternating between two paths 1 m apart deviate by 1 m from the previous plan."""
        samples = []
        for index in range(5):
            stamp = index * _PERIOD
            ego = _ego(stamp, plan=_plan(stamp, offset=float(index % 2)))
            samples.append((stamp, ego, ego))

        consistency = _evaluate(samples, {"metrics": ["consistency"]})["consistency"]

        assert consistency["_value_"] == pytest.approx(1.0)
        assert consistency["coverage"] == 1.0

    def test_plans_kept_over_time(self):
        """A planner keeping to its plan is consistent."""
        samples = []
        for index in range(5):
            ego = _ego(index * _PERIOD, plan=_plan(index * _PERIOD))
            samples.append((index * _PERIOD, ego, ego))

        assert _evaluate(samples, {"metrics": ["consistency"]})["consistency"]["_value_"] == pytest.approx(0.0, abs=1e-9)

    def test_reads_the_evaluated_input_only(self):
        """The consistency of a planner needs no reference."""
        evaluation = TrajectoryDeviation()
        evaluation.configure({"metrics": ["consistency"]})

        assert list(evaluation.inputs()) == ["evaluated"]


class TestConfiguration:
    """Tests configuring the trajectory deviation evaluation."""

    @pytest.mark.parametrize(
        "config, error",
        [
            ({"evaluated_source": "predicted"}, "to be one of"),
            ({"reference_source": "recorded"}, "to be one of"),
            ({"horizon": 0.0}, "to be positive"),
            ({"time_step": -0.1}, "to be positive"),
        ],
    )
    def test_rejects_invalid_configurations(self, config, error):
        """Sources must be known and times positive."""
        with pytest.raises(ValueError, match=error):
            TrajectoryDeviation().configure(config)

    def test_reads_an_evaluated_and_a_reference_ego_vehicle(self):
        """The deviation metrics compare two inputs in their roles."""
        inputs = TrajectoryDeviation().inputs()

        assert {name: declared.role for name, declared in inputs.items()} == {"evaluated": "evaluated", "reference": "reference"}

    def test_headline_value_is_the_average_displacement(self):
        """The headline value of the evaluation is the average displacement error."""
        samples = [(stamp, _ego(stamp, plan=_plan(stamp, offset=math.sqrt(2.0))), _ego(stamp)) for stamp in (0.0, 0.1, 0.2)]

        metrics = _evaluate(samples)

        assert metrics["_value_"] == metrics["displacement"]["_value_"] == pytest.approx(math.sqrt(2.0))
