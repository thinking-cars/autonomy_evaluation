# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for the trajectory deviation evaluation: controller, planner against ground truth, and two planners.

The ego vehicle drives along the x axis at 10 m/s, sampled at 10 Hz. At each sample, a planner plans
a trajectory from the current position on, every 0.1 s, deviating from the driven trajectory in a
known way, e.g. 0.5 m to the left or 2 m/s too fast, so that the expected deviations follow by hand.
The plan is part of the ``EgoData`` or a ``trajectory_planning_msgs/Trajectory`` of the rear axle in
the frame of the vehicle, ``base_link``, 1.5 m behind the geometric center.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from autonomy_evaluation.evaluations.Metric import Sample
from autonomy_evaluation.evaluations.planning.trajectory_deviation.trajectories import message_trajectory
from autonomy_evaluation.evaluations.planning.trajectory_deviation.TrajectoryDeviation import TrajectoryDeviation
from perception_msgs.msg import EgoData
from tests.evaluations.driving_behavior.messages import (
    ego_data,
    moving_tf_buffer,
    planned_state,
    static_tf_buffer,
    trajectory,
)
from trajectory_planning_msgs.msg import DRIVABLE, REFERENCE, Trajectory

_PERIOD = 0.1
_SPEED = 10.0
# distance of the rear axle, the origin of base_link, behind the geometric center
_REAR_AXLE = 1.5
# first stamp of samples whose plans are transformed, as tf2 looks up the latest transform at stamp 0
_START = 1.0


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


def _planned_trajectory(stamp: float, duration: float = 1.0, offset: float = 0.0):
    """Plan to drive the rear axle on at the speed, offset to the left, as Trajectory message in the frame of the vehicle."""
    steps = round(duration / _PERIOD)
    return trajectory(stamp, [(step * _PERIOD, _SPEED * step * _PERIOD, offset, _SPEED, 0.0) for step in range(steps + 1)])


def _vehicle_frame(stamps) -> object:
    """Transforms of base_link, at the rear axle of the ego vehicle driving along the x axis, into map at the stamps."""
    return moving_tf_buffer("map", "base_link", [(stamp, _SPEED * stamp - _REAR_AXLE, 0.0, 0.0) for stamp in stamps])


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


class TestTrajectoryMessages:
    """Tests trajectories read from trajectory_planning_msgs/Trajectory, e.g. as OpenADStack plans them."""

    _CONTROLLER = {"evaluated_source": "driven", "reference_source": "trajectory", "fixed_frame": "map", "horizon": 1.0}

    def setup_method(self):
        """Plan 0.5 m to the left of the driven path at each of 21 samples, in the frame of the vehicle."""
        self.stamps = [_START + index * _PERIOD for index in range(21)]
        self.samples = [(stamp, _ego(stamp), _planned_trajectory(stamp, offset=0.5)) for stamp in self.stamps]

    def test_driving_to_the_right_of_a_trajectory_message(self):
        """Moved from the rear axle to the geometric center and into map, the plans run 0.5 m left of the driven path."""
        config = {**self._CONTROLLER, "base_link_to_geometric_center": [_REAR_AXLE, 0.0]}

        metrics = _evaluate(self.samples, config, _vehicle_frame(self.stamps))

        assert metrics["lateral"]["_value_"] == pytest.approx(0.5)
        assert metrics["lateral"]["mean"] == pytest.approx(-0.5)
        assert metrics["longitudinal"]["max_abs"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["heading"]["max_abs"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["velocity"]["max_abs"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["displacement"]["coverage"] == pytest.approx(20 / 21)

    def test_without_translation_the_rear_axle_is_compared_with_the_geometric_center(self):
        """Taking the rear axle for the geometric center leaves the driven path 1.5 m ahead of the plans."""
        metrics = _evaluate(self.samples, self._CONTROLLER, _vehicle_frame(self.stamps))

        assert metrics["longitudinal"]["mean"] == pytest.approx(_REAR_AXLE)
        assert metrics["displacement"]["_value_"] == pytest.approx(math.hypot(_REAR_AXLE, 0.5))

    def test_without_transform_the_sample_cannot_be_compared(self):
        """Plans in the frame of the vehicle need its transform into the frame compared in."""
        with pytest.raises(ValueError, match="cannot be transformed"):
            _evaluate(self.samples[:1], self._CONTROLLER)

    def test_reference_trajectory_heads_along_its_path(self):
        """A REFERENCE trajectory has no heading of its own: it heads along its path, here the y axis of map."""
        plan = trajectory(
            0.0,
            [(step * _PERIOD, 0.0, _SPEED * step * _PERIOD - _REAR_AXLE, _SPEED, None) for step in range(11)],
            REFERENCE.TYPE_ID,
            "map",
        )
        reference = _ego(
            0.0, plan=[planned_state(step * _PERIOD, 0.0, _SPEED * step * _PERIOD, math.pi / 2, _SPEED) for step in range(11)]
        )
        config = {
            "evaluated_source": "trajectory",
            "reference_source": "planned",
            "fixed_frame": "map",
            "base_link_to_geometric_center": [_REAR_AXLE, 0.0],
        }

        metrics = _evaluate([(0.0, plan, reference)], config)

        assert metrics["displacement"]["max"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["heading"]["max_abs"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["displacement"]["coverage"] == 1.0

    def test_consistency_of_plans_in_the_frame_of_the_vehicle(self):
        """Plans of the same path, each in the frame of the vehicle at its stamp, are consistent once in map."""
        stamps = [_START + index * _PERIOD for index in range(5)]
        samples = [(stamp, _planned_trajectory(stamp, duration=5.0), _ego(stamp)) for stamp in stamps]
        config = {"metrics": ["consistency"], "evaluated_source": "trajectory", "fixed_frame": "map"}

        consistency = _evaluate(samples, config, _vehicle_frame(stamps))["consistency"]

        assert consistency["_value_"] == pytest.approx(0.0, abs=1e-9)
        assert consistency["coverage"] == 1.0

    def test_sample_position_is_that_of_the_ego_vehicle(self):
        """A teleport is detected by the position of the ego vehicle, whichever input it is."""
        evaluation = TrajectoryDeviation()
        evaluation.configure(self._CONTROLLER)
        sample = Sample({"evaluated": _ego(1.0), "reference": _planned_trajectory(1.0)})

        assert evaluation.sample_position(sample) == pytest.approx((_SPEED, 0.0))

    def test_two_trajectory_messages_have_no_sample_position(self):
        """Two planners publishing trajectory messages report no position of the ego vehicle."""
        evaluation = TrajectoryDeviation()
        evaluation.configure({"evaluated_source": "trajectory", "reference_source": "trajectory", "fixed_frame": "map"})
        sample = Sample({"evaluated": _planned_trajectory(0.0), "reference": _planned_trajectory(0.0)})

        assert evaluation.sample_position(sample) is None


class TestReadingTrajectoryMessages:
    """Tests reading a trajectory_planning_msgs/Trajectory as trajectory of the geometric center."""

    @staticmethod
    def _yaw(points, type_id=REFERENCE.TYPE_ID, to_geometric_center=(0.0, 0.0)):
        """Read the headings of a trajectory of ``(t, x, y, v)`` sample points."""
        message = trajectory(0.0, [(t, x, y, v, 0.0) for t, x, y, v in points], type_id, "map")
        return message_trajectory(message, to_geometric_center)

    def test_heading_of_a_drivable_trajectory_is_its_own(self):
        """A DRIVABLE trajectory states its heading, which the geometric center is moved along."""
        message = trajectory(0.0, [(0.0, 0.0, 0.0, 1.0, math.pi / 2), (1.0, 0.0, 1.0, 1.0, math.pi / 2)], DRIVABLE.TYPE_ID)

        read = message_trajectory(message, [2.0, 0.0])

        assert read.yaw == pytest.approx([math.pi / 2] * 2)
        assert read.x == pytest.approx([0.0, 0.0], abs=1e-12)
        assert read.y == pytest.approx([2.0, 3.0])
        assert read.t == pytest.approx([0.0, 1.0])

    def test_reversing_heads_against_the_direction_of_travel(self):
        """Reversing along -x at a negative velocity heads along +x."""
        read = self._yaw([(0.0, 0.0, 0.0, -1.0), (1.0, -1.0, 0.0, -1.0)])

        assert np.cos(read.yaw) == pytest.approx([1.0, 1.0])

    def test_heading_while_standing_is_that_of_the_movement(self):
        """Standing first and then driving along y, the vehicle heads along y throughout."""
        read = self._yaw([(0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (2.0, 0.0, 1.0, 1.0), (3.0, 0.0, 1.0, 0.0)])

        assert read.yaw == pytest.approx([math.pi / 2] * 4)

    def test_standstill_heads_along_the_frame(self):
        """Without movement, the heading is that of the frame, e.g. forward in the frame of the vehicle."""
        read = self._yaw([(0.0, 1.0, 1.0, 0.0), (1.0, 1.0, 1.0, 0.0)], to_geometric_center=(1.0, 0.0))

        assert read.yaw == pytest.approx([0.0, 0.0])
        assert read.x == pytest.approx([2.0, 2.0])

    def test_rejects_malformed_states(self):
        """States must form whole sample points of a known type."""
        with pytest.raises(ValueError, match="no multiple"):
            message_trajectory(Trajectory(type_id=DRIVABLE.TYPE_ID, states=[0.0] * 5), (0.0, 0.0))
        with pytest.raises(ValueError):
            message_trajectory(Trajectory(type_id=7, states=[0.0] * 8), (0.0, 0.0))


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
            ({"reference_source": "trajectory"}, "to name a fixed frame"),
            ({"base_link_to_geometric_center": [1.0]}, r"to be \[x, y\]"),
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

    def test_inputs_are_read_as_the_messages_of_their_sources(self):
        """A trajectory source reads trajectory_planning_msgs/Trajectory, the others perception_msgs/EgoData."""
        evaluation = TrajectoryDeviation()
        evaluation.configure({"evaluated_source": "driven", "reference_source": "trajectory", "fixed_frame": "map"})

        assert {name: declared.msg_type for name, declared in evaluation.inputs().items()} == {
            "evaluated": EgoData,
            "reference": Trajectory,
        }

    def test_consistency_reads_trajectory_messages_of_the_evaluated_input(self):
        """The consistency of a planner publishing trajectory messages reads these."""
        evaluation = TrajectoryDeviation()
        evaluation.configure({"metrics": ["consistency"], "evaluated_source": "trajectory", "fixed_frame": "map"})

        assert evaluation.inputs()["evaluated"].msg_type is Trajectory

    def test_headline_value_is_the_average_displacement(self):
        """The headline value of the evaluation is the average displacement error."""
        samples = [(stamp, _ego(stamp, plan=_plan(stamp, offset=math.sqrt(2.0))), _ego(stamp)) for stamp in (0.0, 0.1, 0.2)]

        metrics = _evaluate(samples)

        assert metrics["_value_"] == metrics["displacement"]["_value_"] == pytest.approx(math.sqrt(2.0))
