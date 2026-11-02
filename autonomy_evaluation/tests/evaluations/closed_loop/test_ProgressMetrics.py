# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for the progress metrics of the closed-loop evaluation: the distance the ego vehicle drives.

The ego vehicle drives along the x axis, sampled at 10 Hz, at speeds simple enough to compute the
distance and the time standing still by hand.
"""

from __future__ import annotations

import pytest
from autonomy_evaluation.evaluations.closed_loop.ClosedLoop import ClosedLoop
from tests.evaluations.closed_loop.messages import ego_data

_PERIOD = 0.1


def _evaluate(egos, scene_ids=None, config=None) -> dict:
    """Record one sample per set of ``ego_data`` keyword arguments, 0.1 s apart, and aggregate them."""
    evaluation = ClosedLoop()
    evaluation.configure({"metrics": ["progress"], **(config or {})})
    for index, ego in enumerate(egos):
        stamp = index * _PERIOD
        evaluation.record_sample(
            sample_id=str(index),
            scene_id=scene_ids[index] if scene_ids else None,
            stamp=stamp,
            ego_data=ego_data(stamp=stamp, **ego),
        )
    return evaluation.finalize()["metrics"]["progress"]


class TestDistanceDriven:
    """Tests the distance the ego vehicle drives, its mean speed and the time it stands still."""

    def test_driving_and_waiting(self):
        """Driving 10 m/s for 1 s covers 10 m, then waiting for five samples stands still for 0.5 s."""
        egos = [{"x": 10.0 * _PERIOD * index, "vel_lon": 10.0} for index in range(11)]
        egos += [{"x": 10.0, "vel_lon": 0.0} for _ in range(5)]

        distance = _evaluate(egos)["distance"]

        assert distance["_value_"] == pytest.approx(10.0)
        assert distance["mean_speed"] == pytest.approx(10.0 / 1.5)
        assert distance["standstill_time"] == pytest.approx(0.5)
        assert distance["standstill_share"] == pytest.approx(5 / 16)

    def test_distance_does_not_run_between_scenes(self):
        """The ego vehicle of the next scene starts elsewhere, which is no distance driven."""
        egos = [{"x": 0.0}, {"x": 1.0}, {"x": 100.0}, {"x": 101.0}]

        distance = _evaluate(egos, scene_ids=["a", "a", "b", "b"])["distance"]

        assert distance["_value_"] == pytest.approx(2.0)
        assert distance["standstill_share"] is None

    def test_single_sample_has_no_mean_speed(self):
        """Without time passing, there is no mean speed."""
        assert _evaluate([{"x": 0.0, "vel_lon": 5.0}])["distance"]["mean_speed"] is None
