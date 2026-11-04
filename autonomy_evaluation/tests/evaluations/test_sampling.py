# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for splitting the recorded samples into contiguous sequences and describing their gaps.

Metrics that depend on the order of the samples, e.g. a jerk or a post-encroachment time, are
computed per sequence, which must neither span two scenes nor bridge a long interruption. Gaps
within a sequence, i.e. missing samples, are reported, as they make such metrics less accurate.
"""

from __future__ import annotations

import pytest
from autonomy_evaluation.evaluations.sampling import sampling_period, sampling_statistics, split_into_sequences


def _entries(*samples) -> list:
    """Build recorded samples from ``(scene_id, stamp)`` pairs."""
    return [{"sample_id": str(index), "scene_id": scene_id, "stamp": stamp} for index, (scene_id, stamp) in enumerate(samples)]


def _sample_ids(sequences) -> list:
    """Reduce sequences to the IDs of their samples."""
    return [[entry["sample_id"] for entry in sequence] for sequence in sequences]


class TestSplitIntoSequences:
    """Tests deciding which consecutive samples form a contiguous sequence."""

    def test_samples_of_a_scene_form_a_sequence(self):
        """Regularly stamped samples of a scene are one sequence."""
        entries = _entries(("a", 0.0), ("a", 0.1), ("a", 0.2))

        assert _sample_ids(split_into_sequences(entries, max_sample_interval=1.0)) == [["0", "1", "2"]]

    def test_a_new_scene_starts_a_new_sequence(self):
        """Samples of different scenes never form a sequence, however close their stamps are."""
        entries = _entries(("a", 0.0), ("a", 0.1), ("b", 0.2))

        assert _sample_ids(split_into_sequences(entries, max_sample_interval=1.0)) == [["0", "1"], ["2"]]

    def test_a_scene_recorded_earlier_starts_a_new_sequence(self):
        """A dataset may continue with a scene recorded before the previous one, whose stamps jump back."""
        entries = _entries((None, 100.0), (None, 100.1), (None, 50.0), (None, 50.1))

        assert _sample_ids(split_into_sequences(entries, max_sample_interval=1.0)) == [["0", "1"], ["2", "3"]]

    def test_samples_recorded_out_of_order_are_put_in_stamp_order(self):
        """A sample whose messages were matched within a tolerance may complete after a later sample."""
        entries = _entries(("a", 0.0), ("a", 0.1), ("a", 0.05), ("a", 0.15), ("a", 0.25), ("a", 0.2))

        assert _sample_ids(split_into_sequences(entries, max_sample_interval=1.0)) == [["0", "2", "1", "3", "5", "4"]]
        assert sampling_statistics(entries, 1.0)["num_gaps"] == 0

    def test_a_restart_starts_a_new_sequence(self):
        """A stamp further back than the maximum interval, e.g. of a restarted simulation, is not reordered."""
        entries = _entries(("a", 10.0), ("a", 10.1), ("a", 0.0), ("a", 0.1), ("a", 10.05))

        assert _sample_ids(split_into_sequences(entries, max_sample_interval=1.0)) == [["0", "1"], ["2", "3"], ["4"]]

    def test_a_long_interruption_starts_a_new_sequence(self):
        """Samples further apart than the maximum interval belong to different sequences."""
        entries = _entries(("a", 0.0), ("a", 0.5), ("a", 2.0))

        assert _sample_ids(split_into_sequences(entries, max_sample_interval=1.0)) == [["0", "1"], ["2"]]

    def test_a_jump_of_the_position_starts_a_new_sequence(self):
        """A simulation teleporting the ego vehicle 350 m within 0.05 s, e.g. to the start of a scenario, is no drive."""
        entries = _entries(("a", 0.0), ("a", 0.05), ("a", 0.1), ("a", 0.15))
        for entry, x in zip(entries, (0.0, 0.5, 350.5, 351.0)):
            entry["position"] = (x, 0.0)

        assert _sample_ids(split_into_sequences(entries, 1.0, max_jump_speed=100.0)) == [["0", "1"], ["2", "3"]]
        assert sampling_statistics(entries, 1.0, max_jump_speed=100.0)["num_jumps"] == 1

    def test_samples_without_position_never_jump(self):
        """An evaluation that reports no positions splits its samples by time and scene only."""
        entries = _entries(("a", 0.0), ("a", 0.05))
        entries[0]["position"] = (0.0, 0.0)

        assert _sample_ids(split_into_sequences(entries, 1.0, max_jump_speed=100.0)) == [["0", "1"]]

    def test_samples_without_stamp_continue_the_sequence(self):
        """Without stamps no interval is known, so the samples of a scene stay together."""
        entries = _entries(("a", None), ("a", None), ("b", None))

        assert _sample_ids(split_into_sequences(entries, max_sample_interval=1.0)) == [["0", "1"], ["2"]]

    def test_no_samples_form_no_sequence(self):
        """An evaluation without samples has no sequence."""
        assert split_into_sequences([], max_sample_interval=1.0) == []


class TestSamplingStatistics:
    """Tests describing how regularly the samples were evaluated."""

    def test_sampling_period_is_the_median_interval(self):
        """The median interval is the period the samples were meant to be published at, despite gaps."""
        assert sampling_period(_entries(("a", 0.0), ("a", 0.1), ("a", 0.2), ("a", 0.5))) == pytest.approx(0.1)

    def test_sampling_period_needs_two_samples(self):
        """A single sample has no interval."""
        assert sampling_period(_entries(("a", 0.0))) is None

    def test_reports_the_gaps_within_sequences(self):
        """Missing samples within a sequence are gaps, an interruption between sequences is not."""
        entries = _entries(("a", 0.0), ("a", 0.1), ("a", 0.4), ("a", 0.5), ("a", 0.6), ("a", 5.0), ("a", 5.1))

        statistics = sampling_statistics(entries, max_sample_interval=1.0)

        assert statistics["num_sequences"] == 2
        assert statistics["num_gaps"] == 1
        assert statistics["period"] == pytest.approx(0.1)
        assert statistics["max_interval"] == pytest.approx(0.3)

    def test_regular_samples_have_no_gaps(self):
        """Samples evaluated at their period leave no gap."""
        entries = _entries(*(("a", index * 0.1) for index in range(10)))

        assert sampling_statistics(entries, max_sample_interval=1.0)["num_gaps"] == 0

    def test_samples_without_interval_have_no_period(self):
        """Without two samples in a sequence, no period is known."""
        assert sampling_statistics(_entries(("a", 0.0), ("b", 0.1)), max_sample_interval=1.0) == {
            "num_sequences": 2,
            "period": None,
            "num_gaps": 0,
            "max_interval": None,
            "num_jumps": 0,
        }
