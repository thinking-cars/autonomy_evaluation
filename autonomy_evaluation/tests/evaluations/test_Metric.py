# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for the building blocks evaluations are composed of: inputs, samples and aggregations.

How an evaluation composes, selects, configures and aggregates its metrics is tested with the
Evaluation base class.
"""

from __future__ import annotations

import pytest
from autonomy_evaluation.evaluations.Metric import Aggregation, CONTEXT, EVALUATED, Input, REFERENCE, Sample


class TestInput:
    """Tests declaring the inputs a metric reads."""

    @pytest.mark.parametrize("role", [EVALUATED, REFERENCE, CONTEXT])
    def test_accepts_the_roles_of_an_input(self, role):
        """An input is evaluated, the reference it is compared with, or context needed to compute a metric."""
        assert Input(object, role).role == role

    def test_rejects_an_unknown_role(self):
        """A role outside of the known ones is a typo, e.g. 'ground_truth' instead of 'reference'."""
        with pytest.raises(ValueError, match="'ground_truth' is none of"):
            Input(object, "ground_truth")

    def test_inputs_are_alike_if_all_their_properties_are(self):
        """Metrics that read the same topic declare it alike, which is checked by comparing their inputs."""
        assert Input(object, REFERENCE, optional=True) == Input(object, REFERENCE, optional=True)
        assert Input(object, REFERENCE) != Input(object, CONTEXT)

    def test_only_inputs_of_single_samples_are_synchronized(self):
        """Static context and inputs passed with their latest message are not matched by their stamps."""
        assert Input(object, EVALUATED).synchronized
        assert not Input(object, CONTEXT, static=True).synchronized
        assert not Input(object, CONTEXT, latest=True).synchronized

    def test_rejects_a_static_input_with_a_latest_message(self):
        """A static input is no topic, so it has no messages."""
        with pytest.raises(ValueError, match="no latest message"):
            Input(object, CONTEXT, static=True, latest=True)


class TestSample:
    """Tests the messages of a sample and the data derived from them."""

    def test_messages_are_accessed_by_input_name(self):
        """A metric reads the message of an input by its name."""
        assert Sample({"objects": [1, 2]})["objects"] == [1, 2]

    def test_derived_data_is_computed_once(self):
        """Data several metrics derive from a sample, e.g. a matching, is computed by the first and reused by the others."""
        sample = Sample({})
        calls = []

        first = sample.memo("matching", lambda: calls.append(1) or "matched")
        second = sample.memo("matching", lambda: calls.append(1) or "matched again")

        assert first == second == "matched"
        assert len(calls) == 1

    def test_derived_data_is_distinguished_by_its_key(self):
        """Data derived with different parameters, e.g. at different thresholds, is kept apart."""
        sample = Sample({})

        assert sample.memo(("matches", 1.0), lambda: 1) == 1
        assert sample.memo(("matches", 2.0), lambda: 2) == 2


class TestAggregation:
    """Tests the samples metrics are aggregated over."""

    _ENTRIES = [
        {"sample_id": "0", "scene_id": "a", "stamp": 0.0, "metrics": {"count": 1}},
        {"sample_id": "1", "scene_id": "a", "stamp": 0.1, "metrics": {"count": 2}},
        {"sample_id": "2", "scene_id": "b", "stamp": 0.2, "metrics": {"count": 3}},
    ]

    def test_records_of_a_metric_follow_the_samples(self):
        """The records of a metric are returned in the order of the samples."""
        assert Aggregation(self._ENTRIES).records("count") == [1, 2, 3]

    def test_records_of_given_samples(self):
        """The records of a subset of the samples, e.g. of a sequence, are returned on request."""
        assert Aggregation(self._ENTRIES).records("count", self._ENTRIES[1:]) == [2, 3]

    def test_samples_are_split_into_sequences_by_scene(self):
        """Samples of different scenes never form a sequence."""
        sequences = Aggregation(self._ENTRIES, config={"max_sample_interval": 1.0}).sequences()

        assert [[entry["sample_id"] for entry in sequence] for sequence in sequences] == [["0", "1"], ["2"]]

    def test_samples_are_split_into_sequences_by_the_configured_interval(self):
        """Samples further apart than 'max_sample_interval' belong to different sequences."""
        entries = [{**entry, "scene_id": None} for entry in self._ENTRIES]

        assert len(Aggregation(entries, config={"max_sample_interval": 0.05}).sequences()) == 3

    def test_recorded_samples_default_to_the_aggregated_ones(self):
        """Without a separate store of all recorded samples, the aggregated samples are all of them."""
        assert Aggregation(self._ENTRIES).recorded_entries is self._ENTRIES
