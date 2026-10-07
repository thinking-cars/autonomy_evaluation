# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for the topics and the result store of the Evaluation base class.

An evaluation reads the topics of a system under test and, if it compares them with a reference,
ground-truth topics. Metrics are reported on three levels: for every single sample, aggregated over
the samples of each scene of the dataset, and aggregated over all evaluated samples. Minimal
evaluations whose metrics are trivial to predict are used, so that the tests cover the topics and
the grouping and not a metric definition.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest
from autonomy_evaluation.evaluations.Evaluation import Evaluation


class CountingEvaluation(Evaluation):
    """Minimal evaluation counting the objects of a sample and summing them up when aggregating."""

    def __init__(self) -> None:
        """Name the evaluation."""
        super().__init__(name="counting", description="counts objects")

    def required_inputs(self) -> Dict[str, Any]:
        """Declare the evaluated input, which this evaluation does not read from ROS messages."""
        return {"prediction": object}

    def required_ground_truth(self) -> Dict[str, Any]:
        """Declare the ground truth the input is compared with."""
        return {"label": object}

    def compute_sample_metrics(self, prediction: Any, label: Any, sample_id: Optional[str] = None) -> Dict[str, Any]:
        """Report the given prediction and label counts of a single sample."""
        return {"num_predictions": prediction, "num_labels": label}

    def compute_aggregated_metrics(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Sum the counts of the given samples."""
        return {
            "num_predictions": sum(entry["metrics"]["num_predictions"] for entry in sample_results),
            "num_labels": sum(entry["metrics"]["num_labels"] for entry in sample_results),
        }


class ClosestObjectEvaluation(Evaluation):
    """Minimal evaluation of inputs only, reporting the closest object without any ground truth."""

    def __init__(self) -> None:
        """Name the evaluation."""
        super().__init__(name="closest_object")

    def required_inputs(self) -> Dict[str, Any]:
        """Declare the ego position and the object distances, as a closed-loop planner is evaluated on."""
        return {"ego_position": object, "object_positions": object}

    def compute_sample_metrics(self, ego_position: float, object_positions: List[float], sample_id: Optional[str] = None):
        """Report the distance of the closest object of a single sample."""
        return {"min_distance": min(abs(position - ego_position) for position in object_positions)}

    def compute_aggregated_metrics(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Report the closest distance of all given samples."""
        return {"min_distance": min(entry["metrics"]["min_distance"] for entry in sample_results)}


class _TopicsEvaluation(CountingEvaluation):
    """Counting evaluation with freely declared topics, to test their validation."""

    def __init__(
        self,
        inputs: Dict[str, Any],
        ground_truth: Dict[str, Any],
        derived_topics=None,
        optional_ground_truth=None,
    ) -> None:
        """Declare the given topics."""
        super().__init__()
        self._inputs, self._ground_truth, self._derived_topics = inputs, ground_truth, derived_topics or {}
        self._optional_ground_truth = optional_ground_truth or {}

    def required_inputs(self) -> Dict[str, Any]:
        """Declare the given inputs."""
        return self._inputs

    def required_ground_truth(self) -> Dict[str, Any]:
        """Declare the given ground truth."""
        return self._ground_truth

    def optional_ground_truth(self) -> Dict[str, Any]:
        """Declare the given optional ground truth."""
        return self._optional_ground_truth

    def derived_topics(self):
        """Declare the given derived topics."""
        return self._derived_topics


def _evaluation_of(samples) -> CountingEvaluation:
    """Record ``(sample_id, scene_id, num_predictions, num_labels)`` samples in an evaluation."""
    evaluation = CountingEvaluation()
    for sample_id, scene_id, num_predictions, num_labels in samples:
        evaluation.record_sample(prediction=num_predictions, label=num_labels, sample_id=sample_id, scene_id=scene_id)
    return evaluation


class TestTopics:
    """Tests declaring the topics an evaluation reads."""

    def test_inputs_are_followed_by_the_ground_truth(self):
        """The topics are passed to the evaluation in the order inputs first, ground truth second."""
        assert list(CountingEvaluation().all_inputs()) == ["prediction", "label"]

    def test_evaluation_of_inputs_only_needs_no_ground_truth(self):
        """An evaluation computing its metrics from the system under test alone declares no ground truth."""
        evaluation = ClosestObjectEvaluation()

        assert evaluation.required_ground_truth() == {}
        assert evaluation.derived_topics() == {}
        assert list(evaluation.all_inputs()) == ["ego_position", "object_positions"]

    def test_rejects_a_topic_declared_as_input_and_as_ground_truth(self):
        """A topic name identifies a single message of a sample, so it cannot play both roles."""
        with pytest.raises(ValueError, match="as input and as ground truth"):
            _TopicsEvaluation({"objects": object}, {"objects": object}).all_inputs()

    def test_rejects_an_evaluation_without_topics(self):
        """An evaluation that reads no topic would never evaluate a sample."""
        with pytest.raises(ValueError, match="no topic"):
            _TopicsEvaluation({}, {}).all_inputs()

    @pytest.mark.parametrize(
        "derived_topics",
        [{"meta_info": ("label", "/meta_info")}, {"label": ("objects", "/meta_info")}, {"label": ("label", "/meta_info")}],
    )
    def test_rejects_a_derived_topic_of_unknown_inputs(self, derived_topics):
        """A derived topic and the topic it is derived from must both be read by the evaluation."""
        with pytest.raises(ValueError, match="derives the topic"):
            _TopicsEvaluation({"prediction": object}, {"label": object}, derived_topics).all_inputs()

    def test_optional_ground_truth_follows_the_required_topics(self):
        """Optional ground truth is read next to the required topics, after them."""
        evaluation = _TopicsEvaluation(
            {"prediction": object}, {"label": object}, optional_ground_truth={"label_meta_info": object}
        )

        assert list(evaluation.all_inputs()) == ["prediction", "label", "label_meta_info"]

    def test_rejects_a_topic_declared_as_required_and_as_optional(self):
        """A topic is either waited for or not, so it cannot be required and optional at once."""
        with pytest.raises(ValueError, match="as ground truth and as optional ground truth"):
            _TopicsEvaluation({"prediction": object}, {"label": object}, optional_ground_truth={"label": object}).all_inputs()

    def test_rejects_an_evaluation_of_optional_topics_only(self):
        """A sample is only evaluated once its required topics have been received, so at least one is needed."""
        with pytest.raises(ValueError, match="no topic"):
            _TopicsEvaluation({}, {}, optional_ground_truth={"label": object}).all_inputs()

    def test_optional_topic_may_be_derived_from_a_required_one(self):
        """Meta information that not every dataset publishes follows the topic of the labels it belongs to."""
        evaluation = _TopicsEvaluation(
            {"prediction": object},
            {"label": object},
            derived_topics={"label_meta_info": ("label", "/meta_info")},
            optional_ground_truth={"label_meta_info": object},
        )

        assert "label_meta_info" in evaluation.all_inputs()


class TestSampleResults:
    """Tests recording samples with the scene of the dataset they belong to."""

    # The recorded samples themselves are no longer reported alongside the aggregated
    # results, while 'sample_results' is commented out in Evaluation.finalize()
    # def test_records_sample_with_its_scene(self):
    #     """A recorded sample keeps its ID, its scene and its metrics."""
    #     evaluation = _evaluation_of([("0", "scene_a", 2, 3)])
    #
    #     assert evaluation.finalize()["sample_results"] == [
    #         {"sample_id": "0", "scene_id": "scene_a", "metrics": {"num_predictions": 2, "num_labels": 3}}
    #     ]

    def test_scene_can_be_set_after_the_sample_was_recorded(self):
        """An evaluation loop that learns the scene late sets it on the returned entry."""
        evaluation = CountingEvaluation()

        entry = evaluation.record_sample(prediction=1, label=1, sample_id="0")
        assert entry["scene_id"] is None
        entry["scene_id"] = "scene_a"

        assert evaluation.sample_results_by_scene() == {"scene_a": [entry]}


class TestFinalize:
    """Tests aggregating the recorded samples per scene and over the whole evaluation."""

    def test_aggregates_per_sample_scene_and_evaluation(self):
        """Metrics are reported for every sample, every scene and all samples together."""
        results = _evaluation_of(
            [
                ("0", "scene_a", 1, 1),
                ("1", "scene_a", 2, 3),
                ("2", "scene_b", 4, 5),
            ]
        ).finalize()

        assert results["num_samples"] == 3
        assert results["num_scenes"] == 2
        assert results["metrics"] == {"num_predictions": 7, "num_labels": 9}
        assert results["scenes"]["scene_a"] == {
            "num_samples": 2,
            "sample_ids": ["0", "1"],
            "metrics": {"num_predictions": 3, "num_labels": 4},
        }
        assert results["scenes"]["scene_b"]["metrics"] == {"num_predictions": 4, "num_labels": 5}
        # The metrics of the single samples are no longer reported alongside the aggregated
        # results, while 'sample_results' is commented out in Evaluation.finalize()
        # assert [entry["metrics"] for entry in results["sample_results"]] == [
        #     {"num_predictions": 1, "num_labels": 1},
        #     {"num_predictions": 2, "num_labels": 3},
        #     {"num_predictions": 4, "num_labels": 5},
        # ]

    def test_groups_samples_of_a_scene_that_are_not_recorded_consecutively(self):
        """Samples are grouped by their scene, not by the order they were recorded in."""
        results = _evaluation_of(
            [
                ("0", "scene_a", 1, 0),
                ("1", "scene_b", 2, 0),
                ("2", "scene_a", 4, 0),
            ]
        ).finalize()

        assert results["scenes"]["scene_a"]["sample_ids"] == ["0", "2"]
        assert results["scenes"]["scene_a"]["metrics"]["num_predictions"] == 5
        assert results["scenes"]["scene_b"]["sample_ids"] == ["1"]

    def test_samples_without_a_scene_are_only_aggregated_over_the_evaluation(self):
        """A sample that cannot be attributed to a scene still counts for the whole evaluation."""
        results = _evaluation_of([("0", "scene_a", 1, 0), ("1", None, 2, 0)]).finalize()

        assert results["num_samples"] == 2
        assert results["num_scenes"] == 1
        assert results["metrics"]["num_predictions"] == 3
        assert results["scenes"]["scene_a"]["metrics"]["num_predictions"] == 1

    def test_aggregates_an_evaluation_of_inputs_only(self):
        """Samples of an evaluation without ground truth are recorded from their inputs alone."""
        evaluation = ClosestObjectEvaluation()
        evaluation.record_sample(sample_id="0", scene_id="scene_a", ego_position=0.0, object_positions=[4.0, -2.5])
        evaluation.record_sample(sample_id="1", scene_id="scene_a", ego_position=1.0, object_positions=[4.0])

        results = evaluation.finalize()

        assert results["metrics"] == {"min_distance": 2.5}
        assert results["scenes"]["scene_a"]["num_samples"] == 2

    def test_reports_no_scene_without_recorded_scenes(self):
        """Samples recorded without a scene aggregate to no scene results at all."""
        results = _evaluation_of([("0", None, 1, 0)]).finalize()

        assert results["num_scenes"] == 0
        assert results["scenes"] == {}


class TestSaveResults:
    """Tests writing the results of all three levels to a JSON file."""

    def test_writes_sample_scene_and_evaluation_metrics(self, tmp_path):
        """The stored results hold the metrics of every sample, every scene and the evaluation."""
        evaluation = _evaluation_of([("0", "scene_a", 1, 1), ("1", "scene_b", 2, 2)])

        output_path = evaluation.save_results(str(tmp_path / "results" / "counting.json"))

        stored = json.loads(open(output_path).read())
        assert stored["metrics"] == {"num_predictions": 3, "num_labels": 3}
        assert sorted(stored["scenes"]) == ["scene_a", "scene_b"]
        # The single samples are no longer written alongside the aggregated
        # results, while 'sample_results' is commented out in Evaluation.finalize()
        # assert [entry["sample_id"] for entry in stored["sample_results"]] == ["0", "1"]

    def test_writes_previously_computed_results(self, tmp_path):
        """Results that have already been computed are written as they are."""
        evaluation = _evaluation_of([("0", "scene_a", 1, 1)])
        results = evaluation.finalize()

        output_path = evaluation.save_results(str(tmp_path / "counting.json"), results=results)

        assert json.loads(open(output_path).read())["metrics"] == results["metrics"]


class TestIncompleteResults:
    """Tests marking the results of an evaluation that did not process all samples."""

    def test_results_of_all_samples_are_complete(self):
        """Results aggregated after the last sample of the evaluation are marked as complete."""
        assert _evaluation_of([("0", "scene_a", 1, 1)]).finalize()["complete"] is True

    def test_interrupted_results_are_marked_incomplete(self):
        """Results aggregated before the last sample, e.g. after Ctrl-C, are marked as incomplete."""
        results = _evaluation_of([("0", "scene_a", 1, 1)]).finalize(complete=False)

        assert results["complete"] is False
        # the samples that were evaluated are still reported
        assert results["num_samples"] == 1
        assert results["metrics"] == {"num_predictions": 1, "num_labels": 1}

    def test_writes_incomplete_results_to_the_results_file(self, tmp_path):
        """The results file of an interrupted evaluation marks the results it holds as incomplete."""
        evaluation = _evaluation_of([("0", "scene_a", 1, 1)])

        output_path = evaluation.save_results(str(tmp_path / "counting.json"), complete=False)

        stored = json.loads(open(output_path).read())
        assert stored["complete"] is False
        assert stored["metrics"] == {"num_predictions": 1, "num_labels": 1}

    def test_written_results_keep_the_flag_they_were_finalized_with(self, tmp_path):
        """A given payload is written as it is, marked the way it was finalized."""
        evaluation = _evaluation_of([("0", "scene_a", 1, 1)])
        results = evaluation.finalize(complete=False)

        output_path = evaluation.save_results(str(tmp_path / "counting.json"), results=results)

        assert json.loads(open(output_path).read())["complete"] is False
