# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for the topics, the metrics, the configuration and the result store of the Evaluation base class.

An evaluation reads the topics of a system under test and, if it compares them with a reference,
ground-truth topics. It is either composed of metric building blocks, which are selected and
configured per evaluation, or computes its metrics itself. Metrics are reported on three levels:
for every single sample, aggregated over the samples of each scene of the dataset, and aggregated
over all evaluated samples. Minimal evaluations whose metrics are trivial to predict are used, so
that the tests cover the topics, the composition and the grouping and not a metric definition.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest
from autonomy_evaluation.evaluations.Evaluation import Evaluation, flatten_config, unflatten_config
from autonomy_evaluation.evaluations.Metric import Aggregation, CONTEXT, EVALUATED, Input, Metric, MetricGroup, Sample


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
            "sampling": {"num_sequences": 1, "period": None, "num_gaps": 0, "max_interval": None, "num_jumps": 0},
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


# --- evaluations composed of metric building blocks ---

# Number of times the objects of a sample were derived, to check that the metrics share them
_DERIVATIONS: List[str] = []


def _objects(sample: Sample) -> List[float]:
    """Derive the sorted object positions of a sample, which all metrics below share."""

    def derive() -> List[float]:
        _DERIVATIONS.append(sample.sample_id)
        return sorted(sample["objects"])

    return sample.memo("objects", derive)


class _NumObjects(Metric):
    """Counts the objects of the samples."""

    NAME = "num_objects"

    def inputs(self) -> Dict[str, Input]:
        """Read the object positions."""
        return {"objects": Input(object, EVALUATED)}

    def compute_sample(self, sample: Sample) -> int:
        """Count the objects of a sample."""
        return len(_objects(sample))

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Sum the objects of all samples."""
        return {"_value_": sum(aggregation.records(self.name))}


class _ClosestObject(Metric):
    """Reports the distance of the closest object, scaled by its configuration."""

    NAME = "closest"

    def inputs(self) -> Dict[str, Input]:
        """Read the ego position and the object positions."""
        return {"ego": Input(object, EVALUATED), "objects": Input(object, EVALUATED)}

    def default_config(self) -> Dict[str, Any]:
        """Scale the distances by 1 by default."""
        return {"scale": 1.0}

    def compute_sample(self, sample: Sample) -> float:
        """Report the scaled distance of the closest object of a sample."""
        return min(abs(position - sample["ego"]) for position in _objects(sample)) * self.config["scale"]

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Report the closest distance of all samples."""
        return {"_value_": min(aggregation.records(self.name))}


class _FarthestObject(Metric):
    """Reports the distance of the farthest object, reading an optional map it does not need."""

    NAME = "farthest"

    def inputs(self) -> Dict[str, Input]:
        """Read the ego position, the object positions and, if published, a map."""
        return {
            "ego": Input(object, EVALUATED),
            "objects": Input(object, EVALUATED),
            "map": Input(object, CONTEXT, optional=True),
        }

    def compute_sample(self, sample: Sample) -> float:
        """Report the distance of the farthest object of a sample."""
        return max(abs(position - sample["ego"]) for position in _objects(sample))

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Report the farthest distance of all samples."""
        return {"_value_": max(aggregation.records(self.name))}

    def visualization_outputs(self) -> Dict[str, Any]:
        """Publish the farthest distance."""
        return {"farthest": float}

    def visualize_sample(self, sample: Sample) -> Dict[str, Any]:
        """Publish the farthest distance of a sample."""
        return {"farthest": self.compute_sample(sample)}


class _ObjectsPerSample(Metric):
    """Averages the number of objects per sample, building on the number of objects."""

    NAME = "objects_per_sample"

    def inputs(self) -> Dict[str, Input]:
        """Read the object positions, which the number of objects is counted from."""
        return {"objects": Input(object, EVALUATED)}

    def requires(self):
        """Build on the number of objects."""
        return ("num_objects",)

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Divide the number of objects by the number of samples."""
        return {"_value_": aggregation.results["num_objects"]["_value_"] / len(aggregation.entries)}


class _EgoSpeed(Metric):
    """Reports the highest speed of the ego position, which depends on the order and the stamps of the samples."""

    NAME = "speed"

    def inputs(self) -> Dict[str, Input]:
        """Read the ego position."""
        return {"ego": Input(object, EVALUATED)}

    def compute_sample(self, sample: Sample) -> tuple:
        """Record the stamp and the ego position of a sample."""
        return sample.stamp, sample["ego"]

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Differentiate the ego positions of each sequence of samples."""
        speeds = []
        for sequence in aggregation.sequences():
            records = aggregation.records(self.name, sequence)
            speeds += [
                abs(ego - previous_ego) / (stamp - previous)
                for (previous, previous_ego), (stamp, ego) in zip(records, records[1:])
            ]
        return {"_value_": max(speeds, default=None)}


class _Census(Metric):
    """Internal metric counting the samples, which other metrics require."""

    NAME = "census"
    INTERNAL = True

    def inputs(self) -> Dict[str, Input]:
        """Read the object positions."""
        return {"objects": Input(object, EVALUATED)}

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Count the samples."""
        return {"_value_": len(aggregation.entries)}


class _SampleShare(Metric):
    """Reports the share of the scene's samples among all recorded samples, building on the internal census."""

    NAME = "share"

    def inputs(self) -> Dict[str, Input]:
        """Read the object positions."""
        return {"objects": Input(object, EVALUATED)}

    def requires(self):
        """Build on the census."""
        return ("census",)

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Divide the samples by all recorded samples."""
        return {"_value_": aggregation.results["census"]["_value_"] / len(aggregation.recorded_entries)}


class _SelfRequiring(_NumObjects):
    """Metric that requires itself, which cannot be computed."""

    def requires(self):
        """Require itself."""
        return ("num_objects",)


class _ObjectsAsContext(_NumObjects):
    """Metric declaring the objects in another role than the other metrics do."""

    NAME = "objects_as_context"

    def inputs(self) -> Dict[str, Input]:
        """Read the object positions as context."""
        return {"objects": Input(object, CONTEXT)}


class _ComposedEvaluation(Evaluation):
    """Evaluation composed of the minimal metrics above, with a group of proximity metrics."""

    HEADLINE_METRIC = "num_objects"

    def __init__(self, blocks=None, evaluation_config=None) -> None:
        """Compose the given metrics, the default ones if none are given."""
        super().__init__(name="composed", description="composed of metric building blocks")
        self._blocks = blocks
        self._evaluation_config = {"unit": "m"} if evaluation_config is None else evaluation_config

    def metrics(self):
        """Compose the evaluation of the given or the default metrics."""
        if self._blocks is not None:
            return self._blocks
        return [_NumObjects(), MetricGroup("proximity", [_ClosestObject(), _FarthestObject()]), _ObjectsPerSample(), _EgoSpeed()]

    def default_config(self) -> Dict[str, Any]:
        """Configure the evaluation-level entries given on construction."""
        return dict(self._evaluation_config)


def _record(evaluation: Evaluation, samples) -> Evaluation:
    """Record ``(scene_id, stamp, ego, objects)`` samples in a composed evaluation."""
    for index, (scene_id, stamp, ego, objects) in enumerate(samples):
        evaluation.record_sample(sample_id=str(index), scene_id=scene_id, stamp=stamp, ego=ego, objects=objects, map=None)
    return evaluation


_SAMPLES = [("a", 0.0, 0.0, [1.0, 4.0]), ("a", 0.5, 1.0, [3.0]), ("b", 1.0, 5.0, [2.0, 9.0])]


class TestComposedEvaluation:
    """Tests computing, selecting and reporting the metrics an evaluation is composed of."""

    def setup_method(self):
        """Forget the objects derived by previous tests."""
        _DERIVATIONS.clear()

    def test_reports_all_metrics_by_default(self):
        """Without a selection, every metric is computed and reported, those of a group nested under its name."""
        metrics = _record(_ComposedEvaluation(), _SAMPLES).finalize()["metrics"]

        assert metrics == {
            "_value_": 5,
            "num_objects": {"_value_": 5},
            "proximity": {"closest": {"_value_": 1.0}, "farthest": {"_value_": 4.0}},
            "objects_per_sample": {"_value_": 5 / 3},
            "speed": {"_value_": 2.0},
        }

    def test_metrics_are_aggregated_per_scene(self):
        """The metrics of a scene are aggregated over its samples only."""
        scenes = _record(_ComposedEvaluation(), _SAMPLES).finalize()["scenes"]

        assert scenes["a"]["metrics"]["num_objects"] == {"_value_": 3}
        assert scenes["b"]["metrics"]["proximity"]["farthest"] == {"_value_": 4.0}

    def test_selecting_a_group_selects_its_metrics(self):
        """A group is selected by its name."""
        evaluation = _ComposedEvaluation()
        evaluation.configure({"metrics": ["proximity"]})

        assert list(evaluation.selected_metrics()) == ["proximity.closest", "proximity.farthest"]

    def test_selecting_a_metric_of_a_group(self):
        """A single metric of a group is selected as '<group>.<metric>'."""
        evaluation = _ComposedEvaluation()
        evaluation.configure({"metrics": ["proximity.closest"]})

        assert _record(evaluation, _SAMPLES).finalize()["metrics"] == {
            "_value_": None,
            "proximity": {"closest": {"_value_": 1.0}},
        }

    def test_reads_only_the_topics_of_the_computed_metrics(self):
        """An evaluation subscribes to the inputs of the metrics it computes, the launch file remaps all of them."""
        evaluation = _ComposedEvaluation()
        evaluation.configure({"metrics": ["num_objects"]})

        assert list(evaluation.inputs()) == ["objects"]
        assert list(evaluation.available_inputs()) == ["objects", "ego", "map"]

    def test_inputs_keep_their_roles_and_optionality(self):
        """The inputs carry the role and optionality their metrics declare."""
        inputs = _ComposedEvaluation().inputs()

        assert inputs["objects"] == Input(object, EVALUATED)
        assert inputs["map"] == Input(object, CONTEXT, optional=True)

    def test_required_metric_is_computed_but_not_reported(self):
        """A metric another one builds on is computed first, but only reported if selected itself."""
        evaluation = _ComposedEvaluation()
        evaluation.configure({"metrics": ["objects_per_sample"]})

        metrics = _record(evaluation, _SAMPLES).finalize()["metrics"]

        assert list(evaluation.computed_metrics()) == ["num_objects", "objects_per_sample"]
        assert metrics == {"_value_": 5, "objects_per_sample": {"_value_": 5 / 3}}

    def test_headline_is_only_reported_while_its_metric_is_computed(self):
        """Without its headline metric, an evaluation reports no overall value."""
        evaluation = _ComposedEvaluation()
        evaluation.configure({"metrics": ["proximity"]})

        assert _record(evaluation, _SAMPLES).finalize()["metrics"]["_value_"] is None

    def test_internal_metric_is_computed_for_others_only(self):
        """An internal metric provides data for others, but is neither selectable nor reported."""
        evaluation = _ComposedEvaluation(blocks=[_Census(), _SampleShare()])

        results = _record(evaluation, _SAMPLES).finalize()

        assert evaluation.config["metrics"] == ["share"]
        assert list(evaluation.computed_metrics()) == ["census", "share"]
        assert results["metrics"] == {"_value_": None, "share": {"_value_": 1.0}}
        assert results["scenes"]["a"]["metrics"]["share"] == {"_value_": 2 / 3}
        with pytest.raises(ValueError, match="has no metric 'census'"):
            evaluation.configure({"metrics": ["census"]})

    def test_metrics_share_the_data_they_derive_from_a_sample(self):
        """All metrics of a sample derive its objects only once."""
        _record(_ComposedEvaluation(), _SAMPLES[:1])

        assert _DERIVATIONS == ["0"]

    def test_visualization_reuses_the_recorded_sample(self):
        """Visualizing the sample just recorded reuses the data the metrics derived from it."""
        evaluation = _ComposedEvaluation()
        messages = {"ego": 0.0, "objects": [1.0, 4.0], "map": None}
        evaluation.record_sample(sample_id="0", stamp=0.0, **messages)

        visualization = evaluation.visualize_sample(sample_id="0", **messages)

        assert evaluation.visualization_outputs() == {"farthest": float}
        assert visualization == {"farthest": 4.0}
        assert _DERIVATIONS == ["0"]

    def test_temporal_metric_is_computed_per_sequence(self):
        """The ego jumps between the scenes, which is no speed, as the scenes are no sequence."""
        evaluation = _record(_ComposedEvaluation(), _SAMPLES)

        assert evaluation.finalize()["metrics"]["speed"] == {"_value_": 2.0}

    @pytest.mark.parametrize(
        "selection, reason",
        [(["unknown"], "has no metric 'unknown'"), ([], "selects no metric"), (["proximity.unknown"], "has no metric")],
    )
    def test_rejects_a_selection_of_no_metric(self, selection, reason):
        """A selection must name metrics or groups of the evaluation."""
        with pytest.raises(ValueError, match=reason):
            _ComposedEvaluation().configure({"metrics": selection})

    def test_rejects_a_requirement_of_an_unknown_metric(self):
        """A metric can only build on metrics of its evaluation."""
        with pytest.raises(ValueError, match="requires unknown metric 'num_objects'"):
            _ComposedEvaluation(blocks=[_ObjectsPerSample()]).configure()

    def test_rejects_a_metric_requiring_itself(self):
        """A metric cannot build on its own result."""
        with pytest.raises(ValueError, match="requires itself"):
            _ComposedEvaluation(blocks=[_SelfRequiring()]).configure()

    def test_rejects_metrics_of_the_same_name(self):
        """Each metric is reported and configured under a name of its own."""
        with pytest.raises(ValueError, match="declares 'num_objects' twice"):
            _ComposedEvaluation(blocks=[_NumObjects(), _NumObjects()]).available_metrics()

    def test_rejects_an_input_declared_differently_by_two_metrics(self):
        """Metrics reading the same topic must agree on its type, role and optionality."""
        with pytest.raises(ValueError, match="reads input 'objects' as"):
            _ComposedEvaluation(blocks=[_NumObjects(), _ObjectsAsContext()]).inputs()

    def test_rejects_visualization_outputs_of_the_same_name(self):
        """Two metrics cannot publish their visualization on the same topic."""

        class _OtherFarthest(_FarthestObject):
            NAME = "other_farthest"

        with pytest.raises(ValueError, match="publishes visualization output 'farthest' twice"):
            _ComposedEvaluation(blocks=[_FarthestObject(), _OtherFarthest()]).visualization_outputs()


class TestConfiguration:
    """Tests configuring an evaluation and its metrics."""

    def test_default_configuration_holds_the_selection_and_every_metric(self):
        """The configuration selects all metrics and holds the entries of each, nested under its group."""
        assert _ComposedEvaluation().config_defaults() == {
            "max_sample_interval": 1.0,
            "max_jump_speed": 100.0,
            "limits": [],
            "unit": "m",
            "metrics": ["num_objects", "proximity", "objects_per_sample", "speed"],
            "proximity": {"closest": {"scale": 1.0}},
        }

    def test_configures_a_metric(self):
        """A configured entry reaches its metric, an integer is accepted for a floating-point entry."""
        evaluation = _ComposedEvaluation()
        evaluation.configure({"proximity": {"closest": {"scale": 2}}})

        assert evaluation.available_metrics()["proximity.closest"].config == {"scale": 2.0}
        assert _record(evaluation, _SAMPLES).finalize()["metrics"]["proximity"]["closest"] == {"_value_": 2.0}

    def test_unconfigured_entries_keep_their_defaults(self):
        """Only the configured entries change."""
        effective = _ComposedEvaluation().configure({"max_sample_interval": 2.0})

        assert effective["max_sample_interval"] == 2.0
        assert effective["proximity"] == {"closest": {"scale": 1.0}}

    def test_evaluation_is_configured_with_its_defaults_on_first_use(self):
        """An evaluation that is never configured uses its default configuration."""
        assert _ComposedEvaluation().config == _ComposedEvaluation().config_defaults()

    @pytest.mark.parametrize(
        "config, reason",
        [
            ({"proximity": {"closest": {"offset": 1.0}}}, r"has no configuration \['proximity.closest.offset'\]"),
            ({"max_sample_interval": "long"}, "to be of type float"),
            ({"max_sample_interval": True}, "to be of type float"),
            ({"metrics": "proximity"}, "to be a list"),
            ({"proximity": 1.0}, "to be nested entries"),
        ],
    )
    def test_rejects_an_invalid_configuration(self, config, reason):
        """Unknown entries and entries of another type than their default are rejected."""
        with pytest.raises(ValueError, match=reason):
            _ComposedEvaluation().configure(config)

    def test_rejects_an_evaluation_level_entry_named_like_a_metric(self):
        """Evaluation-level entries and metrics share the namespace of the configuration."""
        with pytest.raises(ValueError, match="next to metrics of the same name"):
            _ComposedEvaluation(evaluation_config={"speed": 1.0}).config_defaults()

    def test_evaluation_without_building_blocks_is_configured_too(self):
        """An evaluation computing its metrics itself has the configuration every evaluation has."""
        assert CountingEvaluation().config == {"max_sample_interval": 1.0, "max_jump_speed": 100.0, "limits": []}

    def test_flattened_configuration_names_nested_entries_by_their_joined_keys(self):
        """ROS parameters name nested entries by their keys joined with '.', lists are values."""
        config = {"metrics": ["ap"], "ap": {"min_precision": 0.1, "matching_thresholds": [0.5, 1.0]}}

        flattened = flatten_config(config, "object_detection_3d.")

        assert flattened == {
            "object_detection_3d.metrics": ["ap"],
            "object_detection_3d.ap.min_precision": 0.1,
            "object_detection_3d.ap.matching_thresholds": [0.5, 1.0],
        }
        assert unflatten_config(flatten_config(config)) == config


class TestResultsDescription:
    """Tests describing how the results were computed next to the metrics."""

    def test_results_describe_the_inputs_the_configuration_and_the_sampling(self):
        """The results name the topic and role of each input and hold the configuration and the sampling."""
        evaluation = _ComposedEvaluation()
        evaluation.input_topics = {"objects": "/simulation/objects"}

        results = _record(evaluation, _SAMPLES).finalize()

        assert results["inputs"]["objects"] == {
            "type": "object",
            "role": "evaluated",
            "optional": False,
            "latest": False,
            "topic": "/simulation/objects",
        }
        assert results["inputs"]["map"]["role"] == "context"
        assert results["config"] == evaluation.config
        assert results["sampling"] == {"num_sequences": 2, "period": 0.5, "num_gaps": 0, "max_interval": 0.5, "num_jumps": 0}

    def test_inputs_of_an_evaluation_without_building_blocks_have_roles(self):
        """Inputs are evaluated, ground truth is the reference."""
        inputs = _evaluation_of([("0", "scene_a", 1, 1)]).finalize()["inputs"]

        assert {name: description["role"] for name, description in inputs.items()} == {
            "prediction": "evaluated",
            "label": "reference",
        }


class TestLimits:
    """Tests judging the metrics of an evaluation and of each scene by optional limits."""

    def test_results_hold_no_verdict_without_limits(self):
        """Limits are optional, without them the results are the same as before."""
        results = _record(_ComposedEvaluation(), _SAMPLES).finalize()

        assert "limits" not in results
        assert all("limits" not in scene for scene in results["scenes"].values())

    def test_limits_are_checked_on_all_samples_and_on_each_scene(self):
        """Scene 'a' has three objects and passes, scene 'b' has two objects and fails, all samples have five."""
        evaluation = _ComposedEvaluation()
        evaluation.configure({"limits": ["num_objects >= 3"]})

        results = _record(evaluation, _SAMPLES).finalize()

        assert results["limits"]["passed"] is True
        assert results["limits"]["checks"] == [{"limit": "num_objects >= 3", "value": 5.0, "passed": True}]
        assert results["scenes"]["a"]["limits"]["passed"] is True
        assert results["scenes"]["b"]["limits"]["passed"] is False

    def test_limits_may_judge_the_headline_value(self):
        """The headline value of the evaluation is its '_value_'."""
        evaluation = _ComposedEvaluation()
        evaluation.configure({"limits": ["_value_ < 5"]})

        assert _record(evaluation, _SAMPLES).finalize()["limits"]["passed"] is False

    @pytest.mark.parametrize(
        "limits, error",
        [
            (["num_objects >> 3"], "is not of the form"),
            (["speed < 2"], "names none of the selected metrics"),  # not selected below
            (["unknown.metric < 2"], "names none of the selected metrics"),
        ],
    )
    def test_rejects_limits_it_cannot_check(self, limits, error):
        """A malformed limit, or one on a metric that is not computed, is rejected when configuring."""
        with pytest.raises(ValueError, match=error):
            _ComposedEvaluation().configure({"metrics": ["num_objects"], "limits": limits})
