# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Abstract base class for all evaluations of automated driving tasks.

Each evaluation defines which topics it reads and how to compute per-sample and
aggregated metrics from their messages. An evaluation may read the topics of a
system under test only, e.g. the ego state and the surrounding objects of a
closed-loop planner to compute its time to collision, or compare them with
ground-truth topics, e.g. the predictions of a perception algorithm with the
labels of a dataset. Ground truth that enriches an evaluation without being
necessary for it, e.g. dataset annotations that not every dataset publishes, is
declared as optional. Concrete subclasses must override the abstract methods.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from tf2_ros import BufferInterface


class Evaluation(ABC):
    """Meta-class (abstract base class) for the evaluations of automated driving tasks.

    An evaluation is responsible for:
    * declaring the topics it reads, split into the inputs from the system
      under test and, if it compares them with a reference, the required and
      optional ground truth,
    * computing per-sample metrics from the messages of these topics,
    * aggregating per-sample metrics into scene-level and dataset-level metrics,
    * persisting results to JSON.

    The messages of all topics that belong to the same sample are passed to
    :meth:`compute_sample_metrics` as keyword arguments named after the topics
    (see :meth:`all_inputs`), so an implementation names its parameters like
    its topics.  The message of optional ground truth that is not published is
    passed as ``None``.

    Messages given in different frames are related through :attr:`tf_buffer`,
    which the node fills with the transforms published on ``/tf`` and
    ``/tf_static``.

    Subclasses declare the version of their evaluation via the class attributes
    :attr:`VERSION` and :attr:`RELEASE_NOTES`.
    """

    #: Version of the evaluation implementation.
    VERSION: str = "0.0.0"

    #: Mapping of version strings to their release notes.
    RELEASE_NOTES: Dict[str, str] = {}

    def __init__(self, name: str, description: str = "") -> None:
        """Initialize an evaluation definition and empty result store."""
        self.name: str = name
        self.description: str = description
        self.version = self.VERSION
        self.release_notes = self.RELEASE_NOTES
        # Transforms between the frames of the messages, set by the node; None while no transform is available
        self.tf_buffer: Optional[BufferInterface] = None
        self._sample_results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Abstract interface – must be implemented by every concrete evaluation
    # ------------------------------------------------------------------

    @abstractmethod
    def required_inputs(self) -> Dict[str, Any]:
        """Define the topics of the system under test that are evaluated.

        These are the outputs of the system under test, e.g. the predictions
        of a perception algorithm, and, for an evaluation that needs no
        reference, the topics of the scenario it runs in, e.g. the ego state
        and the surrounding objects a closed-loop planner is evaluated on.

        Returns
        -------
        A dictionary mapping input names to their ROS message types.  The
        names are the keyword arguments of :meth:`compute_sample_metrics`
        and the node-relative topics the node subscribes to.
        """

    def required_ground_truth(self) -> Dict[str, Any]:
        """Define the ground-truth topics the inputs are compared with.

        Ground truth is a reference the output of the system under test is
        measured against, e.g. the labels of a dataset.  An evaluation that
        computes its metrics from the inputs alone needs none.

        Returns
        -------
        A dictionary mapping ground-truth names to their ROS message types,
        empty by default.  The names are used like those of
        :meth:`required_inputs`.
        """
        return {}

    def optional_ground_truth(self) -> Dict[str, Any]:
        """Define the ground-truth topics that are compared with the inputs while they are published.

        Optional ground truth is a reference that not every source of ground
        truth provides, e.g. meta information published next to the labels
        of a dataset that ``perception_msgs`` cannot express.  A sample is
        evaluated once the messages of all required topics have been received,
        waiting for the message of an optional topic only while that topic has
        a publisher.  Otherwise, its message is passed to
        :meth:`compute_sample_metrics` as ``None``.

        Returns
        -------
        A dictionary mapping ground-truth names to their ROS message types,
        empty by default. The names are used like those of
        :meth:`required_inputs`.
        """
        return {}

    def derived_topics(self) -> Dict[str, Tuple[str, str]]:
        """Define the inputs that are published next to the topic of another input.

        A topic that is published next to another one, e.g. the meta
        information of an object list on ``<object list topic>/meta_info``,
        follows that topic instead of having to be configured on its own.

        Returns
        -------
        A dictionary mapping the name of such an input to the name of the
        input it is published next to and the suffix appended to that
        input's topic, empty by default.
        """
        return {}

    @abstractmethod
    def compute_sample_metrics(self, sample_id: Optional[str] = None, **messages: Any) -> Dict[str, Any]:
        """Compute metrics for a single sample.

        Parameters
        ----------
        sample_id:
            An optional identifier for the sample.
        messages:
            One message per topic of :meth:`all_inputs`, passed by the name
            of its topic; ``None`` for optional ground truth that is not
            published.

        Returns
        -------
        A dictionary mapping metric names to their values.
        """

    @abstractmethod
    def compute_aggregated_metrics(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate per-sample metrics over the full dataset.

        Parameters
        ----------
        sample_results:
            A list of dictionaries, each returned by
            :meth:`compute_sample_metrics`.

        Returns
        -------
        A dictionary mapping aggregated metric names to their values (e.g.
        mean-AP, precision, recall).  A metric composed of sub-metrics is a
        nested dictionary that holds its aggregated value under ``_value_``
        next to its sub-metrics.
        """

    # ------------------------------------------------------------------
    # Topics
    # ------------------------------------------------------------------

    def all_inputs(self) -> Dict[str, Any]:
        """Combine the inputs and the ground truth into the topics the evaluation reads.

        Returns
        -------
        A dictionary mapping the names of the inputs, followed by those of the
        required and of the optional ground truth, to their ROS message types.

        Raises
        ------
        ValueError
            If the evaluation requires no topic, declares a topic name twice,
            e.g. as input and as ground truth, or a derived topic refers to a
            topic it does not read.
        """
        declared_topics = {
            "input": self.required_inputs(),
            "ground truth": self.required_ground_truth(),
            "optional ground truth": self.optional_ground_truth(),
        }
        inputs: Dict[str, Any] = {}
        roles: Dict[str, str] = {}
        for role, topics in declared_topics.items():
            for name, msg_type in topics.items():
                if name in roles:
                    raise ValueError(f"Evaluation '{self.name}' declares '{name}' as {roles[name]} and as {role}")
                inputs[name] = msg_type
                roles[name] = role
        if not declared_topics["input"] and not declared_topics["ground truth"]:
            raise ValueError(f"Evaluation '{self.name}' requires no topic to evaluate")
        for name, (source, _) in self.derived_topics().items():
            if name not in inputs or source not in inputs or source == name:
                raise ValueError(f"Evaluation '{self.name}' derives the topic of '{name}' from that of '{source}'")
        return inputs

    # ------------------------------------------------------------------
    # Optional visualization interface
    # ------------------------------------------------------------------

    def visualization_outputs(self) -> Dict[str, Any]:
        """Define the ROS message types this evaluation publishes for visualization.

        Returns
        -------
        A dictionary mapping output names to their ROS message types, empty for
        an evaluation that offers no visualization. The names are node-relative
        topics and match the keys of :meth:`visualize_sample`.
        """
        return {}

    def visualize_sample(self, sample_id: Optional[str] = None, **messages: Any) -> Dict[str, Any]:
        """Build the visualization messages for a single sample.

        Called once per sample while visualization is enabled, with the same
        arguments as :meth:`compute_sample_metrics`.

        Returns
        -------
        A dictionary mapping the names of :meth:`visualization_outputs` to ready
        ROS messages, empty for an evaluation that offers no visualization.
        """
        return {}

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def record_sample(self, sample_id: Optional[str] = None, scene_id: Optional[str] = None, **messages: Any) -> Dict[str, Any]:
        """Compute and store per-sample metrics.

        This is the main entry point used by the evaluation loop. The
        *messages* of the sample are forwarded verbatim to
        :meth:`compute_sample_metrics`.

        Parameters
        ----------
        sample_id:
            An optional identifier for the sample.
        scene_id:
            The scene of the dataset the sample belongs to, which
            :meth:`finalize` aggregates the samples by. An evaluation loop
            that learns the scene only after the sample has been evaluated may
            set it on the returned entry instead of passing it here.

        Returns
        -------
        The stored entry of the sample, as ``{"sample_id", "scene_id",
        "metrics"}``.
        """
        metrics = self.compute_sample_metrics(sample_id=sample_id, **messages)
        entry: Dict[str, Any] = {"sample_id": sample_id, "scene_id": scene_id, "metrics": metrics}
        self._sample_results.append(entry)
        return entry

    def sample_results_by_scene(self) -> Dict[str, List[Dict[str, Any]]]:
        """Group the recorded samples by the scene of the dataset they belong to.

        Returns
        -------
        A dictionary mapping each scene to its recorded samples, in the order
        the samples were recorded. Samples recorded without a scene are left
        out, as they cannot be attributed to one.
        """
        scenes: Dict[str, List[Dict[str, Any]]] = {}
        for entry in self._sample_results:
            scene_id = entry.get("scene_id")
            if scene_id is not None:
                scenes.setdefault(str(scene_id), []).append(entry)
        return scenes

    def finalize(self, complete: bool = True) -> Dict[str, Any]:
        """Compute aggregated metrics and return the full results payload.

        Metrics are reported on two levels: ``metrics`` over all evaluated
        samples, and the ``metrics`` of each scene in ``scenes`` over the
        samples of that scene.

        Parameters
        ----------
        complete:
            Whether all samples of the evaluation have been evaluated. An
            evaluation that was interrupted, e.g. with Ctrl-C, still reports the
            samples it did evaluate, marked as ``"complete": false`` so that
            they are not mistaken for the results over the whole dataset.
        """
        aggregated = self.compute_aggregated_metrics(self._sample_results)
        scenes = self.sample_results_by_scene()
        return {
            "evaluation": self.name,
            "version": self.version,
            "description": self.description,
            "complete": complete,
            "num_samples": len(self._sample_results),
            "num_scenes": len(scenes),
            "metrics": aggregated,
            "scenes": {
                scene_id: {
                    "num_samples": len(entries),
                    "sample_ids": [entry["sample_id"] for entry in entries],
                    "metrics": self.compute_aggregated_metrics(entries),
                }
                for scene_id, entries in scenes.items()
            },
        }

    def save_results(self, output_path: str, results: Optional[Dict[str, Any]] = None, complete: bool = True) -> str:
        """Finalize and write results to a JSON file.

        Parameters
        ----------
        output_path:
            Path to the output JSON file.
        results:
            A results payload previously obtained from :meth:`finalize`, which
            is computed here when omitted.
        complete:
            Whether all samples of the evaluation have been evaluated, see
            :meth:`finalize`.  Only used while the results are computed here; a
            given payload is written with the flag it was finalized with.

        Returns
        -------
        The absolute path of the written file.
        """
        results = self.finalize(complete=complete) if results is None else results
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        return os.path.abspath(output_path)

    def reset(self) -> None:
        """Clear accumulated sample results."""
        self._sample_results.clear()
