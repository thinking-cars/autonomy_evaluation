# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Base class of all evaluations of automated driving tasks.

An evaluation is composed of metrics, the building blocks of
:mod:`~autonomy_evaluation.evaluations.Metric`, e.g. the average precision and
the translation error of a detection, or the time to collision and the jerk of
the ego vehicle of a closed-loop simulation. Each metric declares the topics it
reads, each in a role: *evaluated*, e.g. the predictions of a detector or the
state of the ego vehicle a planner drives, *reference*, e.g. the labels of a
dataset the predictions are compared with, or *context*, e.g. the road users
the ego vehicle interacts with. An evaluation therefore may read the topics of
a system under test only, or compare them with ground truth. Ground truth that
enriches a metric without being necessary for it, e.g. dataset annotations that
not every dataset publishes, is declared as optional.

Which metrics are computed and how is configured per evaluation (see
:meth:`Evaluation.config_defaults`), e.g. by ROS parameters of the node. The
evaluation only reads the topics of the metrics it computes.

An evaluation without metric building blocks, e.g. one implemented in another
package before they existed, declares its topics with
:meth:`Evaluation.required_inputs`, :meth:`Evaluation.required_ground_truth`,
:meth:`Evaluation.optional_ground_truth` and :meth:`Evaluation.derived_topics`,
and computes its metrics itself by overriding
:meth:`Evaluation.compute_sample_metrics` and
:meth:`Evaluation.compute_aggregated_metrics`.
"""

from __future__ import annotations

import copy
import json
import os
from abc import ABC
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from autonomy_evaluation.evaluations.Metric import (
    Aggregation,
    EVALUATED,
    Input,
    Metric,
    MetricGroup,
    REFERENCE,
    Sample,
)
from autonomy_evaluation.evaluations.sampling import sampling_statistics
from tf2_ros import BufferInterface

#: Configuration every evaluation has, with its default values.
_BASE_CONFIG: Dict[str, Any] = {
    # seconds between two consecutive samples beyond which they belong to different sequences (see sampling.py)
    "max_sample_interval": 1.0,
}

#: Configuration key selecting the computed metrics.
_METRICS_KEY = "metrics"

#: Separator of the keys of nested configuration entries in their flattened names, e.g. "ap.matching_thresholds".
CONFIG_SEPARATOR = "."


class Evaluation(ABC):
    """Base class of the evaluations of automated driving tasks.

    An evaluation is responsible for:
    * composing its metrics (:meth:`metrics`) and computing those that are
      selected by its configuration (:meth:`configure`),
    * declaring the topics it reads, i.e. the inputs of the computed metrics
      (:meth:`inputs`),
    * computing the records of the metrics for each sample
      (:meth:`record_sample`) and aggregating them into scene-level and
      dataset-level metrics (:meth:`finalize`),
    * persisting results to JSON.

    The messages of all topics that belong to the same sample are passed to
    :meth:`record_sample` as keyword arguments named after the inputs (see
    :meth:`inputs`).  The message of an optional input that is not published is
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

    #: Qualified name of the metric whose aggregated value the evaluation reports as its own under ``_value_``, e.g.
    #: the detection score of a detection; ``None`` for an evaluation without a single overall value.
    HEADLINE_METRIC: Optional[str] = None

    def __init__(self, name: str, description: str = "") -> None:
        """Initialize an evaluation definition and empty result store."""
        self.name: str = name
        self.description: str = description
        self.version = self.VERSION
        self.release_notes = self.RELEASE_NOTES
        # Transforms between the frames of the messages, set by the node; None while no transform is available
        self.tf_buffer: Optional[BufferInterface] = None
        # Topic each input is subscribed on by input name, set by the node and reported with the results
        self.input_topics: Dict[str, str] = {}
        self._sample_results: List[Dict[str, Any]] = []
        # Metrics by qualified name and the qualified names of the metrics of each group, collected on first use, as
        # subclasses configure their metrics after initializing the base class
        self._metrics: Optional[Dict[str, Metric]] = None
        self._groups: Dict[str, List[str]] = {}
        # Effective configuration, and the selected and computed metrics it implies; set on first use or by configure()
        self._config: Optional[Dict[str, Any]] = None
        self._selected: List[str] = []
        self._computed: List[str] = []
        # Last recorded sample, whose derived data the visualization of the same sample reuses
        self._last_sample: Optional[Sample] = None

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def metrics(self) -> Sequence[Union[Metric, MetricGroup]]:
        """Compose the evaluation of its metrics.

        Returns:
            The metrics and groups of metrics of the evaluation, in the order
            they are reported in; empty for an evaluation without metric
            building blocks.
        """
        return []

    def available_metrics(self) -> Dict[str, Metric]:
        """Collect the metrics of the evaluation by their qualified names.

        The qualified name of a metric is its :attr:`~Metric.NAME`, prefixed
        by the name of its group, e.g. ``criticality.ttc``.

        Returns:
            Qualified name to metric, in the order of :meth:`metrics`.

        Raises:
            ValueError: A metric has no name, or two metrics or groups share a
                qualified name.
        """
        if self._metrics is None:
            metrics: Dict[str, Metric] = {}
            groups: Dict[str, List[str]] = {}
            for block in self.metrics():
                group = block.name if isinstance(block, MetricGroup) else ""
                if group:
                    if group in groups or group in metrics:
                        raise ValueError(f"Evaluation '{self.name}' declares '{group}' twice")
                    groups[group] = []
                for metric in block.metrics if group else [block]:
                    name = f"{group}{CONFIG_SEPARATOR}{metric.NAME}" if group else metric.NAME
                    if not metric.NAME:
                        raise ValueError(f"Evaluation '{self.name}' declares a {type(metric).__name__} without NAME")
                    if name in metrics or name in groups:
                        raise ValueError(f"Evaluation '{self.name}' declares '{name}' twice")
                    metric.name = name
                    metrics[name] = metric
                    if group:
                        groups[group].append(name)
            self._metrics, self._groups = metrics, groups
        return self._metrics

    def selected_metrics(self) -> Dict[str, Metric]:
        """Return the metrics selected by the configuration, which are reported.

        Returns:
            Qualified name to metric, in the order of :meth:`metrics`.
        """
        self._ensure_configured()
        return {name: self.available_metrics()[name] for name in self._selected}

    def computed_metrics(self) -> Dict[str, Metric]:
        """Return the metrics that are computed: the selected ones and those they require.

        Returns:
            Qualified name to metric, each after the metrics it requires.
        """
        self._ensure_configured()
        return {name: self.available_metrics()[name] for name in self._computed}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def default_config(self) -> Dict[str, Any]:
        """Declare the evaluation-level configuration with its default values.

        Evaluation-level entries configure what all metrics share, e.g. the
        detection range of the objects of a detection. They are passed to the
        metrics in :attr:`Sample.config` and :attr:`Aggregation.config`.

        Returns:
            Configuration key to default value, empty by default; values follow
            the rules of :meth:`Metric.default_config`.
        """
        return {}

    def config_defaults(self) -> Dict[str, Any]:
        """Return the full configuration of the evaluation with its default values.

        The configuration holds:

        * ``max_sample_interval``: seconds between two consecutive samples
          beyond which they belong to different sequences (default ``1.0``),
        * the evaluation-level entries of :meth:`default_config`,
        * ``metrics``: the selected metrics, each by the name of a metric, of
          a group of metrics or of a metric of a group as ``<group>.<metric>``;
          all metrics by default,
        * the configuration of each metric (:meth:`Metric.default_config`)
          under its name, nested under the name of its group.

        Returns:
            The nested configuration with its default values.

        Raises:
            ValueError: An evaluation-level entry shares its key with a metric
                or a group.
        """
        defaults = {**copy.deepcopy(_BASE_CONFIG), **self.default_config()}
        metrics = self.available_metrics()
        if not metrics:
            return defaults

        sections: Dict[str, Any] = {}
        for name, metric in metrics.items():
            metric_defaults = metric.default_config()
            if metric_defaults:
                group, _, key = name.rpartition(CONFIG_SEPARATOR)
                (sections.setdefault(group, {}) if group else sections)[key] = metric_defaults
        names = {_METRICS_KEY, *self._groups, *metrics}
        clashes = sorted(key for key in defaults if key in names)
        if clashes:
            raise ValueError(f"Evaluation '{self.name}' configures {clashes} next to metrics of the same name")
        return {**defaults, _METRICS_KEY: self._top_level_names(), **sections}

    @property
    def config(self) -> Dict[str, Any]:
        """Effective configuration of the evaluation, the default configuration unless :meth:`configure` was called."""
        self._ensure_configured()
        return self._config

    def _ensure_configured(self) -> None:
        """Apply the default configuration unless the evaluation has been configured already."""
        if self._config is None:
            self.configure()

    def configure(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply a configuration to the evaluation and its metrics.

        Every entry that is not given keeps its default value. An integer is
        accepted for a floating-point entry.

        Args:
            config: Nested configuration entries that differ from
                :meth:`config_defaults`, e.g.
                ``{"metrics": ["ap"], "ap": {"min_precision": 0.2}}``.

        Returns:
            The effective configuration.

        Raises:
            ValueError: An entry is unknown or of the wrong type, the selection
                names no or an unknown metric, or a metric requires an unknown
                metric or itself.
        """
        effective = _merged(self.config_defaults(), config or {}, f"Evaluation '{self.name}'")
        metrics = self.available_metrics()
        if metrics:
            selected = self._resolve_selection(effective[_METRICS_KEY])
            computed = self._with_requirements(selected)
            for name, metric in metrics.items():
                group, _, key = name.rpartition(CONFIG_SEPARATOR)
                section = effective.get(group, {}) if group else effective
                metric.config = copy.deepcopy(section[key]) if metric.default_config() else {}
            self._selected, self._computed = selected, computed
        self._config = effective
        return effective

    def _top_level_names(self) -> List[str]:
        """Return the names of the groups and of the selectable metrics outside of groups, in the order of :meth:`metrics`."""
        names = []
        for block in self.metrics():
            if isinstance(block, MetricGroup):
                names.append(block.name)
            elif not block.INTERNAL:
                names.append(block.NAME)
        return names

    def _resolve_selection(self, selection: Iterable[str]) -> List[str]:
        """Resolve the selected names into the qualified names of the selected metrics, in the order of :meth:`metrics`."""
        metrics = self.available_metrics()
        selected = set()
        for name in selection:
            if name in self._groups:
                selected.update(member for member in self._groups[name] if not metrics[member].INTERNAL)
            elif name in metrics and not metrics[name].INTERNAL:
                selected.add(name)
            else:
                selectable = [*self._groups, *(member for member, metric in metrics.items() if not metric.INTERNAL)]
                raise ValueError(f"Evaluation '{self.name}' has no metric '{name}', expected one of {selectable}")
        if not selected:
            raise ValueError(f"Evaluation '{self.name}' selects no metric")
        return [name for name in metrics if name in selected]

    def _with_requirements(self, selected: List[str]) -> List[str]:
        """Add the metrics the selected metrics require, ordering each metric after the metrics it requires."""
        metrics = self.available_metrics()
        ordered: List[str] = []
        visiting: set = set()

        def visit(name: str, required_by: Optional[str]) -> None:
            if name in ordered:
                return
            if name not in metrics:
                raise ValueError(f"Metric '{required_by}' of evaluation '{self.name}' requires unknown metric '{name}'")
            if name in visiting:
                raise ValueError(f"Metric '{name}' of evaluation '{self.name}' requires itself")
            visiting.add(name)
            for required in metrics[name].requires():
                visit(required, name)
            visiting.discard(name)
            ordered.append(name)

        for name in selected:
            visit(name, None)
        return ordered

    # ------------------------------------------------------------------
    # Topics
    # ------------------------------------------------------------------

    def inputs(self) -> Dict[str, Input]:
        """Declare the topics the evaluation reads: the inputs of the computed metrics.

        Returns:
            Input name to :class:`~autonomy_evaluation.evaluations.Metric.Input`,
            in the order the computed metrics declare them. The names are the
            keyword arguments of :meth:`record_sample` and the node-relative
            topics the node subscribes to.

        Raises:
            ValueError: Two metrics declare the same input differently, the
                evaluation requires no topic, or a topic is published next to a
                topic the evaluation does not read.
        """
        if not self.available_metrics():
            return self._declared_inputs()
        return self._merged_inputs(self.computed_metrics().values())

    def available_inputs(self) -> Dict[str, Input]:
        """Declare the topics any metric of the evaluation reads, whether it is computed or not.

        Returns:
            Input name to input of all metrics, like :meth:`inputs`.
        """
        if not self.available_metrics():
            return self._declared_inputs()
        return self._merged_inputs(self.available_metrics().values())

    def all_inputs(self) -> Dict[str, Any]:
        """Return the ROS message types of the topics the evaluation reads.

        Returns:
            Input name to ROS message type, in the order of :meth:`inputs`.
        """
        return {name: declared.msg_type for name, declared in self.inputs().items()}

    def _merged_inputs(self, metrics: Iterable[Metric]) -> Dict[str, Input]:
        """Merge the inputs of the given metrics, which must declare a shared input alike."""
        inputs: Dict[str, Input] = {}
        declared_by: Dict[str, str] = {}
        for metric in metrics:
            for name, declared in metric.inputs().items():
                if name in inputs and inputs[name] != declared:
                    raise ValueError(
                        f"Evaluation '{self.name}' reads input '{name}' as {inputs[name]} for metric "
                        f"'{declared_by[name]}' and as {declared} for metric '{metric.name}'"
                    )
                inputs.setdefault(name, declared)
                declared_by.setdefault(name, metric.name)
        return self._validated(inputs)

    def _declared_inputs(self) -> Dict[str, Input]:
        """Turn the topics an evaluation without metric building blocks declares into inputs.

        Its inputs are evaluated, its ground truth is the reference.
        """
        declared_topics = {
            "input": (self.required_inputs(), EVALUATED, False),
            "ground truth": (self.required_ground_truth(), REFERENCE, False),
            "optional ground truth": (self.optional_ground_truth(), REFERENCE, True),
        }
        derived_topics = self.derived_topics()
        inputs: Dict[str, Input] = {}
        roles: Dict[str, str] = {}
        for role, (topics, input_role, optional) in declared_topics.items():
            for name, msg_type in topics.items():
                if name in roles:
                    raise ValueError(f"Evaluation '{self.name}' declares '{name}' as {roles[name]} and as {role}")
                inputs[name] = Input(msg_type, input_role, optional=optional, next_to=derived_topics.get(name))
                roles[name] = role
        for name, (source, _) in derived_topics.items():
            if name not in inputs or source not in inputs or source == name:
                raise ValueError(f"Evaluation '{self.name}' derives the topic of '{name}' from that of '{source}'")
        return self._validated(inputs)

    def _validated(self, inputs: Dict[str, Input]) -> Dict[str, Input]:
        """Check that the inputs require a topic and are only published next to topics that are read."""
        if all(declared.optional or declared.static for declared in inputs.values()):
            raise ValueError(f"Evaluation '{self.name}' requires no topic to evaluate")
        for name, declared in inputs.items():
            source = inputs.get(declared.next_to[0]) if declared.next_to is not None else None
            if declared.next_to is not None and (source is None or source.static or declared.next_to[0] == name):
                raise ValueError(f"Evaluation '{self.name}' derives the topic of '{name}' from that of '{declared.next_to[0]}'")
        return inputs

    def required_inputs(self) -> Dict[str, Any]:
        """Define the evaluated topics of an evaluation without metric building blocks.

        These are the outputs of the system under test, e.g. the predictions
        of a perception algorithm, and, for an evaluation that needs no
        reference, the topics of the scenario it runs in, e.g. the ego state
        and the surrounding objects a closed-loop planner is evaluated on.

        Returns
        -------
        A dictionary mapping input names to their ROS message types, empty by
        default.  The names are the keyword arguments of
        :meth:`compute_sample_metrics` and the node-relative topics the node
        subscribes to.
        """
        return {}

    def required_ground_truth(self) -> Dict[str, Any]:
        """Define the ground-truth topics the inputs of an evaluation without metric building blocks are compared with.

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
        """Define the ground-truth topics of an evaluation without metric building blocks that are read while published.

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
        """Define the topics of an evaluation without metric building blocks that are published next to another one.

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

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def compute_sample_metrics(
        self, sample_id: Optional[str] = None, stamp: Optional[float] = None, **messages: Any
    ) -> Dict[str, Any]:
        """Compute the records of the computed metrics for a single sample.

        An evaluation without metric building blocks overrides this method,
        naming its parameters like its topics; it is called without ``stamp``.

        Parameters
        ----------
        sample_id:
            An optional identifier for the sample.
        stamp:
            The stamp of the sample in seconds, if known.
        messages:
            One message per topic of :meth:`inputs`, passed by the name
            of its input; ``None`` for an optional input that is not
            published.

        Returns
        -------
        A dictionary mapping the qualified names of the computed metrics to
        their records of the sample.
        """
        sample = Sample(messages, sample_id=sample_id, stamp=stamp, config=self.config, tf_buffer=self.tf_buffer)
        self._last_sample = sample
        return {name: metric.compute_sample(sample) for name, metric in self.computed_metrics().items()}

    def compute_aggregated_metrics(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate per-sample metrics over the given samples.

        Each computed metric is aggregated after the metrics it requires, of
        which only the selected ones are reported. An evaluation without metric
        building blocks overrides this method.

        Parameters
        ----------
        sample_results:
            A list of entries recorded by :meth:`record_sample`.

        Returns
        -------
        A dictionary mapping aggregated metric names to their values (e.g.
        mean-AP, precision, recall).  A metric composed of sub-metrics is a
        nested dictionary that holds its aggregated value under ``_value_``
        next to its sub-metrics.  The metrics of a group are nested under the
        name of the group; ``_value_`` holds the aggregated value of
        :attr:`HEADLINE_METRIC`, ``None`` without one.
        """
        aggregation = Aggregation(sample_results, self._sample_results, self.config)
        for name, metric in self.computed_metrics().items():
            aggregation.results[name] = metric.aggregate(aggregation)

        headline = aggregation.results.get(self.HEADLINE_METRIC) if self.HEADLINE_METRIC else None
        reported: Dict[str, Any] = {"_value_": headline["_value_"] if headline else None}
        for name in self._selected:
            group, _, key = name.rpartition(CONFIG_SEPARATOR)
            (reported.setdefault(group, {}) if group else reported)[key] = aggregation.results[name]
        return reported

    # ------------------------------------------------------------------
    # Optional visualization interface
    # ------------------------------------------------------------------

    def visualization_outputs(self) -> Dict[str, Any]:
        """Define the ROS message types this evaluation publishes for visualization.

        Returns
        -------
        A dictionary mapping output names to their ROS message types: those of
        the computed metrics, empty for an evaluation that offers no
        visualization. The names are node-relative topics and match the keys
        of :meth:`visualize_sample`.

        Raises
        ------
        ValueError
            If two computed metrics publish an output of the same name.
        """
        outputs: Dict[str, Any] = {}
        for metric in self.computed_metrics().values():
            for name, msg_type in metric.visualization_outputs().items():
                if name in outputs:
                    raise ValueError(f"Evaluation '{self.name}' publishes visualization output '{name}' twice")
                outputs[name] = msg_type
        return outputs

    def visualize_sample(self, sample_id: Optional[str] = None, stamp: Optional[float] = None, **messages: Any) -> Dict[str, Any]:
        """Build the visualization messages for a single sample.

        Called once per sample while visualization is enabled, with the same
        arguments as :meth:`compute_sample_metrics`. The data the metrics
        derived from the sample while it was recorded is reused.

        Returns
        -------
        A dictionary mapping the names of :meth:`visualization_outputs` to ready
        ROS messages, empty for an evaluation that offers no visualization.
        """
        sample = self._last_sample
        if (
            sample is None
            or sample.sample_id != sample_id
            or sample.messages.keys() != messages.keys()
            or any(sample.messages[name] is not message for name, message in messages.items())
        ):
            sample = Sample(messages, sample_id=sample_id, stamp=stamp, config=self.config, tf_buffer=self.tf_buffer)
        return self.visualize(sample)

    def visualize(self, sample: Sample) -> Dict[str, Any]:
        """Build the visualization messages of the computed metrics for a single sample.

        Args:
            sample: Messages of the sample and the data derived from them.

        Returns:
            Output name of :meth:`visualization_outputs` to ready ROS message.
        """
        visualization: Dict[str, Any] = {}
        for metric in self.computed_metrics().values():
            visualization.update(metric.visualize_sample(sample))
        return visualization

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def record_sample(
        self,
        sample_id: Optional[str] = None,
        scene_id: Optional[str] = None,
        stamp: Optional[float] = None,
        **messages: Any,
    ) -> Dict[str, Any]:
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
        stamp:
            The stamp of the sample in seconds, which orders the samples in
            time for the metrics that depend on it.

        Returns
        -------
        The stored entry of the sample, as ``{"sample_id", "scene_id",
        "stamp", "metrics"}``.
        """
        if self.available_metrics():
            metrics = self.compute_sample_metrics(sample_id=sample_id, stamp=stamp, **messages)
        else:
            metrics = self.compute_sample_metrics(sample_id=sample_id, **messages)
        entry: Dict[str, Any] = {"sample_id": sample_id, "scene_id": scene_id, "stamp": stamp, "metrics": metrics}
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
        samples of that scene. Next to them, the results describe how regularly
        the samples were evaluated (``sampling``, see
        :func:`~autonomy_evaluation.evaluations.sampling.sampling_statistics`),
        the topics that were read (``inputs``) and the configuration the
        metrics were computed with (``config``).

        Parameters
        ----------
        complete:
            Whether all samples of the evaluation have been evaluated. An
            evaluation that was interrupted, e.g. with Ctrl-C, still reports the
            samples it did evaluate, marked as ``"complete": false`` so that
            they are not mistaken for the results over the whole dataset.
        """
        max_sample_interval = self.config["max_sample_interval"]
        aggregated = self.compute_aggregated_metrics(self._sample_results)
        scenes = self.sample_results_by_scene()
        return {
            "evaluation": self.name,
            "version": self.version,
            "description": self.description,
            "complete": complete,
            "num_samples": len(self._sample_results),
            "num_scenes": len(scenes),
            "sampling": sampling_statistics(self._sample_results, max_sample_interval),
            "inputs": {
                name: {
                    "type": _type_name(declared.msg_type),
                    "role": declared.role,
                    "optional": declared.optional,
                    "topic": self.input_topics.get(name),
                }
                for name, declared in self.inputs().items()
            },
            "config": copy.deepcopy(self.config),
            "metrics": aggregated,
            "scenes": {
                scene_id: {
                    "num_samples": len(entries),
                    "sample_ids": [entry["sample_id"] for entry in entries],
                    "sampling": sampling_statistics(entries, max_sample_interval),
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
        self._last_sample = None


def flatten_config(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested configuration into entries named by their joined keys, as ROS parameters name them.

    Args:
        config: Nested configuration, e.g. ``{"ap": {"min_precision": 0.1}}``.
        prefix: Prefix of the flattened names, e.g. the name of the evaluation
            followed by :data:`CONFIG_SEPARATOR`.

    Returns:
        Flattened name to value, e.g. ``{"ap.min_precision": 0.1}``; lists are
        values, not nested entries.
    """
    flattened: Dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flattened.update(flatten_config(value, f"{prefix}{key}{CONFIG_SEPARATOR}"))
        else:
            flattened[f"{prefix}{key}"] = value
    return flattened


def unflatten_config(flattened: Dict[str, Any]) -> Dict[str, Any]:
    """Nest the entries of a flattened configuration again, inverting :func:`flatten_config`.

    Args:
        flattened: Flattened name to value, e.g. ``{"ap.min_precision": 0.1}``.

    Returns:
        The nested configuration, e.g. ``{"ap": {"min_precision": 0.1}}``.
    """
    config: Dict[str, Any] = {}
    for name, value in flattened.items():
        *parents, key = name.split(CONFIG_SEPARATOR)
        section = config
        for parent in parents:
            section = section.setdefault(parent, {})
        section[key] = value
    return config


def _merged(defaults: Dict[str, Any], config: Dict[str, Any], owner: str, path: str = "") -> Dict[str, Any]:
    """Apply configured entries to the default configuration, checking each against the type of its default.

    Args:
        defaults: Nested default configuration.
        config: Nested configured entries, each with a default.
        owner: Evaluation the configuration belongs to, for error messages.
        path: Flattened name of the nested configuration, for error messages.

    Returns:
        A copy of the default configuration with the configured entries.

    Raises:
        ValueError: An entry has no default or is of another type than it.
    """
    unknown = sorted(key for key in config if key not in defaults)
    if unknown:
        raise ValueError(f"{owner} has no configuration {[path + key for key in unknown]}, expected one of {sorted(defaults)}")
    merged: Dict[str, Any] = {}
    for key, default in defaults.items():
        name = f"{path}{key}"
        if key not in config:
            merged[key] = copy.deepcopy(default)
        elif isinstance(default, dict):
            if not isinstance(config[key], dict):
                raise ValueError(f"{owner} expects configuration '{name}' to be nested entries, not {config[key]!r}")
            merged[key] = _merged(default, config[key], owner, f"{name}{CONFIG_SEPARATOR}")
        else:
            merged[key] = _coerced(default, config[key], owner, name)
    return merged


def _coerced(default: Any, value: Any, owner: str, name: str) -> Any:
    """Check a configured value against the type of its default, accepting integers for floating-point values.

    Raises:
        ValueError: The value is of another type than its default.
    """
    if isinstance(default, (list, tuple)):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{owner} expects configuration '{name}' to be a list, not {value!r}")
        return [_coerced(default[0], element, owner, name) for element in value] if default else list(value)
    if isinstance(default, bool) or isinstance(value, bool):
        valid = isinstance(default, bool) and isinstance(value, bool)
    elif isinstance(default, float):
        valid = isinstance(value, (int, float))
        value = float(value) if valid else value
    else:
        valid = isinstance(value, type(default))
    if not valid:
        raise ValueError(f"{owner} expects configuration '{name}' to be of type {type(default).__name__}, not {value!r}")
    return value


def _type_name(msg_type: Any) -> str:
    """Name a ROS message type like ROS does, e.g. ``perception_msgs/msg/ObjectList``."""
    module = getattr(msg_type, "__module__", "")
    name = getattr(msg_type, "__name__", str(msg_type))
    package, _, submodule = module.partition(".")
    return f"{package}/msg/{name}" if submodule.startswith("msg") else name
