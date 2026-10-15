# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Building blocks of the evaluations: metrics, the inputs they read and the data they are computed from.

A metric computes one aspect of an evaluation, e.g. the average precision of a
detector or the time to collision of the ego vehicle. It declares the inputs it
reads, each by name, ROS message type and role, so that an evaluation only reads
the topics of the metrics it computes, and it declares its configuration with
its default values. Which metrics an evaluation computes and how they are
configured is set per evaluation, see
:class:`~autonomy_evaluation.evaluations.Evaluation.Evaluation`.

Like an evaluation, a metric is computed in two passes:

1. :meth:`Metric.compute_sample` turns the messages of a single sample into a
   record, e.g. the true and false positives of a frame or the time to
   collision with each object. Data that several metrics derive from the same
   sample, e.g. the matching of predictions and labels, is derived once and
   shared through :meth:`Sample.memo`.
2. :meth:`Metric.aggregate` aggregates the records of the samples of a scene or
   of the whole evaluation into the reported metric. A metric that depends on
   the order of the samples, e.g. a post-encroachment time or a jerk,
   aggregates the records of each contiguous sequence of samples
   (:meth:`Aggregation.sequences`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple

from autonomy_evaluation.evaluations.sampling import split_into_sequences

#: Role of an input that is evaluated: the output of the system under test, e.g. the predictions of a detector, or its
#: behavior, e.g. the state of the ego vehicle driven by a closed-loop planner.
EVALUATED = "evaluated"

#: Role of an input the evaluated inputs are compared with, e.g. the labels of a dataset.
REFERENCE = "reference"

#: Role of an input that a metric needs, but neither evaluates nor compares with, e.g. the road users the ego vehicle
#: of a closed-loop simulation interacts with.
CONTEXT = "context"

#: Roles an input can play, in the order the node reports its inputs in.
ROLES: Tuple[str, ...] = (EVALUATED, REFERENCE, CONTEXT)


@dataclass(frozen=True)
class Input:
    """An input a metric reads: a topic, the ROS message type published on it and the role it plays.

    The role is a property of the metric, e.g. the predictions of a detection
    metric are evaluated and its labels are the reference. Which topic an input
    is read from is configured when the evaluation is launched, by remapping the
    name of the input onto the topic.

    Attributes:
        msg_type: ROS message type of the topic.
        role: Role of the input, one of :data:`ROLES`.
        optional: Whether a sample is evaluated without the message of the
            input while its topic has no publisher, e.g. meta information that
            not every dataset publishes. The message of an optional input that
            is not published is ``None``.
        next_to: Name of another input and the suffix appended to the topic of
            that input, for a topic published next to it, e.g.
            ``("label", "/meta_info")``. Such a topic follows the topic of the
            other input instead of being configured on its own.
    """

    msg_type: Any
    role: str
    optional: bool = False
    next_to: Optional[Tuple[str, str]] = None

    def __post_init__(self) -> None:
        """Reject a role that is none of :data:`ROLES`."""
        if self.role not in ROLES:
            raise ValueError(f"Input role '{self.role}' is none of {list(ROLES)}")


class Sample:
    """The messages of a single sample, and the data the metrics derive from them.

    Attributes:
        messages: Messages of the sample by input name; ``None`` for an
            optional input that is not published.
        sample_id: Identifier of the sample, if known.
        stamp: Stamp of the sample in seconds, if known.
        config: Configuration of the evaluation the sample is evaluated by,
            whose evaluation-level entries configure the data derived from the
            sample, e.g. the detection range of the objects of a detection.
        tf_buffer: Transforms between the frames of the messages, ``None`` if
            no transform is available.
    """

    def __init__(
        self,
        messages: Dict[str, Any],
        sample_id: Optional[str] = None,
        stamp: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None,
        tf_buffer: Any = None,
    ) -> None:
        """Hold the messages of a sample and start without derived data."""
        self.messages = messages
        self.sample_id = sample_id
        self.stamp = stamp
        self.config: Dict[str, Any] = config or {}
        self.tf_buffer = tf_buffer
        self._memo: Dict[Hashable, Any] = {}

    def __getitem__(self, name: str) -> Any:
        """Return the message of an input of the sample by its name."""
        return self.messages[name]

    def memo(self, key: Hashable, compute: Callable[[], Any]) -> Any:
        """Return data derived from the sample, deriving it once on its first request.

        Several metrics derive the same data from a sample, e.g. all metrics of
        a detection match the same predictions with the same labels. The first
        metric to request the data derives it, the others reuse it.

        Args:
            key: Identifies the derived data, including everything it depends
                on besides the messages, e.g. ``("matches", 2.0)``.
            compute: Derives the data, called only on the first request.

        Returns:
            The derived data.
        """
        if key not in self._memo:
            self._memo[key] = compute()
        return self._memo[key]


class Aggregation:
    """The recorded samples metrics are aggregated over, and the results of the metrics aggregated before them.

    Attributes:
        entries: Recorded samples to aggregate, those of a scene or all of
            them, in the order they were recorded. Each is a
            ``{"sample_id", "scene_id", "stamp", "metrics"}`` entry, whose
            ``metrics`` hold the records of the metrics by qualified name.
        recorded_entries: All samples the evaluation recorded so far, e.g. to
            evaluate every scene on the classes of the labels of all samples.
        config: Configuration of the evaluation.
        results: Aggregated results of the metrics aggregated before, by
            qualified name, among them the results of the metrics a metric
            requires (see :meth:`Metric.requires`).
    """

    def __init__(
        self,
        entries: List[Dict[str, Any]],
        recorded_entries: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
        results: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Hold the samples to aggregate and start without derived data."""
        self.entries = entries
        self.recorded_entries = entries if recorded_entries is None else recorded_entries
        self.config: Dict[str, Any] = config or {}
        self.results: Dict[str, Any] = {} if results is None else results
        self._memo: Dict[Hashable, Any] = {}

    def records(self, metric: str, entries: Optional[List[Dict[str, Any]]] = None) -> List[Any]:
        """Return the records of a metric for the given samples.

        Args:
            metric: Qualified name of the metric.
            entries: Recorded samples, :attr:`entries` by default.

        Returns:
            The record of the metric for each sample, in the order of the samples.
        """
        return [entry["metrics"][metric] for entry in (self.entries if entries is None else entries)]

    def sequences(self) -> List[List[Dict[str, Any]]]:
        """Split the samples into contiguous sequences.

        Two consecutive samples belong to the same sequence if they belong to
        the same scene, are in order and at most the configured
        ``max_sample_interval`` apart (see
        :func:`~autonomy_evaluation.evaluations.sampling.split_into_sequences`).

        Returns:
            The sequences of :attr:`entries`, each in recording order.
        """
        return self.memo(
            "sequences", lambda: split_into_sequences(self.entries, self.config.get("max_sample_interval", float("inf")))
        )

    def memo(self, key: Hashable, compute: Callable[[], Any]) -> Any:
        """Return data derived from the samples, deriving it once on its first request, like :meth:`Sample.memo`."""
        if key not in self._memo:
            self._memo[key] = compute()
        return self._memo[key]


class Metric(ABC):
    """Building block of an evaluation that computes one metric.

    A metric declares the inputs it reads (:meth:`inputs`) and its configuration
    (:meth:`default_config`), turns each sample into a record
    (:meth:`compute_sample`) and aggregates the records into its result
    (:meth:`aggregate`). The result is a dictionary that holds the aggregated
    value of the metric under ``_value_`` next to its sub-metrics.

    A metric may build on the results of other metrics of its evaluation
    (:meth:`requires`), which are then computed as well, but only reported if
    they are selected themselves. A metric that only provides data for others
    is :attr:`INTERNAL`.
    """

    #: Name of the metric in the results and the configuration of its evaluation, unique within its group.
    NAME: str = ""

    #: Whether the metric only provides data that other metrics require. An internal metric is computed whenever a
    #: computed metric requires it, but can neither be selected nor is it reported.
    INTERNAL: bool = False

    def __init__(self) -> None:
        """Start with the default configuration, under the name of the metric."""
        #: Qualified name of the metric within its evaluation, e.g. ``criticality.ttc`` for a metric of a group; set
        #: when the evaluation collects its metrics.
        self.name: str = self.NAME
        #: Configuration of the metric: :meth:`default_config` with the values configured for its evaluation.
        self.config: Dict[str, Any] = self.default_config()

    @abstractmethod
    def inputs(self) -> Dict[str, Input]:
        """Declare the inputs the metric reads.

        Returns:
            Input name to :class:`Input`. The names are the keys of
            :attr:`Sample.messages`; metrics of the same evaluation that read
            the same input declare it under the same name.
        """

    def default_config(self) -> Dict[str, Any]:
        """Declare the configuration of the metric with its default values.

        Returns:
            Configuration key to default value, empty by default. A value is a
            ``bool``, ``int``, ``float``, ``str``, a list of one of them, or a
            nested dictionary of such values, so that it can be configured by
            ROS parameters.
        """
        return {}

    def requires(self) -> Tuple[str, ...]:
        """Declare the metrics whose aggregated results the metric builds on.

        Returns:
            Qualified names of metrics of the same evaluation, empty by default.
            Their results are passed in :attr:`Aggregation.results`.
        """
        return ()

    def compute_sample(self, sample: Sample) -> Any:
        """Turn the messages of a single sample into the record the metric is aggregated from.

        Args:
            sample: Messages of the sample and the data derived from them.

        Returns:
            The record of the sample, ``None`` by default for a metric that
            aggregates the results of other metrics only.
        """
        return None

    @abstractmethod
    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Aggregate the records of the samples into the result of the metric.

        Args:
            aggregation: Samples to aggregate, with the records of the metric
                (:meth:`Aggregation.records`) and the results of the metrics it
                requires.

        Returns:
            The result of the metric, holding its aggregated value under
            ``_value_`` next to its sub-metrics.
        """

    def visualization_outputs(self) -> Dict[str, Any]:
        """Declare the ROS message types the metric publishes for visualization.

        Returns:
            Output name to ROS message type, empty by default. The names are
            node-relative topics, unique within the evaluation.
        """
        return {}

    def visualize_sample(self, sample: Sample) -> Dict[str, Any]:
        """Build the visualization messages of a single sample.

        Args:
            sample: Messages of the sample and the data derived from them.

        Returns:
            Output name of :meth:`visualization_outputs` to ready ROS message.
        """
        return {}


class MetricGroup:
    """Named group of related metrics of an evaluation, e.g. the criticality metrics of a closed-loop evaluation.

    The metrics of a group are selected all at once by the name of the group or
    one by one as ``<group>.<metric>``, configured under the name of the group,
    and reported nested under it.

    Attributes:
        name: Name of the group in the results and the configuration.
        metrics: Metrics of the group, in the order they are reported in.
    """

    def __init__(self, name: str, metrics: Sequence[Metric]) -> None:
        """Group the given metrics under the given name."""
        self.name = name
        self.metrics = list(metrics)
