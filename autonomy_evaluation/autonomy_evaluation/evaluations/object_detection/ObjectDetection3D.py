# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""3D object detection evaluation on the classes of ``perception_msgs/ObjectClassification``.

Evaluates 3D bounding boxes, e.g. detected in lidar point clouds, camera images
or both, against the labels of any dataset that ``autonomy_datasets`` provides.
The metrics follow the nuScenes detection benchmark
(https://www.nuscenes.org/object-detection), which serves as reference; how the
evaluation deviates from it is documented in ``docs/IMPLEMENTATION.md``.

Objects are represented as plain ``dict`` records extracted directly from the
``perception_msgs/ObjectList`` messages. Each record carries the fields the
metrics consume: ``x``, ``y``, ``width``, ``length``, ``height``, ``yaw``,
``vx``, ``vy``, ``types``, ``classes``, ``confidence_score`` and, for
labels, ``positive``, ``velocity_set`` and ``bike_rack``.

Classes:

- The evaluated classes are those of ``perception_msgs/ObjectClassification``:
  pedestrian, bicycle, motorcycle, car, utility, bus, animal, vru and micro.
  Deprecated classification types are evaluated as the types replacing them,
  types the message does not define as UNKNOWN.
- The classes of an object are the types of its classifications that share the
  highest probability. A dataset that cannot tell which class an object has
  assigns all its possible classes with the same probability.
- UNKNOWN (definitely none of the defined classes) and UNCLASSIFIED (unknown
  classification, i.e. any class) are never evaluated.

Scoring:

- Compatibility: a prediction may only match a label if all its classes are
  possible classes of the label. An UNCLASSIFIED label may be any class.
- Labels whose possible classes are all evaluated count exactly once, as true
  positive or false negative. Labels that may also be UNKNOWN or UNCLASSIFIED
  are don't-care objects: a prediction matched to one is ignored, and they are
  never a false negative. Labels that can only be UNKNOWN are matched by no
  prediction, so a prediction on them is a false positive.
- Classes that the labels do not distinguish are evaluated together, as one
  class named after its members, e.g. ``pedestrian|vru``: the possible classes
  of every label of the evaluated samples end up in the same evaluated class.

Key evaluation settings:

- Frames:     Predictions in another frame than the labels are transformed into
              the frame of the labels with tf2.
- Matching:  Predictions, highest confidence first, each claim the nearest
              unclaimed compatible label whose BEV center is closer than the
              threshold.
- Thresholds: 0.5, 1.0, 2.0, 4.0 m (applied uniformly to all classes).
- Ranges:     Per-class max detection distance (car/utility/bus < 50 m, all
              other classes < 40 m). An object of several possible classes uses
              the largest of their ranges.
- Bike-rack:  Bicycles and motorcycles whose BEV center falls inside a bike rack
              are excluded from both predictions and labels. Bike racks are
              recognized by the ``original_class`` of the label meta information,
              published on the optional ``<label topic>/meta_info`` topic.
- AP filter:  P-R points with cum_precision <= 0.1 or cum_recall <= 0.1 are
              excluded from the 101-point recall interpolation.
- TP metrics: ATE, ASE, AOE and AVE measured at the 2.0 m threshold. AVE is only
              computed against labels whose velocity is set, i.e. not marked as
              invalid in their state covariance; without any such label it is
              unavailable.
- Score:      Detection score combining mAP and the available TP metrics with the
              NDS formula of nuScenes, i.e. weighting mAP 5 and each TP metric 1.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

import numpy as np
import rclpy.logging
import tf2_perception_msgs  # noqa: F401, registers the tf2 transform of perception_msgs/ObjectList
from autonomy_datasets_msgs.msg import ObjectListMetaInfo
from autonomy_evaluation.evaluations.Evaluation import Evaluation
from autonomy_evaluation.utils.ObjectDetectionUtils import ObjectDetectionUtils
from perception_msgs.msg import ObjectClassification, ObjectList
from perception_msgs_utils.checks import InvalidStateCovarianceSizeError
from perception_msgs_utils.constants import CONTINUOUS_STATE_COVARIANCE_INVALID
from perception_msgs_utils.convenience_state_getters import get_continuous_state_covariance_at
from perception_msgs_utils.state_getters import (
    get_height,
    get_length,
    get_vel_lat,
    get_vel_lon,
    get_width,
    get_x,
    get_y,
    get_yaw,
)
from perception_msgs_utils.state_index import index_vel_lat, index_vel_lon
from shapely.affinity import rotate as shapely_rotate
from shapely.geometry import box as shapely_box
from shapely.geometry import Point
from tf2_ros import TransformException

# Evaluated classes of perception_msgs/ObjectClassification by their names, in the order of their types
_CLASS_NAMES: Dict[int, str] = {
    ObjectClassification.PEDESTRIAN: "pedestrian",
    ObjectClassification.BICYCLE: "bicycle",
    ObjectClassification.MOTORCYCLE: "motorcycle",
    ObjectClassification.CAR: "car",
    ObjectClassification.UTILITY: "utility",
    ObjectClassification.BUS: "bus",
    ObjectClassification.ANIMAL: "animal",
    ObjectClassification.VRU: "vru",
    ObjectClassification.MICRO: "micro",
}

# Types that are never evaluated: UNKNOWN is definitely none of the defined classes, UNCLASSIFIED may be any of them
_NOT_EVALUATED_TYPES: FrozenSet[int] = frozenset({ObjectClassification.UNKNOWN, ObjectClassification.UNCLASSIFIED})

# Deprecated types by the types replacing them, as perception_msgs/ObjectClassification documents. TRUCK and MOTORBIKE
# share the values of UTILITY and MOTORCYCLE, and BIKE_UNION is replaced by both classes it unites.
_DEPRECATED_TYPES: Dict[int, Tuple[int, ...]] = {
    ObjectClassification.VAN: (ObjectClassification.CAR,),
    ObjectClassification.CAR_UNION: (ObjectClassification.CAR,),
    ObjectClassification.TRAIN: (ObjectClassification.UTILITY,),
    ObjectClassification.TRAILER: (ObjectClassification.UTILITY,),
    ObjectClassification.TRUCK_UNION: (ObjectClassification.UTILITY,),
    ObjectClassification.BIKE_UNION: (ObjectClassification.BICYCLE, ObjectClassification.MOTORCYCLE),
    ObjectClassification.ROAD_OBSTACLE: (ObjectClassification.UNKNOWN,),
}

# 'original_class' annotations of bike racks in the label meta information (nuScenes, MAN TruckScenes)
_BIKE_RACK_ORIGINAL_CLASSES: FrozenSet[str] = frozenset({"static_object.bicycle_rack"})

# Separator joining the names of the classes an evaluated class or a label consists of, e.g. "pedestrian|vru"
_CLASS_SEPARATOR = "|"

# TP error metrics, of which AVE is only available for labels with a velocity
_TP_METRICS: Tuple[str, ...] = ("ate", "ase", "aoe", "ave")

_ORIGIN: Dict[str, float] = {"x": 0.0, "y": 0.0}

_LOGGER = rclpy.logging.get_logger("object_detection_3d")

# Undefined classification types that have been warned about, so that each is reported once
_WARNED_UNDEFINED_TYPES: set = set()


def _normalized_types(class_type: int) -> Tuple[int, ...]:
    """Map a classification type onto the types it is evaluated as.

    Args:
        class_type: Type of a ``perception_msgs/ObjectClassification``.

    Returns:
        The type itself for an evaluated class, UNKNOWN and UNCLASSIFIED; the
        types replacing a deprecated type; UNKNOWN for a type that
        ``perception_msgs/ObjectClassification`` does not define, which is
        warned about once.
    """

    if class_type in _CLASS_NAMES or class_type in _NOT_EVALUATED_TYPES:
        return (class_type,)
    if class_type in _DEPRECATED_TYPES:
        return _DEPRECATED_TYPES[class_type]
    if class_type not in _WARNED_UNDEFINED_TYPES:
        _WARNED_UNDEFINED_TYPES.add(class_type)
        _LOGGER.warning(
            f"Classification type {class_type} is not defined by perception_msgs/ObjectClassification, evaluating it as UNKNOWN"
        )
    return (ObjectClassification.UNKNOWN,)


def _class_types(classifications: Iterable[Any]) -> FrozenSet[int]:
    """Determine the classes of an object from its classifications.

    The classes are the normalized types that share the highest probability. A
    dataset assigns all possible classes of an object it cannot tell the class
    of with the same probability, so they are kept together; a detection
    usually has a single most probable class.

    Args:
        classifications: The ``perception_msgs/ObjectClassification`` entries of
            the object's state.

    Returns:
        The types of the object's classes; UNCLASSIFIED if it carries no
        classification.
    """

    probabilities: Dict[int, float] = {}
    for classification in classifications:
        for class_type in _normalized_types(classification.type):
            if classification.probability > probabilities.get(class_type, -math.inf):
                probabilities[class_type] = classification.probability
    if not probabilities:
        return frozenset({ObjectClassification.UNCLASSIFIED})
    highest = max(probabilities.values())
    return frozenset(class_type for class_type, probability in probabilities.items() if probability == highest)


def _velocity_is_set(obj: Any) -> bool:
    """Decide whether the velocity of a label is set.

    ``perception_msgs`` marks a state entry that is not set with
    ``CONTINUOUS_STATE_COVARIANCE_INVALID`` on the diagonal of the state
    covariance. Datasets that annotate no velocity leave it there, which
    ``perception_msgs_utils`` setters replace when a velocity is set.

    Args:
        obj: A ``perception_msgs/Object``.

    Returns:
        Whether the longitudinal and lateral velocity are both set; ``False``
        for an object without state covariance.
    """

    model_id = obj.state.model_id
    try:
        return all(
            get_continuous_state_covariance_at(obj, index, index) != CONTINUOUS_STATE_COVARIANCE_INVALID
            for index in (index_vel_lon(model_id), index_vel_lat(model_id))
        )
    except InvalidStateCovarianceSizeError:
        return False


def _evaluated_classes(label_classes: Iterable[str]) -> Dict[str, Tuple[str, ...]]:
    """Join the classes that the labels do not distinguish into the evaluated classes.

    Every class of ``perception_msgs/ObjectClassification`` belongs to exactly
    one evaluated class, and all possible classes of a label belong to the same
    one. Classes linked through the possible classes of labels are therefore
    joined, e.g. Waymo's vehicles into ``motorcycle|car|utility|bus|micro``.

    Args:
        label_classes: Possible classes of the positive labels, each as their
            names joined by ``|``.

    Returns:
        The names of the classes each evaluated class consists of, in the order
        of their types, by the name of the evaluated class, which joins them
        with ``|``.
    """

    joined: Dict[str, set] = {name: {name} for name in _CLASS_NAMES.values()}
    for key in label_classes:
        members = set().union(*(joined[name] for name in key.split(_CLASS_SEPARATOR)))
        for name in members:
            joined[name] = members
    evaluated_classes: Dict[str, Tuple[str, ...]] = {}
    for name in _CLASS_NAMES.values():
        members = tuple(member for member in _CLASS_NAMES.values() if member in joined[name])
        evaluated_classes.setdefault(_CLASS_SEPARATOR.join(members), members)
    return evaluated_classes


class ObjectDetection3D(Evaluation):
    """Evaluation of 3D object detection on the classes of ``perception_msgs/ObjectClassification``.

    See the module docstring for the full evaluation settings. Predictions and
    labels are compared on the classes of
    ``perception_msgs/ObjectClassification``, so models can be evaluated on
    datasets of different class taxonomies; classes that the labels of a
    dataset do not distinguish are evaluated together.
    """

    VERSION = "1.0.0"
    RELEASE_NOTES = {
        "1.0.0": "Initial implementation based on adapted nuScenes detection benchmark",
    }

    def __init__(self) -> None:
        """Configure the thresholds, class ranges and filters, following nuScenes."""
        super().__init__(
            name="object_detection_3d",
            description=(
                "3D bounding-box object detection on the classes of perception_msgs/ObjectClassification, "
                "scored with the metrics of the nuScenes detection benchmark."
            ),
        )

        # BEV distance thresholds used for AP and TP-metric computation.
        self.matching_thresholds: List[float] = [0.5, 1.0, 2.0, 4.0]
        # Distance threshold used exclusively for TP error metrics.
        self.tp_metric_threshold: float = 2.0
        # Minimum precision required for a P-R point to enter AP integration.
        self.min_precision: float = 0.1
        # Minimum recall required for a P-R point to enter AP integration.
        self.min_recall: float = 0.1
        # Per-class maximum BEV detection distance in meters, following the ranges of the corresponding nuScenes classes;
        # vru, micro and animal, which nuScenes does not evaluate, use the range of pedestrians. Predictions and labels
        # at or beyond the range of their classes are excluded before matching; an object of several possible classes
        # uses the largest of their ranges, and a class without range is not limited.
        self.class_ranges: Dict[str, float] = {
            "pedestrian": 40.0,
            "bicycle": 40.0,
            "motorcycle": 40.0,
            "car": 50.0,
            "utility": 50.0,
            "bus": 50.0,
            "animal": 40.0,
            "vru": 40.0,
            "micro": 40.0,
        }
        # Classes excluded from predictions and labels when their BEV center falls inside a bike rack.
        self.bike_rack_filtered_classes: FrozenSet[str] = frozenset({"bicycle", "motorcycle"})
        # Weight of mAP in the detection score, relative to a weight of 1 per available TP metric.
        self.map_weight: float = 5.0
        self.compute_dist_func = ObjectDetectionUtils.compute_dist_bev
        self.compute_ap_func = ObjectDetectionUtils.compute_ap_101_point

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def required_inputs(self) -> Dict[str, Any]:
        """Define the evaluated output of the system under test.

        Returns:
            Input name to ROS message type: the ``prediction`` object list of
            the detector, matching the ``compute_sample_metrics`` parameter.
        """

        return {"prediction": ObjectList}

    def required_ground_truth(self) -> Dict[str, Any]:
        """Define the labels the predictions are compared with.

        Returns:
            Ground-truth name to ROS message type: the ``label`` object list,
            matching the ``compute_sample_metrics`` parameter.
        """

        return {"label": ObjectList}

    def optional_ground_truth(self) -> Dict[str, Any]:
        """Define the meta information of the labels, which not every dataset publishes.

        ``label_meta_info`` carries the dataset annotations of the ``label``
        object list that ``perception_msgs/Object`` cannot express. Only the
        ``original_class`` of bike racks is read, for the bike-rack filter.

        Returns:
            Ground-truth name to ROS message type, matching the
            ``compute_sample_metrics`` parameter.
        """

        return {"label_meta_info": ObjectListMetaInfo}

    def derived_topics(self) -> Dict[str, Tuple[str, str]]:
        """Follow the topic of the labels with the topic of their meta information.

        The dataset publishes the meta information of an object list next to
        it, on ``<label topic>/meta_info``, so only the ``label`` topic needs
        to be configured.

        Returns:
            ``label_meta_info`` derived from ``label`` with suffix ``/meta_info``.
        """

        return {"label_meta_info": ("label", "/meta_info")}

    def compute_sample_metrics(
        self,
        prediction: Any,
        label: Any,
        sample_id: Optional[str] = None,
        label_meta_info: Any = None,
    ) -> Dict[str, Any]:
        """Pass 1 - match predictions to labels for one frame.

        Transforms the predictions into the frame of the labels
        (:meth:`_in_label_frame`), extracts and filters the objects
        (:meth:`_prepare_objects`), then greedily matches them by BEV distance
        at every threshold
        (:meth:`_match`). Matching does not depend on how the classes are
        evaluated, which is only decided once all samples are known
        (:meth:`compute_aggregated_metrics`).

        Args:
            prediction: Predicted objects (``perception_msgs/ObjectList``).
            label: Ground-truth objects (``perception_msgs/ObjectList``).
            sample_id: Optional frame identifier (unused).
            label_meta_info: The dataset annotations of ``label``
                (``autonomy_datasets_msgs/ObjectListMetaInfo``) received on
                ``<label topic>/meta_info``, or ``None`` if the dataset
                publishes none.

        Returns:
            ``{"sample_prediction_num", "sample_ground_truth_num",
            "sample_dont_care_num", "label_classes", "velocity_label_classes",
            "prediction_classes", "match_records"}``: the numbers of matched
            predictions, positive labels and don't-care labels; the numbers of
            positive labels, of positive labels with velocity and of predictions
            by their classes joined with ``|``; and ``match_records`` =
            ``threshold → [entry]`` with one entry per prediction that is a
            true or false positive (see :meth:`_entry`).

        Raises:
            ValueError: ``prediction`` cannot be transformed into the frame of ``label``.
        """

        predictions, labels = self._prepare_objects(self._in_label_frame(prediction, label), label, label_meta_info)
        positives = [record for record in labels if record["positive"]]
        candidates = self._candidates(predictions, labels)

        match_records: Dict[float, List[Dict[str, Any]]] = {}
        for threshold in self.matching_thresholds:
            entries: List[Dict[str, Any]] = []
            with_errors = threshold == self.tp_metric_threshold
            for record, label_index in zip(predictions, self._match(candidates, threshold)):
                if label_index == -1:
                    entries.append(self._entry(record))
                elif labels[label_index]["positive"]:
                    entries.append(self._entry(record, labels[label_index], with_errors))
                # a prediction matched to a don't-care label is neither a true nor a false positive
            match_records[threshold] = entries

        return {
            "sample_prediction_num": len(predictions),
            "sample_ground_truth_num": len(positives),
            "sample_dont_care_num": len(labels) - len(positives),
            "label_classes": Counter(_CLASS_SEPARATOR.join(record["classes"]) for record in positives),
            "velocity_label_classes": Counter(
                _CLASS_SEPARATOR.join(record["classes"]) for record in positives if record["velocity_set"]
            ),
            "prediction_classes": Counter(_CLASS_SEPARATOR.join(record["classes"]) for record in predictions),
            "match_records": match_records,
        }

    def visualization_outputs(self) -> Dict[str, Any]:
        """Define the ROS message types published when visualization is enabled.

        Returns:
            Output name to ROS message type. The keys match those of
            :meth:`visualize_sample` and are node-relative topic names.
        """

        return {
            "true_positives": ObjectList,
            "false_positives": ObjectList,
            "false_negatives": ObjectList,
            "ignored": ObjectList,
        }

    def visualize_sample(
        self,
        prediction: Any,
        label: Any,
        sample_id: Optional[str] = None,
        label_meta_info: Any = None,
    ) -> Dict[str, Any]:
        """Split one frame's objects into true positives, false positives, false negatives and ignored objects.

        Runs the same extraction, filtering and matching as
        :meth:`compute_sample_metrics` at the TP metric threshold
        (:attr:`tp_metric_threshold`), and hands back the source messages behind
        the outcome: a prediction that claimed a positive label is a true
        positive, a prediction that claimed none is a false positive, and a
        positive label no prediction claimed is a false negative. Don't-care
        labels and the predictions that claimed them are ignored.

        Objects dropped before matching (predictions without evaluated class,
        labels of UNKNOWN class only, bicycles in bike racks, objects beyond the
        range of their classes) appear in none of the lists.

        Args:
            prediction: Predicted objects (``perception_msgs/ObjectList``).
            label: Ground-truth objects (``perception_msgs/ObjectList``).
            sample_id: Optional frame identifier (unused).
            label_meta_info: The dataset annotations of ``label``
                (``autonomy_datasets_msgs/ObjectListMetaInfo``), or ``None``.

        Returns:
            Output name to ``perception_msgs/ObjectList``, keyed as in
            :meth:`visualization_outputs`. The positives carry predicted
            objects and the header of ``prediction``, both transformed into the
            frame of ``label``; the false negatives carry ground-truth objects
            and the header of ``label``, as do the ignored objects, which hold
            both don't-care labels and the predictions that claimed them.

        Raises:
            ValueError: ``prediction`` cannot be transformed into the frame of ``label``.
        """

        prediction = self._in_label_frame(prediction, label)
        predictions, labels = self._prepare_objects(prediction, label, label_meta_info)
        matches = self._match(self._candidates(predictions, labels), self.tp_metric_threshold)

        true_positives: List[Any] = []
        false_positives: List[Any] = []
        ignored: List[Any] = []
        for record, label_index in zip(predictions, matches):
            if label_index == -1:
                false_positives.append(record["object"])
            elif labels[label_index]["positive"]:
                true_positives.append(record["object"])
            else:
                ignored.append(record["object"])
        claimed = set(matches)
        false_negatives = [record["object"] for index, record in enumerate(labels) if record["positive"] and index not in claimed]
        ignored += [record["object"] for record in labels if not record["positive"]]

        return {
            "true_positives": ObjectList(header=prediction.header, objects=true_positives),
            "false_positives": ObjectList(header=prediction.header, objects=false_positives),
            "false_negatives": ObjectList(header=label.header, objects=false_negatives),
            "ignored": ObjectList(header=label.header, objects=ignored),
        }

    def compute_aggregated_metrics(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pass 2 - aggregate per-frame match records into dataset-level metrics.

        The evaluated classes are derived from the labels of all samples
        recorded so far together with the given ones, so that the results of
        each scene are reported on the same classes as the results of all
        samples. Each true positive is attributed to the evaluated class of its
        label, each false positive to the evaluated classes of its own classes.

        Args:
            sample_results: List of ``{"sample_id": ..., "metrics": {...,
                "match_records": ...}}`` entries produced by
                :meth:`record_sample`.

        Returns:
            The metrics, each holding its aggregated value under ``_value_``
            next to its sub-metrics: the detection score (``_value_``), the
            numbers of positive labels (``num_labels``) and of predictions
            (``num_predictions``) in total and per evaluated class with labels
            or predictions, the labels of each class also by their possible
            classes joined with ``|``, the AP (``ap``) as mAP over all
            thresholds, per class (``classes``) and per threshold
            (``dist_{threshold}``) as mAP with the AP of each class under
            ``classes``, and the TP errors at the 2 m threshold (``ate``,
            ``ase``, ``aoe``, ``ave``) as mean and per class. AP and TP errors
            are reported for the evaluated classes with labels only.
        """

        all_metrics = [entry["metrics"] for entry in sample_results]
        recorded_label_classes = (key for entry in self._sample_results for key in entry["metrics"]["label_classes"])
        evaluated_classes = _evaluated_classes(
            {*recorded_label_classes, *(key for m in all_metrics for key in m["label_classes"])}
        )
        class_of = {member: name for name, members in evaluated_classes.items() for member in members}

        label_counts = sum((Counter(metrics["label_classes"]) for metrics in all_metrics), Counter())
        velocity_label_counts = sum((Counter(metrics["velocity_label_classes"]) for metrics in all_metrics), Counter())
        prediction_counts = sum((Counter(metrics["prediction_classes"]) for metrics in all_metrics), Counter())
        num_labels = self._counts_by_evaluated_class(label_counts, class_of)
        num_velocity_labels = self._counts_by_evaluated_class(velocity_label_counts, class_of)
        num_predictions = self._counts_by_evaluated_class(prediction_counts, class_of)
        # Objects are counted for the evaluated classes with labels or predictions, metrics are computed for those with
        # labels, both in the order of their classes.
        counted_classes = [name for name in evaluated_classes if num_labels.get(name) or num_predictions.get(name)]
        num_labels = {name: num_labels[name] for name in evaluated_classes if num_labels.get(name)}
        # The labels of each evaluated class by their possible classes, which all belong to it, e.g. NVIDIA's persons
        # as "pedestrian|vru" and its strollers as "vru" within "pedestrian|vru".
        label_classes: Dict[str, Dict[str, int]] = {name: {} for name in counted_classes}
        for key, count in label_counts.items():
            label_classes[class_of[key.split(_CLASS_SEPARATOR)[0]]][key] = count

        entries = self._entries_by_evaluated_class(all_metrics, class_of)
        ap = self._compute_ap(entries, num_labels)
        tp_metrics = self._compute_tp_metrics(entries.get(self.tp_metric_threshold, {}), num_labels, num_velocity_labels)

        return {
            "_value_": self._detection_score(ap["_value_"], {metric: tp_metrics[metric]["_value_"] for metric in _TP_METRICS}),
            "num_labels": {
                "_value_": sum(label_counts.values()),
                **{name: {"_value_": num_labels.get(name, 0), **label_classes[name]} for name in counted_classes},
            },
            # a prediction of classes of several evaluated classes counts for each of them, but once in total
            "num_predictions": {
                "_value_": sum(prediction_counts.values()),
                **{name: num_predictions.get(name, 0) for name in counted_classes},
            },
            "ap": ap,
            **tp_metrics,
        }

    # ------------------------------------------------------------------
    # Helper functions for extraction and matching
    # ------------------------------------------------------------------

    @staticmethod
    def _index_meta_info(meta_info: Any) -> Dict[int, Dict[str, List[str]]]:
        """Index an ``ObjectListMetaInfo`` message by object ID.

        Args:
            meta_info: An ``autonomy_datasets_msgs/ObjectListMetaInfo``, or
                ``None`` when the frame carries no meta information.

        Returns:
            Object ID to that object's key-to-values annotations. A key may be
            published more than once per object (e.g. several attributes), so
            values are lists in publication order. Objects without any meta
            information are absent from the index.
        """

        index: Dict[int, Dict[str, List[str]]] = {}
        if meta_info is None:
            return index
        for entry in meta_info.objects:
            annotations = index.setdefault(entry.id, {})
            for key_value in entry.info:
                annotations.setdefault(key_value.key, []).append(key_value.value)
        return index

    def _extract_objects(self, objects: Any, meta_info: Any = None, is_label: bool = False) -> List[Dict[str, Any]]:
        """Extract per-object metric records from a frame's ``ObjectList``.

        Position, dimensions, yaw and velocity are read via each object's own
        motion model (selected by ``state.model_id``); ``types`` are the
        object's classes (:func:`_class_types`), ``classes`` the names of the
        evaluated ones among them, and ``confidence_score`` is its
        ``existence_probability``. Labels additionally record whether they are
        positive, whether their velocity is set and, from their meta
        information, whether they are a bike rack.

        Args:
            objects: A ``perception_msgs/ObjectList``.
            meta_info: The matching ``autonomy_datasets_msgs/ObjectListMetaInfo``,
                whose entries are matched to the objects by ID, or ``None``.
            is_label: ``True`` for ground-truth labels, ``False`` for predictions.

        Returns:
            Object records (dicts); ``object`` is the ``perception_msgs/Object``
            each came from, so the matching outcome can be published for
            visualization.

        Raises:
            UnknownStateEntryError: an object whose ``state.model_id`` is not a
                known motion model (raised by the perception getters).
        """

        meta_index = self._index_meta_info(meta_info)

        records: List[Dict[str, Any]] = []
        for obj in objects.objects:
            yaw = get_yaw(obj)
            vel_lon = get_vel_lon(obj)
            vel_lat = get_vel_lat(obj)
            types = _class_types(obj.state.classifications)
            record = {
                "x": get_x(obj),
                "y": get_y(obj),
                "width": get_width(obj),
                "length": get_length(obj),
                "height": get_height(obj),
                "yaw": yaw,
                "vx": vel_lon * math.cos(yaw) - vel_lat * math.sin(yaw),
                "vy": vel_lon * math.sin(yaw) + vel_lat * math.cos(yaw),
                "types": types,
                # names of the evaluated classes among the types, in the order of their types
                "classes": tuple(name for class_type, name in _CLASS_NAMES.items() if class_type in types),
                "confidence_score": float(obj.existence_probability),
                "object": obj,
            }
            if is_label:
                # a label whose possible classes are all evaluated counts as true positive or false negative,
                # any other is a don't-care object
                record["positive"] = types.isdisjoint(_NOT_EVALUATED_TYPES)
                record["velocity_set"] = _velocity_is_set(obj)
                original_classes = meta_index.get(obj.id, {}).get("original_class", [])
                record["bike_rack"] = not _BIKE_RACK_ORIGINAL_CLASSES.isdisjoint(original_classes)
            records.append(record)
        return records

    def _in_label_frame(self, prediction: Any, label: Any) -> Any:
        """Transform the predictions into the frame of the labels, so that both can be compared.

        Predictions in another frame than the labels, e.g. lidar-frame
        detections of vehicle-frame labels, are transformed with the transform
        :attr:`tf_buffer` holds at the stamp of the predictions, which is warned
        about once. Only the poses of the objects are transformed; their
        velocities are given along their heading and follow its yaw. An empty
        frame cannot be checked, so it is assumed to be the frame of the other
        object list.

        Args:
            prediction: Predicted objects (``perception_msgs/ObjectList``).
            label: Ground-truth objects (``perception_msgs/ObjectList``).

        Returns:
            ``prediction`` itself if it is given in the frame of ``label``, else
            a copy transformed into that frame.

        Raises:
            ValueError: no transform from the frame of ``prediction`` into the
                frame of ``label`` is available at the stamp of ``prediction``.
        """

        prediction_frame, label_frame = prediction.header.frame_id, label.header.frame_id
        if not prediction_frame or not label_frame or prediction_frame == label_frame:
            return prediction
        _LOGGER.warning(
            f"Predictions in frame '{prediction_frame}' are transformed into frame '{label_frame}' of the labels",
            once=True,
        )
        error = f"Predictions in frame '{prediction_frame}' cannot be transformed into frame '{label_frame}' of the labels"
        if self.tf_buffer is None:
            raise ValueError(f"{error}, as no transforms are available")
        try:
            # the tf2 transform of perception_msgs/ObjectList modifies the message it transforms
            return self.tf_buffer.transform(copy.deepcopy(prediction), label_frame)
        except TransformException as exception:
            raise ValueError(f"{error}: {exception}") from exception

    def _prepare_objects(
        self,
        prediction: Any,
        label: Any,
        label_meta_info: Any = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract the objects of a frame and apply the filters before matching.

        1. **Class filter**: drop predictions without an evaluated class and
           labels of UNKNOWN class only, which no prediction can match.
        2. **Bike-rack filter**: drop bicycle/motorcycle predictions and labels
           whose BEV center falls inside a bike rack (labels whose meta
           information names them a bike rack).
        3. **Range filter**: drop objects at or beyond the range of their
           classes (:attr:`class_ranges`).

        Args:
            prediction: Predicted objects (``perception_msgs/ObjectList``) in
                the frame of ``label``.
            label: Ground-truth objects (``perception_msgs/ObjectList``).
            label_meta_info: The dataset annotations of ``label``, or ``None``.

        Returns:
            ``(predictions, labels)`` ready for matching, the predictions in
            descending order of confidence.
        """

        predictions = self._extract_objects(prediction)
        labels = self._extract_objects(label, label_meta_info, is_label=True)
        racks = self._bike_rack_footprints([record for record in labels if record["bike_rack"]])

        predictions = [record for record in predictions if record["classes"] and self._is_kept(record, racks)]
        labels = [
            record
            for record in labels
            if (ObjectClassification.UNCLASSIFIED in record["types"] or record["classes"]) and self._is_kept(record, racks)
        ]
        return sorted(predictions, key=lambda record: -record["confidence_score"]), labels

    def _is_kept(self, record: Dict[str, Any], racks: List[Any]) -> bool:
        """Apply the bike-rack and range filters to an object.

        Args:
            record: Prediction or label record.
            racks: BEV footprints of the bike racks of the frame.

        Returns:
            Whether the object is neither a bicycle or motorcycle inside a bike
            rack nor at or beyond the range of its classes.
        """

        names = record["classes"]
        if racks and names and set(names) <= self.bike_rack_filtered_classes:
            center = Point(record["x"], record["y"])
            if any(rack.contains(center) for rack in racks):
                return False
        if ObjectClassification.UNCLASSIFIED in record["types"]:
            names = tuple(_CLASS_NAMES.values())  # an unclassified object may be of any class
        detection_range = max((self.class_ranges.get(name, math.inf) for name in names), default=math.inf)
        return self.compute_dist_func(record, _ORIGIN) < detection_range

    @staticmethod
    def _bike_rack_footprints(racks: List[Dict[str, Any]]) -> List[Any]:
        """Build the BEV footprint of each bike rack.

        Args:
            racks: Bike-rack label records.

        Returns:
            One Shapely polygon per rack: a rectangle of its length along its
            heading and its width across, centered at the rack.
        """

        footprints = []
        for rack in racks:
            half_length, half_width = rack["length"] / 2.0, rack["width"] / 2.0
            footprint = shapely_box(
                rack["x"] - half_length, rack["y"] - half_width, rack["x"] + half_length, rack["y"] + half_width
            )
            footprints.append(shapely_rotate(footprint, rack["yaw"], origin=(rack["x"], rack["y"]), use_radians=True))
        return footprints

    def _candidates(self, predictions: List[Dict[str, Any]], labels: List[Dict[str, Any]]) -> List[List[Tuple[float, bool, int]]]:
        """List the labels each prediction may match, nearest first.

        Args:
            predictions: Prediction records, highest confidence first.
            labels: Label records.

        Returns:
            Per prediction, ``(distance, don't-care, label index)`` of every
            label it is compatible with, ascending: of equally distant labels,
            positive ones precede don't-care ones and earlier ones later ones.
        """

        return [
            sorted(
                (self.compute_dist_func(prediction, label), not label["positive"], index)
                for index, label in enumerate(labels)
                # all classes of the prediction must be possible classes of the label, any class for an unclassified one
                if ObjectClassification.UNCLASSIFIED in label["types"] or prediction["types"] <= label["types"]
            )
            for prediction in predictions
        ]

    @staticmethod
    def _match(candidates: List[List[Tuple[float, bool, int]]], threshold: float) -> List[int]:
        """Greedily match the predictions of a frame to its labels.

        Predictions, highest confidence first, each claim the nearest unclaimed
        label they are compatible with whose BEV center is closer than
        ``threshold``. For labels of a single class this is the per-class
        matching of nuScenes. Shared by scoring (:meth:`compute_sample_metrics`)
        and visualization (:meth:`visualize_sample`) so both classify
        identically.

        Args:
            candidates: Labels each prediction may match, from :meth:`_candidates`.
            threshold: BEV center distance a match must stay below.

        Returns:
            One label index per prediction, in the order of the predictions:
            the label it claimed, or ``-1`` when it matched nothing.
        """

        claimed: set = set()
        matches: List[int] = []
        for prediction_candidates in candidates:
            match = next(
                (index for distance, _, index in prediction_candidates if distance < threshold and index not in claimed),
                -1,
            )
            if match != -1:
                claimed.add(match)
            matches.append(match)
        return matches

    def _entry(
        self,
        prediction: Dict[str, Any],
        label: Optional[Dict[str, Any]] = None,
        with_errors: bool = False,
    ) -> Dict[str, Any]:
        """Record the outcome of a prediction for aggregation.

        Args:
            prediction: Prediction record.
            label: The positive label the prediction matched, or ``None`` for a
                false positive.
            with_errors: Whether to compute the TP errors of a true positive.

        Returns:
            ``{"classes", "confidence_score", "is_tp", "label_classes"}`` with
            the names of the prediction's and its label's evaluated classes
            joined by ``|`` (``label_classes`` is ``None`` for a false
            positive); a true positive computed ``with_errors`` also holds
            ``ate``, ``ase``, ``aoe`` and ``ave``, of which ``ave`` is ``None``
            if the velocity of the label is not set.
        """

        entry: Dict[str, Any] = {
            "classes": _CLASS_SEPARATOR.join(prediction["classes"]),
            "confidence_score": prediction["confidence_score"],
            "is_tp": label is not None,
            "label_classes": None if label is None else _CLASS_SEPARATOR.join(label["classes"]),
        }
        if label is None or not with_errors:
            return entry

        # ATE: BEV Euclidean translation error.
        entry["ate"] = math.hypot(prediction["x"] - label["x"], prediction["y"] - label["y"])
        # ASE: 1 - 3D IoU after aligning centers and orientation.
        intersection = (
            min(prediction["width"], label["width"])
            * min(prediction["height"], label["height"])
            * min(prediction["length"], label["length"])
        )
        union = (
            prediction["width"] * prediction["height"] * prediction["length"] + label["width"] * label["height"] * label["length"]
        )
        union -= intersection
        entry["ase"] = 1.0 - (intersection / union if union > 0 else 0.0)
        # AOE: smallest yaw angle difference.
        yaw_difference = (prediction["yaw"] - label["yaw"]) % (2.0 * math.pi)
        entry["aoe"] = min(yaw_difference, 2.0 * math.pi - yaw_difference)
        # AVE: velocity error in m/s, only against a label whose velocity is set.
        entry["ave"] = (
            math.hypot(prediction["vx"] - label["vx"], prediction["vy"] - label["vy"]) if label["velocity_set"] else None
        )
        return entry

    # ------------------------------------------------------------------
    # Helper functions for metric computation
    # ------------------------------------------------------------------

    @staticmethod
    def _counts_by_evaluated_class(counts: Dict[str, int], class_of: Dict[str, str]) -> Dict[str, int]:
        """Attribute counts of objects to the evaluated classes they belong to.

        Args:
            counts: The classes joined by ``|`` to the number of objects of
                these classes.
            class_of: Class name to the name of its evaluated class.

        Returns:
            Evaluated class to the number of objects of which one of the
            classes belongs to it; an object of classes of several evaluated
            classes counts for each of them.
        """

        totals: Dict[str, int] = {}
        for classes, count in counts.items():
            for name in {class_of[member] for member in classes.split(_CLASS_SEPARATOR)}:
                totals[name] = totals.get(name, 0) + count
        return totals

    def _entries_by_evaluated_class(
        self,
        all_metrics: List[Dict[str, Any]],
        class_of: Dict[str, str],
    ) -> Dict[float, Dict[str, List[Dict[str, Any]]]]:
        """Gather the per-frame entries of all thresholds by evaluated class.

        A true positive belongs to the evaluated class of its label, which also
        holds its classes. A false positive counts against the evaluated class
        of each of its classes.

        Args:
            all_metrics: Per-frame metrics from :meth:`compute_sample_metrics`.
            class_of: Class name to the name of its evaluated class.

        Returns:
            ``threshold → evaluated class → [entry]``, in frame order.
        """

        entries: Dict[float, Dict[str, List[Dict[str, Any]]]] = {}
        for metrics in all_metrics:
            for threshold, frame_entries in metrics["match_records"].items():
                by_class = entries.setdefault(threshold, {})
                for entry in frame_entries:
                    classes = entry["label_classes"] if entry["is_tp"] else entry["classes"]
                    for name in {class_of[member] for member in classes.split(_CLASS_SEPARATOR)}:
                        by_class.setdefault(name, []).append(entry)
        return entries

    def _average_precision(self, entries: List[Dict[str, Any]], num_labels: int) -> float:
        """Compute the AP of one class at one threshold.

        Sorts the entries globally by confidence descending and accumulates
        precision and recall over the full dataset, which are integrated by
        :attr:`compute_ap_func` with the ``min_recall`` / ``min_precision``
        gating (a perfect detector scores AP = 1.0).

        Args:
            entries: True and false positives of the class.
            num_labels: Number of positive labels of the class.

        Returns:
            The average precision, 0.0 without predictions.
        """

        if not entries:
            return 0.0
        is_tp = np.array([entry["is_tp"] for entry in sorted(entries, key=lambda entry: -entry["confidence_score"])])
        cum_tp = np.cumsum(is_tp)
        cum_fp = np.cumsum(~is_tp)
        precision = cum_tp / (cum_tp + cum_fp)
        recall = cum_tp / num_labels
        return self.compute_ap_func(recall, precision, self.min_recall, self.min_precision)

    def _compute_ap(
        self,
        entries: Dict[float, Dict[str, List[Dict[str, Any]]]],
        num_labels: Dict[str, int],
    ) -> Dict[str, Any]:
        """Compute the AP per threshold and class, and the mAP per threshold, per class and overall.

        AP is computed for every evaluated class with labels; a class without
        labels has no AP and does not enter the mAP, as its predictions can only
        be false positives.

        Args:
            entries: ``threshold → evaluated class → [entry]``.
            num_labels: Evaluated classes with labels to their number of labels.

        Returns:
            ``{"_value_": mAP, "classes": {class: mAP}, "dist_{threshold}":
            {"_value_": mAP, "classes": {class: AP}}}`` with the mAP over all
            thresholds and classes, the mAP of each class over all thresholds,
            and the mAP and APs of each threshold; the mAPs over classes are
            ``None`` without any labels.
        """

        thresholds: Dict[str, Dict[str, Any]] = {}
        for threshold in self.matching_thresholds:
            by_class = entries.get(threshold, {})
            aps = {name: self._average_precision(by_class.get(name, []), count) for name, count in num_labels.items()}
            thresholds[f"dist_{threshold}"] = {"_value_": float(np.mean(list(aps.values()))) if aps else None, "classes": aps}
        class_maps = {name: float(np.mean([metrics["classes"][name] for metrics in thresholds.values()])) for name in num_labels}
        return {"_value_": float(np.mean(list(class_maps.values()))) if class_maps else None, "classes": class_maps, **thresholds}

    def _compute_tp_metrics(
        self,
        entries: Dict[str, List[Dict[str, Any]]],
        num_labels: Dict[str, int],
        num_velocity_labels: Dict[str, int],
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """Compute the TP error metrics per class and their means, at the TP metric threshold.

        Cumulative-mean errors (by descending confidence) are interpolated to
        101 recall points and averaged over ``[min_recall, max_recall]``. A
        class whose recall stays below ``min_recall`` falls back to worst-case
        ``1.0`` errors. ``ave`` only averages true positives whose label has a
        velocity: it is ``None`` for a class without any label with velocity,
        and ``1.0`` if none of its true positives has one. The mean of a metric
        is the macro mean over the classes, skipping ``None`` values.

        Args:
            entries: Evaluated class to its entries at the TP metric threshold.
            num_labels: Evaluated classes with labels to their number of labels.
            num_velocity_labels: Evaluated class to its number of labels with
                velocity.

        Returns:
            ``{metric: {"_value_": mean, class: error}}`` for ``ate``, ``ase``,
            ``aoe`` and ``ave``; the mean is ``None`` without any available
            error, only ``ave`` can be ``None`` per class.
        """

        class_errors: Dict[str, Dict[str, Optional[float]]] = {}
        for name, count in num_labels.items():
            velocity_available = num_velocity_labels.get(name, 0) > 0
            worst: Dict[str, Optional[float]] = {"ate": 1.0, "ase": 1.0, "aoe": 1.0, "ave": 1.0 if velocity_available else None}

            recall: List[float] = []
            curves: Dict[str, List[float]] = {metric: [] for metric in _TP_METRICS}
            sums = {metric: 0.0 for metric in _TP_METRICS}
            counts = {metric: 0 for metric in _TP_METRICS}
            cum_tp = 0
            for entry in sorted(entries.get(name, []), key=lambda entry: -entry["confidence_score"]):
                if entry["is_tp"]:
                    cum_tp += 1
                    for metric in _TP_METRICS:
                        if entry[metric] is not None:
                            sums[metric] += entry[metric]
                            counts[metric] += 1
                recall.append(cum_tp / count)
                for metric in _TP_METRICS:
                    curves[metric].append(sums[metric] / counts[metric] if counts[metric] else 0.0)

            average = ObjectDetectionUtils.compute_tp_101_point
            if not recall or average(np.array(recall), curves["ate"], self.min_recall) is None:
                class_errors[name] = worst
                continue
            class_errors[name] = {
                metric: average(np.array(recall), curves[metric], self.min_recall) for metric in ("ate", "ase", "aoe")
            }
            if not velocity_available:
                class_errors[name]["ave"] = None
            elif counts["ave"] == 0:
                class_errors[name]["ave"] = 1.0  # no true positive against a label with velocity
            else:
                class_errors[name]["ave"] = average(np.array(recall), curves["ave"], self.min_recall)

        tp_metrics: Dict[str, Dict[str, Optional[float]]] = {}
        for metric in _TP_METRICS:
            errors = {name: class_metrics[metric] for name, class_metrics in class_errors.items()}
            available = [error for error in errors.values() if error is not None]
            tp_metrics[metric] = {"_value_": float(np.mean(available)) if available else None, **errors}
        return tp_metrics

    def _detection_score(self, map_value: Optional[float], mean_errors: Dict[str, Optional[float]]) -> Optional[float]:
        """Combine the mAP and the available mean TP errors into the detection score.

        ::

            detection_score = (5·mAP + Σ max(1 - m{metric}, 0)) / (5 + number of TP metrics)

        This is the NDS formula of nuScenes over the available TP metrics, i.e.
        weighting mAP with :attr:`map_weight` and each TP metric with 1: AVE is
        left out when no label has a velocity.

        Args:
            map_value: mAP over all thresholds and evaluated classes, ``None``
                without any labels.
            mean_errors: TP metric to its mean error over the evaluated classes,
                ``None`` if unavailable.

        Returns:
            The detection score, ``None`` without any labels.
        """

        if map_value is None:
            return None
        available = [error for error in mean_errors.values() if error is not None]
        tp_scores = sum(max(1.0 - error, 0.0) for error in available)
        return (self.map_weight * map_value + tp_scores) / (self.map_weight + len(available))
