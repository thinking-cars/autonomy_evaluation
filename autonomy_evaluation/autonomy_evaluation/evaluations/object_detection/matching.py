# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Matching of predicted with labeled 3D objects, which all metrics of the 3D object detection share.

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

Settings, configured by the evaluation (see ``ObjectDetection3D``):

- Frames:     Predictions in another frame than the labels are transformed into
              the frame of the labels with tf2.
- Matching:   Predictions, highest confidence first, each claim the nearest
              unclaimed compatible label whose BEV center is closer than the
              threshold.
- Ranges:     Per-class max detection distance (``class_ranges``). An object of
              several possible classes uses the largest of their ranges.
- Bike-rack:  Bicycles and motorcycles (``bike_rack_filtered_classes``) whose BEV
              center falls inside a bike rack are excluded from both predictions
              and labels. Bike racks are recognized by the ``original_class`` of
              the label meta information, published on the optional
              ``<label topic>/meta_info`` topic.
- TP errors:  True positives at ``tp_metric_threshold`` carry their translation,
              scale, orientation and velocity errors.

Everything derived from a sample is derived once and shared among the metrics
through :meth:`~autonomy_evaluation.evaluations.Metric.Sample.memo`.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

import rclpy.logging
import tf2_perception_msgs  # noqa: F401, registers the tf2 transform of perception_msgs/ObjectList
from autonomy_evaluation.evaluations.Metric import Sample
from autonomy_evaluation.utils.ObjectDetectionUtils import ObjectDetectionUtils
from perception_msgs.msg import ObjectClassification
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
CLASS_NAMES: Dict[int, str] = {
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
CLASS_SEPARATOR = "|"

_ORIGIN: Dict[str, float] = {"x": 0.0, "y": 0.0}

_LOGGER = rclpy.logging.get_logger("object_detection_3d")

# Undefined classification types that have been warned about, so that each is reported once
_WARNED_UNDEFINED_TYPES: set = set()


# ----------------------------------------------------------------------
# Classes of objects
# ----------------------------------------------------------------------


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

    if class_type in CLASS_NAMES or class_type in _NOT_EVALUATED_TYPES:
        return (class_type,)
    if class_type in _DEPRECATED_TYPES:
        return _DEPRECATED_TYPES[class_type]
    if class_type not in _WARNED_UNDEFINED_TYPES:
        _WARNED_UNDEFINED_TYPES.add(class_type)
        _LOGGER.warning(
            f"Classification type {class_type} is not defined by perception_msgs/ObjectClassification, evaluating it as UNKNOWN"
        )
    return (ObjectClassification.UNKNOWN,)


def class_types(classifications: Iterable[Any]) -> FrozenSet[int]:
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


def evaluated_classes(label_classes: Iterable[str]) -> Dict[str, Tuple[str, ...]]:
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

    joined: Dict[str, set] = {name: {name} for name in CLASS_NAMES.values()}
    for key in label_classes:
        members = set().union(*(joined[name] for name in key.split(CLASS_SEPARATOR)))
        for name in members:
            joined[name] = members
    evaluated: Dict[str, Tuple[str, ...]] = {}
    for name in CLASS_NAMES.values():
        members = tuple(member for member in CLASS_NAMES.values() if member in joined[name])
        evaluated.setdefault(CLASS_SEPARATOR.join(members), members)
    return evaluated


def velocity_is_set(obj: Any) -> bool:
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


# ----------------------------------------------------------------------
# Data derived from a sample, shared by the metrics
# ----------------------------------------------------------------------


def prediction_in_label_frame(sample: Sample) -> Any:
    """Transform the predictions of a sample into the frame of its labels, so that both can be compared.

    Predictions in another frame than the labels, e.g. lidar-frame
    detections of vehicle-frame labels, are transformed with the transform
    :attr:`Sample.tf_buffer` holds at the stamp of the predictions, which is
    warned about once. Only the poses of the objects are transformed; their
    velocities are given along their heading and follow its yaw. An empty
    frame cannot be checked, so it is assumed to be the frame of the other
    object list.

    Args:
        sample: Sample holding the ``prediction`` and ``label`` object lists.

    Returns:
        ``prediction`` itself if it is given in the frame of ``label``, else
        a copy transformed into that frame.

    Raises:
        ValueError: no transform from the frame of ``prediction`` into the
            frame of ``label`` is available at the stamp of ``prediction``.
    """

    def transform() -> Any:
        prediction, label = sample["prediction"], sample["label"]
        prediction_frame, label_frame = prediction.header.frame_id, label.header.frame_id
        if not prediction_frame or not label_frame or prediction_frame == label_frame:
            return prediction
        _LOGGER.warning(
            f"Predictions in frame '{prediction_frame}' are transformed into frame '{label_frame}' of the labels",
            once=True,
        )
        error = f"Predictions in frame '{prediction_frame}' cannot be transformed into frame '{label_frame}' of the labels"
        if sample.tf_buffer is None:
            raise ValueError(f"{error}, as no transforms are available")
        try:
            # the tf2 transform of perception_msgs/ObjectList modifies the message it transforms
            return sample.tf_buffer.transform(copy.deepcopy(prediction), label_frame)
        except TransformException as exception:
            raise ValueError(f"{error}: {exception}") from exception

    return sample.memo("prediction_in_label_frame", transform)


def prepared_objects(sample: Sample) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract the objects of a sample and apply the filters before matching.

    1. **Class filter**: drop predictions without an evaluated class and
       labels of UNKNOWN class only, which no prediction can match.
    2. **Bike-rack filter**: drop the predictions and labels of the classes of
       ``bike_rack_filtered_classes`` whose BEV center falls inside a bike rack
       (labels whose meta information names them a bike rack).
    3. **Range filter**: drop objects at or beyond the range of their classes
       (``class_ranges``).

    Args:
        sample: Sample holding the ``prediction`` and ``label`` object lists
            and, if published, the ``label_meta_info`` of the labels.

    Returns:
        ``(predictions, labels)`` ready for matching, the predictions in the
        frame of the labels and in descending order of confidence.
    """

    def prepare() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        predictions = _extract_objects(prediction_in_label_frame(sample))
        labels = _extract_objects(sample["label"], sample.messages.get("label_meta_info"), is_label=True)
        racks = _bike_rack_footprints([record for record in labels if record["bike_rack"]])
        filtered_classes = frozenset(sample.config["bike_rack_filtered_classes"])
        class_ranges = sample.config["class_ranges"]

        def is_kept(record: Dict[str, Any]) -> bool:
            return _is_kept(record, racks, filtered_classes, class_ranges)

        predictions = [record for record in predictions if record["classes"] and is_kept(record)]
        labels = [
            record
            for record in labels
            if (ObjectClassification.UNCLASSIFIED in record["types"] or record["classes"]) and is_kept(record)
        ]
        return sorted(predictions, key=lambda record: -record["confidence_score"]), labels

    return sample.memo("prepared_objects", prepare)


def matches(sample: Sample, threshold: float) -> List[int]:
    """Greedily match the predictions of a sample to its labels at a threshold.

    Predictions, highest confidence first, each claim the nearest unclaimed
    label they are compatible with whose BEV center is closer than
    ``threshold``. For labels of a single class this is the per-class
    matching of nuScenes. Shared by the metrics and the visualization, so that
    both classify identically.

    Args:
        sample: Sample holding the ``prediction`` and ``label`` object lists.
        threshold: BEV center distance a match must stay below.

    Returns:
        One label index into the labels of :func:`prepared_objects` per
        prediction, in the order of the predictions: the label it claimed, or
        ``-1`` when it matched nothing.
    """

    def match() -> List[int]:
        claimed: set = set()
        label_indices: List[int] = []
        for prediction_candidates in _candidates(sample):
            label_index = next(
                (index for distance, _, index in prediction_candidates if distance < threshold and index not in claimed),
                -1,
            )
            if label_index != -1:
                claimed.add(label_index)
            label_indices.append(label_index)
        return label_indices

    return sample.memo(("matches", threshold), match)


def match_entries(sample: Sample, threshold: float) -> List[Dict[str, Any]]:
    """Record the outcome of each prediction of a sample at a threshold for aggregation.

    Args:
        sample: Sample holding the ``prediction`` and ``label`` object lists.
        threshold: BEV center distance a match must stay below.

    Returns:
        One entry per prediction that is a true or false positive, see
        :func:`_entry`; a prediction matched to a don't-care label is neither.
        At ``tp_metric_threshold``, the entries of true positives carry their
        TP errors.
    """

    def record() -> List[Dict[str, Any]]:
        predictions, labels = prepared_objects(sample)
        with_errors = threshold == sample.config["tp_metric_threshold"]
        entries: List[Dict[str, Any]] = []
        for prediction, label_index in zip(predictions, matches(sample, threshold)):
            if label_index == -1:
                entries.append(_entry(prediction))
            elif labels[label_index]["positive"]:
                entries.append(_entry(prediction, labels[label_index], with_errors))
        return entries

    return sample.memo(("match_entries", threshold), record)


# ----------------------------------------------------------------------
# Helper functions for extraction and matching
# ----------------------------------------------------------------------


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


def _extract_objects(objects: Any, meta_info: Any = None, is_label: bool = False) -> List[Dict[str, Any]]:
    """Extract per-object metric records from a frame's ``ObjectList``.

    Position, dimensions, yaw and velocity are read via each object's own
    motion model (selected by ``state.model_id``); ``types`` are the
    object's classes (:func:`class_types`), ``classes`` the names of the
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

    meta_index = _index_meta_info(meta_info)

    records: List[Dict[str, Any]] = []
    for obj in objects.objects:
        yaw = get_yaw(obj)
        vel_lon = get_vel_lon(obj)
        vel_lat = get_vel_lat(obj)
        types = class_types(obj.state.classifications)
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
            "classes": tuple(name for class_type, name in CLASS_NAMES.items() if class_type in types),
            "confidence_score": float(obj.existence_probability),
            "object": obj,
        }
        if is_label:
            # a label whose possible classes are all evaluated counts as true positive or false negative,
            # any other is a don't-care object
            record["positive"] = types.isdisjoint(_NOT_EVALUATED_TYPES)
            record["velocity_set"] = velocity_is_set(obj)
            original_classes = meta_index.get(obj.id, {}).get("original_class", [])
            record["bike_rack"] = not _BIKE_RACK_ORIGINAL_CLASSES.isdisjoint(original_classes)
        records.append(record)
    return records


def _is_kept(record: Dict[str, Any], racks: List[Any], filtered_classes: FrozenSet[str], class_ranges: Dict[str, float]) -> bool:
    """Apply the bike-rack and range filters to an object.

    Args:
        record: Prediction or label record.
        racks: BEV footprints of the bike racks of the frame.
        filtered_classes: Classes excluded inside a bike rack.
        class_ranges: Maximum BEV detection distance per class; a class without
            range is not limited.

    Returns:
        Whether the object is neither of the filtered classes inside a bike
        rack nor at or beyond the range of its classes.
    """

    names = record["classes"]
    if racks and names and set(names) <= filtered_classes:
        center = Point(record["x"], record["y"])
        if any(rack.contains(center) for rack in racks):
            return False
    if ObjectClassification.UNCLASSIFIED in record["types"]:
        names = tuple(CLASS_NAMES.values())  # an unclassified object may be of any class
    detection_range = max((class_ranges.get(name, math.inf) for name in names), default=math.inf)
    return ObjectDetectionUtils.compute_dist_bev(record, _ORIGIN) < detection_range


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
        footprint = shapely_box(rack["x"] - half_length, rack["y"] - half_width, rack["x"] + half_length, rack["y"] + half_width)
        footprints.append(shapely_rotate(footprint, rack["yaw"], origin=(rack["x"], rack["y"]), use_radians=True))
    return footprints


def _candidates(sample: Sample) -> List[List[Tuple[float, bool, int]]]:
    """List the labels each prediction of a sample may match, nearest first.

    Args:
        sample: Sample holding the ``prediction`` and ``label`` object lists.

    Returns:
        Per prediction, ``(distance, don't-care, label index)`` of every
        label it is compatible with, ascending: of equally distant labels,
        positive ones precede don't-care ones and earlier ones later ones.
    """

    def list_candidates() -> List[List[Tuple[float, bool, int]]]:
        predictions, labels = prepared_objects(sample)
        return [
            sorted(
                (ObjectDetectionUtils.compute_dist_bev(prediction, label), not label["positive"], index)
                for index, label in enumerate(labels)
                # all classes of the prediction must be possible classes of the label, any class for an unclassified one
                if ObjectClassification.UNCLASSIFIED in label["types"] or prediction["types"] <= label["types"]
            )
            for prediction in predictions
        ]

    return sample.memo("candidates", list_candidates)


def _entry(
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
        "classes": CLASS_SEPARATOR.join(prediction["classes"]),
        "confidence_score": prediction["confidence_score"],
        "is_tp": label is not None,
        "label_classes": None if label is None else CLASS_SEPARATOR.join(label["classes"]),
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
    union = prediction["width"] * prediction["height"] * prediction["length"] + label["width"] * label["height"] * label["length"]
    union -= intersection
    entry["ase"] = 1.0 - (intersection / union if union > 0 else 0.0)
    # AOE: smallest yaw angle difference.
    yaw_difference = (prediction["yaw"] - label["yaw"]) % (2.0 * math.pi)
    entry["aoe"] = min(yaw_difference, 2.0 * math.pi - yaw_difference)
    # AVE: velocity error in m/s, only against a label whose velocity is set.
    entry["ave"] = math.hypot(prediction["vx"] - label["vx"], prediction["vy"] - label["vy"]) if label["velocity_set"] else None
    return entry
