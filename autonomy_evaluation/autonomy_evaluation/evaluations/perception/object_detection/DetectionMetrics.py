# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Metrics of the 3D object detection, following the nuScenes detection benchmark.

Every metric compares the ``prediction`` object list of a detector with the
``label`` object list of a dataset, optionally with the ``label_meta_info``
published next to the labels, and builds on the matching of
:mod:`~autonomy_evaluation.evaluations.perception.object_detection.matching`:

- ``num_labels``, ``num_predictions``: the numbers of positive labels and of
  predictions, in total and per evaluated class.
- ``ap``: the average precision per matching threshold and evaluated class,
  and their means.
- ``ate``, ``ase``, ``aoe``, ``ave``: the translation, scale, orientation and
  velocity errors of the true positives at the TP metric threshold.
- ``detection_score``: the mAP and the available mean TP errors combined with
  the NDS formula of nuScenes.

The evaluated classes and the numbers of objects per class are shared by all
metrics through the internal ``objects`` metric.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from autonomy_datasets_msgs.msg import ObjectListMetaInfo
from autonomy_evaluation.evaluations.Metric import Aggregation, EVALUATED, Input, Metric, REFERENCE, Sample
from autonomy_evaluation.evaluations.perception.object_detection.matching import (
    CLASS_SEPARATOR,
    evaluated_classes,
    match_entries,
    prepared_objects,
)
from autonomy_evaluation.utils.ObjectDetectionUtils import ObjectDetectionUtils
from perception_msgs.msg import ObjectList

# TP error metrics, of which AVE is only available for labels with a velocity
TP_METRICS: Tuple[str, ...] = ("ate", "ase", "aoe", "ave")


def detection_inputs() -> Dict[str, Input]:
    """Declare the inputs every metric of the 3D object detection reads.

    Returns:
        The evaluated ``prediction`` object list of the detector, the reference
        ``label`` object list and, optionally, the ``label_meta_info``
        published next to the labels on ``<label topic>/meta_info``, which
        carries the dataset annotations ``perception_msgs/Object`` cannot
        express; only the ``original_class`` of bike racks is read.
    """
    return {
        "prediction": Input(ObjectList, EVALUATED),
        "label": Input(ObjectList, REFERENCE),
        "label_meta_info": Input(ObjectListMetaInfo, REFERENCE, optional=True, next_to=("label", "/meta_info")),
    }


class DetectedObjects(Metric):
    """Internal metric counting the objects of each class, from which all detection metrics derive their classes.

    Its result is no metric, but the data the other metrics share: the
    evaluated classes, derived from the labels of all samples recorded so far,
    so that the results of each scene are reported on the same classes as the
    results of all samples, and the numbers of objects of the aggregated
    samples per evaluated class.
    """

    NAME = "objects"
    INTERNAL = True

    def inputs(self) -> Dict[str, Input]:
        """Read the predictions and the labels."""
        return detection_inputs()

    def compute_sample(self, sample: Sample) -> Dict[str, Any]:
        """Count the matched objects of a sample by their classes.

        Returns:
            ``{"num_predictions", "num_labels", "num_dont_care",
            "label_classes", "velocity_label_classes", "prediction_classes"}``:
            the numbers of matched predictions, positive labels and don't-care
            labels; the numbers of positive labels, of positive labels with
            velocity and of predictions by their classes joined with ``|``.
        """
        predictions, labels = prepared_objects(sample)
        positives = [record for record in labels if record["positive"]]
        return {
            "num_predictions": len(predictions),
            "num_labels": len(positives),
            "num_dont_care": len(labels) - len(positives),
            "label_classes": Counter(CLASS_SEPARATOR.join(record["classes"]) for record in positives),
            "velocity_label_classes": Counter(
                CLASS_SEPARATOR.join(record["classes"]) for record in positives if record["velocity_set"]
            ),
            "prediction_classes": Counter(CLASS_SEPARATOR.join(record["classes"]) for record in predictions),
        }

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Derive the evaluated classes and count the objects of the aggregated samples per evaluated class.

        Returns:
            ``{"evaluated_classes", "class_of", "label_counts",
            "prediction_counts", "num_labels", "num_velocity_labels",
            "num_predictions", "counted_classes", "label_classes"}``: the
            members of each evaluated class, the evaluated class of each class,
            the numbers of labels and predictions by their joined classes, the
            numbers of labels (of evaluated classes with labels only), labels
            with velocity and predictions per evaluated class, the evaluated
            classes with labels or predictions, and the labels of each counted
            class by their possible classes.
        """
        records = aggregation.records(self.name)
        recorded_label_classes = (
            key for record in aggregation.records(self.name, aggregation.recorded_entries) for key in record["label_classes"]
        )
        classes = evaluated_classes({*recorded_label_classes, *(key for record in records for key in record["label_classes"])})
        class_of = {member: name for name, members in classes.items() for member in members}

        label_counts = sum((Counter(record["label_classes"]) for record in records), Counter())
        velocity_label_counts = sum((Counter(record["velocity_label_classes"]) for record in records), Counter())
        prediction_counts = sum((Counter(record["prediction_classes"]) for record in records), Counter())
        num_labels = _counts_by_evaluated_class(label_counts, class_of)
        num_predictions = _counts_by_evaluated_class(prediction_counts, class_of)
        # Objects are counted for the evaluated classes with labels or predictions, metrics are computed for those with
        # labels, both in the order of their classes.
        counted_classes = [name for name in classes if num_labels.get(name) or num_predictions.get(name)]
        # The labels of each evaluated class by their possible classes, which all belong to it, e.g. NVIDIA's persons
        # as "pedestrian|vru" and its strollers as "vru" within "pedestrian|vru".
        label_classes: Dict[str, Dict[str, int]] = {name: {} for name in counted_classes}
        for key, count in label_counts.items():
            label_classes[class_of[key.split(CLASS_SEPARATOR)[0]]][key] = count
        return {
            "evaluated_classes": classes,
            "class_of": class_of,
            "label_counts": label_counts,
            "prediction_counts": prediction_counts,
            "num_labels": {name: num_labels[name] for name in classes if num_labels.get(name)},
            "num_velocity_labels": _counts_by_evaluated_class(velocity_label_counts, class_of),
            "num_predictions": num_predictions,
            "counted_classes": counted_classes,
            "label_classes": label_classes,
        }


class _DetectionMetric(Metric):
    """Metric of the 3D object detection, which reads the detection inputs and builds on the counted objects."""

    def inputs(self) -> Dict[str, Input]:
        """Read the predictions and the labels."""
        return detection_inputs()

    def requires(self) -> Tuple[str, ...]:
        """Build on the evaluated classes and the numbers of objects per class."""
        return (DetectedObjects.NAME,)


class NumLabels(_DetectionMetric):
    """Number of positive labels, in total and per evaluated class with labels or predictions."""

    NAME = "num_labels"

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Report the number of positive labels.

        Returns:
            ``{"_value_": total, class: {"_value_": number, classes: number}}``
            for every evaluated class with labels or predictions, the labels of
            each class also by their possible classes joined with ``|``.
        """
        objects = aggregation.results[DetectedObjects.NAME]
        return {
            "_value_": sum(objects["label_counts"].values()),
            **{
                name: {"_value_": objects["num_labels"].get(name, 0), **objects["label_classes"][name]}
                for name in objects["counted_classes"]
            },
        }


class NumPredictions(_DetectionMetric):
    """Number of predictions of an evaluated class, in total and per evaluated class with labels or predictions."""

    NAME = "num_predictions"

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Report the number of predictions.

        Returns:
            ``{"_value_": total, class: number}`` for every evaluated class with
            labels or predictions; a prediction of classes of several evaluated
            classes counts for each of them, but once in total.
        """
        objects = aggregation.results[DetectedObjects.NAME]
        return {
            "_value_": sum(objects["prediction_counts"].values()),
            **{name: objects["num_predictions"].get(name, 0) for name in objects["counted_classes"]},
        }


class AveragePrecision(_DetectionMetric):
    """Average precision per matching threshold and evaluated class, and its means.

    Configuration:
        ``matching_thresholds``: BEV center distances in meters below which a
        prediction matches a label, each evaluated on its own.
        ``min_precision``: precision offset subtracted before integrating the
        precision-recall curve; P-R points at or below it are excluded.
    """

    NAME = "ap"

    def default_config(self) -> Dict[str, Any]:
        """Match at 0.5, 1, 2 and 4 m, and integrate precisions above 0.1, following nuScenes."""
        return {"matching_thresholds": [0.5, 1.0, 2.0, 4.0], "min_precision": 0.1}

    def compute_sample(self, sample: Sample) -> Dict[float, List[Dict[str, Any]]]:
        """Match the predictions of a sample to its labels at every threshold.

        Returns:
            ``threshold → [entry]`` with one entry per prediction that is a
            true or false positive (see ``matching.match_entries``).
        """
        return {threshold: match_entries(sample, threshold) for threshold in self.config["matching_thresholds"]}

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Compute the AP per threshold and class, and the mAP per threshold, per class and overall.

        AP is computed for every evaluated class with labels; a class without
        labels has no AP and does not enter the mAP, as its predictions can only
        be false positives.

        Returns:
            ``{"_value_": mAP, "classes": {class: mAP}, "dist_{threshold}":
            {"_value_": mAP, "classes": {class: AP}}}`` with the mAP over all
            thresholds and classes, the mAP of each class over all thresholds,
            and the mAP and APs of each threshold; the mAPs over classes are
            ``None`` without any labels.
        """
        objects = aggregation.results[DetectedObjects.NAME]
        num_labels = objects["num_labels"]
        thresholds: Dict[str, Dict[str, Any]] = {}
        for threshold in self.config["matching_thresholds"]:
            by_class = _entries_by_evaluated_class(aggregation, self.name, threshold, objects["class_of"])
            aps = {
                name: self._average_precision(by_class.get(name, []), count, aggregation) for name, count in num_labels.items()
            }
            thresholds[f"dist_{threshold}"] = {"_value_": float(np.mean(list(aps.values()))) if aps else None, "classes": aps}
        class_maps = {name: float(np.mean([metrics["classes"][name] for metrics in thresholds.values()])) for name in num_labels}
        return {"_value_": float(np.mean(list(class_maps.values()))) if class_maps else None, "classes": class_maps, **thresholds}

    def _average_precision(self, entries: List[Dict[str, Any]], num_labels: int, aggregation: Aggregation) -> float:
        """Compute the AP of one class at one threshold.

        Sorts the entries globally by confidence descending and accumulates
        precision and recall over the full dataset, which are integrated with
        101-point recall interpolation, gating by ``min_recall`` and
        ``min_precision`` (a perfect detector scores AP = 1.0).

        Args:
            entries: True and false positives of the class.
            num_labels: Number of positive labels of the class.
            aggregation: Aggregated samples, whose configuration holds ``min_recall``.

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
        return ObjectDetectionUtils.compute_ap_101_point(
            recall, precision, aggregation.config["min_recall"], self.config["min_precision"]
        )


class _TPError(_DetectionMetric):
    """Error of the true positives at the TP metric threshold, per evaluated class and as mean over the classes.

    Cumulative-mean errors (by descending confidence) are interpolated to 101
    recall points and averaged over ``[min_recall, max_recall]``. A class whose
    recall stays below ``min_recall`` falls back to the worst-case error
    ``1.0``. The mean is the macro mean over the classes, skipping ``None``
    values.
    """

    def compute_sample(self, sample: Sample) -> List[Dict[str, Any]]:
        """Match the predictions of a sample to its labels at the TP metric threshold.

        Returns:
            One entry per true or false positive, the true positives with their
            TP errors (see ``matching.match_entries``).
        """
        return match_entries(sample, sample.config["tp_metric_threshold"])

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Compute the error per evaluated class with labels and its mean.

        Returns:
            ``{"_value_": mean, class: error}``; the mean is ``None`` without
            any available error.
        """
        objects = aggregation.results[DetectedObjects.NAME]
        entries = _entries_by_evaluated_class(
            aggregation, self.name, aggregation.config["tp_metric_threshold"], objects["class_of"]
        )
        errors = {
            name: self._class_error(entries.get(name, []), count, objects["num_velocity_labels"].get(name, 0), aggregation)
            for name, count in objects["num_labels"].items()
        }
        available = [error for error in errors.values() if error is not None]
        return {"_value_": float(np.mean(available)) if available else None, **errors}

    def _class_error(
        self, entries: List[Dict[str, Any]], num_labels: int, num_velocity_labels: int, aggregation: Aggregation
    ) -> Optional[float]:
        """Average the error of the true positives of one evaluated class over its recall.

        Args:
            entries: True and false positives of the class at the TP metric threshold.
            num_labels: Number of positive labels of the class.
            num_velocity_labels: Number of positive labels of the class with velocity.
            aggregation: Aggregated samples, whose configuration holds ``min_recall``.

        Returns:
            The error of the class, ``1.0`` if its recall stays below ``min_recall``.
        """
        recall: List[float] = []
        curve: List[float] = []
        error_sum, error_count, cum_tp = 0.0, 0, 0
        for entry in sorted(entries, key=lambda entry: -entry["confidence_score"]):
            if entry["is_tp"]:
                cum_tp += 1
                if entry[self.NAME] is not None:
                    error_sum += entry[self.NAME]
                    error_count += 1
            recall.append(cum_tp / num_labels)
            curve.append(error_sum / error_count if error_count else 0.0)

        min_recall = aggregation.config["min_recall"]
        if not recall or ObjectDetectionUtils.compute_tp_101_point(np.array(recall), curve, min_recall) is None:
            return 1.0
        return ObjectDetectionUtils.compute_tp_101_point(np.array(recall), curve, min_recall)


class TranslationError(_TPError):
    """Average translation error (ATE): BEV Euclidean distance in meters between a true positive and its label."""

    NAME = "ate"


class ScaleError(_TPError):
    """Average scale error (ASE): ``1 - IoU`` between a true positive and its label after aligning centers and orientation."""

    NAME = "ase"


class OrientationError(_TPError):
    """Average orientation error (AOE): smallest yaw angle difference in radians between a true positive and its label."""

    NAME = "aoe"


class VelocityError(_TPError):
    """Average velocity error (AVE): absolute velocity error in m/s between a true positive and its label.

    Only computed against labels whose velocity is set, i.e. not marked as
    invalid in their state covariance: a class without any label with velocity
    has no velocity error (``None``), and a class of whose true positives none
    has a label with velocity has the worst-case error ``1.0``.
    """

    NAME = "ave"

    def _class_error(
        self, entries: List[Dict[str, Any]], num_labels: int, num_velocity_labels: int, aggregation: Aggregation
    ) -> Optional[float]:
        """Average the velocity error of one class, unavailable without labels with velocity."""
        if not num_velocity_labels:
            return None
        if not any(entry["is_tp"] and entry["ave"] is not None for entry in entries):
            return 1.0  # no true positive against a label with velocity
        return super()._class_error(entries, num_labels, num_velocity_labels, aggregation)


class DetectionScore(_DetectionMetric):
    """Detection score combining the mAP and the available mean TP errors with the NDS formula of nuScenes.

    ::

        detection_score = (map_weight·mAP + Σ max(1 - m{metric}, 0)) / (map_weight + number of TP metrics)

    i.e. weighting mAP with ``map_weight`` and each TP metric with 1: AVE is left
    out when no label has a velocity.

    Configuration:
        ``map_weight``: weight of the mAP relative to a weight of 1 per
        available TP metric.
    """

    NAME = "detection_score"

    def default_config(self) -> Dict[str, Any]:
        """Weight the mAP 5, following the NDS of nuScenes."""
        return {"map_weight": 5.0}

    def requires(self) -> Tuple[str, ...]:
        """Build on the mAP and the mean TP errors."""
        return ("ap", *TP_METRICS)

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Combine the mAP and the mean TP errors.

        Returns:
            ``{"_value_": detection score}``, ``None`` without any labels.
        """
        mean_errors = {metric: aggregation.results[metric]["_value_"] for metric in TP_METRICS}
        return {"_value_": self.score(aggregation.results["ap"]["_value_"], mean_errors)}

    def score(self, map_value: Optional[float], mean_errors: Dict[str, Optional[float]]) -> Optional[float]:
        """Combine the mAP and the available mean TP errors into the detection score.

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
        return (self.config["map_weight"] * map_value + tp_scores) / (self.config["map_weight"] + len(available))


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
        for name in {class_of[member] for member in classes.split(CLASS_SEPARATOR)}:
            totals[name] = totals.get(name, 0) + count
    return totals


def _entries_by_evaluated_class(
    aggregation: Aggregation, metric: str, threshold: float, class_of: Dict[str, str]
) -> Dict[str, List[Dict[str, Any]]]:
    """Gather the entries of the aggregated samples at a threshold by evaluated class.

    A true positive belongs to the evaluated class of its label, which also
    holds its classes. A false positive counts against the evaluated class
    of each of its classes. The entries at a threshold are the same for all
    metrics, so they are gathered once per threshold.

    Args:
        aggregation: Aggregated samples.
        metric: Qualified name of a metric whose records hold the entries at
            the threshold, either directly or by threshold.
        threshold: BEV center distance threshold of the entries.
        class_of: Class name to the name of its evaluated class.

    Returns:
        ``evaluated class → [entry]``, in sample order.
    """

    def gather() -> Dict[str, List[Dict[str, Any]]]:
        by_class: Dict[str, List[Dict[str, Any]]] = {}
        for record in aggregation.records(metric):
            for entry in record[threshold] if isinstance(record, dict) else record:
                classes = entry["label_classes"] if entry["is_tp"] else entry["classes"]
                for name in {class_of[member] for member in classes.split(CLASS_SEPARATOR)}:
                    by_class.setdefault(name, []).append(entry)
        return by_class

    return aggregation.memo(("entries_by_class", threshold), gather)
