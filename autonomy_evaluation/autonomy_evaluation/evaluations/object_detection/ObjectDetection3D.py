# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""3D object detection evaluation on the classes of ``perception_msgs/ObjectClassification``.

Evaluates 3D bounding boxes, e.g. detected in lidar point clouds, camera images
or both, against the labels of any dataset that ``autonomy_datasets`` provides.
The metrics follow the nuScenes detection benchmark
(https://www.nuscenes.org/object-detection), which serves as reference; how the
evaluation deviates from it is documented in ``docs/IMPLEMENTATION.md``.

The evaluation is composed of the metrics of
:mod:`~autonomy_evaluation.evaluations.object_detection.DetectionMetrics`, which
all build on the matching of predictions and labels of
:mod:`~autonomy_evaluation.evaluations.object_detection.matching` (classes,
compatibility, don't-care labels, filters and matching are documented there).

Key evaluation settings, all configurable:

- Thresholds: 0.5, 1.0, 2.0, 4.0 m (applied uniformly to all classes,
              ``ap.matching_thresholds``).
- Ranges:     Per-class max detection distance (car/utility/bus < 50 m, all
              other classes < 40 m, ``class_ranges``). An object of several
              possible classes uses the largest of their ranges.
- Bike-rack:  Bicycles and motorcycles whose BEV center falls inside a bike rack
              are excluded from both predictions and labels
              (``bike_rack_filtered_classes``).
- AP filter:  P-R points with cum_precision <= 0.1 or cum_recall <= 0.1 are
              excluded from the 101-point recall interpolation
              (``ap.min_precision``, ``min_recall``).
- TP metrics: ATE, ASE, AOE and AVE measured at the 2.0 m threshold
              (``tp_metric_threshold``). AVE is only computed against labels
              whose velocity is set, i.e. not marked as invalid in their state
              covariance; without any such label it is unavailable.
- Score:      Detection score combining mAP and the available TP metrics with the
              NDS formula of nuScenes, i.e. weighting mAP 5 and each TP metric 1
              (``detection_score.map_weight``).
"""

from __future__ import annotations

from typing import Any, Dict, List

from autonomy_evaluation.evaluations.Evaluation import Evaluation
from autonomy_evaluation.evaluations.Metric import Sample
from autonomy_evaluation.evaluations.object_detection.DetectionMetrics import (
    AveragePrecision,
    DetectedObjects,
    DetectionScore,
    NumLabels,
    NumPredictions,
    OrientationError,
    ScaleError,
    TranslationError,
    VelocityError,
)
from autonomy_evaluation.evaluations.object_detection.matching import matches, prediction_in_label_frame, prepared_objects
from perception_msgs.msg import ObjectList


class ObjectDetection3D(Evaluation):
    """Evaluation of 3D object detection on the classes of ``perception_msgs/ObjectClassification``.

    See the module docstring for the full evaluation settings. Predictions and
    labels are compared on the classes of
    ``perception_msgs/ObjectClassification``, so models can be evaluated on
    datasets of different class taxonomies; classes that the labels of a
    dataset do not distinguish are evaluated together.
    """

    VERSION = "1.1.0"
    RELEASE_NOTES = {
        "1.0.0": "Initial implementation based on adapted nuScenes detection benchmark",
        "1.1.0": "Composed of selectable and configurable metrics, reporting the detection score also as metric "
        "'detection_score'; the values of all metrics are unchanged",
    }

    HEADLINE_METRIC = DetectionScore.NAME

    def __init__(self) -> None:
        """Name the evaluation."""
        super().__init__(
            name="object_detection_3d",
            description=(
                "3D bounding-box object detection on the classes of perception_msgs/ObjectClassification, "
                "scored with the metrics of the nuScenes detection benchmark."
            ),
        )

    def metrics(self) -> List[Any]:
        """Compose the evaluation of the counts of objects, the AP, the TP errors and the detection score."""
        return [
            DetectedObjects(),
            NumLabels(),
            NumPredictions(),
            AveragePrecision(),
            TranslationError(),
            ScaleError(),
            OrientationError(),
            VelocityError(),
            DetectionScore(),
        ]

    def default_config(self) -> Dict[str, Any]:
        """Configure the filters and the matching shared by all metrics, following nuScenes.

        Returns:
            ``class_ranges``: per-class maximum BEV detection distance in meters,
            following the ranges of the corresponding nuScenes classes; vru,
            micro and animal, which nuScenes does not evaluate, use the range of
            pedestrians. Predictions and labels at or beyond the range of their
            classes are excluded before matching; an object of several possible
            classes uses the largest of their ranges.
            ``bike_rack_filtered_classes``: classes excluded from predictions and
            labels when their BEV center falls inside a bike rack.
            ``tp_metric_threshold``: BEV center distance in meters below which a
            prediction matches a label for the TP errors and the visualization.
            ``min_recall``: recall at or below which P-R points are excluded from
            the AP and TP errors are not averaged.
        """
        return {
            "class_ranges": {
                "pedestrian": 40.0,
                "bicycle": 40.0,
                "motorcycle": 40.0,
                "car": 50.0,
                "utility": 50.0,
                "bus": 50.0,
                "animal": 40.0,
                "vru": 40.0,
                "micro": 40.0,
            },
            "bike_rack_filtered_classes": ["bicycle", "motorcycle"],
            "tp_metric_threshold": 2.0,
            "min_recall": 0.1,
        }

    def visualization_outputs(self) -> Dict[str, Any]:
        """Define the ROS message types published when visualization is enabled.

        Returns:
            Output name to ROS message type. The keys match those of
            :meth:`visualize` and are node-relative topic names.
        """

        return {
            "true_positives": ObjectList,
            "false_positives": ObjectList,
            "false_negatives": ObjectList,
            "ignored": ObjectList,
        }

    def visualize(self, sample: Sample) -> Dict[str, Any]:
        """Split one frame's objects into true positives, false positives, false negatives and ignored objects.

        Reuses the extraction, filtering and matching of the metrics at the TP
        metric threshold (``tp_metric_threshold``), and hands back the source
        messages behind the outcome: a prediction that claimed a positive label
        is a true positive, a prediction that claimed none is a false positive,
        and a positive label no prediction claimed is a false negative.
        Don't-care labels and the predictions that claimed them are ignored.

        Objects dropped before matching (predictions without evaluated class,
        labels of UNKNOWN class only, bicycles in bike racks, objects beyond the
        range of their classes) appear in none of the lists.

        Args:
            sample: Sample holding the ``prediction`` and ``label`` object lists
                and, if published, the ``label_meta_info`` of the labels.

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

        prediction, label = prediction_in_label_frame(sample), sample["label"]
        predictions, labels = prepared_objects(sample)
        label_indices = matches(sample, sample.config["tp_metric_threshold"])

        true_positives: List[Any] = []
        false_positives: List[Any] = []
        ignored: List[Any] = []
        for record, label_index in zip(predictions, label_indices):
            if label_index == -1:
                false_positives.append(record["object"])
            elif labels[label_index]["positive"]:
                true_positives.append(record["object"])
            else:
                ignored.append(record["object"])
        claimed = set(label_indices)
        false_negatives = [record["object"] for index, record in enumerate(labels) if record["positive"] and index not in claimed]
        ignored += [record["object"] for record in labels if not record["positive"]]

        return {
            "true_positives": ObjectList(header=prediction.header, objects=true_positives),
            "false_positives": ObjectList(header=prediction.header, objects=false_positives),
            "false_negatives": ObjectList(header=label.header, objects=false_negatives),
            "ignored": ObjectList(header=label.header, objects=ignored),
        }
