# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ObjectDetection3D evaluation.

Tests focus on public API methods: compute_sample_metrics(),
compute_aggregated_metrics(), visualize_sample(), and evaluation configuration.

Inputs are real ROS messages (built via the helpers below), matching how
``_extract_objects`` reads them: geometry through the ``perception_msgs_utils``
state getters (which require a real ``ObjectState`` with a valid ``model_id``),
and the classes of predictions and labels from the ``ObjectClassification``
entries of their state. A label lists all its possible classes with the same
probability, as ``autonomy_datasets`` publishes a class its dataset cannot
tell. The optional ``autonomy_datasets_msgs/ObjectListMetaInfo`` of the labels,
correlated with the ``ObjectList`` by object ID, only names bike racks.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

import pytest
from autonomy_datasets_msgs.msg import ObjectListMetaInfo, ObjectMetaInfo
from autonomy_evaluation.evaluations.object_detection.ObjectDetection3D import _class_types, ObjectDetection3D
from diagnostic_msgs.msg import KeyValue
from geometry_msgs.msg import TransformStamped
from perception_msgs.msg import HEXAMOTION, Object, ObjectClassification, ObjectList
from perception_msgs_utils import initialize_state, set_continuous_state_covariance_to_unknown_at
from tf2_ros import Buffer

CAR = ObjectClassification.CAR
UTILITY = ObjectClassification.UTILITY
BUS = ObjectClassification.BUS
PEDESTRIAN = ObjectClassification.PEDESTRIAN
BICYCLE = ObjectClassification.BICYCLE
MOTORCYCLE = ObjectClassification.MOTORCYCLE
VRU = ObjectClassification.VRU
MICRO = ObjectClassification.MICRO
ANIMAL = ObjectClassification.ANIMAL
UNKNOWN = ObjectClassification.UNKNOWN
UNCLASSIFIED = ObjectClassification.UNCLASSIFIED

# Possible classes of a Waymo vehicle and pedestrian, as autonomy_datasets publishes them
WAYMO_VEHICLE = (CAR, MOTORCYCLE, UTILITY, BUS, MICRO)
WAYMO_PEDESTRIAN = (PEDESTRIAN, VRU)
WAYMO_VEHICLE_CLASS = "motorcycle|car|utility|bus|micro"

# A ground-truth object paired with its dataset annotations (``None`` when the
# dataset publishes no meta information for it).
GroundTruth = Tuple[Object, Optional[List[Tuple[str, str]]]]


def _object(
    x: float,
    y: float,
    width: float,
    length: float,
    height: float,
    yaw: float,
    vel_lon: float,
    vel_lat: float,
    classes: Sequence[int],
    probability: float,
    velocity_set: bool,
) -> Object:
    """Build an ``Object`` with a HEXAMOTION state, whose covariance marks the velocity as set or not."""
    obj = Object()
    initialize_state(obj, HEXAMOTION.MODEL_ID)
    for index, value in (
        (HEXAMOTION.X, x),
        (HEXAMOTION.Y, y),
        (HEXAMOTION.WIDTH, width),
        (HEXAMOTION.LENGTH, length),
        (HEXAMOTION.HEIGHT, height),
        (HEXAMOTION.YAW, yaw),
        (HEXAMOTION.VEL_LON, vel_lon),
        (HEXAMOTION.VEL_LAT, vel_lat),
    ):
        obj.state.continuous_state[index] = value
    if velocity_set:
        for index in (HEXAMOTION.VEL_LON, HEXAMOTION.VEL_LAT):
            set_continuous_state_covariance_to_unknown_at(obj.state, index, index)
    obj.state.classifications = [ObjectClassification(type=class_type, probability=probability) for class_type in classes]
    return obj


def _pred(
    x=0.0,
    y=0.0,
    width=1.0,
    length=1.0,
    height=1.0,
    yaw=0.0,
    vel_lon=0.0,
    vel_lat=0.0,
    class_type=CAR,
    confidence=0.9,
    obj_id=0,
    classes: Optional[Sequence[int]] = None,
) -> Object:
    """Build a prediction ``Object``.

    Its classes are classified with its confidence, which is also its
    ``existence_probability``; ``classes`` lists several equally probable ones.
    Predictions carry no dataset meta information.
    """
    obj = _object(x, y, width, length, height, yaw, vel_lon, vel_lat, classes or (class_type,), confidence, velocity_set=False)
    obj.id = obj_id
    obj.existence_probability = confidence
    return obj


def _gt(
    x=0.0,
    y=0.0,
    width=1.0,
    length=1.0,
    height=1.0,
    yaw=0.0,
    vel_lon=0.0,
    vel_lat=0.0,
    classes: Sequence[int] = (CAR,),
    velocity_set=True,
    original_class: Optional[str] = None,
) -> GroundTruth:
    """Build a ground-truth ``Object`` and its dataset annotations.

    All possible classes are classified with probability 1, as the dataset
    publishes them. ``original_class`` becomes the only annotation, turned into
    an ``ObjectMetaInfo`` entry by :func:`_label`.
    """
    obj = _object(x, y, width, length, height, yaw, vel_lon, vel_lat, classes, 1.0, velocity_set)
    obj.existence_probability = 1.0
    return obj, [("original_class", original_class)] if original_class else None


def _msg(objs, frame_id: str = "base_link") -> ObjectList:
    """Wrap objects in a ``perception_msgs/ObjectList`` message."""
    msg = ObjectList(objects=objs)
    msg.header.frame_id = frame_id
    return msg


def _label(gts: List[GroundTruth], frame_id: str = "base_link") -> Tuple[ObjectList, ObjectListMetaInfo]:
    """Split ``_gt`` pairs into the label and label meta info messages.

    Object IDs are assigned per frame and are what correlates both messages;
    objects without annotations get no ``ObjectMetaInfo`` entry, as published by
    the dataset.
    """
    objects: List[Object] = []
    meta_entries: List[ObjectMetaInfo] = []
    for object_id, (obj, object_annotations) in enumerate(gts):
        obj.id = object_id
        objects.append(obj)
        if object_annotations:
            info = [KeyValue(key=key, value=value) for key, value in object_annotations]
            meta_entries.append(ObjectMetaInfo(id=object_id, info=info))
    label = _msg(objects, frame_id)
    meta_info = ObjectListMetaInfo(objects=meta_entries)
    meta_info.header.frame_id = frame_id
    return label, meta_info


def _lidar_tf_buffer(x: float, yaw: float) -> Buffer:
    """Build a tf2 buffer holding the static transform of a 'lidar_top' frame at ``x`` turned by ``yaw`` in 'base_link'."""
    transform = TransformStamped()
    transform.header.frame_id = "base_link"
    transform.child_frame_id = "lidar_top"
    transform.transform.translation.x = x
    transform.transform.rotation.z = math.sin(yaw / 2.0)
    transform.transform.rotation.w = math.cos(yaw / 2.0)
    tf_buffer = Buffer()
    tf_buffer.set_transform_static(transform, "test")
    return tf_buffer


class TestObjectDetection3D:
    """Tests for the ObjectDetection3D evaluation.

    Covers the classes of objects, sample metrics computation (Pass 1),
    aggregated metrics (Pass 2), the detection score and the visualization.
    """

    def setup_method(self):
        """Create a fresh evaluation instance for each test."""
        self.bm = ObjectDetection3D()

    def _metrics(self, pred_objs, gts, meta_info: bool = True) -> dict:
        """Compute sample metrics from prediction objects and ``_gt`` pairs.

        Mirrors the node's synchronized callback, which hands the evaluation the
        label object list and, if the dataset publishes it, its meta info message.
        """
        label, label_meta_info = _label(gts)
        return self.bm.compute_sample_metrics(_msg(pred_objs), label, label_meta_info=label_meta_info if meta_info else None)

    def _entries(self, pred_objs, gts, threshold: float = 2.0) -> list:
        """Return the true and false positives of a sample at a threshold."""
        return self._metrics(pred_objs, gts)["match_records"][threshold]

    def _aggregate(self, *frames, scene_ids: Optional[Sequence[str]] = None) -> dict:
        """Record ``(pred_objs, gts)`` frames and aggregate them like the node does at the end."""
        for index, (pred_objs, gts) in enumerate(frames):
            label, label_meta_info = _label(gts)
            scene_id = scene_ids[index] if scene_ids else None
            self.bm.record_sample(
                sample_id=str(index), scene_id=scene_id, prediction=_msg(pred_objs), label=label, label_meta_info=label_meta_info
            )
        return self.bm.compute_aggregated_metrics(self.bm._sample_results)

    def _visualize(self, pred_objs, gts) -> dict:
        """Build the visualization object lists the way the node's callback does."""
        label, label_meta_info = _label(gts)
        return self.bm.visualize_sample(_msg(pred_objs), label, label_meta_info=label_meta_info)

    # --- classes of an object ---

    def test_single_classification_is_the_class(self):
        """An object classified once has that class."""
        assert _class_types([ObjectClassification(type=CAR, probability=0.4)]) == {CAR}

    def test_equally_probable_classifications_are_the_possible_classes(self):
        """A dataset that cannot tell the class assigns all possible ones with the same probability."""
        classifications = [ObjectClassification(type=class_type, probability=1.0) for class_type in WAYMO_VEHICLE]

        assert _class_types(classifications) == set(WAYMO_VEHICLE)

    def test_most_probable_classification_is_the_class(self):
        """Of a detection's class probabilities, only the highest one counts."""
        classifications = [ObjectClassification(type=CAR, probability=0.7), ObjectClassification(type=UTILITY, probability=0.3)]

        assert _class_types(classifications) == {CAR}

    def test_object_without_classification_is_unclassified(self):
        """An object without any classification may be of any class."""
        assert _class_types([]) == {UNCLASSIFIED}

    @pytest.mark.parametrize(
        "deprecated, replacement",
        [
            (ObjectClassification.VAN, {CAR}),
            (ObjectClassification.CAR_UNION, {CAR}),
            (ObjectClassification.TRAIN, {UTILITY}),
            (ObjectClassification.TRAILER, {UTILITY}),
            (ObjectClassification.TRUCK_UNION, {UTILITY}),
            (ObjectClassification.BIKE_UNION, {BICYCLE, MOTORCYCLE}),
            (ObjectClassification.ROAD_OBSTACLE, {UNKNOWN}),
        ],
    )
    def test_deprecated_types_are_evaluated_as_their_replacement(self, deprecated, replacement):
        """Deprecated types are replaced as perception_msgs/ObjectClassification documents it."""
        assert _class_types([ObjectClassification(type=deprecated, probability=1.0)]) == replacement

    def test_undefined_type_is_evaluated_as_unknown(self):
        """A type the message does not define, e.g. Waymo's sign (20), is none of the defined classes."""
        assert _class_types([ObjectClassification(type=20, probability=1.0)]) == {UNKNOWN}

    def test_deprecated_prediction_matches_its_replacement(self):
        """A detector publishing the deprecated TRAILER is evaluated as UTILITY."""
        entries = self._entries([_pred(class_type=ObjectClassification.TRAILER)], [_gt(classes=(UTILITY,))])

        assert entries[0]["is_tp"] is True

    # --- matching ---

    def test_empty_inputs(self):
        """Verify empty inputs produce valid match records for all thresholds."""
        result = self._metrics([], [])
        assert result["sample_prediction_num"] == 0
        assert result["sample_ground_truth_num"] == 0
        for thr in [0.5, 1.0, 2.0, 4.0]:
            assert result["match_records"][thr] == []

    def test_basic_matching_produces_tp(self):
        """Verify identical boxes produce true positives at tight threshold."""
        entry = self._entries([_pred(x=0.0)], [_gt(x=0.0)], threshold=0.5)[0]
        assert entry["is_tp"] is True
        assert entry["label_classes"] == "car"

    def test_distant_pred_is_fp_at_tight_threshold(self):
        """Verify distant predictions are false positives at tight thresholds."""
        assert self._entries([_pred(x=10.0)], [_gt(x=0.0)], threshold=0.5)[0]["is_tp"] is False

    def test_match_distance_must_stay_below_the_threshold(self):
        """A prediction exactly at the threshold distance does not match, as in nuScenes."""
        assert self._entries([_pred(x=2.0)], [_gt(x=0.0)], threshold=2.0)[0]["is_tp"] is False

    def test_prediction_of_another_class_is_a_false_positive(self):
        """A truck predicted on a car is no match: the car is missed, the truck a false positive."""
        metrics = self._aggregate(([_pred(class_type=UTILITY)], [_gt(classes=(CAR,))]))

        assert metrics["ap"]["dist_2.0"]["classes"] == {"car": 0.0}
        assert metrics["num_labels"] == {"_value_": 1, "car": {"_value_": 1, "car": 1}, "utility": {"_value_": 0}}
        assert metrics["num_predictions"] == {"_value_": 1, "car": 0, "utility": 1}

    def test_prediction_of_a_possible_class_matches(self):
        """Any possible class of a label matches it, e.g. a truck on a Waymo vehicle."""
        entry = self._entries([_pred(class_type=UTILITY)], [_gt(classes=WAYMO_VEHICLE)])[0]

        assert entry["is_tp"] is True
        assert entry["label_classes"] == WAYMO_VEHICLE_CLASS

    def test_prediction_of_several_classes_must_fit_them_all(self):
        """Hedging over classes earns no match a single class would not."""
        hedging = _pred(classes=(CAR, UTILITY))

        assert self._entries([hedging], [_gt(classes=(CAR,))])[0]["is_tp"] is False
        assert self._entries([hedging], [_gt(classes=WAYMO_VEHICLE)])[0]["is_tp"] is True

    def test_second_prediction_on_a_label_of_several_classes_is_a_duplicate(self):
        """A Waymo vehicle is one object, so a car predicted next to the truck on it is a false positive."""
        entries = self._entries(
            [_pred(class_type=UTILITY, confidence=0.9), _pred(class_type=CAR, confidence=0.7)], [_gt(classes=WAYMO_VEHICLE)]
        )

        assert [(entry["classes"], entry["is_tp"]) for entry in entries] == [("utility", True), ("car", False)]

    def test_unclassified_label_is_dont_care(self):
        """An unclassified object may be anything: the first prediction on it is ignored, it is never missed."""
        result = self._metrics([_pred(x=0.0, confidence=0.9), _pred(x=0.5, confidence=0.8)], [_gt(classes=(UNCLASSIFIED,))])

        # the less confident prediction finds the object claimed and is a duplicate
        assert [entry["confidence_score"] for entry in result["match_records"][2.0]] == [0.8]
        assert result["sample_ground_truth_num"] == 0
        assert result["sample_dont_care_num"] == 1

    def test_label_that_may_be_unknown_is_dont_care(self):
        """The 'other_vehicle' of NVIDIA may be a motorcycle or none of the classes, but never a car."""
        other_vehicle = (UNKNOWN, MOTORCYCLE)

        assert self._entries([_pred(class_type=MOTORCYCLE)], [_gt(classes=other_vehicle)]) == []
        assert self._entries([_pred(class_type=CAR)], [_gt(classes=other_vehicle)])[0]["is_tp"] is False

    def test_prediction_on_an_unknown_label_is_a_false_positive(self):
        """A car predicted on a barrier, which is definitely none of the classes, is a false positive."""
        result = self._metrics([_pred(class_type=CAR)], [_gt(classes=(UNKNOWN,))])

        assert result["match_records"][2.0][0]["is_tp"] is False
        assert result["sample_ground_truth_num"] == 0
        assert result["sample_dont_care_num"] == 0

    @pytest.mark.parametrize("classes", [(UNKNOWN,), (UNCLASSIFIED,), ()])
    def test_predictions_without_evaluated_class_are_dropped(self, classes):
        """Predictions of no evaluated class, e.g. a detector's barriers, are not scored."""
        prediction = _pred()
        prediction.state.classifications = [ObjectClassification(type=class_type, probability=0.9) for class_type in classes]

        assert self._metrics([prediction], [])["sample_prediction_num"] == 0

    def test_prediction_claims_the_nearest_label(self):
        """A prediction on a don't-care object does not steal the positive label of the next prediction."""
        rider = _gt(x=20.0, classes=(BICYCLE, MOTORCYCLE))
        other_vehicle = _gt(x=21.7, classes=(UNKNOWN, MOTORCYCLE))
        on_other_vehicle = _pred(x=21.7, y=0.1, class_type=MOTORCYCLE, confidence=0.9)
        on_rider = _pred(x=19.6, class_type=MOTORCYCLE, confidence=0.8)

        entries = self._entries([on_other_vehicle, on_rider], [rider, other_vehicle])

        assert [(entry["confidence_score"], entry["is_tp"]) for entry in entries] == [(0.8, True)]
        assert entries[0]["ate"] == pytest.approx(0.4)

    def test_positive_label_is_claimed_before_an_equally_distant_dont_care_label(self):
        """Of two labels at the same distance, the prediction claims the one that counts."""
        entries = self._entries([_pred(x=0.0)], [_gt(x=1.0, classes=(UNCLASSIFIED,)), _gt(x=-1.0)])

        assert entries[0]["is_tp"] is True

    # --- evaluated classes ---

    def test_single_class_labels_are_evaluated_per_class(self):
        """Datasets that label every object with one class, e.g. nuScenes, are evaluated per class."""
        metrics = self._aggregate(
            ([_pred(class_type=CAR), _pred(x=5.0, class_type=UTILITY)], [_gt(classes=(CAR,)), _gt(x=5.0, classes=(UTILITY,))])
        )

        assert list(metrics["ap"]["dist_2.0"]["classes"]) == ["car", "utility"]
        assert metrics["ap"]["_value_"] == 1.0

    def test_classes_a_dataset_does_not_distinguish_are_evaluated_together(self):
        """Waymo's vehicles, pedestrians and cyclists become the evaluated classes."""
        metrics = self._aggregate(
            (
                [_pred(x=0.0, class_type=UTILITY), _pred(x=5.0, class_type=PEDESTRIAN), _pred(x=10.0, class_type=BICYCLE)],
                [_gt(x=0.0, classes=WAYMO_VEHICLE), _gt(x=5.0, classes=WAYMO_PEDESTRIAN), _gt(x=10.0, classes=(BICYCLE,))],
            )
        )

        assert list(metrics["ap"]["dist_2.0"]["classes"]) == ["pedestrian|vru", "bicycle", WAYMO_VEHICLE_CLASS]
        assert metrics["ap"]["dist_2.0"]["classes"][WAYMO_VEHICLE_CLASS] == 1.0

    def test_overlapping_label_classes_are_evaluated_together(self):
        """Persons of NVIDIA may be pedestrians or VRUs, so its strollers are evaluated with them."""
        person, stroller = _gt(x=0.0, classes=WAYMO_PEDESTRIAN), _gt(x=5.0, classes=(VRU,))

        metrics = self._aggregate(([_pred(x=0.0, class_type=VRU), _pred(x=5.0, class_type=PEDESTRIAN)], [person, stroller]))

        # the labels of the evaluated class by their possible classes
        assert metrics["num_labels"] == {"_value_": 2, "pedestrian|vru": {"_value_": 2, "pedestrian|vru": 1, "vru": 1}}
        # the stroller is definitely a VRU, so the pedestrian predicted on it is a false positive
        assert metrics["ate"]["pedestrian|vru"] == 0.0
        assert metrics["ap"]["dist_2.0"]["classes"]["pedestrian|vru"] < 1.0

    def test_labels_that_may_be_unknown_join_no_classes(self):
        """A label that may be none of the classes is don't-care, so it does not join the classes it lists."""
        metrics = self._aggregate(
            (
                [_pred(x=0.0, class_type=CAR), _pred(x=5.0, class_type=UTILITY)],
                [_gt(x=0.0, classes=(CAR,)), _gt(x=5.0, classes=(UTILITY,)), _gt(x=10.0, classes=(UNKNOWN, CAR, UTILITY))],
            )
        )

        assert list(metrics["ap"]["dist_2.0"]["classes"]) == ["car", "utility"]

    def test_scene_metrics_use_the_classes_of_all_samples(self):
        """A scene without a label joining classes is still evaluated on the classes of the whole evaluation."""
        self._aggregate(
            ([_pred(class_type=PEDESTRIAN)], [_gt(classes=WAYMO_PEDESTRIAN)]),
            ([_pred(class_type=VRU)], [_gt(classes=(VRU,))]),
            scene_ids=["persons", "strollers"],
        )

        results = self.bm.finalize()

        assert list(results["scenes"]["strollers"]["metrics"]["ap"]["dist_2.0"]["classes"]) == ["pedestrian|vru"]
        assert list(results["metrics"]["ap"]["dist_2.0"]["classes"]) == ["pedestrian|vru"]

    # --- aggregation ---

    def test_labels_fed_back_as_predictions_score_perfectly(self):
        """Labels of any taxonomy, published as predictions, are a perfect detection."""
        gts = [
            _gt(x=0.0, classes=WAYMO_VEHICLE, vel_lon=3.0),
            _gt(x=5.0, classes=WAYMO_PEDESTRIAN, vel_lon=1.0),
            _gt(x=10.0, classes=(BICYCLE, MOTORCYCLE)),
            _gt(x=15.0, classes=(CAR,), vel_lon=8.0),
        ]
        preds = [
            _pred(x=0.0, classes=WAYMO_VEHICLE, vel_lon=3.0),
            _pred(x=5.0, classes=WAYMO_PEDESTRIAN, vel_lon=1.0),
            _pred(x=10.0, classes=(BICYCLE, MOTORCYCLE)),
            _pred(x=15.0, classes=(CAR,), vel_lon=8.0),
        ]

        metrics = self._aggregate((preds, gts))

        assert metrics["ap"]["_value_"] == 1.0
        assert [metrics[metric]["_value_"] for metric in ("ate", "ase", "aoe", "ave")] == [0.0, 0.0, 0.0, 0.0]
        assert metrics["_value_"] == 1.0

    def test_perfect_matching_yields_high_map(self):
        """Verify perfect matching produces high mean average precision."""
        n = 10
        preds = [_pred(x=float(i), confidence=1.0 - i * 0.01) for i in range(n)]
        gts = [_gt(x=float(i)) for i in range(n)]
        assert self._aggregate((preds, gts))["ap"]["_value_"] >= 0.8

    def test_no_pred_yields_zero_map(self):
        """Verify missing predictions yield zero mAP."""
        assert self._aggregate(([], [_gt(x=0.0)]))["ap"]["_value_"] == 0.0

    def test_predictions_without_labels_yield_no_map(self):
        """Without any label, no class is evaluated, so there is no mAP."""
        metrics = self._aggregate(([_pred(x=float(i)) for i in range(5)], []))

        assert metrics["ap"]["_value_"] is None
        assert metrics["_value_"] is None
        assert metrics["num_labels"] == {"_value_": 0, "car": {"_value_": 0}}

    def test_class_without_labels_does_not_enter_map(self):
        """A false positive bus in a sample without buses has no AP to dilute the mAP with."""
        n = 10
        preds = [_pred(x=float(i), confidence=1.0 - i * 0.01) for i in range(n)]
        gts = [_gt(x=float(i)) for i in range(n)]
        perfect = self._aggregate((preds, gts))["ap"]["_value_"]
        self.bm.reset()

        metrics = self._aggregate((preds + [_pred(x=5.0, y=20.0, class_type=BUS)], gts))

        assert metrics["ap"]["_value_"] == perfect
        assert "bus" not in metrics["ap"]["dist_2.0"]["classes"]
        assert metrics["num_labels"]["bus"] == {"_value_": 0}

    def test_class_without_predictions_counts_as_zero(self):
        """A class with labels that the model never predicts is a real failure and stays in the mAP."""
        n = 10
        preds = [_pred(x=float(i), confidence=1.0 - i * 0.01) for i in range(n)]
        gts = [_gt(x=float(i)) for i in range(n)] + [_gt(x=5.0, y=20.0, classes=(PEDESTRIAN,))]

        metrics = self._aggregate((preds, gts))

        assert metrics["ap"]["_value_"] < 0.6
        assert list(metrics["ap"]["dist_2.0"]["classes"]) == ["pedestrian", "car"]
        assert metrics["num_predictions"]["pedestrian"] == 0
        assert metrics["ate"]["pedestrian"] == 1.0

    def test_class_map_averages_the_ap_of_the_class_over_the_thresholds(self):
        """A car predicted 1.5 m off only matches at the 2 m and 4 m thresholds, a pedestrian on its label at all."""
        metrics = self._aggregate(
            ([_pred(x=1.5), _pred(x=10.0, class_type=PEDESTRIAN)], [_gt(x=0.0), _gt(x=10.0, classes=(PEDESTRIAN,))])
        )

        assert [metrics["ap"][f"dist_{threshold}"]["classes"]["car"] for threshold in (0.5, 1.0, 2.0, 4.0)] == [
            0.0,
            0.0,
            1.0,
            1.0,
        ]
        assert metrics["ap"]["classes"] == {"pedestrian": 1.0, "car": 0.5}
        assert metrics["ap"]["_value_"] == 0.75

    def test_multi_frame_accumulation(self):
        """Verify label counts accumulate correctly across multiple frames."""
        metrics = self._aggregate(([_pred(x=0.0, confidence=0.9)], [_gt(x=0.0)]), ([_pred(x=0.0, confidence=0.8)], [_gt(x=0.0)]))
        assert metrics["num_labels"] == {"_value_": 2, "car": {"_value_": 2, "car": 2}}
        assert metrics["num_predictions"] == {"_value_": 2, "car": 2}

    def test_prediction_of_several_evaluated_classes_counts_once_in_total(self):
        """A prediction hedging over a car and a pedestrian counts for both classes, but as one prediction."""
        metrics = self._aggregate(([_pred(classes=(CAR, PEDESTRIAN))], [_gt(classes=(CAR,)), _gt(x=5.0, classes=(PEDESTRIAN,))]))

        assert metrics["num_predictions"] == {"_value_": 1, "pedestrian": 1, "car": 1}
        assert metrics["num_labels"] == {
            "_value_": 2,
            "pedestrian": {"_value_": 1, "pedestrian": 1},
            "car": {"_value_": 1, "car": 1},
        }

    def test_metrics_hold_their_aggregated_value_next_to_their_sub_metrics(self):
        """Each metric reports its aggregate under '_value_', followed by its thresholds or classes."""
        metrics = self._aggregate(([_pred(x=0.0)], [_gt(x=0.0)]))

        assert list(metrics) == ["_value_", "num_labels", "num_predictions", "ap", "ate", "ase", "aoe", "ave"]
        assert list(metrics["ap"]) == ["_value_", "classes", "dist_0.5", "dist_1.0", "dist_2.0", "dist_4.0"]
        assert metrics["num_labels"] == {"_value_": 1, "car": {"_value_": 1, "car": 1}}
        assert metrics["ap"]["dist_0.5"] == {"_value_": 1.0, "classes": {"car": 1.0}}
        assert metrics["ate"] == {"_value_": 0.0, "car": 0.0}

    def test_reproduces_the_former_nuscenes_evaluation(self):
        """With labels of single classes, AP and TP errors equal those of the former NuscenesLidarObjectDetection.

        The expected values were captured from NuscenesLidarObjectDetection on the
        same frames, before it became ObjectDetection3D.
        """
        for labels, predictions in _golden_frames():
            gts = [_gt(*values, classes=(class_type,)) for class_type, *values in labels]
            preds = [
                _pred(*values, class_type=class_type, confidence=confidence) for class_type, *values, confidence in predictions
            ]
            label, _ = _label(gts)
            self.bm.record_sample(prediction=_msg(preds), label=label)

        metrics = self.bm.compute_aggregated_metrics(self.bm._sample_results)

        for threshold, aps in _GOLDEN_AP.items():
            assert metrics["ap"][f"dist_{threshold}"]["classes"] == pytest.approx(aps, abs=1e-12)
        for name, errors in _GOLDEN_TP_ERRORS.items():
            assert [metrics[metric][name] for metric in ("ate", "ase", "aoe", "ave")] == pytest.approx(errors, abs=1e-12)
        assert metrics["ap"]["_value_"] == pytest.approx(_GOLDEN_MAP, abs=1e-12)

    # --- filters ---

    def test_range_filtering_active(self):
        """Cars beyond 50 m are excluded from matching, those inside are kept."""
        assert self._metrics([_pred(x=51.0)], [_gt(x=51.0)])["match_records"][2.0] == []
        assert self._metrics([_pred(x=49.0)], [_gt(x=49.0)])["sample_ground_truth_num"] == 1

    def test_range_excludes_its_boundary(self):
        """An object exactly at the range of its class is excluded, as in nuScenes."""
        result = self._metrics([_pred(x=50.0)], [_gt(x=50.0)])

        assert result["sample_prediction_num"] == 0
        assert result["sample_ground_truth_num"] == 0

    def test_label_of_several_classes_uses_their_largest_range(self):
        """A Waymo vehicle may be a car, so it is evaluated up to 50 m, a motorcycle only up to 40 m."""
        assert self._metrics([], [_gt(x=45.0, classes=WAYMO_VEHICLE)])["sample_ground_truth_num"] == 1
        assert self._metrics([], [_gt(x=45.0, classes=(MOTORCYCLE,))])["sample_ground_truth_num"] == 0

    def test_unclassified_label_uses_the_largest_range(self):
        """An unclassified object may be a car, so it is don't-care up to 50 m."""
        assert self._metrics([], [_gt(x=45.0, classes=(UNCLASSIFIED,))])["sample_dont_care_num"] == 1

    def test_bike_rack_filters_bicycles_inside(self):
        """Bicycles inside a bike rack are excluded; the rack spans its length along its heading."""
        rack = _gt(
            x=10.0, width=1.0, length=6.0, yaw=math.pi / 2, classes=(UNKNOWN,), original_class="static_object.bicycle_rack"
        )
        inside = _gt(x=10.0, y=2.5, classes=(BICYCLE,))
        outside = _gt(x=12.5, y=0.0, classes=(BICYCLE,))

        result = self._metrics([_pred(x=10.0, y=2.4, class_type=BICYCLE)], [rack, inside, outside])

        assert result["sample_prediction_num"] == 0
        assert result["sample_ground_truth_num"] == 1

    def test_bike_racks_need_the_label_meta_info(self):
        """Without meta information, e.g. on datasets that publish none, no bike rack is known."""
        rack = _gt(x=10.0, width=4.0, length=4.0, classes=(UNKNOWN,), original_class="static_object.bicycle_rack")

        result = self._metrics([_pred(x=10.0, class_type=BICYCLE)], [rack, _gt(x=10.0, classes=(BICYCLE,))], meta_info=False)

        assert result["match_records"][2.0][0]["is_tp"] is True

    def test_bike_rack_keeps_other_classes(self):
        """Only bicycles and motorcycles are filtered by bike racks."""
        rack = _gt(x=10.0, width=4.0, length=4.0, classes=(UNKNOWN,), original_class="static_object.bicycle_rack")

        result = self._metrics([_pred(x=10.0, class_type=PEDESTRIAN)], [rack, _gt(x=10.0, classes=(PEDESTRIAN,))])

        assert result["match_records"][2.0][0]["is_tp"] is True

    # --- TP metrics and score ---

    def test_velocity_error_needs_labels_with_velocity(self):
        """Without labels whose velocity is set, AVE is unavailable and left out of the detection score."""
        metrics = self._aggregate(([_pred(x=0.0, vel_lon=5.0)], [_gt(x=0.0, velocity_set=False)]))

        assert metrics["ave"] == {"_value_": None, "car": None}
        # a perfect detection without AVE, which would otherwise count as error 1
        assert metrics["_value_"] == 1.0

    def test_velocity_error_against_labels_with_velocity(self):
        """The velocity error compares the predicted with the labeled velocity."""
        metrics = self._aggregate(([_pred(x=0.0, vel_lon=5.0)], [_gt(x=0.0, vel_lon=3.0)]))

        assert metrics["ave"] == {"_value_": 2.0, "car": 2.0}
        # AVE enters the detection score with the error clamped to 1
        assert metrics["_value_"] == pytest.approx((5.0 * 1.0 + 3.0) / 9.0)

    def test_velocity_error_skips_labels_without_velocity(self):
        """True positives against labels without velocity do not enter the velocity error."""
        preds = [_pred(x=0.0, vel_lon=5.0, confidence=0.9), _pred(x=10.0, vel_lon=9.0, confidence=0.8)]
        gts = [_gt(x=0.0, vel_lon=3.0), _gt(x=10.0, velocity_set=False)]

        assert self._aggregate((preds, gts))["ave"]["car"] == 2.0

    def test_detection_score_with_all_tp_metrics(self):
        """With all TP metrics available, the detection score weights them like the NDS of nuScenes without AAE."""
        score = self.bm._detection_score(0.5, {"ate": 0.2, "ase": 0.1, "aoe": 0.3, "ave": 0.4})

        assert score == pytest.approx((5.0 * 0.5 + 0.8 + 0.9 + 0.7 + 0.6) / 9.0)

    def test_detection_score_without_velocity(self):
        """Without AVE, the detection score is normalized over the remaining TP metrics."""
        score = self.bm._detection_score(0.5, {"ate": 0.2, "ase": 0.1, "aoe": 0.3, "ave": None})

        assert score == pytest.approx((5.0 * 0.5 + 0.8 + 0.9 + 0.7) / 8.0)

    def test_detection_score_clamps_tp_errors(self):
        """Verify TP errors above 1.0 are clamped in the detection score."""
        assert self.bm._detection_score(0.0, {"ate": 2.0, "ase": 3.0, "aoe": 5.0, "ave": 1.5}) == 0.0

    # --- topics ---

    def test_compares_the_predictions_with_the_labels_as_ground_truth(self):
        """The prediction of the detector is evaluated against the labels, optionally with their meta information."""
        assert self.bm.required_inputs() == {"prediction": ObjectList}
        assert self.bm.required_ground_truth() == {"label": ObjectList}
        assert self.bm.optional_ground_truth() == {"label_meta_info": ObjectListMetaInfo}

    def test_label_meta_info_follows_the_label_topic(self):
        """The dataset publishes the meta information next to the labels, on '<label topic>/meta_info'."""
        assert self.bm.derived_topics() == {"label_meta_info": ("label", "/meta_info")}

    def test_sample_metrics_are_computed_from_the_messages_by_topic_name(self):
        """The node passes the messages of a sample by the names of their topics, without unpublished meta information."""
        label, _ = _label([_gt(x=0.0)])

        result = self.bm.record_sample(sample_id="0", prediction=_msg([_pred(x=0.0)]), label=label, label_meta_info=None)

        assert result["metrics"]["sample_ground_truth_num"] == 1

    def test_transforms_predictions_into_the_frame_of_the_labels(self):
        """Lidar-frame detections are compared with vehicle-frame labels after transforming them with tf2."""
        self.bm.tf_buffer = _lidar_tf_buffer(x=1.0, yaw=math.pi / 2)
        # the lidar 1 m ahead of the vehicle origin faces to the left, so 10 m to its right are 11 m ahead of the origin
        label, _ = _label([_gt(x=11.0, yaw=math.pi / 2)], frame_id="base_link")

        result = self.bm.compute_sample_metrics(_msg([_pred(y=-10.0, yaw=0.0)], frame_id="lidar_top"), label)

        entry = result["match_records"][2.0][0]
        assert entry["is_tp"] is True
        assert entry["ate"] == pytest.approx(0.0, abs=1e-6)
        assert entry["aoe"] == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("has_tf_buffer", [False, True])
    def test_rejects_predictions_that_cannot_be_transformed_into_the_frame_of_the_labels(self, has_tf_buffer):
        """Boxes of different frames cannot be compared without a transform between them."""
        self.bm.tf_buffer = Buffer() if has_tf_buffer else None
        label, _ = _label([_gt(x=0.0)], frame_id="base_link")

        with pytest.raises(ValueError, match="'lidar_top' cannot be transformed into frame 'base_link'"):
            self.bm.compute_sample_metrics(_msg([_pred(x=0.0)], frame_id="lidar_top"), label)

    def test_accepts_objects_without_frame(self):
        """An empty frame cannot be checked, so it is assumed to be the frame of the other object list."""
        label, _ = _label([_gt(x=0.0)], frame_id="base_link")

        result = self.bm.compute_sample_metrics(_msg([_pred(x=0.0)], frame_id=""), label)

        assert result["match_records"][2.0][0]["is_tp"] is True

    # --- visualization of the per-sample matching outcome ---

    def test_visualization_outputs_are_object_lists(self):
        """The evaluation declares one ObjectList output per matching outcome."""
        assert self.bm.visualization_outputs() == {
            "true_positives": ObjectList,
            "false_positives": ObjectList,
            "false_negatives": ObjectList,
            "ignored": ObjectList,
        }

    def test_visualization_splits_outcomes_by_source_object(self):
        """A matched prediction is a TP, an unmatched one an FP, a missed GT an FN."""
        preds = [_pred(x=0.0, obj_id=100), _pred(x=15.0, obj_id=101)]
        # _label assigns the GT IDs 0 and 1, in order.
        gts = [_gt(x=0.0), _gt(x=30.0)]
        outputs = self._visualize(preds, gts)
        assert [obj.id for obj in outputs["true_positives"].objects] == [100]
        assert [obj.id for obj in outputs["false_positives"].objects] == [101]
        assert [obj.id for obj in outputs["false_negatives"].objects] == [1]
        assert outputs["ignored"].objects == []

    def test_visualization_shows_ignored_objects(self):
        """Don't-care labels and the predictions they absorbed are neither positives nor negatives."""
        preds = [_pred(x=0.0, class_type=MOTORCYCLE, obj_id=100)]
        gts = [_gt(x=0.0, classes=(UNKNOWN, MOTORCYCLE)), _gt(x=10.0, classes=(UNCLASSIFIED,))]

        outputs = self._visualize(preds, gts)

        assert [obj.id for obj in outputs["ignored"].objects] == [100, 0, 1]
        assert outputs["true_positives"].objects == outputs["false_positives"].objects == outputs["false_negatives"].objects == []

    def test_visualization_agrees_with_the_scored_matching(self):
        """The split reproduces what was scored at the TP metric threshold."""
        preds = [_pred(x=0.0, obj_id=100), _pred(x=15.0, obj_id=101), _pred(x=20.0, obj_id=102)]
        gts = [_gt(x=0.0), _gt(x=30.0), _gt(x=20.0, classes=(UNCLASSIFIED,))]
        entries = self._entries(preds, gts, threshold=self.bm.tp_metric_threshold)
        outputs = self._visualize(preds, gts)
        assert sum(entry["is_tp"] for entry in entries) == len(outputs["true_positives"].objects)
        assert sum(not entry["is_tp"] for entry in entries) == len(outputs["false_positives"].objects)

    def test_visualization_keeps_the_source_headers(self):
        """Positives carry the prediction header, negatives the label header."""
        # Distinct stamps only to show which header each output copies; a real
        # synchronized frame has the same header on both inputs.
        prediction = _msg([_pred(x=0.0, obj_id=100), _pred(x=15.0, obj_id=101)])
        prediction.header.stamp.sec = 1
        label, label_meta_info = _label([_gt(x=30.0)])
        label.header.stamp.sec = 2
        outputs = self.bm.visualize_sample(prediction, label, label_meta_info=label_meta_info)
        assert outputs["true_positives"].header.stamp.sec == 1
        assert outputs["false_positives"].header.stamp.sec == 1
        assert outputs["false_negatives"].header.stamp.sec == 2
        assert outputs["ignored"].header.stamp.sec == 2

    def test_visualization_shows_predictions_in_the_frame_of_the_labels(self):
        """Predictions of another frame are published as they were compared, transformed into the frame of the labels."""
        self.bm.tf_buffer = _lidar_tf_buffer(x=1.0, yaw=0.0)
        label, label_meta_info = _label([_gt(x=11.0)], frame_id="base_link")

        outputs = self.bm.visualize_sample(_msg([_pred(x=10.0)], frame_id="lidar_top"), label, label_meta_info=label_meta_info)

        assert outputs["true_positives"].header.frame_id == "base_link"
        assert outputs["true_positives"].objects[0].state.continuous_state[HEXAMOTION.X] == pytest.approx(11.0)

    def test_visualization_omits_objects_dropped_before_matching(self):
        """Objects the pre-matching filters remove appear in none of the lists."""
        gts = [
            _gt(x=1.0, classes=(UNKNOWN,)),  # definitely none of the evaluated classes
            _gt(x=60.0),  # beyond the 50 m car range
        ]
        outputs = self._visualize([_pred(x=1.0, class_type=UNKNOWN)], gts)
        assert all(output.objects == [] for output in outputs.values())


# --- frames of single-class labels on which the former nuScenes evaluation was captured ---

_GOLDEN_CLASSES = [CAR, BUS, PEDESTRIAN, BICYCLE, MOTORCYCLE]
# width, length, height
_GOLDEN_SIZES = {
    CAR: (1.9, 4.5, 1.6),
    BUS: (2.9, 11.0, 3.4),
    PEDESTRIAN: (0.7, 0.7, 1.75),
    BICYCLE: (0.6, 1.7, 1.3),
    MOTORCYCLE: (0.8, 2.1, 1.5),
}
# stays inside the 50 m / 40 m ranges, away from their boundaries
_GOLDEN_MAX_DISTANCE = {CAR: 47.0, BUS: 47.0, PEDESTRIAN: 37.0, BICYCLE: 37.0, MOTORCYCLE: 37.0}

_GOLDEN_AP = {
    0.5: {
        "car": 0.006438655561462578,
        "pedestrian": 0.0010158730158730158,
        "motorcycle": 0.00017018043684710363,
        "bus": 0.021245159262588454,
        "bicycle": 0.005559082892416226,
    },
    1.0: {
        "car": 0.13281661340226872,
        "pedestrian": 0.05530632432455618,
        "motorcycle": 0.4022536749655658,
        "bus": 0.267667634048353,
        "bicycle": 0.3114164694593545,
    },
    2.0: {
        "car": 0.24698008294499524,
        "pedestrian": 0.6083969965702566,
        "motorcycle": 0.6399097926325685,
        "bus": 0.6000144127415942,
        "bicycle": 0.5387271131871522,
    },
    4.0: {
        "car": 0.24698008294499524,
        "pedestrian": 0.6083969965702566,
        "motorcycle": 0.754652715806359,
        "bus": 0.6000144127415942,
        "bicycle": 0.6536572243004992,
    },
}
# ATE, ASE, AOE, AVE
_GOLDEN_TP_ERRORS = {
    "car": [0.7755723767215861, 0.254035467593683, 0.3531286311864389, 2.3415086211280576],
    "pedestrian": [1.225186829878604, 0.23432575401126457, 0.2758240374924547, 1.074174945374688],
    "motorcycle": [0.8353333611443023, 0.24469258206819075, 0.36219585892398903, 2.2845158301969084],
    "bus": [0.8538862750819984, 0.24882790667399152, 0.28481636715337927, 2.417204848354466],
    "bicycle": [0.7300753657997088, 0.2692939995456252, 0.2587320404330374, 2.6244096910800643],
}
_GOLDEN_MAP = 0.33508097489047783


def _golden_frames(seed: int = 2026, num_frames: int = 12) -> list:
    """Return frames of (labels, predictions) as plain tuples.

    A label is (class, x, y, width, length, height, yaw, vel_lon, vel_lat), a
    prediction the same followed by its confidence. Detections are noisy,
    sometimes confused with another class or missed, and every frame holds a
    few false positives.
    """
    rng = random.Random(seed)
    frames = []
    for _ in range(num_frames):
        labels, predictions = [], []
        for _ in range(rng.randint(6, 12)):
            cls = rng.choice(_GOLDEN_CLASSES)
            distance, angle = rng.uniform(2.0, _GOLDEN_MAX_DISTANCE[cls]), rng.uniform(-math.pi, math.pi)
            width, length, height = (size * rng.uniform(0.9, 1.1) for size in _GOLDEN_SIZES[cls])
            yaw = rng.uniform(-math.pi, math.pi)
            vel_lon = rng.uniform(0.0, 2.0 if cls == PEDESTRIAN else 12.0)
            vel_lat = rng.uniform(-0.5, 0.5)
            label = (cls, distance * math.cos(angle), distance * math.sin(angle), width, length, height, yaw, vel_lon, vel_lat)
            labels.append(label)
            if rng.random() < 0.8:
                predicted_cls = cls if rng.random() < 0.9 else rng.choice(_GOLDEN_CLASSES)
                predictions.append(
                    (
                        predicted_cls,
                        label[1] + rng.gauss(0.0, 0.7),
                        label[2] + rng.gauss(0.0, 0.7),
                        width * rng.uniform(0.8, 1.2),
                        length * rng.uniform(0.8, 1.2),
                        height * rng.uniform(0.8, 1.2),
                        yaw + rng.gauss(0.0, 0.4),
                        vel_lon + rng.gauss(0.0, 1.0),
                        vel_lat + rng.gauss(0.0, 0.3),
                        rng.uniform(0.2, 1.0),
                    )
                )
        for _ in range(rng.randint(0, 4)):
            cls = rng.choice(_GOLDEN_CLASSES)
            distance, angle = rng.uniform(2.0, _GOLDEN_MAX_DISTANCE[cls]), rng.uniform(-math.pi, math.pi)
            width, length, height = _GOLDEN_SIZES[cls]
            predictions.append(
                (
                    cls,
                    distance * math.cos(angle),
                    distance * math.sin(angle),
                    width,
                    length,
                    height,
                    rng.uniform(-math.pi, math.pi),
                    rng.uniform(0.0, 5.0),
                    0.0,
                    rng.uniform(0.05, 0.7),
                )
            )
        frames.append((labels, predictions))
    return frames
