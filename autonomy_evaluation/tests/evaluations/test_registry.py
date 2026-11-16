# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for selecting the evaluations the node runs by their names or the names of their areas.

Evaluations of this package are selected by their registered name, all evaluations of an area by the
name of the area, evaluations implemented in another package by ``<module>:<class>``, and several of
them as comma-separated list. A name that selects no evaluation stops the node, so it has to be
rejected with a message naming the problem.
"""

from __future__ import annotations

import pytest
from autonomy_evaluation.evaluations import (
    AREA_EVALUATIONS,
    Evaluation,
    EVALUATIONS,
    load_evaluation,
    load_evaluations,
    resolve_selection,
)
from autonomy_evaluation.evaluations.Evaluation import AREAS
from autonomy_evaluation.evaluations.perception.object_detection.ObjectDetection3D import ObjectDetection3D


class TestLoadEvaluation:
    """Tests instantiating an evaluation by its name."""

    @pytest.mark.parametrize("name", sorted(EVALUATIONS))
    def test_loads_every_registered_evaluation(self, name):
        """Every evaluation of this package can be selected by its registered name."""
        assert isinstance(load_evaluation(name), Evaluation)

    @pytest.mark.parametrize("name", sorted(EVALUATIONS))
    def test_every_registered_evaluation_has_an_area(self, name):
        """Every evaluation of this package tells its area, and lies in the package of that area."""
        evaluation = load_evaluation(name)

        assert evaluation.AREA in AREAS
        assert type(evaluation).__module__.startswith(f"autonomy_evaluation.evaluations.{evaluation.AREA}.")

    def test_loads_a_registered_evaluation_by_its_name(self):
        """The registered name selects the evaluation it is registered for."""
        assert isinstance(load_evaluation("object_detection_3d"), ObjectDetection3D)

    def test_loads_an_evaluation_by_its_module_and_class(self):
        """An evaluation of another package is selected as '<module>:<class>'."""
        evaluation = load_evaluation(f"{ObjectDetection3D.__module__}:{ObjectDetection3D.__name__}")

        assert isinstance(evaluation, ObjectDetection3D)

    @pytest.mark.parametrize(
        "name, reason",
        [
            ("unknown_evaluation", "Unknown evaluation"),
            ("autonomy_evaluation.evaluations:", "Unknown evaluation"),
            ("no_such_package.evaluations:Evaluation", "Cannot load"),
            ("autonomy_evaluation.evaluations:NoSuchEvaluation", "Cannot load"),
            ("autonomy_evaluation.utils.ObjectDetectionUtils:ObjectDetectionUtils", "no subclass"),
        ],
    )
    def test_rejects_a_name_that_selects_no_evaluation(self, name, reason):
        """Names of no evaluation are rejected with the reason they cannot be loaded."""
        with pytest.raises(ValueError, match=reason):
            load_evaluation(name)


class TestAreas:
    """Tests the evaluations registered for each area."""

    def test_every_area_has_evaluations(self):
        """Each area can be selected by its name."""
        assert list(AREA_EVALUATIONS) == list(AREAS)
        assert all(AREA_EVALUATIONS[area] for area in AREAS)

    @pytest.mark.parametrize("area", AREAS)
    def test_every_evaluation_of_an_area_belongs_to_it(self, area):
        """An area selects exactly the evaluations that tell it as their area."""
        assert [load_evaluation(name).AREA for name in AREA_EVALUATIONS[area]] == [area] * len(AREA_EVALUATIONS[area])

    def test_every_evaluation_belongs_to_one_area(self):
        """No evaluation of this package is left out of the selection of its area, or selected by two areas."""
        assert sorted(name for names in AREA_EVALUATIONS.values() for name in names) == sorted(EVALUATIONS)

    def test_area_and_evaluation_names_differ(self):
        """A name selects either an area or an evaluation."""
        assert not set(AREA_EVALUATIONS) & set(EVALUATIONS)


class TestResolveSelection:
    """Tests resolving a selection into the names of the selected evaluations."""

    @pytest.mark.parametrize(
        "selection, names",
        [
            ("comfort", ["comfort"]),
            ("driving_behavior", ["criticality", "rule_compliance", "progress", "comfort"]),
            ("criticality,comfort", ["criticality", "comfort"]),
            (" comfort , criticality ", ["comfort", "criticality"]),
            ("perception,planning", ["object_detection_3d", "trajectory_deviation"]),
            ("comfort,driving_behavior", ["comfort", "criticality", "rule_compliance", "progress"]),
            ("my_package.evaluations:MyEvaluation,comfort", ["my_package.evaluations:MyEvaluation", "comfort"]),
        ],
    )
    def test_resolves_evaluations_and_areas_in_their_order(self, selection, names):
        """An area selects all its evaluations, and each evaluation is selected once, where it is selected first."""
        assert resolve_selection(selection) == names

    @pytest.mark.parametrize(
        "selection, reason", [("", "names no evaluation"), (" , ", "names no evaluation"), ("comfort,driving", "Unknown")]
    )
    def test_rejects_a_selection_of_no_or_an_unknown_evaluation(self, selection, reason):
        """A selection naming nothing or something unknown is rejected."""
        with pytest.raises(ValueError, match=reason):
            resolve_selection(selection)


class TestLoadEvaluations:
    """Tests instantiating the evaluations of a selection."""

    def test_loads_all_evaluations_of_an_area(self):
        """The evaluations of an area are loaded in their order."""
        assert [evaluation.name for evaluation in load_evaluations("driving_behavior")] == [
            "criticality",
            "rule_compliance",
            "progress",
            "comfort",
        ]

    def test_rejects_evaluations_of_the_same_name(self):
        """Evaluations are configured and report their results by their names, which therefore must differ."""
        with pytest.raises(ValueError, match="several evaluations named"):
            load_evaluations(f"object_detection_3d,{ObjectDetection3D.__module__}:{ObjectDetection3D.__name__}")
