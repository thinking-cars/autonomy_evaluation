# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for selecting the evaluation the node runs by its name.

Evaluations of this package are selected by their registered name, evaluations implemented in
another package by ``<module>:<class>``. A name that selects no evaluation stops the node, so it
has to be rejected with a message naming the problem.
"""

from __future__ import annotations

import pytest
from autonomy_evaluation.evaluations import Evaluation, EVALUATIONS, load_evaluation
from autonomy_evaluation.evaluations.object_detection.ObjectDetection3D import ObjectDetection3D


class TestLoadEvaluation:
    """Tests instantiating an evaluation by its name."""

    @pytest.mark.parametrize("name", sorted(EVALUATIONS))
    def test_loads_every_registered_evaluation(self, name):
        """Every evaluation of this package can be selected by its registered name."""
        assert isinstance(load_evaluation(name), Evaluation)

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
