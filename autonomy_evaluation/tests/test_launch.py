# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for the remapping of the topics of the selected evaluation in the launch file.

The launch file does not know the topics of an evaluation in advance, so it remaps whichever topics
the selected evaluation reads onto the launch arguments of their names.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from autonomy_evaluation.evaluations import load_evaluation

_LAUNCH_FILE = Path(__file__).resolve().parents[1] / "launch" / "autonomy_evaluation.launch.py"


def _launch_module():
    """Import the launch file, which is no module of the package."""
    spec = importlib.util.spec_from_file_location("autonomy_evaluation_launch", _LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestInputRemappings:
    """Tests remapping the topics of an evaluation onto the given launch arguments."""

    def setup_method(self):
        """Select the 3D object detection, reading a prediction and a label, optionally with its meta information."""
        self.input_remappings = _launch_module().input_remappings
        self.evaluation = load_evaluation("object_detection_3d")

    def test_remaps_the_topics_given_as_launch_arguments(self):
        """A topic given by a launch argument of its name is subscribed there."""
        remappings = self.input_remappings(self.evaluation, {"prediction": "/object_list/prediction", "name": "evaluation"})

        assert ("prediction", "/object_list/prediction") in remappings

    def test_subscribes_topics_without_launch_argument_in_the_private_namespace(self):
        """A topic that is not given is subscribed in the private namespace of the node."""
        remappings = self.input_remappings(self.evaluation, {})

        assert remappings == [("prediction", "~/prediction"), ("label", "~/label")]

    def test_leaves_a_derived_topic_to_the_node_unless_it_is_given(self):
        """The node derives the meta information topic from the label topic, unless it is given itself."""
        assert "label_meta_info" not in dict(self.input_remappings(self.evaluation, {"label": "/object_list/lidar_01"}))

        remappings = self.input_remappings(self.evaluation, {"label_meta_info": "/meta_info"})

        assert ("label_meta_info", "/meta_info") in remappings
