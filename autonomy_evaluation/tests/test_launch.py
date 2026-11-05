# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for the remapping of the topics and the RViz configuration of the selected evaluations in the launch file.

The launch file does not know the topics of the evaluations in advance, so it remaps whichever topics
the selected evaluations read onto the launch arguments of their names, and shows their visualization
with the RViz configuration of the selection, of an evaluation or of its area, if the package has one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from autonomy_evaluation.evaluations import (
    Aggregation,
    CONTEXT,
    EVALUATED,
    Evaluation,
    Input,
    load_evaluation,
    load_evaluations,
    Metric,
)

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
        remappings = self.input_remappings([self.evaluation], {"prediction": "/object_list/prediction", "name": "evaluation"})

        assert ("prediction", "/object_list/prediction") in remappings

    def test_subscribes_topics_without_launch_argument_in_the_private_namespace(self):
        """A topic that is not given is subscribed in the private namespace of the node."""
        remappings = self.input_remappings([self.evaluation], {})

        assert remappings == [("prediction", "~/prediction"), ("label", "~/label")]

    def test_leaves_a_derived_topic_to_the_node_unless_it_is_given(self):
        """The node derives the meta information topic from the label topic, unless it is given itself."""
        assert "label_meta_info" not in dict(self.input_remappings([self.evaluation], {"label": "/object_list/lidar_01"}))

        remappings = self.input_remappings([self.evaluation], {"label_meta_info": "/meta_info"})

        assert ("label_meta_info", "/meta_info") in remappings


class _Speed(Metric):
    """Metric reading the ego state only."""

    NAME = "speed"

    def inputs(self) -> dict:
        """Read the ego state."""
        return {"ego_data": Input(object, EVALUATED)}

    def aggregate(self, aggregation: Aggregation) -> dict:
        """Report no value, as only the topics are tested."""
        return {"_value_": None}


class _Distance(_Speed):
    """Metric reading the ego state and the objects around it."""

    NAME = "distance"

    def inputs(self) -> dict:
        """Read the ego state and the objects."""
        return {"ego_data": Input(object, EVALUATED), "objects": Input(object, CONTEXT)}


class _PlannerEvaluation(Evaluation):
    """Evaluation of a speed and a distance metric, of which only the speed is selected."""

    def __init__(self):
        """Select the speed only."""
        super().__init__(name="planner")
        self.configure({"metrics": ["speed"]})

    def metrics(self):
        """Compose the evaluation of the speed and the distance."""
        return [_Speed(), _Distance()]


class TestInputRemappingsOfSelectedMetrics:
    """Tests remapping the topics of the metrics an evaluation may compute."""

    def test_remaps_the_topics_of_all_metrics(self):
        """The metrics are only selected by the configuration the node reads, so the topics of all of them are remapped."""
        remappings = _launch_module().input_remappings([_PlannerEvaluation()], {"objects": "/simulation/objects"})

        assert remappings == [("ego_data", "~/ego_data"), ("objects", "/simulation/objects")]


class TestInputRemappingsOfSeveralEvaluations:
    """Tests remapping the topics of several evaluations that run together, e.g. those of an area."""

    def test_remaps_each_topic_once(self):
        """Evaluations reading an input of the same name read it from the same topic, while the map is no topic."""
        remappings = _launch_module().input_remappings(
            load_evaluations("driving_behavior"), {"ego_data": "/ego_data", "objects": "/objects"}
        )

        assert remappings == [("ego_data", "/ego_data"), ("objects", "/objects"), ("route", "~/route")]


class TestRvizConfig:
    """Tests choosing the RViz configuration showing the visualization of the evaluation."""

    def test_area_with_a_configuration_of_its_name(self, tmp_path):
        """The evaluations of an area whose visualization differs from the default share an RViz configuration of its name."""
        for name in ("conf.rviz", "driving_behavior.rviz"):
            (tmp_path / name).write_text("")
        rviz_config = _launch_module().rviz_config

        assert rviz_config("driving_behavior", load_evaluations("driving_behavior"), str(tmp_path)) == str(
            tmp_path / "driving_behavior.rviz"
        )
        assert rviz_config("criticality", [load_evaluation("criticality")], str(tmp_path)) == str(
            tmp_path / "driving_behavior.rviz"
        )

    def test_evaluation_with_a_configuration_of_its_name(self, tmp_path):
        """A configuration of the name of an evaluation takes precedence over that of its area."""
        for name in ("conf.rviz", "driving_behavior.rviz", "criticality.rviz"):
            (tmp_path / name).write_text("")

        assert _launch_module().rviz_config("criticality", [load_evaluation("criticality")], str(tmp_path)) == str(
            tmp_path / "criticality.rviz"
        )

    def test_default_configuration(self, tmp_path):
        """Any other evaluation, e.g. one of another package, is shown with the default configuration."""
        (tmp_path / "conf.rviz").write_text("")
        evaluations = [SimpleNamespace(name="my_evaluation", AREA="")]

        assert _launch_module().rviz_config("my_package.evaluations:MyEvaluation", evaluations, str(tmp_path)) == str(
            tmp_path / "conf.rviz"
        )

    def test_package_holds_a_configuration_per_visualizing_area(self):
        """The evaluations of the driving behavior show their outputs in a configuration of their area."""
        config_directory = Path(__file__).resolve().parents[1] / "config"

        assert (config_directory / "driving_behavior.rviz").is_file()
        assert (config_directory / "conf.rviz").is_file()
