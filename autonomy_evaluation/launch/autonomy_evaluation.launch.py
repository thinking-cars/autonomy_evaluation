#!/usr/bin/env python3

# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Mapping

from ament_index_python.packages import get_package_share_directory
from autonomy_evaluation.evaluations import Evaluation, load_evaluation
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue


def input_remappings(evaluation: Evaluation, launch_configurations: Mapping[str, str]) -> list[tuple[str, str]]:
    """Remap the topics the evaluation may read onto the topics given by the launch arguments of their names.

    Every topic any metric of the evaluation reads, whatever its role and whether it is optional, is configured by
    a launch argument of its name, e.g. ``prediction:=/object_list/prediction``. As the metrics the evaluation
    computes are only selected by its configuration, the topics of all its metrics are remapped; the node only
    subscribes to those of the computed metrics. A topic without such an argument is subscribed in the private
    namespace of the node, except for a topic published next to another one, which the node derives from the
    topic of that one.

    Args:
        evaluation (Evaluation): evaluation to run
        launch_configurations (Mapping[str, str]): values of the launch arguments by name

    Returns:
        list[tuple[str, str]]: remappings of the node-relative input names
    """
    remappings = []
    for name, declared in evaluation.available_inputs().items():
        if name in launch_configurations:
            remappings.append((name, launch_configurations[name]))
        elif declared.next_to is None:
            remappings.append((name, f"~/{name}"))
    return remappings


def rviz_config(evaluation_name: str, config_directory: str) -> str:
    """Choose the RViz configuration showing the visualization of the evaluation.

    Args:
        evaluation_name (str): name the evaluation is selected with
        config_directory (str): directory holding the RViz configurations

    Returns:
        str: path of '<evaluation>.rviz' if the directory holds one, of the default 'conf.rviz' otherwise
    """
    path = os.path.join(config_directory, f"{evaluation_name}.rviz")
    return path if os.path.isfile(path) else os.path.join(config_directory, "conf.rviz")


def launch_node(context: LaunchContext, remappings: list) -> list[Node]:
    """Launch the node with the topics of the selected evaluation remapped, and RViz showing its visualization.

    Args:
        context (LaunchContext): launch context holding the launch arguments
        remappings (list): remappings of the names other than the topics of the evaluation

    Returns:
        list[Node]: node and, if visualizing, RViz to launch
    """
    evaluation_name = LaunchConfiguration("evaluation").perform(context)
    evaluation = load_evaluation(evaluation_name)
    # a parameter file configures the evaluation, e.g. which metrics it computes, by parameters named after it
    config = LaunchConfiguration("config").perform(context)
    node = Node(
        package="autonomy_evaluation",
        executable="autonomy_evaluation",
        namespace=LaunchConfiguration("namespace"),
        name=LaunchConfiguration("name"),
        parameters=[
            *([config] if config else []),
            {"evaluation": LaunchConfiguration("evaluation")},
            {"visualize": ParameterValue(LaunchConfiguration("visualize"), value_type=bool)},
            {"sample_source": LaunchConfiguration("sample_source")},
            {"sync_tolerance": ParameterValue(LaunchConfiguration("sync_tolerance"), value_type=float)},
            {"samples_per_request": ParameterValue(LaunchConfiguration("samples_per_request"), value_type=int)},
            {"sample_ids": ParameterValue(LaunchConfiguration("sample_ids"), value_type=str)},
            {"evaluation_timeout": ParameterValue(LaunchConfiguration("evaluation_timeout"), value_type=float)},
            {"results_path": ParameterValue(LaunchConfiguration("results_path"), value_type=str)},
        ],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
        remappings=input_remappings(evaluation, context.launch_configurations) + remappings,
        output="screen",
        emulate_tty=True,
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=[
            "-d",
            rviz_config(evaluation_name, os.path.join(get_package_share_directory("autonomy_evaluation"), "config")),
        ],
        condition=IfCondition(LaunchConfiguration("visualize")),
        output="screen",
    )
    return [node, rviz]


def generate_launch_description():
    """Create and return the launch description for the autonomy_evaluation node."""

    # Service of the dataset node the evaluation requests the samples to evaluate from, remapped
    # onto the topic its argument resolves to just like the evaluation's data inputs.
    remappable_topics = [
        DeclareLaunchArgument(
            "request_samples",
            default_value="~/request_samples",
            description="service of the dataset node used to request the samples to evaluate",
        ),
    ]

    args = [
        *remappable_topics,
        DeclareLaunchArgument(
            "evaluation",
            default_value="object_detection_3d",
            description="name of an evaluation of this package, or '<module>:<class>' of an evaluation implemented in "
            "another package; each topic the evaluation reads is set by an argument of its name, e.g. prediction:=/topic",
        ),
        DeclareLaunchArgument(
            "config",
            default_value="",
            description="path of a ROS parameter file configuring the evaluation, e.g. which metrics it computes, by "
            "parameters named '<evaluation>.<entry>' (default configuration if empty)",
        ),
        DeclareLaunchArgument("name", default_value="autonomy_evaluation", description="node name"),
        DeclareLaunchArgument("namespace", default_value="", description="node namespace"),
        DeclareLaunchArgument(
            "log_level", default_value="info", description="ROS logging level (debug, info, warn, error, fatal)"
        ),
        DeclareLaunchArgument("use_sim_time", default_value="true", description="use simulation clock"),
        DeclareLaunchArgument(
            "visualize",
            default_value="false",
            choices=["true", "false"],
            description="publish the per-sample visualization of the evaluation and open RViz on it",
        ),
        DeclareLaunchArgument(
            "sample_source",
            default_value="dataset",
            choices=["dataset", "external"],
            description="'dataset' requests the samples to evaluate from the dataset one after another; 'external' "
            "evaluates the samples published by others, e.g. by a simulation or the playback panel in RViz",
        ),
        DeclareLaunchArgument(
            "sync_tolerance",
            default_value="0.0",
            description="seconds by which the stamps of the messages of a sample may differ (0 matches exact stamps only)",
        ),
        DeclareLaunchArgument(
            "samples_per_request",
            default_value="1",
            description="number of samples to request from the dataset at a time (0 requests all remaining samples at once)",
        ),
        DeclareLaunchArgument(
            "sample_ids",
            default_value="",
            description="comma-separated IDs of the dataset samples to evaluate (all samples if empty)",
        ),
        DeclareLaunchArgument(
            "evaluation_timeout",
            default_value="60.0",
            description="seconds to wait for a published sample to be evaluated before continuing without it",
        ),
        DeclareLaunchArgument(
            "results_path",
            default_value="",
            description="path of the JSON file the evaluation results are written to (results are only logged if empty)",
        ),
    ]

    # The topics the evaluation reads and the visualization RViz shows depend on the selected
    # evaluation, so the node and RViz are only created once the launch arguments are known
    remappings = [(la.default_value[0].text, LaunchConfiguration(la.name)) for la in remappable_topics]
    nodes = OpaqueFunction(function=launch_node, kwargs={"remappings": remappings})

    return LaunchDescription(
        [
            *args,
            SetParameter("use_sim_time", LaunchConfiguration("use_sim_time")),
            nodes,
        ]
    )
