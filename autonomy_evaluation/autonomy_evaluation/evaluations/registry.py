# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Registry of the evaluations the node runs, selected by their name or by the name of their area.

The node runs one or more evaluations, selected by a comma-separated list of
names. Each name is the name of an evaluation of this package, the name of an
area, which selects all evaluations of the area, e.g. ``driving_behavior``, or
``<module>:<class>`` of an evaluation implemented in another package, e.g.
``my_package.evaluations:MyPlannerEvaluation``, so that a system under test can
be evaluated without adding its evaluation to this package.
"""

from __future__ import annotations

import importlib
from typing import Dict, List, Tuple

from autonomy_evaluation.evaluations.Evaluation import Evaluation

# Evaluations of this package by the name they are selected with, as (module, class), ordered by their area (see
# Evaluation.AREA). A module is only imported once its evaluation is selected, so that running one evaluation does
# not require the dependencies of all others.
EVALUATIONS: Dict[str, Tuple[str, str]] = {
    # perception
    "object_detection_3d": (
        "autonomy_evaluation.evaluations.perception.object_detection.ObjectDetection3D",
        "ObjectDetection3D",
    ),
    # planning
    "trajectory_deviation": (
        "autonomy_evaluation.evaluations.planning.trajectory_deviation.TrajectoryDeviation",
        "TrajectoryDeviation",
    ),
    # driving behavior
    "criticality": (
        "autonomy_evaluation.evaluations.driving_behavior.criticality.Criticality",
        "Criticality",
    ),
    "rule_compliance": (
        "autonomy_evaluation.evaluations.driving_behavior.rule_compliance.RuleCompliance",
        "RuleCompliance",
    ),
    "progress": (
        "autonomy_evaluation.evaluations.driving_behavior.progress.Progress",
        "Progress",
    ),
    "comfort": (
        "autonomy_evaluation.evaluations.driving_behavior.comfort.Comfort",
        "Comfort",
    ),
}

# Evaluations of this package of each area (see Evaluation.AREAS), in the order they run in when the area is selected
# by its name.
AREA_EVALUATIONS: Dict[str, Tuple[str, ...]] = {
    "perception": ("object_detection_3d",),
    "planning": ("trajectory_deviation",),
    "driving_behavior": ("criticality", "rule_compliance", "progress", "comfort"),
}

#: Separator of the names of a selection of several evaluations, e.g. ``criticality,comfort``.
SELECTION_SEPARATOR = ","


def resolve_selection(selection: str) -> List[str]:
    """Resolve a selection of evaluations into the names of the selected evaluations.

    Parameters
    ----------
    selection:
        Comma-separated names, each of an evaluation of this package (see
        :data:`EVALUATIONS`), of an area, which selects all evaluations of the
        area (see :data:`AREA_EVALUATIONS`), or ``<module>:<class>`` of an
        evaluation implemented in another package, e.g.
        ``"driving_behavior"`` or ``"criticality,comfort"``.

    Returns
    -------
    The names of the selected evaluations, in the order they are selected in,
    each once.

    Raises
    ------
    ValueError
        If the selection names nothing, or a name is neither an evaluation of
        this package, an area nor ``<module>:<class>``.
    """
    names: List[str] = []
    for name in (name.strip() for name in selection.split(SELECTION_SEPARATOR)):
        if not name:
            continue
        if name in AREA_EVALUATIONS:
            selected = list(AREA_EVALUATIONS[name])
        elif name in EVALUATIONS or ":" in name:
            selected = [name]
        else:
            raise ValueError(
                f"Unknown evaluation '{name}', expected one of {sorted(EVALUATIONS)}, an area of {list(AREA_EVALUATIONS)} "
                "or '<module>:<class>'"
            )
        names.extend(selected_name for selected_name in selected if selected_name not in names)
    if not names:
        raise ValueError(f"Selection '{selection}' names no evaluation")
    return names


def load_evaluation(name: str) -> Evaluation:
    """Instantiate the evaluation selected by its name.

    Parameters
    ----------
    name:
        Name of an evaluation of this package (see :data:`EVALUATIONS`), or
        ``<module>:<class>`` of an :class:`Evaluation` subclass implemented in
        another package, which is instantiated without arguments.

    Returns
    -------
    The selected evaluation.

    Raises
    ------
    ValueError
        If no evaluation of this name exists, or its class cannot be loaded.
    """
    if name in EVALUATIONS:
        module_name, class_name = EVALUATIONS[name]
    else:
        module_name, _, class_name = name.partition(":")
        if not module_name or not class_name:
            raise ValueError(f"Unknown evaluation '{name}', expected one of {sorted(EVALUATIONS)} or '<module>:<class>'")

    try:
        evaluation_class = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exception:
        raise ValueError(f"Cannot load evaluation '{name}' from '{module_name}:{class_name}': {exception}") from exception
    if not isinstance(evaluation_class, type) or not issubclass(evaluation_class, Evaluation):
        raise ValueError(f"Evaluation '{name}' is no subclass of {Evaluation.__module__}.{Evaluation.__name__}")
    return evaluation_class()


def load_evaluations(selection: str) -> List[Evaluation]:
    """Instantiate the evaluations of a selection.

    Parameters
    ----------
    selection:
        Comma-separated names of evaluations or areas, see :func:`resolve_selection`.

    Returns
    -------
    The selected evaluations, in the order they are selected in.

    Raises
    ------
    ValueError
        If the selection names no or an unknown evaluation, an evaluation
        cannot be loaded, or two selected evaluations share their name, under
        which their configuration and their results are kept apart.
    """
    evaluations = [load_evaluation(name) for name in resolve_selection(selection)]
    names = [evaluation.name for evaluation in evaluations]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Selection '{selection}' selects several evaluations named {duplicates}")
    return evaluations
