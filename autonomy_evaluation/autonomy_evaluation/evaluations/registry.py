# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Registry of the evaluations the node runs, selected by their name.

Besides the evaluations of this package, an evaluation implemented in another
package is selected as ``<module>:<class>``, e.g.
``my_package.evaluations:TimeToCollision``, so that a system under test can be
evaluated without adding its evaluation to this package.
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

from autonomy_evaluation.evaluations.Evaluation import Evaluation

# Evaluations of this package by the name they are selected with, as (module, class). A module is
# only imported once its evaluation is selected, so that running one evaluation does not require
# the dependencies of all others.
EVALUATIONS: Dict[str, Tuple[str, str]] = {
    "object_detection_3d": (
        "autonomy_evaluation.evaluations.object_detection.ObjectDetection3D",
        "ObjectDetection3D",
    ),
}


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
