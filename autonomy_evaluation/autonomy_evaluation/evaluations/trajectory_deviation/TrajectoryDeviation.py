# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Evaluation of the deviation of an evaluated trajectory from a reference trajectory.

Compares two trajectories of the ego vehicle, each read from an
``perception_msgs/EgoData`` input and taken either from its planned trajectory
or from the states actually driven (see
:mod:`~autonomy_evaluation.evaluations.trajectory_deviation.trajectories`), so
that one evaluation covers several questions by the sources and topics of its
inputs:

- How well does the controller follow the plan? The ``evaluated`` trajectory
  is the one driven, the ``reference`` the one planned, both read from the same
  topic.
- How far does a planner deviate from a ground truth, e.g. the recorded
  trajectory of a human driver (open-loop planning)? The ``evaluated``
  trajectory is the planned one, the ``reference`` the one driven by the ground
  truth.
- How far do two planners deviate from each other? Both trajectories are
  planned, read from the topics of the two planners.

Metrics, selectable via ``metrics``:

- ``displacement``, ``longitudinal``, ``lateral``, ``heading``, ``velocity``:
  deviations of the evaluated from the reference trajectory.
- ``consistency``: deviation of the planned trajectories of the evaluated input
  from the ones planned at the previous sample.

See :mod:`~autonomy_evaluation.evaluations.trajectory_deviation.DeviationMetrics`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from autonomy_evaluation.evaluations.Evaluation import Evaluation
from autonomy_evaluation.evaluations.trajectory_deviation.DeviationMetrics import (
    Consistency,
    Deviations,
    Displacement,
    Heading,
    Lateral,
    Longitudinal,
    Velocity,
)
from autonomy_evaluation.evaluations.trajectory_deviation.trajectories import DRIVEN, PLANNED, SOURCES


class TrajectoryDeviation(Evaluation):
    """Evaluation of the deviation of an evaluated from a reference trajectory.

    See the module docstring for the use cases and the metrics.
    """

    VERSION = "1.0.0"
    RELEASE_NOTES = {
        "1.0.0": "Initial implementation",
    }
    HEADLINE_METRIC = Displacement.NAME

    def __init__(self) -> None:
        """Name the evaluation."""
        super().__init__(
            name="trajectory_deviation",
            description="Deviation of an evaluated trajectory of the ego vehicle, planned or driven, from a reference "
            "trajectory, e.g. of the driven from the planned trajectory, of a planner from a ground truth, or of two "
            "planners from each other.",
        )

    def metrics(self) -> List[Any]:
        """Compose the evaluation of the deviation metrics and the consistency of the planned trajectories."""
        return [Deviations(), Displacement(), Longitudinal(), Lateral(), Heading(), Velocity(), Consistency()]

    def default_config(self) -> Dict[str, Any]:
        """Configure where the trajectories are taken from and how they are compared, shared by all metrics.

        Returns:
            ``evaluated_source``, ``reference_source``: whether the trajectory
            of each input is the ``planned`` one or the one ``driven``; by
            default, the planned trajectory is compared with the one driven.
            ``horizon``: seconds after each sample over which the trajectories
            are compared. ``time_step``: seconds between the compared times.
        """
        return {"evaluated_source": PLANNED, "reference_source": DRIVEN, "horizon": 5.0, "time_step": 0.1}

    def configure(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply a configuration, checking that the sources are known and the times positive.

        Raises:
            ValueError: The configuration is invalid, see :meth:`Evaluation.configure`.
        """
        effective = super().configure(config)
        for key in ("evaluated_source", "reference_source"):
            if effective[key] not in SOURCES:
                raise ValueError(f"Evaluation '{self.name}' expects '{key}' to be one of {list(SOURCES)}, not {effective[key]!r}")
        for key in ("horizon", "time_step"):
            if effective[key] <= 0.0:
                raise ValueError(f"Evaluation '{self.name}' expects '{key}' to be positive, not {effective[key]!r}")
        return effective
