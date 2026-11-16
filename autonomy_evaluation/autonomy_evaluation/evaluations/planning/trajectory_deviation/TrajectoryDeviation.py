# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Evaluation of the deviation of an evaluated trajectory from a reference trajectory.

Compares two trajectories of the ego vehicle, each read from an input and
taken from its configured source (see
:mod:`~autonomy_evaluation.evaluations.planning.trajectory_deviation.trajectories`): the
trajectory planned in a ``perception_msgs/EgoData`` (``planned``), the states
of ``perception_msgs/EgoData`` actually driven (``driven``), or a
``trajectory_planning_msgs/Trajectory`` (``trajectory``). One evaluation thus
covers several questions by the sources and topics of its inputs:

- How well does the controller follow the plan? The ``evaluated`` trajectory
  is the one driven, the ``reference`` the one planned, read from the same
  ``EgoData`` topic or from the ``Trajectory`` topic of the planner.
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

See :mod:`~autonomy_evaluation.evaluations.planning.trajectory_deviation.DeviationMetrics`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from autonomy_evaluation.evaluations.Evaluation import Evaluation
from autonomy_evaluation.evaluations.Metric import Sample
from autonomy_evaluation.evaluations.planning.trajectory_deviation.DeviationMetrics import (
    Consistency,
    Deviations,
    Displacement,
    Heading,
    Lateral,
    Longitudinal,
    Velocity,
)
from autonomy_evaluation.evaluations.planning.trajectory_deviation.trajectories import (
    DRIVEN,
    PLANNED,
    SOURCES,
    state_pose,
    TRAJECTORY,
)


class TrajectoryDeviation(Evaluation):
    """Evaluation of the deviation of an evaluated from a reference trajectory.

    See the module docstring for the use cases and the metrics.
    """

    VERSION = "1.0.0"
    RELEASE_NOTES = {
        "1.0.0": "Initial implementation",
    }

    AREA = "planning"

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
            of each input is the one ``planned`` in its ``EgoData``, the one
            ``driven``, or a ``trajectory`` message; by default, the planned
            trajectory is compared with the one driven.
            ``fixed_frame``: fixed frame the trajectories are compared in, e.g.
            ``map``; if empty, the frame of the reference.
            ``base_link_to_geometric_center``: translation ``[x, y]`` in meters
            from the origin of ``base_link``, the point of the vehicle the
            sample points of every ``trajectory`` input describe, e.g. its rear
            axle, to its geometric center.
            ``horizon``: seconds after each sample over which the trajectories
            are compared. ``time_step``: seconds between the compared times.
        """
        return {
            "evaluated_source": PLANNED,
            "reference_source": DRIVEN,
            "fixed_frame": "",
            "base_link_to_geometric_center": [0.0, 0.0],
            "horizon": 5.0,
            "time_step": 0.1,
        }

    def sample_position(self, sample: Sample) -> Optional[Tuple[float, float]]:
        """Report the position of the ego vehicle, so that a teleport splits the driven trajectories.

        Returns:
            The position of the ego vehicle of the reference, else of the
            evaluated input, whichever is read as ``EgoData``; ``None`` if both
            are trajectory messages.
        """
        for name in ("reference", "evaluated"):
            message = sample.messages.get(name)
            if message is not None and self.config[f"{name}_source"] != TRAJECTORY:
                x, y, _, _ = state_pose(message)
                return x, y
        return None

    def configure(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply a configuration, checking it and declaring the message types of the inputs by their sources.

        Raises:
            ValueError: The configuration is invalid, see
                :meth:`Evaluation.configure`, e.g. a ``trajectory`` source
                without ``fixed_frame``, as trajectory messages are often given in
                the frame of the vehicle, which moves.
        """
        effective = super().configure(config)
        for key in ("evaluated_source", "reference_source"):
            if effective[key] not in SOURCES:
                raise ValueError(f"Evaluation '{self.name}' expects '{key}' to be one of {list(SOURCES)}, not {effective[key]!r}")
        for key in ("horizon", "time_step"):
            if effective[key] <= 0.0:
                raise ValueError(f"Evaluation '{self.name}' expects '{key}' to be positive, not {effective[key]!r}")
        if len(effective["base_link_to_geometric_center"]) != 2:
            raise ValueError(
                f"Evaluation '{self.name}' expects 'base_link_to_geometric_center' to be [x, y], not "
                f"{effective['base_link_to_geometric_center']!r}"
            )
        sources = {name: effective[f"{name}_source"] for name in ("evaluated", "reference")}
        if TRAJECTORY in sources.values() and not effective["fixed_frame"]:
            raise ValueError(
                f"Evaluation '{self.name}' expects 'fixed_frame' to name a fixed frame to compare in, e.g. 'map', as "
                f"'{TRAJECTORY}' messages are often given in the frame of the vehicle"
            )
        for metric in self.available_metrics().values():
            metric.sources = sources
        return effective
