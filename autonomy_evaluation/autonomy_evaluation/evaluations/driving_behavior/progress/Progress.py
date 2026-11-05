# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Evaluation of the progress of the ego vehicle.

Whether the ego vehicle a system under test drives gets ahead. Reads the
``ego_data`` of the ego vehicle (evaluated) and, if published, the ``route``
planned for it (context). Metrics, selectable via ``metrics``:

- ``distance``: distance the ego vehicle drives, its mean speed and how long it
  stands still.
- ``route``: how far the ego vehicle gets along its route, towards its
  destination.

See :mod:`~autonomy_evaluation.evaluations.driving_behavior.progress.ProgressMetrics`.
"""

from __future__ import annotations

from typing import Any, List

from autonomy_evaluation.evaluations.driving_behavior.DrivingBehaviorEvaluation import DrivingBehaviorEvaluation
from autonomy_evaluation.evaluations.driving_behavior.progress.ProgressMetrics import DistanceDriven, RouteProgress


class Progress(DrivingBehaviorEvaluation):
    """Evaluation of the progress of the ego vehicle, see the module docstring."""

    VERSION = "1.0.0"
    RELEASE_NOTES = {
        "1.0.0": "Initial implementation",
    }

    def __init__(self) -> None:
        """Name the evaluation."""
        super().__init__(
            name="progress",
            description="Progress of the ego vehicle a system under test drives: the distance it drives, how long it "
            "stands still, and how far it gets along its route.",
        )

    def metrics(self) -> List[Any]:
        """Compose the evaluation of the distance driven and the progress along the route."""
        return [DistanceDriven(), RouteProgress()]
