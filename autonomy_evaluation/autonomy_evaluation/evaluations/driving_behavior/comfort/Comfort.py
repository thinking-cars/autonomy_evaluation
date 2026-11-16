# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Evaluation of the comfort of the ego vehicle.

How smoothly the ego vehicle a system under test drives. Reads the ``ego_data``
of the ego vehicle (evaluated) only. Metrics, selectable via ``metrics``:

- ``acceleration``: longitudinal and lateral acceleration.
- ``jerk``: time derivative of the longitudinal and lateral acceleration.
- ``yaw_rate``: rate of turn.

All of them build on the motion of the ego vehicle, which the internal
``ego_motion`` metric derives once. See
:mod:`~autonomy_evaluation.evaluations.driving_behavior.comfort.ComfortMetrics`.
"""

from __future__ import annotations

from typing import Any, List

from autonomy_evaluation.evaluations.driving_behavior.comfort.ComfortMetrics import Acceleration, EgoMotion, Jerk, YawRate
from autonomy_evaluation.evaluations.driving_behavior.DrivingBehaviorEvaluation import DrivingBehaviorEvaluation


class Comfort(DrivingBehaviorEvaluation):
    """Evaluation of the comfort of the ego vehicle, see the module docstring."""

    VERSION = "1.0.0"
    RELEASE_NOTES = {
        "1.0.0": "Initial implementation",
    }

    def __init__(self) -> None:
        """Name the evaluation."""
        super().__init__(
            name="comfort",
            description="Comfort of the ego vehicle a system under test drives: its acceleration, jerk and yaw rate.",
        )

    def metrics(self) -> List[Any]:
        """Compose the evaluation of the acceleration, the jerk and the yaw rate, building on the motion of the ego vehicle."""
        return [EgoMotion(), Acceleration(), Jerk(), YawRate()]
