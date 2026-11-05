# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Evaluation of the criticality of the ego vehicle towards the other road users.

How close the ego vehicle a system under test drives comes to colliding with
the other road users, or does. Reads the ``ego_data`` of the ego vehicle
(evaluated), the ``objects`` around it (context) and, if given, the Lanelet2
``map`` of the current scene (context). Metrics, selectable via ``metrics``:

- ``collision``: collisions of the ego vehicle, and whether it is at fault.
- ``ttc``: time to collision, with the time exposed to and the time integrated
  over small times to collision.
- ``thw``: time headway to the road user the ego vehicle follows.
- ``pet``: post-encroachment time between the ego vehicle and the road users
  passing the same spot.

See :mod:`~autonomy_evaluation.evaluations.driving_behavior.criticality.CriticalityMetrics`.
"""

from __future__ import annotations

from typing import Any, Dict, List

from autonomy_evaluation.evaluations.driving_behavior.criticality.CriticalityMetrics import (
    Collision,
    PostEncroachmentTime,
    TimeHeadway,
    TimeToCollision,
)
from autonomy_evaluation.evaluations.driving_behavior.DrivingBehaviorEvaluation import DrivingBehaviorEvaluation


class Criticality(DrivingBehaviorEvaluation):
    """Evaluation of the criticality of the ego vehicle towards the other road users, see the module docstring."""

    VERSION = "1.0.0"
    RELEASE_NOTES = {
        "1.0.0": "Initial implementation",
    }

    def __init__(self) -> None:
        """Name the evaluation."""
        super().__init__(
            name="criticality",
            description="Criticality of the ego vehicle a system under test drives towards the other road users: how "
            "close it comes to colliding with them, or does.",
        )

    def metrics(self) -> List[Any]:
        """Compose the evaluation of the collision, the time to collision, the time headway and the post-encroachment time."""
        return [Collision(), TimeToCollision(), TimeHeadway(), PostEncroachmentTime()]

    def default_config(self) -> Dict[str, Any]:
        """Configure how the other road users are read, shared by all metrics.

        Returns:
            ``ego_object_id``: ID of the ego vehicle among the objects, which
            some simulations list, and which is left out of them; ``-1`` if
            the objects do not contain the ego vehicle.
        """
        return {"ego_object_id": -1}
