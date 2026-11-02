# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Closed-loop evaluation of the driving behavior of the ego vehicle.

Evaluates the ego vehicle a system under test drives, e.g. a planner in a
closed-loop simulation, by its behavior over the whole run: its criticality
towards the other road users and its comfort, i.e. how smoothly it drives.

The evaluation reads the ``ego_data`` of the ego vehicle (evaluated) and, for
the metrics that involve other road users, the ``objects`` around it
(context). Usually, a recorded simulation run is replayed by
``autonomy_datasets`` with the dataset as sample source, so that the samples
are attributed to the scenes of the recordings; with an external sample source,
the run ends when its publisher calls the ``~/finish`` service of the node.

Metrics, grouped and selectable via ``metrics``:

- ``criticality``: ``collision``, ``ttc``, ``thw`` and ``pet`` of the ego
  vehicle towards the other road users
  (:mod:`~autonomy_evaluation.evaluations.closed_loop.CriticalityMetrics`).
- ``comfort``: ``acceleration``, ``jerk`` and ``yaw_rate`` of the ego vehicle
  (:mod:`~autonomy_evaluation.evaluations.closed_loop.ComfortMetrics`).

Every metric over time is computed per contiguous sequence of samples, see
:mod:`~autonomy_evaluation.evaluations.sampling`.
"""

from __future__ import annotations

from typing import Any, Dict, List

from autonomy_evaluation.evaluations.closed_loop.ComfortMetrics import Acceleration, EgoMotion, Jerk, YawRate
from autonomy_evaluation.evaluations.closed_loop.CriticalityMetrics import (
    Collision,
    PostEncroachmentTime,
    TimeHeadway,
    TimeToCollision,
)
from autonomy_evaluation.evaluations.Evaluation import Evaluation
from autonomy_evaluation.evaluations.Metric import MetricGroup


class ClosedLoop(Evaluation):
    """Closed-loop evaluation of the criticality and the comfort of the ego vehicle.

    See the module docstring for the metrics and their inputs.
    """

    VERSION = "1.0.0"
    RELEASE_NOTES = {
        "1.0.0": "Initial implementation",
    }

    def __init__(self) -> None:
        """Name the evaluation."""
        super().__init__(
            name="closed_loop",
            description="Driving behavior of the ego vehicle a system under test drives in closed loop: its criticality "
            "towards other road users and its comfort.",
        )

    def metrics(self) -> List[Any]:
        """Compose the evaluation of the criticality and the comfort metrics."""
        return [
            EgoMotion(),
            MetricGroup("criticality", [Collision(), TimeToCollision(), TimeHeadway(), PostEncroachmentTime()]),
            MetricGroup("comfort", [Acceleration(), Jerk(), YawRate()]),
        ]

    def default_config(self) -> Dict[str, Any]:
        """Configure how the other road users are read, shared by all metrics involving them.

        Returns:
            ``ego_object_id``: ID of the ego vehicle among the objects, which
            some simulations list, and which is left out of them; ``-1`` if
            the objects do not contain the ego vehicle.
        """
        return {"ego_object_id": -1}
