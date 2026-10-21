# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Closed-loop evaluation of the driving behavior of the ego vehicle.

Evaluates the ego vehicle a system under test drives, e.g. a planner in a
closed-loop simulation, by its behavior over the whole run: its comfort, i.e.
how smoothly it drives, and its criticality towards the other road users.

The evaluation reads the ``ego_data`` of the ego vehicle (evaluated) and, for
the metrics that involve other road users, the ``objects`` around it
(context). Usually, a recorded simulation run is replayed by
``autonomy_datasets`` with the dataset as sample source, so that the samples
are attributed to the scenes of the recordings; with an external sample source,
the run ends when its publisher calls the ``~/finish`` service of the node.

Metrics, grouped and selectable via ``metrics``:

- ``comfort``: ``acceleration``, ``jerk`` and ``yaw_rate`` of the ego vehicle
  (:mod:`~autonomy_evaluation.evaluations.closed_loop.ComfortMetrics`).

Every metric over time is computed per contiguous sequence of samples, see
:mod:`~autonomy_evaluation.evaluations.sampling`.
"""

from __future__ import annotations

from typing import Any, List

from autonomy_evaluation.evaluations.closed_loop.ComfortMetrics import Acceleration, EgoMotion, Jerk, YawRate
from autonomy_evaluation.evaluations.Evaluation import Evaluation
from autonomy_evaluation.evaluations.Metric import MetricGroup


class ClosedLoop(Evaluation):
    """Closed-loop evaluation of the comfort of the ego vehicle.

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
            description="Driving behavior of the ego vehicle a system under test drives in closed loop.",
        )

    def metrics(self) -> List[Any]:
        """Compose the evaluation of the comfort metrics, building on the motion of the ego vehicle."""
        return [
            EgoMotion(),
            MetricGroup("comfort", [Acceleration(), Jerk(), YawRate()]),
        ]
