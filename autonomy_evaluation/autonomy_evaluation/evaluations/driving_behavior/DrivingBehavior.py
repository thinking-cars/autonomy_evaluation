# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Evaluations of the driving behavior of the ego vehicle.

Evaluates the ego vehicle a system under test drives, e.g. an automated driving
stack in a closed-loop simulation, by its behavior over the whole run: its
criticality towards the other road users, its compliance with the traffic
rules, its progress and its comfort, i.e. how smoothly it drives.

The evaluation reads the ``ego_data`` of the ego vehicle (evaluated) and, for
the metrics that involve other road users, the ``objects`` around it
(context). Usually, a recorded simulation run is replayed by
``autonomy_datasets`` with the dataset as sample source, so that the samples
are attributed to the scenes of the recordings; with an external sample source,
the run ends when its publisher calls the ``~/finish`` service of the node.

The area ``driving_behavior`` holds one evaluation per aspect, each a group of
metrics, which all run together and are selected via ``metrics``:

- ``criticality``: ``collision``, ``ttc``, ``thw`` and ``pet`` of the ego
  vehicle towards the other road users
  (:mod:`~autonomy_evaluation.evaluations.driving_behavior.CriticalityMetrics`).
- ``rule_compliance``: ``drivable_area``, ``driving_direction`` and
  ``speed_limit`` of the ego vehicle, checked against the Lanelet2 map
  (:mod:`~autonomy_evaluation.evaluations.driving_behavior.RuleComplianceMetrics`).
- ``progress``: ``distance`` the ego vehicle drives and its progress along
  its ``route``
  (:mod:`~autonomy_evaluation.evaluations.driving_behavior.ProgressMetrics`).
- ``comfort``: ``acceleration``, ``jerk`` and ``yaw_rate`` of the ego vehicle
  (:mod:`~autonomy_evaluation.evaluations.driving_behavior.ComfortMetrics`).

Every metric over time is computed per contiguous sequence of samples, see
:mod:`~autonomy_evaluation.evaluations.sampling`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from autonomy_evaluation.evaluations.driving_behavior.ComfortMetrics import Acceleration, EgoMotion, Jerk, YawRate
from autonomy_evaluation.evaluations.driving_behavior.CriticalityMetrics import (
    Collision,
    PostEncroachmentTime,
    TimeHeadway,
    TimeToCollision,
)
from autonomy_evaluation.evaluations.driving_behavior.ProgressMetrics import DistanceDriven, RouteProgress
from autonomy_evaluation.evaluations.driving_behavior.RuleComplianceMetrics import (
    DrivableArea,
    DrivingDirection,
    SpeedLimit,
)
from autonomy_evaluation.evaluations.driving_behavior.states import ego_state
from autonomy_evaluation.evaluations.Evaluation import Evaluation
from autonomy_evaluation.evaluations.Metric import MetricGroup, Sample


class DrivingBehavior(Evaluation):
    """Evaluation of the criticality, the rule compliance, the progress and the comfort of the ego vehicle.

    See the module docstring for the metrics and their inputs.
    """

    VERSION = "1.2.0"
    RELEASE_NOTES = {
        "1.0.0": "Initial implementation",
        "1.1.0": "Rule compliance metrics: drivable area, driving direction and speed limit; progress: distance driven",
        "1.2.0": "Progress along the planned route",
    }

    AREA = "driving_behavior"

    def __init__(self) -> None:
        """Name the evaluation."""
        super().__init__(
            name="driving_behavior",
            description="Driving behavior of the ego vehicle a system under test drives, e.g. in a closed-loop "
            "simulation: its criticality towards other road users, its compliance with the traffic rules, its progress "
            "and its comfort.",
        )

    def metrics(self) -> List[Any]:
        """Compose the evaluation of the criticality, the rule compliance, the progress and the comfort metrics."""
        return [
            EgoMotion(),
            MetricGroup("criticality", [Collision(), TimeToCollision(), TimeHeadway(), PostEncroachmentTime()]),
            MetricGroup("rule_compliance", [DrivableArea(), DrivingDirection(), SpeedLimit()]),
            MetricGroup("progress", [DistanceDriven(), RouteProgress()]),
            MetricGroup("comfort", [Acceleration(), Jerk(), YawRate()]),
        ]

    def sample_position(self, sample: Sample) -> Optional[Tuple[float, float]]:
        """Report the position of the ego vehicle, so that a teleport, e.g. to the start of a scenario, splits the sequence."""
        ego = ego_state(sample)
        return ego.x, ego.y

    def default_config(self) -> Dict[str, Any]:
        """Configure how the other road users are read, shared by all metrics involving them.

        Returns:
            ``ego_object_id``: ID of the ego vehicle among the objects, which
            some simulations list, and which is left out of them; ``-1`` if
            the objects do not contain the ego vehicle.
        """
        return {"ego_object_id": -1}
