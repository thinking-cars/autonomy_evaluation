# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Base class of the evaluations of the driving behavior of the ego vehicle.

The area ``driving_behavior`` evaluates the ego vehicle a system under test
drives, e.g. an automated driving stack in a closed-loop simulation, by its
behavior over the whole run, in one evaluation per aspect:

- ``criticality``: how close the ego vehicle comes to colliding with the other
  road users, or does
  (:mod:`~autonomy_evaluation.evaluations.driving_behavior.criticality.Criticality`).
- ``rule_compliance``: whether the ego vehicle drives as the traffic rules
  allow, checked against the Lanelet2 map
  (:mod:`~autonomy_evaluation.evaluations.driving_behavior.rule_compliance.RuleCompliance`).
- ``progress``: whether the ego vehicle gets ahead, along its route
  (:mod:`~autonomy_evaluation.evaluations.driving_behavior.progress.Progress`).
- ``comfort``: how smoothly the ego vehicle drives
  (:mod:`~autonomy_evaluation.evaluations.driving_behavior.comfort.Comfort`).

All of them read the ``ego_data`` of the ego vehicle (evaluated) and share the
states derived from it (see
:mod:`~autonomy_evaluation.evaluations.driving_behavior.states`). Usually, a
recorded simulation run is replayed by ``autonomy_datasets`` with the dataset as
sample source, so that the samples are attributed to the scenes of the
recordings; with an external sample source, the run ends when its publisher
calls the ``~/finish`` service of the node. As they evaluate the same run of the
same ego vehicle, the evaluations of the area usually run together in one node,
selected by the name of the area.

Every metric over time is computed per contiguous sequence of samples, see
:mod:`~autonomy_evaluation.evaluations.sampling`.
"""

from __future__ import annotations

from typing import Optional, Tuple

from autonomy_evaluation.evaluations.driving_behavior.states import ego_state
from autonomy_evaluation.evaluations.Evaluation import Evaluation
from autonomy_evaluation.evaluations.Metric import Sample


class DrivingBehaviorEvaluation(Evaluation):
    """Base class of the evaluations of the driving behavior of the ego vehicle, see the module docstring."""

    AREA = "driving_behavior"

    def sample_position(self, sample: Sample) -> Optional[Tuple[float, float]]:
        """Report the position of the ego vehicle, so that a teleport, e.g. to the start of a scenario, splits the sequence."""
        ego = ego_state(sample)
        return ego.x, ego.y
