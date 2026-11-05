# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Evaluation of the compliance of the ego vehicle with the traffic rules.

Whether the ego vehicle a system under test drives drives as the traffic rules
allow. Reads the ``ego_data`` of the ego vehicle (evaluated) and checks it
against the Lanelet2 ``map`` of the current scene (context), without which no
rule is checked. Metrics, selectable via ``metrics``:

- ``drivable_area``: whether the ego vehicle stays within the lanelets vehicles
  may drive in.
- ``driving_direction``: whether the ego vehicle drives in the direction of
  travel of its lane.
- ``speed_limit``: whether the ego vehicle keeps the speed limit of its lane.

See :mod:`~autonomy_evaluation.evaluations.driving_behavior.rule_compliance.RuleComplianceMetrics`.
"""

from __future__ import annotations

from typing import Any, List

from autonomy_evaluation.evaluations.driving_behavior.DrivingBehaviorEvaluation import DrivingBehaviorEvaluation
from autonomy_evaluation.evaluations.driving_behavior.rule_compliance.RuleComplianceMetrics import (
    DrivableArea,
    DrivingDirection,
    SpeedLimit,
)


class RuleCompliance(DrivingBehaviorEvaluation):
    """Evaluation of the compliance of the ego vehicle with the traffic rules, see the module docstring."""

    VERSION = "1.0.0"
    RELEASE_NOTES = {
        "1.0.0": "Initial implementation",
    }

    def __init__(self) -> None:
        """Name the evaluation."""
        super().__init__(
            name="rule_compliance",
            description="Compliance of the ego vehicle a system under test drives with the traffic rules: whether it "
            "stays on the road, drives in the direction of travel and keeps the speed limit.",
        )

    def metrics(self) -> List[Any]:
        """Compose the evaluation of the drivable area, the driving direction and the speed limit."""
        return [DrivableArea(), DrivingDirection(), SpeedLimit()]
