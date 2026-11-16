# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Reading the states of ``perception_msgs`` the way all evaluations of road users read them.

- Positions are those of the geometric centers: the reference point of a state
  is moved by its ``translation_to_geometric_center``, rotated by the yaw.
- An entry that ``perception_msgs`` marks as not set, i.e.
  ``CONTINUOUS_STATE_COVARIANCE_INVALID`` on the diagonal of the state
  covariance, is unavailable. A state without covariance has all its entries
  set.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple

from perception_msgs.msg import ObjectState
from perception_msgs_utils.checks import InvalidStateCovarianceSizeError
from perception_msgs_utils.constants import CONTINUOUS_STATE_COVARIANCE_INVALID
from perception_msgs_utils.convenience_state_getters import get_continuous_state_covariance_at
from perception_msgs_utils.state_index import UnknownStateEntryError


class PerceptionMsgsUtils:
    """Helpers reading positions and entries of ``perception_msgs`` states."""

    @staticmethod
    def rotated(along: float, across: float, yaw: float) -> Tuple[float, float]:
        """Rotate a vector given along and across a heading into the frame the heading is given in."""
        return along * math.cos(yaw) - across * math.sin(yaw), along * math.sin(yaw) + across * math.cos(yaw)

    @staticmethod
    def geometric_center(state: ObjectState, x: float, y: float, yaw: float) -> Tuple[float, float]:
        """Move the BEV position of the reference point of a state to the geometric center."""
        translation = state.reference_point.translation_to_geometric_center
        dx, dy = PerceptionMsgsUtils.rotated(translation.x, translation.y, yaw)
        return x + dx, y + dy

    @staticmethod
    def entry(obj: Any, getter: Callable[[Any], float], index: Callable[[int], int]) -> Optional[float]:
        """Read a state entry, ``None`` if the motion model has no such entry or ``perception_msgs`` marks it as not set.

        Args:
            obj: A message holding a state, e.g. ``perception_msgs/Object``, or an ``ObjectState`` itself.
            getter: ``perception_msgs_utils`` getter of the entry, e.g. ``get_vel_lon``.
            index: ``perception_msgs_utils`` index of the entry, e.g. ``index_vel_lon``.
        """
        try:
            value = float(getter(obj))
        except UnknownStateEntryError:
            return None
        state = obj if isinstance(obj, ObjectState) else obj.state
        try:
            position = index(state.model_id)
            if get_continuous_state_covariance_at(obj, position, position) == CONTINUOUS_STATE_COVARIANCE_INVALID:
                return None
        except InvalidStateCovarianceSizeError:
            pass  # a state without covariance has all its entries set
        return value
