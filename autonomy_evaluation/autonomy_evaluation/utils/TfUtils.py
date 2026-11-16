# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Planar transforms between the frames of messages that evaluations compare, looked up with tf2."""

from __future__ import annotations

import math
from typing import Any, Tuple

import numpy as np
from rclpy.time import Time
from tf2_ros import TransformException

#: A planar transform as ``(dx, dy, dyaw)``: the rotation about the vertical axis by ``dyaw``, then the translation.
PlanarTransform = Tuple[float, float, float]

#: The planar transform that keeps poses as they are.
IDENTITY: PlanarTransform = (0.0, 0.0, 0.0)


class TfUtils:
    """Helpers looking up and applying planar transforms."""

    @staticmethod
    def planar_transform(tf_buffer: Any, source: str, target: str, stamp: Any, what: str = "Messages") -> PlanarTransform:
        """Look up the planar transform from one frame into another.

        Args:
            tf_buffer: Transforms between the frames, ``None`` if none is available.
            source, target: Frames; an empty frame is taken as the other one.
            stamp: ``builtin_interfaces/Time`` to look the transform up at.
            what: What is transformed, for the error message, e.g. ``"The route"``.

        Returns:
            The transform that takes poses in ``source`` to ``target``, the
            identity for the same frame.

        Raises:
            ValueError: No transform between the frames is available at the stamp.
        """
        if not source or not target or source == target:
            return IDENTITY
        error = f"{what} in frame '{source}' cannot be transformed into frame '{target}'"
        if tf_buffer is None:
            raise ValueError(f"{error}, as no transforms are available")
        try:
            transform = tf_buffer.lookup_transform(target, source, Time.from_msg(stamp)).transform
        except TransformException as exception:
            raise ValueError(f"{error}: {exception}") from exception
        rotation = transform.rotation
        dyaw = math.atan2(2.0 * (rotation.w * rotation.z + rotation.x * rotation.y), 1.0 - 2.0 * (rotation.y**2 + rotation.z**2))
        return transform.translation.x, transform.translation.y, dyaw

    @staticmethod
    def transform_points(x: np.ndarray, y: np.ndarray, transform: PlanarTransform) -> Tuple[np.ndarray, np.ndarray]:
        """Apply a planar transform to positions."""
        dx, dy, dyaw = transform
        cos, sin = math.cos(dyaw), math.sin(dyaw)
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        return cos * x - sin * y + dx, sin * x + cos * y + dy
