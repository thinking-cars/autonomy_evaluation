# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Progress metrics of a closed-loop evaluation: whether the ego vehicle gets ahead.

- ``distance``: distance the ego vehicle drives, its mean speed and how long it
  stands still, e.g. to tell an ego vehicle that drives safely from one that
  never starts.

Like the other metrics over time, the distance follows the positions of the
ego vehicle over each contiguous sequence of samples, and each sample counts
for the sampling period of its sequence.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from autonomy_evaluation.evaluations.closed_loop.ComfortMetrics import ego_inputs
from autonomy_evaluation.evaluations.closed_loop.states import ego_state
from autonomy_evaluation.evaluations.Metric import Aggregation, Input, Metric, Sample
from autonomy_evaluation.evaluations.sampling import sampling_period


class DistanceDriven(Metric):
    """Distance the ego vehicle drives, its mean speed, and how long it stands still.

    The distance is the length of the path through the positions of the ego
    vehicle over each contiguous sequence of samples, the mean speed that
    distance over the duration of the sequences.

    Configuration:
        ``standstill_speed``: speed in m/s below which the ego vehicle stands still.
    """

    NAME = "distance"

    def inputs(self) -> Dict[str, Input]:
        """Read the ego vehicle."""
        return ego_inputs()

    def default_config(self) -> Dict[str, Any]:
        """Take the ego vehicle as standing still below 0.1 m/s, like collisions do."""
        return {"standstill_speed": 0.1}

    def compute_sample(self, sample: Sample) -> Tuple[Optional[float], float, float, Optional[float]]:
        """Record the position and the speed of the ego vehicle at a sample.

        Returns:
            ``(stamp, x, y, speed)``, the speed ``None`` without velocity.
        """
        ego = ego_state(sample)
        return sample.stamp, ego.x, ego.y, ego.speed

    def aggregate(self, aggregation: Aggregation) -> Dict[str, Any]:
        """Report the distance driven, the mean speed and the time standing still.

        Returns:
            ``{"_value_", "mean_speed", "standstill_time", "standstill_share"}``:
            the distance driven in meters; the mean speed in m/s, ``None`` if no
            time passes; the time the ego vehicle stands still in seconds; and
            the share of the samples with velocity at which it stands still,
            ``None`` without such samples.
        """
        standstill_speed = self.config["standstill_speed"]
        distance, duration, standstill_time = 0.0, 0.0, 0.0
        num_with_speed, num_standing = 0, 0
        for sequence in aggregation.sequences():
            records = aggregation.records(self.name, sequence)
            period = sampling_period(sequence) or 0.0
            for (_, x, y, _), (_, next_x, next_y, _) in zip(records, records[1:]):
                distance += math.hypot(next_x - x, next_y - y)
            if records[0][0] is not None and records[-1][0] is not None:
                duration += records[-1][0] - records[0][0]
            for _, _, _, speed in records:
                if speed is not None:
                    num_with_speed += 1
                    if speed < standstill_speed:
                        num_standing += 1
                        standstill_time += period
        return {
            "_value_": distance,
            "mean_speed": distance / duration if duration > 0.0 else None,
            "standstill_time": standstill_time,
            "standstill_share": num_standing / num_with_speed if num_with_speed else None,
        }
