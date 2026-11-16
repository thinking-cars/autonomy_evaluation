# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Contiguous sequences of the recorded samples and the gaps between them.

Most metrics treat every sample on its own, e.g. the average precision of a
detector. Others depend on the order of the samples and the time between them,
e.g. the jerk of the ego vehicle or the time between two road users passing the
same spot. Such metrics are computed on contiguous sequences of samples, which
must neither span two scenes of a dataset nor bridge a long interruption.

Samples are recorded in the order they are evaluated, i.e. once all their
messages have arrived. A sample can thus be recorded after a later one, e.g.
when its messages are matched within a tolerance and one arrives late, so the
samples of a scene are put in stamp order, unless a stamp lies more than
``max_sample_interval`` seconds before the latest one recorded, as when a
dataset continues with a scene recorded before the scene played before it, or a
simulation restarts. Two consecutive samples in this order then belong to the
same sequence if

- they belong to the same scene, as far as their scenes are known,
- the later one is stamped after the earlier one,
- they are at most ``max_sample_interval`` seconds apart, and
- where the evaluation reports a position per sample, e.g. that of the ego
  vehicle, the position does not jump between them faster than
  ``max_jump_speed``, as when a simulation teleports the ego vehicle to the
  start of a scenario.

Within a sequence, an interval of more than :data:`GAP_FACTOR` times the
sampling period is a gap: samples are missing, e.g. because a system under test
skipped them or the evaluation did not keep up with a live system. Gaps are
reported with the results, as they make metrics computed over time less
accurate.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional

#: Factor by which an interval between two samples of a sequence must exceed the sampling period to be a gap.
GAP_FACTOR = 1.5


def split_into_sequences(
    entries: List[Dict[str, Any]], max_sample_interval: float, max_jump_speed: float = math.inf
) -> List[List[Dict[str, Any]]]:
    """Split recorded samples into contiguous sequences.

    Args:
        entries: Recorded samples, each holding its ``scene_id`` and ``stamp``
            (seconds, ``None`` if unknown) and, optionally, its ``position``
            ``(x, y)``, in the order they were recorded.
        max_sample_interval: Seconds between two consecutive samples beyond
            which they belong to different sequences.
        max_jump_speed: Speed in m/s beyond which the position jumps between
            two consecutive samples, which then belong to different sequences.

    Returns:
        The sequences in the order they were recorded, each a list of
        consecutive samples in stamp order. Samples without stamp continue the
        sequence of the previous sample of their scene, as no interval can be
        measured, and keep the samples around them in recording order.
    """

    sequences: List[List[Dict[str, Any]]] = []
    for run in _runs_in_stamp_order(entries, max_sample_interval):
        sequences.append([run[0]])
        for entry in run[1:]:
            if _continues(sequences[-1][-1], entry, max_sample_interval) and not _jumps(sequences[-1][-1], entry, max_jump_speed):
                sequences[-1].append(entry)
            else:
                sequences.append([entry])
    return sequences


def _runs_in_stamp_order(entries: List[Dict[str, Any]], max_sample_interval: float) -> List[List[Dict[str, Any]]]:
    """Group the samples recorded one after the other that may form sequences, and sort each group by stamp.

    A group ends where the scene changes or a stamp lies more than
    ``max_sample_interval`` before the latest stamp of the group, e.g. when a
    simulation restarts. Within a group, samples recorded out of order are put
    in stamp order, unless a sample has no stamp to order it by.
    """
    runs: List[List[Dict[str, Any]]] = []
    latest: Optional[float] = None
    for entry in entries:
        stamp = entry.get("stamp")
        restarts = stamp is not None and latest is not None and stamp < latest - max_sample_interval
        if runs and entry.get("scene_id") == runs[-1][-1].get("scene_id") and not restarts:
            runs[-1].append(entry)
        else:
            runs.append([entry])
            latest = None
        if stamp is not None:
            latest = stamp if latest is None else max(latest, stamp)
    return [_in_stamp_order(run) for run in runs]


def _in_stamp_order(run: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort samples by stamp, unless a sample has no stamp to order it by."""
    if any(entry.get("stamp") is None for entry in run):
        return run
    return sorted(run, key=lambda entry: entry["stamp"])


def _continues(previous: Dict[str, Any], entry: Dict[str, Any], max_sample_interval: float) -> bool:
    """Decide whether a sample continues the sequence of the sample recorded before it."""
    if previous.get("scene_id") != entry.get("scene_id"):
        return False
    if previous.get("stamp") is None or entry.get("stamp") is None:
        return True
    return 0.0 < entry["stamp"] - previous["stamp"] <= max_sample_interval


def _jumps(previous: Dict[str, Any], entry: Dict[str, Any], max_jump_speed: float) -> bool:
    """Decide whether the position jumps from a sample to the next one, faster than any road user drives."""
    if previous.get("position") is None or entry.get("position") is None:
        return False
    if previous.get("stamp") is None or entry.get("stamp") is None or entry["stamp"] <= previous["stamp"]:
        return False
    distance = math.hypot(entry["position"][0] - previous["position"][0], entry["position"][1] - previous["position"][1])
    return distance > max_jump_speed * (entry["stamp"] - previous["stamp"])


def intervals(sequence: List[Dict[str, Any]]) -> List[float]:
    """Return the seconds between the consecutive samples of a sequence whose stamps are known."""
    stamps = [entry["stamp"] for entry in sequence if entry.get("stamp") is not None]
    return [later - earlier for earlier, later in zip(stamps, stamps[1:])]


def sampling_period(sequence: List[Dict[str, Any]]) -> Optional[float]:
    """Return the sampling period of a sequence, as the median of the intervals between its samples.

    The median is robust against the gaps of a sequence, so it is the period at
    which its samples were meant to be published.

    Args:
        sequence: Consecutive samples of a sequence.

    Returns:
        The sampling period in seconds, ``None`` for a sequence of fewer than
        two samples with stamps.
    """
    sequence_intervals = intervals(sequence)
    return statistics.median(sequence_intervals) if sequence_intervals else None


def sampling_statistics(
    entries: List[Dict[str, Any]], max_sample_interval: float, max_jump_speed: float = math.inf
) -> Dict[str, Any]:
    """Describe how regularly the recorded samples were evaluated.

    Args:
        entries: Recorded samples, in the order they were recorded.
        max_sample_interval: Seconds between two consecutive samples beyond
            which they belong to different sequences.
        max_jump_speed: Speed in m/s beyond which the position jumps between
            two consecutive samples, which then belong to different sequences.

    Returns:
        ``{"num_sequences", "period", "num_gaps", "max_interval", "num_jumps"}``:
        the number of contiguous sequences, the median interval between the
        consecutive samples of all sequences in seconds, the number of gaps,
        i.e. intervals of more than :data:`GAP_FACTOR` times the sampling period
        of their sequence, the longest interval within a sequence in seconds,
        and the number of jumps of the position that split sequences; period
        and longest interval are ``None`` without any interval.
    """

    sequences = split_into_sequences(entries, max_sample_interval, max_jump_speed)
    all_intervals: List[float] = []
    num_gaps = 0
    for sequence in sequences:
        sequence_intervals = intervals(sequence)
        if sequence_intervals:
            period = statistics.median(sequence_intervals)
            num_gaps += sum(interval > GAP_FACTOR * period for interval in sequence_intervals)
            all_intervals += sequence_intervals
    return {
        "num_sequences": len(sequences),
        "period": statistics.median(all_intervals) if all_intervals else None,
        "num_gaps": num_gaps,
        "max_interval": max(all_intervals) if all_intervals else None,
        "num_jumps": sum(
            _continues(previous, entry, max_sample_interval) and _jumps(previous, entry, max_jump_speed)
            for previous, entry in zip(entries, entries[1:])
        ),
    }
