# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Contiguous sequences of the recorded samples and the gaps between them.

Most metrics treat every sample on its own, e.g. the average precision of a
detector. Others depend on the order of the samples and the time between them,
e.g. the jerk of the ego vehicle or the time between two road users passing the
same spot. Such metrics are computed on contiguous sequences of samples, which
must neither span two scenes of a dataset nor bridge a long interruption.

Samples are recorded in the order they are evaluated, i.e. published. Two
consecutive samples belong to the same sequence if

- they belong to the same scene, as far as their scenes are known,
- the later one is stamped after the earlier one, as a dataset may continue
  with a scene that was recorded before the scene played before it, and
- they are at most ``max_sample_interval`` seconds apart.

Within a sequence, an interval of more than :data:`GAP_FACTOR` times the
sampling period is a gap: samples are missing, e.g. because a system under test
skipped them or the evaluation did not keep up with a live system. Gaps are
reported with the results, as they make metrics computed over time less
accurate.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

#: Factor by which an interval between two samples of a sequence must exceed the sampling period to be a gap.
GAP_FACTOR = 1.5


def split_into_sequences(entries: List[Dict[str, Any]], max_sample_interval: float) -> List[List[Dict[str, Any]]]:
    """Split recorded samples into contiguous sequences.

    Args:
        entries: Recorded samples, each holding its ``scene_id`` and ``stamp``
            (seconds, ``None`` if unknown), in the order they were recorded.
        max_sample_interval: Seconds between two consecutive samples beyond
            which they belong to different sequences.

    Returns:
        The sequences, each a list of consecutive samples, in recording order.
        Samples without stamp continue the sequence of the previous sample of
        their scene, as no interval can be measured.
    """

    sequences: List[List[Dict[str, Any]]] = []
    for entry in entries:
        if sequences and _continues(sequences[-1][-1], entry, max_sample_interval):
            sequences[-1].append(entry)
        else:
            sequences.append([entry])
    return sequences


def _continues(previous: Dict[str, Any], entry: Dict[str, Any], max_sample_interval: float) -> bool:
    """Decide whether a sample continues the sequence of the sample recorded before it."""
    if previous.get("scene_id") != entry.get("scene_id"):
        return False
    if previous.get("stamp") is None or entry.get("stamp") is None:
        return True
    return 0.0 < entry["stamp"] - previous["stamp"] <= max_sample_interval


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


def sampling_statistics(entries: List[Dict[str, Any]], max_sample_interval: float) -> Dict[str, Any]:
    """Describe how regularly the recorded samples were evaluated.

    Args:
        entries: Recorded samples, in the order they were recorded.
        max_sample_interval: Seconds between two consecutive samples beyond
            which they belong to different sequences.

    Returns:
        ``{"num_sequences", "period", "num_gaps", "max_interval"}``: the number
        of contiguous sequences, the median interval between the consecutive
        samples of all sequences in seconds, the number of gaps, i.e. intervals
        of more than :data:`GAP_FACTOR` times the sampling period of their
        sequence, and the longest interval within a sequence in seconds; period
        and longest interval are ``None`` without any interval.
    """

    sequences = split_into_sequences(entries, max_sample_interval)
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
    }
