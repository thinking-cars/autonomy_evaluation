# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Evaluations of automated driving tasks, each composed of the metrics of one task."""

from autonomy_evaluation.evaluations.Evaluation import Evaluation, merged_inputs
from autonomy_evaluation.evaluations.Metric import (
    Aggregation,
    CONTEXT,
    EVALUATED,
    Input,
    Metric,
    MetricGroup,
    REFERENCE,
    ROLES,
    Sample,
)
from autonomy_evaluation.evaluations.registry import (
    AREA_EVALUATIONS,
    EVALUATIONS,
    load_evaluation,
    load_evaluations,
    resolve_selection,
)

__all__ = [
    "AREA_EVALUATIONS",
    "Aggregation",
    "CONTEXT",
    "EVALUATED",
    "Evaluation",
    "EVALUATIONS",
    "Input",
    "load_evaluation",
    "load_evaluations",
    "merged_inputs",
    "Metric",
    "MetricGroup",
    "REFERENCE",
    "resolve_selection",
    "ROLES",
    "Sample",
]
