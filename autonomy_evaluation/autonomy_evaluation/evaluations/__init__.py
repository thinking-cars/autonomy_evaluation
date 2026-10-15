# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Evaluations of automated driving tasks, each composed of the metrics of one task."""

from autonomy_evaluation.evaluations.Evaluation import Evaluation
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
from autonomy_evaluation.evaluations.registry import EVALUATIONS, load_evaluation

__all__ = [
    "Aggregation",
    "CONTEXT",
    "EVALUATED",
    "Evaluation",
    "EVALUATIONS",
    "Input",
    "load_evaluation",
    "Metric",
    "MetricGroup",
    "REFERENCE",
    "ROLES",
    "Sample",
]
