# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Optional limits on the metrics of an evaluation, which judge its results as passed or failed.

Metrics only measure, e.g. the smallest time to collision of the ego vehicle.
Whether a measured value is acceptable depends on the application, so limits
are configured for an evaluation, if at all, in its ``limits`` entry, one limit
per string ``"<metric> <operator> <bound>"``, e.g.::

    limits: ["criticality.ttc >= 1.5", "criticality.collision.at_fault == 0", "comfort.jerk.longitudinal.max_abs < 5"]

``<metric>`` is the path of a value in the aggregated metrics, its keys joined
by dots; a path to a metric with sub-metrics means its ``_value_``. The
operator is one of ``<``, ``<=``, ``>``, ``>=`` and ``==``, the bound a number.

The limits are checked on the metrics over all samples and on those of each
scene. A value that is ``null``, e.g. the smallest time to collision while no
road user was ever on collision course, or that the path does not lead to,
cannot be checked, and neither passes nor fails. The results pass if every
limit passes, fail if any limit fails, and are undecided otherwise. Without
limits, the results hold no verdict at all.
"""

from __future__ import annotations

import math
import operator
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: Operators a limit compares the value of its metric with its bound by.
OPERATORS: Dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
}

# Separator of the keys of a metric path
_SEPARATOR = "."

# Key of the aggregated value of a metric with sub-metrics
_VALUE_KEY = "_value_"

_LIMIT_PATTERN = re.compile(r"^\s*(?P<metric>[^\s<>=]+)\s*(?P<operator><=|>=|==|<|>)\s*(?P<bound>\S+)\s*$")


@dataclass(frozen=True)
class Limit:
    """A limit on one value of the aggregated metrics.

    Attributes:
        text: The limit as configured, e.g. ``"criticality.ttc >= 1.5"``.
        metric: Path of the value, e.g. ``"criticality.ttc"``.
        operator: Comparison of the value with the bound, one of :data:`OPERATORS`.
        bound: Bound the value is compared with.
    """

    text: str
    metric: str
    operator: str
    bound: float

    @classmethod
    def parse(cls, text: str) -> "Limit":
        """Read a limit from its configuration ``"<metric> <operator> <bound>"``.

        Raises:
            ValueError: The text is no such limit, or its bound is no finite number.
        """
        match = _LIMIT_PATTERN.match(text)
        if match is None:
            raise ValueError(f"Limit '{text}' is not of the form '<metric> <operator> <bound>', operators: {list(OPERATORS)}")
        try:
            bound = float(match["bound"])
        except ValueError:
            bound = math.nan
        if not math.isfinite(bound):
            raise ValueError(f"Limit '{text}' has no finite number as bound")
        return cls(text=text.strip(), metric=match["metric"], operator=match["operator"], bound=bound)

    def check(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Check the limit on aggregated metrics.

        Args:
            metrics: Aggregated metrics of an evaluation or of one of its scenes.

        Returns:
            ``{"limit", "value", "passed"}``: the limit as configured, the value
            of its metric and whether the value is within the limit, ``None`` if
            it cannot be checked; with an ``error`` naming why the path leads to
            no number, if it does not.
        """
        value, error = _resolve(metrics, self.metric)
        check: Dict[str, Any] = {"limit": self.text, "value": value, "passed": None}
        if error is not None:
            check["error"] = error
        elif value is not None:
            check["passed"] = OPERATORS[self.operator](value, self.bound)
        return check


def parse_limits(texts: Sequence[str]) -> List[Limit]:
    """Read the configured limits of an evaluation, see :meth:`Limit.parse`."""
    return [Limit.parse(text) for text in texts]


def check_limits(limits: Sequence[Limit], metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Check limits on aggregated metrics.

    Args:
        limits: Limits to check.
        metrics: Aggregated metrics of an evaluation or of one of its scenes.

    Returns:
        ``{"passed", "num_passed", "num_failed", "checks"}``: whether the
        metrics pass all limits, ``False`` if any limit fails and ``None`` if
        none fails but some cannot be checked; the numbers of passed and failed
        limits; and the check of each limit (see :meth:`Limit.check`).
    """
    checks = [limit.check(metrics) for limit in limits]
    verdicts = [check["passed"] for check in checks]
    passed = False if False in verdicts else (None if None in verdicts else True)
    return {"passed": passed, "num_passed": verdicts.count(True), "num_failed": verdicts.count(False), "checks": checks}


def _resolve(metrics: Any, path: str) -> Tuple[Optional[float], Optional[str]]:
    """Find the value a metric path leads to.

    Keys may contain the separator themselves, e.g. ``dist_0.5``, so each level
    takes the longest key the remaining path starts with.

    Returns:
        ``(value, error)``: the number the path leads to, ``None`` for ``null``;
        and why the path leads to no number, ``None`` if it does.
    """
    node, remaining = metrics, path
    while remaining:
        if not isinstance(node, dict):
            return None, f"'{path}' leads beyond a value"
        keys = [key for key in node if remaining == key or remaining.startswith(f"{key}{_SEPARATOR}")]
        if not keys:
            return None, f"'{path}' is no metric"
        key = max(keys, key=len)
        node, remaining = node[key], remaining[len(key) + len(_SEPARATOR) :]
    if isinstance(node, dict):
        node = node.get(_VALUE_KEY)
    if node is None:
        return None, None
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        return None, f"'{path}' is no number"
    return float(node), None
