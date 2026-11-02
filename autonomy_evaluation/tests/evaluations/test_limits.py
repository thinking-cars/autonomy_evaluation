# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

"""Tests for the optional limits that judge the aggregated metrics of an evaluation as passed or failed."""

from __future__ import annotations

import pytest
from autonomy_evaluation.evaluations.limits import check_limits, Limit, parse_limits

_METRICS = {
    "_value_": None,
    "criticality": {
        "ttc": {"_value_": 0.8, "share": 0.25},
        "collision": {"_value_": 1, "at_fault": 0, "types": {"active_front": 0}},
        "pet": {"_value_": None},
    },
    "ap": {"_value_": 0.6, "dist_0.5": {"_value_": 0.4}, "dist_1.0": {"_value_": 0.7}},
    "sources": ["state"],
}


class TestParse:
    """Tests reading limits from their configuration."""

    def test_reads_metric_operator_and_bound(self):
        """Spaces around the operator are optional."""
        assert Limit.parse(" criticality.ttc>=1.5 ") == Limit("criticality.ttc>=1.5", "criticality.ttc", ">=", 1.5)
        assert Limit.parse("ap.dist_0.5 < 1e-1").bound == pytest.approx(0.1)

    @pytest.mark.parametrize("text", ["criticality.ttc", "criticality.ttc => 1", "criticality.ttc >= fast", ">= 1", "ttc >= inf"])
    def test_rejects_malformed_limits(self, text):
        """A limit names a metric, an operator and a finite bound."""
        with pytest.raises(ValueError, match="Limit"):
            parse_limits([text])


class TestCheck:
    """Tests checking limits on aggregated metrics."""

    @pytest.mark.parametrize(
        "text, value, passed",
        [
            ("criticality.ttc >= 1.5", 0.8, False),  # a metric with sub-metrics means its aggregated value
            ("criticality.ttc._value_ < 1.5", 0.8, True),
            ("criticality.ttc.share <= 0.25", 0.25, True),
            ("criticality.collision.at_fault == 0", 0.0, True),
            ("criticality.collision.types.active_front > 0", 0.0, False),
            ("ap.dist_0.5 >= 0.5", 0.4, False),  # keys may contain dots themselves
            ("ap.dist_1.0 >= 0.5", 0.7, True),
        ],
    )
    def test_compares_the_value_of_the_metric_with_the_bound(self, text, value, passed):
        """Each limit compares one value of the metrics with its bound."""
        check = Limit.parse(text).check(_METRICS)

        assert check == {"limit": text, "value": pytest.approx(value), "passed": passed}

    def test_null_value_cannot_be_checked(self):
        """A value that could not be computed neither passes nor fails."""
        assert Limit.parse("criticality.pet >= 1.0").check(_METRICS) == {
            "limit": "criticality.pet >= 1.0",
            "value": None,
            "passed": None,
        }

    @pytest.mark.parametrize(
        "text, error",
        [
            ("criticality.thw >= 1.0", "is no metric"),
            ("criticality.ttc.share.max >= 1.0", "leads beyond a value"),
            ("sources == 1", "is no number"),
        ],
    )
    def test_path_to_no_number_cannot_be_checked(self, text, error):
        """A path that leads to no number is reported with the reason."""
        check = Limit.parse(text).check(_METRICS)

        assert check["passed"] is None
        assert error in check["error"]

    def test_metrics_fail_if_any_limit_fails(self):
        """One failed limit fails the metrics, whatever the others."""
        verdict = check_limits(parse_limits(["ap >= 0.5", "criticality.ttc >= 1.5", "criticality.pet >= 1"]), _METRICS)

        assert (verdict["passed"], verdict["num_passed"], verdict["num_failed"]) == (False, 1, 1)
        assert [check["passed"] for check in verdict["checks"]] == [True, False, None]

    def test_metrics_are_undecided_if_a_limit_cannot_be_checked(self):
        """Without any failed limit, a limit that cannot be checked leaves the verdict open."""
        assert check_limits(parse_limits(["ap >= 0.5", "criticality.pet >= 1"]), _METRICS)["passed"] is None

    def test_metrics_pass_if_all_limits_pass(self):
        """The metrics pass if every limit passes."""
        assert check_limits(parse_limits(["ap >= 0.5", "criticality.collision.at_fault == 0"]), _METRICS)["passed"] is True
