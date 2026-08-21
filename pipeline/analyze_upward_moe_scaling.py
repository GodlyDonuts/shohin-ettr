#!/usr/bin/env python3
"""Normalize completed Shohin MoE screens and fit the upward scaling curve."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

QWEN_SCHEMA = "shohin-q36-mtr-multi-trajectory-screen-result-v1"
QWEN_REVISION_SCHEMA = "shohin-q36-mtr-external-screen-result-summary-v1"
QWEN_EXTERNAL_SCORE_SCHEMA = "shohin-q36-mtr-external-score-v1"
MATCHED_SCHEMAS = frozenset(
    {
        "shohin-nemotron-super-fixed-draft-screen-score-v1",
        "shohin-mixtral-8x22b-fixed-draft-screen-score-v1",
        "shohin-nemotron-ultra-fixed-draft-screen-score-v1",
        "shohin-gpt-oss-120b-fixed-draft-screen-score-v1",
    }
)
REPORT_SCHEMA = "shohin-upward-moe-scaling-analysis-v1"
DOMAINS = ("bbh_logic", "math500", "mbpp")
Z95 = 1.959963984540054


class UpwardMoEScalingError(RuntimeError):
    """A completed MoE point was not comparable with the upward curve."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise UpwardMoEScalingError(f"{label} differs")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UpwardMoEScalingError(f"{label} differs")
    result = float(value)
    if not math.isfinite(result):
        raise UpwardMoEScalingError(f"{label} differs")
    return result


def _billions(value: Any, label: str) -> int:
    if isinstance(value, str) and value.endswith("B"):
        try:
            parsed = float(value[:-1]) * 1_000_000_000
        except ValueError as error:
            raise UpwardMoEScalingError(f"{label} differs") from error
        if not parsed.is_integer():
            raise UpwardMoEScalingError(f"{label} differs")
        value = int(parsed)
    return _integer(value, label, 1)


def _domain_counts(
    treatment: dict[str, Any], unchanged: dict[str, Any], totals: dict[str, Any]
) -> dict[str, dict[str, int]]:
    if set(totals) != set(DOMAINS):
        raise UpwardMoEScalingError("domain geometry differs")
    output: dict[str, dict[str, int]] = {}
    for domain in DOMAINS:
        total = _integer(totals[domain], f"{domain}.total", 1)
        treatment_correct = _integer(
            treatment.get(domain), f"{domain}.treatment_correct"
        )
        unchanged_correct = _integer(
            unchanged.get(domain), f"{domain}.unchanged_correct"
        )
        if treatment_correct > total or unchanged_correct > total:
            raise UpwardMoEScalingError("domain correct count exceeds total")
        output[domain] = {
            "treatment_correct": treatment_correct,
            "unchanged_correct": unchanged_correct,
            "total": total,
            "delta_correct": treatment_correct - unchanged_correct,
        }
    return output


def _normalize_qwen(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("status") != "complete_promoted":
        raise UpwardMoEScalingError("Qwen point is not complete")
    host = payload.get("host")
    screen = payload.get("source_disjoint_screen")
    result = payload.get("result")
    controls = payload.get("matched_controls")
    if not all(isinstance(value, dict) for value in (host, screen, result, controls)):
        raise UpwardMoEScalingError("Qwen point structure differs")
    treatment = result.get("multi_trajectory")
    unchanged = result.get("unchanged")
    self_refinement = controls.get("self_refinement")
    if not all(
        isinstance(value, dict) for value in (treatment, unchanged, self_refinement)
    ):
        raise UpwardMoEScalingError("Qwen arms differ")
    rows = _integer(screen.get("rows"), "rows", 1)
    treatment_correct = _integer(treatment.get("correct"), "treatment.correct")
    unchanged_correct = _integer(unchanged.get("correct"), "unchanged.correct")
    self_correct = _integer(self_refinement.get("correct"), "self.correct")
    delta = treatment_correct - unchanged_correct
    if (
        _integer(result.get("absolute_gain_correct"), "gain") != delta
        or _integer(result.get("paired_multi_only_correct"), "paired wins")
        - _integer(result.get("paired_unchanged_only_correct"), "paired losses")
        != delta
    ):
        raise UpwardMoEScalingError("Qwen paired delta differs")
    domains = _domain_counts(
        treatment.get("domains", {}),
        unchanged.get("domains", {}),
        screen.get("domain_rows", {}),
    )
    if (
        sum(domain["total"] for domain in domains.values()) != rows
        or sum(domain["treatment_correct"] for domain in domains.values())
        != treatment_correct
        or sum(domain["unchanged_correct"] for domain in domains.values())
        != unchanged_correct
    ):
        raise UpwardMoEScalingError("Qwen domain accounting differs")
    retained = _integer(result.get("unchanged_correct_retained"), "retained")
    retention = _finite(result.get("unchanged_correct_retention"), "retention")
    if unchanged_correct == 0 or retained / unchanged_correct != retention:
        raise UpwardMoEScalingError("Qwen retention differs")
    return {
        "host": host.get("model"),
        "architecture_series": "multi_trajectory_best_system",
        "total_parameters": _billions(host.get("total_parameters"), "total parameters"),
        "active_parameters": _billions(
            host.get("active_parameters"), "active parameters"
        ),
        "rows": rows,
        "treatment_arm": "multi_trajectory",
        "treatment_correct": treatment_correct,
        "unchanged_correct": unchanged_correct,
        "self_refinement_correct": self_correct,
        "gain_over_unchanged_count": delta,
        "gain_over_unchanged_percentage_points": 100.0 * delta / rows,
        "gain_over_self_refinement_count": treatment_correct - self_correct,
        "paired_wins": result["paired_multi_only_correct"],
        "paired_losses": result["paired_unchanged_only_correct"],
        "mcnemar_exact_two_sided_p": _finite(
            result.get("mcnemar_exact_two_sided_p"), "p value"
        ),
        "unchanged_correct_retained": retained,
        "unchanged_correct_retention": retention,
        "domains": domains,
    }


def _normalize_qwen_revision(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("status") != "complete" or payload.get("rows") != 256:
        raise UpwardMoEScalingError("Qwen revision point is not complete")
    arms = payload.get("arms")
    if not isinstance(arms, dict):
        raise UpwardMoEScalingError("Qwen revision arms differ")
    required = {"unchanged", "self_refinement", "revision"}
    if not required.issubset(arms) or not all(
        isinstance(arms[arm], dict) for arm in required
    ):
        raise UpwardMoEScalingError("Qwen revision arms differ")
    correct = {
        arm: _integer(arms[arm].get("correct"), f"{arm}.correct") for arm in required
    }
    rows = 256
    delta = correct["revision"] - correct["unchanged"]
    wins = _integer(arms["revision"].get("arm_only_correct"), "paired wins")
    losses = _integer(arms["revision"].get("unchanged_only_correct"), "paired losses")
    if (
        _integer(arms["revision"].get("gain_over_unchanged_count"), "gain") != delta
        or wins - losses != delta
    ):
        raise UpwardMoEScalingError("Qwen revision paired delta differs")
    totals: dict[str, int] = {}
    treatment_domains: dict[str, int] = {}
    unchanged_domains: dict[str, int] = {}
    self_domains: dict[str, int] = {}
    for domain in DOMAINS:
        for arm, destination in (
            ("revision", treatment_domains),
            ("unchanged", unchanged_domains),
            ("self_refinement", self_domains),
        ):
            value = arms[arm].get("domains", {}).get(domain)
            if (
                not isinstance(value, list)
                or len(value) != 2
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in value
                )
            ):
                raise UpwardMoEScalingError("Qwen revision domain differs")
            destination[domain] = _integer(value[0], f"{arm}.{domain}.correct")
            total = _integer(value[1], f"{arm}.{domain}.total", 1)
            if domain in totals and totals[domain] != total:
                raise UpwardMoEScalingError("Qwen revision domain totals differ")
            totals[domain] = total
    domains = _domain_counts(treatment_domains, unchanged_domains, totals)
    if (
        sum(totals.values()) != rows
        or sum(treatment_domains.values()) != correct["revision"]
        or sum(unchanged_domains.values()) != correct["unchanged"]
        or sum(self_domains.values()) != correct["self_refinement"]
    ):
        raise UpwardMoEScalingError("Qwen revision domain accounting differs")
    retained = _integer(arms["revision"].get("unchanged_correct_retained"), "retained")
    retention = _finite(
        arms["revision"].get("unchanged_correct_retention"), "retention"
    )
    if correct["unchanged"] == 0 or retained / correct["unchanged"] != retention:
        raise UpwardMoEScalingError("Qwen revision retention differs")
    return {
        "host": payload.get("host_model"),
        "architecture_series": "trained_revision",
        "total_parameters": 35_000_000_000,
        "active_parameters": 3_000_000_000,
        "rows": rows,
        "treatment_arm": "revision",
        "treatment_correct": correct["revision"],
        "unchanged_correct": correct["unchanged"],
        "self_refinement_correct": correct["self_refinement"],
        "gain_over_unchanged_count": delta,
        "gain_over_unchanged_percentage_points": 100.0 * delta / rows,
        "gain_over_self_refinement_count": (
            correct["revision"] - correct["self_refinement"]
        ),
        "paired_wins": wins,
        "paired_losses": losses,
        "mcnemar_exact_two_sided_p": _finite(
            arms["revision"].get("mcnemar_exact_two_sided_p"), "p value"
        ),
        "unchanged_correct_retained": retained,
        "unchanged_correct_retention": retention,
        "domains": domains,
    }


def _normalize_qwen_external_score(payload: dict[str, Any]) -> dict[str, Any]:
    rows = _integer(payload.get("rows"), "rows", 1)
    if payload.get("status") != "complete" or rows != 256:
        raise UpwardMoEScalingError("Qwen external score is not complete")
    arms = payload.get("arms")
    outcomes = payload.get("outcomes")
    required = {"unchanged", "self_refinement", "revision"}
    if (
        not isinstance(arms, dict)
        or not required.issubset(arms)
        or not all(isinstance(arms[arm], dict) for arm in required)
        or not isinstance(outcomes, list)
        or len(outcomes) != rows
    ):
        raise UpwardMoEScalingError("Qwen external score arms differ")
    identities: set[str] = set()
    outcome_correct: dict[str, int] = {arm: 0 for arm in required}
    retained = 0
    wins = 0
    losses = 0
    outcome_domains = {arm: {domain: 0 for domain in DOMAINS} for arm in required}
    domain_totals = {domain: 0 for domain in DOMAINS}
    for row in outcomes:
        if not isinstance(row, dict):
            raise UpwardMoEScalingError("Qwen external outcome differs")
        identity = row.get("identity_sha256")
        task = row.get("task")
        correct = row.get("correct")
        if (
            not isinstance(identity, str)
            or len(identity) != 64
            or any(value not in "0123456789abcdef" for value in identity)
            or identity in identities
            or task not in DOMAINS
            or not isinstance(correct, dict)
            or not all(isinstance(correct.get(arm), bool) for arm in required)
        ):
            raise UpwardMoEScalingError("Qwen external outcome differs")
        identities.add(identity)
        domain_totals[task] += 1
        for arm in required:
            if correct[arm]:
                outcome_correct[arm] += 1
                outcome_domains[arm][task] += 1
        retained += int(correct["unchanged"] and correct["revision"])
        wins += int(correct["revision"] and not correct["unchanged"])
        losses += int(correct["unchanged"] and not correct["revision"])

    arm_correct = {
        arm: _integer(arms[arm].get("correct"), f"{arm}.correct") for arm in required
    }
    if arm_correct != outcome_correct:
        raise UpwardMoEScalingError("Qwen external outcome totals differ")
    for arm in required:
        raw_domains = arms[arm].get("domains")
        if not isinstance(raw_domains, dict) or set(raw_domains) != set(DOMAINS):
            raise UpwardMoEScalingError("Qwen external domains differ")
        for domain in DOMAINS:
            value = raw_domains[domain]
            if (
                not isinstance(value, dict)
                or _integer(value.get("correct"), f"{arm}.{domain}.correct")
                != outcome_domains[arm][domain]
                or _integer(value.get("total"), f"{arm}.{domain}.total", 1)
                != domain_totals[domain]
            ):
                raise UpwardMoEScalingError("Qwen external domains differ")
    revision_pair = arms["revision"].get("paired_vs_unchanged")
    delta = arm_correct["revision"] - arm_correct["unchanged"]
    if (
        not isinstance(revision_pair, dict)
        or _integer(revision_pair.get("arm_only_correct"), "paired wins") != wins
        or _integer(revision_pair.get("unchanged_only_correct"), "paired losses")
        != losses
        or _integer(arms["revision"].get("gain_over_unchanged_count"), "gain") != delta
        or wins - losses != delta
        or arm_correct["unchanged"] == 0
    ):
        raise UpwardMoEScalingError("Qwen external paired delta differs")
    domains = _domain_counts(
        outcome_domains["revision"], outcome_domains["unchanged"], domain_totals
    )
    return {
        "host": "Qwen3.6-35B-A3B",
        "architecture_series": "trained_revision",
        "total_parameters": 35_000_000_000,
        "active_parameters": 3_000_000_000,
        "rows": rows,
        "treatment_arm": "revision",
        "treatment_correct": arm_correct["revision"],
        "unchanged_correct": arm_correct["unchanged"],
        "self_refinement_correct": arm_correct["self_refinement"],
        "gain_over_unchanged_count": delta,
        "gain_over_unchanged_percentage_points": 100.0 * delta / rows,
        "gain_over_self_refinement_count": (
            arm_correct["revision"] - arm_correct["self_refinement"]
        ),
        "paired_wins": wins,
        "paired_losses": losses,
        "mcnemar_exact_two_sided_p": _finite(
            revision_pair.get("mcnemar_exact_two_sided_p"), "p value"
        ),
        "unchanged_correct_retained": retained,
        "unchanged_correct_retention": retained / arm_correct["unchanged"],
        "domains": domains,
    }


def _normalize_matched(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("status") != "complete":
        raise UpwardMoEScalingError("matched point is not complete")
    arms = payload.get("arms")
    paired = payload.get("revision_vs_unchanged")
    if not isinstance(arms, dict) or not isinstance(paired, dict):
        raise UpwardMoEScalingError("matched point structure differs")
    if set(arms) != {"unchanged", "self_refinement", "revision"}:
        raise UpwardMoEScalingError("matched arms differ")
    rows = _integer(payload.get("rows"), "rows", 1)
    correct = {
        arm: _integer(arms[arm].get("correct"), f"{arm}.correct") for arm in arms
    }
    if any(value > rows for value in correct.values()):
        raise UpwardMoEScalingError("arm correct count exceeds rows")
    delta = correct["revision"] - correct["unchanged"]
    wins = _integer(paired.get("left_only_correct"), "paired wins")
    losses = _integer(paired.get("right_only_correct"), "paired losses")
    if (
        wins - losses != delta
        or _integer(paired.get("net_correct"), "paired net", -rows) != delta
    ):
        raise UpwardMoEScalingError("matched paired delta differs")
    totals: dict[str, int] = {}
    treatment_domains: dict[str, int] = {}
    unchanged_domains: dict[str, int] = {}
    self_domains: dict[str, int] = {}
    for domain in DOMAINS:
        revision_domain = arms["revision"].get("domains", {}).get(domain)
        unchanged_domain = arms["unchanged"].get("domains", {}).get(domain)
        if not isinstance(revision_domain, dict) or not isinstance(
            unchanged_domain, dict
        ):
            raise UpwardMoEScalingError("matched domain structure differs")
        self_domain = arms["self_refinement"].get("domains", {}).get(domain)
        if not isinstance(self_domain, dict):
            raise UpwardMoEScalingError("matched self-refinement domain differs")
        totals[domain] = _integer(revision_domain.get("total"), f"{domain}.total", 1)
        if (
            _integer(unchanged_domain.get("total"), f"{domain}.unchanged_total", 1)
            != totals[domain]
        ):
            raise UpwardMoEScalingError("matched domain totals differ")
        treatment_domains[domain] = _integer(
            revision_domain.get("correct"), f"{domain}.revision_correct"
        )
        unchanged_domains[domain] = _integer(
            unchanged_domain.get("correct"), f"{domain}.unchanged_correct"
        )
        if (
            _integer(self_domain.get("total"), f"{domain}.self_total", 1)
            != totals[domain]
        ):
            raise UpwardMoEScalingError("matched self-refinement domain total differs")
        self_domains[domain] = _integer(
            self_domain.get("correct"), f"{domain}.self_correct"
        )
    domains = _domain_counts(treatment_domains, unchanged_domains, totals)
    if (
        sum(totals.values()) != rows
        or sum(treatment_domains.values()) != correct["revision"]
        or sum(unchanged_domains.values()) != correct["unchanged"]
        or sum(self_domains.values()) != correct["self_refinement"]
    ):
        raise UpwardMoEScalingError("matched domain accounting differs")
    retained = _integer(arms["revision"].get("unchanged_correct_retained"), "retained")
    retention = _finite(
        arms["revision"].get("unchanged_correct_retention"), "retention"
    )
    if correct["unchanged"] == 0 or retained / correct["unchanged"] != retention:
        raise UpwardMoEScalingError("matched retention differs")
    return {
        "host": payload.get("host"),
        "architecture_series": "trained_revision",
        "total_parameters": _billions(
            payload.get("total_parameters"), "total parameters"
        ),
        "active_parameters": _billions(
            payload.get("active_parameters"), "active parameters"
        ),
        "rows": rows,
        "treatment_arm": "revision",
        "treatment_correct": correct["revision"],
        "unchanged_correct": correct["unchanged"],
        "self_refinement_correct": correct["self_refinement"],
        "gain_over_unchanged_count": delta,
        "gain_over_unchanged_percentage_points": 100.0 * delta / rows,
        "gain_over_self_refinement_count": (
            correct["revision"] - correct["self_refinement"]
        ),
        "paired_wins": wins,
        "paired_losses": losses,
        "mcnemar_exact_two_sided_p": _finite(
            paired.get("mcnemar_exact_two_sided_p"), "p value"
        ),
        "unchanged_correct_retained": retained,
        "unchanged_correct_retention": retention,
        "domains": domains,
    }


def normalize_point(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise UpwardMoEScalingError("point path is not a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UpwardMoEScalingError("point is unreadable") from error
    if not isinstance(payload, dict):
        raise UpwardMoEScalingError("point payload differs")
    schema = payload.get("schema")
    if schema == QWEN_SCHEMA:
        point = _normalize_qwen(payload)
    elif schema == QWEN_REVISION_SCHEMA:
        point = _normalize_qwen_revision(payload)
    elif schema == QWEN_EXTERNAL_SCORE_SCHEMA:
        point = _normalize_qwen_external_score(payload)
    elif schema in MATCHED_SCHEMAS:
        point = _normalize_matched(payload)
    else:
        raise UpwardMoEScalingError("point schema is not an admitted MoE screen")
    if not isinstance(point["host"], str) or not point["host"]:
        raise UpwardMoEScalingError("host differs")
    point["source_path"] = str(path.resolve())
    point["source_sha256"] = sha256_file(path)
    return point


def _ols(points: list[dict[str, Any]], parameter_field: str) -> dict[str, float]:
    xs = [math.log10(point[parameter_field]) for point in points]
    ys = [point["gain_over_unchanged_percentage_points"] for point in points]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator == 0:
        raise UpwardMoEScalingError("parameter scale is not distinct")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    intercept = y_mean - slope * x_mean
    predicted = [intercept + slope * x for x in xs]
    residual = sum((y - estimate) ** 2 for y, estimate in zip(ys, predicted))
    total = sum((y - y_mean) ** 2 for y in ys)
    return {
        "slope_percentage_points_per_log10_parameter": slope,
        "intercept_percentage_points": intercept,
        "r_squared": 1.0 - residual / total if total else 1.0,
    }


def _paired_gain_sampling(point: dict[str, Any]) -> dict[str, Any]:
    rows = _integer(point.get("rows"), "rows", 2)
    wins = _integer(point.get("paired_wins"), "paired wins")
    losses = _integer(point.get("paired_losses"), "paired losses")
    treatment = _integer(point.get("treatment_correct"), "treatment correct")
    unchanged = _integer(point.get("unchanged_correct"), "unchanged correct")
    if (
        wins + losses > rows
        or wins - losses != treatment - unchanged
        or treatment > rows
        or unchanged > rows
    ):
        raise UpwardMoEScalingError("paired sampling accounting differs")
    mean = (wins - losses) / rows
    sum_squares = wins + losses
    # A perfectly agreeing finite screen has zero empirical discordance but
    # must not receive infinite weight in a cross-host fit. One conservative
    # effective discordant row supplies a deterministic finite variance floor.
    effective_sum_squares = max(sum_squares, 1)
    sample_variance = (effective_sum_squares - rows * mean * mean) / (rows - 1)
    if sample_variance <= 0.0:
        raise UpwardMoEScalingError("paired sampling variance differs")
    standard_error = 100.0 * math.sqrt(sample_variance / rows)
    mean_percentage_points = 100.0 * mean
    return {
        "model": "paired_outcome_normal_approximation",
        "gain_percentage_points": mean_percentage_points,
        "standard_error_percentage_points": standard_error,
        "observed_discordant_rows": sum_squares,
        "variance_floor_discordant_rows": effective_sum_squares,
        "ci95_percentage_points": [
            mean_percentage_points - Z95 * standard_error,
            mean_percentage_points + Z95 * standard_error,
        ],
    }


def _sampling_weighted_ols(
    points: list[dict[str, Any]], parameter_field: str
) -> dict[str, Any]:
    xs = [math.log10(point[parameter_field]) for point in points]
    ys = [point["gain_over_unchanged_percentage_points"] for point in points]
    standard_errors = [
        point["paired_gain_sampling"]["standard_error_percentage_points"]
        for point in points
    ]
    weights = [
        1.0 / (standard_error * standard_error) for standard_error in standard_errors
    ]
    total_weight = sum(weights)
    x_mean = sum(weight * x for weight, x in zip(weights, xs)) / total_weight
    y_mean = sum(weight * y for weight, y in zip(weights, ys)) / total_weight
    denominator = sum(weight * (x - x_mean) ** 2 for weight, x in zip(weights, xs))
    if denominator <= 0.0:
        raise UpwardMoEScalingError("weighted parameter scale is not distinct")
    slope = (
        sum(
            weight * (x - x_mean) * (y - y_mean)
            for weight, x, y in zip(weights, xs, ys)
        )
        / denominator
    )
    intercept = y_mean - slope * x_mean
    slope_standard_error = math.sqrt(1.0 / denominator)
    predicted = [intercept + slope * x for x in xs]
    residual = sum(
        weight * (y - estimate) ** 2
        for weight, y, estimate in zip(weights, ys, predicted)
    )
    total = sum(weight * (y - y_mean) ** 2 for weight, y in zip(weights, ys))
    lower = slope - Z95 * slope_standard_error
    upper = slope + Z95 * slope_standard_error
    return {
        "sampling_model": "independent_marginal_paired_outcome_normal_approximation",
        "slope_percentage_points_per_log10_parameter": slope,
        "slope_standard_error_percentage_points": slope_standard_error,
        "slope_ci95_percentage_points": [lower, upper],
        "intercept_percentage_points": intercept,
        "weighted_r_squared": 1.0 - residual / total if total else 1.0,
        "positive_slope_ci95_lower_bound_above_zero": lower > 0.0,
        "cross_host_identity_covariance_modeled": False,
    }


def analyze(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise UpwardMoEScalingError("at least one completed point is required")
    points = sorted(
        (normalize_point(path) for path in paths),
        key=lambda point: point["active_parameters"],
    )
    if len({point["host"] for point in points}) != len(points):
        raise UpwardMoEScalingError("host is duplicated")
    series = {point["architecture_series"] for point in points}
    if len(series) != 1:
        raise UpwardMoEScalingError("architecture series is not comparable")
    geometry = {
        (
            point["rows"],
            tuple((domain, point["domains"][domain]["total"]) for domain in DOMAINS),
        )
        for point in points
    }
    if len(geometry) != 1:
        raise UpwardMoEScalingError("screen geometry is not comparable")
    distinct_total = len({point["total_parameters"] for point in points})
    distinct_active = len({point["active_parameters"] for point in points})
    curve_ready = len(points) >= 3 and distinct_total >= 3 and distinct_active >= 3
    for point in points:
        point["paired_gain_sampling"] = _paired_gain_sampling(point)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "complete_curve" if curve_ready else "complete_insufficient_points",
        "point_count": len(points),
        "minimum_points_for_curve": 3,
        "architecture_series": next(iter(series)),
        "points": points,
        "all_points_positive_vs_unchanged": all(
            point["gain_over_unchanged_count"] > 0 for point in points
        ),
        "all_points_positive_vs_self_refinement": all(
            point["gain_over_self_refinement_count"] > 0 for point in points
        ),
        "all_points_retention_at_least_95_percent": all(
            point["unchanged_correct_retention"] >= 0.95 for point in points
        ),
        "all_domains_nonnegative_at_every_point": all(
            domain["delta_correct"] >= 0
            for point in points
            for domain in point["domains"].values()
        ),
        "curve": None,
        "capability_curve_claim": "insufficient_completed_moe_points",
        "conservative_retention_curve_claim": "insufficient_completed_moe_points",
        "claim": "insufficient_completed_moe_points_for_scaling_curve",
    }
    if curve_ready:
        total_fit = _ols(points, "total_parameters")
        active_fit = _ols(points, "active_parameters")
        monotonic_gain = all(
            left["gain_over_unchanged_percentage_points"]
            <= right["gain_over_unchanged_percentage_points"]
            for left, right in zip(points, points[1:])
        )
        report["curve"] = {
            "total_parameter_fit": total_fit,
            "active_parameter_fit": active_fit,
            "paired_sampling_weighted_total_parameter_fit": (
                _sampling_weighted_ols(points, "total_parameters")
            ),
            "paired_sampling_weighted_active_parameter_fit": (
                _sampling_weighted_ols(points, "active_parameters")
            ),
            "gain_monotonic_by_active_parameters": monotonic_gain,
        }
        capability_supported = (
            report["all_points_positive_vs_unchanged"]
            and report["all_points_positive_vs_self_refinement"]
            and report["all_domains_nonnegative_at_every_point"]
            and active_fit["slope_percentage_points_per_log10_parameter"] > 0
        )
        conservative_supported = (
            capability_supported and report["all_points_retention_at_least_95_percent"]
        )
        report["capability_curve_claim"] = (
            "positive_upward_cross_family_moe_capability_scaling_supported"
            if capability_supported
            else "positive_upward_moe_capability_scaling_not_supported"
        )
        report["conservative_retention_curve_claim"] = (
            "positive_upward_cross_family_moe_scaling_with_retention_supported"
            if conservative_supported
            else "conservative_retention_scaling_not_supported"
        )
        if conservative_supported:
            report["claim"] = "positive_upward_cross_family_moe_scaling_supported"
        elif capability_supported:
            report["claim"] = (
                "positive_moe_capability_scaling_with_conservative_retention_not_supported"
            )
        else:
            report["claim"] = "moe_transfer_measured_without_positive_scaling_law"
    return report


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise UpwardMoEScalingError("output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    analysis = analyze(arguments.point)
    atomic_json(arguments.output, analysis)
    print(json.dumps({"status": analysis["status"], "claim": analysis["claim"]}))
