#!/usr/bin/env python3
"""Select the strongest qualified larger MoE host for exact temporal transfer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from analyze_upward_moe_scaling import DOMAINS, normalize_point

REPORT_SCHEMA = "shohin-upward-moe-temporal-promotion-v1"
EXPECTED_HOSTS = {
    "NVIDIA-Nemotron-3-Super-120B-A12B-FP8": "nemotron-super",
    "mistralai/Mixtral-8x22B-Instruct-v0.1": "mixtral-8x22b",
}
MINIMUM_RETENTION = 0.95


class UpwardMoETemporalPromotionError(RuntimeError):
    """The completed larger-host evidence differed from the promotion contract."""


def _qualification(point: dict[str, Any]) -> dict[str, bool]:
    return {
        "positive_vs_unchanged": point["gain_over_unchanged_count"] > 0,
        "positive_vs_self_refinement": point["gain_over_self_refinement_count"] > 0,
        "all_domains_nonnegative_vs_unchanged": all(
            point["domains"][domain]["delta_correct"] >= 0 for domain in DOMAINS
        ),
        "retains_at_least_95_percent_unchanged_correct": (
            point["unchanged_correct_retention"] >= MINIMUM_RETENTION
        ),
    }


def _rank(point: dict[str, Any]) -> tuple[int, int, int, float, int, int, str]:
    """Rank capability first, then retention and scale, with a stable host tie-break."""
    return (
        point["gain_over_unchanged_count"],
        point["gain_over_self_refinement_count"],
        point["treatment_correct"],
        point["unchanged_correct_retention"],
        point["active_parameters"],
        point["total_parameters"],
        point["host"],
    )


def select(paths: list[Path]) -> dict[str, Any]:
    if len(paths) != len(EXPECTED_HOSTS):
        raise UpwardMoETemporalPromotionError(
            "exactly the Super and Mixtral score reports are required"
        )
    try:
        points = [normalize_point(path) for path in paths]
    except Exception as error:
        raise UpwardMoETemporalPromotionError(
            "larger-host score validation failed"
        ) from error
    by_host = {point["host"]: point for point in points}
    if len(by_host) != len(points) or set(by_host) != set(EXPECTED_HOSTS):
        raise UpwardMoETemporalPromotionError("larger-host identity differs")
    if any(point["architecture_series"] != "trained_revision" for point in points):
        raise UpwardMoETemporalPromotionError("architecture series differs")
    if len({point["rows"] for point in points}) != 1 or points[0]["rows"] != 256:
        raise UpwardMoETemporalPromotionError("screen geometry differs")
    domain_geometry = {
        tuple((domain, point["domains"][domain]["total"]) for domain in DOMAINS)
        for point in points
    }
    if len(domain_geometry) != 1:
        raise UpwardMoETemporalPromotionError("domain geometry differs")

    candidates: list[dict[str, Any]] = []
    for host in sorted(EXPECTED_HOSTS):
        point = by_host[host]
        checks = _qualification(point)
        candidates.append(
            {
                "host": host,
                "dispatcher_host": EXPECTED_HOSTS[host],
                "source_path": point["source_path"],
                "source_sha256": point["source_sha256"],
                "total_parameters": point["total_parameters"],
                "active_parameters": point["active_parameters"],
                "rows": point["rows"],
                "revision_correct": point["treatment_correct"],
                "unchanged_correct": point["unchanged_correct"],
                "self_refinement_correct": point["self_refinement_correct"],
                "gain_over_unchanged_count": point["gain_over_unchanged_count"],
                "gain_over_self_refinement_count": point[
                    "gain_over_self_refinement_count"
                ],
                "paired_wins": point["paired_wins"],
                "paired_losses": point["paired_losses"],
                "mcnemar_exact_two_sided_p": point["mcnemar_exact_two_sided_p"],
                "unchanged_correct_retention": point["unchanged_correct_retention"],
                "domain_deltas": {
                    domain: point["domains"][domain]["delta_correct"]
                    for domain in DOMAINS
                },
                "qualification_checks": checks,
                "qualifies": all(checks.values()),
            }
        )

    qualified_points = [
        point for point in points if all(_qualification(point).values())
    ]
    selected = max(qualified_points, key=_rank) if qualified_points else None
    return {
        "schema": REPORT_SCHEMA,
        "status": "promote" if selected is not None else "no_qualifying_larger_host",
        "contract": {
            "required_hosts": sorted(EXPECTED_HOSTS),
            "minimum_unchanged_correct_retention": MINIMUM_RETENTION,
            "required_conditions": [
                "positive_vs_unchanged",
                "positive_vs_self_refinement",
                "all_domains_nonnegative_vs_unchanged",
                "retains_at_least_95_percent_unchanged_correct",
            ],
            "ranking": [
                "gain_over_unchanged_count",
                "gain_over_self_refinement_count",
                "revision_correct",
                "unchanged_correct_retention",
                "active_parameters",
                "total_parameters",
                "host",
            ],
        },
        "candidates": candidates,
        "selected_host": selected["host"] if selected is not None else None,
        "selected_dispatcher_host": (
            EXPECTED_HOSTS[selected["host"]] if selected is not None else None
        ),
        "next_action": (
            "launch_exact_host_owned_temporal_transfer"
            if selected is not None
            else "preserve_both_results_and_do_not_launch_temporal_transfer"
        ),
        "automatic_launch": False,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise UpwardMoETemporalPromotionError("output exists")
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
    parser.add_argument("--score", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    decision = select(arguments.score)
    atomic_json(arguments.output, decision)
    print(
        json.dumps(
            {
                "status": decision["status"],
                "selected_dispatcher_host": decision["selected_dispatcher_host"],
            },
            sort_keys=True,
        )
    )
