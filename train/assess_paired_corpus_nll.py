#!/usr/bin/env python3
"""Assess a candidate checkpoint against baseline on identical NLL windows."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from eval_corpus_nll import REPORT_SCHEMA
from pipeline.tokenize_shards import (
    canonical_payload_sha256,
    file_receipt,
)


ASSESSMENT_SCHEMA = "shohin-paired-corpus-nll-assessment-v1"


class PairedNllError(ValueError):
    """The NLL reports do not form a valid paired comparison."""


def _report(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = file_receipt(path)
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedNllError(f"NLL report is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise PairedNllError("NLL report is not an object")
    claimed = value.get("payload_sha256")
    unsigned = dict(value)
    unsigned.pop("payload_sha256", None)
    if (
        value.get("schema") != REPORT_SCHEMA
        or not isinstance(claimed, str)
        or canonical_payload_sha256(unsigned) != claimed
    ):
        raise PairedNllError("NLL report payload differs")
    return value, receipt


def assess_paired_nll(
    *,
    baseline_path: Path,
    candidate_path: Path,
    output: Path,
    maximum_allowed_mean_regression: float = 0.0,
) -> dict[str, Any]:
    if (
        output.exists()
        or output.is_symlink()
        or not math.isfinite(maximum_allowed_mean_regression)
        or maximum_allowed_mean_regression < 0
    ):
        raise PairedNllError("paired assessment arguments differ")
    baseline, baseline_receipt = _report(baseline_path)
    candidate, candidate_receipt = _report(candidate_path)
    comparison_fields = (
        baseline.get("corpus", {}).get("manifest_payload_sha256")
        == candidate.get("corpus", {}).get("manifest_payload_sha256"),
        baseline.get("sampling") == candidate.get("sampling"),
        baseline.get("metric", {}).get("training_zloss_excluded") is True,
        candidate.get("metric", {}).get("training_zloss_excluded") is True,
    )
    baseline_windows = baseline.get("metric", {}).get("window_mean_nll")
    candidate_windows = candidate.get("metric", {}).get("window_mean_nll")
    if (
        not all(comparison_fields)
        or not isinstance(baseline_windows, list)
        or not isinstance(candidate_windows, list)
        or len(baseline_windows) != len(candidate_windows)
        or len(baseline_windows) < 2
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in [*baseline_windows, *candidate_windows]
        )
    ):
        raise PairedNllError("NLL reports are not paired on identical windows")
    deltas = [
        float(candidate_value) - float(baseline_value)
        for baseline_value, candidate_value in zip(
            baseline_windows,
            candidate_windows,
        )
    ]
    count = len(deltas)
    mean_delta = fmean(deltas)
    variance = sum((value - mean_delta) ** 2 for value in deltas) / (
        count - 1
    )
    standard_error = math.sqrt(variance / count)
    ci_low = mean_delta - 1.96 * standard_error
    ci_high = mean_delta + 1.96 * standard_error
    ordered = sorted(deltas)

    def quantile(fraction: float) -> float:
        index = round(fraction * (count - 1))
        return ordered[index]

    assessment = {
        "schema": ASSESSMENT_SCHEMA,
        "baseline": {
            "report": baseline_receipt,
            "report_payload_sha256": baseline["payload_sha256"],
            "checkpoint": baseline["checkpoint"],
        },
        "candidate": {
            "report": candidate_receipt,
            "report_payload_sha256": candidate["payload_sha256"],
            "checkpoint": candidate["checkpoint"],
        },
        "corpus_manifest_payload_sha256": baseline["corpus"][
            "manifest_payload_sha256"
        ],
        "sampling": baseline["sampling"],
        "paired_windows": count,
        "delta_definition": "candidate_mean_nll_minus_baseline_mean_nll",
        "statistics": {
            "mean_delta_nll": mean_delta,
            "standard_error": standard_error,
            "normal_95pct_ci": [ci_low, ci_high],
            "median_delta_nll": quantile(0.5),
            "p05_delta_nll": quantile(0.05),
            "p95_delta_nll": quantile(0.95),
            "candidate_window_win_fraction": sum(
                value < 0 for value in deltas
            )
            / count,
        },
        "gate": {
            "maximum_allowed_mean_regression": (
                maximum_allowed_mean_regression
            ),
            "upper_95pct_ci_within_regression_limit": (
                ci_high <= maximum_allowed_mean_regression
            ),
            "strict_improvement_upper_95pct_ci_below_zero": ci_high < 0,
        },
    }
    assessment["payload_sha256"] = canonical_payload_sha256(assessment)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as destination:
        json.dump(assessment, destination, indent=2, sort_keys=True)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    return assessment


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--maximum-allowed-mean-regression",
        type=float,
        default=0.0,
    )
    arguments = parser.parse_args(argv)
    result = assess_paired_nll(
        baseline_path=arguments.baseline,
        candidate_path=arguments.candidate,
        output=arguments.output,
        maximum_allowed_mean_regression=(
            arguments.maximum_allowed_mean_regression
        ),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
