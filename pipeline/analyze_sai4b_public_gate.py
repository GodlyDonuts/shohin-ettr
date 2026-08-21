#!/usr/bin/env python3
"""Apply Sai's benchmark-first promotion gate to five official score reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "sai-4b-public-benchmark-score-v1"
SCHEMA = "sai-4b-public-benchmark-gate-v1"
BENCHMARKS = ("humaneval_plus", "mbpp_plus", "ifeval", "musr", "correctbench")
SHA256_KEYS = (
    "benchmark_source_sha256",
    "identity_order_sha256",
    "prompt_contract_sha256",
    "decoding_contract_sha256",
    "original_checkpoint_sha256",
    "equal_compute_checkpoint_sha256",
    "candidate_checkpoint_sha256",
)


class SaiPublicGateError(RuntimeError):
    """A score report or its matched-comparison binding is invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SaiPublicGateError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 100.0:
        raise SaiPublicGateError(f"{field} is outside [0, 100]")
    return result


def _validate_report(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise SaiPublicGateError("score report must be an object")
    if report.get("schema") != REPORT_SCHEMA or report.get("status") != "complete":
        raise SaiPublicGateError("score report schema/status differs")
    benchmark = report.get("benchmark")
    if benchmark not in BENCHMARKS:
        raise SaiPublicGateError("benchmark identity differs")
    rows = report.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise SaiPublicGateError("benchmark row count differs")
    if (
        not isinstance(report.get("benchmark_version"), str)
        or not report["benchmark_version"]
    ):
        raise SaiPublicGateError("benchmark version is missing")
    for key in SHA256_KEYS:
        value = report.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise SaiPublicGateError(f"{key} differs")
        try:
            bytes.fromhex(value)
        except ValueError as error:
            raise SaiPublicGateError(f"{key} differs") from error
    return {
        **report,
        "original_score": _score(report.get("original_score"), "original_score"),
        "equal_compute_score": _score(
            report.get("equal_compute_score"), "equal_compute_score"
        ),
        "candidate_score": _score(report.get("candidate_score"), "candidate_score"),
    }


def run(paths: list[Path], output: Path) -> dict[str, Any]:
    if len(paths) != len(BENCHMARKS):
        raise SaiPublicGateError("exactly five score reports are required")
    reports = [_validate_report(json.loads(path.read_text())) for path in paths]
    if {report["benchmark"] for report in reports} != set(BENCHMARKS):
        raise SaiPublicGateError("one report per required benchmark is required")

    first = reports[0]
    matched = (
        "original_checkpoint_sha256",
        "equal_compute_checkpoint_sha256",
        "candidate_checkpoint_sha256",
        "prompt_contract_sha256",
        "decoding_contract_sha256",
    )
    for report in reports[1:]:
        for key in matched[:3]:
            if report[key] != first[key]:
                raise SaiPublicGateError(f"cross-benchmark {key} differs")

    ordered = sorted(reports, key=lambda report: BENCHMARKS.index(report["benchmark"]))
    per_benchmark: dict[str, dict[str, Any]] = {}
    for report in ordered:
        candidate = report["candidate_score"]
        original = report["original_score"]
        control = report["equal_compute_score"]
        per_benchmark[report["benchmark"]] = {
            "rows": report["rows"],
            "benchmark_version": report["benchmark_version"],
            "original_score": original,
            "equal_compute_score": control,
            "candidate_score": candidate,
            "candidate_vs_original_points": candidate - original,
            "candidate_vs_equal_compute_points": candidate - control,
            "original_nonregression": candidate >= original - 1.0,
            "equal_compute_nonregression": candidate >= control - 1.0,
        }

    macro_original = sum(item["original_score"] for item in per_benchmark.values()) / 5
    macro_control = (
        sum(item["equal_compute_score"] for item in per_benchmark.values()) / 5
    )
    macro_candidate = (
        sum(item["candidate_score"] for item in per_benchmark.values()) / 5
    )
    checks = {
        "macro_beats_original_by_at_least_1_point": macro_candidate
        >= macro_original + 1.0,
        "macro_beats_equal_compute_by_at_least_1_point": macro_candidate
        >= macro_control + 1.0,
        "no_benchmark_regresses_over_1_point_vs_original": all(
            item["original_nonregression"] for item in per_benchmark.values()
        ),
        "no_benchmark_regresses_over_1_point_vs_equal_compute": all(
            item["equal_compute_nonregression"] for item in per_benchmark.values()
        ),
        "beats_original_on_at_least_four_benchmarks": sum(
            item["candidate_vs_original_points"] > 0 for item in per_benchmark.values()
        )
        >= 4,
        "beats_equal_compute_on_at_least_four_benchmarks": sum(
            item["candidate_vs_equal_compute_points"] > 0
            for item in per_benchmark.values()
        )
        >= 4,
        "musr_nonnegative_vs_both": min(
            per_benchmark["musr"]["candidate_vs_original_points"],
            per_benchmark["musr"]["candidate_vs_equal_compute_points"],
        )
        >= 0,
        "correctbench_nonnegative_vs_both": min(
            per_benchmark["correctbench"]["candidate_vs_original_points"],
            per_benchmark["correctbench"]["candidate_vs_equal_compute_points"],
        )
        >= 0,
    }
    promote = all(checks.values())
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "decision": "promote_sai_candidate" if promote else "reject_sai_candidate",
        "architecture_locked": False,
        "promote_to_full_confirmation": promote,
        "stop_candidate": not promote,
        "reports": [
            {"path": str(path.resolve()), "sha256": _sha256_file(path)}
            for path in paths
        ],
        "checkpoints": {key: first[key] for key in matched[:3]},
        "macro": {
            "original_score": macro_original,
            "equal_compute_score": macro_control,
            "candidate_score": macro_candidate,
            "candidate_vs_original_points": macro_candidate - macro_original,
            "candidate_vs_equal_compute_points": macro_candidate - macro_control,
        },
        "benchmarks": per_benchmark,
        "checks": checks,
    }
    if output.exists():
        raise SaiPublicGateError("gate output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run(args.score, args.output)
    print(json.dumps({"decision": payload["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
