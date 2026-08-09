#!/usr/bin/env python3
"""Apply the frozen OBR1 broad-owner development qualification gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


class OBR1ComparisonError(RuntimeError):
    """An OBR1 development or custody receipt differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(report: dict[str, Any], fit: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema") != "shohin-idr1-revision-evaluation-v1" or report.get("status") != "complete":
        raise OBR1ComparisonError("OBR1 evaluation report differs")
    if fit.get("schema") != "shohin-rme1-product-training-v1" or fit.get("status") != "complete":
        raise OBR1ComparisonError("OBR1 fit report differs")
    config = fit.get("rme1_config", {})
    custody = fit.get("sequence_custody", {})
    if (
        fit.get("architecture") != "shohin-rme1-moe-revision-v1"
        or fit.get("rme1_draft_control") != "draft_unavailable"
        or config.get("mode") != "shared"
        or config.get("controlled_layers") != 16
        or config.get("rank") != 18
        or fit.get("trainable_parameters") != 1_179_648
        or fit.get("updates") != 2048
        or fit.get("protected_router_expert_trainables") != 0
        or custody.get("overflow_rows") != 0
    ):
        raise OBR1ComparisonError("OBR1 geometry or sequence custody differs")
    if (
        data.get("schema") != "shohin-obr1-broad-owner-data-report-v1"
        or data.get("status") != "complete"
        or data.get("holdout_used") is not False
        or data.get("zero_exact_development_overlap") is not True
        or data.get("zero_ngram_development_overlap") is not True
        or data.get("complete_retention") is not True
    ):
        raise OBR1ComparisonError("OBR1 data custody differs")
    metrics = report.get("metrics", {})
    score = int(metrics.get("overall", {}).get("generated_correct", -1))
    domains = {
        task: int(metrics.get(task, {}).get("generated_correct", -1))
        for task in ("math500", "bbh_logic", "mbpp")
    }
    gates = {
        "overall_at_least_300": score >= 300,
        "math_at_least_75": domains["math500"] >= 75,
        "logic_science_at_least_215": domains["bbh_logic"] >= 215,
        "code_at_least_10": domains["mbpp"] >= 10,
        "complete_custody": True,
    }
    return {
        "schema": "shohin-obr1-development-comparison-v1",
        "status": "pass" if all(gates.values()) else "fail",
        "score": score,
        "domains": domains,
        "gates": gates,
        "qualified_for_temporal_gate": all(gates.values()),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise OBR1ComparisonError("refusing existing OBR1 comparison")
    result = compare(
        json.loads(args.evaluation_report.read_text()),
        json.loads(args.fit_report.read_text()),
        json.loads(args.data_report.read_text()),
    )
    result["inputs"] = {
        "evaluation_report": {"path": str(args.evaluation_report.resolve()), "sha256": sha256_file(args.evaluation_report)},
        "fit_report": {"path": str(args.fit_report.resolve()), "sha256": sha256_file(args.fit_report)},
        "data_report": {"path": str(args.data_report.resolve()), "sha256": sha256_file(args.data_report)},
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
