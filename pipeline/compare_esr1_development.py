#!/usr/bin/env python3
"""Apply the frozen ESR1 development gate to matched complete reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-esr1-development-comparison-v1"
EVAL_SCHEMA = "shohin-idr1-revision-evaluation-v1"
TRAIN_SCHEMA = "shohin-hf-product-reasoning-training-v1"
DOMAINS = ("math500", "bbh_logic", "mbpp")


class ESR1ComparisonError(RuntimeError):
    """The frozen ESR1 comparison contract was violated."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_complete(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema") != schema
        or value.get("status") != "complete"
    ):
        raise ESR1ComparisonError(f"incomplete ESR1 artifact: {path}")
    return value


def accuracy(report: dict[str, Any], domain: str = "overall") -> float:
    metrics = report.get("metrics", {}).get(domain)
    if not isinstance(metrics, dict) or int(metrics.get("total", 0)) <= 0:
        raise ESR1ComparisonError(f"missing ESR1 metrics: {domain}")
    return int(metrics["generated_correct"]) / int(metrics["total"])


def correct(report: dict[str, Any], domain: str) -> int:
    return int(report["metrics"][domain]["generated_correct"])


def complete_coverage(report: dict[str, Any]) -> bool:
    return (
        report.get("split") == "development"
        and report.get("full_row_count") == 1289
        and report.get("merged_from_shards") is True
        and report.get("shard_count") == 8
    )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing comparison: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def compare(args: argparse.Namespace) -> dict[str, Any]:
    syndrome = load_complete(args.syndrome_report, EVAL_SCHEMA)
    ettr = load_complete(args.ettr_report, EVAL_SCHEMA)
    always = load_complete(args.always_report, EVAL_SCHEMA)
    syndrome_fit = load_complete(args.syndrome_fit_report, TRAIN_SCHEMA)
    ettr_fit = load_complete(args.ettr_fit_report, TRAIN_SCHEMA)
    if syndrome_fit.get("arm") != "syndrome" or ettr_fit.get("arm") != "ettr":
        raise ESR1ComparisonError("ESR1 fit arm differs")
    if syndrome_fit.get("architecture") != "shohin-error-syndrome-revision-v1":
        raise ESR1ComparisonError("ESR1 treatment architecture differs")
    matched_fit_keys = (
        "model_root",
        "model_revision",
        "data_sha256",
        "selected_rows",
        "seed",
        "data_seed",
        "lora_layers",
        "lora_rank",
        "lora_alpha",
        "unfreeze_layers",
        "workspace_config",
        "updates",
        "gradient_accumulation",
        "batch_size",
        "max_sequence_length",
        "learning_rate",
    )
    if any(syndrome_fit.get(key) != ettr_fit.get(key) for key in matched_fit_keys):
        raise ESR1ComparisonError("ESR1 treatment/control fit settings differ")
    matched_eval_keys = ("split", "model_revision", "data_sha256", "data_report_sha256")
    if any(
        report.get(key) != syndrome.get(key)
        for report in (ettr, always)
        for key in matched_eval_keys
    ):
        raise ESR1ComparisonError("ESR1 evaluation settings differ")
    if not all(complete_coverage(report) for report in (syndrome, ettr, always)):
        raise ESR1ComparisonError("ESR1 development coverage differs")

    syndrome_accuracy = accuracy(syndrome)
    ettr_accuracy = accuracy(ettr)
    always_accuracy = accuracy(always)
    domain_deltas = {
        domain: correct(syndrome, domain) - correct(always, domain)
        for domain in DOMAINS
    }
    gates = {
        "syndrome_beats_always_revise_by_5_points": syndrome_accuracy
        >= always_accuracy + 0.05,
        "syndrome_beats_workspace_control_by_3_points": syndrome_accuracy
        >= ettr_accuracy + 0.03,
        "all_domain_deltas_vs_always_nonnegative": all(
            delta >= 0 for delta in domain_deltas.values()
        ),
        "complete_identity_coverage": True,
    }
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "split": "development",
        "model_revision": syndrome["model_revision"],
        "data_sha256": syndrome["data_sha256"],
        "reports": {
            "syndrome": {
                "path": str(args.syndrome_report.resolve()),
                "sha256": sha256_file(args.syndrome_report),
                "accuracy": syndrome_accuracy,
            },
            "workspace_control": {
                "path": str(args.ettr_report.resolve()),
                "sha256": sha256_file(args.ettr_report),
                "accuracy": ettr_accuracy,
            },
            "always_revise": {
                "path": str(args.always_report.resolve()),
                "sha256": sha256_file(args.always_report),
                "accuracy": always_accuracy,
            },
        },
        "syndrome_minus_always_points": 100 * (syndrome_accuracy - always_accuracy),
        "syndrome_minus_workspace_control_points": 100
        * (syndrome_accuracy - ettr_accuracy),
        "domain_correct_count_deltas_vs_always": domain_deltas,
        "gates": gates,
        "gate_pass": all(gates.values()),
        "holdout_authorized": all(gates.values()),
    }
    atomic_json(args.output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--syndrome-report", type=Path, required=True)
    parser.add_argument("--ettr-report", type=Path, required=True)
    parser.add_argument("--always-report", type=Path, required=True)
    parser.add_argument("--syndrome-fit-report", type=Path, required=True)
    parser.add_argument("--ettr-fit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args)
    print(json.dumps({"gate_pass": result["gate_pass"], "gates": result["gates"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
