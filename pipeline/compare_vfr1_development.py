#!/usr/bin/env python3
"""Apply the frozen matched VFR1 development capability gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-vfr1-development-comparison-v1"
EVALUATION_SCHEMA = "shohin-vfr1-capability-evaluation-v1"
FIT_SCHEMA = "shohin-hf-product-reasoning-training-v1"
DOMAINS = ("math500", "bbh_logic", "mbpp")
FLOORS = {"overall": 603, "math500": 223, "bbh_logic": 349, "mbpp": 17}


class VFR1ComparisonError(RuntimeError):
    """A VFR1 capability receipt differs from the frozen contract."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "complete":
        raise VFR1ComparisonError(f"incomplete report: {path}")
    return value


def metric(report: dict[str, Any], domain: str = "overall") -> tuple[int, int]:
    value = report.get("metrics", {}).get(domain)
    if not isinstance(value, dict) or int(value.get("total", 0)) <= 0:
        raise VFR1ComparisonError(f"missing metrics for {domain}")
    return int(value["generated_correct"]), int(value["total"])


def complete_evaluation(report: dict[str, Any]) -> bool:
    return (
        report.get("schema") == EVALUATION_SCHEMA
        and report.get("merged_from_shards") is True
        and report.get("full_row_count") == 1289
        and int(report.get("shard_count", 0)) >= 2
        and float(report.get("parse_fraction", 0.0)) >= 0.95
    )


def complete_fit(report: dict[str, Any], data_path: Path) -> bool:
    return (
        report.get("schema") == FIT_SCHEMA
        and report.get("updates") == 256
        and report.get("batch_size") == 1
        and report.get("gradient_accumulation") == 8
        and report.get("max_sequence_length") == 4096
        and report.get("learning_rate") == 2e-5
        and report.get("lora_layers") == 4
        and report.get("lora_rank") == 8
        and report.get("lora_alpha") == 16.0
        and report.get("warm_start_update") == 256
        and report.get("data_sha256") == sha256_file(data_path)
    )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise VFR1ComparisonError(f"refusing existing comparison: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def compare(args: argparse.Namespace) -> dict[str, Any]:
    treatment = load(args.treatment_report)
    shuffled = load(args.shuffled_report)
    treatment_fit = load(args.treatment_fit)
    shuffled_fit = load(args.shuffled_fit)
    data_report = load(args.data_report)

    if not complete_evaluation(treatment) or not complete_evaluation(shuffled):
        raise VFR1ComparisonError("VFR1 evaluation coverage differs")
    for key in (
        "model_root",
        "model_revision",
        "data_sha256",
        "data_report_sha256",
        "max_new_tokens",
        "batch_size",
        "seed",
    ):
        if treatment.get(key) != shuffled.get(key):
            raise VFR1ComparisonError(f"evaluation setting differs: {key}")

    outputs = data_report.get("outputs", {})
    treatment_data = Path(str(outputs.get("train_treatment", {}).get("path", "")))
    shuffled_data = Path(str(outputs.get("train_shuffled", {}).get("path", "")))
    if not complete_fit(treatment_fit, treatment_data) or not complete_fit(
        shuffled_fit, shuffled_data
    ):
        raise VFR1ComparisonError("VFR1 fit receipt differs")
    for key in (
        "model_root",
        "model_revision",
        "model_loader",
        "warm_start_sha256",
        "warm_start_update",
        "trainable_parameters",
        "charged_tokens",
        "selected_rows",
        "seed",
        "data_seed",
    ):
        if treatment_fit.get(key) != shuffled_fit.get(key):
            raise VFR1ComparisonError(f"matched fit differs: {key}")

    treatment_correct, total = metric(treatment)
    shuffled_correct, shuffled_total = metric(shuffled)
    if total != 1289 or shuffled_total != total:
        raise VFR1ComparisonError("VFR1 evaluation totals differ")
    domains = {domain: metric(treatment, domain)[0] for domain in DOMAINS}
    gates = {
        "treatment_at_least_603": treatment_correct >= FLOORS["overall"],
        "treatment_beats_shuffled_by_10": treatment_correct >= shuffled_correct + 10,
        "math_at_least_223": domains["math500"] >= FLOORS["math500"],
        "logic_at_least_349": domains["bbh_logic"] >= FLOORS["bbh_logic"],
        "code_at_least_17": domains["mbpp"] >= FLOORS["mbpp"],
        "strict_parse_at_least_0_95": treatment["parse_fraction"] >= 0.95,
        "complete_matched_receipts": True,
    }
    gate_pass = all(gates.values())
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "split": "development",
        "arms": {
            "treatment": {
                "correct": treatment_correct,
                "accuracy": treatment_correct / total,
                "parse_fraction": treatment["parse_fraction"],
                "report_sha256": sha256_file(args.treatment_report),
            },
            "shuffled_fault": {
                "correct": shuffled_correct,
                "accuracy": shuffled_correct / total,
                "parse_fraction": shuffled["parse_fraction"],
                "report_sha256": sha256_file(args.shuffled_report),
            },
        },
        "treatment_minus_shuffled_answers": treatment_correct - shuffled_correct,
        "treatment_minus_shuffled_points": 100.0
        * (treatment_correct - shuffled_correct)
        / total,
        "domain_correct": domains,
        "charged_target_tokens_per_fit": treatment_fit["charged_tokens"],
        "trainable_parameters_per_fit": treatment_fit["trainable_parameters"],
        "gates": gates,
        "gate_pass": gate_pass,
        "holdout_authorized": gate_pass,
        "decision": (
            "allow_one_sealed_vfr1_holdout_evaluation"
            if gate_pass
            else "close_exact_vfr1_without_rescue"
        ),
        "receipts": {
            "data_report_sha256": sha256_file(args.data_report),
            "treatment_fit_sha256": sha256_file(args.treatment_fit),
            "shuffled_fit_sha256": sha256_file(args.shuffled_fit),
        },
    }
    atomic_json(args.output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "treatment_report",
        "shuffled_report",
        "treatment_fit",
        "shuffled_fit",
        "data_report",
        "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    result = compare(parser.parse_args())
    print(json.dumps({"gate_pass": result["gate_pass"], "gates": result["gates"]}, indent=2))
    return 0 if result["gate_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
