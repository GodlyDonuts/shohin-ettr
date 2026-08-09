#!/usr/bin/env python3
"""Apply the frozen matched NDR1 development gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-ndr1-development-comparison-v1"
EVAL_SCHEMA = "shohin-idr1-revision-evaluation-v1"
FIT_SCHEMA = "shohin-hf-product-reasoning-training-v1"


class NDR1ComparisonError(RuntimeError):
    """NDR1 fit or evaluation evidence differs from the frozen contract."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("status") != "complete":
        raise NDR1ComparisonError(f"incomplete report: {path}")
    return value


def score(report: dict[str, Any], domain: str = "overall") -> tuple[int, int]:
    value = report.get("metrics", {}).get(domain)
    if not isinstance(value, dict) or int(value.get("total", 0)) <= 0:
        raise NDR1ComparisonError(f"missing score for {domain}")
    return int(value["generated_correct"]), int(value["total"])


def complete_eval(report: dict[str, Any]) -> bool:
    return (
        report.get("schema") == EVAL_SCHEMA
        and report.get("split") == "development"
        and report.get("full_row_count") == 1289
        and report.get("merged_from_shards") is True
        and int(report.get("shard_count", 0)) >= 2
    )


def complete_fit(report: dict[str, Any], data: Path) -> bool:
    return (
        report.get("schema") == FIT_SCHEMA
        and report.get("updates") == 512
        and report.get("batch_size") == 1
        and report.get("gradient_accumulation") == 8
        and report.get("max_sequence_length") == 4096
        and report.get("learning_rate") == 2e-5
        and report.get("lora_layers") == 4
        and report.get("lora_rank") == 8
        and report.get("lora_alpha") == 16.0
        and report.get("warm_start_update") == 256
        and report.get("seed") == 2026080921
        and report.get("data_seed") == 2026080920
        and report.get("data_sha256") == sha256_file(data)
    )


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise NDR1ComparisonError(f"refusing existing comparison: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def compare(args: argparse.Namespace) -> dict[str, Any]:
    aligned = load(args.aligned_report)
    shuffled = load(args.shuffled_report)
    aligned_fit = load(args.aligned_fit)
    shuffled_fit = load(args.shuffled_fit)
    data_report = load(args.data_report)
    if data_report.get("schema") != "shohin-ndr1-natural-revision-data-report-v1":
        raise NDR1ComparisonError("NDR1 data report differs")
    if not complete_eval(aligned) or not complete_eval(shuffled):
        raise NDR1ComparisonError("NDR1 evaluation coverage differs")
    for key in ("model_root", "model_revision", "data_sha256", "data_report_sha256"):
        if aligned.get(key) != shuffled.get(key):
            raise NDR1ComparisonError(f"NDR1 evaluation setting differs: {key}")
    outputs = data_report.get("outputs", {})
    aligned_data = Path(str(outputs.get("aligned", {}).get("path", "")))
    shuffled_data = Path(str(outputs.get("shuffled", {}).get("path", "")))
    if not complete_fit(aligned_fit, aligned_data) or not complete_fit(shuffled_fit, shuffled_data):
        raise NDR1ComparisonError("NDR1 fit receipt differs")
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
        if aligned_fit.get(key) != shuffled_fit.get(key):
            raise NDR1ComparisonError(f"NDR1 matched fit differs: {key}")

    aligned_correct, total = score(aligned)
    shuffled_correct, shuffled_total = score(shuffled)
    if total != 1289 or shuffled_total != total:
        raise NDR1ComparisonError("NDR1 evaluation totals differ")
    domains = {
        "math500": score(aligned, "math500")[0],
        "bbh_logic": score(aligned, "bbh_logic")[0],
        "mbpp": score(aligned, "mbpp")[0],
    }
    aligned_exhausted = int(aligned.get("max_token_exhausted", -1))
    shuffled_exhausted = int(shuffled.get("max_token_exhausted", -1))
    gates = {
        "aligned_at_least_603": aligned_correct >= 603,
        "aligned_beats_shuffled_by_10": aligned_correct >= shuffled_correct + 10,
        "math_at_least_223": domains["math500"] >= 223,
        "logic_at_least_349": domains["bbh_logic"] >= 349,
        "code_at_least_17": domains["mbpp"] >= 17,
        "aligned_exhaustion_at_most_400": 0 <= aligned_exhausted <= 400,
        "aligned_exhaustion_within_25_of_shuffled": (
            shuffled_exhausted >= 0 and aligned_exhausted <= shuffled_exhausted + 25
        ),
        "complete_matched_receipts": True,
    }
    passed = all(gates.values())
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "split": "development",
        "arms": {
            "aligned": {
                "correct": aligned_correct,
                "accuracy": aligned_correct / total,
                "max_token_exhausted": aligned_exhausted,
                "report_sha256": sha256_file(args.aligned_report),
            },
            "shuffled": {
                "correct": shuffled_correct,
                "accuracy": shuffled_correct / total,
                "max_token_exhausted": shuffled_exhausted,
                "report_sha256": sha256_file(args.shuffled_report),
            },
        },
        "aligned_minus_shuffled_answers": aligned_correct - shuffled_correct,
        "domain_correct": domains,
        "charged_target_tokens_per_fit": aligned_fit["charged_tokens"],
        "trainable_parameters_per_fit": aligned_fit["trainable_parameters"],
        "gates": gates,
        "gate_pass": passed,
        "holdout_authorized": passed,
        "decision": (
            "allow_one_sealed_ndr1_holdout_evaluation"
            if passed
            else "close_exact_ndr1_without_rescue"
        ),
    }
    atomic_json(args.output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "aligned_report",
        "shuffled_report",
        "aligned_fit",
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
