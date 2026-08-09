#!/usr/bin/env python3
"""Apply the frozen Stage-1 gate to the all-layer ECR1 depth follow-up."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-ecr1-depth-development-comparison-v1"
DOMAINS = ("math500", "bbh_logic", "mbpp")


class ECR1DepthComparisonError(RuntimeError):
    """The frozen ECR1 depth-follow-up inputs differ."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "complete":
        raise ECR1DepthComparisonError(f"incomplete report: {path}")
    return value


def metric(report: dict[str, Any], domain: str = "overall") -> tuple[int, int]:
    value = report.get("metrics", {}).get(domain)
    if not isinstance(value, dict) or int(value.get("total", 0)) <= 0:
        raise ECR1DepthComparisonError(f"missing metrics for {domain}")
    return int(value["generated_correct"]), int(value["total"])


def complete_evaluation(report: dict[str, Any]) -> bool:
    return (
        report.get("schema") == "shohin-idr1-revision-evaluation-v1"
        and report.get("split") == "development"
        and report.get("full_row_count") == 1289
        and report.get("merged_from_shards") is True
        and int(report.get("shard_count", 0)) == 8
        and report.get("ecr_code_intervention") == "normal"
    )


def complete_fit(
    report: dict[str, Any], *, parameters: int, mode: str
) -> bool:
    config = report.get("ecr1_config", {})
    custody = report.get("sequence_custody", {})
    return (
        report.get("schema") == "shohin-ecr1-product-training-v1"
        and report.get("updates") == 256
        and report.get("selected_rows") == 9651
        and report.get("trainable_parameters") == parameters
        and config.get("mode") == mode
        and config.get("controlled_layers") == 16
        and config.get("rank") == 8
        and config.get("alpha") == 8.0
        and report.get("ecr1_draft_control") == "normal"
        and report.get("protected_router_expert_trainables") == 0
        and custody.get("overflow_rows") == 0
        and custody.get("source_retention") == 1.0
        and custody.get("draft_retention") == 1.0
        and custody.get("target_retention") == 1.0
    )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing comparison: {path}")
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
    shared = load(args.shared_report)
    treatment_fit = load(args.treatment_fit)
    shared_fit = load(args.shared_fit)
    final_four = load(args.final_four_comparison)
    unchanged = load(args.unchanged_report)
    mtr = load(args.mtr_report)

    if final_four.get("schema") != "shohin-ecr1-development-comparison-v1":
        raise ECR1DepthComparisonError("final-four comparison differs")
    final_four_arms = final_four.get("arms", {})
    eligibility = (
        int(final_four_arms.get("treatment", {}).get("correct", -1)) >= 204
        and int(final_four_arms.get("shared", {}).get("correct", -1)) >= 204
        and final_four.get("holdout_authorized") is False
    )
    if not eligibility:
        raise ECR1DepthComparisonError("conditional depth eligibility is false")
    if not complete_evaluation(treatment) or not complete_evaluation(shared):
        raise ECR1DepthComparisonError("depth evaluation coverage differs")
    for key in ("model_revision", "data_sha256", "data_report_sha256"):
        if treatment.get(key) != shared.get(key):
            raise ECR1DepthComparisonError("depth evaluation inputs differ")
    if not complete_fit(
        treatment_fit, parameters=532_480, mode="expert_conditioned"
    ) or not complete_fit(shared_fit, parameters=524_288, mode="shared"):
        raise ECR1DepthComparisonError("depth fit receipt differs")
    if treatment_fit.get("charged_tokens") != shared_fit.get("charged_tokens"):
        raise ECR1DepthComparisonError("depth fit token budgets differ")

    treatment_correct, total = metric(treatment)
    shared_correct, shared_total = metric(shared)
    if total != shared_total:
        raise ECR1DepthComparisonError("depth evaluation totals differ")
    unchanged_correct = metric(unchanged)[0]
    domains = {domain: metric(treatment, domain)[0] for domain in DOMAINS}
    gates = {
        "treatment_at_least_256": treatment_correct >= 256,
        "treatment_beats_shared_by_39": treatment_correct >= shared_correct + 39,
        "domain_floors": domains["math500"] >= 40
        and domains["bbh_logic"] >= 145
        and domains["mbpp"] >= 5,
        "complete_receipts": True,
    }
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "split": "development",
        "conditional_eligibility": eligibility,
        "arms": {
            "treatment": {
                "correct": treatment_correct,
                "accuracy": treatment_correct / total,
                "report_sha256": sha256_file(args.treatment_report),
            },
            "shared": {
                "correct": shared_correct,
                "accuracy": shared_correct / total,
                "report_sha256": sha256_file(args.shared_report),
            },
        },
        "references": {
            "unchanged": unchanged_correct,
            "mtr": metric(mtr)[0],
            "final_four_ecr": int(final_four_arms["treatment"]["correct"]),
            "final_four_shared": int(final_four_arms["shared"]["correct"]),
        },
        "domain_correct": domains,
        "charged_target_tokens_per_fit": treatment_fit["charged_tokens"],
        "treatment_minus_shared_points": 100
        * ((treatment_correct - shared_correct) / total),
        "treatment_minus_unchanged_points": 100
        * ((treatment_correct - unchanged_correct) / total),
        "gates": gates,
        "stage_two_causal_controls_authorized": all(gates.values()),
        "holdout_authorized": False,
        "close_ecr1_if_false": not all(gates.values()),
    }
    atomic_json(args.output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "treatment_report",
        "shared_report",
        "treatment_fit",
        "shared_fit",
        "final_four_comparison",
        "unchanged_report",
        "mtr_report",
        "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    result = compare(parser.parse_args())
    print(
        json.dumps(
            {
                "stage_two_causal_controls_authorized": result[
                    "stage_two_causal_controls_authorized"
                ],
                "gates": result["gates"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
