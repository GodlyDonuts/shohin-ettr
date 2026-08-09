#!/usr/bin/env python3
"""Apply the frozen ECR1 development gate to complete matched reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-ecr1-development-comparison-v1"
DOMAINS = ("math500", "bbh_logic", "mbpp")


class ECR1ComparisonError(RuntimeError):
    """A frozen ECR1 comparison input differs."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "complete":
        raise ECR1ComparisonError(f"incomplete report: {path}")
    return value


def metric(report: dict[str, Any], domain: str = "overall") -> tuple[int, int]:
    value = report.get("metrics", {}).get(domain)
    if not isinstance(value, dict) or int(value.get("total", 0)) <= 0:
        raise ECR1ComparisonError(f"missing metrics for {domain}")
    return int(value["generated_correct"]), int(value["total"])


def complete(report: dict[str, Any]) -> bool:
    return (
        report.get("schema") == "shohin-idr1-revision-evaluation-v1"
        and report.get("split") == "development"
        and report.get("full_row_count") == 1289
        and report.get("merged_from_shards") is True
        and int(report.get("shard_count", 0)) == 8
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
    paths = {
        "treatment": args.treatment_report,
        "shared": args.shared_report,
        "draft_unavailable": args.draft_report,
        "code_zero": args.zero_report,
        "code_mean": args.mean_report,
        "code_permutation": args.permutation_report,
    }
    reports = {name: load(path) for name, path in paths.items()}
    unchanged = load(args.unchanged_report)
    mtr = load(args.mtr_report)
    fits = {
        "treatment": load(args.treatment_fit),
        "shared": load(args.shared_fit),
        "draft_unavailable": load(args.draft_fit),
    }
    semantic = load(args.semantic_report)
    if not all(complete(report) for report in reports.values()):
        raise ECR1ComparisonError("ECR1 development coverage differs")
    shared_keys = ("model_revision", "data_sha256", "data_report_sha256")
    reference = reports["treatment"]
    if any(
        report.get(key) != reference.get(key)
        for report in reports.values()
        for key in shared_keys
    ):
        raise ECR1ComparisonError("ECR1 evaluation arms differ")
    expected_controls = {
        "treatment": "normal",
        "shared": "normal",
        "draft_unavailable": "normal",
        "code_zero": "zero",
        "code_mean": "mean",
        "code_permutation": "permutation",
    }
    if any(
        reports[name].get("ecr_code_intervention") != control
        for name, control in expected_controls.items()
    ):
        raise ECR1ComparisonError("ECR1 code intervention differs")
    expected_fits = {
        "treatment": (515_840, "expert_conditioned", "normal"),
        "shared": (524_288, "shared", "normal"),
        "draft_unavailable": (515_840, "expert_conditioned", "draft_unavailable"),
    }
    charged = None
    for name, (parameters, mode, draft_control) in expected_fits.items():
        fit = fits[name]
        if (
            fit.get("schema") != "shohin-ecr1-product-training-v1"
            or fit.get("updates") != 256
            or fit.get("selected_rows") != 9651
            or fit.get("trainable_parameters") != parameters
            or fit.get("ecr1_config", {}).get("mode") != mode
            or fit.get("ecr1_draft_control") != draft_control
            or fit.get("protected_router_expert_trainables") != 0
            or fit.get("sequence_custody", {}).get("overflow_rows") != 0
        ):
            raise ECR1ComparisonError(f"{name} fit receipt differs")
        if charged is None:
            charged = fit.get("charged_tokens")
        elif fit.get("charged_tokens") != charged:
            raise ECR1ComparisonError("ECR1 fit token budgets differ")
    treatment_correct, total = metric(reference)
    arm_correct = {name: metric(report)[0] for name, report in reports.items()}
    strongest_control = max(arm_correct["shared"], arm_correct["draft_unavailable"])
    unchanged_correct = metric(unchanged)[0]
    domain_counts = {domain: metric(reference, domain)[0] for domain in DOMAINS}
    semantic_counts = semantic.get("counts", {})
    gates = {
        "treatment_at_least_256": treatment_correct >= 256,
        "treatment_beats_strongest_new_control_by_39": treatment_correct
        >= strongest_control + 39,
        "domain_floors": domain_counts["math500"] >= 40
        and domain_counts["bbh_logic"] >= 145
        and domain_counts["mbpp"] >= 5,
        "draft_causality_margin_13": treatment_correct
        >= arm_correct["draft_unavailable"] + 13,
        "code_zero_margin_13": treatment_correct >= arm_correct["code_zero"] + 13,
        "code_mean_margin_13": treatment_correct >= arm_correct["code_mean"] + 13,
        "code_permutation_margin_13": treatment_correct
        >= arm_correct["code_permutation"] + 13,
        "at_least_25_possible_semantic_repairs": int(
            semantic_counts.get("remaining_possible_semantic_repairs", -1)
        )
        >= 25,
        "semantic_net_at_least_20": int(
            semantic_counts.get("remaining_possible_semantic_repairs", -1)
        )
        - int(semantic_counts.get("strict_breaks", 10**9))
        >= 20,
        "complete_receipts": all(complete(report) for report in reports.values()),
    }
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "split": "development",
        "model_revision": reference["model_revision"],
        "arms": {
            name: {
                "correct": arm_correct[name],
                "accuracy": arm_correct[name] / total,
                "report_sha256": sha256_file(paths[name]),
            }
            for name in reports
        },
        "references": {
            "unchanged": unchanged_correct,
            "mtr": metric(mtr)[0],
        },
        "treatment_minus_unchanged_points": 100
        * ((treatment_correct - unchanged_correct) / total),
        "treatment_minus_shared_points": 100
        * ((treatment_correct - arm_correct["shared"]) / total),
        "domain_correct": domain_counts,
        "semantic_counts": semantic_counts,
        "charged_target_tokens_per_fit": charged,
        "gates": gates,
        "gate_pass": all(gates.values()),
        "holdout_authorized": all(gates.values()),
    }
    atomic_json(args.output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "treatment_report",
        "shared_report",
        "draft_report",
        "zero_report",
        "mean_report",
        "permutation_report",
        "unchanged_report",
        "mtr_report",
        "treatment_fit",
        "shared_fit",
        "draft_fit",
        "semantic_report",
        "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    result = compare(parser.parse_args())
    print(json.dumps({"gate_pass": result["gate_pass"], "gates": result["gates"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
