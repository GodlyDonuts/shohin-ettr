#!/usr/bin/env python3
"""Apply the frozen SER1 Stage-1 development gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


class SER1ComparisonError(RuntimeError):
    """A SER1 Stage-1 input differs from the frozen contract."""


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "complete":
        raise SER1ComparisonError(f"incomplete report: {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(report: dict[str, Any], domain: str = "overall") -> tuple[int, int]:
    value = report.get("metrics", {}).get(domain)
    if not isinstance(value, dict) or int(value.get("total", 0)) <= 0:
        raise SER1ComparisonError(f"missing metric: {domain}")
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


def complete_fit(report: dict[str, Any], *, mode: str, rank: int, parameters: int) -> bool:
    config = report.get("ser1_config", {})
    custody = report.get("sequence_custody", {})
    return (
        report.get("schema") == "shohin-ser1-product-training-v1"
        and report.get("updates") == 256
        and report.get("selected_rows") == 9651
        and report.get("trainable_parameters") == parameters
        and config.get("mode") == mode
        and config.get("controlled_layers") == 16
        and config.get("rank") == rank
        and config.get("alpha") == float(rank)
        and report.get("ser1_draft_control") == "normal"
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
    paths = {
        "treatment": args.treatment_report,
        "shared_flop": args.shared_flop_report,
        "shared_parameter": args.shared_parameter_report,
    }
    reports = {name: load(path) for name, path in paths.items()}
    fits = {
        "treatment": load(args.treatment_fit),
        "shared_flop": load(args.shared_flop_fit),
        "shared_parameter": load(args.shared_parameter_fit),
    }
    if not all(complete_evaluation(report) for report in reports.values()):
        raise SER1ComparisonError("SER1 evaluation coverage differs")
    reference = reports["treatment"]
    for report in reports.values():
        for key in ("model_revision", "data_sha256", "data_report_sha256"):
            if report.get(key) != reference.get(key):
                raise SER1ComparisonError("SER1 evaluation inputs differ")
    expected = {
        "treatment": ("selected_expert", 1, 4_194_304),
        "shared_flop": ("shared", 8, 524_288),
        "shared_parameter": ("shared", 64, 4_194_304),
    }
    charged = None
    for name, (mode, rank, parameters) in expected.items():
        if not complete_fit(fits[name], mode=mode, rank=rank, parameters=parameters):
            raise SER1ComparisonError(f"{name} fit receipt differs")
        if charged is None:
            charged = fits[name].get("charged_tokens")
        elif fits[name].get("charged_tokens") != charged:
            raise SER1ComparisonError("SER1 token budgets differ")
    treatment_correct, total = metric(reference)
    arm_correct = {name: metric(report)[0] for name, report in reports.items()}
    strongest = max(arm_correct["shared_flop"], arm_correct["shared_parameter"])
    domains = {
        domain: metric(reference, domain)[0]
        for domain in ("math500", "bbh_logic", "mbpp")
    }
    gates = {
        "treatment_at_least_280": treatment_correct >= 280,
        "treatment_beats_strongest_control_by_26": treatment_correct >= strongest + 26,
        "domain_floors": domains["math500"] >= 55
        and domains["bbh_logic"] >= 180
        and domains["mbpp"] >= 5,
        "complete_receipts": True,
    }
    result = {
        "schema": "shohin-ser1-development-comparison-v1",
        "status": "complete",
        "split": "development",
        "arms": {
            name: {
                "correct": arm_correct[name],
                "accuracy": arm_correct[name] / total,
                "report_sha256": sha256_file(paths[name]),
            }
            for name in reports
        },
        "references": {"unchanged": 191, "mtr1": 204, "ecr1_all_layer": 240},
        "domain_correct": domains,
        "charged_target_tokens_per_fit": charged,
        "gates": gates,
        "stage_two_authorized": all(gates.values()),
        "holdout_authorized": False,
        "close_ser1_if_false": not all(gates.values()),
    }
    atomic_json(args.output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "treatment_report",
        "shared_flop_report",
        "shared_parameter_report",
        "treatment_fit",
        "shared_flop_fit",
        "shared_parameter_fit",
        "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    result = compare(parser.parse_args())
    print(json.dumps({"stage_two_authorized": result["stage_two_authorized"], "gates": result["gates"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
