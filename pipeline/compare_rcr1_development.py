#!/usr/bin/env python3
"""Apply the frozen RCR1 development gate to complete matched reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-rcr1-development-comparison-v1"
DOMAINS = ("math500", "bbh_logic", "mbpp")


class RCR1ComparisonError(RuntimeError):
    """The frozen RCR1 comparison contract differs."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "complete":
        raise RCR1ComparisonError(f"incomplete report: {path}")
    return value


def metric(report: dict[str, Any], domain: str = "overall") -> tuple[int, int]:
    value = report.get("metrics", {}).get(domain)
    if not isinstance(value, dict) or int(value.get("total", 0)) <= 0:
        raise RCR1ComparisonError(f"missing metrics for {domain}")
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
    with temporary.open("x", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def compare(args: argparse.Namespace) -> dict[str, Any]:
    router = load(args.router_report)
    attention = load(args.attention_report)
    unchanged = load(args.unchanged_report)
    prior = load(args.prior_treatment_report)
    router_fit = load(args.router_fit_report)
    attention_fit = load(args.attention_fit_report)
    for report in (router, attention, prior):
        if not complete(report):
            raise RCR1ComparisonError("revision report coverage differs")
    shared = ("model_revision", "data_sha256", "data_report_sha256")
    if any(router.get(key) != attention.get(key) for key in shared):
        raise RCR1ComparisonError("RCR1 arms are not matched")
    if any(router.get(key) != unchanged.get(key) for key in shared):
        raise RCR1ComparisonError("unchanged control is not matched")
    if unchanged.get("control") != "unchanged_second_pass":
        raise RCR1ComparisonError("unchanged control differs")
    expected_fit = {
        "router": (router_fit, "router", 67584),
        "attention": (attention_fit, "token_mixer", 65536),
    }
    for name, (fit, scope, parameters) in expected_fit.items():
        if (
            fit.get("updates") != 256
            or fit.get("charged_tokens") != 342896
            or fit.get("lora_scope") != scope
            or fit.get("trainable_parameters") != parameters
        ):
            raise RCR1ComparisonError(f"{name} fit budget differs")
    parameter_ratio = 67584 / 65536
    router_correct, total = metric(router)
    attention_correct, attention_total = metric(attention)
    unchanged_correct, unchanged_total = metric(unchanged)
    if len({total, attention_total, unchanged_total}) != 1:
        raise RCR1ComparisonError("RCR1 report totals differ")
    domain_deltas = {
        domain: metric(router, domain)[0] - metric(unchanged, domain)[0]
        for domain in DOMAINS
    }
    router_accuracy = router_correct / total
    attention_accuracy = attention_correct / total
    unchanged_accuracy = unchanged_correct / total
    gates = {
        "router_beats_unchanged_by_5_points": router_accuracy
        >= unchanged_accuracy + 0.05,
        "router_beats_matched_attention_by_3_points": router_accuracy
        >= attention_accuracy + 0.03,
        "all_domain_deltas_vs_unchanged_nonnegative": all(
            value >= 0 for value in domain_deltas.values()
        ),
        "complete_identity_coverage": all(
            complete(report) for report in (router, attention, prior)
        ),
        "trainable_parameters_within_5_percent": parameter_ratio <= 1.05,
    }
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "split": "development",
        "model_revision": router["model_revision"],
        "data_sha256": router["data_sha256"],
        "data_report_sha256": router["data_report_sha256"],
        "arms": {
            "router": {
                "correct": router_correct,
                "accuracy": router_accuracy,
                "report_sha256": sha256_file(args.router_report),
            },
            "matched_attention": {
                "correct": attention_correct,
                "accuracy": attention_accuracy,
                "report_sha256": sha256_file(args.attention_report),
            },
            "unchanged_second_pass": {
                "correct": unchanged_correct,
                "accuracy": unchanged_accuracy,
                "report_sha256": sha256_file(args.unchanged_report),
            },
            "prior_rank8_attention": {
                "correct": metric(prior)[0],
                "accuracy": metric(prior)[0] / total,
                "report_sha256": sha256_file(args.prior_treatment_report),
            },
        },
        "router_minus_unchanged_points": 100
        * (router_accuracy - unchanged_accuracy),
        "router_minus_matched_attention_points": 100
        * (router_accuracy - attention_accuracy),
        "domain_correct_count_deltas_vs_unchanged": domain_deltas,
        "parameter_ratio_router_to_attention": parameter_ratio,
        "gates": gates,
        "gate_pass": all(gates.values()),
        "holdout_authorized": all(gates.values()),
    }
    atomic_json(args.output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-report", type=Path, required=True)
    parser.add_argument("--attention-report", type=Path, required=True)
    parser.add_argument("--unchanged-report", type=Path, required=True)
    parser.add_argument("--prior-treatment-report", type=Path, required=True)
    parser.add_argument("--router-fit-report", type=Path, required=True)
    parser.add_argument("--attention-fit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args)
    print(json.dumps({"gate_pass": result["gate_pass"], "gates": result["gates"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

