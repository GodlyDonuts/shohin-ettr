#!/usr/bin/env python3
"""Apply the frozen CTSR1 development gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


class CTSR1ComparisonError(RuntimeError):
    pass


STATIC_SHARED_SHA256 = "03f076b2b866541ed336a2333e293a60d8e65a12df1c431d8166efe7f863d75d"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "complete":
        raise CTSR1ComparisonError(f"incomplete report: {path}")
    return value


def metric(report: dict[str, Any], domain: str = "overall") -> tuple[int, int]:
    value = report.get("metrics", {}).get(domain)
    if not isinstance(value, dict):
        raise CTSR1ComparisonError(f"missing metric: {domain}")
    correct, total = int(value.get("generated_correct", -1)), int(value.get("total", 0))
    if total <= 0 or not 0 <= correct <= total:
        raise CTSR1ComparisonError(f"invalid metric: {domain}")
    return correct, total


def complete_evaluation(report: dict[str, Any]) -> bool:
    return (
        report.get("schema") == "shohin-idr1-revision-evaluation-v1"
        and report.get("split") == "development"
        and report.get("full_row_count") == 1289
        and report.get("merged_from_shards") is True
        and int(report.get("shard_count", 0)) == 8
        and report.get("ecr_code_intervention") == "normal"
    )


def complete_fit(report: dict[str, Any], mode: str) -> bool:
    config = report.get("ctsr1_config", {})
    custody = report.get("sequence_custody", {})
    return (
        report.get("schema") == "shohin-ctsr1-product-training-v1"
        and report.get("updates") == 256
        and report.get("selected_rows") == 9651
        and report.get("charged_tokens") == 338620
        and report.get("trainable_parameters") == 1594752
        and report.get("adapter_macs_per_token_per_layer") == 488576
        and report.get("protected_router_expert_trainables") == 0
        and config.get("mode") == mode
        and config.get("controlled_layers") == 16
        and config.get("state_width") == 64
        and config.get("head_width") == 32
        and config.get("residual_rank") == 18
        and config.get("residual_alpha") == 18.0
        and config.get("router_scale") == 1.0
        and config.get("entropy_floor") == 0.80
        and config.get("collapse_weight") == 0.01
        and report.get("ctsr1_draft_control") == "normal"
        and custody.get("overflow_rows") == 0
        and custody.get("source_retention") == 1.0
        and custody.get("draft_retention") == 1.0
        and custody.get("target_retention") == 1.0
    )


def route_gate(report: dict[str, Any]) -> tuple[bool, dict[str, float]]:
    receipts = report.get("routing_receipts")
    if not isinstance(receipts, list) or len(receipts) != 8:
        return False, {}
    minimum_load = 1.0
    changed = tokens = 0.0
    for receipt in receipts:
        layers = receipt.get("layers") if isinstance(receipt, dict) else None
        if not isinstance(layers, list) or len(layers) != 16:
            return False, {}
        for layer in layers:
            values = [
                float(layer.get("load_entropy", float("nan"))),
                float(layer.get("mean_token_entropy_normalized", float("nan"))),
                float(layer.get("route_probability_l1_mean", float("nan"))),
                float(layer.get("mean_state_norm", float("nan"))),
                float(layer.get("mean_residual_norm", float("nan"))),
            ]
            if not all(math.isfinite(value) for value in values):
                return False, {}
            if int(layer.get("active_experts", 0)) != 64:
                return False, {}
            layer_tokens = int(layer.get("tokens", 0))
            if layer_tokens <= 0:
                return False, {}
            minimum_load = min(minimum_load, values[0])
            changed += float(layer.get("top1_change_rate", 0.0)) * layer_tokens
            tokens += layer_tokens
    mean_change = changed / tokens
    return minimum_load >= 0.80 and mean_change >= 0.01, {
        "minimum_load_entropy": minimum_load,
        "mean_top1_change_rate": mean_change,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def compare(args: argparse.Namespace) -> dict[str, Any]:
    reports = {
        "treatment": load(args.treatment_report),
        "temporal_shared": load(args.temporal_shared_report),
        "static_shared": load(args.static_shared_report),
    }
    if sha256_file(args.static_shared_report) != STATIC_SHARED_SHA256:
        raise CTSR1ComparisonError("static shared reference differs")
    if not all(complete_evaluation(report) for report in reports.values()):
        raise CTSR1ComparisonError("evaluation coverage differs")
    reference = reports["treatment"]
    for report in reports.values():
        for key in ("model_revision", "data_sha256", "data_report_sha256"):
            if report.get(key) != reference.get(key):
                raise CTSR1ComparisonError("evaluation inputs differ")
    fits = {
        "treatment": load(args.treatment_fit),
        "temporal_shared": load(args.temporal_shared_fit),
    }
    if not complete_fit(fits["treatment"], "temporal_router"):
        raise CTSR1ComparisonError("treatment fit receipt differs")
    if not complete_fit(fits["temporal_shared"], "temporal_shared"):
        raise CTSR1ComparisonError("temporal shared fit receipt differs")
    treatment, total = metric(reports["treatment"])
    temporal_shared = metric(reports["temporal_shared"])[0]
    static_shared = metric(reports["static_shared"])[0]
    if static_shared != 248:
        raise CTSR1ComparisonError("static shared score differs")
    domains = {
        domain: metric(reports["treatment"], domain)[0]
        for domain in ("math500", "bbh_logic", "mbpp")
    }
    routes_pass, route_summary = route_gate(reports["treatment"])
    gates = {
        "treatment_at_least_280": treatment >= 280,
        "treatment_beats_temporal_shared_by_26": treatment >= temporal_shared + 26,
        "treatment_beats_static_shared_by_13": treatment >= static_shared + 13,
        "domain_floors": domains["math500"] >= 55
        and domains["bbh_logic"] >= 180
        and domains["mbpp"] >= 5,
        "causal_state_changes_routes_without_collapse": routes_pass,
        "complete_receipts": True,
    }
    passed = all(gates.values())
    result = {
        "schema": "shohin-ctsr1-development-comparison-v1",
        "status": "complete",
        "split": "development",
        "arms": {
            name: {
                "correct": metric(report)[0],
                "accuracy": metric(report)[0] / total,
                "report_sha256": sha256_file(path),
            }
            for name, report, path in (
                ("treatment", reports["treatment"], args.treatment_report),
                ("temporal_shared", reports["temporal_shared"], args.temporal_shared_report),
                ("static_shared", reports["static_shared"], args.static_shared_report),
            )
        },
        "domain_correct": domains,
        "route_summary": route_summary,
        "charged_target_tokens_per_fit": 338620,
        "gates": gates,
        "causal_controls_authorized": passed,
        "holdout_authorized": False,
        "close_ctsr1_if_false": not passed,
    }
    atomic_json(args.output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "treatment_report", "temporal_shared_report", "static_shared_report",
        "treatment_fit", "temporal_shared_fit", "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    result = compare(parser.parse_args())
    print(json.dumps({"authorized": result["causal_controls_authorized"], "gates": result["gates"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

