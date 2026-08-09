#!/usr/bin/env python3
"""Apply the frozen MPR1 practical MoE revision development gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-mpr1-development-comparison-v1"
EVAL_SCHEMA = "shohin-idr1-revision-evaluation-v1"
FIT_SCHEMA = "shohin-rme1-product-training-v1"


class MPR1ComparisonError(RuntimeError):
    """MPR1 evidence differs from its frozen contract."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("status") != "complete":
        raise MPR1ComparisonError(f"incomplete report: {path}")
    return value


def score(report: dict[str, Any], domain: str = "overall") -> tuple[int, int]:
    value = report.get("metrics", {}).get(domain)
    if not isinstance(value, dict) or int(value.get("total", 0)) <= 0:
        raise MPR1ComparisonError(f"missing score: {domain}")
    return int(value["generated_correct"]), int(value["total"])


def complete_eval(report: dict[str, Any]) -> bool:
    return (
        report.get("schema") == EVAL_SCHEMA
        and report.get("split") == "development"
        and report.get("full_row_count") == 1289
        and report.get("merged_from_shards") is True
        and int(report.get("shard_count", 0)) >= 2
    )


def complete_fit(report: dict[str, Any], control: str, data_sha: str) -> bool:
    config = report.get("rme1_config", {})
    return (
        report.get("schema") == FIT_SCHEMA
        and report.get("status") == "complete"
        and report.get("updates") == 256
        and report.get("batch_size") == 1
        and report.get("gradient_accumulation") == 8
        and report.get("max_sequence_length") == 4096
        and report.get("learning_rate") == 2e-5
        and report.get("seed") == 2026080901
        and report.get("data_seed") == 2026080814
        and report.get("data_sha256") == data_sha
        and report.get("trainable_parameters") == 1_179_648
        and report.get("protected_router_expert_trainables") == 0
        and report.get("rme1_draft_control") == control
        and config.get("mode") == "shared"
        and config.get("controlled_layers") == 16
        and config.get("rank") == 18
        and config.get("alpha") == 18.0
    )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise MPR1ComparisonError(f"refusing existing output: {path}")
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
        name: load(path)
        for name, path in (
            ("aligned", args.aligned_report),
            ("shuffled", args.shuffled_report),
            ("hidden", args.hidden_report),
            ("unchanged", args.unchanged_report),
        )
    }
    if any(not complete_eval(report) for report in reports.values()):
        raise MPR1ComparisonError("MPR1 evaluation coverage differs")
    for key in ("model_root", "model_revision", "data_sha256", "data_report_sha256"):
        values = {report.get(key) for report in reports.values()}
        if len(values) != 1:
            raise MPR1ComparisonError(f"MPR1 evaluation setting differs: {key}")

    data = load(args.data_report)
    if (
        data.get("schema") != "shohin-mpr1-revision-data-report-v1"
        or data.get("complete_retention") is not True
        or data.get("holdout_used") is not False
    ):
        raise MPR1ComparisonError("MPR1 data custody differs")
    outputs = data.get("outputs", {})
    aligned_sha = str(outputs.get("aligned", {}).get("sha256", ""))
    shuffled_sha = str(outputs.get("shuffled", {}).get("sha256", ""))
    fits = {
        "aligned": load(args.aligned_fit),
        "shuffled": load(args.shuffled_fit),
        "hidden": load(args.hidden_fit),
    }
    if not complete_fit(fits["aligned"], "normal", aligned_sha):
        raise MPR1ComparisonError("MPR1 aligned fit differs")
    if not complete_fit(fits["shuffled"], "normal", shuffled_sha):
        raise MPR1ComparisonError("MPR1 shuffled fit differs")
    if not complete_fit(fits["hidden"], "zero_attention", aligned_sha):
        raise MPR1ComparisonError("MPR1 hidden fit differs")
    for key in (
        "model_root", "model_revision", "updates", "charged_tokens",
        "selected_rows", "trainable_parameters", "adapter_macs_per_token_per_layer",
        "seed", "data_seed",
    ):
        if len({fit.get(key) for fit in fits.values()}) != 1:
            raise MPR1ComparisonError(f"MPR1 matched fit differs: {key}")

    scores = {name: score(report)[0] for name, report in reports.items()}
    if any(score(report)[1] != 1289 for report in reports.values()):
        raise MPR1ComparisonError("MPR1 totals differ")
    domains = {
        domain: score(reports["aligned"], domain)[0]
        for domain in ("math500", "bbh_logic", "mbpp")
    }
    semantic = load(args.semantic_attribution)
    if semantic.get("schema") != "shohin-moe-semantic-repair-attribution-v1":
        raise MPR1ComparisonError("MPR1 semantic attribution differs")
    counts = semantic.get("counts", {})
    semantic_net = int(counts.get("remaining_possible_semantic_repairs", -1)) - int(
        counts.get("strict_breaks", -1)
    )
    gates = {
        "aligned_at_least_230": scores["aligned"] >= 230,
        "aligned_beats_unchanged_by_39": scores["aligned"] >= scores["unchanged"] + 39,
        "aligned_beats_shuffled_by_13": scores["aligned"] >= scores["shuffled"] + 13,
        "aligned_beats_hidden_by_13": scores["aligned"] >= scores["hidden"] + 13,
        "math_at_least_40": domains["math500"] >= 40,
        "logic_at_least_145": domains["bbh_logic"] >= 145,
        "code_at_least_5": domains["mbpp"] >= 5,
        "semantic_net_at_least_13": semantic_net >= 13,
        "complete_matched_receipts": True,
    }
    passed = all(gates.values())
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "split": "development",
        "scores": scores,
        "domain_correct": domains,
        "semantic_counts": counts,
        "semantic_net": semantic_net,
        "margins": {
            "aligned_minus_unchanged": scores["aligned"] - scores["unchanged"],
            "aligned_minus_shuffled": scores["aligned"] - scores["shuffled"],
            "aligned_minus_hidden": scores["aligned"] - scores["hidden"],
        },
        "charged_target_tokens_per_fit": fits["aligned"]["charged_tokens"],
        "trainable_parameters_per_fit": 1_179_648,
        "gates": gates,
        "gate_pass": passed,
        "holdout_authorized": passed,
        "decision": "open_one_sealed_mpr1_holdout" if passed else "close_exact_mpr1",
        "inputs": {
            "data_report_sha256": sha256_file(args.data_report),
            "semantic_attribution_sha256": sha256_file(args.semantic_attribution),
            **{f"{name}_report_sha256": sha256_file(path) for name, path in (
                ("aligned", args.aligned_report), ("shuffled", args.shuffled_report),
                ("hidden", args.hidden_report), ("unchanged", args.unchanged_report),
            )},
        },
    }
    atomic_json(args.output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "aligned_report", "shuffled_report", "hidden_report", "unchanged_report",
        "aligned_fit", "shuffled_fit", "hidden_fit", "data_report",
        "semantic_attribution", "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    result = compare(parser.parse_args())
    print(json.dumps({"gate_pass": result["gate_pass"], "gates": result["gates"]}, indent=2))
    return 0 if result["gate_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

