#!/usr/bin/env python3
"""Apply the frozen MPR3 practical temporal-revision development gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EVAL_SCHEMA = "shohin-idr1-revision-evaluation-v1"
FIT_SCHEMA = "shohin-rme1-product-training-v1"
DATA_SCHEMA = "shohin-mpr2-revision-data-report-v1"
SEMANTIC_SCHEMA = "shohin-moe-semantic-repair-attribution-v1"


class MPR3ComparisonError(RuntimeError):
    """MPR3 result receipts do not satisfy the frozen comparison contract."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("status") != "complete":
        raise MPR3ComparisonError(f"incomplete MPR3 input: {path}")
    return value


def score(report: dict[str, Any]) -> int:
    return int(report["metrics"]["overall"]["generated_correct"])


def domains(report: dict[str, Any]) -> dict[str, int]:
    return {
        task: int(report["metrics"][task]["generated_correct"])
        for task in ("math500", "bbh_logic", "mbpp")
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise MPR3ComparisonError(f"refusing existing comparison: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def compare(args: argparse.Namespace) -> dict[str, Any]:
    evaluations = {
        name: load(getattr(args, f"{name}_report"))
        for name in ("aligned", "shuffled", "hidden", "owner")
    }
    if any(
        report.get("schema") != EVAL_SCHEMA
        or report.get("split") != "development"
        or int(report.get("full_row_count", -1)) != 1289
        for report in evaluations.values()
    ):
        raise MPR3ComparisonError("MPR3 evaluation geometry differs")
    fits = {
        name: load(getattr(args, f"{name}_fit"))
        for name in ("aligned", "shuffled", "hidden")
    }
    data = load(args.data_report)
    semantic = load(args.semantic_attribution)
    if (
        data.get("schema") != DATA_SCHEMA
        or data.get("holdout_used") is not False
        or data.get("complete_retention") is not True
        or int(data.get("owner_development_score", -1)) < 300
        or semantic.get("schema") != SEMANTIC_SCHEMA
    ):
        raise MPR3ComparisonError("MPR3 data or semantic receipt differs")
    expected_data = {
        "aligned": data["outputs"]["train_aligned"]["sha256"],
        "shuffled": data["outputs"]["train_shuffled"]["sha256"],
        "hidden": data["outputs"]["train_aligned"]["sha256"],
    }
    for arm, fit in fits.items():
        expected_control = "draft_unavailable" if arm == "hidden" else "normal"
        config = fit.get("rme1_config", {})
        if (
            fit.get("schema") != FIT_SCHEMA
            or int(fit.get("updates", -1)) != 256
            or int(fit.get("trainable_parameters", -1)) != 1_179_648
            or int(fit.get("protected_router_expert_trainables", -1)) != 0
            or fit.get("data_sha256") != expected_data[arm]
            or fit.get("rme1_draft_control") != expected_control
            or config.get("mode") != "shared"
            or int(config.get("controlled_layers", -1)) != 16
            or int(config.get("rank", -1)) != 18
        ):
            raise MPR3ComparisonError(f"MPR3 {arm} fit differs")
    scores = {name: score(report) for name, report in evaluations.items()}
    aligned_domains = domains(evaluations["aligned"])
    owner_domains = domains(evaluations["owner"])
    counts = semantic.get("counts", {})
    semantic_net = int(counts.get("remaining_possible_semantic_repairs", -1)) - int(
        counts.get("strict_breaks", -1)
    )
    gates = {
        "aligned_plus_39_over_owner": scores["aligned"] - scores["owner"] >= 39,
        "aligned_plus_13_over_shuffled": scores["aligned"] - scores["shuffled"] >= 13,
        "aligned_plus_13_over_hidden": scores["aligned"] - scores["hidden"] >= 13,
        "math_nonnegative": aligned_domains["math500"] >= owner_domains["math500"],
        "logic_nonnegative": aligned_domains["bbh_logic"] >= owner_domains["bbh_logic"],
        "code_nonnegative": aligned_domains["mbpp"] >= owner_domains["mbpp"],
        "semantic_net_at_least_13": semantic_net >= 13,
    }
    gate_pass = all(gates.values())
    payload = {
        "schema": "shohin-mpr3-development-comparison-v1",
        "status": "complete",
        "split": "development",
        "scores": scores,
        "margins": {
            "aligned_minus_owner": scores["aligned"] - scores["owner"],
            "aligned_minus_shuffled": scores["aligned"] - scores["shuffled"],
            "aligned_minus_hidden": scores["aligned"] - scores["hidden"],
        },
        "aligned_domains": aligned_domains,
        "owner_domains": owner_domains,
        "semantic_counts": counts,
        "semantic_net": semantic_net,
        "gates": gates,
        "gate_pass": gate_pass,
        "holdout_authorized": gate_pass,
        "decision": "open_one_sealed_holdout" if gate_pass else "close_practical_small_olmoe_route",
        "inputs": {
            name: sha256_file(path)
            for name, path in vars(args).items()
            if isinstance(path, Path) and path.exists()
        },
    }
    atomic_json(args.output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "aligned_report",
        "shuffled_report",
        "hidden_report",
        "owner_report",
        "aligned_fit",
        "shuffled_fit",
        "hidden_fit",
        "data_report",
        "semantic_attribution",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = compare(parser.parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
