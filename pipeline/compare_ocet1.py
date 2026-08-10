#!/usr/bin/env python3
"""Merge OCET1 shards and apply the frozen on-policy transduction gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path


SCHEMA = "shohin-ocet1-comparison-v1"
EVAL_SCHEMA = "shohin-rift1-fixed-point-evaluation-v1"
ISET_CORRECT = 1838


class OCET1ComparisonError(RuntimeError):
    """OCET1 shard custody or geometry differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_arm(paths: list[Path], arm: str) -> tuple[list[dict], dict]:
    if len(paths) != 8:
        raise OCET1ComparisonError(f"{arm} shard count differs")
    rows, inputs, shards = [], [], set()
    receipt = None
    for path in paths:
        report = json.loads(path.read_text())
        if (
            report.get("schema") != EVAL_SCHEMA
            or report.get("status") != "complete"
            or report.get("holdout_used") is not False
            or report.get("arm") != arm
            or int(report.get("shard_count", -1)) != 8
        ):
            raise OCET1ComparisonError(f"{arm} report differs")
        shard = int(report["shard_index"])
        if shard in shards:
            raise OCET1ComparisonError(f"{arm} duplicate shard")
        shards.add(shard)
        bound = (
            report["checkpoint_sha256"],
            report["data_sha256"],
            report["data_report_sha256"],
        )
        if receipt is None:
            receipt = bound
        elif receipt != bound:
            raise OCET1ComparisonError(f"{arm} receipt differs")
        report_rows = report.get("results")
        if not isinstance(report_rows, list) or len(report_rows) != int(report["row_count"]):
            raise OCET1ComparisonError(f"{arm} row count differs")
        rows.extend(report_rows)
        inputs.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    identities = [str(row["identity_sha256"]) for row in rows]
    if shards != set(range(8)) or len(rows) != 1908 or len(set(identities)) != 1908:
        raise OCET1ComparisonError(f"{arm} coverage differs")
    return rows, {"receipt": receipt, "inputs": inputs}


def summarize(rows: list[dict]) -> dict:
    totals = Counter()
    family, member = defaultdict(Counter), defaultdict(Counter)
    actions = Counter()
    for row in rows:
        totals["rows"] += 1
        for metric in ("proposal_correct", "commit_valid", "final_correct"):
            value = int(bool(row[metric]))
            totals[metric] += value
            family[str(row["corruption_family"])][metric] += value
            member[str(row["pair_member"])][metric] += value
        family[str(row["corruption_family"])]["rows"] += 1
        member[str(row["pair_member"])]["rows"] += 1
        totals["commit_errors"] += int(row.get("commit_error") is not None)
        totals["max_token_exhausted"] += int(bool(row.get("max_token_exhausted")))
        actions[str(row.get("commit_action"))] += 1
    return {
        **dict(totals),
        "final_accuracy": totals["final_correct"] / totals["rows"],
        "commit_valid_accuracy": totals["commit_valid"] / totals["rows"],
        "family": {name: dict(values) for name, values in family.items()},
        "member": {name: dict(values) for name, values in member.items()},
        "commit_actions": dict(actions),
    }


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise OCET1ComparisonError("OCET1 output exists")
    loaded = {}
    receipts = {}
    for arm, paths in (("aligned", args.aligned), ("swapped", args.swapped), ("hidden", args.hidden)):
        rows, receipts[arm] = load_arm(paths, arm)
        loaded[arm] = rows
    aligned_by_id = {str(row["identity_sha256"]): row for row in loaded["aligned"]}
    for arm in ("swapped", "hidden"):
        by_id = {str(row["identity_sha256"]): row for row in loaded[arm]}
        if set(aligned_by_id) != set(by_id):
            raise OCET1ComparisonError(f"aligned/{arm} identities differ")
        for identity, left in aligned_by_id.items():
            right = by_id[identity]
            for field in ("pair_identity_sha256", "pair_member", "corruption_family"):
                if left[field] != right[field]:
                    raise OCET1ComparisonError(f"aligned/{arm} metadata differs")
        if receipts["aligned"]["receipt"][1:] != receipts[arm]["receipt"][1:]:
            raise OCET1ComparisonError(f"aligned/{arm} data receipt differs")
    summaries = {arm: summarize(rows) for arm, rows in loaded.items()}
    aligned = summaries["aligned"]
    if set(aligned["family"]) != {"numeric_final", "choice_final"} or set(
        aligned["member"]
    ) != {"clean", "fault"}:
        raise OCET1ComparisonError("OCET1 group geometry differs")
    clean = aligned["member"]["clean"]
    fault = aligned["member"]["fault"]
    choice = aligned["family"]["choice_final"]
    gate = {
        "aligned_final_ge_1874": aligned["final_correct"] >= 1874,
        "commit_valid_ge_0_95": aligned["commit_valid_accuracy"] >= 0.95,
        "choice_final_ge_220": choice["final_correct"] >= 220,
        "clean_final_ge_945": clean["final_correct"] >= 945,
        "fault_final_ge_859": fault["final_correct"] >= 859,
        "aligned_minus_iset_ge_13": aligned["final_correct"] - ISET_CORRECT >= 13,
        "aligned_minus_swapped_ge_13": (
            aligned["final_correct"] - summaries["swapped"]["final_correct"] >= 13
        ),
        "aligned_minus_hidden_ge_13": (
            aligned["final_correct"] - summaries["hidden"]["final_correct"] >= 13
        ),
        "zero_exhaustion": aligned["max_token_exhausted"] == 0,
    }
    passed = all(gate.values())
    payload = {
        "schema": SCHEMA,
        "status": "pass" if passed else "fail",
        "holdout_used": False,
        "holdout_authorized": passed,
        "gate": gate,
        **summaries,
        "iset_correct": ISET_CORRECT,
        "aligned_minus_iset": aligned["final_correct"] - ISET_CORRECT,
        "aligned_minus_swapped": aligned["final_correct"] - summaries["swapped"]["final_correct"],
        "aligned_minus_hidden": aligned["final_correct"] - summaries["hidden"]["final_correct"],
        "receipts": receipts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned", type=Path, nargs="+", required=True)
    parser.add_argument("--swapped", type=Path, nargs="+", required=True)
    parser.add_argument("--hidden", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
