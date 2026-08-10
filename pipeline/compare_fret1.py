#!/usr/bin/env python3
"""Merge FRET1 shards and apply the frozen always-rewrite ceiling gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path


SCHEMA = "shohin-fret1-comparison-v1"
EVAL_SCHEMA = "shohin-fret1-always-rewrite-evaluation-v1"


class FRET1ComparisonError(RuntimeError):
    """FRET1 shard custody or geometry differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_arm(paths: list[Path], arm: str) -> tuple[list[dict], dict]:
    if len(paths) != 8:
        raise FRET1ComparisonError(f"{arm} shard count differs")
    rows = []
    receipt = None
    shard_indices = set()
    inputs = []
    for path in paths:
        report = json.loads(path.read_text())
        if (
            report.get("schema") != EVAL_SCHEMA
            or report.get("status") != "complete"
            or report.get("holdout_used") is not False
            or report.get("arm") != arm
            or int(report.get("shard_count", -1)) != 8
        ):
            raise FRET1ComparisonError(f"{arm} report differs")
        shard_index = int(report["shard_index"])
        if shard_index in shard_indices:
            raise FRET1ComparisonError(f"{arm} duplicate shard")
        shard_indices.add(shard_index)
        bound = (
            report["checkpoint_sha256"],
            report["data_sha256"],
            report["data_report_sha256"],
        )
        if receipt is None:
            receipt = bound
        elif receipt != bound:
            raise FRET1ComparisonError(f"{arm} receipt differs")
        report_rows = report.get("results")
        if not isinstance(report_rows, list) or len(report_rows) != int(report["row_count"]):
            raise FRET1ComparisonError(f"{arm} row count differs")
        rows.extend(report_rows)
        inputs.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    if shard_indices != set(range(8)) or len(rows) != 1908:
        raise FRET1ComparisonError(f"{arm} coverage differs")
    identities = [str(row["identity_sha256"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise FRET1ComparisonError(f"{arm} duplicate identity")
    return rows, {"receipt": receipt, "inputs": inputs}


def summarize(rows: list[dict]) -> dict:
    family = defaultdict(Counter)
    member = defaultdict(Counter)
    totals = Counter()
    for row in rows:
        for metric in ("pointer_exact", "replacement_exact", "program_exact", "execution_correct"):
            value = int(bool(row[metric]))
            totals[metric] += value
            family[str(row["corruption_family"])][metric] += value
            member[str(row["pair_member"])][metric] += value
        totals["rows"] += 1
        family[str(row["corruption_family"])]["rows"] += 1
        member[str(row["pair_member"])]["rows"] += 1
        totals["execution_errors"] += int(row.get("execution_error") is not None)
        totals["max_token_exhausted"] += int(bool(row.get("max_token_exhausted")))
        totals["copy_characters"] += int(row["copy_characters"])
        totals["draft_characters"] += int(row["draft_characters"])
    return {
        **dict(totals),
        "family": {name: dict(values) for name, values in family.items()},
        "member": {name: dict(values) for name, values in member.items()},
        "program_accuracy": totals["program_exact"] / totals["rows"],
        "execution_accuracy": totals["execution_correct"] / totals["rows"],
        "copy_rate": totals["copy_characters"] / totals["draft_characters"],
    }


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FRET1ComparisonError("FRET1 output exists")
    aligned_rows, aligned_receipt = load_arm(args.aligned, "aligned")
    hidden_rows, hidden_receipt = load_arm(args.hidden, "hidden")
    aligned_by_id = {str(row["identity_sha256"]): row for row in aligned_rows}
    hidden_by_id = {str(row["identity_sha256"]): row for row in hidden_rows}
    if set(aligned_by_id) != set(hidden_by_id):
        raise FRET1ComparisonError("aligned/hidden identities differ")
    for identity in aligned_by_id:
        left, right = aligned_by_id[identity], hidden_by_id[identity]
        for field in ("pair_identity_sha256", "pair_member", "corruption_family"):
            if left[field] != right[field]:
                raise FRET1ComparisonError("aligned/hidden metadata differs")
    if aligned_receipt["receipt"][1:] != hidden_receipt["receipt"][1:]:
        raise FRET1ComparisonError("aligned/hidden data receipt differs")
    aligned = summarize(aligned_rows)
    hidden = summarize(hidden_rows)
    families = aligned["family"]
    members = aligned["member"]
    expected_families = {"numeric_final", "choice_final"}
    if set(families) != expected_families or set(members) != {"clean", "fault"}:
        raise FRET1ComparisonError("FRET1 group geometry differs")
    family_program_gate = all(
        values["program_exact"] / values["rows"] >= 0.95 for values in families.values()
    )
    gate = {
        "aligned_program_ge_0_95": aligned["program_accuracy"] >= 0.95,
        "each_family_program_ge_0_95": family_program_gate,
        "aligned_execution_ge_0_95": aligned["execution_accuracy"] >= 0.95,
        "clean_execution_ge_0_99": members["clean"]["execution_correct"] / members["clean"]["rows"] >= 0.99,
        "fault_execution_ge_0_90": members["fault"]["execution_correct"] / members["fault"]["rows"] >= 0.90,
        "aligned_minus_hidden_ge_13": aligned["execution_correct"] - hidden["execution_correct"] >= 13,
        "zero_errors": aligned["execution_errors"] == 0,
        "zero_exhaustion": aligned["max_token_exhausted"] == 0,
        "copy_rate_ge_0_95": aligned["copy_rate"] >= 0.95,
    }
    payload = {
        "schema": SCHEMA,
        "status": "pass" if all(gate.values()) else "fail",
        "holdout_used": False,
        "holdout_authorized": False,
        "gate": gate,
        "aligned": aligned,
        "hidden": hidden,
        "aligned_minus_hidden_execution": aligned["execution_correct"] - hidden["execution_correct"],
        "receipts": {"aligned": aligned_receipt, "hidden": hidden_receipt},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned", type=Path, nargs="+", required=True)
    parser.add_argument("--hidden", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
