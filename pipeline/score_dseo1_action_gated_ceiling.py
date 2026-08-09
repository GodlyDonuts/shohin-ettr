#!/usr/bin/env python3
"""Score the deterministic KEEP-copy/FIX-rewrite ceiling from DSEO1 outputs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any


REPORT_SCHEMAS = {
    "shohin-dseo1-paired-evaluation-v1",
    "shohin-dseo1-paired-evaluation-merged-v1",
}
OUTPUT_SCHEMA = "shohin-dsec0-action-gated-ceiling-v1"


class DSEC0Error(RuntimeError):
    """The frozen DSEO1 result cannot support the requested ceiling."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_report(path: Path, arm: str) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if (
        report.get("schema") not in REPORT_SCHEMAS
        or report.get("status") != "complete"
        or report.get("arm") != arm
        or report.get("holdout_used") is True
        or int(report.get("row_count", -1)) != len(report.get("results", []))
    ):
        raise DSEC0Error(f"DSEO1 {arm} report differs")
    return report


def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    family: dict[str, Counter[str]] = defaultdict(Counter)
    member: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        correct = bool(row[key])
        counts["rows"] += 1
        counts["correct"] += int(correct)
        for groups, name in (
            (family, str(row["corruption_family"])),
            (member, str(row["pair_member"])),
        ):
            groups[name]["rows"] += 1
            groups[name]["correct"] += int(correct)

    def finalize(counter: Counter[str]) -> dict[str, Any]:
        total = int(counter["rows"])
        correct = int(counter["correct"])
        return {"rows": total, "correct": correct, "accuracy": correct / total}

    return {
        **finalize(counts),
        "family": {name: finalize(value) for name, value in sorted(family.items())},
        "member": {name: finalize(value) for name, value in sorted(member.items())},
    }


def score(aligned: dict[str, Any], final_only: dict[str, Any]) -> dict[str, Any]:
    final_rows = {
        str(row["identity_sha256"]): row for row in final_only["results"]
    }
    if len(final_rows) != len(final_only["results"]):
        raise DSEC0Error("final-only identities are not unique")
    rows = []
    for row in aligned["results"]:
        identity = str(row["identity_sha256"])
        control = final_rows.get(identity)
        if control is None or any(
            row.get(name) != control.get(name)
            for name in (
                "pair_identity_sha256",
                "pair_member",
                "corruption_family",
                "gold_answer",
            )
        ):
            raise DSEC0Error("aligned/final-only identity geometry differs")
        member = str(row["pair_member"])
        predicted = row.get("predicted_action")
        observed = member == "clean" if predicted == "<KEEP>" else bool(row["answer_correct"])
        oracle = member == "clean" if member == "clean" else bool(row["answer_correct"])
        rows.append(
            {
                "identity_sha256": identity,
                "pair_identity_sha256": row["pair_identity_sha256"],
                "pair_member": member,
                "corruption_family": row["corruption_family"],
                "predicted_action": predicted,
                "generated_answer_correct": bool(row["answer_correct"]),
                "observed_action_gated_correct": observed,
                "oracle_action_gated_correct": oracle,
                "final_only_correct": bool(control["answer_correct"]),
            }
        )
    if len(rows) != len(final_rows):
        raise DSEC0Error("aligned/final-only row count differs")
    observed = summarize(rows, "observed_action_gated_correct")
    oracle = summarize(rows, "oracle_action_gated_correct")
    baseline = summarize(rows, "final_only_correct")
    return {
        "rows": rows,
        "metrics": {
            "observed_action_gated": observed,
            "oracle_action_gated": oracle,
            "final_only": baseline,
        },
        "margins": {
            "observed_minus_final_only_answers": observed["correct"] - baseline["correct"],
            "oracle_minus_final_only_answers": oracle["correct"] - baseline["correct"],
        },
        "gates": {
            "observed_beats_final_only_by_13": observed["correct"] - baseline["correct"] >= 13,
            "observed_fault_repair_ge_0_90": observed["member"]["fault"]["accuracy"] >= 0.90,
            "oracle_fault_repair_ge_0_90": oracle["member"]["fault"]["accuracy"] >= 0.90,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise DSEC0Error("output exists")
    aligned = load_report(args.aligned, "aligned")
    final_only = load_report(args.final_only, "final_only")
    scored = score(aligned, final_only)
    passed = all(scored["gates"].values())
    report = {
        "schema": OUTPUT_SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "inputs": {
            "aligned": {"path": str(args.aligned.resolve()), "sha256": sha256_file(args.aligned)},
            "final_only": {"path": str(args.final_only.resolve()), "sha256": sha256_file(args.final_only)},
        },
        **scored,
        "passed": passed,
        "decision": "implement_binary_action_gate" if passed else "kill_binary_action_gate",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--final-only", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(json.dumps({"metrics": report["metrics"], "margins": report["margins"], "gates": report["gates"], "passed": report["passed"]}, sort_keys=True))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
