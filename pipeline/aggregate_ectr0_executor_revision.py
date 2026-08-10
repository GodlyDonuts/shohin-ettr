#!/usr/bin/env python3
"""Aggregate the frozen ECTR0 receipt arms and apply prospective gates."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-ectr0-executor-conditioned-revision-v1"
CONTROLS = ("aligned", "receipt_absent", "receipt_shuffled")


class ECTR0AggregateError(RuntimeError):
    """ECTR0 shards or their frozen boundaries differ."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_arm(paths: list[Path], control: str) -> dict[str, Any]:
    if not paths:
        raise ECTR0AggregateError(f"{control} has no reports")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    immutable = (
        "model_revision",
        "adapter_checkpoint_sha256",
        "data_sha256",
        "ctf_report_sha256",
        "seed",
        "max_new_tokens",
        "max_sequence_length",
        "batch_size",
        "shard_count",
    )
    reference = reports[0]
    for report in reports:
        if (
            report.get("schema") != SCHEMA
            or report.get("status") != "complete"
            or report.get("control") != control
            or report.get("holdout_used") is not False
            or report.get("public_test_opened") is not False
        ):
            raise ECTR0AggregateError(f"{control} report boundary differs")
        if any(report.get(key) != reference.get(key) for key in immutable):
            raise ECTR0AggregateError(f"{control} immutable receipt differs")
    shard_count = int(reference["shard_count"])
    if len(reports) != shard_count or {int(report["shard_index"]) for report in reports} != set(range(shard_count)):
        raise ECTR0AggregateError(f"{control} shard coverage differs")
    details = [detail for report in reports for detail in report["details"]]
    identities = [str(detail["identity_sha256"]) for detail in details]
    if len(details) != 666 or len(set(identities)) != 666:
        raise ECTR0AggregateError(f"{control} identity coverage differs")
    counts: Counter[str] = Counter()
    for report in reports:
        counts.update({key: int(value) for key, value in report["counts"].items()})
    return {
        "control": control,
        "reports": [str(path.resolve()) for path in paths],
        "report_sha256": [sha256_file(path) for path in paths],
        "counts": dict(sorted(counts.items())),
        "details": {str(detail["identity_sha256"]): detail for detail in details},
        "immutable": {key: reference[key] for key in immutable},
        "generated_tokens": sum(int(report["generated_tokens"]) for report in reports),
        "elapsed_seconds_sum": sum(float(report["elapsed_seconds"]) for report in reports),
        "peak_gpu_memory_bytes_max": max(int(report["peak_gpu_memory_bytes"]) for report in reports),
        "max_input_tokens": max(int(report["max_input_tokens"]) for report in reports),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise ECTR0AggregateError(f"refusing to replace output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    arms = {
        "aligned": load_arm(args.aligned_report, "aligned"),
        "receipt_absent": load_arm(args.absent_report, "receipt_absent"),
        "receipt_shuffled": load_arm(args.shuffled_report, "receipt_shuffled"),
    }
    immutable = arms["aligned"]["immutable"]
    if any(arm["immutable"] != immutable for arm in arms.values()):
        raise ECTR0AggregateError("cross-arm immutable receipt differs")
    identity_sets = [set(arm["details"]) for arm in arms.values()]
    if any(identity_set != identity_sets[0] for identity_set in identity_sets[1:]):
        raise ECTR0AggregateError("cross-arm identity coverage differs")
    aligned = arms["aligned"]["counts"]
    absent = arms["receipt_absent"]["counts"]
    shuffled = arms["receipt_shuffled"]["counts"]
    direct_correct = int(aligned["direct_correct"])
    if direct_correct != 487 or any(int(arm["counts"]["direct_correct"]) != direct_correct for arm in arms.values()):
        raise ECTR0AggregateError("frozen direct-owner baseline differs")
    aligned_correct = int(aligned["correct"])
    gates = {
        "aligned_at_least_500": aligned_correct >= 500,
        "aligned_minus_absent_at_least_13": aligned_correct - int(absent["correct"]) >= 13,
        "aligned_minus_shuffled_at_least_13": aligned_correct - int(shuffled["correct"]) >= 13,
        "repair_minus_break_at_least_13": int(aligned["repairs"]) - int(aligned["breaks"]) >= 13,
        "explicit_final_at_least_650": int(aligned["explicit_final"]) >= 650,
    }
    compact_arms = {
        name: {key: value for key, value in arm.items() if key != "details"}
        for name, arm in arms.items()
    }
    report = {
        "schema": "shohin-ectr0-aggregate-v1",
        "status": "complete",
        "holdout_used": False,
        "public_test_opened": False,
        "gate_pass": all(gates.values()),
        "gates": gates,
        "deltas": {
            "aligned_minus_direct": aligned_correct - direct_correct,
            "aligned_minus_absent": aligned_correct - int(absent["correct"]),
            "aligned_minus_shuffled": aligned_correct - int(shuffled["correct"]),
            "aligned_repairs_minus_breaks": int(aligned["repairs"]) - int(aligned["breaks"]),
        },
        "arms": compact_arms,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-report", type=Path, action="append", required=True)
    parser.add_argument("--absent-report", type=Path, action="append", required=True)
    parser.add_argument("--shuffled-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
