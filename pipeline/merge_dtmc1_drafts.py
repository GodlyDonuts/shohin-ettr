#!/usr/bin/env python3
"""Verify and merge all immutable DTMC1 draft shards with typed targets."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

SHARD_SCHEMA = "shohin-dtmc1-model-owned-draft-shard-v1"
MERGED_SCHEMA = "shohin-dtmc1-model-owned-draft-corpus-v1"


class DTMC1MergeError(ValueError):
    """Draft shards or their typed-target join differ from the frozen corpus."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise DTMC1MergeError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise DTMC1MergeError(f"refusing existing report: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.shard_count != 8:
        raise DTMC1MergeError("shard count differs")
    if sha256_file(args.program_data) != args.expected_program_sha256:
        raise DTMC1MergeError("typed target corpus SHA-256 differs")
    programs = load_jsonl(args.program_data)
    if len(programs) != 6333:
        raise DTMC1MergeError("typed target population differs")
    identities = [str(row["identity_sha256"]) for row in programs]
    if len(set(identities)) != len(identities):
        raise DTMC1MergeError("typed target identities are not unique")

    drafts: dict[str, dict[str, Any]] = {}
    shard_receipts = []
    for shard_index in range(args.shard_count):
        shard = args.shard_root / f"shard_{shard_index:02d}.jsonl"
        report_path = args.shard_root / f"shard_{shard_index:02d}.report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows = load_jsonl(shard)
        expected_rows = sum(
            index % args.shard_count == shard_index for index in range(len(programs))
        )
        if (
            report.get("schema") != SHARD_SCHEMA
            or report.get("status") != "complete"
            or report.get("public_test_opened") is not False
            or report.get("holdout_used") is not False
            or report.get("shard_index") != shard_index
            or report.get("shard_count") != args.shard_count
            or report.get("rows") != expected_rows
            or report.get("output_sha256") != sha256_file(shard)
            or report.get("checkpoint_sha256") != args.expected_checkpoint_sha256
            or report.get("data_sha256") != args.expected_direct_sha256
            or len(rows) != expected_rows
        ):
            raise DTMC1MergeError(f"shard {shard_index} receipt differs")
        for row in rows:
            identity = str(row["identity_sha256"])
            if (
                row.get("schema") != SHARD_SCHEMA
                or row.get("shard_index") != shard_index
                or row.get("shard_count") != args.shard_count
                or identity in drafts
            ):
                raise DTMC1MergeError(f"shard {shard_index} row differs")
            drafts[identity] = row
        shard_receipts.append(
            {
                "shard_index": shard_index,
                "data_sha256": sha256_file(shard),
                "report_sha256": sha256_file(report_path),
                "rows": len(rows),
                "elapsed_seconds": report["elapsed_seconds"],
            }
        )
    if set(drafts) != set(identities):
        raise DTMC1MergeError("draft identity coverage differs")

    merged = []
    draft_correct = 0
    exhausted = 0
    generated_tokens = 0
    for index, program in enumerate(programs):
        identity = identities[index]
        draft = drafts[identity]
        if (
            index % args.shard_count != draft["shard_index"]
            or draft["original_question"] != program["original_question"]
            or str(draft["gold_answer"]) != str(program["gold_answer"])
            or draft["register_depth"] != program["register_depth"]
        ):
            raise DTMC1MergeError("draft-to-target join differs")
        draft_correct += int(draft["draft_correct"])
        exhausted += int(draft["exhausted"])
        generated_tokens += int(draft["generated_tokens"])
        merged.append(
            {
                "schema": MERGED_SCHEMA,
                "identity_sha256": identity,
                "original_question": program["original_question"],
                "gold_answer": program["gold_answer"],
                "gold_program": program["gold_program"],
                "register_depth": program["register_depth"],
                "draft": draft["draft"],
                "draft_prediction": draft["draft_prediction"],
                "draft_correct": draft["draft_correct"],
                "generated_tokens": draft["generated_tokens"],
                "exhausted": draft["exhausted"],
            }
        )
    atomic_jsonl(args.output, merged)
    by_depth = Counter()
    by_depth_correct = Counter()
    for row in merged:
        depth = str(row["register_depth"])
        by_depth[depth] += 1
        by_depth_correct[depth] += int(row["draft_correct"])
    report = {
        "schema": MERGED_SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "public_test_opened": False,
        "rows": len(merged),
        "unique_identities": len(set(identities)),
        "draft_correct": draft_correct,
        "draft_accuracy": draft_correct / len(merged),
        "exhausted": exhausted,
        "generated_tokens": generated_tokens,
        "by_depth_rows": dict(sorted(by_depth.items())),
        "by_depth_correct": dict(sorted(by_depth_correct.items())),
        "program_data_sha256": args.expected_program_sha256,
        "direct_data_sha256": args.expected_direct_sha256,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "shards": shard_receipts,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
    }
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--program-data", type=Path, required=True)
    parser.add_argument("--expected-program-sha256", required=True)
    parser.add_argument("--expected-direct-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
