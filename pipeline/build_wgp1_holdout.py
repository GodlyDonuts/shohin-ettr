#!/usr/bin/env python3
"""Open the single WGP1 confirmation from immutable RG-v4 held-out seeds."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from build_btt1_byte_supervision import compile_row as compile_btt1
from build_mltc1_lexical_supervision import compile_row as compile_mltc1
from build_pstc1_stack_supervision import compile_actions, execute_actions


SCHEMA = "shohin-wgp1-heldout-seed-report-v1"
FAMILIES = {"basic_arithmetic", "chain_sum", "decimal_arithmetic", "decimal_chain_sum", "products"}


class WGP1HoldoutError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def answer_fraction(value: object) -> Fraction:
    if not isinstance(value, str):
        raise WGP1HoldoutError("answer is not a string")
    try:
        return Fraction(value.strip().rstrip(".").replace(",", ""))
    except (ValueError, ZeroDivisionError) as error:
        raise WGP1HoldoutError("answer is not an exact fraction") from error


def existing_questions(paths: list[Path]) -> set[str]:
    values = set()
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                values.add(json.loads(line)["source_question_sha256"])
    return values


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            payload = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists():
        raise WGP1HoldoutError("refusing existing output root")
    if sha256_file(args.source) != args.expected_source_sha256:
        raise WGP1HoldoutError("holdout source SHA-256 differs")
    protected = existing_questions(args.protected)
    admitted, exclusions = [], []
    counts: Counter[str] = Counter()
    seen = set()
    for line in args.source.read_text().splitlines():
        if not line.strip():
            continue
        source = json.loads(line)
        family = source.get("family")
        if source.get("split") != "eval_seed" or family not in FAMILIES:
            continue
        counts[f"source:{family}"] += 1
        question = source.get("question")
        question_hash = hashlib.sha256(str(question).strip().encode()).hexdigest()
        if question_hash in protected or question_hash in seen:
            raise WGP1HoldoutError("holdout question overlaps or duplicates")
        seen.add(question_hash)
        try:
            actions, spans, _ = compile_actions(question, family)
            records, terminal, maximum_stack = execute_actions(actions, spans)
            if terminal != answer_fraction(source.get("answer")):
                raise WGP1HoldoutError("terminal differs from heldout answer")
            identity = hashlib.sha256(f"shohin-wgp1-holdout-v1\0{question_hash}".encode()).hexdigest()
            pstc = {
                "schema": "shohin-pstc1-stack-supervision-v1",
                "identity_sha256": identity,
                "source_question_sha256": question_hash,
                "split": "holdout",
                "family": family,
                "question": question,
                "number_spans": spans,
                "actions": actions,
                "action_count": len(actions),
                "maximum_stack": maximum_stack,
                "binary_record_count": len(records),
            }
            admitted.append(compile_btt1(compile_mltc1(pstc)))
            counts[f"admitted:{family}"] += 1
        except (ValueError, KeyError, TypeError) as error:
            exclusions.append({"source_question_sha256": question_hash, "family": family, "reason": str(error)})
            counts[f"excluded:{family}"] += 1
    for family in sorted(FAMILIES):
        if counts[f"source:{family}"] != 500:
            raise WGP1HoldoutError(f"{family} source population differs")
        if counts[f"admitted:{family}"] < 495:
            raise WGP1HoldoutError(f"{family} admission below frozen 99% floor")
    args.output_root.mkdir(parents=True)
    data_path = args.output_root / "holdout.jsonl"
    exclusion_path = args.output_root / "exclusions.jsonl"
    data_sha = atomic_jsonl(data_path, admitted)
    exclusion_sha = atomic_jsonl(exclusion_path, exclusions)
    report = {
        "schema": SCHEMA, "status": "complete", "holdout_used": True,
        "source": str(args.source.resolve()), "source_sha256": args.expected_source_sha256,
        "protected": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in args.protected],
        "counts": dict(sorted(counts.items())), "rows": len(admitted), "exclusions": len(exclusions),
        "data": str(data_path.resolve()), "data_sha256": data_sha,
        "exclusion_path": str(exclusion_path.resolve()), "exclusion_sha256": exclusion_sha,
        "zero_question_overlap": True, "extensional_parity": True,
    }
    (args.output_root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--protected", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
