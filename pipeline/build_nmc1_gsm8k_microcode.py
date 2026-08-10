#!/usr/bin/env python3
"""Build source-disjoint GSM8K result-free microcode and direct-control rows."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re

from natural_microcode_program import (
    NaturalMicrocodeError,
    action_count,
    compile_gsm8k_answer,
    execute_fraction,
    register_depth,
    render_program,
    result_fields_absent,
)

SCHEMA = "shohin-nmc1-gsm8k-data-v1"
REPORT_SCHEMA = "shohin-nmc1-gsm8k-data-report-v1"


class NMC1DataError(ValueError):
    """NMC1 source custody or data invariant differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_question(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def identity(text: str) -> str:
    return hashlib.sha256(normalized_question(text).encode()).hexdigest()


def _rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write(path: Path, rows: list[dict[str, object]]) -> str:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            payload = (
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists() or args.development_modulus != 10:
        raise NMC1DataError("NMC1 output or split geometry differs")
    if sha256_file(args.train) != args.expected_train_sha256:
        raise NMC1DataError("train source SHA-256 differs")
    if sha256_file(args.test) != args.expected_test_sha256:
        raise NMC1DataError("test source SHA-256 differs")
    test_ids = {identity(str(row["question"])) for row in _rows(args.test)}
    if len(test_ids) != 1319:
        raise NMC1DataError("public-test identity population differs")

    program_train: list[dict[str, object]] = []
    direct_train: list[dict[str, object]] = []
    development: list[dict[str, object]] = []
    counters: Counter[str] = Counter()
    seen: set[str] = set()
    for source in _rows(args.train):
        counters["source_rows"] += 1
        question, answer = source.get("question"), source.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            counters["rejected:malformed"] += 1
            continue
        row_identity = identity(question)
        if row_identity in seen:
            raise NMC1DataError("duplicate normalized train question")
        seen.add(row_identity)
        if row_identity in test_ids:
            raise NMC1DataError("train/test question overlap")
        try:
            program, final = compile_gsm8k_answer(answer)
            serialized = render_program(program)
            if execute_fraction(program) != final:
                raise NaturalMicrocodeError("Fraction execution differs")
            if not result_fields_absent(serialized, str(final)):
                raise NaturalMicrocodeError("result field leaked")
        except NaturalMicrocodeError as error:
            counters[f"rejected:{error}"] += 1
            continue
        split = "development" if int(row_identity[:16], 16) % 10 == 0 else "train"
        common = {
            "schema": SCHEMA,
            "identity_sha256": row_identity,
            "split": split,
            "source": "gsm8k_train",
            "original_question": question,
            "gold_program": serialized,
            "gold_answer": str(final),
            "register_depth": register_depth(program),
            "action_count": action_count(program),
        }
        if split == "development":
            development.append(common)
        else:
            program_train.append(
                {
                    **common,
                    "question": (
                        "Compile the word problem into MICROCODE_V1. Emit only the "
                        "program. Do not emit calculated results.\n\nPROBLEM:\n"
                        + question
                    ),
                    "response": serialized,
                    "arm": "program",
                }
            )
            direct_train.append(
                {
                    **common,
                    "question": question,
                    "response": answer,
                    "arm": "direct",
                }
            )
        counters[f"admitted:{split}"] += 1
        counters["records"] += register_depth(program)
        counters["actions"] += action_count(program)

    if not program_train or len(program_train) != len(direct_train) or not development:
        raise NMC1DataError("NMC1 output population differs")
    if {row["identity_sha256"] for row in program_train} & {
        row["identity_sha256"] for row in development
    }:
        raise NMC1DataError("train/development identity overlap")
    args.output.mkdir(parents=True)
    files = {
        "program_train": ("program_train.jsonl", program_train),
        "direct_train": ("direct_train.jsonl", direct_train),
        "development": ("development.jsonl", development),
    }
    outputs = {}
    for name, (filename, rows) in files.items():
        path = args.output / filename
        outputs[name] = {
            "path": str(path.resolve()),
            "rows": len(rows),
            "sha256": _write(path, rows),
        }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "public_test_opened": False,
        "sources": {
            "train": {
                "path": str(args.train.resolve()),
                "sha256": args.expected_train_sha256,
            },
            "test_overlap_only": {
                "path": str(args.test.resolve()),
                "sha256": args.expected_test_sha256,
            },
        },
        "development_modulus": args.development_modulus,
        "counters": dict(sorted(counters.items())),
        "outputs": outputs,
        "result_fields_absent": True,
        "fraction_execution_exact": True,
        "zero_split_overlap": True,
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-test-sha256", required=True)
    parser.add_argument("--development-modulus", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
