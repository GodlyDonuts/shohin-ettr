#!/usr/bin/env python3
"""Materialize exact structured ledgers as compact source-to-ledger SFT rows."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable


SCHEMA = "shohin-structured-ledger-sft-v1"
REPORT_SCHEMA = "shohin-structured-ledger-sft-report-v1"
LEDGER_OPEN = "<LEDGER_V1>"
LEDGER_CLOSE = "</LEDGER_V1>"
RECORD_RE = re.compile(
    r"R(?P<address>\d+)\|(?P<operation>ADD|SUB|MUL|DIV|POW)\|"
    r"(?P<left>@R\d+|-?\d+(?:/[1-9]\d*)?)\|"
    r"(?P<right>@R\d+|-?\d+(?:/[1-9]\d*)?)\|"
    r"(?P<result>-?\d+(?:/[1-9]\d*)?)"
)
COMMIT_RE = re.compile(r"COMMIT\|@R(?P<address>\d+)\|(?P<value>-?\d+(?:/[1-9]\d*)?)")


class LedgerMaterializationError(ValueError):
    """Raised when ledger custody or canonical encoding differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _fraction(value: dict[str, Any]) -> str:
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise LedgerMaterializationError("fraction differs")
    return str(numerator) if denominator == 1 else f"{numerator}/{denominator}"


def _dependency_map(record: dict[str, Any]) -> dict[str, int]:
    dependencies: dict[str, int] = {}
    for dependency in record.get("dependencies", []):
        role = dependency.get("operand_role")
        index = dependency.get("record_index")
        if role not in {"left", "right"} or type(index) is not int or index < 0:
            raise LedgerMaterializationError("dependency differs")
        if role in dependencies:
            raise LedgerMaterializationError("duplicate dependency role")
        dependencies[role] = index
    return dependencies


def encode_ledger(records: list[dict[str, Any]], terminal_value: dict[str, Any]) -> str:
    if not records:
        raise LedgerMaterializationError("ledger is empty")
    lines = [LEDGER_OPEN]
    for expected_address, record in enumerate(records):
        if record.get("address") != expected_address:
            raise LedgerMaterializationError("record address differs")
        operation = record.get("operation")
        if operation not in {"ADD", "SUB", "MUL", "DIV", "POW"}:
            raise LedgerMaterializationError("operation differs")
        operands = record.get("operands")
        if not isinstance(operands, list) or len(operands) != 2:
            raise LedgerMaterializationError("operand geometry differs")
        dependencies = _dependency_map(record)
        rendered_operands = []
        for role, operand in zip(("left", "right"), operands, strict=True):
            if role in dependencies:
                dependency = dependencies[role]
                if dependency >= expected_address:
                    raise LedgerMaterializationError("dependency is not causal")
                if records[dependency].get("result") != operand:
                    raise LedgerMaterializationError("dependency value differs")
                rendered_operands.append(f"@R{dependency}")
            else:
                rendered_operands.append(_fraction(operand))
        lines.append(
            f"R{expected_address}|{operation}|{rendered_operands[0]}|"
            f"{rendered_operands[1]}|{_fraction(record['result'])}"
        )
    if records[-1].get("result") != terminal_value:
        raise LedgerMaterializationError("terminal value differs")
    lines.append(f"COMMIT|@R{len(records) - 1}|{_fraction(terminal_value)}")
    lines.append(LEDGER_CLOSE)
    return "\n".join(lines)


def parse_ledger(text: str) -> dict[str, Any]:
    lines = text.strip().splitlines()
    if len(lines) < 4 or lines[0] != LEDGER_OPEN or lines[-1] != LEDGER_CLOSE:
        raise LedgerMaterializationError("ledger envelope differs")
    records: list[dict[str, Any]] = []
    for line in lines[1:-2]:
        match = RECORD_RE.fullmatch(line)
        if match is None:
            raise LedgerMaterializationError("record syntax differs")
        address = int(match.group("address"))
        if address != len(records):
            raise LedgerMaterializationError("record sequence differs")
        records.append(match.groupdict())
    commit = COMMIT_RE.fullmatch(lines[-2])
    if commit is None or not records:
        raise LedgerMaterializationError("commit syntax differs")
    if int(commit.group("address")) != len(records) - 1:
        raise LedgerMaterializationError("commit address differs")
    if commit.group("value") != records[-1]["result"]:
        raise LedgerMaterializationError("commit value differs")
    return {"records": records, "commit": commit.groupdict()}


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise LedgerMaterializationError(
                    f"invalid JSON at line {line_number}"
                ) from error
            if not isinstance(row, dict):
                raise LedgerMaterializationError(f"non-object row at line {line_number}")
            yield row


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    if path.exists():
        raise LedgerMaterializationError(f"refusing existing output: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    count = 0
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            payload = canonical_json_bytes(row)
            handle.write(payload)
            digest.update(payload)
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count, digest.hexdigest()


def _materialized_rows(source: Path, counters: Counter[str]) -> Iterable[dict[str, Any]]:
    seen: set[str] = set()
    for row in _iter_jsonl(source):
        identity = row.get("identity_sha256")
        question = row.get("question")
        if not isinstance(identity, str) or len(identity) != 64 or not isinstance(question, str):
            raise LedgerMaterializationError("source identity or question differs")
        if identity in seen:
            raise LedgerMaterializationError("duplicate source identity")
        seen.add(identity)
        response = encode_ledger(row.get("records", []), row.get("terminal_value", {}))
        parsed = parse_ledger(response)
        if len(parsed["records"]) != len(row["records"]):
            raise LedgerMaterializationError("round-trip record count differs")
        counters["rows"] += 1
        counters["records"] += len(row["records"])
        counters[f"family:{row.get('family', '')}"] += 1
        yield {
            "schema": SCHEMA,
            "identity_sha256": identity,
            "source_question_sha256": row.get("source_question_sha256"),
            "split": row.get("split"),
            "family": row.get("family"),
            "question": (
                "Compile the task into the canonical operation ledger. "
                "Emit only the ledger.\n\nTASK:\n" + question.strip()
            ),
            "response": response,
            "record_count": len(row["records"]),
            "terminal_value": row["terminal_value"],
        }


def materialize(
    train_source: Path,
    development_source: Path,
    output_root: Path,
    *,
    expected_train_sha256: str | None = None,
    expected_development_sha256: str | None = None,
) -> dict[str, Any]:
    if output_root.exists():
        raise LedgerMaterializationError(f"refusing existing output root: {output_root}")
    source_hashes = {
        "train": sha256_file(train_source),
        "development": sha256_file(development_source),
    }
    if expected_train_sha256 and source_hashes["train"] != expected_train_sha256:
        raise LedgerMaterializationError("train source SHA-256 differs")
    if (
        expected_development_sha256
        and source_hashes["development"] != expected_development_sha256
    ):
        raise LedgerMaterializationError("development source SHA-256 differs")

    output_root.mkdir(parents=True)
    counters: dict[str, Counter[str]] = {}
    outputs: dict[str, dict[str, Any]] = {}
    for split, source in (
        ("train", train_source),
        ("development", development_source),
    ):
        split_counters: Counter[str] = Counter()
        path = output_root / f"{split}.jsonl"
        rows, digest = _atomic_jsonl(path, _materialized_rows(source, split_counters))
        counters[split] = split_counters
        outputs[split] = {"path": str(path.resolve()), "rows": rows, "sha256": digest}

    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "sources": {
            "train": {"path": str(train_source.resolve()), "sha256": source_hashes["train"]},
            "development": {
                "path": str(development_source.resolve()),
                "sha256": source_hashes["development"],
            },
        },
        "outputs": outputs,
        "counters": {
            split: dict(sorted(split_counters.items()))
            for split, split_counters in counters.items()
        },
        "canonical_round_trip_verified": True,
        "holdout_used": False,
    }
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--development-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-train-sha256")
    parser.add_argument("--expected-development-sha256")
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(
                args.train_source,
                args.development_source,
                args.output,
                expected_train_sha256=args.expected_train_sha256,
                expected_development_sha256=args.expected_development_sha256,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
