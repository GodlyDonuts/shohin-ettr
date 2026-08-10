#!/usr/bin/env python3
"""Build complete raw-byte lexical supervision from admitted MLTC1 rows."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from build_mltc1_lexical_supervision import compile_selected


SCHEMA = "shohin-btt1-byte-supervision-v1"
REPORT_SCHEMA = "shohin-btt1-byte-supervision-report-v1"


class BTT1SupervisionError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise BTT1SupervisionError(f"invalid JSON at line {line_number}") from error
            if not isinstance(row, dict):
                raise BTT1SupervisionError(f"non-object row at line {line_number}")
            yield row


def compile_byte_roles(question: str, candidates: list[dict[str, Any]]) -> list[str]:
    try:
        question.encode("ascii")
    except UnicodeEncodeError as error:
        raise BTT1SupervisionError("non-ASCII source is outside byte schema") from error
    roles = ["IGNORE"] * len(question)
    for candidate in candidates:
        role, start, end = candidate.get("role"), candidate.get("start"), candidate.get("end")
        if role == "IGNORE":
            continue
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(question):
            raise BTT1SupervisionError("candidate span differs")
        if any(existing != "IGNORE" for existing in roles[start:end]):
            raise BTT1SupervisionError("selected byte spans overlap")
        if role == "NUMBER":
            roles[start] = "NUM_BEGIN"
            for position in range(start + 1, end):
                roles[position] = "NUM_CONT"
        else:
            if end != start + 1:
                raise BTT1SupervisionError("operator byte span differs")
            roles[start] = role
    return roles


def execute_byte_roles(question: str, roles: list[str]) -> tuple[list[dict[str, Any]], bool]:
    candidates = []
    cursor = 0
    while cursor < len(roles):
        role = roles[cursor]
        if role == "IGNORE":
            cursor += 1
            continue
        if role == "NUM_BEGIN":
            end = cursor + 1
            while end < len(roles) and roles[end] == "NUM_CONT":
                end += 1
            surface = question[cursor:end]
            try:
                float(surface)
            except ValueError:
                return [{"action": "STOP"}], False
            candidates.append({"role": "NUMBER", "source_index": len(candidates), "surface": surface})
            cursor = end
            continue
        if role == "NUM_CONT":
            return [{"action": "STOP"}], False
        candidates.append({"role": role, "source_index": -1})
        cursor += 1
    compiled, valid = compile_selected(candidates)
    actions = []
    for action in compiled:
        if action["action"] == "PUSH":
            candidate = candidates[action["candidate_index"]]
            actions.append({"action": "PUSH", "surface": candidate["surface"]})
        else:
            actions.append({"action": action["action"]})
    return actions, valid


def compile_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("schema") != "shohin-mltc1-lexical-supervision-v1":
        raise BTT1SupervisionError("source schema differs")
    question = row.get("question")
    candidates, gold_actions, spans = row.get("candidates"), row.get("gold_actions"), row.get("number_spans")
    if not isinstance(question, str) or not isinstance(candidates, list) or not isinstance(gold_actions, list) or not isinstance(spans, list):
        raise BTT1SupervisionError("source fields differ")
    roles = compile_byte_roles(question, candidates)
    compiled, valid = execute_byte_roles(question, roles)
    normalized_gold = []
    for action in gold_actions:
        if action["action"] == "PUSH":
            normalized_gold.append({"action": "PUSH", "surface": spans[action["source_index"]]["surface"]})
        else:
            normalized_gold.append({"action": action["action"]})
    if not valid or compiled != normalized_gold:
        raise BTT1SupervisionError("byte execution does not reproduce stack program")
    return {
        "schema": SCHEMA,
        "identity_sha256": row["identity_sha256"],
        "source_question_sha256": row["source_question_sha256"],
        "split": row["split"],
        "family": row["family"],
        "question": question,
        "byte_roles": roles,
        "gold_actions": normalized_gold,
        "byte_count": len(roles),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            payload = canonical_json_bytes(row)
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists():
        raise BTT1SupervisionError("refusing existing output root")
    args.output_root.mkdir(parents=True)
    outputs = {}
    for split, path, expected, expected_rows in (
        ("train", args.train, args.expected_train_sha256, 75935),
        ("development", args.development, args.expected_development_sha256, 3917),
    ):
        if sha256_file(path) != expected:
            raise BTT1SupervisionError(f"{split} source SHA-256 differs")
        rows, counts = [], Counter()
        for source in _iter_jsonl(path):
            row = compile_row(source)
            rows.append(row)
            counts[f"family:{row['family']}"] += 1
            for role in row["byte_roles"]:
                counts[f"role:{role}"] += 1
        if len(rows) != expected_rows:
            raise BTT1SupervisionError(f"{split} population differs")
        output = args.output_root / f"{split}.jsonl"
        outputs[split] = {
            "path": str(output.resolve()), "sha256": _write_jsonl(output, rows), "rows": len(rows),
            "maximum_bytes": max(row["byte_count"] for row in rows), "counts": dict(sorted(counts.items())),
        }
    report = {
        "schema": REPORT_SCHEMA, "status": "complete", "holdout_used": False, "extensional_parity": True,
        "sources": {
            "train": {"path": str(args.train.resolve()), "sha256": args.expected_train_sha256},
            "development": {"path": str(args.development.resolve()), "sha256": args.expected_development_sha256},
        },
        "outputs": outputs,
    }
    report_path = args.output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-development-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
