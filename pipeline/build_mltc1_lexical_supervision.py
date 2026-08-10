#!/usr/bin/env python3
"""Build monotonic lexical-role supervision from admitted PSTC1 programs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

from build_pstc1_stack_supervision import NUMBER_RE, extract_expression


SCHEMA = "shohin-mltc1-lexical-supervision-v1"
REPORT_SCHEMA = "shohin-mltc1-lexical-supervision-report-v1"
SYMBOL_ROLE = {
    "+": "ADD",
    "-": "SUB",
    "*": "MUL",
    "/": "DIV",
    "(": "LPAREN",
    ")": "RPAREN",
}
OPERATOR_ROLES = {"ADD", "SUB", "MUL", "DIV", "NEGATE"}
PRECEDENCE = {"ADD": 1, "SUB": 1, "MUL": 2, "DIV": 2, "NEGATE": 3}


class MLTC1SupervisionError(ValueError):
    """Raised when lexical supervision is not extensionally exact."""


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
                raise MLTC1SupervisionError(f"invalid JSON at line {line_number}") from error
            if not isinstance(row, dict):
                raise MLTC1SupervisionError(f"non-object row at line {line_number}")
            yield row


def candidate_spans(question: str) -> list[dict[str, Any]]:
    candidates = [
        {
            "start": match.start(),
            "end": match.end(),
            "surface": match.group(0),
            "surface_type": "NUMBER",
        }
        for match in NUMBER_RE.finditer(question)
    ]
    candidates.extend(
        {
            "start": index,
            "end": index + 1,
            "surface": character,
            "surface_type": {
                "+": "PLUS",
                "-": "MINUS",
                "*": "MUL",
                "/": "DIV",
                "(": "LPAREN",
                ")": "RPAREN",
            }[character],
        }
        for index, character in enumerate(question)
        if character in SYMBOL_ROLE
    )
    candidates.sort(key=lambda item: (item["start"], item["end"]))
    for left, right in zip(candidates, candidates[1:]):
        if left["end"] > right["start"]:
            raise MLTC1SupervisionError("lexical candidates overlap")
    return candidates


def expression_roles(question: str, family: str) -> dict[tuple[int, int], str]:
    expression, expression_start = extract_expression(question, family)
    roles: dict[tuple[int, int], str] = {}
    previous = "START"
    cursor = 0
    while cursor < len(expression):
        if expression[cursor].isspace():
            cursor += 1
            continue
        number = NUMBER_RE.match(expression, cursor)
        if number is not None:
            span = (expression_start + number.start(), expression_start + number.end())
            roles[span] = "NUMBER"
            previous = "VALUE"
            cursor = number.end()
            continue
        character = expression[cursor]
        if character not in SYMBOL_ROLE:
            raise MLTC1SupervisionError("expression lexical symbol differs")
        role = SYMBOL_ROLE[character]
        if character == "-" and previous in {"START", "OPERATOR", "LPAREN"}:
            role = "NEGATE"
        elif character == "+" and previous in {"START", "OPERATOR", "LPAREN"}:
            # Unary plus is semantically identity and is omitted from the program.
            cursor += 1
            continue
        span = (expression_start + cursor, expression_start + cursor + 1)
        roles[span] = role
        if role == "LPAREN":
            previous = "LPAREN"
        elif role == "RPAREN":
            previous = "VALUE"
        elif role in OPERATOR_ROLES:
            previous = "OPERATOR"
        cursor += 1
    return roles


def compile_selected(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """Compile predicted lexical roles using a generic shunting-yard executor."""
    output: list[dict[str, Any]] = []
    operators: list[str] = []
    valid = True
    for candidate_index, candidate in enumerate(candidates):
        role = candidate["role"]
        if role == "IGNORE":
            continue
        if role == "NUMBER":
            output.append({"action": "PUSH", "candidate_index": candidate_index})
            continue
        if role == "LPAREN":
            operators.append(role)
            continue
        if role == "RPAREN":
            while operators and operators[-1] != "LPAREN":
                operator = operators.pop()
                output.append({"action": "NEGATE" if operator == "NEGATE" else f"APPLY_{operator}"})
            if not operators or operators.pop() != "LPAREN":
                valid = False
                break
            continue
        if role not in OPERATOR_ROLES:
            valid = False
            break
        while operators and operators[-1] in OPERATOR_ROLES:
            top = operators[-1]
            should_pop = PRECEDENCE[top] > PRECEDENCE[role] or (
                PRECEDENCE[top] == PRECEDENCE[role] and role != "NEGATE"
            )
            if not should_pop:
                break
            operator = operators.pop()
            output.append({"action": "NEGATE" if operator == "NEGATE" else f"APPLY_{operator}"})
        operators.append(role)
    while valid and operators:
        operator = operators.pop()
        if operator in {"LPAREN", "RPAREN"}:
            valid = False
            break
        output.append({"action": "NEGATE" if operator == "NEGATE" else f"APPLY_{operator}"})
    output.append({"action": "STOP"})
    return output, valid


def compile_row(row: dict[str, Any]) -> dict[str, Any]:
    question = row.get("question")
    family = row.get("family")
    gold_actions = row.get("actions")
    number_spans = row.get("number_spans")
    if not isinstance(question, str) or not isinstance(family, str):
        raise MLTC1SupervisionError("source row metadata differs")
    if not isinstance(gold_actions, list) or not isinstance(number_spans, list):
        raise MLTC1SupervisionError("source stack program differs")
    roles = expression_roles(question, family)
    candidates = candidate_spans(question)
    source_number_lookup = {
        (int(span["start"]), int(span["end"])): index for index, span in enumerate(number_spans)
    }
    for candidate in candidates:
        span = (candidate["start"], candidate["end"])
        candidate["role"] = roles.get(span, "IGNORE")
        candidate["source_index"] = source_number_lookup.get(span, -1)
        if candidate["role"] == "NUMBER" and candidate["source_index"] < 0:
            raise MLTC1SupervisionError("selected number has no source owner")
    predicted, valid = compile_selected(candidates)
    normalized = []
    for action in predicted:
        if action["action"] == "PUSH":
            source_index = candidates[action["candidate_index"]]["source_index"]
            normalized.append({"action": "PUSH", "source_index": source_index})
        else:
            normalized.append({"action": action["action"]})
    if not valid or normalized != gold_actions:
        raise MLTC1SupervisionError("lexical execution does not reproduce stack program")
    return {
        "schema": SCHEMA,
        "identity_sha256": row["identity_sha256"],
        "source_question_sha256": row["source_question_sha256"],
        "split": row["split"],
        "family": family,
        "question": question,
        "number_spans": number_spans,
        "gold_actions": gold_actions,
        "candidates": candidates,
        "candidate_count": len(candidates),
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
        raise MLTC1SupervisionError("refusing existing output root")
    args.output_root.mkdir(parents=True)
    outputs = {}
    for split, path, expected, expected_rows in (
        ("train", args.train, args.expected_train_sha256, 75935),
        ("development", args.development, args.expected_development_sha256, 3917),
    ):
        if sha256_file(path) != expected:
            raise MLTC1SupervisionError(f"{split} source SHA-256 differs")
        rows = []
        counts: Counter[str] = Counter()
        for source in _iter_jsonl(path):
            compiled = compile_row(source)
            rows.append(compiled)
            counts[f"family:{compiled['family']}"] += 1
            counts[f"candidates:{compiled['candidate_count']}"] += 1
            for candidate in compiled["candidates"]:
                counts[f"role:{candidate['role']}"] += 1
        if len(rows) != expected_rows:
            raise MLTC1SupervisionError(f"{split} population differs")
        output = args.output_root / f"{split}.jsonl"
        outputs[split] = {
            "path": str(output.resolve()),
            "sha256": _write_jsonl(output, rows),
            "rows": len(rows),
            "maximum_candidates": max(row["candidate_count"] for row in rows),
            "counts": dict(sorted(counts.items())),
        }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "sources": {
            "train": {"path": str(args.train.resolve()), "sha256": args.expected_train_sha256},
            "development": {"path": str(args.development.resolve()), "sha256": args.expected_development_sha256},
        },
        "outputs": outputs,
        "extensional_parity": True,
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
