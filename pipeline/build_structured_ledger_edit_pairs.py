#!/usr/bin/env python3
"""Build paired clean/fault record-edit transactions over canonical ledgers."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from materialize_structured_ledger_sft import (
    LEDGER_CLOSE,
    LEDGER_OPEN,
    LedgerMaterializationError,
    parse_ledger,
    sha256_file,
)


SCHEMA = "shohin-structured-ledger-edit-pair-v1"
REPORT_SCHEMA = "shohin-structured-ledger-edit-pair-report-v1"
EDIT_OPEN = "<EDIT_V1>"
EDIT_CLOSE = "</EDIT_V1>"


class LedgerEditPairError(ValueError):
    """Raised when paired edit custody or exact execution differs."""


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise LedgerEditPairError(f"invalid JSON at line {line_number}") from error
            if not isinstance(row, dict):
                raise LedgerEditPairError(f"non-object row at line {line_number}")
            yield row


def _changed_fraction(value: str) -> str:
    parsed = Fraction(value)
    changed = parsed + 1
    return str(changed.numerator) if changed.denominator == 1 else f"{changed.numerator}/{changed.denominator}"


def fault_ledger(gold: str, identity: str) -> tuple[str, int, str]:
    parsed = parse_ledger(gold)
    records = parsed["records"]
    if len(records) < 2:
        raise LedgerEditPairError("one-record ledger has no nonterminal fault site")
    address = int(identity[:16], 16) % (len(records) - 1)
    lines = gold.strip().splitlines()
    fields = lines[address + 1].split("|")
    if len(fields) != 5:
        raise LedgerEditPairError("gold record syntax differs")
    fields[-1] = _changed_fraction(fields[-1])
    corrupted = "|".join(fields)
    if corrupted == lines[address + 1]:
        raise LedgerEditPairError("fault mutation is identity")
    lines[address + 1] = corrupted
    fault = "\n".join(lines)
    parse_ledger(fault)
    return fault, address, corrupted


def keep_script() -> str:
    return f"{EDIT_OPEN}\nKEEP\nCOMMIT\n{EDIT_CLOSE}"


def replace_script(address: int, gold_record_line: str) -> str:
    return (
        f"{EDIT_OPEN}\nREPLACE|R{address}|{gold_record_line}\n"
        f"COMMIT\n{EDIT_CLOSE}"
    )


def apply_edit_script(draft: str, script: str) -> str:
    draft_lines = draft.strip().splitlines()
    parse_ledger(draft)
    script_lines = script.strip().splitlines()
    if len(script_lines) < 4 or script_lines[0] != EDIT_OPEN or script_lines[-1] != EDIT_CLOSE:
        raise LedgerEditPairError("edit envelope differs")
    if script_lines[-2] != "COMMIT":
        raise LedgerEditPairError("edit commit differs")
    actions = script_lines[1:-2]
    if actions == ["KEEP"]:
        return draft.strip()
    if len(actions) != 1 or not actions[0].startswith("REPLACE|R"):
        raise LedgerEditPairError("edit action differs")
    _, address_text, replacement = actions[0].split("|", 2)
    if not address_text.startswith("R") or not address_text[1:].isdigit():
        raise LedgerEditPairError("edit address differs")
    address = int(address_text[1:])
    if address < 0 or address >= len(draft_lines) - 3:
        raise LedgerEditPairError("edit address is outside draft")
    if not replacement.startswith(f"R{address}|"):
        raise LedgerEditPairError("replacement address differs")
    draft_lines[address + 1] = replacement
    executed = "\n".join(draft_lines)
    parse_ledger(executed)
    return executed


def editor_prompt(source: str, draft: str) -> str:
    return (
        "Emit one canonical edit transaction for the draft ledger. Preserve all "
        "correct records. Use KEEP or one addressed REPLACE, then COMMIT.\n\n"
        f"<SOURCE_STREAM>\n{source.strip()}\n</SOURCE_STREAM>\n\n"
        f"<DRAFT_STREAM>\n{draft.strip()}\n</DRAFT_STREAM>"
    )


def _presentations(source: Path, counters: Counter[str]) -> Iterable[dict[str, Any]]:
    seen: set[str] = set()
    for row in _iter_jsonl(source):
        identity = str(row.get("identity_sha256", ""))
        if len(identity) != 64 or identity in seen:
            raise LedgerEditPairError("source identity differs")
        seen.add(identity)
        gold = str(row.get("response", ""))
        parsed = parse_ledger(gold)
        if len(parsed["records"]) < 2:
            counters["excluded_depth_one"] += 1
            continue
        fault, address, corrupted_line = fault_ledger(gold, identity)
        gold_line = gold.strip().splitlines()[address + 1]
        clean_script = keep_script()
        fault_script = replace_script(address, gold_line)
        if apply_edit_script(gold, clean_script) != gold:
            raise LedgerEditPairError("clean execution differs")
        if apply_edit_script(fault, fault_script) != gold:
            raise LedgerEditPairError("fault repair execution differs")
        pair_id = hashlib.sha256(f"{SCHEMA}\0{identity}".encode()).hexdigest()
        common = {
            "schema": SCHEMA,
            "pair_identity_sha256": pair_id,
            "source_identity_sha256": identity,
            "source_question_sha256": row.get("source_question_sha256"),
            "split": row.get("split"),
            "family": row.get("family"),
            "record_count": row.get("record_count"),
            "fault_address": address,
            "fault_kind": "nonterminal_result_plus_one",
            "gold_ledger_sha256": hashlib.sha256(gold.encode()).hexdigest(),
        }
        for presentation, draft, response in (
            ("clean", gold, clean_script),
            ("fault", fault, fault_script),
        ):
            counters["presentations"] += 1
            counters[f"presentation:{presentation}"] += 1
            counters[f"family:{row.get('family', '')}"] += 1
            counters["draft_records"] += len(parsed["records"])
            counters["copied_records"] += len(parsed["records"]) - int(
                presentation == "fault"
            )
            yield {
                **common,
                "identity_sha256": hashlib.sha256(
                    f"{pair_id}\0{presentation}".encode()
                ).hexdigest(),
                "presentation": presentation,
                "question": editor_prompt(str(row["question"]), draft),
                "response": response,
                "draft": draft,
                "gold_ledger": gold,
                "corrupted_record_line": corrupted_line if presentation == "fault" else None,
            }
        counters["pairs"] += 1


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    count = 0
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


def build(
    train_source: Path,
    development_source: Path,
    output_root: Path,
    *,
    expected_train_sha256: str | None = None,
    expected_development_sha256: str | None = None,
) -> dict[str, Any]:
    if output_root.exists():
        raise LedgerEditPairError(f"refusing existing output root: {output_root}")
    hashes = {"train": sha256_file(train_source), "development": sha256_file(development_source)}
    if expected_train_sha256 and hashes["train"] != expected_train_sha256:
        raise LedgerEditPairError("train source SHA-256 differs")
    if expected_development_sha256 and hashes["development"] != expected_development_sha256:
        raise LedgerEditPairError("development source SHA-256 differs")
    output_root.mkdir(parents=True)
    outputs = {}
    counters = {}
    for split, source in (("train", train_source), ("development", development_source)):
        split_counters: Counter[str] = Counter()
        path = output_root / f"{split}.jsonl"
        rows, digest = _atomic_jsonl(path, _presentations(source, split_counters))
        outputs[split] = {"path": str(path.resolve()), "rows": rows, "sha256": digest}
        counters[split] = dict(sorted(split_counters.items()))
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "sources": {
            "train": {"path": str(train_source.resolve()), "sha256": hashes["train"]},
            "development": {"path": str(development_source.resolve()), "sha256": hashes["development"]},
        },
        "outputs": outputs,
        "counters": counters,
        "exact_clean_identity_verified": True,
        "exact_fault_repair_verified": True,
        "maximum_edit_actions": 1,
        "record_copy_fraction": {
            split: (
                counters[split].get("copied_records", 0)
                / counters[split]["draft_records"]
                if counters[split].get("draft_records", 0)
                else None
            )
            for split in counters
        },
        "semantic_host_repair_calls": 0,
        "holdout_used": False,
    }
    temporary = output_root / f".report.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_root / "report.json")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--development-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-train-sha256")
    parser.add_argument("--expected-development-sha256")
    args = parser.parse_args()
    print(json.dumps(build(
        args.train_source,
        args.development_source,
        args.output,
        expected_train_sha256=args.expected_train_sha256,
        expected_development_sha256=args.expected_development_sha256,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
