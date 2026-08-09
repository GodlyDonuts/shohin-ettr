#!/usr/bin/env python3
"""Build token-boundary PSET1 pairs from the immutable DSET1 development data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-pset1-pointer-pair-v1"
REPORT_SCHEMA = "shohin-pset1-pointer-data-report-v1"
DSET_SCHEMA = "shohin-dset1-span-edit-presentation-v1"
DSET_REPORT_SCHEMA = "shohin-dset1-span-edit-data-report-v1"
SOURCE_PREFIX = "Original problem:\n"
DRAFT_MARKER = "\n\nInternal draft:\n"


class PSET1DataError(RuntimeError):
    """The PSET1 source, token boundary, or split differs from the contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_rows(path: Path, rows: list[dict[str, Any]]) -> str:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            payload = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def extract_source(question: str) -> str:
    if SOURCE_PREFIX not in question or DRAFT_MARKER not in question:
        raise PSET1DataError("PSET1 source/draft markers are absent")
    source = question.split(SOURCE_PREFIX, 1)[1].split(DRAFT_MARKER, 1)[0]
    if not source.strip() or question.count(source) < 2:
        raise PSET1DataError("PSET1 repeated source custody differs")
    return source


def tokenize_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[list[int]]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")
    if not isinstance(ids, list) or not isinstance(offsets, list) or len(ids) != len(offsets):
        raise PSET1DataError("PSET1 tokenizer lacks offset mappings")
    return [int(value) for value in ids], [list(map(int, value)) for value in offsets]


def character_coverage(offsets: list[list[int]], length: int) -> bool:
    covered = [0] * length
    for left, right in offsets:
        if not 0 <= left <= right <= length:
            return False
        for index in range(left, right):
            covered[index] += 1
    return bool(covered) and all(value == 1 for value in covered)


def exact_surface_bytes(surface: str, maximum: int) -> list[int] | None:
    payload = list(surface.encode("utf-8"))
    try:
        decoded = bytes(payload).decode("utf-8")
    except UnicodeDecodeError:
        return None
    return payload if payload and len(payload) <= maximum and decoded == surface else None


def convert_pair(tokenizer: Any, pair: list[dict[str, Any]], maximum_replacement: int):
    members = {str(row["pair_member"]): row for row in pair}
    if set(members) != {"clean", "fault"}:
        raise PSET1DataError("PSET1 DSET pair membership differs")
    clean, fault = members["clean"], members["fault"]
    if clean["source_identity_sha256"] != fault["source_identity_sha256"]:
        raise PSET1DataError("PSET1 DSET source identity differs")
    source = extract_source(str(clean["question"]))
    if source != extract_source(str(fault["question"])):
        raise PSET1DataError("PSET1 pair source text differs")
    start, end = map(int, clean["changed_character_span"])
    old = str(fault["old_surface"])
    new = str(fault["new_surface"])
    if clean["draft"][start:end] != new or fault["draft"][start:end] != old:
        return None, "registered_character_span_differs"
    old_ids = exact_surface_bytes(old, maximum_replacement)
    new_ids = exact_surface_bytes(new, maximum_replacement)
    if old_ids is None or new_ids is None:
        return None, "replacement_not_exact_or_over_budget"
    source_ids = [int(value) for value in tokenizer.encode(source, add_special_tokens=False)]
    encoded = {}
    for name, row in members.items():
        draft_ids, offsets = tokenize_with_offsets(tokenizer, str(row["draft"]))
        if not character_coverage(offsets, len(str(row["draft"]))):
            return None, "draft_character_coverage_differs"
        encoded[name] = {
            "draft": row["draft"],
            "draft_sha256": row["draft_sha256"],
            "draft_token_count": len(draft_ids),
            "pointer_start": start,
            "pointer_end": end - 1,
        }
    output = {
        "schema": SCHEMA,
        "identity_sha256": sha256_text(f"pset1\0{clean['pair_identity_sha256']}"),
        "source_identity_sha256": clean["source_identity_sha256"],
        "dset_pair_identity_sha256": clean["pair_identity_sha256"],
        "training_group": clean["training_group"],
        "task": clean["task"],
        "corruption_family": clean["corruption_family"],
        "source": source,
        "source_sha256": sha256_text(source),
        "source_token_count": len(source_ids),
        "final_response": clean["final_response"],
        "old_surface": old,
        "old_byte_ids": old_ids,
        "new_surface": new,
        "new_byte_ids": new_ids,
        "changed_character_span": [start, end],
        "members": {
            "clean": {
                **encoded["clean"],
                "action": "KEEP",
                "replacement_byte_ids": [],
                "permuted_action": "REPLACE",
                "permuted_replacement_byte_ids": old_ids,
            },
            "fault": {
                **encoded["fault"],
                "action": "REPLACE",
                "replacement_byte_ids": new_ids,
                "permuted_action": "KEEP",
                "permuted_replacement_byte_ids": [],
            },
        },
    }
    return output, None


def select(rows: list[dict[str, Any]], total: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    quotas = {"numeric_final": total * 7 // 8, "choice_final": total // 8}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["corruption_family"])].append(row)
    selected = []
    for family, quota in quotas.items():
        ordered = sorted(grouped[family], key=lambda row: row["source_identity_sha256"])
        if len(ordered) < quota:
            raise PSET1DataError(f"PSET1 {family} floor fails: {len(ordered)} < {quota}")
        selected.extend(ordered[:quota])
    selected.sort(key=lambda row: row["source_identity_sha256"])
    return selected, quotas


def load_pairs(path: Path, report: dict[str, Any], split: str) -> list[list[dict[str, Any]]]:
    expected = report.get("outputs", {}).get(split, {})
    if Path(str(expected.get("path", ""))).resolve() != path.resolve() or expected.get("sha256") != sha256_file(path):
        raise PSET1DataError(f"PSET1 DSET {split} binding differs")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if line:
            row = json.loads(line)
            if row.get("schema") != DSET_SCHEMA:
                raise PSET1DataError("PSET1 DSET row schema differs")
            grouped[str(row["pair_identity_sha256"])].append(row)
    return [grouped[key] for key in sorted(grouped)]


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise PSET1DataError("PSET1 output exists")
    report = json.loads(args.dset_report.read_text())
    if report.get("schema") != DSET_REPORT_SCHEMA or report.get("status") != "complete" or report.get("holdout_used") is not False:
        raise PSET1DataError("PSET1 DSET report differs")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    converted = {}
    drops = {}
    maxima = Counter()
    for split, path in (("train", args.dset_train), ("diagnostic", args.dset_diagnostic)):
        rows = []
        counts = Counter()
        for pair in load_pairs(path, report, split):
            row, reason = convert_pair(tokenizer, pair, args.max_replacement_tokens)
            if reason:
                counts[reason] += 1
                continue
            assert row is not None
            if row["source_token_count"] > args.max_sequence_length or any(
                member["draft_token_count"] > args.max_sequence_length
                for member in row["members"].values()
            ):
                counts["stream_overflow"] += 1
                continue
            maxima["source"] = max(maxima["source"], row["source_token_count"])
            maxima["draft"] = max(
                maxima["draft"], *(member["draft_token_count"] for member in row["members"].values())
            )
            maxima["replacement"] = max(maxima["replacement"], len(row["new_byte_ids"]), len(row["old_byte_ids"]))
            rows.append(row)
        converted[split] = rows
        drops[split] = dict(counts)
    train, train_quotas = select(converted["train"], args.train_sources)
    diagnostic, diagnostic_quotas = select(converted["diagnostic"], args.diagnostic_sources)
    train_ids = {row["source_identity_sha256"] for row in train}
    diagnostic_ids = {row["source_identity_sha256"] for row in diagnostic}
    if train_ids & diagnostic_ids:
        raise PSET1DataError("PSET1 train/diagnostic overlap")
    args.output.mkdir(parents=True)
    outputs = {}
    for split, rows in (("train", train), ("diagnostic", diagnostic)):
        path = args.output / f"{split}.jsonl"
        outputs[split] = {
            "path": str(path.resolve()),
            "sha256": atomic_rows(path, rows),
            "sources": len(rows),
        }
    payload = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "dset_report_sha256": sha256_file(args.dset_report),
        "dset_train_sha256": sha256_file(args.dset_train),
        "dset_diagnostic_sha256": sha256_file(args.dset_diagnostic),
        "model_root": str(args.model_root.resolve()),
        "model_config_sha256": sha256_file(args.model_root / "config.json"),
        "max_sequence_length": args.max_sequence_length,
        "max_replacement_tokens": args.max_replacement_tokens,
        "maximum_tokens": dict(maxima),
        "drops": drops,
        "quotas": {"train": train_quotas, "diagnostic": diagnostic_quotas},
        "train_diagnostic_source_overlap": 0,
        "outputs": outputs,
    }
    atomic_json(args.output / "report.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dset-train", type=Path, required=True)
    parser.add_argument("--dset-diagnostic", type=Path, required=True)
    parser.add_argument("--dset-report", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-sources", type=int, default=4096)
    parser.add_argument("--diagnostic-sources", type=int, default=256)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--max-replacement-tokens", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
