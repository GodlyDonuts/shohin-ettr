#!/usr/bin/env python3
"""Retokenize one frozen semantic pointer corpus without changing its text.

This creates a paired tokenizer view of exactly the same questions, programs,
labels, split identities, and character spans.  Only token IDs, token offsets,
token bags, and token counts may change.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
from pathlib import Path

from tokenizers import Tokenizer

from build_referential_literal_pointer_factorized_corpus import (
    SPLITS,
    artifact_paths,
    audit_splits,
    write_jsonl,
)
from semantic_compiler_falsifier import (
    attach_token_targets,
    canonical_json,
    sha256_bytes,
    sha256_file,
)


TOKEN_FIELDS = {"spans", "token_count", "token_ids_sha256", "token_bag"}


def semantic_payload(row: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key not in TOKEN_FIELDS}


def retokenize_row(row: dict[str, object], tokenizer: Tokenizer) -> dict[str, object]:
    question = row.get("question")
    spans = row.get("spans")
    if not isinstance(question, str) or not isinstance(spans, dict):
        raise ValueError("source row lacks question or spans")
    character_spans = {
        label: {
            "start": int(target["start"]),
            "end": int(target["end"]),
            "text": str(target["text"]),
        }
        for label, target in spans.items()
    }
    encoding, token_targets = attach_token_targets(question, character_spans, tokenizer)
    derived = copy.deepcopy(row)
    derived["spans"] = token_targets
    derived["token_count"] = len(encoding.ids)
    derived["token_ids_sha256"] = sha256_bytes(canonical_json(encoding.ids).encode())
    derived["token_bag"] = sorted(collections.Counter(encoding.ids).items())
    if semantic_payload(derived) != semantic_payload(row):
        raise ValueError("retokenization changed semantic row content")
    return derived


def load_bound_rows(source_dir: Path, report: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    rows = {}
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("source report lacks artifacts")
    for split in SPLITS:
        path = source_dir / f"{split}.jsonl"
        expected = artifacts.get(split)
        if not path.is_file() or not isinstance(expected, dict):
            raise ValueError(f"source split is absent: {split}")
        if sha256_file(path) != expected.get("sha256"):
            raise ValueError(f"source report does not bind {split}")
        rows[split] = [json.loads(line) for line in path.read_text().splitlines() if line]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    source_report_path = source_dir / "report.json"
    target_dir = Path(args.out_dir)
    target_paths = artifact_paths(target_dir)
    target_report_path = target_dir / "report.json"
    if any(path.exists() for path in (*target_paths.values(), target_report_path)):
        raise SystemExit("refusing to overwrite corpus output")
    source_report = json.loads(source_report_path.read_text())
    if (
        source_report.get("schema")
        != "r12_referential_literal_pointer_factorized_corpus_v1"
        or not source_report.get("all_gates_pass")
    ):
        raise SystemExit("source corpus is not an admitted factorized board")

    source_rows = load_bound_rows(source_dir, source_report)
    tokenizer_path = Path(args.tokenizer)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    derived_rows = {
        split: [retokenize_row(row, tokenizer) for row in rows]
        for split, rows in source_rows.items()
    }
    source_semantic_sha256 = sha256_bytes(
        canonical_json({
            split: [semantic_payload(row) for row in rows]
            for split, rows in source_rows.items()
        }).encode()
    )
    target_semantic_sha256 = sha256_bytes(
        canonical_json({
            split: [semantic_payload(row) for row in rows]
            for split, rows in derived_rows.items()
        }).encode()
    )
    if source_semantic_sha256 != target_semantic_sha256:
        raise SystemExit("semantic corpus identity changed during retokenization")

    report = audit_splits(derived_rows, tokenizer_path, Path(__file__))
    report["seeds"] = source_report.get("seeds")
    report["groups"] = source_report.get("groups")
    report["derivation"] = {
        "schema": "r12_factorized_semantic_identity_retokenization_v1",
        "source_report": str(source_report_path.resolve()),
        "source_report_sha256": sha256_file(source_report_path),
        "source_tokenizer_sha256": source_report.get("tokenizer_sha256"),
        "source_semantic_sha256": source_semantic_sha256,
        "target_semantic_sha256": target_semantic_sha256,
        "semantic_identity_exact": True,
    }
    target_dir.mkdir(parents=True, exist_ok=True)
    report["artifacts"] = {
        split: write_jsonl(target_paths[split], rows)
        for split, rows in derived_rows.items()
    }
    target_report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "all_gates_pass": report["all_gates_pass"],
        "semantic_identity_sha256": source_semantic_sha256,
        "artifacts": report["artifacts"],
        "report": str(target_report_path.resolve()),
    }, sort_keys=True))
    if not report["all_gates_pass"]:
        raise SystemExit("retokenized corpus audit failed")


if __name__ == "__main__":
    main()
