#!/usr/bin/env python3
"""Audit complete DTMC1 train/development input and pointer geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import torch

from dtmc1_inputs import DraftExample, load_examples, tokenize_draft_sources
from eval_dtmc1_development import load_drafts
from eval_tmc1_development import load_rows, row_graph, source_shuffle

SCHEMA = "shohin-dtmc1-input-geometry-audit-v1"


class DTMC1AuditError(ValueError):
    """DTMC1 complete-corpus input geometry differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_examples(tokenizer, examples: Sequence[DraftExample]) -> dict[str, int]:
    maximum = 0
    charged = 0
    for offset in range(0, len(examples), 64):
        _, _, receipt = tokenize_draft_sources(
            tokenizer, examples[offset : offset + 64], torch.device("cpu"), 1024
        )
        maximum = max(maximum, receipt["maximum_tokens"])
        charged += receipt["charged_source_draft_tokens"]
    return {"rows": len(examples), "maximum_tokens": maximum, "tokens": charged}


def development_examples(
    rows: list[dict[str, object]],
    drafts: dict[str, dict[str, object]],
    control: str,
) -> list[DraftExample]:
    donors = source_shuffle(rows)
    result = []
    for row in rows:
        identity = str(row["identity_sha256"])
        donor = donors[identity]
        source_row = donor if control == "source_draft_shuffled" else row
        draft_identity = (
            str(donor["identity_sha256"])
            if control in {"draft_shuffled", "source_draft_shuffled"}
            else identity
        )
        detail = drafts[draft_identity]
        result.append(
            DraftExample(
                str(source_row["identity_sha256"]),
                row_graph(source_row),
                str(detail["completion"]),
                bool(detail["answer_correct"]),
                bool(detail["exhausted"]),
            )
        )
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_root, trust_remote_code=True, use_fast=True
    )
    if not getattr(tokenizer, "is_fast", False):
        raise DTMC1AuditError("fast tokenizer offsets unavailable")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    train_examples = load_examples(args.train_data, args.train_data_sha256, 6333)
    rows = load_rows(args.development_data, args.development_data_sha256)
    drafts = load_drafts(
        args.draft_report,
        args.draft_report_sha256,
        args.development_data_sha256,
    )
    receipts = {"train": audit_examples(tokenizer, train_examples)}
    for control in ("normal", "draft_shuffled", "source_draft_shuffled"):
        receipts[control] = audit_examples(
            tokenizer, development_examples(rows, drafts, control)
        )
    if any(receipt["maximum_tokens"] > 1024 for receipt in receipts.values()):
        raise DTMC1AuditError("frozen context is insufficient")
    report = {
        "schema": SCHEMA,
        "status": "pass",
        "holdout_used": False,
        "public_test_opened": False,
        "train_data_sha256": args.train_data_sha256,
        "development_data_sha256": args.development_data_sha256,
        "draft_report_sha256": args.draft_report_sha256,
        "tokenizer_sha256": sha256_file(args.model_root / "tokenizer.json"),
        "receipts": receipts,
    }
    if args.output.exists():
        raise DTMC1AuditError("refusing existing report")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--train-data-sha256", required=True)
    parser.add_argument("--development-data", type=Path, required=True)
    parser.add_argument("--development-data-sha256", required=True)
    parser.add_argument("--draft-report", type=Path, required=True)
    parser.add_argument("--draft-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
