#!/usr/bin/env python3
"""Audit complete ECR1 source/draft/target retention without loading a model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from hf_product_reasoning_train import reservoir_rows_with_sha256
from train_ecr1_product import tokenize_complete_revision_rows


def main() -> int:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=9655)
    parser.add_argument("--data-seed", type=int, default=2026080814)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    rows, observed_sha256 = reservoir_rows_with_sha256(args.data, args.rows, args.data_seed)
    if observed_sha256 != args.data_sha256 or len(rows) != args.rows:
        parser.error("data hash or row count differs")
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    _, _, _, custody = tokenize_complete_revision_rows(
        tokenizer, rows, args.max_sequence_length
    )
    report = {
        "schema": "shohin-ecr1-sequence-custody-v1",
        "status": "complete",
        "model_root": str(args.model_root.resolve()),
        "data": str(args.data.resolve()),
        "data_sha256": observed_sha256,
        "data_seed": args.data_seed,
        "sequence_custody": custody,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    print(
        f"[ecr1-custody] rows={custody['rows']} max={custody['maximum_observed_tokens']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
