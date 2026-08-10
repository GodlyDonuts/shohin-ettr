#!/usr/bin/env python3
"""Audit exact prompt/ledger token geometry with the pinned host tokenizer."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

from hf_product_reasoning_train import PRODUCT_SYSTEM_PROMPT, render_reasoning_messages


SCHEMA = "shohin-structured-ledger-token-geometry-v1"


class LedgerTokenAuditError(ValueError):
    """Raised when tokenizer or sequence custody differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise LedgerTokenAuditError(
                    f"invalid JSON at line {line_number}"
                ) from error
            if not isinstance(row, dict):
                raise LedgerTokenAuditError(f"non-object row at line {line_number}")
            yield row


def quantiles(values: list[int]) -> dict[str, int]:
    if not values:
        raise LedgerTokenAuditError("empty token population")
    ordered = sorted(values)
    result: dict[str, int] = {}
    for label, fraction in (
        ("p50", 0.50),
        ("p90", 0.90),
        ("p95", 0.95),
        ("p99", 0.99),
        ("p999", 0.999),
        ("max", 1.0),
    ):
        index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
        result[label] = ordered[index]
    return result


def audit_file(
    path: Path,
    tokenizer: Any,
    *,
    max_sequence_length: int,
) -> dict[str, Any]:
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    total_lengths: list[int] = []
    overflow_by_family: Counter[str] = Counter()
    rows_by_family: Counter[str] = Counter()
    rows_by_depth: Counter[int] = Counter()
    overflow_by_depth: Counter[int] = Counter()
    max_identity = None
    max_total = -1
    for row in _iter_jsonl(path):
        question = row.get("question")
        response = row.get("response")
        identity = row.get("identity_sha256")
        if not isinstance(question, str) or not isinstance(response, str):
            raise LedgerTokenAuditError("row lacks question or response")
        rendered = render_reasoning_messages(
            tokenizer,
            [
                {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            enable_thinking=False,
        )
        prompt_length = len(tokenizer.encode(rendered, add_special_tokens=False))
        response_length = len(tokenizer.encode(response, add_special_tokens=False)) + 1
        total = prompt_length + response_length
        prompt_lengths.append(prompt_length)
        response_lengths.append(response_length)
        total_lengths.append(total)
        family = str(row.get("family", ""))
        depth = int(row.get("record_count", 0))
        rows_by_family[family] += 1
        rows_by_depth[depth] += 1
        if total > max_sequence_length:
            overflow_by_family[family] += 1
            overflow_by_depth[depth] += 1
        if total > max_total:
            max_total = total
            max_identity = identity
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": len(total_lengths),
        "max_sequence_length": max_sequence_length,
        "prompt_tokens": quantiles(prompt_lengths),
        "response_tokens_including_eos": quantiles(response_lengths),
        "total_tokens": quantiles(total_lengths),
        "overflow_rows": sum(overflow_by_family.values()),
        "overflow_by_family": dict(sorted(overflow_by_family.items())),
        "overflow_by_depth": {
            str(key): value for key, value in sorted(overflow_by_depth.items())
        },
        "rows_by_family": dict(sorted(rows_by_family.items())),
        "rows_by_depth": {str(key): value for key, value in sorted(rows_by_depth.items())},
        "maximum_identity_sha256": max_identity,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.output.exists():
        raise LedgerTokenAuditError(f"refusing existing output: {args.output}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    splits = {
        "train": audit_file(
            args.train, tokenizer, max_sequence_length=args.max_sequence_length
        ),
        "development": audit_file(
            args.development, tokenizer, max_sequence_length=args.max_sequence_length
        ),
    }
    if args.expected_train_sha256 and splits["train"]["sha256"] != args.expected_train_sha256:
        raise LedgerTokenAuditError("train SHA-256 differs")
    if (
        args.expected_development_sha256
        and splits["development"]["sha256"] != args.expected_development_sha256
    ):
        raise LedgerTokenAuditError("development SHA-256 differs")
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "tokenizer_hashes": {
            name: sha256_file(args.model_root / name)
            for name in ("tokenizer.json", "tokenizer_config.json")
        },
        "splits": splits,
        "zero_truncation": all(split["overflow_rows"] == 0 for split in splits.values()),
        "holdout_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sequence-length", type=int, required=True)
    parser.add_argument("--expected-train-sha256")
    parser.add_argument("--expected-development-sha256")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["zero_truncation"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
