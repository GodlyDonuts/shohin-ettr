#!/usr/bin/env python3
"""Prove complete CTE1 prompt/target retention under the frozen tokenizer."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from hf_product_reasoning_train import PRODUCT_SYSTEM_PROMPT, render_reasoning_messages


SCHEMA = "shohin-cte1-token-geometry-v1"


class CTE1TokenError(ValueError):
    """CTE1 tokenizer custody or complete-sequence geometry differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(path: Path, expected_sha256: str, expected_rows: int, tokenizer) -> dict:
    if sha256_file(path) != expected_sha256:
        raise CTE1TokenError("data SHA-256 differs")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != expected_rows:
        raise CTE1TokenError("data population differs")
    counts: Counter[str] = Counter()
    maximum_prompt = 0
    maximum_response = 0
    maximum_complete = 0
    for row in rows:
        rendered = render_reasoning_messages(
            tokenizer,
            [
                {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                {"role": "user", "content": str(row["question"])},
            ],
            enable_thinking=False,
        )
        prompt = tokenizer.encode(rendered, add_special_tokens=False)
        response = tokenizer.encode(str(row["response"]), add_special_tokens=False)
        complete = len(prompt) + len(response) + 1
        if complete > 1024:
            raise CTE1TokenError("complete sequence exceeds 1,024 tokens")
        counts["rows"] += 1
        counts["prompt_tokens"] += len(prompt)
        counts["response_tokens"] += len(response) + 1
        maximum_prompt = max(maximum_prompt, len(prompt))
        maximum_response = max(maximum_response, len(response) + 1)
        maximum_complete = max(maximum_complete, complete)
    return {
        "sha256": expected_sha256,
        "counts": dict(sorted(counts.items())),
        "maximum_prompt_tokens": maximum_prompt,
        "maximum_response_tokens": maximum_response,
        "maximum_complete_tokens": maximum_complete,
        "retained_truncation": 0,
    }


def run(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    if sha256_file(args.model_root / "tokenizer.json") != args.tokenizer_sha256:
        raise CTE1TokenError("tokenizer SHA-256 differs")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_root, trust_remote_code=True, use_fast=True
    )
    report = {
        "schema": SCHEMA,
        "status": "pass",
        "holdout_used": False,
        "public_test_opened": False,
        "tokenizer_sha256": args.tokenizer_sha256,
        "train": audit(args.train, args.expected_train_sha256, 6333, tokenizer),
        "development": audit(
            args.development, args.expected_development_sha256, 666, tokenizer
        ),
    }
    if args.output.exists():
        raise CTE1TokenError("refusing existing output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--expected-development-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

