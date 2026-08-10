#!/usr/bin/env python3
"""Audit exact NMC1/direct token geometry before either matched fit."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from hf_product_reasoning_train import PRODUCT_SYSTEM_PROMPT, render_reasoning_messages

SCHEMA = "shohin-nmc1-token-geometry-v1"


class NMC1TokenError(ValueError):
    """NMC1 tokenizer custody or zero-truncation invariant differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(
    path: Path, expected_sha256: str, tokenizer, maximum: int
) -> dict[str, object]:
    if sha256_file(path) != expected_sha256:
        raise NMC1TokenError("training data SHA-256 differs")
    counts: Counter[str] = Counter()
    longest: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
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
            total = len(prompt) + len(response) + 1
            counts["rows"] += 1
            counts["prompt_tokens"] += len(prompt)
            counts["response_tokens"] += len(response) + 1
            counts["over_limit"] += int(total > maximum)
            counts["max_prompt"] = max(counts["max_prompt"], len(prompt))
            counts["max_response_with_eos"] = max(
                counts["max_response_with_eos"], len(response) + 1
            )
            counts["max_total"] = max(counts["max_total"], total)
            longest.append(
                {
                    "identity_sha256": row["identity_sha256"],
                    "prompt_tokens": len(prompt),
                    "response_tokens_with_eos": len(response) + 1,
                    "total_tokens": total,
                }
            )
    if not counts["rows"]:
        raise NMC1TokenError("training data is empty")
    return {
        "counts": dict(sorted(counts.items())),
        "longest": sorted(
            longest, key=lambda row: int(row["total_tokens"]), reverse=True
        )[:10],
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    from transformers import AutoTokenizer

    if args.output.exists() or args.maximum_sequence_length != 1024:
        raise NMC1TokenError("output or token geometry differs")
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    program = audit(
        args.program,
        args.expected_program_sha256,
        tokenizer,
        args.maximum_sequence_length,
    )
    direct = audit(
        args.direct,
        args.expected_direct_sha256,
        tokenizer,
        args.maximum_sequence_length,
    )
    gates = {
        "program_zero_truncation": program["counts"]["over_limit"] == 0,
        "direct_zero_truncation": direct["counts"]["over_limit"] == 0,
        "matched_rows": program["counts"]["rows"] == direct["counts"]["rows"],
    }
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "model_root": str(args.model_root.resolve()),
        "tokenizer_sha256": sha256_file(args.model_root / "tokenizer.json"),
        "maximum_sequence_length": args.maximum_sequence_length,
        "program": program,
        "direct": direct,
        "gates": gates,
        "overall_pass": all(gates.values()),
        "holdout_used": False,
    }
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--expected-program-sha256", required=True)
    parser.add_argument("--expected-direct-sha256", required=True)
    parser.add_argument("--maximum-sequence-length", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args())
    return 0 if result["overall_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
