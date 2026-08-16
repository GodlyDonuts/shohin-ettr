#!/usr/bin/env python3
"""Measure exact Q36 revision-training token geometry without loading weights."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from hf_product_reasoning_train import PRODUCT_SYSTEM_PROMPT, render_reasoning_messages
from ttr1_revision import tokenize_with_draft_mask


def summarize(rows: list[tuple[int, int, int, int]], limit: int) -> dict[str, Any]:
    if not rows or limit < 1:
        raise ValueError("Q36 token audit geometry differs")
    ordered = sorted(rows, key=lambda row: (row[1], row[0]))
    totals = [row[1] for row in ordered]
    over = [row for row in ordered if row[1] > limit + 1]
    return {
        "rows": len(rows),
        "limit": limit,
        "eos_allowance": 1,
        "maximum_total_tokens": totals[-1],
        "maximum_row_index": ordered[-1][0],
        "over_limit_count": len(over),
        "over_limit": [
            {
                "row_index": index,
                "total_tokens": total,
                "prompt_tokens": prompt,
                "response_tokens_including_eos": response,
            }
            for index, total, prompt, response in reversed(over)
        ],
    }


def run(model_root: Path, data: Path, output: Path, limit: int) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_root, local_files_only=True, trust_remote_code=True
    )
    geometry = []
    with data.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            rendered = render_reasoning_messages(
                tokenizer,
                [
                    {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                    {"role": "user", "content": str(row["question"])},
                ],
                enable_thinking=False,
            )
            prompt, _, _ = tokenize_with_draft_mask(tokenizer, rendered)
            response = tokenizer.encode(str(row["response"]), add_special_tokens=False)
            geometry.append(
                (index, len(prompt) + len(response) + 1, len(prompt), len(response) + 1)
            )
    report = summarize(geometry, limit)
    report.update(
        {
            "schema": "shohin-q36-mtr-training-token-audit-v1",
            "status": "complete",
            "model_root": str(model_root.resolve(strict=True)),
            "data": str(data.resolve(strict=True)),
        }
    )
    if output.exists() or output.is_symlink():
        raise ValueError("Q36 token audit output exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4096)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.model_root, args.data, args.output, args.limit), sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
