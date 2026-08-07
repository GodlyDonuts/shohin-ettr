#!/usr/bin/env python3
"""Measure zero-shot NCP1 alias binding on natural evidence statements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from diverge_ncp1_runtime import greedy_ctc_decode, load_pointer, tensorize_commands


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_pointer(args.checkpoint, args.checkpoint_sha256)
    model.to(device)
    rows: list[dict[str, Any]] = []
    targets: list[tuple[int, ...]] = []
    for episode in _load_jsonl(args.public_data):
        aliases = tuple(str(value) for value in episode["aliases"])
        for evidence in episode["evidence"]:
            operation = str(evidence["operation"])
            rows.append({"source_text": evidence["source_text"], "aliases": aliases})
            targets.append((aliases.index(operation),))

    exact = 0
    lengths: dict[int, int] = {}
    examples = []
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            tensors = tensorize_commands(batch, device)
            predictions = greedy_ctc_decode(model(*tensors[:4]), tensors[4])
            for offset, prediction in enumerate(predictions):
                target = targets[start + offset]
                exact += int(prediction == target)
                lengths[len(prediction)] = lengths.get(len(prediction), 0) + 1
                if prediction != target and len(examples) < 10:
                    examples.append(
                        {
                            "prediction": list(prediction),
                            "target": list(target),
                            "source": batch[offset]["source_text"],
                        }
                    )
    print(
        json.dumps(
            {
                "device": str(device),
                "exact": exact,
                "rate": exact / len(rows),
                "total": len(rows),
                "prediction_lengths": lengths,
                "examples": examples,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
