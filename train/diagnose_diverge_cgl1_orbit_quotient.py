#!/usr/bin/env python3
"""Read-only permutation-orbit attribution for a closed CGL1 arm."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from diverge_gti1_runtime import expected_transaction
from eval_diverge_cgl1 import _load_board, _load_model, _score
from eval_diverge_ccr1 import _referent_records
from eval_diverge_pqi1 import sha256_path


SCHEMA = "shohin-diverge-cgl1-orbit-attribution-v1"


def mapped_swap_scores(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 2 or scores.shape[1] != 2:
        raise ValueError("CGL1 orbit attribution expects two candidate scores")
    return scores.flip(dims=(1,))


def _summary(
    records: Sequence[Mapping[str, Any]], scores: torch.Tensor
) -> dict[str, Any]:
    predictions = scores.argmax(dim=-1).tolist()
    overall = Counter()
    by_mode: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_renderer: defaultdict[str, Counter[str]] = defaultdict(Counter)
    margins = []
    for row, (record, prediction) in enumerate(zip(records, predictions, strict=True)):
        expected = expected_transaction(record)
        exact = int(prediction) == expected
        margins.append(float(scores[row, expected] - scores[row, 1 - expected]))
        for counter in (
            overall,
            by_mode[str(record["mode"])],
            by_renderer[str(int(record["renderer"]))],
        ):
            counter["total"] += 1
            counter["exact"] += exact
    return {
        "overall": dict(overall),
        "by_mode": {key: dict(value) for key, value in sorted(by_mode.items())},
        "by_renderer": {
            key: dict(value) for key, value in sorted(by_renderer.items())
        },
        "mean_signed_margin": sum(margins) / len(margins),
        "prediction_sha256": hashlib.sha256(
            json.dumps(predictions, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "score_sha256": hashlib.sha256(
            scores.contiguous().numpy().tobytes()
        ).hexdigest(),
        "_predictions": predictions,
    }


def _public(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if not key.startswith("_")}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--source-result-sha256", required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("CGL1 orbit-attribution output exists")
    if sha256_path(args.source_result) != args.source_result_sha256:
        raise SystemExit("CGL1 source-result hash differs")
    source_result = json.loads(args.source_result.read_text(encoding="utf-8"))
    if (
        source_result.get("board_type") != "development"
        or source_result.get("checkpoint_sha256") != args.checkpoint_sha256
        or source_result.get("data_sha256") != args.data_sha256
    ):
        raise SystemExit("CGL1 source-result receipt differs")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CGL1 orbit attribution requested unavailable CUDA")

    device = torch.device(args.device)
    board = _load_board(args.data, args.data_sha256, "development")
    records = [row for row in _referent_records(board) if row["stage"] == "QUERY"]
    model, checkpoint = _load_model(
        args.checkpoint,
        args.checkpoint_sha256,
        args.base,
        args.base_sha256,
        args.tokenizer,
        args.tokenizer_sha256,
        device,
    )
    normal = _score(model, records, device=device, batch_size=args.batch_size)
    swapped = _score(
        model,
        records,
        device=device,
        batch_size=args.batch_size,
        control="swap_mentions",
    )
    mapped = mapped_swap_scores(swapped["_scores"])
    orbit_product = normal["_scores"] + mapped
    normal_summary = _summary(records, normal["_scores"])
    mapped_summary = _summary(records, mapped)
    orbit_summary = _summary(records, orbit_product)
    disagreements = sum(
        left != right
        for left, right in zip(
            normal_summary["_predictions"],
            mapped_summary["_predictions"],
            strict=True,
        )
    )
    report = {
        "schema": SCHEMA,
        "claim_boundary": (
            "Read-only attribution on the opened CGL1 development board. "
            "Orbit-product scores cannot rescue or promote the closed CGL1 gate."
        ),
        "frozen_rule": (
            "Map the alpha/beta-swapped candidate scores into the original "
            "physical-identity frame, then add normal and mapped log evidence."
        ),
        "normal": _public(normal_summary),
        "mapped_swap": _public(mapped_summary),
        "orbit_product": _public(orbit_summary),
        "normal_swap_prediction_disagreements": disagreements,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "adapter_state_sha256": checkpoint["adapter_state_sha256"],
        "source_result": str(args.source_result),
        "source_result_sha256": args.source_result_sha256,
        "base": str(args.base),
        "base_sha256": args.base_sha256,
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": args.tokenizer_sha256,
        "data": str(args.data),
        "data_sha256": args.data_sha256,
    }
    _atomic_json(args.output, report)
    os.chmod(args.output, 0o444)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
                "normal": report["normal"]["overall"],
                "mapped_swap": report["mapped_swap"]["overall"],
                "orbit_product": report["orbit_product"]["overall"],
                "disagreements": disagreements,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
