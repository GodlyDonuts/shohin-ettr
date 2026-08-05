#!/usr/bin/env python3
"""Decompose the frozen Smol lexical-CSDC tuple failure by source field."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from csdc_smollm2_lexical_bridge import (
    EXPECTED_REASONER_SHA256,
    LexicalBridgeConfig,
    LexicalChallengeParser,
    gather_lexical_targets,
    render_lexical_source,
    sha256_file,
)
from frozen_pointer_backbone import load_frozen_pointer_backbone
from prompt_selected_presented_algebra import PresentedAlgebraConfig
from pspa_presented_reasoning import FAMILIES, batch_sha256, generate_batch


SCHEMA = "shohin-csdc-smollm2-lexical-bridge-diagnostic-v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_parser(
    base_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[LexicalChallengeParser, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != "shohin-csdc-smollm2-lexical-bridge-v1":
        raise ValueError("unsupported lexical bridge checkpoint")
    model, _, receipt = load_frozen_pointer_backbone(base_path, device=device)
    config = LexicalBridgeConfig(**checkpoint["parser_config"])
    parser = LexicalChallengeParser(model, config).to(device)
    missing, unexpected = parser.load_state_dict(
        checkpoint["adapter_state"], strict=False
    )
    if unexpected or any(not name.startswith("model.") for name in missing):
        raise ValueError(
            f"lexical adapter mismatch missing={missing} unexpected={unexpected}"
        )
    parser.eval().requires_grad_(False)
    return parser, {
        "checkpoint_receipt": {
            key: checkpoint[key]
            for key in (
                "base_sha256",
                "tokenizer_sha256",
                "warm_adapter_sha256",
                "reasoner_checkpoint_sha256",
            )
        },
        "backbone_format": receipt.checkpoint_format,
        "base_import": receipt.base_import,
    }


@torch.inference_mode()
def diagnose_cohort(
    parser: LexicalChallengeParser,
    tokenizer: Tokenizer,
    algebra: PresentedAlgebraConfig,
    *,
    family: int,
    length: int,
    count: int,
    batch_size: int,
    seed: int,
    renderer_seed: int,
    templates: tuple[int, ...],
    shifted_aliases: bool,
    device: torch.device,
) -> dict[str, Any]:
    challenge_metrics = {
        "record": 0,
        "start": 0,
        "outcome": 0,
        "length": 0,
        "word": 0,
        "valid": 0,
        "tuple": 0,
    }
    episode_metrics = {"valid": 0, "tuple": 0}
    record_kind_correct = 0
    record_kind_total = 0
    batch_hashes = []
    processed = 0
    challenge_total = 0
    while processed < count:
        current = min(batch_size, count - processed)
        batch = generate_batch(
            current,
            length,
            algebra,
            seed=seed + processed * 1009,
            family=family,
        )
        batch_hashes.append(batch_sha256(batch))
        source = render_lexical_source(
            batch,
            algebra,
            tokenizer,
            seed=renderer_seed + processed * 1013,
            templates=templates,
            shifted_aliases=shifted_aliases,
            seq_len=parser.model.cfg.seq_len,
        ).to(device)
        logits = parser(source)
        decoded, valid = parser.decode(logits, source, algebra)
        true_record, true_start, true_outcome, true_length, true_word = (
            gather_lexical_targets(source, decoded.record_index)
        )
        positions = torch.arange(algebra.maximum_word_length, device=device)
        true_word_mask = positions[None, None] < true_length[..., None]
        start_exact = decoded.start.eq(true_start)
        outcome_exact = decoded.outcome.eq(true_outcome)
        length_exact = decoded.length.eq(true_length)
        word_exact = (decoded.word.eq(true_word) | ~true_word_mask).all(-1)
        tuple_exact = (
            true_record
            & start_exact
            & outcome_exact
            & length_exact
            & word_exact
            & valid
        )
        values = {
            "record": true_record,
            "start": start_exact,
            "outcome": outcome_exact,
            "length": length_exact,
            "word": word_exact,
            "valid": valid,
            "tuple": tuple_exact,
        }
        for name, value in values.items():
            challenge_metrics[name] += int(value.sum().item())
        episode_metrics["valid"] += int(valid.all(-1).sum().item())
        episode_metrics["tuple"] += int(tuple_exact.all(-1).sum().item())
        predicted_kind = logits.kind.argmax(-1)
        record_kind_correct += int(
            predicted_kind[source.record_mask]
            .eq(source.challenge_record[source.record_mask].long())
            .sum()
            .item()
        )
        record_kind_total += int(source.record_mask.sum().item())
        challenge_total += int(valid.numel())
        processed += current
    return {
        "family": FAMILIES[family],
        "length": length,
        "count": count,
        "templates": list(templates),
        "shifted_aliases": shifted_aliases,
        "batch_sha256": hashlib.sha256(
            "\n".join(batch_hashes).encode("ascii")
        ).hexdigest(),
        "record_kind_accuracy": record_kind_correct / record_kind_total,
        "per_challenge": {
            name: value / challenge_total
            for name, value in challenge_metrics.items()
        },
        "per_episode": {
            name: value / count for name, value in episode_metrics.items()
        },
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for split in ("development", "lexical_shift"):
        selected = [row for row in rows if row["split"] == split]
        episodes = sum(row["count"] for row in selected)
        challenges = episodes * 8
        result[split] = {
            "episodes": episodes,
            "challenges": challenges,
            "record_kind_accuracy": sum(
                row["record_kind_accuracy"] * row["count"] for row in selected
            ) / episodes,
            "per_challenge": {
                metric: sum(
                    row["per_challenge"][metric] * row["count"]
                    for row in selected
                ) / episodes
                for metric in selected[0]["per_challenge"]
            },
            "per_episode": {
                metric: sum(
                    row["per_episode"][metric] * row["count"]
                    for row in selected
                ) / episodes
                for metric in selected[0]["per_episode"]
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--reasoner-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("diagnostic requires CUDA")
    if sha256_file(args.reasoner_checkpoint) != EXPECTED_REASONER_SHA256:
        raise SystemExit("reasoner hash differs")
    device = torch.device("cuda")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    lexical_parser, receipts = load_parser(args.base, args.adapter, device)
    reasoner = torch.load(
        args.reasoner_checkpoint, map_location="cpu", weights_only=True
    )
    algebra = PresentedAlgebraConfig(**reasoner["algebra_config"])
    rows = []
    for split, templates, shifted, offset in (
        ("development", (0, 1, 2), False, 0),
        ("lexical_shift", (3,), True, 100_000),
    ):
        for family in range(len(FAMILIES)):
            for length in (8, 12):
                row = diagnose_cohort(
                    lexical_parser,
                    tokenizer,
                    algebra,
                    family=family,
                    length=length,
                    count=args.count,
                    batch_size=args.batch_size,
                    seed=202608053000 + offset + family * 100 + length,
                    renderer_seed=202608054000 + offset + family * 100 + length,
                    templates=templates,
                    shifted_aliases=shifted,
                    device=device,
                )
                row["split"] = split
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "adapter": str(args.adapter),
        "adapter_sha256": sha256_file(args.adapter),
        "reasoner_checkpoint_sha256": EXPECTED_REASONER_SHA256,
        "receipts": receipts,
        "cohorts": rows,
        "summary": aggregate(rows),
        "training_updates": 0,
        "claim_boundary": "read-only field decomposition of the frozen failed gate",
    }
    _atomic_json(args.output, report)
    print(json.dumps({"summary": report["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()

