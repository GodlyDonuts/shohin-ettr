#!/usr/bin/env python3
"""Evaluate FSTC1 skeleton compilation and its frozen causal controls."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import torch

from fixed_slot_typed_compiler import (
    MAX_SLOTS,
    MAX_SOURCE_NUMBERS,
    FixedSlotSkeletonCompiler,
    TypedProgram,
    compile_typed_program,
    decode_fraction,
    skeleton_labels,
)
from hf_product_reasoning_train import (
    load_product_backbone,
    resolve_product_backbone_layout,
)
from train_fstc1_skeleton import sha256_file, tokenize_sources


SCHEMA = "shohin-fstc1-skeleton-evaluation-v1"
CONTROLS = {"normal", "source_shuffled", "recurrence_reset"}


class FSTC1EvaluationError(RuntimeError):
    """Raised when FSTC1 evaluation custody or geometry differs."""


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_rows(path: Path, expected_sha256: str) -> list[tuple[dict[str, Any], TypedProgram]]:
    if sha256_file(path) != expected_sha256:
        raise FSTC1EvaluationError("development data SHA-256 differs")
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append((row, compile_typed_program(row)))
    if len(rows) != 3917 or len({program.identity_sha256 for _, program in rows}) != len(rows):
        raise FSTC1EvaluationError("development population differs")
    return rows


def source_shuffle_indices(rows: list[tuple[dict[str, Any], TypedProgram]], seed: int) -> list[int]:
    groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for index, (row, program) in enumerate(rows):
        groups[(str(row.get("family")), len(program.slots), len(program.number_spans))].append(index)
    mapping = list(range(len(rows)))
    generator = random.Random(seed)
    for key, members in sorted(groups.items()):
        if len(members) < 2:
            raise FSTC1EvaluationError(f"source-shuffle singleton bucket: {key}")
        rotated = list(members)
        shift = generator.randrange(1, len(members))
        rotated = rotated[shift:] + rotated[:shift]
        for target, source in zip(members, rotated, strict=True):
            if target == source:
                raise FSTC1EvaluationError("source shuffle retained identity")
            mapping[target] = source
    return mapping


def _operand_value(program: TypedProgram, slot: int, reference: int, polarity: int) -> Fraction | None:
    if reference < MAX_SOURCE_NUMBERS:
        if reference >= len(program.number_spans):
            return None
        value = program.number_spans[reference].magnitude
    else:
        state = reference - MAX_SOURCE_NUMBERS
        if state >= slot or state >= len(program.slots):
            return None
        value = decode_fraction(program.slots[state].result)
    return -value if polarity else value


def _gold_operand(program: TypedProgram, slot: int, side: str) -> Fraction:
    reference = getattr(program.slots[slot], side)
    value = _operand_value(program, slot, MAX_SOURCE_NUMBERS + reference.index if reference.kind else reference.index, reference.polarity)
    if value is None:
        raise FSTC1EvaluationError("gold operand is unresolved")
    return value


def evaluate_batch(
    output: Any,
    targets: list[TypedProgram],
    sources: list[TypedProgram],
) -> list[dict[str, Any]]:
    active = output.active_logits.argmax(-1).cpu()
    operation = output.operation_logits.argmax(-1).cpu()
    left_reference = output.left_reference_logits.argmax(-1).cpu()
    right_reference = output.right_reference_logits.argmax(-1).cpu()
    left_polarity = output.left_polarity_logits.argmax(-1).cpu()
    right_polarity = output.right_polarity_logits.argmax(-1).cpu()
    details = []
    for row, (target, source) in enumerate(zip(targets, sources, strict=True)):
        gold_depth = len(target.slots)
        predicted_depth = MAX_SLOTS
        for slot in range(MAX_SLOTS):
            if int(active[row, slot]) == 0:
                predicted_depth = slot
                break
        active_exact = predicted_depth == gold_depth
        operation_exact = active_exact and all(
            int(operation[row, slot]) == target.slots[slot].operation
            for slot in range(gold_depth)
        )
        reference_exact = True
        reference_kind_exact = True
        polarity_exact = True
        operand_value_exact = True
        invalid_reference = False
        for slot in range(gold_depth):
            for side, references, polarities in (
                ("left", left_reference, left_polarity),
                ("right", right_reference, right_polarity),
            ):
                predicted_reference = int(references[row, slot])
                predicted_polarity = int(polarities[row, slot])
                gold_reference = getattr(target.slots[slot], side)
                gold_class = (
                    gold_reference.index
                    if gold_reference.kind == 0
                    else MAX_SOURCE_NUMBERS + gold_reference.index
                )
                reference_exact &= predicted_reference == gold_class
                reference_kind_exact &= (
                    predicted_reference >= MAX_SOURCE_NUMBERS
                ) == bool(gold_reference.kind)
                polarity_exact &= predicted_polarity == gold_reference.polarity
                predicted_value = _operand_value(
                    source, slot, predicted_reference, predicted_polarity
                )
                invalid_reference |= predicted_value is None
                operand_value_exact &= (
                    predicted_value is not None
                    and predicted_value == _gold_operand(target, slot, side)
                )
        complete = active_exact and operation_exact and operand_value_exact
        details.append(
            {
                "identity_sha256": target.identity_sha256,
                "source_identity_sha256": source.identity_sha256,
                "gold_depth": gold_depth,
                "predicted_depth": predicted_depth,
                "depth_exact": active_exact,
                "operation_sequence_exact": operation_exact,
                "reference_class_exact": reference_exact,
                "reference_kind_exact": reference_kind_exact,
                "polarity_exact": polarity_exact,
                "operand_value_exact": operand_value_exact,
                "complete_skeleton_exact": complete,
                "invalid_reference": invalid_reference,
            }
        )
    return details


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.output.exists() or args.control not in CONTROLS:
        raise FSTC1EvaluationError("output exists or control differs")
    if not 0 <= args.shard_index < args.shard_count:
        raise FSTC1EvaluationError("shard geometry differs")
    checkpoint_sha256 = sha256_file(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != "shohin-fstc1-skeleton-training-v1"
        or payload.get("model_revision") != args.model_revision
        or payload.get("data_sha256") != args.expected_train_sha256
    ):
        raise FSTC1EvaluationError("checkpoint custody differs")
    rows = load_rows(args.data, args.expected_data_sha256)
    mapping = (
        source_shuffle_indices(rows, args.shuffle_seed)
        if args.control == "source_shuffled"
        else list(range(len(rows)))
    )
    selected = [index for index in range(len(rows)) if index % args.shard_count == args.shard_index]
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_root, trust_remote_code=True, use_fast=True
    )
    backbone, loader = load_product_backbone(
        args.model_root,
        args.model_loader,
        dtype=torch.bfloat16,
        device_map={"": 0},
        quantization="none",
    )
    text_model, _, source_width, layout = resolve_product_backbone_layout(backbone)
    backbone.eval().requires_grad_(False)
    config = payload["config"]
    if int(config["source_width"]) != source_width:
        raise FSTC1EvaluationError("checkpoint source width differs")
    compiler = FixedSlotSkeletonCompiler(
        source_width,
        width=int(config["width"]),
        encoder_layers=int(config["encoder_layers"]),
        heads=int(config["heads"]),
    ).to(device=device, dtype=torch.bfloat16)
    compiler.load_state_dict(payload["state_dict"], strict=True)
    compiler.eval()
    details = []
    started = time.time()
    generated_slots = 0
    with torch.inference_mode():
        for start in range(0, len(selected), args.batch_size):
            indices = selected[start : start + args.batch_size]
            targets = [rows[index][1] for index in indices]
            sources = [rows[mapping[index]][1] for index in indices]
            encoded, candidate_mask, _ = tokenize_sources(
                tokenizer, sources, device, args.max_source_tokens
            )
            labels = skeleton_labels(targets, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                source_features = text_model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state
                output = compiler(
                    source_features,
                    encoded["attention_mask"].bool(),
                    candidate_mask,
                    labels["candidate_count"],
                    feedback="hard",
                    reset_recurrence=args.control == "recurrence_reset",
                )
            details.extend(evaluate_batch(output, targets, sources))
            generated_slots += len(indices) * MAX_SLOTS
    counters: Counter[str] = Counter()
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    by_depth: dict[int, Counter[str]] = defaultdict(Counter)
    row_by_identity = {program.identity_sha256: row for row, program in rows}
    for detail in details:
        counters["rows"] += 1
        family = str(row_by_identity[detail["identity_sha256"]].get("family"))
        depth = int(detail["gold_depth"])
        by_family[family]["rows"] += 1
        by_depth[depth]["rows"] += 1
        for metric in (
            "depth_exact",
            "operation_sequence_exact",
            "reference_class_exact",
            "reference_kind_exact",
            "polarity_exact",
            "operand_value_exact",
            "complete_skeleton_exact",
            "invalid_reference",
        ):
            value = int(bool(detail[metric]))
            counters[metric] += value
            by_family[family][metric] += value
            by_depth[depth][metric] += value
    elapsed = time.time() - started
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "control": args.control,
        "model_revision": args.model_revision,
        "model_loader": loader,
        "backbone_layout": layout,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "data": str(args.data.resolve()),
        "data_sha256": args.expected_data_sha256,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "counts": dict(sorted(counters.items())),
        "rates": {
            key: value / counters["rows"]
            for key, value in sorted(counters.items())
            if key != "rows"
        },
        "by_family": {
            family: dict(sorted(counts.items())) for family, counts in sorted(by_family.items())
        },
        "by_depth": {
            str(depth): dict(sorted(counts.items())) for depth, counts in sorted(by_depth.items())
        },
        "generated_slots": generated_slots,
        "elapsed_seconds": elapsed,
        "rows_per_second": counters["rows"] / elapsed,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-loader", default="auto")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", choices=sorted(CONTROLS), required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-source-tokens", type=int, default=256)
    parser.add_argument("--shuffle-seed", type=int, default=2026081033)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
