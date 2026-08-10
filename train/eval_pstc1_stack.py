#!/usr/bin/env python3
"""Evaluate PSTC1 and frozen source/stack causal interventions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import torch

from hf_product_reasoning_train import load_product_backbone, resolve_product_backbone_layout
from pushdown_stack_typed_compiler import (
    ACTIONS,
    MAX_ACTIONS,
    PUSH,
    STOP,
    PushdownStackCompiler,
    StackProgram,
    load_stack_program,
    stack_labels,
)
from train_pstc1_stack import sha256_file, tokenize_sources


SCHEMA = "shohin-pstc1-stack-evaluation-v1"
CONTROLS = {"normal", "source_shuffled", "stack_reset", "stack_top_permuted"}


class PSTC1EvaluationError(RuntimeError):
    """Raised when PSTC1 evaluation custody or execution differs."""


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_rows(path: Path, expected_sha256: str) -> list[StackProgram]:
    if sha256_file(path) != expected_sha256:
        raise PSTC1EvaluationError("development data SHA-256 differs")
    rows = [
        load_stack_program(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != 3917 or len({row.identity_sha256 for row in rows}) != len(rows):
        raise PSTC1EvaluationError("development population differs")
    return rows


def source_shuffle_indices(rows: list[StackProgram], seed: int) -> list[int]:
    groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row.family, len(row.actions), len(row.number_spans))].append(index)
    mapping = list(range(len(rows)))
    generator = random.Random(seed)
    for key, members in sorted(groups.items()):
        if len(members) < 2:
            raise PSTC1EvaluationError(f"source-shuffle singleton bucket: {key}")
        shift = generator.randrange(1, len(members))
        rotated = members[shift:] + members[:shift]
        for target, source in zip(members, rotated, strict=True):
            mapping[target] = source
    if any(index == source for index, source in enumerate(mapping)):
        raise PSTC1EvaluationError("source shuffle retained identity")
    return mapping


def execute_symbolic(
    actions: list[int], pointers: list[int], source: StackProgram
) -> tuple[Any | None, bool, int, int]:
    stack = []
    invalid = 0
    stopped_at = MAX_ACTIONS
    for index, action in enumerate(actions):
        if action == PUSH:
            pointer = pointers[index]
            if not 0 <= pointer < len(source.number_spans):
                invalid += 1
                break
            stack.append(("VALUE", str(source.number_spans[pointer].magnitude)))
        elif ACTIONS[action] == "NEGATE":
            if not stack:
                invalid += 1
                break
            stack[-1] = ("NEGATE", stack[-1])
        elif ACTIONS[action].startswith("APPLY_"):
            if len(stack) < 2:
                invalid += 1
                break
            right = stack.pop()
            left = stack.pop()
            stack.append((ACTIONS[action], left, right))
        elif action == STOP:
            stopped_at = index
            if len(stack) != 1:
                invalid += 1
            break
        else:
            invalid += 1
            break
        if len(stack) > 6:
            invalid += 1
            break
    valid = invalid == 0 and stopped_at < MAX_ACTIONS and len(stack) == 1
    return (stack[0] if valid else None), valid, stopped_at + 1, invalid


def evaluate_batch(output: Any, targets: list[StackProgram], sources: list[StackProgram]) -> list[dict[str, Any]]:
    actions = output.chosen_actions.cpu().tolist()
    pointers = output.chosen_pointers.cpu().tolist()
    details = []
    for predicted_actions, predicted_pointers, target, source in zip(
        actions, pointers, targets, sources, strict=True
    ):
        tree, valid, predicted_length, invalid = execute_symbolic(
            predicted_actions, predicted_pointers, source
        )
        gold_actions = [item.action for item in target.actions]
        gold_pointers = [item.source_index for item in target.actions]
        gold_tree, gold_valid, gold_length, _ = execute_symbolic(
            gold_actions + [STOP] * (MAX_ACTIONS - len(gold_actions)),
            [max(0, value) for value in gold_pointers]
            + [0] * (MAX_ACTIONS - len(gold_pointers)),
            target,
        )
        if not gold_valid or gold_length != len(target.actions):
            raise PSTC1EvaluationError("gold symbolic execution differs")
        action_length_exact = predicted_length == len(target.actions)
        action_sequence_exact = action_length_exact and predicted_actions[:predicted_length] == gold_actions
        pointer_value_exact = action_length_exact
        for index, gold_action in enumerate(gold_actions):
            if gold_action != PUSH:
                continue
            predicted_pointer = predicted_pointers[index]
            if not 0 <= predicted_pointer < len(source.number_spans):
                pointer_value_exact = False
                continue
            predicted_value = source.number_spans[predicted_pointer].magnitude
            gold_value = target.number_spans[gold_pointers[index]].magnitude
            pointer_value_exact &= predicted_value == gold_value
        exact = valid and tree == gold_tree
        question = target.question
        details.append(
            {
                "identity_sha256": target.identity_sha256,
                "source_identity_sha256": source.identity_sha256,
                "family": target.family,
                "gold_action_count": len(target.actions),
                "predicted_action_count": predicted_length,
                "action_length_exact": action_length_exact,
                "action_sequence_exact": action_sequence_exact,
                "pointer_value_exact": pointer_value_exact,
                "valid_program": valid,
                "exact_skeleton": exact,
                "invalid_transitions": invalid,
                "mixed_precedence": ("*" in question or "/" in question) and ("+" in question or "-" in question),
                "unary_group": "-(" in question.replace(" ", ""),
                "parenthesis_count": question.count("("),
            }
        )
    return details


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.output.exists() or args.control not in CONTROLS:
        raise PSTC1EvaluationError("output exists or control differs")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != "shohin-pstc1-stack-training-v1"
        or payload.get("model_revision") != args.model_revision
        or payload.get("data_sha256") != args.expected_train_sha256
    ):
        raise PSTC1EvaluationError("checkpoint custody differs")
    rows = load_rows(args.data, args.expected_data_sha256)
    mapping = source_shuffle_indices(rows, args.shuffle_seed) if args.control == "source_shuffled" else list(range(len(rows)))
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True, use_fast=True)
    backbone, loader = load_product_backbone(
        args.model_root, args.model_loader, dtype=torch.bfloat16, device_map={"": 0}, quantization="none"
    )
    text_model, _, source_width, layout = resolve_product_backbone_layout(backbone)
    backbone.eval().requires_grad_(False)
    config = payload["config"]
    compiler = PushdownStackCompiler(
        source_width,
        width=int(config["width"]),
        encoder_layers=int(config["encoder_layers"]),
        heads=int(config["heads"]),
    ).to(device=device, dtype=torch.bfloat16)
    compiler.load_state_dict(payload["state_dict"], strict=True)
    compiler.eval()
    details = []
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            targets = rows[start : start + args.batch_size]
            sources = [rows[mapping[index]] for index in range(start, min(start + args.batch_size, len(rows)))]
            encoded, candidate_mask, _ = tokenize_sources(tokenizer, sources, device, args.max_source_tokens)
            labels = stack_labels(targets, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                source_features = text_model(
                    input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], use_cache=False, return_dict=True
                ).last_hidden_state
                output = compiler(
                    source_features,
                    encoded["attention_mask"].bool(),
                    candidate_mask,
                    labels["candidate_count"],
                    feedback="hard",
                    reset_stack=args.control == "stack_reset",
                    permute_stack_top=args.control == "stack_top_permuted",
                )
            details.extend(evaluate_batch(output, targets, sources))
    counts: Counter[str] = Counter()
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for detail in details:
        counts["rows"] += 1
        for metric in (
            "action_length_exact", "action_sequence_exact", "pointer_value_exact", "valid_program", "exact_skeleton"
        ):
            counts[metric] += int(detail[metric])
        counts["invalid_transitions"] += int(detail["invalid_transitions"])
        labels = [
            f"family:{detail['family']}",
            f"mixed:{str(detail['mixed_precedence']).lower()}",
            f"unary:{str(detail['unary_group']).lower()}",
            "parentheses:3+" if detail["parenthesis_count"] >= 3 else "parentheses:<3",
        ]
        for label in labels:
            groups[label]["rows"] += 1
            groups[label]["exact_skeleton"] += int(detail["exact_skeleton"])
    elapsed = time.time() - started
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "control": args.control,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "data": str(args.data.resolve()),
        "data_sha256": args.expected_data_sha256,
        "model_revision": args.model_revision,
        "model_loader": loader,
        "backbone_layout": layout,
        "counts": dict(sorted(counts.items())),
        "rates": {key: value / counts["rows"] for key, value in sorted(counts.items()) if key not in {"rows", "invalid_transitions"}},
        "groups": {name: {**dict(sorted(value.items())), "exact_rate": value["exact_skeleton"] / value["rows"]} for name, value in sorted(groups.items())},
        "elapsed_seconds": elapsed,
        "rows_per_second": counts["rows"] / elapsed,
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-source-tokens", type=int, default=256)
    parser.add_argument("--shuffle-seed", type=int, default=2026081043)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
