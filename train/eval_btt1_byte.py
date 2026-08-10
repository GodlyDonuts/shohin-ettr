#!/usr/bin/env python3
"""Evaluate BTT1 raw-byte compilation and frozen interventions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from build_mltc1_lexical_supervision import compile_selected
from byte_tape_compiler import ROLES, ByteProgram, ByteTapeCompiler, byte_batch
from eval_mltc1_lexical import flat_compile_selected
from train_btt1_byte import load_programs, sha256_file


SCHEMA = "shohin-btt1-evaluation-v1"
CONTROLS = {"normal", "source_shuffled", "zero_bytes", "flat_executor"}


class BTT1EvaluationError(RuntimeError):
    pass


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def binary_depth(program: ByteProgram) -> int:
    return sum(str(action["action"]).startswith("APPLY_") for action in program.gold_actions)


def source_shuffle_indices(rows: list[ByteProgram]) -> list[int]:
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row.family, binary_depth(row))].append(index)
    mapping = list(range(len(rows)))
    for key, members in sorted(groups.items()):
        if len(members) < 2:
            raise BTT1EvaluationError(f"source-shuffle singleton bucket: {key}")
        members = sorted(members, key=lambda index: (len(rows[index].question), rows[index].identity_sha256))
        for target, source in zip(members, members[1:] + members[:1], strict=True):
            mapping[target] = source
    if any(target == source for target, source in enumerate(mapping)):
        raise BTT1EvaluationError("source shuffle retained identity")
    return mapping


def selected_lexemes(question: str, roles: list[str]) -> tuple[list[dict[str, Any]], bool]:
    lexemes = []
    cursor = 0
    while cursor < len(question):
        role = roles[cursor]
        if role == "IGNORE":
            cursor += 1
            continue
        if role == "NUM_BEGIN":
            end = cursor + 1
            while end < len(question) and roles[end] == "NUM_CONT":
                end += 1
            surface = question[cursor:end]
            try:
                float(surface)
            except ValueError:
                return lexemes, False
            lexemes.append({"role": "NUMBER", "source_index": len(lexemes), "surface": surface})
            cursor = end
            continue
        if role == "NUM_CONT":
            return lexemes, False
        lexemes.append({"role": role, "source_index": -1})
        cursor += 1
    return lexemes, True


def compile_roles(question: str, roles: list[str], *, flat: bool) -> tuple[list[dict[str, Any]], bool]:
    lexemes, valid = selected_lexemes(question, roles)
    if not valid:
        return [{"action": "STOP"}], False
    compiled, parse_valid = (flat_compile_selected if flat else compile_selected)(lexemes)
    actions = []
    for action in compiled:
        if action["action"] == "PUSH":
            actions.append({"action": "PUSH", "surface": lexemes[action["candidate_index"]]["surface"]})
        else:
            actions.append({"action": action["action"]})
    return actions, parse_valid


def execute(actions: list[dict[str, Any]]) -> tuple[Any | None, bool]:
    stack: list[Any] = []
    stopped = False
    for action in actions:
        name = action["action"]
        if name == "PUSH":
            stack.append(("VALUE", action["surface"]))
        elif name == "NEGATE":
            if not stack:
                return None, False
            stack[-1] = ("NEGATE", stack[-1])
        elif name.startswith("APPLY_"):
            if len(stack) < 2:
                return None, False
            right, left = stack.pop(), stack.pop()
            stack.append((name, left, right))
        elif name == "STOP":
            stopped = True
            break
        else:
            return None, False
    return (stack[0] if stopped and len(stack) == 1 else None), stopped and len(stack) == 1


def signature(question: str, roles: list[str]):
    lexemes, valid = selected_lexemes(question, roles)
    if not valid:
        return None
    return [(item["role"], item.get("surface")) for item in lexemes]


def evaluate_batch(output: Any, targets: list[ByteProgram], sources: list[ByteProgram], control: str):
    chosen = output.chosen_roles.cpu().tolist()
    details = []
    for predicted, target, source in zip(chosen, targets, sources, strict=True):
        predicted_names = [ROLES[index] for index in predicted[: len(source.question)]]
        target_names = [ROLES[index] for index in target.byte_roles]
        actions, parse_valid = compile_roles(source.question, predicted_names, flat=control == "flat_executor")
        predicted_tree, execution_valid = execute(actions)
        gold_tree, gold_valid = execute(list(target.gold_actions))
        if not gold_valid:
            raise BTT1EvaluationError("gold execution differs")
        question = target.question
        details.append(
            {
                "identity_sha256": target.identity_sha256,
                "source_identity_sha256": source.identity_sha256,
                "family": target.family,
                "binary_depth": binary_depth(target),
                "byte_role_sequence_exact": len(source.question) == len(target.question) and predicted_names == target_names,
                "selected_byte_sequence_exact": signature(source.question, predicted_names) == signature(target.question, target_names),
                "action_sequence_exact": [item["action"] for item in actions] == [item["action"] for item in target.gold_actions],
                "valid_program": parse_valid and execution_valid,
                "exact_skeleton": parse_valid and execution_valid and predicted_tree == gold_tree,
                "mixed_precedence": ("*" in question or "/" in question) and ("+" in question or "-" in question),
                "unary_group": "-(" in question.replace(" ", ""),
                "parenthesis_count": question.count("("),
            }
        )
    return details


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.control not in CONTROLS:
        raise BTT1EvaluationError("output exists or control differs")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "shohin-btt1-training-v1" or payload.get("data_sha256") != args.expected_train_sha256:
        raise BTT1EvaluationError("checkpoint custody differs")
    rows = load_programs(args.data, args.expected_data_sha256, 3917)
    mapping = source_shuffle_indices(rows) if args.control == "source_shuffled" else list(range(len(rows)))
    device = torch.device("cuda")
    config = payload["config"]
    model = ByteTapeCompiler(width=int(config["width"]), encoder_layers=int(config["encoder_layers"]), heads=int(config["heads"])).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    details = []
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            targets = rows[start : start + args.batch_size]
            sources = [rows[mapping[index]] for index in range(start, min(start + args.batch_size, len(rows)))]
            batch = byte_batch(sources, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch["byte_ids"], batch["mask"], zero_bytes=args.control == "zero_bytes")
            details.extend(evaluate_batch(output, targets, sources, args.control))
    counts = Counter()
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for detail in details:
        for metric in ("byte_role_sequence_exact", "selected_byte_sequence_exact", "action_sequence_exact", "valid_program", "exact_skeleton"):
            counts[metric] += int(detail[metric])
        counts["rows"] += 1
        buckets = [
            f"family:{detail['family']}", f"mixed:{str(detail['mixed_precedence']).lower()}",
            f"unary:{str(detail['unary_group']).lower()}",
            "parentheses:3+" if detail["parenthesis_count"] >= 3 else "parentheses:<3",
            "hierarchical:true" if detail["mixed_precedence"] or detail["unary_group"] or detail["parenthesis_count"] else "hierarchical:false",
        ]
        for bucket in buckets:
            groups[bucket]["rows"] += 1
            groups[bucket]["exact_skeleton"] += int(detail["exact_skeleton"])
    elapsed = time.time() - started
    report = {
        "schema": SCHEMA, "status": "complete", "holdout_used": False, "control": args.control,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256_file(args.checkpoint),
        "data": str(args.data.resolve()), "data_sha256": args.expected_data_sha256,
        "counts": dict(counts),
        "rates": {key: counts[key] / counts["rows"] for key in ("byte_role_sequence_exact", "selected_byte_sequence_exact", "action_sequence_exact", "valid_program", "exact_skeleton")},
        "groups": {key: {"rows": value["rows"], "exact_skeleton": value["exact_skeleton"], "exact_rate": value["exact_skeleton"] / value["rows"]} for key, value in sorted(groups.items())},
        "details": details, "elapsed_seconds": elapsed, "rows_per_second": len(details) / elapsed,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
    }
    atomic_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", choices=sorted(CONTROLS), required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
