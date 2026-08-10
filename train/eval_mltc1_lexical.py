#!/usr/bin/env python3
"""Evaluate MLTC1 lexical transduction and frozen causal interventions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from build_mltc1_lexical_supervision import OPERATOR_ROLES, compile_selected
from hf_product_reasoning_train import load_product_backbone, resolve_product_backbone_layout
from monotonic_lexical_compiler import ROLES, LexicalProgram, MonotonicLexicalCompiler, lexical_labels
from train_mltc1_lexical import load_programs, sha256_file, tokenize_sources


SCHEMA = "shohin-mltc1-evaluation-v1"
CONTROLS = {"normal", "source_shuffled", "flat_executor", "candidate_state_permuted"}


class MLTC1EvaluationError(RuntimeError):
    """Raised when MLTC1 evaluation custody differs."""


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def binary_depth(program: LexicalProgram) -> int:
    return sum(str(action["action"]).startswith("APPLY_") for action in program.gold_actions)


def source_shuffle_indices(rows: list[LexicalProgram]) -> list[int]:
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row.family, binary_depth(row))].append(index)
    mapping = list(range(len(rows)))
    for key, members in sorted(groups.items()):
        if len(members) < 2:
            raise MLTC1EvaluationError(f"source-shuffle singleton bucket: {key}")
        members = sorted(members, key=lambda index: (len(rows[index].candidates), rows[index].identity_sha256))
        for target, source in zip(members, members[1:] + members[:1], strict=True):
            mapping[target] = source
    if any(target == source for target, source in enumerate(mapping)):
        raise MLTC1EvaluationError("source shuffle retained identity")
    return mapping


def flat_compile_selected(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """Remove parentheses and binary precedence while preserving lexical roles."""
    output: list[dict[str, Any]] = []
    operators: list[str] = []
    for candidate_index, candidate in enumerate(candidates):
        role = candidate["role"]
        if role in {"IGNORE", "LPAREN", "RPAREN"}:
            continue
        if role == "NUMBER":
            output.append({"action": "PUSH", "candidate_index": candidate_index})
            while operators and operators[-1] == "NEGATE":
                operators.pop()
                output.append({"action": "NEGATE"})
            continue
        if role == "NEGATE":
            operators.append(role)
            continue
        if role not in OPERATOR_ROLES:
            return output + [{"action": "STOP"}], False
        while operators:
            operator = operators.pop()
            output.append({"action": "NEGATE" if operator == "NEGATE" else f"APPLY_{operator}"})
        operators.append(role)
    while operators:
        operator = operators.pop()
        output.append({"action": "NEGATE" if operator == "NEGATE" else f"APPLY_{operator}"})
    output.append({"action": "STOP"})
    return output, True


def normalized_actions(compiled: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for action in compiled:
        if action["action"] == "PUSH":
            result.append({"action": "PUSH", "source_index": candidates[action["candidate_index"]]["source_index"]})
        else:
            result.append({"action": action["action"]})
    return result


def execute(actions: list[dict[str, Any]], source: LexicalProgram) -> tuple[Any | None, bool]:
    stack: list[Any] = []
    stopped = False
    for action in actions:
        name = action["action"]
        if name == "PUSH":
            index = action.get("source_index", -1)
            if not 0 <= index < len(source.number_spans):
                return None, False
            stack.append(("VALUE", str(source.number_spans[index]["magnitude"])))
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


def lexical_signature(program: LexicalProgram, roles: list[int] | None = None):
    signature = []
    for index, candidate in enumerate(program.candidates):
        role = ROLES[candidate.role if roles is None else roles[index]]
        if role == "IGNORE":
            continue
        value = str(program.number_spans[candidate.source_index]["magnitude"]) if role == "NUMBER" and candidate.source_index >= 0 else None
        signature.append((role, value))
    return signature


def evaluate_batch(output: Any, targets: list[LexicalProgram], sources: list[LexicalProgram], control: str):
    chosen = output.chosen_roles.cpu().tolist()
    details = []
    for predicted_roles, target, source in zip(chosen, targets, sources, strict=True):
        predicted_candidates = []
        for candidate, role_id in zip(source.candidates, predicted_roles, strict=False):
            predicted_candidates.append(
                {
                    "role": ROLES[role_id],
                    "source_index": candidate.source_index,
                }
            )
        compiler = flat_compile_selected if control == "flat_executor" else compile_selected
        compiled, lexical_valid = compiler(predicted_candidates)
        actions = normalized_actions(compiled, predicted_candidates)
        predicted_tree, execution_valid = execute(actions, source)
        gold_tree, gold_valid = execute(list(target.gold_actions), target)
        if not gold_valid:
            raise MLTC1EvaluationError("gold execution differs")
        action_names = [action["action"] for action in actions]
        gold_action_names = [action["action"] for action in target.gold_actions]
        exact_roles = len(source.candidates) == len(target.candidates) and predicted_roles[: len(source.candidates)] == [item.role for item in target.candidates]
        selected_exact = lexical_signature(source, predicted_roles) == lexical_signature(target)
        question = target.question
        details.append(
            {
                "identity_sha256": target.identity_sha256,
                "source_identity_sha256": source.identity_sha256,
                "family": target.family,
                "binary_depth": binary_depth(target),
                "lexical_role_sequence_exact": exact_roles,
                "selected_lexical_sequence_exact": selected_exact,
                "action_sequence_exact": action_names == gold_action_names,
                "valid_program": lexical_valid and execution_valid,
                "exact_skeleton": lexical_valid and execution_valid and predicted_tree == gold_tree,
                "mixed_precedence": ("*" in question or "/" in question) and ("+" in question or "-" in question),
                "unary_group": "-(" in question.replace(" ", ""),
                "parenthesis_count": question.count("("),
            }
        )
    return details


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.output.exists() or args.control not in CONTROLS:
        raise MLTC1EvaluationError("output exists or control differs")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "shohin-mltc1-training-v1" or payload.get("model_revision") != args.model_revision or payload.get("data_sha256") != args.expected_train_sha256:
        raise MLTC1EvaluationError("checkpoint custody differs")
    rows = load_programs(args.data, args.expected_data_sha256, 3917)
    mapping = source_shuffle_indices(rows) if args.control == "source_shuffled" else list(range(len(rows)))
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True, use_fast=True)
    backbone, loader = load_product_backbone(
        args.model_root, args.model_loader, dtype=torch.bfloat16, device_map={"": 0}, quantization="none"
    )
    text_model, _, source_width, layout = resolve_product_backbone_layout(backbone)
    backbone.eval().requires_grad_(False)
    config = payload["config"]
    compiler = MonotonicLexicalCompiler(
        source_width, width=int(config["width"]), encoder_layers=int(config["encoder_layers"]), heads=int(config["heads"])
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
            labels = lexical_labels(sources, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                source_features = text_model(
                    input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
                    use_cache=False, return_dict=True,
                ).last_hidden_state
                output = compiler(
                    source_features, candidate_mask, labels["surface"], labels["candidate_count"],
                    permute_candidate_states=args.control == "candidate_state_permuted",
                )
            details.extend(evaluate_batch(output, targets, sources, args.control))
    counts = Counter()
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for detail in details:
        for metric in ("lexical_role_sequence_exact", "selected_lexical_sequence_exact", "action_sequence_exact", "valid_program", "exact_skeleton"):
            counts[metric] += int(detail[metric])
        counts["rows"] += 1
        buckets = [
            f"family:{detail['family']}", f"mixed:{str(detail['mixed_precedence']).lower()}",
            f"unary:{str(detail['unary_group']).lower()}",
            "parentheses:3+" if detail["parenthesis_count"] >= 3 else "parentheses:<3",
            "hierarchical:true" if detail["mixed_precedence"] or detail["unary_group"] or detail["parenthesis_count"] else "hierarchical:false",
            "depth:3+" if detail["binary_depth"] >= 3 else "depth:<3",
        ]
        for bucket in buckets:
            groups[bucket]["rows"] += 1
            groups[bucket]["exact_skeleton"] += int(detail["exact_skeleton"])
    report = {
        "schema": SCHEMA, "status": "complete", "holdout_used": False, "control": args.control,
        "model_revision": args.model_revision, "model_loader": loader, "backbone_layout": layout,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256_file(args.checkpoint),
        "data": str(args.data.resolve()), "data_sha256": args.expected_data_sha256,
        "counts": dict(counts),
        "rates": {key: counts[key] / counts["rows"] for key in ("lexical_role_sequence_exact", "selected_lexical_sequence_exact", "action_sequence_exact", "valid_program", "exact_skeleton")},
        "groups": {key: {"rows": value["rows"], "exact_skeleton": value["exact_skeleton"], "exact_rate": value["exact_skeleton"] / value["rows"]} for key, value in sorted(groups.items())},
        "details": details, "elapsed_seconds": time.time() - started,
        "rows_per_second": len(details) / (time.time() - started),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
    }
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
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
