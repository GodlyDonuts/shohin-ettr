#!/usr/bin/env python3
"""Evaluate the frozen draft-conditioned typed compiler on development."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from dtmc1_inputs import DraftExample, tokenize_draft_sources
from eval_tmc1_development import (
    graph_payload,
    is_multi_digit,
    load_microcode,
    load_rows,
    row_graph,
    source_shuffle,
    structural_counts,
)
from hf_product_reasoning_eval import _load_model
from learned_arithmetic_microcode import LearnedArithmeticError
from train_lam1_microcode import candidate_fraction
from typed_microcode_compiler import TypedMicrocodeCompiler, decode_graphs
from typed_microcode_graph import TypedMicrocodeGraphError, execute_learned

SCHEMA = "shohin-dtmc1-development-evaluation-v1"


class DTMC1EvaluationError(ValueError):
    """Frozen DTMC1 evaluation custody differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_drafts(
    path: Path, expected_sha256: str, expected_data_sha256: str
) -> dict[str, dict[str, object]]:
    if sha256_file(path) != expected_sha256:
        raise DTMC1EvaluationError("direct development report SHA-256 differs")
    report = json.loads(path.read_text(encoding="utf-8"))
    details = report.get("details")
    if (
        report.get("schema") != "shohin-nmc1-development-evaluation-v1"
        or report.get("status") != "complete"
        or report.get("arm") != "direct"
        or report.get("control") != "normal"
        or report.get("public_test_opened") is not False
        or report.get("development_data_sha256") != expected_data_sha256
        or report.get("checkpoint_sha256")
        != "8a2b6550f4083368eff2493f1b14d59a1d9dd2e4b39d4da58ff23cdd1b250b53"
        or report.get("seed") != 2026081053
        or report.get("max_new_tokens") != 512
        or not isinstance(details, list)
        or len(details) != 666
    ):
        raise DTMC1EvaluationError("direct development report custody differs")
    by_identity = {str(detail["identity_sha256"]): detail for detail in details}
    if len(by_identity) != 666:
        raise DTMC1EvaluationError("direct draft identities differ")
    return by_identity


def load_compiler(path: Path) -> tuple[TypedMicrocodeCompiler, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != "shohin-dtmc1-typed-compiler-training-v1"
        or payload.get("updates") != 4096
        or payload.get("maximum_source_tokens") != 1024
    ):
        raise DTMC1EvaluationError("compiler checkpoint differs")
    config = payload.get("config")
    if config != {
        "source_width": 1024,
        "width": 512,
        "source_layers": 2,
        "decoder_layers": 4,
        "heads": 8,
    }:
        raise DTMC1EvaluationError("compiler geometry differs")
    model = TypedMicrocodeCompiler(**config)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device="cuda:0", dtype=torch.bfloat16).eval(), payload


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise DTMC1EvaluationError("refusing existing evaluation output")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, object]:
    from transformers import AutoTokenizer

    if args.batch_size != 32 or args.control not in {
        "normal",
        "draft_shuffled",
        "source_draft_shuffled",
    }:
        raise DTMC1EvaluationError("evaluation geometry differs")
    if sha256_file(args.compiler_checkpoint) != args.expected_compiler_sha256:
        raise DTMC1EvaluationError("compiler checkpoint SHA-256 differs")
    if sha256_file(args.owner_checkpoint) != args.expected_owner_sha256:
        raise DTMC1EvaluationError("owner checkpoint SHA-256 differs")
    rows = load_rows(args.data, args.expected_data_sha256)
    draft_details = load_drafts(
        args.draft_report,
        args.expected_draft_report_sha256,
        args.expected_data_sha256,
    )
    donors = source_shuffle(rows)
    gold_graphs = [row_graph(row) for row in rows]
    source_rows = [
        (
            donors[str(row["identity_sha256"])]
            if args.control == "source_draft_shuffled"
            else row
        )
        for row in rows
    ]
    source_graphs = [row_graph(row) for row in source_rows]
    draft_identities = [
        (
            str(donors[str(row["identity_sha256"])]["identity_sha256"])
            if args.control in {"draft_shuffled", "source_draft_shuffled"}
            else str(row["identity_sha256"])
        )
        for row in rows
    ]
    examples = [
        DraftExample(
            str(source_row["identity_sha256"]),
            graph,
            str(draft_details[draft_identity]["completion"]),
            bool(draft_details[draft_identity]["answer_correct"]),
            bool(draft_details[draft_identity]["exhausted"]),
        )
        for source_row, graph, draft_identity in zip(
            source_rows, source_graphs, draft_identities, strict=True
        )
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_root, trust_remote_code=True, use_fast=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    owner, metadata, loader = _load_model(
        args.model_root, args.owner_checkpoint, "auto"
    )
    if (
        metadata is None
        or metadata.get("update") != 1024
        or metadata.get("model_revision") != args.model_revision
        or metadata.get("data_sha256") != args.expected_owner_data_sha256
    ):
        raise DTMC1EvaluationError("owner metadata differs")
    owner.eval().requires_grad_(False)
    compiler, compiler_receipt = load_compiler(args.compiler_checkpoint)
    if compiler_receipt.get("data_sha256") != args.expected_train_data_sha256:
        raise DTMC1EvaluationError("compiler training corpus differs")
    microcode = load_microcode(args.lam_checkpoint)
    counts: Counter[str] = Counter()
    details = []
    charged_tokens = 0
    maximum_tokens = 0
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for offset in range(0, len(rows), args.batch_size):
        batch_rows = rows[offset : offset + args.batch_size]
        batch_examples = examples[offset : offset + args.batch_size]
        batch_gold = gold_graphs[offset : offset + args.batch_size]
        encoded, candidate_mask, receipt = tokenize_draft_sources(
            tokenizer, batch_examples, torch.device("cuda:0"), 1024
        )
        charged_tokens += receipt["charged_source_draft_tokens"]
        maximum_tokens = max(maximum_tokens, receipt["maximum_tokens"])
        source_count = torch.tensor(
            [len(example.graph.number_spans) for example in batch_examples],
            dtype=torch.long,
            device="cuda:0",
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            source_states = owner.text_model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            output = compiler(
                source_states,
                encoded["attention_mask"].bool(),
                candidate_mask,
                source_count,
            )
        predicted_graphs = decode_graphs(
            output, [example.graph.source for example in batch_examples]
        )
        for row, example, draft_identity, predicted, gold in zip(
            batch_rows,
            batch_examples,
            draft_identities[offset : offset + args.batch_size],
            predicted_graphs,
            batch_gold,
            strict=True,
        ):
            counts["rows"] += 1
            structure = structural_counts(predicted, gold)
            counts.update(structure)
            expected = Fraction(str(row["gold_answer"]))
            detail: dict[str, object] = {
                "identity_sha256": row["identity_sha256"],
                "source_identity_sha256": example.identity_sha256,
                "draft_identity_sha256": draft_identity,
                "draft_correct": example.draft_correct,
                "draft_exhausted": example.exhausted,
                "predicted_graph": graph_payload(predicted),
                "instruction_count_exact": bool(structure["instruction_count_exact"]),
                "graph_exact": bool(structure["graph_exact"]),
            }
            intervention_correct = {}
            for intervention in ("normal", "carry_reset", "opcode_permuted"):
                try:
                    prediction = candidate_fraction(
                        execute_learned(microcode, predicted, intervention=intervention)
                    )
                    correct = prediction == expected
                    intervention_correct[intervention] = correct
                    counts[f"{intervention}:valid"] += 1
                    counts[f"{intervention}:correct"] += int(correct)
                    detail[f"{intervention}_prediction"] = str(prediction)
                    detail[f"{intervention}_correct"] = correct
                except (
                    LearnedArithmeticError,
                    TypedMicrocodeGraphError,
                    ZeroDivisionError,
                ):
                    intervention_correct[intervention] = False
                    counts[f"{intervention}:invalid"] += 1
                    detail[f"{intervention}_correct"] = False
            if intervention_correct.get("normal", False) and is_multi_digit(gold):
                counts["normal_correct_multi_digit_rows"] += 1
                counts["carry_reset:normal_correct_multi_digit_correct"] += int(
                    intervention_correct.get("carry_reset", False)
                )
            details.append(detail)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "public_test_opened": False,
        "control": args.control,
        "model_revision": args.model_revision,
        "model_loader": loader,
        "owner_checkpoint_sha256": args.expected_owner_sha256,
        "compiler_checkpoint_sha256": args.expected_compiler_sha256,
        "compiler_updates": compiler_receipt["updates"],
        "compiler_training_data_sha256": args.expected_train_data_sha256,
        "development_data_sha256": args.expected_data_sha256,
        "draft_report_sha256": args.expected_draft_report_sha256,
        "lam_checkpoint_sha256": sha256_file(args.lam_checkpoint),
        "charged_source_draft_tokens": charged_tokens,
        "maximum_source_tokens": maximum_tokens,
        "counts": dict(sorted(counts.items())),
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "details": details,
    }
    atomic_json(args.output, report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "details"},
            indent=2,
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control",
        choices=("normal", "draft_shuffled", "source_draft_shuffled"),
        required=True,
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--owner-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-owner-sha256", required=True)
    parser.add_argument("--expected-owner-data-sha256", required=True)
    parser.add_argument("--compiler-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-compiler-sha256", required=True)
    parser.add_argument("--expected-train-data-sha256", required=True)
    parser.add_argument("--lam-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--draft-report", type=Path, required=True)
    parser.add_argument("--expected-draft-report-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
