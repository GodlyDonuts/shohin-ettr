#!/usr/bin/env python3
"""Evaluate KCR1 transaction generation and deterministic execution."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
from typing import Any

from hf_product_reasoning_eval import (
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
    _render_prompt,
)
from hf_product_reasoning_rollouts import score_completion
from hf_vcr1_evaluate_reviser import _atomic_json, _atomic_lines, sha256_file
from kcr1_branch_transducer import ACTIONS, KEEP, KCR1TransducerError, execute_transaction, parse_transaction


DATA_SCHEMA = "shohin-kcr1-development-canary-v1"
DATA_REPORT_SCHEMA = "shohin-kcr1-development-canary-report-v1"
CANDIDATE_SCHEMA = "shohin-kcr1-transaction-candidate-v1"
REPORT_SCHEMA = "shohin-kcr1-transaction-evaluation-v1"


class KCR1EvaluationError(RuntimeError):
    """The KCR1 evaluator or transaction contract differs."""


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = row.get("identity_sha256")
            if (
                row.get("schema") != DATA_SCHEMA
                or row.get("split") != "development"
                or not isinstance(identity, str)
                or identity in identities
                or row.get("runtime_fields") != ["question"]
                or row.get("assessor_fields_visible_to_model") is not False
                or row.get("expected_action") not in ACTIONS
            ):
                raise KCR1EvaluationError("KCR1 canary row differs")
            identities.add(identity)
            if not all(
                isinstance(row.get(key), str) and row[key]
                for key in (
                    "source_identity_sha256",
                    "question",
                    "draft",
                    "expected_execution",
                )
            ) or not isinstance(row.get("assessor"), dict):
                raise KCR1EvaluationError("KCR1 canary assessor binding differs")
            rows.append(row)
    if not rows or len(rows) % 3:
        raise KCR1EvaluationError("KCR1 canary population differs")
    return rows


def shard_bounds(total: int, shard_index: int, shard_count: int, batch_size: int) -> tuple[int, int]:
    if total <= 0 or shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise KCR1EvaluationError("KCR1 shard geometry is invalid")
    if batch_size <= 0:
        raise KCR1EvaluationError("KCR1 batch size is invalid")
    batch_count = (total + batch_size - 1) // batch_size
    batch_start = batch_count * shard_index // shard_count
    batch_end = batch_count * (shard_index + 1) // shard_count
    start = min(total, batch_start * batch_size)
    end = min(total, batch_end * batch_size)
    if start >= end:
        raise KCR1EvaluationError("KCR1 shard is empty")
    return start, end


def summarize(rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != len(results):
        raise KCR1EvaluationError("KCR1 result population differs")
    branch_total: Counter[str] = Counter()
    branch_correct: Counter[str] = Counter()
    source_actions: dict[str, list[bool]] = defaultdict(list)
    action_correct = semantic_correct = exact_execution = keep_exact = keep_total = 0
    malformed = exhausted = 0
    for row, result in zip(rows, results, strict=True):
        if row["identity_sha256"] != result.get("identity_sha256"):
            raise KCR1EvaluationError("KCR1 result identity order differs")
        action = str(row["expected_action"])
        branch_total[action] += 1
        is_action_correct = result.get("action_correct") is True
        branch_correct[action] += int(is_action_correct)
        action_correct += int(is_action_correct)
        semantic_correct += int(result.get("correct") is True)
        exact_execution += int(result.get("execution_exact") is True)
        malformed += int(result.get("valid_transaction") is not True)
        exhausted += int(result.get("max_token_exhausted") is True)
        source_actions[str(row["source_identity_sha256"])].append(is_action_correct)
        if action == KEEP:
            keep_total += 1
            keep_exact += int(result.get("keep_byte_preserved") is True)
    if set(branch_total) != set(ACTIONS) or any(
        total <= 0 for total in branch_total.values()
    ):
        raise KCR1EvaluationError("KCR1 branch coverage differs")
    if any(len(values) != 3 for values in source_actions.values()):
        raise KCR1EvaluationError("KCR1 source counterfactual coverage differs")
    counterfactual_correct = sum(all(values) for values in source_actions.values())
    total = len(rows)
    source_total = len(source_actions)
    action_accuracy = action_correct / total
    branch_accuracy = {
        action: branch_correct[action] / branch_total[action] for action in ACTIONS
    }
    semantic_accuracy = semantic_correct / total
    keep_accuracy = keep_exact / keep_total
    counterfactual_accuracy = counterfactual_correct / source_total
    gates = {
        "action_accuracy_at_least_0_95": action_accuracy >= 0.95,
        "each_branch_action_accuracy_at_least_0_95": min(branch_accuracy.values())
        >= 0.95,
        "executed_semantic_accuracy_at_least_0_95": semantic_accuracy >= 0.95,
        "keep_byte_preservation_at_least_0_99": keep_accuracy >= 0.99,
        "counterfactual_consistency_at_least_0_90": counterfactual_accuracy >= 0.90,
    }
    return {
        "treatment_mechanics_gate": gates,
        "treatment_mechanics_pass": all(gates.values()),
        "control_gate_pending": True,
        "rows": total,
        "sources": source_total,
        "action_correct": action_correct,
        "action_accuracy": action_accuracy,
        "branch_total": dict(branch_total),
        "branch_correct": dict(branch_correct),
        "branch_accuracy": branch_accuracy,
        "semantic_correct": semantic_correct,
        "semantic_accuracy": semantic_accuracy,
        "exact_execution": exact_execution,
        "exact_execution_accuracy": exact_execution / total,
        "keep_byte_preserved": keep_exact,
        "keep_total": keep_total,
        "keep_byte_preservation_accuracy": keep_accuracy,
        "counterfactual_sources_correct": counterfactual_correct,
        "counterfactual_consistency": counterfactual_accuracy,
        "malformed_transactions": malformed,
        "max_token_exhausted": exhausted,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    if args.report.exists() or args.candidates_output.exists():
        raise KCR1EvaluationError("KCR1 evaluation output already exists")
    data_report = json.loads(args.data_report.read_text(encoding="utf-8"))
    expected = data_report.get("output", {})
    if (
        data_report.get("schema") != DATA_REPORT_SCHEMA
        or data_report.get("status") != "complete"
        or data_report.get("holdout_used") is not False
        or Path(expected.get("path", "")).resolve() != args.data.resolve()
        or expected.get("sha256") != sha256_file(args.data)
    ):
        raise KCR1EvaluationError("KCR1 canary report differs")
    all_rows = load_rows(args.data)
    row_start, row_end = shard_bounds(
        len(all_rows), args.shard_index, args.shard_count, args.batch_size
    )
    rows = all_rows[row_start:row_end]
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, adapter_metadata, model_loader = _load_model(
        args.model_root, args.adapter_checkpoint, args.model_loader
    )
    stop_token_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    results: list[dict[str, Any]] = []
    generated_tokens = exhausted_count = 0
    started = time.monotonic()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        rendered = [_render_prompt(tokenizer, row["question"], True, False) for row in batch]
        completions, usage = _generate_completions(
            model,
            tokenizer,
            rendered,
            True,
            "greedy",
            args.max_new_tokens,
            stop_token_ids,
        )
        for row, completion, (token_count, exhausted) in zip(
            batch, completions, usage, strict=True
        ):
            valid = True
            action = None
            executed = ""
            try:
                transaction = parse_transaction(completion)
                action = transaction.action
                executed = execute_transaction(row["draft"], completion)
            except KCR1TransducerError:
                valid = False
            score = (
                score_completion(row["assessor"], executed, code_timeout=args.code_timeout)
                if valid
                else {
                    "prediction": "",
                    "gold": "",
                    "explicit_final_answer": False,
                    "correct": False,
                    "program": None,
                    "execution": None,
                }
            )
            results.append(
                {
                    "schema": CANDIDATE_SCHEMA,
                    "identity_sha256": row["identity_sha256"],
                    "source_identity_sha256": row["source_identity_sha256"],
                    "task": row["task"],
                    "presentation": row["presentation"],
                    "completion": completion,
                    "generated_tokens": token_count,
                    "max_token_exhausted": exhausted,
                    "valid_transaction": valid,
                    "predicted_action": action,
                    "expected_action": row["expected_action"],
                    "action_correct": action == row["expected_action"],
                    "execution_exact": valid and executed == row["expected_execution"],
                    "keep_byte_preserved": (
                        row["expected_action"] == KEEP
                        and action == KEEP
                        and executed == row["draft"]
                    ),
                    **score,
                }
            )
            generated_tokens += token_count
            exhausted_count += int(exhausted)
        processed = min(start + len(batch), len(rows))
        if processed % 32 == 0 or processed == len(rows):
            print(f"[kcr1-eval] {processed}/{len(rows)}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    candidates_sha256 = _atomic_lines(args.candidates_output, results)
    summary = (
        summarize(rows, results)
        if args.shard_count == 1
        else {"treatment_mechanics_gate": None, "treatment_mechanics_pass": False}
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "split": "development",
        "model_root": str(args.model_source_root.resolve()),
        "loaded_model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": model_loader,
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": sha256_file(args.adapter_checkpoint),
        "adapter_metadata": adapter_metadata,
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "data_report": str(args.data_report.resolve()),
        "data_report_sha256": sha256_file(args.data_report),
        "runtime_fields": ["question"],
        "assessor_fields_visible_to_model": False,
        "generation_mode": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "full_row_count": len(all_rows),
        "row_start": row_start,
        "row_end": row_end,
        "elapsed_seconds": elapsed,
        "generated_tokens": generated_tokens,
        "generated_tokens_per_second": generated_tokens / elapsed,
        "max_token_exhausted": exhausted_count,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "candidates_output": str(args.candidates_output.resolve()),
        "candidates_sha256": candidates_sha256,
        **summary,
    }
    _atomic_json(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-loader", choices=("auto", "causal", "multimodal"), default="auto")
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--code-timeout", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=2026081011)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    report = run(parser.parse_args())
    print(json.dumps({"treatment_mechanics_gate": report["treatment_mechanics_gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
