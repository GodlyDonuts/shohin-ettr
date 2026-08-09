#!/usr/bin/env python3
"""Evaluate model-owned selective whole-trajectory commitment."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import Any

from hf_idr1_evaluate_reviser import shard_bounds
from hf_product_reasoning_eval import (
    _completion_usage,
    _generate_adapter,
    _generate_completions,
    _generation_arguments,
    _generation_stop_token_ids,
    _load_model,
    _render_prompt,
)
from hf_product_reasoning_rollouts import score_completion
from hf_vcr1_evaluate_reviser import (
    _atomic_json,
    _atomic_lines,
    sha256_file,
    summarize as source_summary,
)
from sctr1_commit import selective_commit
from ttr1_revision import tokenize_with_draft_mask


EVAL_SCHEMA = "shohin-sctr1-selective-commit-eval-v1"
DATA_REPORT_SCHEMA = "shohin-sctr1-selective-commit-data-report-v1"
REPORT_SCHEMA = "shohin-sctr1-selective-commit-evaluation-v1"
TASKS = ("math500", "bbh_logic", "mbpp")


class SCTR1EvaluationError(RuntimeError):
    """Selective-commit model, data, or evaluator contract differs."""


def load_rows(path: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != EVAL_SCHEMA or row.get("split") != split:
                raise SCTR1EvaluationError("SCTR1 evaluation schema/split differs")
            identity = row.get("identity_sha256")
            if not isinstance(identity, str) or len(identity) != 64:
                raise SCTR1EvaluationError("SCTR1 identity is invalid")
            if identity in identities:
                raise SCTR1EvaluationError("SCTR1 identity is duplicated")
            identities.add(identity)
            if row.get("task") not in TASKS:
                raise SCTR1EvaluationError("SCTR1 task differs")
            if row.get("runtime_fields") != ["question"]:
                raise SCTR1EvaluationError("SCTR1 runtime fields differ")
            if row.get("expected_command") not in ("keep", "revise"):
                raise SCTR1EvaluationError("SCTR1 expected command differs")
            if row.get("internal_draft_visible") is not True:
                raise SCTR1EvaluationError("SCTR1 draft boundary differs")
            if row.get("external_candidate_text_visible") is not False:
                raise SCTR1EvaluationError("SCTR1 candidate boundary differs")
            question = row.get("question")
            draft = row.get("internal_draft")
            if (
                not isinstance(question, str)
                or not question.strip()
                or not isinstance(draft, dict)
                or draft.get("identity_sha256") != identity
                or not isinstance(draft.get("completion"), str)
            ):
                raise SCTR1EvaluationError("SCTR1 draft binding differs")
            serialized = draft["completion"] if draft["completion"].strip() else "<EMPTY_DRAFT>"
            if serialized not in question:
                raise SCTR1EvaluationError("SCTR1 draft is absent from the prompt")
            assessor = row.get("assessor")
            candidates = row.get("candidates")
            if not isinstance(assessor, dict) or assessor.get("task") != row["task"]:
                raise SCTR1EvaluationError("SCTR1 assessor binding differs")
            if not isinstance(candidates, list) or len(candidates) != 2:
                raise SCTR1EvaluationError("SCTR1 assessor candidates differ")
            rows.append(row)
    if not rows or {row["task"] for row in rows} != set(TASKS):
        raise SCTR1EvaluationError("SCTR1 evaluation coverage differs")
    return rows


def summarize(rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = source_summary(rows, results)
    by_identity = {row["identity_sha256"]: row for row in results}
    commands: Counter[str] = Counter()
    expected: Counter[str] = Counter()
    command_correct = 0
    for row in rows:
        result = by_identity.get(row["identity_sha256"])
        if result is None:
            raise SCTR1EvaluationError("SCTR1 result coverage is incomplete")
        commands[str(result["commit_command"])] += 1
        expected[str(row["expected_command"])] += 1
        command_correct += int(result["commit_command"] == row["expected_command"])
    return {
        "metrics": summary["metrics"],
        "commitment": {
            "predicted": dict(commands),
            "expected": dict(expected),
            "command_correct": command_correct,
            "command_accuracy": command_correct / len(rows),
            "malformed": commands["malformed"],
        },
    }


def masked_completions(
    model: Any,
    tokenizer: Any,
    rendered: list[str],
    max_new_tokens: int,
    stop_token_ids: list[int],
) -> tuple[list[str], list[tuple[int, bool]], int]:
    """Generate at identical prompt geometry with only draft keys hidden."""

    import torch

    token_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    masked_tokens = 0
    for prompt in rendered:
        tokens, attention, _ = tokenize_with_draft_mask(tokenizer, prompt)
        token_rows.append(tokens)
        mask_rows.append(attention)
        masked_tokens += attention.count(0)
    width = max(map(len, token_rows))
    pad_id = int(tokenizer.pad_token_id)
    input_ids = torch.full(
        (len(token_rows), width), pad_id, device="cuda:0", dtype=torch.long
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, (tokens, attention) in enumerate(
        zip(token_rows, mask_rows, strict=True)
    ):
        offset = width - len(tokens)
        input_ids[index, offset:] = torch.tensor(tokens, device="cuda:0")
        attention_mask[index, offset:] = torch.tensor(attention, device="cuda:0")
    arguments = _generation_arguments("greedy", max_new_tokens)
    arguments["eos_token_id"] = (
        stop_token_ids[0] if len(stop_token_ids) == 1 else stop_token_ids
    )
    with torch.inference_mode():
        output = _generate_adapter(
            model,
            {"input_ids": input_ids, "attention_mask": attention_mask},
            arguments,
            pad_id,
        )
    completions = tokenizer.batch_decode(output, skip_special_tokens=True)
    usage = [
        _completion_usage(tokens.tolist(), stop_token_ids, max_new_tokens)
        for tokens in output
    ]
    return completions, usage, masked_tokens


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    if args.report.exists() or args.candidates_output.exists():
        raise SCTR1EvaluationError("SCTR1 evaluation output already exists")
    data_report = json.loads(args.data_report.read_text(encoding="utf-8"))
    expected = data_report.get("outputs", {}).get(args.split, {})
    if (
        data_report.get("schema") != DATA_REPORT_SCHEMA
        or data_report.get("status") != "complete"
        or data_report.get("commitment") != "whole_draft_or_whole_revision"
        or Path(expected.get("path", "")).resolve() != args.data.resolve()
        or expected.get("sha256") != sha256_file(args.data)
    ):
        raise SCTR1EvaluationError("SCTR1 data receipt differs")
    all_rows = load_rows(args.data, args.split)
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
    checkpoint_masks_draft = bool(
        adapter_metadata and adapter_metadata.get("mask_internal_draft")
    )
    if checkpoint_masks_draft != args.mask_internal_draft:
        raise SCTR1EvaluationError("SCTR1 draft-mask checkpoint contract differs")
    stop_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()

    results: list[dict[str, Any]] = []
    generated_tokens = exhausted_count = masked_draft_tokens = 0
    started = time.monotonic()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        rendered = [
            _render_prompt(tokenizer, str(row["question"]), True, False)
            for row in batch
        ]
        if args.mask_internal_draft:
            outputs, usage, masked = masked_completions(
                model, tokenizer, rendered, args.max_new_tokens, stop_ids
            )
            masked_draft_tokens += masked
        else:
            outputs, usage = _generate_completions(
                model,
                tokenizer,
                rendered,
                True,
                "greedy",
                args.max_new_tokens,
                stop_ids,
            )
        for row, output, (token_count, exhausted) in zip(
            batch, outputs, usage, strict=True
        ):
            commit = selective_commit(
                str(row["internal_draft"]["completion"]), output
            )
            score = score_completion(
                row["assessor"], commit.completion, code_timeout=args.code_timeout
            )
            results.append(
                {
                    "schema": "shohin-sctr1-selective-commit-candidate-v1",
                    "identity_sha256": row["identity_sha256"],
                    "task": row["task"],
                    "controller_output": output,
                    "commit_command": commit.command,
                    "commit_valid": commit.valid,
                    "completion": commit.completion,
                    "generated_tokens": token_count,
                    "max_token_exhausted": exhausted,
                    **score,
                }
            )
            generated_tokens += token_count
            exhausted_count += int(exhausted)
        processed = min(start + len(batch), len(rows))
        if processed % 32 == 0 or processed == len(rows):
            print(f"[sctr1-eval] {processed}/{len(rows)}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    candidate_sha256 = _atomic_lines(args.candidates_output, results)
    summary = (
        summarize(rows, results)
        if args.shard_count == 1
        else {"metrics": None, "commitment": None}
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "split": args.split,
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
        "commitment": "whole_draft_or_whole_revision",
        "generation_mode": "greedy",
        "mask_internal_draft": args.mask_internal_draft,
        "masked_draft_tokens": masked_draft_tokens,
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
        "candidates_sha256": candidate_sha256,
        **summary,
    }
    _atomic_json(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--model-loader", choices=("auto", "causal", "multimodal"), default="auto"
    )
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "holdout"), required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=770)
    parser.add_argument("--code-timeout", type=float, default=3.0)
    parser.add_argument("--mask-internal-draft", action="store_true")
    parser.add_argument("--seed", type=int, default=2026080823)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0 or args.code_timeout <= 0:
        parser.error("SCTR1 evaluation dimensions must be positive")
    report = run(args)
    print(json.dumps({"metrics": report["metrics"], "commitment": report["commitment"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
