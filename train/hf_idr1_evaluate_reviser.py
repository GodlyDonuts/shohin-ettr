#!/usr/bin/env python3
"""Evaluate the IDR1 source-plus-internal-draft reviser."""

from __future__ import annotations

import argparse
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
from hf_sdr1_evaluate import FROZEN_FLOORS
from hf_vcr1_evaluate_reviser import (
    _atomic_json,
    _atomic_lines,
    sha256_file,
    summarize as source_summary,
)

EVAL_SCHEMA = "shohin-idr1-revision-eval-v1"
DATA_REPORT_SCHEMA = "shohin-idr1-revision-data-report-v1"
REPORT_SCHEMA = "shohin-idr1-revision-evaluation-v1"
TASKS = ("math500", "bbh_logic", "mbpp")


class IDR1EvaluationError(RuntimeError):
    """The IDR1 model, data, or evaluator contract differs."""


def load_rows(path: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != EVAL_SCHEMA or row.get("split") != split:
                raise IDR1EvaluationError("IDR1 evaluation schema/split differs")
            identity = row.get("identity_sha256")
            if not isinstance(identity, str) or len(identity) != 64:
                raise IDR1EvaluationError("IDR1 identity is invalid")
            if identity in identities:
                raise IDR1EvaluationError("IDR1 identity is duplicated")
            identities.add(identity)
            if row.get("task") not in TASKS:
                raise IDR1EvaluationError("IDR1 task differs")
            if row.get("runtime_fields") != ["question"]:
                raise IDR1EvaluationError("IDR1 runtime fields differ")
            if row.get("internal_draft_visible") is not True:
                raise IDR1EvaluationError("IDR1 internal-draft boundary differs")
            if row.get("external_candidate_text_visible") is not False:
                raise IDR1EvaluationError("IDR1 external-candidate boundary differs")
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                raise IDR1EvaluationError("IDR1 runtime prompt is empty")
            assessor = row.get("assessor")
            if not isinstance(assessor, dict) or assessor.get("task") != row["task"]:
                raise IDR1EvaluationError("IDR1 assessor binding differs")
            draft = row.get("internal_draft")
            if (
                not isinstance(draft, dict)
                or draft.get("identity_sha256") != identity
                or not isinstance(draft.get("completion"), str)
                or draft["completion"] not in row["question"]
            ):
                raise IDR1EvaluationError("IDR1 internal draft is not prompt-bound")
            candidates = row.get("candidates")
            if not isinstance(candidates, list) or len(candidates) != 2:
                raise IDR1EvaluationError("IDR1 requires two assessor candidates")
            if {candidate.get("lineage") for candidate in candidates} != {
                "base",
                "expert",
            }:
                raise IDR1EvaluationError("IDR1 assessor candidate lineages differ")
            rows.append(row)
    if not rows or {row["task"] for row in rows} != set(TASKS):
        raise IDR1EvaluationError("IDR1 evaluation coverage differs")
    return rows


def shard_bounds(total: int, shard_index: int, shard_count: int, batch_size: int) -> tuple[int, int]:
    """Partition rows without changing any full-evaluation batch pair."""
    if total <= 0 or shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise IDR1EvaluationError("IDR1 shard geometry is invalid")
    if batch_size <= 0:
        raise IDR1EvaluationError("IDR1 batch size is invalid")
    batch_count = (total + batch_size - 1) // batch_size
    batch_start = batch_count * shard_index // shard_count
    batch_end = batch_count * (shard_index + 1) // shard_count
    start = min(total, batch_start * batch_size)
    end = min(total, batch_end * batch_size)
    if start >= end:
        raise IDR1EvaluationError("IDR1 shard is empty")
    return start, end


def summarize(
    rows: list[dict[str, Any]], results: list[dict[str, Any]], split: str
) -> dict[str, Any]:
    if split not in FROZEN_FLOORS:
        raise IDR1EvaluationError("IDR1 split has no frozen floors")
    summary = source_summary(rows, results)
    metrics = summary["metrics"]
    floors = FROZEN_FLOORS[split]
    gates = {
        "within_0_02_of_vcr1_overall": metrics["overall"]["generated_correct"]
        >= floors["overall"],
        "math_within_frozen_floor": metrics["math500"]["generated_correct"]
        >= floors["math500"],
        "science_within_frozen_floor": metrics["bbh_logic"]["generated_correct"]
        >= floors["bbh_logic"],
        "code_within_one_answer_of_vcr1": metrics["mbpp"]["generated_correct"]
        >= floors["mbpp"],
        "solves_frozen_both_wrong_floor": metrics["overall"]["both_wrong_repaired"]
        >= floors["both_wrong"],
    }
    return {
        "gate": gates,
        "gate_pass": all(gates.values()),
        "frozen_floors": floors,
        "metrics": metrics,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    if args.report.exists() or args.candidates_output.exists():
        raise IDR1EvaluationError("IDR1 evaluation output already exists")
    data_report = json.loads(args.data_report.read_text(encoding="utf-8"))
    if (
        data_report.get("schema") != DATA_REPORT_SCHEMA
        or data_report.get("status") != "complete"
        or data_report.get("internal_draft_visible") is not True
        or data_report.get("external_candidate_text_visible") is not False
    ):
        raise IDR1EvaluationError("IDR1 data report is incomplete")
    expected = data_report.get("outputs", {}).get(args.split)
    if not isinstance(expected, dict):
        raise IDR1EvaluationError("IDR1 data report lacks requested split")
    if Path(expected.get("path", "")).resolve() != args.data.resolve():
        raise IDR1EvaluationError("IDR1 evaluation data path differs")
    if expected.get("sha256") != sha256_file(args.data):
        raise IDR1EvaluationError("IDR1 evaluation data hash differs")
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
    if hasattr(model, "set_code_intervention"):
        model.set_code_intervention(args.ecr_code_intervention)
    if hasattr(model, "reset_routing_receipt"):
        model.reset_routing_receipt()
    stop_token_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()

    results: list[dict[str, Any]] = []
    generated_tokens = exhausted_count = 0
    started = time.monotonic()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        rendered = [
            _render_prompt(tokenizer, row["question"], True, False) for row in batch
        ]
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
            score = score_completion(
                row["assessor"], completion, code_timeout=args.code_timeout
            )
            results.append(
                {
                    "schema": "shohin-idr1-revision-candidate-v1",
                    "identity_sha256": row["identity_sha256"],
                    "task": row["task"],
                    "completion": completion,
                    "generated_tokens": token_count,
                    "max_token_exhausted": exhausted,
                    **score,
                }
            )
            generated_tokens += token_count
            exhausted_count += int(exhausted)
        processed = min(start + len(batch), len(rows))
        if processed % 32 == 0 or processed == len(rows):
            print(f"[idr1-eval] {processed}/{len(rows)}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    candidate_sha256 = _atomic_lines(args.candidates_output, results)
    summary = (
        summarize(rows, results, args.split)
        if args.shard_count == 1
        else {"gate": None, "gate_pass": False, "frozen_floors": None, "metrics": None}
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
        "ecr_code_intervention": args.ecr_code_intervention,
        "routing_receipt": (
            model.routing_receipt() if hasattr(model, "routing_receipt") else None
        ),
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "data_report": str(args.data_report.resolve()),
        "data_report_sha256": sha256_file(args.data_report),
        "runtime_fields": ["question"],
        "internal_draft_visible": True,
        "external_candidate_text_visible": False,
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
        "--model-loader",
        choices=("auto", "causal", "multimodal"),
        default="auto",
    )
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--split", choices=sorted(FROZEN_FLOORS), required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--code-timeout", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=2026080816)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--ecr-code-intervention",
        choices=("normal", "zero", "mean", "permutation"),
        default="normal",
    )
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"gate": report["gate"], "metrics": report["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
