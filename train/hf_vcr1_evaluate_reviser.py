#!/usr/bin/env python3
"""Evaluate a verifier-supervised VCR1 whole-solution reviser."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
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

EVAL_SCHEMA = "shohin-vcr1-revision-eval-v1"
DATA_REPORT_SCHEMA = "shohin-vcr1-revision-data-report-v1"
REPORT_SCHEMA = "shohin-vcr1-revision-evaluation-v1"
TASKS = ("math500", "bbh_logic", "mbpp")


class VCR1EvaluationError(RuntimeError):
    """The VCR1 model, data, or evaluator contract differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise VCR1EvaluationError(f"refusing existing candidates: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise VCR1EvaluationError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_rows(path: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != EVAL_SCHEMA or row.get("split") != split:
                raise VCR1EvaluationError("VCR1 evaluation schema/split differs")
            identity = row.get("identity_sha256")
            if not isinstance(identity, str) or len(identity) != 64:
                raise VCR1EvaluationError("VCR1 identity is invalid")
            if identity in identities:
                raise VCR1EvaluationError("VCR1 identity is duplicated")
            identities.add(identity)
            if row.get("task") not in TASKS:
                raise VCR1EvaluationError("VCR1 task differs")
            if row.get("runtime_fields") != ["question"]:
                raise VCR1EvaluationError("VCR1 runtime fields differ")
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                raise VCR1EvaluationError("VCR1 runtime prompt is empty")
            assessor = row.get("assessor")
            if not isinstance(assessor, dict) or assessor.get("task") != row["task"]:
                raise VCR1EvaluationError("VCR1 assessor binding differs")
            candidates = row.get("candidates")
            if not isinstance(candidates, list) or len(candidates) != 2:
                raise VCR1EvaluationError("VCR1 requires two source candidates")
            if {candidate.get("lineage") for candidate in candidates} != {
                "base",
                "expert",
            }:
                raise VCR1EvaluationError("VCR1 candidate lineages differ")
            rows.append(row)
    if not rows or {row["task"] for row in rows} != set(TASKS):
        raise VCR1EvaluationError("VCR1 evaluation coverage differs")
    return rows


def summarize(
    rows: list[dict[str, Any]], results: list[dict[str, Any]]
) -> dict[str, Any]:
    by_identity = {row["identity_sha256"]: row for row in results}
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        result = by_identity.get(row["identity_sha256"])
        if result is None:
            raise VCR1EvaluationError("VCR1 result coverage is incomplete")
        candidates = {
            candidate["lineage"]: candidate for candidate in row["candidates"]
        }
        base_correct = bool(candidates["base"]["correct"])
        expert_correct = bool(candidates["expert"]["correct"])
        generated_correct = bool(result["correct"])
        for key in ("overall", str(row["task"])):
            bucket = buckets[key]
            bucket["total"] += 1
            bucket["base_correct"] += int(base_correct)
            bucket["expert_correct"] += int(expert_correct)
            bucket["generated_correct"] += int(generated_correct)
            bucket["oracle_correct"] += int(base_correct or expert_correct)
            bucket["both_wrong"] += int(not base_correct and not expert_correct)
            bucket["both_wrong_repaired"] += int(
                not base_correct and not expert_correct and generated_correct
            )
    metrics: dict[str, Any] = {}
    for key, bucket in sorted(buckets.items()):
        total = bucket["total"]
        metrics[key] = {
            **dict(bucket),
            "base_accuracy": bucket["base_correct"] / total,
            "expert_accuracy": bucket["expert_correct"] / total,
            "generated_accuracy": bucket["generated_correct"] / total,
            "oracle_accuracy": bucket["oracle_correct"] / total,
            "both_wrong_repair_rate": (
                bucket["both_wrong_repaired"] / bucket["both_wrong"]
                if bucket["both_wrong"]
                else None
            ),
        }
    overall = metrics["overall"]
    gates = {
        "generated_beats_expert_by_0_02": overall["generated_accuracy"]
        >= overall["expert_accuracy"] + 0.02,
        "math_no_regression_over_0_02": metrics["math500"]["generated_accuracy"]
        >= metrics["math500"]["expert_accuracy"] - 0.02,
        "science_no_regression_over_0_02": metrics["bbh_logic"]["generated_accuracy"]
        >= metrics["bbh_logic"]["expert_accuracy"] - 0.02,
        "code_not_below_expert": metrics["mbpp"]["generated_correct"]
        >= metrics["mbpp"]["expert_correct"],
        "repairs_at_least_one_both_wrong": overall["both_wrong_repaired"] > 0,
    }
    return {"gate": gates, "gate_pass": all(gates.values()), "metrics": metrics}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    if args.report.exists() or args.candidates_output.exists():
        raise VCR1EvaluationError("VCR1 evaluation output already exists")
    data_report = json.loads(args.data_report.read_text(encoding="utf-8"))
    if (
        data_report.get("schema") != DATA_REPORT_SCHEMA
        or data_report.get("status") != "complete"
    ):
        raise VCR1EvaluationError("VCR1 data report is incomplete")
    expected = data_report.get("outputs", {}).get(args.split)
    if not isinstance(expected, dict):
        raise VCR1EvaluationError("VCR1 data report lacks requested split")
    if Path(expected.get("path", "")).resolve() != args.data.resolve():
        raise VCR1EvaluationError("VCR1 evaluation data path differs")
    if expected.get("sha256") != sha256_file(args.data):
        raise VCR1EvaluationError("VCR1 evaluation data hash differs")
    rows = load_rows(args.data, args.split)

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, adapter_metadata, model_loader = _load_model(
        args.model_root, args.adapter_checkpoint, "multimodal"
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
                    "schema": "shohin-vcr1-revision-candidate-v1",
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
            print(f"[vcr1-eval] {processed}/{len(rows)}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    candidate_sha256 = _atomic_lines(args.candidates_output, results)
    summary = summarize(rows, results)
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
        "generation_mode": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "seed": args.seed,
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
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "holdout"), required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--code-timeout", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=2026080816)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0 or args.code_timeout <= 0:
        parser.error("VCR1 evaluation dimensions must be positive")
    report = run(args)
    print(
        json.dumps(
            {"gate_pass": report["gate_pass"], "metrics": report["metrics"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
