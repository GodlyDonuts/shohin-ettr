#!/usr/bin/env python3
"""Conditionally score VCR1 on the preserved 568-example product board."""

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

DATA_SCHEMA = "shohin-vcr1-product-eval-v1"
DATA_REPORT_SCHEMA = "shohin-vcr1-product-data-report-v1"
SOURCE_REPORT_SCHEMA = "shohin-vcr1-revision-evaluation-v1"
REPORT_SCHEMA = "shohin-vcr1-product-evaluation-v1"
TASKS_BY_DOMAIN = {
    "grade_school_math": ("gsm8k",),
    "competition_math": ("math500",),
    "code": ("humaneval", "mbpp"),
    "science": ("gpqa",),
    "logic": ("bbh_logic",),
}
MAIN_TASKS = tuple(task for tasks in TASKS_BY_DOMAIN.values() for task in tasks)
TASKS = (*MAIN_TASKS, "aime")


class VCR1ProductError(RuntimeError):
    """The conditional VCR1 product contract differs."""


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows or len({row.get("identity_sha256") for row in rows}) != len(rows):
        raise VCR1ProductError("VCR1 product identity coverage differs")
    for row in rows:
        if row.get("schema") != DATA_SCHEMA or row.get("task") not in TASKS:
            raise VCR1ProductError("VCR1 product schema/task differs")
        if row.get("runtime_fields") != ["question"]:
            raise VCR1ProductError("VCR1 product runtime fields differ")
    if {row["task"] for row in rows} != set(TASKS):
        raise VCR1ProductError("VCR1 product task coverage differs")
    return rows


def summarize(
    rows: list[dict[str, Any]], results: list[dict[str, Any]]
) -> dict[str, Any]:
    generated = {row["identity_sha256"]: bool(row["correct"]) for row in results}
    identities = {row["identity_sha256"] for row in rows}
    if len(generated) != len(results) or set(generated) != identities:
        raise VCR1ProductError("VCR1 product result coverage differs")
    tasks: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        candidates = {item["lineage"]: item for item in row["candidates"]}
        counter = tasks[row["task"]]
        counter["total"] += 1
        counter["base_correct"] += int(candidates["base"]["correct"])
        counter["expert_correct"] += int(candidates["expert"]["correct"])
        counter["generated_correct"] += int(generated[row["identity_sha256"]])
        counter["oracle_correct"] += int(
            any(item["correct"] for item in candidates.values())
        )
    task_metrics = {task: dict(tasks[task]) for task in TASKS}
    arms: dict[str, dict[str, Any]] = {}
    for arm, field in (
        ("base", "base_correct"),
        ("expert", "expert_correct"),
        ("generated", "generated_correct"),
        ("oracle", "oracle_correct"),
    ):
        domains: dict[str, dict[str, Any]] = {}
        for domain, domain_tasks in TASKS_BY_DOMAIN.items():
            correct = sum(task_metrics[task][field] for task in domain_tasks)
            total = sum(task_metrics[task]["total"] for task in domain_tasks)
            domains[domain] = {
                "correct": correct,
                "total": total,
                "accuracy": correct / total,
            }
        arms[arm] = {
            "domains": domains,
            "macro_accuracy": sum(item["accuracy"] for item in domains.values())
            / len(domains),
            "solved": sum(task_metrics[task][field] for task in MAIN_TASKS),
            "total": sum(task_metrics[task]["total"] for task in MAIN_TASKS),
            "aime": {
                "correct": task_metrics["aime"][field],
                "total": task_metrics["aime"]["total"],
                "accuracy": task_metrics["aime"][field] / task_metrics["aime"]["total"],
            },
        }
    strongest_name = max(
        ("base", "expert"),
        key=lambda name: (arms[name]["macro_accuracy"], arms[name]["solved"]),
    )
    strongest = arms[strongest_name]
    treatment = arms["generated"]
    deltas = {
        domain: treatment["domains"][domain]["accuracy"]
        - strongest["domains"][domain]["accuracy"]
        for domain in TASKS_BY_DOMAIN
    }
    gates = {
        "code_at_least_30_of_40": treatment["domains"]["code"]["correct"] >= 30,
        "macro_delta_at_least_three_points": treatment["macro_accuracy"]
        >= strongest["macro_accuracy"] + 0.03,
        "solved_delta_at_least_fifteen": treatment["solved"]
        >= strongest["solved"] + 15,
        "improves_at_least_three_domains": sum(delta > 0 for delta in deltas.values())
        >= 3,
        "no_domain_regression_over_two_points": min(deltas.values()) >= -0.02,
    }
    return {
        "tasks": task_metrics,
        "arms": arms,
        "comparison": {
            "strongest_single_lineage": strongest_name,
            "domain_accuracy_deltas": deltas,
            "macro_delta": treatment["macro_accuracy"] - strongest["macro_accuracy"],
            "solved_delta": treatment["solved"] - strongest["solved"],
            "gates": gates,
            "gate_pass": all(gates.values()),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    if args.report.exists() or args.candidates_output.exists():
        raise VCR1ProductError("VCR1 product evaluation output already exists")
    source_report = json.loads(args.source_holdout_report.read_text(encoding="utf-8"))
    if (
        source_report.get("schema") != SOURCE_REPORT_SCHEMA
        or source_report.get("gate_pass") is not True
    ):
        raise VCR1ProductError("VCR1 source holdout did not pass")
    data_report = json.loads(args.data_report.read_text(encoding="utf-8"))
    if (
        data_report.get("schema") != DATA_REPORT_SCHEMA
        or data_report.get("status") != "complete"
    ):
        raise VCR1ProductError("VCR1 product data report is incomplete")
    if Path(data_report.get("output", "")).resolve() != args.data.resolve():
        raise VCR1ProductError("VCR1 product data path differs")
    if data_report.get("output_sha256") != sha256_file(args.data):
        raise VCR1ProductError("VCR1 product data hash differs")
    rows = load_rows(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, loader = _load_model(
        args.model_root, args.adapter_checkpoint, "multimodal"
    )
    stop_token_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    generated_tokens = exhausted = 0
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
        for row, completion, (token_count, hit_cap) in zip(
            batch, completions, usage, strict=True
        ):
            score = score_completion(
                row["assessor"], completion, code_timeout=args.code_timeout
            )
            results.append(
                {
                    "identity_sha256": row["identity_sha256"],
                    "task": row["task"],
                    "completion": completion,
                    "generated_tokens": token_count,
                    "max_token_exhausted": hit_cap,
                    **score,
                }
            )
            generated_tokens += token_count
            exhausted += int(hit_cap)
        processed = min(start + len(batch), len(rows))
        if processed % 32 == 0 or processed == len(rows):
            print(f"[vcr1-product] {processed}/{len(rows)}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    candidates_sha256 = _atomic_lines(args.candidates_output, results)
    summary = summarize(rows, results)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "model_root": str(args.model_source_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": loader,
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": sha256_file(args.adapter_checkpoint),
        "adapter_metadata": metadata,
        "source_holdout_report": str(args.source_holdout_report.resolve()),
        "source_holdout_report_sha256": sha256_file(args.source_holdout_report),
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "data_report": str(args.data_report.resolve()),
        "data_report_sha256": sha256_file(args.data_report),
        "runtime_fields": ["question"],
        "generated_tokens": generated_tokens,
        "generated_tokens_per_second": generated_tokens / elapsed,
        "max_token_exhausted": exhausted,
        "elapsed_seconds": elapsed,
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
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--source-holdout-report", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--code-timeout", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=2026080817)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["comparison"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
