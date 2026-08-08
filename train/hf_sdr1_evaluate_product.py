#!/usr/bin/env python3
"""Conditionally score standalone SDR1 on the preserved product board."""

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
from hf_vcr1_evaluate_product import summarize as lineage_summary
from hf_vcr1_evaluate_reviser import _atomic_json, _atomic_lines, sha256_file

DATA_SCHEMA = "shohin-sdr1-product-eval-v1"
DATA_REPORT_SCHEMA = "shohin-sdr1-product-data-report-v1"
SOURCE_REPORT_SCHEMA = "shohin-sdr1-source-only-evaluation-v1"
REPORT_SCHEMA = "shohin-sdr1-product-evaluation-v1"
TASKS = {
    "aime",
    "bbh_logic",
    "gpqa",
    "gsm8k",
    "humaneval",
    "math500",
    "mbpp",
}


class SDR1ProductError(RuntimeError):
    """The conditional standalone product contract differs."""


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    identities = {row.get("identity_sha256") for row in rows}
    if not rows or len(identities) != len(rows):
        raise SDR1ProductError("SDR1 product identity coverage differs")
    for row in rows:
        if row.get("schema") != DATA_SCHEMA or row.get("task") not in TASKS:
            raise SDR1ProductError("SDR1 product schema/task differs")
        if row.get("runtime_fields") != ["question"]:
            raise SDR1ProductError("SDR1 product runtime fields differ")
        if row.get("candidate_text_visible") is not False:
            raise SDR1ProductError("SDR1 product candidate boundary differs")
    if {row["task"] for row in rows} != TASKS:
        raise SDR1ProductError("SDR1 product task coverage differs")
    return rows


def summarize(
    rows: list[dict[str, Any]], results: list[dict[str, Any]]
) -> dict[str, Any]:
    summary = lineage_summary(rows, results)
    treatment = summary["arms"]["generated"]
    expert = summary["arms"]["expert"]
    deltas = {
        domain: treatment["domains"][domain]["accuracy"]
        - expert["domains"][domain]["accuracy"]
        for domain in treatment["domains"]
    }
    gates = {
        "code_at_least_30_of_40": treatment["domains"]["code"]["correct"] >= 30,
        "macro_at_least_0_70": treatment["macro_accuracy"] >= 0.70,
        "solved_at_least_350": treatment["solved"] >= 350,
        "improves_at_least_three_domains": sum(delta > 0 for delta in deltas.values())
        >= 3,
        "no_domain_regression_over_two_points": min(deltas.values()) >= -0.02,
    }
    summary["standalone_comparison"] = {
        "reference": "expert",
        "domain_accuracy_deltas": deltas,
        "gates": gates,
        "gate_pass": all(gates.values()),
    }
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    if args.report.exists() or args.candidates_output.exists():
        raise SDR1ProductError("SDR1 product output already exists")
    source_report = json.loads(args.source_holdout_report.read_text(encoding="utf-8"))
    if (
        source_report.get("schema") != SOURCE_REPORT_SCHEMA
        or source_report.get("split") != "holdout"
        or source_report.get("gate_pass") is not True
    ):
        raise SDR1ProductError("SDR1 source holdout did not pass")
    data_report = json.loads(args.data_report.read_text(encoding="utf-8"))
    if (
        data_report.get("schema") != DATA_REPORT_SCHEMA
        or data_report.get("status") != "complete"
        or data_report.get("candidate_text_visible") is not False
    ):
        raise SDR1ProductError("SDR1 product data report is incomplete")
    if Path(data_report.get("output", "")).resolve() != args.data.resolve():
        raise SDR1ProductError("SDR1 product data path differs")
    if data_report.get("output_sha256") != sha256_file(args.data):
        raise SDR1ProductError("SDR1 product data hash differs")
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
            print(f"[sdr1-product] {processed}/{len(rows)}", flush=True)
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
        "candidate_text_visible": False,
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
    parser.add_argument("--seed", type=int, default=2026080816)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["standalone_comparison"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
