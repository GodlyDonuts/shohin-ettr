#!/usr/bin/env python3
"""Force wrong DSEO1 action prefixes to test downstream causal control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import torch

from eval_dseo1_paired_action import answer_correct, load_diagnostic
from hf_product_reasoning_eval import (
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
    _render_prompt,
)
from train_dseo1_paired_action import sha256_file


class DSEO1InterventionError(RuntimeError):
    """The frozen DSEO1 action intervention contract differs."""


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def normalized_trajectory(text: str) -> str:
    return " ".join(text.split()).casefold()


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.output.exists() or args.pairs != 128:
        raise DSEO1InterventionError("DSEO1 intervention output or population differs")
    baseline = json.loads(args.baseline.read_text())
    if (
        baseline.get("schema") != "shohin-dseo1-paired-evaluation-merged-v1"
        or baseline.get("status") != "complete"
        or baseline.get("arm") != "aligned"
        or baseline.get("adapter_checkpoint_sha256") != sha256_file(args.adapter_checkpoint)
        or baseline.get("data_sha256") != sha256_file(args.data)
    ):
        raise DSEO1InterventionError("DSEO1 aligned baseline differs")
    pairs = load_diagnostic(args.data, args.data_report)
    selected = sorted(pairs, key=lambda pair: pair[0]["pair_identity_sha256"])[: args.pairs]
    baseline_by_id = {row["identity_sha256"]: row for row in baseline["results"]}
    interventions = []
    for pair in selected:
        fault_action = next(row["action"] for row in pair if row["pair_member"] == "fault")
        for row in pair:
            forced = "<KEEP>" if row["pair_member"] == "fault" else fault_action
            normal = baseline_by_id.get(row["identity_sha256"])
            if normal is None:
                raise DSEO1InterventionError("DSEO1 baseline row is absent")
            interventions.append((row, forced, normal))

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, loader = _load_model(
        args.model_root, args.adapter_checkpoint, "causal"
    )
    stop_ids = _generation_stop_token_ids(tokenizer)
    random.seed(2026080915)
    torch.manual_seed(2026080915)
    torch.cuda.manual_seed_all(2026080915)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    results = []
    for offset in range(0, len(interventions), args.batch_size):
        batch = interventions[offset : offset + args.batch_size]
        rendered = [
            _render_prompt(tokenizer, str(row["question"]), True, False)
            + forced
            + "\n"
            for row, forced, _ in batch
        ]
        completions, usages = _generate_completions(
            model,
            tokenizer,
            rendered,
            True,
            "greedy",
            args.max_new_tokens,
            stop_ids,
        )
        for (row, forced, normal), completion, (used, cap) in zip(
            batch, completions, usages, strict=True
        ):
            results.append(
                {
                    "identity_sha256": row["identity_sha256"],
                    "pair_identity_sha256": row["pair_identity_sha256"],
                    "pair_member": row["pair_member"],
                    "forced_action": forced,
                    "correct_action": row["action"],
                    "normal_answer_correct": bool(normal["answer_correct"]),
                    "forced_answer_correct": answer_correct(row, completion),
                    "normal_trajectory": normal["trajectory"],
                    "forced_trajectory": completion,
                    "trajectory_changed": normalized_trajectory(normal["trajectory"])
                    != normalized_trajectory(completion),
                    "forced_generated_tokens": used,
                    "forced_max_token_exhausted": cap,
                }
            )
    member = {}
    for name in ("clean", "fault"):
        rows = [row for row in results if row["pair_member"] == name]
        member[name] = {
            "rows": len(rows),
            "normal_answer_accuracy": sum(row["normal_answer_correct"] for row in rows) / len(rows),
            "forced_answer_accuracy": sum(row["forced_answer_correct"] for row in rows) / len(rows),
            "answer_accuracy_drop": (
                sum(row["normal_answer_correct"] for row in rows)
                - sum(row["forced_answer_correct"] for row in rows)
            )
            / len(rows),
            "trajectory_change_rate": sum(row["trajectory_changed"] for row in rows) / len(rows),
        }
    gates = {
        "forced_keep_fault_answer_drop_ge_0_05": member["fault"]["answer_accuracy_drop"] >= 0.05,
        "forced_fault_clean_answer_drop_ge_0_05": member["clean"]["answer_accuracy_drop"] >= 0.05,
        "fault_trajectory_change_rate_ge_0_50": member["fault"]["trajectory_change_rate"] >= 0.50,
        "clean_trajectory_change_rate_ge_0_50": member["clean"]["trajectory_change_rate"] >= 0.50,
    }
    elapsed = time.monotonic() - started
    report = {
        "schema": "shohin-dseo1-action-intervention-v1",
        "status": "complete",
        "passed": all(gates.values()),
        "thresholds_frozen_before_output": True,
        "holdout_used": False,
        "pair_count": args.pairs,
        "subset_rule": "lexicographically_first_128_pair_identity_sha256",
        "gates": gates,
        "member_metrics": member,
        "model_root": str((args.model_source_root or args.model_root).resolve()),
        "model_revision": args.model_revision,
        "model_loader": loader,
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": sha256_file(args.adapter_checkpoint),
        "adapter_metadata": metadata,
        "data_sha256": sha256_file(args.data),
        "data_report_sha256": sha256_file(args.data_report),
        "baseline_sha256": sha256_file(args.baseline),
        "max_new_tokens": args.max_new_tokens,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    args = parser.parse_args()
    if min(args.pairs, args.batch_size, args.max_new_tokens) <= 0:
        parser.error("DSEO1 intervention dimensions differ")
    report = run(args)
    print(json.dumps({"passed": report["passed"], "gates": report["gates"]}, sort_keys=True))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
