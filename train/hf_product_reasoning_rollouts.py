#!/usr/bin/env python3
"""Sample and exactly verify fresh product-reasoning trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

from hf_product_reasoning_eval import (
    TASKS,
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
    _render_prompt,
    has_explicit_final_answer,
)


SCHEMA = "shohin-hf-product-reasoning-rollouts-v1"


class ProductRolloutError(RuntimeError):
    """The product rollout contract was violated."""


def score_completion(row: dict[str, Any], completion: str) -> dict[str, Any]:
    task_name = str(row.get("task"))
    task = TASKS.get(task_name)
    if task is None or task["kind"] != "answer":
        raise ProductRolloutError("rollout row task is unsupported")
    prediction = task["extract"](completion)
    gold = task["gold"](row)
    explicit = has_explicit_final_answer(completion)
    correct = explicit and bool(task["match"](prediction, gold))
    return {
        "prediction": prediction,
        "gold": gold,
        "explicit_final_answer": explicit,
        "correct": correct,
    }


def choose_positive(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    correct = [candidate for candidate in candidates if candidate["correct"]]
    if not correct:
        return None
    return min(
        correct,
        key=lambda candidate: (
            candidate["generated_tokens"],
            len(candidate["completion"]),
            candidate["sample_index"],
        ),
    )


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise ProductRolloutError(f"refusing to replace output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
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
        raise ProductRolloutError(f"refusing to replace report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    data_bytes = args.data.read_bytes()
    all_rows = [json.loads(line) for line in data_bytes.splitlines() if line.strip()]
    if args.skip < 0 or args.count <= 0 or args.skip + args.count > len(all_rows):
        raise ProductRolloutError("requested rollout slice is outside the bank")
    rows = all_rows[args.skip : args.skip + args.count]
    identities = [str(row.get("identity_sha256")) for row in rows]
    if any(not identity for identity in identities) or len(set(identities)) != len(rows):
        raise ProductRolloutError("rollout slice identities differ")

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, adapter_metadata, model_loader = _load_model(
        args.model_root, args.adapter_checkpoint, args.model_loader
    )
    stop_token_ids = _generation_stop_token_ids(tokenizer)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()

    candidate_rows: list[dict[str, Any]] = []
    positive_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    started = time.monotonic()
    for local_index, row in enumerate(rows):
        global_index = args.skip + local_index
        row_seed = args.seed + global_index * 1009
        torch.manual_seed(row_seed)
        torch.cuda.manual_seed_all(row_seed)
        rendered = _render_prompt(tokenizer, row["question"], True, False)
        completions, usage = _generate_completions(
            model,
            tokenizer,
            [rendered] * args.samples,
            True,
            "qwen-thinking",
            args.max_new_tokens,
            stop_token_ids,
        )
        per_prompt: list[dict[str, Any]] = []
        for sample_index, (completion, (token_count, exhausted)) in enumerate(
            zip(completions, usage, strict=True)
        ):
            score = score_completion(row, completion)
            candidate = {
                "schema": SCHEMA,
                "identity_sha256": row["identity_sha256"],
                "question": row["question"],
                "task": row["task"],
                "training_group": row["training_group"],
                "sample_index": sample_index,
                "seed": row_seed,
                "completion": completion,
                "generated_tokens": token_count,
                "max_token_exhausted": exhausted,
                **score,
            }
            candidate_rows.append(candidate)
            per_prompt.append(candidate)
            counters["candidates"] += 1
            counters["correct_candidates"] += int(candidate["correct"])
            counters["explicit_candidates"] += int(candidate["explicit_final_answer"])
            counters["max_token_exhausted"] += int(exhausted)
            counters["generated_tokens"] += token_count
        positive = choose_positive(per_prompt)
        if positive is not None:
            negatives = [candidate for candidate in per_prompt if not candidate["correct"]]
            positive_rows.append(
                {
                    "question": row["question"],
                    "response": positive["completion"],
                    "answer": row["answer"],
                    "expected_answer_normalized": row["expected_answer_normalized"],
                    "training_group": row["training_group"],
                    "verification": "student_exact_answer_match_v1",
                    "source_identity_sha256": row["identity_sha256"],
                    "source_adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
                    "chosen_sample_index": positive["sample_index"],
                    "rejected_response": (
                        min(
                            negatives,
                            key=lambda candidate: (
                                candidate["generated_tokens"],
                                candidate["sample_index"],
                            ),
                        )["completion"]
                        if negatives
                        else None
                    ),
                }
            )
            counters["positive_prompts"] += 1
        counters["prompts"] += 1
        if (local_index + 1) % 8 == 0 or local_index + 1 == len(rows):
            print(
                f"[product-rollout] {local_index + 1}/{len(rows)} "
                f"positive={counters['positive_prompts']} "
                f"correct={counters['correct_candidates']}/{counters['candidates']}",
                flush=True,
            )

    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    candidate_sha256 = _atomic_lines(args.candidates_output, candidate_rows)
    positives_sha256 = _atomic_lines(args.positives_output, positive_rows)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "model_root": str((args.model_source_root or args.model_root).resolve()),
        "loaded_model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": model_loader,
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_metadata": adapter_metadata,
        "data": str(args.data.resolve()),
        "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "skip": args.skip,
        "count": args.count,
        "samples": args.samples,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "counters": dict(sorted(counters.items())),
        "prompt_positive_rate": counters["positive_prompts"] / counters["prompts"],
        "candidate_accuracy": counters["correct_candidates"] / counters["candidates"],
        "elapsed_seconds": elapsed,
        "generated_tokens_per_second": counters["generated_tokens"] / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "candidates_output": str(args.candidates_output.resolve()),
        "candidates_sha256": candidate_sha256,
        "positives_output": str(args.positives_output.resolve()),
        "positives_sha256": positives_sha256,
    }
    _atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-loader", choices=("auto", "causal", "multimodal"), default="auto")
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--positives-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    args = parser.parse_args()
    if args.samples <= 1 or args.samples > 8 or args.max_new_tokens <= 0:
        parser.error("samples must be in [2, 8] and generation limit must be positive")
    return args


def main() -> int:
    args = parse_args()
    report = run(args)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
