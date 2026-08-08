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
    _bounded_program_result,
    _generate_completions,
    _finalization_question,
    _generation_stop_token_ids,
    _humaneval_program,
    _load_model,
    _mbpp_program,
    _question,
    _render_prompt,
    _row_identity,
    _task_prompt,
    has_explicit_final_answer,
)

SCHEMA = "shohin-hf-product-reasoning-rollouts-v1"


class ProductRolloutError(RuntimeError):
    """The product rollout contract was violated."""


def score_completion(
    row: dict[str, Any],
    completion: str,
    code_timeout: float = 3.0,
) -> dict[str, Any]:
    task_name = str(row.get("task"))
    task = TASKS.get(task_name)
    if task is None:
        raise ProductRolloutError("rollout row task is unsupported")
    if task["kind"] == "code":
        program = (
            _humaneval_program(row, completion)
            if task_name == "humaneval"
            else _mbpp_program(row, completion)
        )
        execution = _bounded_program_result(program, code_timeout)
        return {
            "prediction": "pass" if execution["passed"] else "fail",
            "gold": "pass",
            "explicit_final_answer": True,
            "correct": bool(execution["passed"]),
            "program": program,
            "execution": execution,
        }
    prediction = task["extract"](completion)
    if task_name == "bbh_logic" and row.get("expected_answer_normalized") is not None:
        gold = str(row["expected_answer_normalized"])
    else:
        gold = task["gold"](row)
    explicit = has_explicit_final_answer(completion)
    correct = explicit and bool(task["match"](prediction, gold))
    return {
        "prediction": prediction,
        "gold": gold,
        "explicit_final_answer": explicit,
        "correct": correct,
        "program": None,
        "execution": None,
    }


def choose_positive(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    correct = [
        candidate
        for candidate in candidates
        if candidate["correct"]
        and not candidate.get("draft_max_token_exhausted", False)
    ]
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


def combine_finalization(
    completion: str,
    exhausted: bool,
    finalization: str | None,
) -> str:
    if exhausted and finalization and has_explicit_final_answer(finalization):
        return f"{completion.rstrip()}\n\n{finalization.strip()}"
    return completion


def validate_generation_geometry(mode: str, samples: int) -> None:
    if mode == "greedy":
        if samples != 1:
            raise ProductRolloutError("greedy rollout collection requires one sample")
        return
    if mode == "qwen-thinking":
        if not 2 <= samples <= 8:
            raise ProductRolloutError(
                "stochastic rollout collection requires 2--8 samples"
            )
        return
    raise ProductRolloutError("unsupported rollout generation mode")


def render_rollout_prompt(
    tokenizer: Any,
    task_prompt: str,
    *,
    adapter: bool,
    enable_thinking: bool,
    bare_prompt_style: str,
) -> str:
    """Render either the established adapter prompt or a direct bare-model task."""

    if adapter or bare_prompt_style == "reasoning":
        return _render_prompt(tokenizer, task_prompt, adapter, enable_thinking)
    if bare_prompt_style != "direct":
        raise ProductRolloutError("unsupported bare prompt style")
    from hf_product_reasoning_train import render_reasoning_messages

    return render_reasoning_messages(
        tokenizer,
        [{"role": "user", "content": task_prompt}],
        enable_thinking=enable_thinking,
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
    identities = [
        str(row.get("identity_sha256") or _row_identity(str(row.get("task")), row))
        for row in rows
    ]
    if len(set(identities)) != len(rows):
        raise ProductRolloutError("rollout slice identities differ")

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, adapter_metadata, model_loader = _load_model(
        args.model_root, args.adapter_checkpoint, args.model_loader
    )
    adapter = args.adapter_checkpoint is not None
    stop_token_ids = _generation_stop_token_ids(tokenizer)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()

    candidate_rows: list[dict[str, Any]] = []
    positive_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    started = time.monotonic()
    processed = 0
    for batch_start in range(0, len(rows), args.prompt_batch_size):
        batch = rows[batch_start : batch_start + args.prompt_batch_size]
        global_batch_start = args.skip + batch_start
        batch_seed = args.seed + global_batch_start * 1009
        torch.manual_seed(batch_seed)
        torch.cuda.manual_seed_all(batch_seed)
        rendered = [
            render_rollout_prompt(
                tokenizer,
                _task_prompt(str(row["task"]), row),
                adapter=adapter,
                enable_thinking=args.enable_thinking,
                bare_prompt_style=args.bare_prompt_style,
            )
            for row in batch
            for _ in range(args.samples)
        ]
        completions, usage = _generate_completions(
            model,
            tokenizer,
            rendered,
            adapter,
            args.generation_mode,
            args.max_new_tokens,
            stop_token_ids,
        )
        finalizations: list[str | None] = [None] * len(completions)
        finalization_usage: list[tuple[int, bool]] = [(0, False)] * len(completions)
        if args.finalize_exhausted and all(
            TASKS[str(row["task"])]["kind"] != "code" for row in batch
        ):
            finalize_indices = [
                index
                for index, (completion, (_, exhausted)) in enumerate(
                    zip(completions, usage, strict=True)
                )
                if exhausted and not has_explicit_final_answer(completion)
            ]
            torch.manual_seed(batch_seed + 1)
            torch.cuda.manual_seed_all(batch_seed + 1)
            for finalize_start in range(
                0, len(finalize_indices), args.finalize_batch_size
            ):
                chunk_indices = finalize_indices[
                    finalize_start : finalize_start + args.finalize_batch_size
                ]
                finalize_rendered = [
                    render_rollout_prompt(
                        tokenizer,
                        _finalization_question(
                            _task_prompt(
                                str(batch[index // args.samples]["task"]),
                                batch[index // args.samples],
                            ),
                            completions[index],
                        ),
                        adapter=adapter,
                        enable_thinking=False,
                        bare_prompt_style=args.bare_prompt_style,
                    )
                    for index in chunk_indices
                ]
                recovered, recovered_usage = _generate_completions(
                    model,
                    tokenizer,
                    finalize_rendered,
                    adapter,
                    "greedy",
                    args.finalize_max_new_tokens,
                    stop_token_ids,
                )
                for index, recovered_text, recovered_count in zip(
                    chunk_indices, recovered, recovered_usage, strict=True
                ):
                    finalizations[index] = recovered_text
                    finalization_usage[index] = recovered_count
        if len(completions) != len(batch) * args.samples:
            raise ProductRolloutError("generation batch cardinality differs")
        for batch_index, row in enumerate(batch):
            global_index = global_batch_start + batch_index
            identity = identities[batch_start + batch_index]
            row_seed = args.seed + global_index * 1009
            sample_start = batch_index * args.samples
            per_prompt: list[dict[str, Any]] = []
            for sample_index, (completion, (token_count, exhausted)) in enumerate(
                zip(
                    completions[sample_start : sample_start + args.samples],
                    usage[sample_start : sample_start + args.samples],
                    strict=True,
                )
            ):
                flat_index = sample_start + sample_index
                finalization = finalizations[flat_index]
                finalize_token_count, finalize_exhausted = finalization_usage[
                    flat_index
                ]
                scoring_completion = combine_finalization(
                    completion, exhausted, finalization
                )
                score = score_completion(
                    row, scoring_completion, code_timeout=args.code_timeout
                )
                candidate = {
                    "schema": SCHEMA,
                    "identity_sha256": identity,
                    "question": _question(row),
                    "task": row["task"],
                    "training_group": row.get("training_group"),
                    "sample_index": sample_index,
                    "batch_seed": batch_seed,
                    "row_seed": row_seed,
                    "completion": scoring_completion,
                    "draft_completion": completion,
                    "finalization": finalization,
                    "generated_tokens": token_count + finalize_token_count,
                    "draft_generated_tokens": token_count,
                    "finalization_generated_tokens": finalize_token_count,
                    "max_token_exhausted": (
                        finalize_exhausted if finalization is not None else exhausted
                    ),
                    "draft_max_token_exhausted": exhausted,
                    "finalization_max_token_exhausted": finalize_exhausted,
                    **score,
                }
                candidate_rows.append(candidate)
                per_prompt.append(candidate)
                counters["candidates"] += 1
                counters["correct_candidates"] += int(candidate["correct"])
                counters["explicit_candidates"] += int(
                    candidate["explicit_final_answer"]
                )
                counters["max_token_exhausted"] += int(candidate["max_token_exhausted"])
                counters["draft_max_token_exhausted"] += int(exhausted)
                counters["finalization_attempts"] += int(finalization is not None)
                counters["finalization_max_token_exhausted"] += int(finalize_exhausted)
                counters["draft_generated_tokens"] += token_count
                counters["finalization_generated_tokens"] += finalize_token_count
                counters["generated_tokens"] += token_count + finalize_token_count
            positive = choose_positive(per_prompt)
            if positive is not None:
                negatives = [
                    candidate for candidate in per_prompt if not candidate["correct"]
                ]
                positive_rows.append(
                    {
                        "question": _question(row),
                        "response": positive["draft_completion"],
                        "answer": row.get("answer"),
                        "expected_answer_normalized": row.get(
                            "expected_answer_normalized"
                        ),
                        "training_group": row.get("training_group"),
                        "verification": (
                            "student_execution_verified_v1"
                            if TASKS[str(row["task"])]["kind"] == "code"
                            else "student_exact_answer_match_v1"
                        ),
                        "source_identity_sha256": identity,
                        "source_adapter_checkpoint": (
                            str(args.adapter_checkpoint.resolve())
                            if args.adapter_checkpoint is not None
                            else None
                        ),
                        "source_model_root": str(
                            (args.model_source_root or args.model_root).resolve()
                        ),
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
        processed += len(batch)
        if processed % 8 == 0 or processed == len(rows):
            print(
                f"[product-rollout] {processed}/{len(rows)} "
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
        "adapter_checkpoint": (
            str(args.adapter_checkpoint.resolve())
            if args.adapter_checkpoint is not None
            else None
        ),
        "adapter_checkpoint_sha256": (
            hashlib.sha256(args.adapter_checkpoint.read_bytes()).hexdigest()
            if args.adapter_checkpoint is not None
            else None
        ),
        "adapter_metadata": adapter_metadata,
        "data": str(args.data.resolve()),
        "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "skip": args.skip,
        "count": args.count,
        "samples": args.samples,
        "generation_mode": args.generation_mode,
        "prompt_batch_size": args.prompt_batch_size,
        "finalize_exhausted": args.finalize_exhausted,
        "finalize_max_new_tokens": args.finalize_max_new_tokens,
        "finalize_batch_size": args.finalize_batch_size,
        "enable_thinking": args.enable_thinking,
        "bare_prompt_style": args.bare_prompt_style,
        "code_timeout": args.code_timeout,
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
    parser.add_argument(
        "--model-loader", choices=("auto", "causal", "multimodal"), default="auto"
    )
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--positives-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument(
        "--generation-mode",
        choices=("greedy", "qwen-thinking"),
        default="qwen-thinking",
    )
    parser.add_argument("--prompt-batch-size", type=int, default=1)
    parser.add_argument("--finalize-exhausted", action="store_true")
    parser.add_argument("--finalize-max-new-tokens", type=int, default=64)
    parser.add_argument("--finalize-batch-size", type=int, default=32)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--bare-prompt-style",
        choices=("reasoning", "direct"),
        default="reasoning",
    )
    parser.add_argument("--code-timeout", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    args = parser.parse_args()
    if (
        args.prompt_batch_size <= 0
        or args.prompt_batch_size > 64
        or args.finalize_max_new_tokens <= 0
        or args.finalize_batch_size <= 0
        or args.max_new_tokens <= 0
        or args.code_timeout <= 0
    ):
        parser.error(
            "prompt batch size must be in [1, 64], and "
            "generation limit must be positive"
        )
    try:
        validate_generation_geometry(args.generation_mode, args.samples)
    except ProductRolloutError as exc:
        parser.error(str(exc))
    return args


def main() -> int:
    args = parse_args()
    report = run(args)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
