#!/usr/bin/env python3
"""Evaluate one frozen PCF1 revision or matched control arm."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from hf_product_reasoning_eval import (
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
    _render_prompt,
)
from pcf1_code_sandbox import (
    BWRAP_SHA256,
    SANDBOX_CONFIG_SHA256,
    atomic_json as sandbox_atomic_json,
    mbpp_allocation_setup_receipts_sha256,
    qualify_allocation,
    qualify_mbpp_assessor_setups,
    score_completion,
)
from pcf1_environment import validate_environment_receipt

EVAL_SCHEMA = "shohin-pcf1-eval-v1"
DATA_REPORT_SCHEMA = "shohin-pcf1-data-report-v1"
REPORT_SCHEMA = "shohin-pcf1-evaluation-v1"
CANDIDATE_SCHEMA = "shohin-pcf1-candidate-v1"
TASKS = ("math500", "bbh_logic", "mbpp")
SPLITS = ("calibration", "confirmation")
ARMS = ("revision", "unchanged", "self_refinement")
PINNED_MODEL_REVISION = "81eaece1948f3875421d9a45bc55487d10e2d894"
EVALUATION_SEED = 2026080816
EXPECTED_TEXT_LAYER_COUNT = 34
EXPECTED_LORA_LAYER_INDICES = (30, 31, 32, 33)


class PCF1EvaluationError(RuntimeError):
    """The PCF1 model, data, generation, or sealed-split contract differs."""


def shard_bounds(
    total: int, shard_index: int, shard_count: int, batch_size: int
) -> tuple[int, int]:
    """Partition rows on full generation batches without legacy-lane imports."""

    if (
        total <= 0
        or shard_count <= 0
        or not 0 <= shard_index < shard_count
        or batch_size <= 0
    ):
        raise PCF1EvaluationError("PCF1 shard geometry is invalid")
    batch_count = (total + batch_size - 1) // batch_size
    batch_start = batch_count * shard_index // shard_count
    batch_end = batch_count * (shard_index + 1) // shard_count
    start = min(total, batch_start * batch_size)
    end = min(total, batch_end * batch_size)
    if start >= end:
        raise PCF1EvaluationError("PCF1 shard is empty")
    return start, end


def self_refinement_prompt(row: dict[str, Any]) -> str:
    """Frozen matched control prompt, kept native to the authorized PCF1 graph."""

    source = str(row["source_prompt"])
    draft = str(row["internal_draft"]["completion"])
    return (
        "Review the attempted solution below for mistakes, then solve the original "
        "problem correctly. Do not only critique the attempt.\n\n"
        f"Original problem:\n{source}\n\nAttempt:\n{draft}\n\n"
        "Follow the original problem's requested output format."
    )


def nonpadding_prompt_tokens(tokenizer: Any, rendered: list[str]) -> int:
    """Count the exact nonpadding input tokens used by generation."""

    encoded = tokenizer(rendered, padding=True, return_attention_mask=True)
    attention = encoded.get("attention_mask")
    if hasattr(attention, "tolist"):
        attention = attention.tolist()
    if (
        not isinstance(attention, list)
        or len(attention) != len(rendered)
        or any(not isinstance(row, list) or not row for row in attention)
        or any(value not in (0, 1) for row in attention for value in row)
    ):
        raise PCF1EvaluationError("PCF1 prompt attention geometry differs")
    return sum(int(value) for row in attention for value in row)


def validate_adapter_trainables(
    model: Any, metadata: Any
) -> dict[str, int | str | list[int]]:
    """Bind the loaded adapter to the qualified final-four Ministral layers."""

    if (
        not isinstance(metadata, dict)
        or metadata.get("arm") != "baseline"
        or metadata.get("model_revision") != PINNED_MODEL_REVISION
        or metadata.get("model_loader") != "multimodal"
        or metadata.get("lora_layers") != 4
        or metadata.get("lora_rank") != 8
        or metadata.get("lora_alpha") != 16
        or metadata.get("lora_scope") != "token_mixer"
    ):
        raise PCF1EvaluationError("PCF1 qualified adapter metadata differs")
    try:
        layer_count = len(model.text_model.layers)
        trainable = sorted(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
    except (AttributeError, TypeError) as error:
        raise PCF1EvaluationError("PCF1 loaded adapter structure differs") from error
    names = [name for name, _ in trainable]
    trainable_parameters = sum(int(parameter.numel()) for _, parameter in trainable)
    name_sha256 = hashlib.sha256("\n".join(names).encode()).hexdigest()
    layer_indices = sorted(
        {
            int(match.group(1))
            for name in names
            for match in [re.search(r"(?:^|\.)layers\.(\d+)\.", name)]
            if match is not None
        }
    )
    if (
        layer_count != EXPECTED_TEXT_LAYER_COUNT
        or layer_indices != list(EXPECTED_LORA_LAYER_INDICES)
        or trainable_parameters <= 0
        or metadata.get("trainable_parameters") != trainable_parameters
        or metadata.get("trainable_parameter_name_sha256") != name_sha256
    ):
        raise PCF1EvaluationError("PCF1 qualified adapter trainables differ")
    return {
        "trainable_parameters": trainable_parameters,
        "trainable_parameter_name_sha256": name_sha256,
        "lora_layer_indices": layer_indices,
    }


def model_visible_runtime_fields(arm: str) -> list[str]:
    if arm not in ARMS:
        raise PCF1EvaluationError("PCF1 arm differs")
    return (
        ["source_prompt", "internal_draft.completion"]
        if arm == "self_refinement"
        else ["question"]
    )


def reject_sealed_path(path: Path) -> None:
    rendered = f"{path}\n{path.resolve(strict=False)}".casefold()
    if any(word in rendered for word in ("holdout", "product", "public")):
        raise PCF1EvaluationError(f"sealed path supplied to PCF1: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise PCF1EvaluationError(f"refusing existing PCF1 output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise PCF1EvaluationError(f"refusing existing PCF1 output: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise PCF1EvaluationError(f"refusing existing PCF1 output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise PCF1EvaluationError(f"refusing existing PCF1 output: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def summarize_results(
    sources: list[dict[str, Any]], results: list[dict[str, Any]]
) -> dict[str, dict[str, int]]:
    by_identity = {row.get("identity_sha256"): row for row in results}
    if len(by_identity) != len(results):
        raise PCF1EvaluationError("PCF1 result identities are duplicated")
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for source in sources:
        result = by_identity.get(source["identity_sha256"])
        if (
            result is None
            or result.get("task") != source.get("task")
            or not isinstance(result.get("correct"), bool)
        ):
            raise PCF1EvaluationError("PCF1 result coverage differs")
        for domain in ("overall", str(source["task"])):
            buckets[domain]["total"] += 1
            buckets[domain]["generated_correct"] += int(result["correct"])
    if set(buckets) != {"overall", *TASKS}:
        raise PCF1EvaluationError("PCF1 result domain coverage differs")
    return {domain: dict(counter) for domain, counter in sorted(buckets.items())}


def load_rows(path: Path, split: str) -> list[dict[str, Any]]:
    reject_sealed_path(path)
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = row.get("identity_sha256")
            if row.get("schema") != EVAL_SCHEMA or row.get("split") != split:
                raise PCF1EvaluationError("PCF1 evaluation schema/split differs")
            if (
                not isinstance(identity, str)
                or len(identity) != 64
                or identity in identities
            ):
                raise PCF1EvaluationError("PCF1 identity is invalid or duplicated")
            identities.add(identity)
            if row.get("task") not in TASKS:
                raise PCF1EvaluationError("PCF1 task differs")
            expected_runtime_fields = (
                ["question", "source_prompt"]
                if split == "confirmation"
                else ["question"]
            )
            if row.get("runtime_fields") != expected_runtime_fields:
                raise PCF1EvaluationError("PCF1 runtime fields differ")
            if row.get("internal_draft_visible") is not True:
                raise PCF1EvaluationError("PCF1 internal draft boundary differs")
            if row.get("external_candidate_text_visible") is not False:
                raise PCF1EvaluationError("PCF1 external candidate boundary differs")
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                raise PCF1EvaluationError("PCF1 runtime prompt is empty")
            assessor = row.get("assessor")
            if split == "calibration":
                if (
                    not isinstance(assessor, dict)
                    or assessor.get("task") != row["task"]
                ):
                    raise PCF1EvaluationError("PCF1 assessor binding differs")
            elif (
                "assessor" in row
                or not isinstance(row.get("source_prompt"), str)
                or not row["source_prompt"].strip()
                or any(
                    field in row
                    for field in ("answer", "correct", "gold", "response", "target")
                )
            ):
                raise PCF1EvaluationError("PCF1 confirmation exposes supervision")
            draft = row.get("internal_draft")
            if (
                not isinstance(draft, dict)
                or draft.get("identity_sha256") != identity
                or not isinstance(draft.get("completion"), str)
                or draft["completion"] not in row["question"]
            ):
                raise PCF1EvaluationError("PCF1 draft is not prompt-bound")
            if row.get("candidates") != []:
                raise PCF1EvaluationError("PCF1 external candidate slot is not empty")
            rows.append(row)
    if not rows or {str(row["task"]) for row in rows} != set(TASKS):
        raise PCF1EvaluationError("PCF1 task coverage differs")
    if split == "confirmation" and len(rows) != 1289:
        raise PCF1EvaluationError("PCF1 confirmation cardinality differs")
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    environment = validate_environment_receipt(
        args.environment_receipt,
        args.environment_receipt_sha256,
        "train/hf_pcf1_evaluate.py",
    )
    import torch
    from transformers import AutoTokenizer

    for path in (
        args.model_root,
        args.model_source_root,
        args.adapter_checkpoint,
        args.data,
        args.data_report,
        args.candidates_output,
        args.report,
        args.environment_receipt,
    ):
        reject_sealed_path(path)
    if (
        args.model_revision != PINNED_MODEL_REVISION
        or args.model_loader != "multimodal"
        or args.seed != EVALUATION_SEED
    ):
        raise PCF1EvaluationError("PCF1 pinned evaluation settings differ")
    if args.report.exists() or args.candidates_output.exists():
        raise PCF1EvaluationError("PCF1 evaluation output already exists")
    sandbox_probe_sha256: str | None = None
    sandbox_receipt_sha256: str | None = None
    if args.split == "calibration":
        if args.sandbox_probe_output is None:
            raise PCF1EvaluationError("PCF1 calibration sandbox probe output is absent")
        reject_sealed_path(args.sandbox_probe_output)
        sandbox_receipt = qualify_allocation()
        sandbox_probe_sha256 = str(sandbox_receipt["probe_sha256"])
        sandbox_receipt_sha256 = sandbox_atomic_json(
            args.sandbox_probe_output, sandbox_receipt
        )
    elif args.sandbox_probe_output is not None:
        raise PCF1EvaluationError(
            "PCF1 confirmation must not run a code-scoring sandbox probe"
        )
    data_report = json.loads(args.data_report.read_text(encoding="utf-8"))
    expected = data_report.get("outputs", {}).get(args.split)
    if (
        data_report.get("schema") != DATA_REPORT_SCHEMA
        or data_report.get("status") != "complete"
        or data_report.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
        or not isinstance(expected, dict)
        or Path(str(expected.get("path", ""))).resolve() != args.data.resolve()
        or expected.get("sha256") != sha256_file(args.data)
    ):
        raise PCF1EvaluationError("PCF1 data receipt differs")
    all_rows = load_rows(args.data, args.split)
    row_start, row_end = shard_bounds(
        len(all_rows), args.shard_index, args.shard_count, args.batch_size
    )
    rows = all_rows[row_start:row_end]
    setup_qualifications: list[dict[str, Any]] = []
    if args.split == "calibration":
        setup_qualifications = qualify_mbpp_assessor_setups(
            [row["assessor"] for row in rows]
        )
    setup_qualifications_sha256 = (
        mbpp_allocation_setup_receipts_sha256(setup_qualifications)
        if args.split == "calibration"
        else None
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, adapter_metadata, model_loader = _load_model(
        args.model_root, args.adapter_checkpoint, args.model_loader
    )
    if model_loader != "multimodal":
        raise PCF1EvaluationError("PCF1 multimodal loader differs")
    trainable_receipt = validate_adapter_trainables(model, adapter_metadata)
    stop_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()

    results: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    started = time.monotonic()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        questions = (
            [self_refinement_prompt(row) for row in batch]
            if args.arm == "self_refinement"
            else [str(row["question"]) for row in batch]
        )
        rendered = [
            _render_prompt(tokenizer, question, True, False) for question in questions
        ]
        counters["prompt_tokens"] += nonpadding_prompt_tokens(tokenizer, rendered)
        completions, usage = _generate_completions(
            model,
            tokenizer,
            rendered,
            True,
            "greedy",
            768,
            stop_ids,
        )
        for row, completion, (token_count, exhausted) in zip(
            batch, completions, usage, strict=True
        ):
            candidate = {
                "schema": CANDIDATE_SCHEMA,
                "arm": args.arm,
                "identity_sha256": row["identity_sha256"],
                "task": row["task"],
                "completion": completion,
                "generated_tokens": token_count,
                "max_token_exhausted": exhausted,
            }
            if args.split == "calibration":
                score = score_completion(row["assessor"], completion)
                candidate.update(score)
                counters["correct"] += int(score["correct"])
                counters["sandbox_executions"] += int(row["task"] == "mbpp")
                execution = score.get("execution")
                counters["capability_policy_rejections"] += int(
                    isinstance(execution, dict)
                    and execution.get("candidate_policy_passed") is False
                )
            results.append(candidate)
            counters["rows"] += 1
            counters["generated_tokens"] += token_count
            counters["max_token_exhausted"] += int(exhausted)
            counters["empty_completions"] += int(not completion.strip())
        processed = min(start + len(batch), len(rows))
        if processed % 32 == 0 or processed == len(rows):
            print(f"[pcf1-{args.arm}] {processed}/{len(rows)}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    counters["sandbox_executions"] += 0
    counters["capability_policy_rejections"] += 0
    candidates_sha256 = atomic_lines(args.candidates_output, results)
    metrics = (
        summarize_results(rows, results)
        if args.split == "calibration" and args.shard_count == 1
        else None
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "arm": args.arm,
        "split": args.split,
        "model_root": str(args.model_source_root.resolve()),
        "loaded_model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": model_loader,
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": sha256_file(args.adapter_checkpoint),
        "adapter_metadata": adapter_metadata,
        "adapter_metadata_sha256": hashlib.sha256(
            json.dumps(adapter_metadata, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        **trainable_receipt,
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "data_report": str(args.data_report.resolve()),
        "data_report_sha256": sha256_file(args.data_report),
        "runtime_fields": model_visible_runtime_fields(args.arm),
        "assessor_fields_visible_to_model": False,
        "assessment_mode": (
            "calibration_immediate"
            if args.split == "calibration"
            else "confirmation_deferred"
        ),
        "assessor_board_access_count": 0,
        "generation_mode": "greedy",
        "max_new_tokens": 768,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "row_start": row_start,
        "row_end": row_end,
        "full_row_count": len(all_rows),
        "candidates_output": str(args.candidates_output.resolve()),
        "candidates_sha256": candidates_sha256,
        "metrics": metrics,
        "counters": dict(sorted(counters.items())),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "code_sandbox_config_sha256": SANDBOX_CONFIG_SHA256,
        "code_sandbox_binary_sha256": BWRAP_SHA256,
        "code_sandbox_probe_sha256": sandbox_receipt_sha256,
        "code_sandbox_probe_result_sha256": sandbox_probe_sha256,
        "sandbox_receipt_sha256": sandbox_receipt_sha256,
        "code_sandbox_status": (
            "passed"
            if args.split == "calibration"
            else "not_applicable_no_code_scoring"
        ),
        "code_sandbox_probe_passed": (
            sandbox_probe_sha256 is not None if args.split == "calibration" else None
        ),
        "mbpp_allocation_setup_status": (
            "passed"
            if args.split == "calibration"
            else "not_applicable_no_code_scoring"
        ),
        "mbpp_allocation_setup_receipts": setup_qualifications,
        "mbpp_allocation_setup_receipt_count": len(setup_qualifications),
        "mbpp_allocation_setup_receipts_sha256": setup_qualifications_sha256,
        "environment_verified": True,
        "environment_receipt_sha256": args.environment_receipt_sha256,
        "environment_tree_sha256": environment["environment_tree"]["sha256"],
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-loader", choices=("multimodal",), default="multimodal")
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sandbox-probe-output", type=Path)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--environment-receipt-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=EVALUATION_SEED)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.shard_count <= 0:
        parser.error("PCF1 batch/shard geometry must be positive")
    return args


def main() -> int:
    report = run(parse_args())
    print(
        json.dumps(
            {
                "arm": report["arm"],
                "rows": report["counters"]["rows"],
                "split": report["split"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
