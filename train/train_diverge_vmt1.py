"""Frozen fit gate for DIVERGE-VMT1 verified multi-trajectory matching."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch

from diverge_vmt1_product import VMT1ProductModel, frozen_parameter_sha256
from diverge_vmt1_workspace import vmt1_architecture_sha256
from hf_product_reasoning_train import (
    PRODUCT_SYSTEM_PROMPT,
    ProductReasoningTrainError,
    _atomic_json,
    _save_checkpoint,
    load_product_backbone,
    render_reasoning_messages,
)


VMT1_BOARD_SCHEMA = "shohin-diverge-vmt1-board-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_exact_board(
    path: Path,
    report_path: Path,
    *,
    expected_rows: int,
) -> tuple[list[dict[str, Any]], str, str]:
    if not path.is_file() or not report_path.is_file():
        raise ProductReasoningTrainError("VMT1 board or report is missing")
    data_sha256 = sha256_file(path)
    report_sha256 = sha256_file(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != VMT1_BOARD_SCHEMA or report.get("status") != "complete":
        raise ProductReasoningTrainError("VMT1 board report schema differs")
    if report.get("output_sha256") != data_sha256:
        raise ProductReasoningTrainError("VMT1 board hash differs from its report")
    if int(report.get("rows", -1)) != expected_rows:
        raise ProductReasoningTrainError("VMT1 board report row count differs")

    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            required = {
                "schema",
                "identity_sha256",
                "question",
                "responses",
                "correct",
                "correct_index",
                "training_group",
                "token_accounting",
            }
            if not required.issubset(row) or row["schema"] != VMT1_BOARD_SCHEMA:
                raise ProductReasoningTrainError("VMT1 board row schema differs")
            identity = str(row["identity_sha256"])
            if identity in identities:
                raise ProductReasoningTrainError("VMT1 board repeats a prompt")
            identities.add(identity)
            responses = row["responses"]
            correctness = [bool(value) for value in row["correct"]]
            if (
                not isinstance(responses, list)
                or len(responses) != 2
                or any(not str(response).strip() for response in responses)
                or len(correctness) != 2
                or sum(correctness) != 1
            ):
                raise ProductReasoningTrainError("VMT1 pair contract differs")
            correct_index = 0 if correctness[0] else 1
            if int(row["correct_index"]) != correct_index:
                raise ProductReasoningTrainError("VMT1 correct index differs")
            if row["training_group"] not in {"math", "science"}:
                raise ProductReasoningTrainError("VMT1 training group differs")
            rows.append(row)
    if len(rows) != expected_rows:
        raise ProductReasoningTrainError("VMT1 exact board row count differs")
    cell_counts: dict[tuple[str, int], int] = {}
    for row in rows:
        cell = (str(row["training_group"]), int(row["correct_index"]))
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
    expected_per_cell = expected_rows // 4
    expected_cells = {
        (group, correct_index): expected_per_cell
        for group in ("math", "science")
        for correct_index in (0, 1)
    }
    if expected_rows % 4 or cell_counts != expected_cells:
        raise ProductReasoningTrainError("VMT1 board cells are not exactly balanced")
    return rows, data_sha256, report_sha256


def tokenize_exact_board(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_sequence_length: int,
    workspace_slots: int,
) -> list[dict[str, Any]]:
    if tokenizer.eos_token_id is None:
        raise ProductReasoningTrainError("VMT1 tokenizer has no EOS token")
    tokenized: list[dict[str, Any]] = []
    for row in rows:
        rendered = render_reasoning_messages(
            tokenizer,
            [
                {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                {"role": "user", "content": str(row["question"])},
            ],
            enable_thinking=False,
        )
        prompt = tokenizer.encode(rendered, add_special_tokens=False)
        raw_responses = [
            tokenizer.encode(str(response), add_special_tokens=False)
            for response in row["responses"]
        ]
        totals = [
            len(prompt) + len(response) + 1 + workspace_slots
            for response in raw_responses
        ]
        accounting = row["token_accounting"]
        if (
            int(accounting.get("prompt_tokens", -1)) != len(prompt)
            or list(accounting.get("response_tokens", []))
            != [len(response) for response in raw_responses]
            or int(accounting.get("workspace_slots", -1)) != workspace_slots
            or int(accounting.get("maximum_total_tokens", -1)) != max(totals)
        ):
            raise ProductReasoningTrainError("VMT1 token accounting drifted")
        if max(totals) > max_sequence_length:
            raise ProductReasoningTrainError("VMT1 row would require truncation")
        if not prompt or any(not response for response in raw_responses):
            raise ProductReasoningTrainError("VMT1 tokenization produced an empty row")
        responses = [response + [tokenizer.eos_token_id] for response in raw_responses]
        tokenized.append(
            {
                "identity_sha256": str(row["identity_sha256"]),
                "training_group": str(row["training_group"]),
                "correct_index": int(row["correct_index"]),
                "prompt_tokens": prompt,
                "response_tokens": responses,
                "correct": [bool(value) for value in row["correct"]],
            }
        )
    return tokenized


def _batch_fields(
    records: list[dict[str, Any]],
) -> tuple[list[list[int]], list[list[list[int]]], list[list[bool]]]:
    return (
        [record["prompt_tokens"] for record in records],
        [record["response_tokens"] for record in records],
        [record["correct"] for record in records],
    )


def audit_fit(
    model: VMT1ProductModel,
    records: list[dict[str, Any]],
    *,
    pad_token_id: int,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    row_metrics: dict[str, list[Any]] = {
        "selected_correct_response_nll_rows": [],
        "matched_trace_cosine_rows": [],
        "crossed_trace_cosine_rows": [],
        "selector_correct_rows": [],
        "swapped_selector_correct_rows": [],
        "internal_trajectory_cosine_rows": [],
        "selected_indices": [],
        "best_assignments": [],
        "validity_logits_rows": [],
    }
    losses: list[float] = []
    charged_tokens: list[int] = []
    with torch.inference_mode():
        for offset in range(0, len(records), batch_size):
            batch = records[offset : offset + batch_size]
            prompt_rows, response_pairs, correctness = _batch_fields(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = model.forward_batch(
                    prompt_rows,
                    response_pairs,
                    correctness,
                    pad_token_id,
                )
            losses.append(float(loss.detach()))
            for key in row_metrics:
                row_metrics[key].extend(metrics[key])
            charged_tokens.extend(
                len(record["response_tokens"][record["correct_index"]])
                for record in batch
            )
    model.train()
    selected = [bool(value) for value in row_metrics["selector_correct_rows"]]
    swapped = [bool(value) for value in row_metrics["swapped_selector_correct_rows"]]
    result: dict[str, Any] = {
        "loss": sum(losses) / len(losses),
        **row_metrics,
        "mean_selected_correct_response_nll": sum(
            row_metrics["selected_correct_response_nll_rows"]
        )
        / len(records),
        "token_weighted_selected_correct_response_nll": sum(
            nll * tokens
            for nll, tokens in zip(
                row_metrics["selected_correct_response_nll_rows"],
                charged_tokens,
                strict=True,
            )
        )
        / sum(charged_tokens),
        "mean_matched_trace_cosine": sum(row_metrics["matched_trace_cosine_rows"])
        / len(records),
        "mean_crossed_trace_cosine": sum(row_metrics["crossed_trace_cosine_rows"])
        / len(records),
        "mean_internal_trajectory_cosine": sum(
            row_metrics["internal_trajectory_cosine_rows"]
        )
        / len(records),
        "selector_correct": sum(selected),
        "swapped_selector_correct": sum(swapped),
        "selector_accuracy": sum(selected) / len(records),
        "swapped_selector_accuracy": sum(swapped) / len(records),
        "charged_tokens_rows": charged_tokens,
        "charged_tokens": sum(charged_tokens),
    }
    result["finite"] = all(
        math.isfinite(float(value))
        for key, value in result.items()
        if isinstance(value, (int, float)) and key != "finite"
    )
    return result


def reduce_fit_gate(
    before: dict[str, Any],
    after: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    frozen_parameters_unchanged: bool,
    training_finite: bool,
) -> dict[str, Any]:
    improved = [
        after_value < before_value
        for before_value, after_value in zip(
            before["selected_correct_response_nll_rows"],
            after["selected_correct_response_nll_rows"],
            strict=True,
        )
    ]
    orientation_correct = {0: 0, 1: 0}
    orientation_total = {0: 0, 1: 0}
    for record, correct in zip(records, after["selector_correct_rows"], strict=True):
        orientation = int(record["correct_index"])
        orientation_total[orientation] += 1
        orientation_correct[orientation] += int(bool(correct))
    trace_advantage = (
        after["mean_matched_trace_cosine"] - after["mean_crossed_trace_cosine"]
    )
    swap_drop = after["selector_accuracy"] - after["swapped_selector_accuracy"]
    checks = {
        "all_16_selected_nll_improve": len(improved) == 16 and all(improved),
        "selector_at_least_15_of_16": after["selector_correct"] >= 15,
        "selector_at_least_7_of_8_correct_0": orientation_correct[0] >= 7,
        "selector_at_least_7_of_8_correct_1": orientation_correct[1] >= 7,
        "matched_trace_cosine_at_least_0_85": (
            after["mean_matched_trace_cosine"] >= 0.85
        ),
        "matched_crossed_advantage_at_least_0_10": trace_advantage >= 0.10,
        "internal_trajectory_cosine_at_most_0_95": (
            after["mean_internal_trajectory_cosine"] <= 0.95
        ),
        "finite": bool(before["finite"] and after["finite"] and training_finite),
        "frozen_parameters_unchanged": frozen_parameters_unchanged,
        "swapped_selector_drop_at_least_0_25": swap_drop >= 0.25,
    }
    return {
        "checks": checks,
        "qualified": all(checks.values()),
        "improved_rows": sum(improved),
        "orientation_correct": {
            str(key): value for key, value in orientation_correct.items()
        },
        "orientation_total": {
            str(key): value for key, value in orientation_total.items()
        },
        "trace_advantage": trace_advantage,
        "swapped_selector_accuracy_drop": swap_drop,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.output.exists():
        raise ProductReasoningTrainError(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows, data_sha256, board_report_sha256 = load_exact_board(
        args.data,
        args.board_report,
        expected_rows=args.expected_rows,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    backbone, resolved_model_loader = load_product_backbone(
        args.model_root,
        args.model_loader,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = VMT1ProductModel(
        backbone,
        lora_layers=args.lora_layers,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        latent_width=args.latent_width,
        trajectory_slots=args.trajectory_slots,
        recurrent_steps=args.recurrent_steps,
        attention_heads=args.attention_heads,
        ff_multiplier=args.ff_multiplier,
        assignment_temperature=args.assignment_temperature,
        validity_margin=args.validity_margin,
        trace_weight=args.trace_weight,
        validity_weight=args.validity_weight,
        halting_weight=args.halting_weight,
    ).to("cuda:0")
    records = tokenize_exact_board(
        tokenizer,
        rows,
        max_sequence_length=args.max_sequence_length,
        workspace_slots=model.sequence_workspace_slots(),
    )
    metadata = {
        "architecture": model.architecture,
        "arm": model.arm,
        "model_root": str((args.model_source_root or args.model_root).resolve()),
        "loaded_model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": resolved_model_loader,
        "backbone_layout": model.backbone_layout,
        "data": str(args.data.resolve()),
        "data_sha256": data_sha256,
        "board_report": str(args.board_report.resolve()),
        "board_report_sha256": board_report_sha256,
        "selected_rows": len(records),
        "seed": args.seed,
        "lora_layers": args.lora_layers,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_projection_count": model.lora_projection_count,
        "trainable_parameters": model.trainable_parameter_count(),
        "workspace_config": asdict(model.workspace_config),
        "workspace_architecture_sha256": vmt1_architecture_sha256(
            model.workspace_config
        ),
        "assignment_temperature": args.assignment_temperature,
        "validity_margin": args.validity_margin,
        "trace_weight": args.trace_weight,
        "validity_weight": args.validity_weight,
        "halting_weight": args.halting_weight,
        "selection_strategy": "validity",
        "gate_mode": args.gate_mode,
    }

    frozen_before = frozen_parameter_sha256(model)
    fit_before = audit_fit(
        model,
        records,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=args.batch_size,
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )
    optimizer.zero_grad(set_to_none=True)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    update = 0
    microstep = 0
    logical_tokens = 0
    candidate_tokens = 0
    trace_target_tokens = 0
    training_finite = True
    trace: list[dict[str, float | int]] = []
    final_checkpoint = args.output / f"checkpoint_{args.updates:07d}.pt"

    while update < args.updates:
        offset = (microstep * args.batch_size) % len(records)
        batch = [
            records[(offset + index) % len(records)] for index in range(args.batch_size)
        ]
        prompt_rows, response_pairs, correctness = _batch_fields(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = model.forward_batch(
                prompt_rows,
                response_pairs,
                correctness,
                tokenizer.pad_token_id,
            )
            scaled_loss = loss / args.gradient_accumulation
        if not torch.isfinite(loss):
            raise ProductReasoningTrainError("VMT1 training loss is nonfinite")
        scaled_loss.backward()
        logical_tokens += int(metrics["logical_charged_tokens"])
        candidate_tokens += int(metrics["candidate_charged_tokens"])
        trace_target_tokens += int(metrics["trace_target_tokens"])
        microstep += 1
        if microstep % args.gradient_accumulation:
            continue

        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        if not torch.isfinite(gradient_norm):
            raise ProductReasoningTrainError("VMT1 gradient norm is nonfinite")
        progress = update / max(args.updates - 1, 1)
        learning_rate = args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update += 1
        if update == 1 or update % args.log_interval == 0:
            elapsed = time.monotonic() - started
            event = {
                "update": update,
                "loss": float(loss.detach()),
                "assignment_loss": metrics["assignment_loss"],
                "validity_loss": metrics["validity_loss"],
                "correct_response_nll": metrics["correct_response_nll"],
                "selected_correct_response_nll": metrics[
                    "selected_correct_response_nll"
                ],
                "matched_trace_cosine": metrics["matched_trace_cosine"],
                "crossed_trace_cosine": metrics["crossed_trace_cosine"],
                "assignment_entropy": metrics["assignment_entropy"],
                "internal_trajectory_cosine": metrics["internal_trajectory_cosine"],
                "halting_loss": metrics["halting_loss"],
                "final_stop_probability": metrics["final_stop_probability"],
                "mean_step_delta": metrics["mean_step_delta"],
                "gradient_norm": float(gradient_norm),
                "learning_rate": learning_rate,
                "logical_charged_tokens": logical_tokens,
                "candidate_charged_tokens": candidate_tokens,
                "trace_target_tokens": trace_target_tokens,
                "logical_tokens_per_second": logical_tokens / elapsed,
                "candidate_tokens_per_second": candidate_tokens / elapsed,
            }
            training_finite = training_finite and all(
                math.isfinite(float(value)) for value in event.values()
            )
            trace.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)
        if update % args.checkpoint_interval == 0 or update == args.updates:
            _save_checkpoint(
                args.output / f"checkpoint_{update:07d}.pt",
                model,
                optimizer,
                update,
                metadata,
            )

    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    fit_after = audit_fit(
        model,
        records,
        pad_token_id=tokenizer.pad_token_id,
        batch_size=args.batch_size,
    )
    frozen_after = frozen_parameter_sha256(model)
    frozen_unchanged = frozen_before == frozen_after
    fit_gate = reduce_fit_gate(
        fit_before,
        fit_after,
        records,
        frozen_parameters_unchanged=frozen_unchanged,
        training_finite=training_finite,
    )
    report = {
        "schema": "shohin-diverge-vmt1-training-v1",
        "status": "complete",
        **metadata,
        "updates": update,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "max_sequence_length": args.max_sequence_length,
        "learning_rate": args.learning_rate,
        "logical_charged_tokens": logical_tokens,
        "candidate_charged_tokens": candidate_tokens,
        "trace_target_tokens": trace_target_tokens,
        "elapsed_seconds": elapsed,
        "logical_tokens_per_second": logical_tokens / elapsed,
        "candidate_tokens_per_second": candidate_tokens / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "frozen_parameter_sha256_before": frozen_before,
        "frozen_parameter_sha256_after": frozen_after,
        "frozen_parameters_unchanged": frozen_unchanged,
        "final_checkpoint": str(final_checkpoint.resolve()),
        "final_checkpoint_sha256": sha256_file(final_checkpoint),
        "fit_before": fit_before,
        "fit_after": fit_after,
        "fit_gate": fit_gate,
        "trace": trace,
    }
    _atomic_json(args.output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--model-loader", choices=("auto", "causal", "multimodal"), default="auto"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--board-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-mode", choices=("smoke", "fit"), required=True)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--expected-rows", type=int, default=16)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-layers", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--latent-width", type=int, default=384)
    parser.add_argument("--trajectory-slots", type=int, default=8)
    parser.add_argument("--recurrent-steps", type=int, default=8)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--ff-multiplier", type=int, default=2)
    parser.add_argument("--assignment-temperature", type=float, default=0.1)
    parser.add_argument("--validity-margin", type=float, default=1.0)
    parser.add_argument("--trace-weight", type=float, default=1.0)
    parser.add_argument("--validity-weight", type=float, default=0.25)
    parser.add_argument("--halting-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026080602)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    args = parser.parse_args()
    positive = (
        args.updates,
        args.batch_size,
        args.gradient_accumulation,
        args.expected_rows,
        args.max_sequence_length,
        args.lora_layers,
        args.lora_rank,
        args.latent_width,
        args.trajectory_slots,
        args.recurrent_steps,
        args.attention_heads,
        args.ff_multiplier,
        args.log_interval,
        args.checkpoint_interval,
    )
    if any(value <= 0 for value in positive) or args.learning_rate <= 0.0:
        parser.error("VMT1 dimensions and learning rate must be positive")
    if args.assignment_temperature <= 0.0 or args.validity_margin < 0.0:
        parser.error("VMT1 assignment settings differ")
    if min(args.trace_weight, args.validity_weight, args.halting_weight) < 0.0:
        parser.error("VMT1 loss weights must be nonnegative")
    if args.max_sequence_length <= args.trajectory_slots + 16:
        parser.error("maximum sequence length leaves no VMT1 token budget")
    return args


def main() -> int:
    args = parse_args()
    report = run(args)
    print(
        "[vmt1-train] "
        f"updates={report['updates']} "
        f"logical_tok/s={report['logical_tokens_per_second']:.1f} "
        f"qualified={report['fit_gate']['qualified']}",
        flush=True,
    )
    if args.gate_mode == "fit" and not report["fit_gate"]["qualified"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
