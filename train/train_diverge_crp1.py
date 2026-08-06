"""Train the bounded complete-trace causal-revision gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import torch

from diverge_crp1_data import tokenize_revision_example
from diverge_crp1_product import CRP1ProductModel, save_crp1_checkpoint
from hf_product_reasoning_train import load_product_backbone


BOARD_SCHEMA = "shohin-diverge-crp1-board-v1"
REPORT_SCHEMA = "shohin-diverge-crp1-training-report-v1"


class CRP1TrainError(RuntimeError):
    """The causal-revision training contract was violated."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        raw = tensor.detach().to("cpu").contiguous().view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise CRP1TrainError(f"refusing to replace report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _read_board(path: Path, expected_split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema") != BOARD_SCHEMA or row.get("split") != expected_split:
            raise CRP1TrainError("CRP1 board schema or split differs")
        identity = str(row.get("identity_sha256") or "")
        if len(identity) != 64 or identity in identities:
            raise CRP1TrainError("CRP1 board identity differs")
        if not 1 <= int(row.get("error_index", 0)) <= int(row.get("depth", 0)) - 2:
            raise CRP1TrainError("CRP1 first-error certificate differs")
        if row.get("answer") == row.get("wrong_answer"):
            raise CRP1TrainError("CRP1 wrong trace has the correct answer")
        identities.add(identity)
        rows.append(row)
    if not rows:
        raise CRP1TrainError("CRP1 board is empty")
    return rows


def _tokenize_pair(tokenizer: Any, row: dict[str, Any], args: argparse.Namespace):
    wrong = tokenize_revision_example(
        tokenizer,
        str(row["problem"]),
        list(map(str, row["wrong_steps"])),
        f"Final answer: \\boxed{{{row['wrong_answer']}}}",
        str(row["wrong_target"]),
        max_sequence_length=args.max_sequence_length,
        workspace_slots=args.workspace_slots,
    )
    correct = tokenize_revision_example(
        tokenizer,
        str(row["problem"]),
        list(map(str, row["correct_steps"])),
        f"Final answer: \\boxed{{{row['answer']}}}",
        str(row["correct_target"]),
        max_sequence_length=args.max_sequence_length,
        workspace_slots=args.workspace_slots,
    )
    if wrong is None or correct is None:
        raise CRP1TrainError("admitted CRP1 row no longer fits")
    if len(wrong.response_ids) != int(row["wrong_target_tokens"]) or len(
        correct.response_ids
    ) != int(row["correct_target_tokens"]):
        raise CRP1TrainError("CRP1 target token accounting differs")
    return wrong, correct


@torch.no_grad()
def _probe(
    model: CRP1ProductModel,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    model.source.eval()
    totals = {
        "loss": 0.0,
        "language_loss": 0.0,
        "localization_loss": 0.0,
        "localization_accuracy": 0.0,
        "wrong_localization_accuracy": 0.0,
        "correct_no_error_accuracy": 0.0,
        "mean_selected_no_error_separation": 0.0,
    }
    count = min(args.development_probe_rows, len(rows))
    for row in rows[:count]:
        wrong, correct = _tokenize_pair(tokenizer, row, args)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = model.forward_batch(
                [wrong.prompt_ids, correct.prompt_ids],
                [wrong.response_ids, correct.response_ids],
                [wrong.problem_mask, correct.problem_mask],
                [wrong.step_masks, correct.step_masks],
                [wrong.final_mask, correct.final_mask],
                [int(row["error_index"]), 0],
                tokenizer.pad_token_id,
            )
        totals["loss"] += float(loss)
        for key in totals:
            if key != "loss":
                totals[key] += float(metrics[key])
    model.train()
    model.source.eval()
    return {key: value / count for key, value in totals.items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.output.exists():
        raise CRP1TrainError(f"output already exists: {args.output}")
    if _sha256_file(args.data) != args.data_sha256:
        raise CRP1TrainError("CRP1 training board hash differs")
    if _sha256_file(args.development_data) != args.development_data_sha256:
        raise CRP1TrainError("CRP1 development board hash differs")
    if _sha256_file(args.source_checkpoint) != args.source_checkpoint_sha256:
        raise CRP1TrainError("CRP1 source checkpoint hash differs")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if not getattr(tokenizer, "is_fast", False):
        raise CRP1TrainError("CRP1 requires a fast tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    backbone, resolved_loader = load_product_backbone(
        args.model_root,
        args.model_loader,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = CRP1ProductModel(
        backbone,
        args.source_checkpoint,
        source_checkpoint_sha256=args.source_checkpoint_sha256,
        source_revision=args.model_revision,
        unguarded=args.arm == "unguarded",
        workspace_width=args.workspace_width,
        workspace_slots=args.workspace_slots,
        recurrent_steps=args.recurrent_steps,
        attention_heads=args.attention_heads,
        ff_multiplier=args.ff_multiplier,
        max_trace_steps=args.max_trace_steps,
        localization_weight=args.localization_weight,
    ).to("cuda:0")
    model.train()
    model.source.eval()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable or any(
        parameter.requires_grad for parameter in model.source.parameters()
    ):
        raise CRP1TrainError("CRP1 optimizer boundary differs")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )
    rows = _read_board(args.data, "train")
    development_rows = _read_board(args.development_data, "development")
    if args.max_rows < len(rows):
        rows = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{args.data_seed}\0{row['identity_sha256']}".encode()
            ).hexdigest(),
        )[: args.max_rows]
    random.Random(args.data_seed).shuffle(rows)
    packet_initial_sha256 = _state_sha256(model.packet)
    source_initial_sha256 = model.frozen_source_sha256()
    initial_probe = _probe(model, tokenizer, development_rows, args)
    metadata = {
        "architecture": model.architecture,
        "arm": args.arm,
        "model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": resolved_loader,
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": args.source_checkpoint_sha256,
        "source_checkpoint_update": model.source_update,
        "data": str(args.data.resolve()),
        "data_sha256": args.data_sha256,
        "development_data": str(args.development_data.resolve()),
        "development_data_sha256": args.development_data_sha256,
        "selected_rows": len(rows),
        "seed": args.seed,
        "data_seed": args.data_seed,
        "packet_config": {
            "backbone_width": model.packet_config.backbone_width,
            "workspace_width": model.packet_config.workspace_width,
            "workspace_slots": model.packet_config.workspace_slots,
            "recurrent_steps": model.packet_config.recurrent_steps,
            "attention_heads": model.packet_config.attention_heads,
            "ff_multiplier": model.packet_config.ff_multiplier,
            "max_trace_steps": model.packet_config.max_trace_steps,
        },
        "trainable_parameters": model.trainable_parameter_count(),
        "localization_weight": args.localization_weight,
        "packet_initial_sha256": packet_initial_sha256,
        "source_initial_sha256": source_initial_sha256,
    }
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    update = 0
    microstep = 0
    charged_tokens = 0
    trace: list[dict[str, float | int]] = []
    metric_keys = (
        "loss",
        "language_loss",
        "localization_loss",
        "localization_accuracy",
        "wrong_localization_accuracy",
        "correct_no_error_accuracy",
        "mean_selected_no_error_separation",
        "mean_step_delta",
    )
    sums = {key: 0.0 for key in metric_keys}
    while update < args.updates:
        row = rows[microstep % len(rows)]
        wrong, correct = _tokenize_pair(tokenizer, row, args)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = model.forward_batch(
                [wrong.prompt_ids, correct.prompt_ids],
                [wrong.response_ids, correct.response_ids],
                [wrong.problem_mask, correct.problem_mask],
                [wrong.step_masks, correct.step_masks],
                [wrong.final_mask, correct.final_mask],
                [int(row["error_index"]), 0],
                tokenizer.pad_token_id,
            )
            objective = loss / args.gradient_accumulation
        objective.backward()
        charged_tokens += int(metrics["charged_tokens"])
        sums["loss"] += float(loss.detach())
        for key in metric_keys:
            if key != "loss":
                sums[key] += float(metrics[key])
        microstep += 1
        if microstep % args.gradient_accumulation:
            continue
        progress = update / max(args.updates - 1, 1)
        learning_rate = args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(gradient_norm):
            raise CRP1TrainError("CRP1 gradient became nonfinite")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update += 1
        if update == 1 or update % args.log_interval == 0:
            elapsed = time.monotonic() - started
            divisor = float(args.gradient_accumulation)
            event: dict[str, float | int] = {
                "update": update,
                **{key: value / divisor for key, value in sums.items()},
                "gradient_norm": float(gradient_norm),
                "learning_rate": learning_rate,
                "charged_tokens": charged_tokens,
                "charged_tokens_per_second": charged_tokens / elapsed,
            }
            trace.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)
        sums = {key: 0.0 for key in metric_keys}

    final_probe = _probe(model, tokenizer, development_rows, args)
    source_final_sha256 = model.frozen_source_sha256()
    if source_initial_sha256 != source_final_sha256:
        raise CRP1TrainError("protected source changed during CRP1 training")
    packet_final_sha256 = _state_sha256(model.packet)
    checkpoint = args.output / f"checkpoint_{update:07d}.pt"
    save_crp1_checkpoint(checkpoint, model, optimizer, update, metadata)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        **metadata,
        "updates": update,
        "gradient_accumulation": args.gradient_accumulation,
        "max_sequence_length": args.max_sequence_length,
        "learning_rate": args.learning_rate,
        "charged_tokens": charged_tokens,
        "elapsed_seconds": elapsed,
        "charged_tokens_per_second": charged_tokens / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "source_final_sha256": source_final_sha256,
        "source_unchanged": True,
        "packet_final_sha256": packet_final_sha256,
        "packet_changed": packet_initial_sha256 != packet_final_sha256,
        "initial_development_probe": initial_probe,
        "final_development_probe": final_probe,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "trace": trace,
    }
    _atomic_json(args.output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-loader", choices=("auto", "causal"), default="causal")
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--development-data", type=Path, required=True)
    parser.add_argument("--development-data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("guarded", "unguarded"), required=True)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=4800)
    parser.add_argument("--development-probe-rows", type=int, default=48)
    parser.add_argument("--workspace-width", type=int, default=256)
    parser.add_argument("--workspace-slots", type=int, default=6)
    parser.add_argument("--recurrent-steps", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--ff-multiplier", type=int, default=4)
    parser.add_argument("--max-trace-steps", type=int, default=12)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--localization-weight", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026080604)
    parser.add_argument("--data-seed", type=int, default=2026080604)
    args = parser.parse_args()
    integer_fields = (
        args.updates,
        args.gradient_accumulation,
        args.max_rows,
        args.development_probe_rows,
        args.workspace_width,
        args.workspace_slots,
        args.recurrent_steps,
        args.attention_heads,
        args.ff_multiplier,
        args.max_trace_steps,
        args.max_sequence_length,
        args.log_interval,
    )
    if any(value <= 0 for value in integer_fields) or args.learning_rate <= 0:
        parser.error("CRP1 training dimensions must be positive")
    if args.localization_weight < 0:
        parser.error("CRP1 localization weight must be nonnegative")
    return args


def main() -> int:
    report = run(parse_args())
    print(
        f"[crp1-train] arm={report['arm']} updates={report['updates']} "
        f"probe={report['final_development_probe']['localization_accuracy']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
