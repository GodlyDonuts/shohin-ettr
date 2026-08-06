"""Train the bounded verified temporal-correction gate."""

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

from diverge_vcr1_data import tokenize_correction_example
from diverge_vcr1_product import VCR1ProductModel, save_vcr1_checkpoint
from hf_product_reasoning_train import load_product_backbone


BOARD_SCHEMA = "shohin-diverge-vcr1-pair-board-v1"
REPORT_SCHEMA = "shohin-diverge-vcr1-training-report-v1"


class VCR1TrainError(RuntimeError):
    """The temporal-correction training contract was violated."""


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
        raise VCR1TrainError(f"refusing to replace report: {path}")
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
            raise VCR1TrainError("VCR1 board schema or split differs")
        identity = str(row.get("identity_sha256") or "")
        if identity in identities:
            raise VCR1TrainError("VCR1 board repeats an identity")
        identities.add(identity)
        rows.append(row)
    if not rows:
        raise VCR1TrainError("VCR1 board is empty")
    return rows


def _tokenize_pair(tokenizer: Any, row: dict[str, Any], args: argparse.Namespace):
    wrong = tokenize_correction_example(
        tokenizer,
        str(row["question"]),
        str(row["wrong_draft"]),
        str(row["target"]),
        max_sequence_length=args.max_sequence_length,
        workspace_slots=args.workspace_slots,
    )
    correct = tokenize_correction_example(
        tokenizer,
        str(row["question"]),
        str(row["correct_draft"]),
        str(row["target"]),
        max_sequence_length=args.max_sequence_length,
        workspace_slots=args.workspace_slots,
    )
    if wrong is None or correct is None:
        raise VCR1TrainError("admitted VCR1 row no longer fits")
    if len(wrong.response_ids) != int(row["target_tokens"]):
        raise VCR1TrainError("VCR1 target token accounting differs")
    return wrong, correct


@torch.no_grad()
def _teacher_draft_probe(
    model: VCR1ProductModel,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    model.source.eval()
    totals: dict[str, float] = {
        "loss": 0.0,
        "language_loss": 0.0,
        "validity_accuracy": 0.0,
        "wrong_correction_strength": 0.0,
        "correct_correction_strength": 0.0,
    }
    count = min(args.development_probe_rows, len(rows))
    for row in rows[:count]:
        wrong, correct = _tokenize_pair(tokenizer, row, args)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = model.forward_batch(
                [wrong.prompt_ids, correct.prompt_ids],
                [wrong.response_ids, correct.response_ids],
                [wrong.question_mask, correct.question_mask],
                [wrong.draft_mask, correct.draft_mask],
                [False, True],
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
        raise VCR1TrainError(f"output already exists: {args.output}")
    if _sha256_file(args.data) != args.data_sha256:
        raise VCR1TrainError("VCR1 training board hash differs")
    if _sha256_file(args.development_data) != args.development_data_sha256:
        raise VCR1TrainError("VCR1 development board hash differs")
    if _sha256_file(args.source_checkpoint) != args.source_checkpoint_sha256:
        raise VCR1TrainError("VCR1 source checkpoint hash differs")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if not getattr(tokenizer, "is_fast", False):
        raise VCR1TrainError("VCR1 requires a fast tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    backbone, resolved_loader = load_product_backbone(
        args.model_root,
        args.model_loader,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = VCR1ProductModel(
        backbone,
        args.source_checkpoint,
        source_checkpoint_sha256=args.source_checkpoint_sha256,
        source_revision=args.model_revision,
        role_blind=args.arm == "role_blind",
        workspace_width=args.workspace_width,
        workspace_slots=args.workspace_slots,
        recurrent_steps=args.recurrent_steps,
        attention_heads=args.attention_heads,
        ff_multiplier=args.ff_multiplier,
        validity_weight=args.validity_weight,
        correction_margin_weight=args.correction_margin_weight,
        correction_margin=args.correction_margin,
    ).to("cuda:0")
    model.train()
    model.source.eval()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable or any(
        parameter.requires_grad for parameter in model.source.parameters()
    ):
        raise VCR1TrainError("VCR1 optimizer boundary differs")
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
    reactor_initial_sha256 = _state_sha256(model.reactor)
    source_initial_sha256 = model.frozen_source_sha256()
    initial_probe = _teacher_draft_probe(model, tokenizer, development_rows, args)

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
        "workspace_config": {
            "backbone_width": model.workspace_config.backbone_width,
            "workspace_width": model.workspace_config.workspace_width,
            "workspace_slots": model.workspace_config.workspace_slots,
            "recurrent_steps": model.workspace_config.recurrent_steps,
            "attention_heads": model.workspace_config.attention_heads,
            "ff_multiplier": model.workspace_config.ff_multiplier,
        },
        "trainable_parameters": model.trainable_parameter_count(),
        "validity_weight": args.validity_weight,
        "correction_margin_weight": args.correction_margin_weight,
        "correction_margin": args.correction_margin,
        "reactor_initial_sha256": reactor_initial_sha256,
        "source_initial_sha256": source_initial_sha256,
    }

    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    update = 0
    microstep = 0
    charged_tokens = 0
    trace: list[dict[str, float | int]] = []
    sums = {
        "loss": 0.0,
        "language_loss": 0.0,
        "validity_loss": 0.0,
        "margin_loss": 0.0,
        "validity_accuracy": 0.0,
        "wrong_correction_strength": 0.0,
        "correct_correction_strength": 0.0,
        "mean_step_delta": 0.0,
    }
    while update < args.updates:
        row = rows[microstep % len(rows)]
        wrong, correct = _tokenize_pair(tokenizer, row, args)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = model.forward_batch(
                [wrong.prompt_ids, correct.prompt_ids],
                [wrong.response_ids, correct.response_ids],
                [wrong.question_mask, correct.question_mask],
                [wrong.draft_mask, correct.draft_mask],
                [False, True],
                tokenizer.pad_token_id,
            )
            objective = loss / args.gradient_accumulation
        objective.backward()
        charged_tokens += int(metrics["charged_tokens"])
        sums["loss"] += float(loss.detach())
        for key in sums:
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
            raise VCR1TrainError("VCR1 gradient became nonfinite")
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
        sums = {key: 0.0 for key in sums}

    final_probe = _teacher_draft_probe(model, tokenizer, development_rows, args)
    source_final_sha256 = model.frozen_source_sha256()
    source_unchanged = source_initial_sha256 == source_final_sha256
    if not source_unchanged:
        raise VCR1TrainError("protected source changed during VCR1 training")
    reactor_final_sha256 = _state_sha256(model.reactor)
    checkpoint = args.output / f"checkpoint_{update:07d}.pt"
    save_vcr1_checkpoint(checkpoint, model, optimizer, update, metadata)
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
        "source_unchanged": source_unchanged,
        "reactor_final_sha256": reactor_final_sha256,
        "reactor_changed": reactor_initial_sha256 != reactor_final_sha256,
        "initial_teacher_draft_probe": initial_probe,
        "final_teacher_draft_probe": final_probe,
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
    parser.add_argument("--arm", choices=("vcr1", "role_blind"), required=True)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=2800)
    parser.add_argument("--development-probe-rows", type=int, default=32)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--workspace-width", type=int, default=384)
    parser.add_argument("--workspace-slots", type=int, default=8)
    parser.add_argument("--recurrent-steps", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--ff-multiplier", type=int, default=4)
    parser.add_argument("--validity-weight", type=float, default=0.20)
    parser.add_argument("--correction-margin-weight", type=float, default=0.10)
    parser.add_argument("--correction-margin", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026080603)
    parser.add_argument("--data-seed", type=int, default=2026080603)
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()
    positive = (
        args.updates,
        args.gradient_accumulation,
        args.max_rows,
        args.development_probe_rows,
        args.max_sequence_length,
        args.learning_rate,
        args.workspace_width,
        args.workspace_slots,
        args.recurrent_steps,
        args.attention_heads,
        args.ff_multiplier,
        args.validity_weight,
        args.correction_margin_weight,
        args.correction_margin,
        args.log_interval,
    )
    if any(value <= 0 for value in positive):
        parser.error("VCR1 training dimensions must be positive")
    return args


def main() -> int:
    report = run(parse_args())
    print(
        f"[vcr1-train] arm={report['arm']} updates={report['updates']} "
        f"tokens/s={report['charged_tokens_per_second']:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
