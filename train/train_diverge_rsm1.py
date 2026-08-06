"""Train the frozen-packet persistent discrete state replay component."""

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

from diverge_rsm1_data import (
    ReplaySupervision,
    ReplayTokens,
    build_replay_supervision,
    tokenize_replay_example,
)
from diverge_rsm1_product import (
    RSM1ProductModel,
    module_state_sha256,
    save_rsm1_checkpoint,
)
from hf_product_reasoning_train import load_product_backbone


BOARD_SCHEMA = "shohin-diverge-crp1-board-v1"
REPORT_SCHEMA = "shohin-diverge-rsm1-training-report-v1"


class RSM1TrainError(RuntimeError):
    """The persistent state replay training contract was violated."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise RSM1TrainError(f"refusing to replace report: {path}")
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
            raise RSM1TrainError("RSM1 board schema or split differs")
        identity = str(row.get("identity_sha256") or "")
        if len(identity) != 64 or identity in identities:
            raise RSM1TrainError("RSM1 board identity differs")
        if not 1 <= int(row.get("error_index", 0)) <= int(row.get("depth", 0)) - 2:
            raise RSM1TrainError("RSM1 first-error certificate differs")
        identities.add(identity)
        rows.append(row)
    if not rows:
        raise RSM1TrainError("RSM1 board is empty")
    return rows


def _tokenize_pair(
    tokenizer: Any,
    row: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[
    tuple[ReplayTokens, ReplaySupervision],
    tuple[ReplayTokens, ReplaySupervision],
]:
    wrong = tokenize_replay_example(
        tokenizer,
        row,
        row["wrong_steps"],
        f"Final answer: \\boxed{{{row['wrong_answer']}}}",
        max_sequence_length=args.max_sequence_length,
        packet_slots=args.packet_slots,
    )
    correct = tokenize_replay_example(
        tokenizer,
        row,
        row["correct_steps"],
        f"Final answer: \\boxed{{{row['answer']}}}",
        max_sequence_length=args.max_sequence_length,
        packet_slots=args.packet_slots,
    )
    if wrong is None or correct is None:
        raise RSM1TrainError("admitted RSM1 row no longer fits")
    wrong_supervision = build_replay_supervision(
        row,
        int(row["error_index"]),
        max_trace_steps=args.max_trace_steps,
    )
    correct_supervision = build_replay_supervision(
        row,
        0,
        max_trace_steps=args.max_trace_steps,
    )
    return (wrong, wrong_supervision), (correct, correct_supervision)


def _collate(
    examples: list[tuple[ReplayTokens, ReplaySupervision]],
) -> dict[str, Any]:
    if not examples:
        raise RSM1TrainError("RSM1 batch is empty")
    tokens = [example[0] for example in examples]
    supervision = [example[1] for example in examples]
    return {
        "prompt_rows": [value.prompt_ids for value in tokens],
        "problem_masks": [value.problem_mask for value in tokens],
        "packet_step_masks": [value.packet_step_masks for value in tokens],
        "operation_masks": [value.operation_masks for value in tokens],
        "final_masks": [value.final_mask for value in tokens],
        "selection_targets": [value.selection for value in supervision],
        "initial_targets": torch.tensor(
            [value.initial for value in supervision], dtype=torch.long
        ),
        "free_targets": torch.tensor(
            [value.free_targets for value in supervision], dtype=torch.long
        ),
        "free_active": torch.tensor(
            [value.free_active for value in supervision], dtype=torch.bool
        ),
        "oracle_predecessors": torch.tensor(
            [value.oracle_predecessors for value in supervision], dtype=torch.long
        ),
        "oracle_targets": torch.tensor(
            [value.oracle_targets for value in supervision], dtype=torch.long
        ),
        "oracle_active": torch.tensor(
            [value.oracle_active for value in supervision], dtype=torch.bool
        ),
        "terminal_targets": torch.tensor(
            [value.terminal for value in supervision], dtype=torch.long
        ),
    }


def _batch_examples(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[tuple[ReplayTokens, ReplaySupervision]]:
    examples: list[tuple[ReplayTokens, ReplaySupervision]] = []
    for row in rows:
        wrong, correct = _tokenize_pair(tokenizer, row, args)
        examples.extend((wrong, correct))
    return examples


@torch.no_grad()
def _probe(
    model: RSM1ProductModel,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    metric_keys = (
        "loss",
        "initial_loss",
        "free_loss",
        "oracle_loss",
        "initial_exact",
        "free_transition_exact",
        "trajectory_exact",
        "terminal_exact",
        "oracle_transition_exact",
        "mean_step_delta",
    )
    totals = {key: 0.0 for key in metric_keys}
    batches = 0
    count = min(args.development_probe_rows, len(rows))
    for start in range(0, count, args.pairs_per_microbatch):
        examples = _batch_examples(
            tokenizer,
            rows[start : start + args.pairs_per_microbatch],
            args,
        )
        batch = _collate(examples)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = model.forward_batch(
                **batch,
                pad_token_id=tokenizer.pad_token_id,
            )
        totals["loss"] += float(loss)
        for key in metric_keys:
            if key != "loss":
                totals[key] += float(metrics[key])
        batches += 1
    model.train()
    return {key: value / batches for key, value in totals.items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.output.exists():
        raise RSM1TrainError(f"output already exists: {args.output}")
    expected_files = (
        (args.data, args.data_sha256, "training board"),
        (
            args.development_data,
            args.development_data_sha256,
            "development board",
        ),
        (args.source_checkpoint, args.source_checkpoint_sha256, "source checkpoint"),
        (args.crp_checkpoint, args.crp_checkpoint_sha256, "CRP1 checkpoint"),
    )
    for path, expected, label in expected_files:
        if _sha256_file(path) != expected:
            raise RSM1TrainError(f"RSM1 {label} hash differs")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RSM1TrainError("RSM1 requires a fast tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    backbone, resolved_loader = load_product_backbone(
        args.model_root,
        args.model_loader,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = RSM1ProductModel(
        backbone,
        args.source_checkpoint,
        args.crp_checkpoint,
        source_checkpoint_sha256=args.source_checkpoint_sha256,
        crp_checkpoint_sha256=args.crp_checkpoint_sha256,
        source_revision=args.model_revision,
        packet_arm=args.packet_arm,
        state_width=args.state_width,
        state_slots=args.state_slots,
        packet_slots=args.packet_slots,
        max_trace_steps=args.max_trace_steps,
        attention_heads=args.attention_heads,
        ff_multiplier=args.ff_multiplier,
    ).to("cuda:0")
    model.train()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable or any(
        parameter.requires_grad for parameter in model.crp.parameters()
    ):
        raise RSM1TrainError("RSM1 optimizer boundary differs")
    if sum(parameter.numel() for parameter in trainable) != sum(
        parameter.numel() for parameter in model.replay.parameters()
    ):
        raise RSM1TrainError("RSM1 optimizer includes non-replay state")
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
    frozen_initial_sha256 = model.frozen_crp_sha256()
    replay_initial_sha256 = module_state_sha256(model.replay)
    initial_probe = _probe(model, tokenizer, development_rows, args)
    metadata = {
        "architecture": model.architecture,
        "packet_arm": args.packet_arm,
        "model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": resolved_loader,
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": args.source_checkpoint_sha256,
        "crp_checkpoint": str(args.crp_checkpoint.resolve()),
        "crp_checkpoint_sha256": args.crp_checkpoint_sha256,
        "crp_checkpoint_update": model.crp_checkpoint_update,
        "data": str(args.data.resolve()),
        "data_sha256": args.data_sha256,
        "development_data": str(args.development_data.resolve()),
        "development_data_sha256": args.development_data_sha256,
        "selected_rows": len(rows),
        "seed": args.seed,
        "data_seed": args.data_seed,
        "replay_config": {
            "backbone_width": model.replay_config.backbone_width,
            "state_width": model.replay_config.state_width,
            "state_slots": model.replay_config.state_slots,
            "packet_slots": model.replay_config.packet_slots,
            "max_trace_steps": model.replay_config.max_trace_steps,
            "attention_heads": model.replay_config.attention_heads,
            "ff_multiplier": model.replay_config.ff_multiplier,
            "state_vocab_size": model.replay_config.state_vocab_size,
        },
        "trainable_parameters": model.trainable_parameter_count(),
        "loss_weights": {
            "selected_boundary": 1.0 / 3.0,
            "free_running": 1.0 / 3.0,
            "oracle_one_step": 1.0 / 3.0,
        },
        "frozen_crp_initial_sha256": frozen_initial_sha256,
        "replay_initial_sha256": replay_initial_sha256,
    }

    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    update = 0
    microstep = 0
    source_tokens = 0
    state_target_tokens = 0
    trace: list[dict[str, float | int]] = []
    metric_keys = (
        "loss",
        "initial_loss",
        "free_loss",
        "oracle_loss",
        "initial_exact",
        "free_transition_exact",
        "trajectory_exact",
        "terminal_exact",
        "oracle_transition_exact",
        "mean_step_delta",
    )
    sums = {key: 0.0 for key in metric_keys}
    while update < args.updates:
        start = (microstep * args.pairs_per_microbatch) % len(rows)
        selected_rows = [
            rows[(start + offset) % len(rows)]
            for offset in range(args.pairs_per_microbatch)
        ]
        examples = _batch_examples(tokenizer, selected_rows, args)
        batch = _collate(examples)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = model.forward_batch(
                **batch,
                pad_token_id=tokenizer.pad_token_id,
            )
            objective = loss / args.gradient_accumulation
        objective.backward()
        source_tokens += int(metrics["source_tokens"])
        state_target_tokens += int(metrics["state_target_tokens"])
        sums["loss"] += float(loss.detach())
        for key in metric_keys:
            if key != "loss":
                sums[key] += float(metrics[key])
        microstep += 1
        if microstep % args.gradient_accumulation:
            continue
        progress = update / max(args.updates - 1, 1)
        learning_rate = args.learning_rate * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(gradient_norm):
            raise RSM1TrainError("RSM1 gradient became nonfinite")
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
                "source_tokens": source_tokens,
                "source_tokens_per_second": source_tokens / elapsed,
                "state_target_tokens": state_target_tokens,
            }
            trace.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)
        sums = {key: 0.0 for key in metric_keys}

    final_probe = _probe(model, tokenizer, development_rows, args)
    frozen_final_sha256 = model.frozen_crp_sha256()
    if frozen_initial_sha256 != frozen_final_sha256:
        raise RSM1TrainError("frozen source or CRP1 packet changed")
    replay_final_sha256 = module_state_sha256(model.replay)
    checkpoint = args.output / f"checkpoint_{update:07d}.pt"
    save_rsm1_checkpoint(checkpoint, model, optimizer, update, metadata)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        **metadata,
        "updates": update,
        "pairs_per_microbatch": args.pairs_per_microbatch,
        "gradient_accumulation": args.gradient_accumulation,
        "identities_per_update": (
            args.pairs_per_microbatch * args.gradient_accumulation
        ),
        "max_sequence_length": args.max_sequence_length,
        "learning_rate": args.learning_rate,
        "source_tokens": source_tokens,
        "state_target_tokens": state_target_tokens,
        "elapsed_seconds": elapsed,
        "source_tokens_per_second": source_tokens / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "frozen_crp_final_sha256": frozen_final_sha256,
        "frozen_crp_unchanged": True,
        "replay_final_sha256": replay_final_sha256,
        "replay_changed": replay_initial_sha256 != replay_final_sha256,
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
    parser.add_argument("--crp-checkpoint", type=Path, required=True)
    parser.add_argument("--crp-checkpoint-sha256", required=True)
    parser.add_argument("--packet-arm", choices=("guarded", "unguarded"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--development-data", type=Path, required=True)
    parser.add_argument("--development-data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1600)
    parser.add_argument("--pairs-per-microbatch", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=4800)
    parser.add_argument("--development-probe-rows", type=int, default=96)
    parser.add_argument("--state-width", type=int, default=256)
    parser.add_argument("--state-slots", type=int, default=24)
    parser.add_argument("--packet-slots", type=int, default=6)
    parser.add_argument("--max-trace-steps", type=int, default=12)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--ff-multiplier", type=int, default=4)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026080605)
    parser.add_argument("--data-seed", type=int, default=2026080605)
    args = parser.parse_args()
    integer_fields = (
        args.updates,
        args.pairs_per_microbatch,
        args.gradient_accumulation,
        args.max_rows,
        args.development_probe_rows,
        args.state_width,
        args.state_slots,
        args.packet_slots,
        args.max_trace_steps,
        args.attention_heads,
        args.ff_multiplier,
        args.max_sequence_length,
        args.log_interval,
    )
    if any(value <= 0 for value in integer_fields) or args.learning_rate <= 0:
        parser.error("RSM1 training dimensions must be positive")
    return args


def main() -> int:
    report = run(parse_args())
    print(
        f"[rsm1-train] arm={report['packet_arm']} updates={report['updates']} "
        f"terminal={report['final_development_probe']['terminal_exact']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
