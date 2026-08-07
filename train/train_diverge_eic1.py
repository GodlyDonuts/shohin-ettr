#!/usr/bin/env python3
"""Train one DIVERGE-EIC1 involution or equal-FLOP control arm."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import torch
from tokenizers import Tokenizer

from diverge_cgl1_data import CGL1DataError, STATE_ORBITS
from diverge_cgl1_runtime import (
    adapter_state_dict,
    adapter_state_sha256,
    frozen_backbone_state_sha256,
)
from diverge_eic1_runtime import EIC1Config, EquivariantIdentityCommitter
from diverge_rrg1_data import SOURCE_ROWS_PER_STAGE
from frozen_pointer_backbone import load_frozen_pointer_backbone
from train_diverge_cgl1 import (
    CONSISTENCY_WEIGHT,
    LEARNING_RATE,
    PAIR_BATCH_SIZE,
    TRAIN_SEED,
    _atomic_checkpoint,
    _atomic_json,
    _load_pairs,
    _pair_losses,
    _training_fit,
    sha256_path,
)


SCHEMA = "shohin-diverge-eic1-training-report-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--backbone-name", choices=("shohin", "smollm2"), required=True)
    parser.add_argument("--projection-mode", choices=("involution", "duplicate"), required=True)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--public-data-sha256", required=True)
    parser.add_argument("--supervisor-data", type=Path, required=True)
    parser.add_argument("--supervisor-data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-batch-size", type=int, default=PAIR_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--consistency-weight", type=float, default=CONSISTENCY_WEIGHT)
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--log-interval", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing EIC1 output: {args.output}")
    if (
        args.pair_batch_size != PAIR_BATCH_SIZE
        or args.learning_rate != LEARNING_RATE
        or args.consistency_weight != CONSISTENCY_WEIGHT
        or args.seed != TRAIN_SEED
    ):
        raise SystemExit("EIC1 frozen training schedule differs")
    for path, expected, label in (
        (args.base, args.base_sha256, "base"),
        (args.tokenizer, args.tokenizer_sha256, "tokenizer"),
    ):
        if sha256_path(path) != expected:
            raise SystemExit(f"EIC1 {label} hash differs")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("EIC1 requested unavailable CUDA")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    pairs = _load_pairs(
        args.public_data,
        args.public_data_sha256,
        args.supervisor_data,
        args.supervisor_data_sha256,
    )
    order = list(range(len(pairs)))
    random.Random(args.seed ^ 0x43474C31).shuffle(order)

    backbone, _, receipt = load_frozen_pointer_backbone(args.base, device=device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    config = EIC1Config(projection_mode=args.projection_mode)
    model = EquivariantIdentityCommitter(backbone, tokenizer, config).to(device)
    parameters = list(model.adapter_parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    updates = math.ceil(len(order) / args.pair_batch_size)
    history = []
    started = time.monotonic()
    model.train()
    for update, start in enumerate(
        range(0, len(order), args.pair_batch_size), start=1
    ):
        batch = [pairs[index] for index in order[start : start + args.pair_batch_size]]
        records = [record for pair in batch for record in pair.records]
        progress = update / updates
        learning_rate = args.learning_rate * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        scores = model.training_scores(records, device=device)
        outcome, consistency = _pair_losses(
            scores,
            batch,
            flip_outcomes=False,
            device=device,
        )
        loss = outcome + args.consistency_weight * consistency
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(gradient_norm):
            raise SystemExit("non-finite EIC1 gradient")
        optimizer.step()
        if update == 1 or update % args.log_interval == 0 or update == updates:
            record = {
                "update": update,
                "updates": updates,
                "learning_rate": learning_rate,
                "loss": float(loss.detach()),
                "outcome_loss": float(outcome.detach()),
                "consistency_loss": float(consistency.detach()),
                "gradient_norm": float(gradient_norm),
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    elapsed = time.monotonic() - started
    frozen_after = frozen_backbone_state_sha256(model.backbone)
    if frozen_after != model.frozen_state_before:
        raise SystemExit("EIC1 training changed a frozen backbone tensor")
    fit = _training_fit(
        model,
        pairs,
        flip_outcomes=False,
        device=device,
        batch_size=args.pair_batch_size,
    )
    args.output.mkdir(parents=True)
    checkpoint_path = args.output / "checkpoint.pt"
    checkpoint = {
        "schema": SCHEMA,
        "config": asdict(config),
        "backbone_name": args.backbone_name,
        "projection_mode": args.projection_mode,
        "base_sha256": args.base_sha256,
        "tokenizer_sha256": args.tokenizer_sha256,
        "public_data_sha256": args.public_data_sha256,
        "supervisor_data_sha256": args.supervisor_data_sha256,
        "seed": args.seed,
        "updates": updates,
        "pair_batch_size": args.pair_batch_size,
        "learning_rate": args.learning_rate,
        "consistency_weight": args.consistency_weight,
        "lora_projection_count": model.lora_projection_count,
        "adapter_state": adapter_state_dict(model),
        "adapter_state_sha256": adapter_state_sha256(model),
        "frozen_backbone_state_sha256": frozen_after,
    }
    _atomic_checkpoint(checkpoint_path, checkpoint)
    report = {
        **{key: value for key, value in checkpoint.items() if key != "adapter_state"},
        "base": str(args.base),
        "tokenizer": str(args.tokenizer),
        "public_data": str(args.public_data),
        "supervisor_data": str(args.supervisor_data),
        "logical_public_rows": len(pairs) * 2 * STATE_ORBITS,
        "compressed_pairs": len(pairs),
        "unique_source_rows": len(pairs) * 2,
        "backbone_forwards_per_source": 2,
        "elapsed_seconds": elapsed,
        "logical_rows_per_second": len(pairs) * 2 * STATE_ORBITS / max(elapsed, 1e-9),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "training_fit": fit,
        "history": history,
        "backbone_receipt": {
            "checkpoint_format": receipt.checkpoint_format,
            "base_step": receipt.base_step,
            "initialization": receipt.initialization,
            "base_import": receipt.base_import,
            "base_rms_norm_eps": receipt.base_rms_norm_eps,
        },
        "objective_receipt": {
            "distinct_outcome_multiplicity": 2,
            "equal_outcome_multiplicity": 1,
            "equal_outcome_gradient": 0,
            "compressed_objective_exact": True,
            "identity_projection": args.projection_mode,
        },
    }
    report_path = args.output / "report.json"
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": report["checkpoint_sha256"],
                "report": str(report_path),
                "report_sha256": sha256_path(report_path),
                "training_fit": fit,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except CGL1DataError as error:
        raise SystemExit(str(error)) from error
