#!/usr/bin/env python3
"""Train one DIVERGE-CGL1 outcome-grounded semantic arm."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from diverge_cgl1_data import (
    CGL1DataError,
    STATE_ORBITS,
    validate_public_record,
    validate_supervisor_record,
)
from diverge_cgl1_runtime import (
    CGL1Config,
    CausalGroundingInterpreter,
    adapter_state_dict,
    adapter_state_sha256,
    frozen_backbone_state_sha256,
)
from diverge_nve1_data import symbol_occurrence_groups
from diverge_rrg1_data import SOURCE_ROWS_PER_STAGE
from frozen_pointer_backbone import load_frozen_pointer_backbone


SCHEMA = "shohin-diverge-cgl1-training-report-v1"
TRAIN_SEED = 2026080702
PAIR_BATCH_SIZE = 32
LEARNING_RATE = 1e-4
CONSISTENCY_WEIGHT = 0.25


class CGL1TrainingError(RuntimeError):
    """The frozen CGL1 training contract was violated."""


@dataclass(frozen=True, slots=True)
class CompressedPair:
    records: tuple[dict[str, Any], dict[str, Any]]
    targets: tuple[int, int]
    physical_to_candidate: tuple[tuple[int, int], tuple[int, int]]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _compress_group(
    members: Sequence[tuple[dict[str, Any], dict[str, Any]]],
) -> CompressedPair:
    if len(members) != 2 * STATE_ORBITS:
        raise CGL1TrainingError("CGL1 compressed pair geometry differs")
    grouped: defaultdict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = (
        defaultdict(list)
    )
    for public, supervisor in members:
        grouped[str(public["source_identity_sha256"])].append((public, supervisor))
    if len(grouped) != 2 or any(len(values) != STATE_ORBITS for values in grouped.values()):
        raise CGL1TrainingError("CGL1 source-orbit grouping differs")

    records = []
    targets = []
    candidate_symbols = []
    for values in grouped.values():
        ordered = sorted(values, key=lambda item: int(item[0]["state_orbit"]))
        if [int(item[0]["state_orbit"]) for item in ordered] != list(
            range(STATE_ORBITS)
        ):
            raise CGL1TrainingError("CGL1 source state coverage differs")
        supports = []
        for public, supervisor in ordered:
            answer = int(supervisor["terminal_answer"])
            support = tuple(
                index
                for index, value in enumerate(public["candidate_values"])
                if int(value) == answer
            )
            supports.append(support)
        if len(supports[0]) != 1 or supports[1] != supports[0] or supports[2] != (0, 1):
            raise CGL1TrainingError("CGL1 outcome support differs")
        record = ordered[0][0]
        groups = symbol_occurrence_groups(
            str(record["source_text"]), tuple(str(value) for value in record["symbols"])
        )
        if len(groups) != 2:
            raise CGL1TrainingError("CGL1 compressed mentions differ")
        records.append(record)
        targets.append(supports[0][0])
        candidate_symbols.append(tuple(group[0] for group in groups))

    physical = sorted(set(candidate_symbols[0]))
    if len(physical) != 2 or set(candidate_symbols[1]) != set(physical):
        raise CGL1TrainingError("CGL1 clause-order symbols differ")
    physical_to_candidate = []
    for symbols in candidate_symbols:
        physical_to_candidate.append(tuple(symbols.index(value) for value in physical))
    return CompressedPair(
        records=(records[0], records[1]),
        targets=(targets[0], targets[1]),
        physical_to_candidate=(physical_to_candidate[0], physical_to_candidate[1]),
    )


def _load_pairs(
    public_path: Path,
    public_sha256: str,
    supervisor_path: Path,
    supervisor_sha256: str,
) -> list[CompressedPair]:
    if sha256_path(public_path) != public_sha256:
        raise CGL1TrainingError("CGL1 public hash differs")
    if sha256_path(supervisor_path) != supervisor_sha256:
        raise CGL1TrainingError("CGL1 supervisor hash differs")
    pairs = []
    pending: list[tuple[dict[str, Any], dict[str, Any]]] = []
    active_identity = None
    public_rows = 0
    with (
        public_path.open(encoding="utf-8") as public_handle,
        supervisor_path.open(encoding="utf-8") as supervisor_handle,
    ):
        while True:
            public_line = public_handle.readline()
            supervisor_line = supervisor_handle.readline()
            if not public_line or not supervisor_line:
                if public_line or supervisor_line:
                    raise CGL1TrainingError("CGL1 public/supervisor length differs")
                break
            public = json.loads(public_line)
            supervisor = json.loads(supervisor_line)
            validate_public_record(public)
            validate_supervisor_record(supervisor, public)
            public_rows += 1
            pair_identity = str(public["pair_identity_sha256"])
            if active_identity is None:
                active_identity = pair_identity
            if pair_identity != active_identity:
                pairs.append(_compress_group(pending))
                pending = []
                active_identity = pair_identity
            pending.append((public, supervisor))
    if pending:
        pairs.append(_compress_group(pending))
    expected_rows = SOURCE_ROWS_PER_STAGE * 2 * STATE_ORBITS
    if public_rows != expected_rows or len(pairs) != SOURCE_ROWS_PER_STAGE:
        raise CGL1TrainingError("CGL1 compressed corpus count differs")
    return pairs


def _pair_losses(
    scores: torch.Tensor,
    pairs: Sequence[CompressedPair],
    *,
    flip_outcomes: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_scores = scores.reshape(len(pairs), 2, 2)
    targets = torch.tensor(
        [
            (1 - target if flip_outcomes else target)
            for pair in pairs
            for target in pair.targets
        ],
        dtype=torch.long,
        device=device,
    )
    # Two distinct-outcome copies supervise each source and the equal-outcome
    # copy contributes exactly zero, so 2/3 preserves the full 300k-row mean.
    outcome = F.cross_entropy(scores.float(), targets) * (2.0 / 3.0)
    mapping = torch.tensor(
        [pair.physical_to_candidate for pair in pairs],
        dtype=torch.long,
        device=device,
    )
    aligned = torch.gather(pair_scores.float(), 2, mapping)
    probabilities = aligned.softmax(dim=-1)
    mean = probabilities.mean(dim=1, keepdim=True)
    consistency = (
        probabilities
        * (probabilities.clamp_min(1e-9).log() - mean.clamp_min(1e-9).log())
    ).sum(dim=-1).mean()
    return outcome, consistency


@torch.no_grad()
def _training_fit(
    model: CausalGroundingInterpreter,
    pairs: Sequence[CompressedPair],
    *,
    flip_outcomes: bool,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    true_exact = 0
    supervision_exact = 0
    total = 0
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        records = [record for pair in batch for record in pair.records]
        scores = model.candidate_scores(
            records, device=device, batch_size=len(records)
        )
        predictions = scores.argmax(dim=-1).tolist()
        true = [target for pair in batch for target in pair.targets]
        supervision = [1 - target if flip_outcomes else target for target in true]
        true_exact += sum(left == right for left, right in zip(predictions, true, strict=True))
        supervision_exact += sum(
            left == right for left, right in zip(predictions, supervision, strict=True)
        )
        total += len(records)
    return {
        "unique_source_rows": total,
        "true_exact": true_exact,
        "true_exact_rate": true_exact / total,
        "supervision_exact": supervision_exact,
        "supervision_exact_rate": supervision_exact / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--backbone-name", choices=("shohin", "smollm2"), required=True)
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--public-data-sha256", required=True)
    parser.add_argument("--supervisor-data", type=Path, required=True)
    parser.add_argument("--supervisor-data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-batch-size", type=int, default=PAIR_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--consistency-weight", type=float, default=CONSISTENCY_WEIGHT)
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--flip-outcomes", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--log-interval", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing CGL1 output: {args.output}")
    if (
        args.pair_batch_size != PAIR_BATCH_SIZE
        or args.learning_rate != LEARNING_RATE
        or args.consistency_weight != CONSISTENCY_WEIGHT
        or args.seed != TRAIN_SEED
    ):
        raise SystemExit("CGL1 frozen training schedule differs")
    for path, expected, label in (
        (args.base, args.base_sha256, "base"),
        (args.tokenizer, args.tokenizer_sha256, "tokenizer"),
    ):
        if sha256_path(path) != expected:
            raise SystemExit(f"CGL1 {label} hash differs")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CGL1 requested unavailable CUDA")

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
    config = CGL1Config()
    model = CausalGroundingInterpreter(backbone, tokenizer, config).to(device)
    parameters = list(model.adapter_parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01
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
        learning_rate = args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        scores = model.training_scores(records, device=device)
        outcome, consistency = _pair_losses(
            scores,
            batch,
            flip_outcomes=args.flip_outcomes,
            device=device,
        )
        loss = outcome + args.consistency_weight * consistency
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(gradient_norm):
            raise SystemExit("non-finite CGL1 gradient")
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
        raise SystemExit("CGL1 training changed a frozen backbone tensor")
    fit = _training_fit(
        model,
        pairs,
        flip_outcomes=args.flip_outcomes,
        device=device,
        batch_size=args.pair_batch_size,
    )
    args.output.mkdir(parents=True)
    checkpoint_path = args.output / "checkpoint.pt"
    checkpoint = {
        "schema": SCHEMA,
        "config": asdict(config),
        "backbone_name": args.backbone_name,
        "base_sha256": args.base_sha256,
        "tokenizer_sha256": args.tokenizer_sha256,
        "public_data_sha256": args.public_data_sha256,
        "supervisor_data_sha256": args.supervisor_data_sha256,
        "seed": args.seed,
        "updates": updates,
        "pair_batch_size": args.pair_batch_size,
        "learning_rate": args.learning_rate,
        "consistency_weight": args.consistency_weight,
        "flip_outcomes": args.flip_outcomes,
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
