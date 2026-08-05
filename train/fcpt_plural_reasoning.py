#!/usr/bin/env python3
"""Three-family plural-inference gate for FCPT and matched particle controls."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from falsification_coupled_particles import (
    FalsificationCoupledParticleCore,
    ParticleArm,
    ParticleConfig,
    ParticleTrajectory,
)


SCHEMA = "shohin-fcpt-plural-reasoning-v1"
MODULUS = 11
FAMILIES = ("noncommuting", "binding", "induction")


class PluralReasoningError(RuntimeError):
    """The plural-reasoning gate violated its frozen contract."""


@dataclass(frozen=True, slots=True)
class BoardConfig:
    width: int = 64
    maximum_evidence: int = 9
    maximum_descriptors: int = 9
    maximum_depth: int = 7
    modulus: int = MODULUS

    def validate(self) -> None:
        if min(
            self.width,
            self.maximum_evidence,
            self.maximum_descriptors,
            self.maximum_depth,
            self.modulus,
        ) <= 0:
            raise PluralReasoningError("board dimensions must be positive")
        if self.maximum_depth + 2 > self.maximum_evidence:
            raise PluralReasoningError("board cannot hold maximum-depth evidence")
        if self.maximum_depth + 2 > self.maximum_descriptors:
            raise PluralReasoningError("board cannot hold maximum-depth descriptors")


@dataclass(frozen=True, slots=True)
class PluralBatch:
    family: torch.Tensor
    descriptor_type: torch.Tensor
    descriptor_value: torch.Tensor
    descriptor_mask: torch.Tensor
    probe_fields: torch.Tensor
    outcomes: torch.Tensor
    evidence_mask: torch.Tensor
    query_fields: torch.Tensor
    answer: torch.Tensor

    def to(self, device: torch.device) -> PluralBatch:
        return PluralBatch(
            **{
                field.name: getattr(self, field.name).to(device)
                for field in fields(self)
            }
        )


def _apply_affine_program(value: int, operations: list[tuple[int, int]]) -> int:
    for opcode, parameter in operations:
        if opcode == 0:
            value = value + parameter
        elif opcode == 1:
            value = value * parameter
        elif opcode == 2:
            value = parameter - value
        else:
            raise PluralReasoningError("unknown affine opcode")
        value %= MODULUS
    return value


def _apply_mapping(mapping: list[int], value: int, hops: int) -> int:
    for _ in range(hops):
        value = mapping[value]
    return value


def _polynomial(coefficients: list[int], value: int) -> int:
    result = 0
    power = 1
    for coefficient in coefficients:
        result = (result + coefficient * power) % MODULUS
        power = (power * value) % MODULUS
    return result


def _empty_row(config: BoardConfig) -> dict[str, Any]:
    return {
        "descriptor_type": [0] * config.maximum_descriptors,
        "descriptor_value": [0] * config.maximum_descriptors,
        "descriptor_mask": [False] * config.maximum_descriptors,
        "probe_fields": [[0, 0, 0] for _ in range(config.maximum_evidence)],
        "outcomes": [0] * config.maximum_evidence,
        "evidence_mask": [False] * config.maximum_evidence,
        "query_fields": [0, 0, 0],
    }


def _noncommuting_row(depth: int, rng: random.Random, config: BoardConfig) -> dict[str, Any]:
    row = _empty_row(config)
    operations = []
    for position in range(depth):
        opcode = rng.randrange(3)
        parameter = rng.randrange(1, MODULUS) if opcode == 1 else rng.randrange(MODULUS)
        operations.append((opcode, parameter))
        row["descriptor_type"][position] = opcode
        row["descriptor_value"][position] = parameter
        row["descriptor_mask"][position] = True
    order = list(range(depth))
    rng.shuffle(order)
    hidden_program = [operations[index] for index in order]
    evidence_count = depth + 2
    values = rng.sample(range(MODULUS), evidence_count + 1)
    for index, value in enumerate(values[:evidence_count]):
        row["probe_fields"][index] = [value, depth, 0]
        row["outcomes"][index] = _apply_affine_program(value, hidden_program)
        row["evidence_mask"][index] = True
    query = values[-1]
    row["query_fields"] = [query, depth, 0]
    row["answer"] = _apply_affine_program(query, hidden_program)
    return row


def _binding_row(depth: int, rng: random.Random, config: BoardConfig) -> dict[str, Any]:
    row = _empty_row(config)
    cardinality = depth + 2
    canonical = list(range(cardinality))
    rng.shuffle(canonical)
    binding = list(range(cardinality))
    rng.shuffle(binding)
    inverse = [0] * cardinality
    for public, internal in enumerate(binding):
        inverse[internal] = public
    public_mapping = [
        inverse[canonical[binding[public]]] for public in range(cardinality)
    ]
    for position, target in enumerate(canonical):
        row["descriptor_type"][position] = 3
        row["descriptor_value"][position] = target
        row["descriptor_mask"][position] = True
    candidates = [
        (value, hops)
        for value in range(cardinality)
        for hops in range(1, 4)
    ]
    rng.shuffle(candidates)
    evidence_count = min(config.maximum_evidence, cardinality + 1)
    for index, (value, hops) in enumerate(candidates[:evidence_count]):
        row["probe_fields"][index] = [value, hops, cardinality]
        row["outcomes"][index] = _apply_mapping(public_mapping, value, hops)
        row["evidence_mask"][index] = True
    query_value, query_hops = candidates[evidence_count]
    row["query_fields"] = [query_value, query_hops, cardinality]
    row["answer"] = _apply_mapping(public_mapping, query_value, query_hops)
    return row


def _induction_row(depth: int, rng: random.Random, config: BoardConfig) -> dict[str, Any]:
    row = _empty_row(config)
    coefficients = [rng.randrange(MODULUS) for _ in range(depth + 1)]
    for position in range(depth + 1):
        row["descriptor_type"][position] = 4
        row["descriptor_value"][position] = position
        row["descriptor_mask"][position] = True
    evidence_count = depth + 1
    values = rng.sample(range(MODULUS), evidence_count + 1)
    for index, value in enumerate(values[:evidence_count]):
        row["probe_fields"][index] = [value, depth, 0]
        row["outcomes"][index] = _polynomial(coefficients, value)
        row["evidence_mask"][index] = True
    query = values[-1]
    row["query_fields"] = [query, depth, 0]
    row["answer"] = _polynomial(coefficients, query)
    return row


def generate_batch(
    batch_size: int,
    depth: int,
    config: BoardConfig,
    *,
    seed: int,
    family: int | None = None,
    device: torch.device | None = None,
) -> PluralBatch:
    config.validate()
    if batch_size <= 0 or not 2 <= depth <= config.maximum_depth:
        raise PluralReasoningError("batch size or depth differs")
    if family is not None and family not in range(len(FAMILIES)):
        raise PluralReasoningError("unknown board family")
    rows = []
    families = []
    for index in range(batch_size):
        row_family = family if family is not None else (index + seed) % len(FAMILIES)
        rng = random.Random(seed * 1_000_003 + index * 97 + row_family * 7919)
        if row_family == 0:
            row = _noncommuting_row(depth, rng, config)
        elif row_family == 1:
            row = _binding_row(depth, rng, config)
        else:
            row = _induction_row(depth, rng, config)
        rows.append(row)
        families.append(row_family)
    target_device = device or torch.device("cpu")
    return PluralBatch(
        family=torch.tensor(families, dtype=torch.long, device=target_device),
        descriptor_type=torch.tensor(
            [row["descriptor_type"] for row in rows],
            dtype=torch.long,
            device=target_device,
        ),
        descriptor_value=torch.tensor(
            [row["descriptor_value"] for row in rows],
            dtype=torch.long,
            device=target_device,
        ),
        descriptor_mask=torch.tensor(
            [row["descriptor_mask"] for row in rows],
            dtype=torch.bool,
            device=target_device,
        ),
        probe_fields=torch.tensor(
            [row["probe_fields"] for row in rows],
            dtype=torch.long,
            device=target_device,
        ),
        outcomes=torch.tensor(
            [row["outcomes"] for row in rows],
            dtype=torch.long,
            device=target_device,
        ),
        evidence_mask=torch.tensor(
            [row["evidence_mask"] for row in rows],
            dtype=torch.bool,
            device=target_device,
        ),
        query_fields=torch.tensor(
            [row["query_fields"] for row in rows],
            dtype=torch.long,
            device=target_device,
        ),
        answer=torch.tensor(
            [row["answer"] for row in rows],
            dtype=torch.long,
            device=target_device,
        ),
    )


def batch_sha256(batch: PluralBatch) -> str:
    digest = hashlib.sha256()
    for field in fields(batch):
        value = getattr(batch, field.name)
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class EpisodeEncoder(nn.Module):
    def __init__(self, config: BoardConfig):
        super().__init__()
        self.config = config
        width = config.width
        self.family = nn.Embedding(len(FAMILIES), width)
        self.value = nn.Embedding(config.modulus, width)
        self.field_position = nn.Embedding(3, width)
        self.descriptor_type = nn.Embedding(5, width)
        self.descriptor_position = nn.Embedding(config.maximum_descriptors, width)
        self.probe_norm = nn.RMSNorm(width)
        self.query_norm = nn.RMSNorm(width)

    def _global(self, batch: PluralBatch) -> torch.Tensor:
        positions = torch.arange(
            self.config.maximum_descriptors,
            device=batch.family.device,
        )
        descriptor = (
            self.descriptor_type(batch.descriptor_type)
            + self.value(batch.descriptor_value)
            + self.descriptor_position(positions)[None]
        )
        weights = batch.descriptor_mask.to(descriptor.dtype).unsqueeze(-1)
        return (descriptor * weights).sum(1) / weights.sum(1).clamp_min(1)

    def _fields(self, fields: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(3, device=fields.device)
        return (self.value(fields) + self.field_position(positions)).sum(-2)

    def forward(
        self, batch: PluralBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_summary = self._global(batch)
        family = self.family(batch.family)
        base = global_summary + family
        probes = self.probe_norm(self._fields(batch.probe_fields) + base[:, None])
        query = self.query_norm(self._fields(batch.query_fields) + base)
        return probes, probes, query


class PluralReasoner(nn.Module):
    def __init__(
        self,
        board_config: BoardConfig,
        particle_config: ParticleConfig,
        arm: ParticleArm,
    ):
        super().__init__()
        if board_config.width != particle_config.width:
            raise PluralReasoningError("board and particle widths differ")
        self.encoder = EpisodeEncoder(board_config)
        self.core = FalsificationCoupledParticleCore(particle_config, arm)

    def forward(self, batch: PluralBatch) -> tuple[torch.Tensor, ParticleTrajectory]:
        source, probes, query = self.encoder(batch)
        return self.core(
            source,
            batch.evidence_mask,
            probes,
            batch.outcomes,
            batch.evidence_mask,
            query,
        )


def behavior_loss(trajectory: ParticleTrajectory, batch: PluralBatch) -> torch.Tensor:
    losses = []
    for step in trajectory.rounds:
        batch_size, candidates, evidence, classes = step.behavior_logits.shape
        labels = batch.outcomes[:, None].expand(-1, candidates, -1)
        per_item = F.cross_entropy(
            step.behavior_logits.reshape(-1, classes),
            labels.reshape(-1),
            reduction="none",
        ).view(batch_size, candidates, evidence)
        mask = batch.evidence_mask[:, None].expand_as(per_item)
        losses.append((per_item * mask).sum() / mask.sum().clamp_min(1))
    return torch.stack(losses).mean()


def behavioral_diversity(trajectory: ParticleTrajectory, batch: PluralBatch) -> torch.Tensor:
    logits = trajectory.rounds[0].behavior_logits
    probabilities = logits.softmax(-1)
    mean_probability = probabilities.mean(1)
    mean_entropy = -(mean_probability * mean_probability.clamp_min(1e-8).log()).sum(-1)
    particle_entropy = -(
        probabilities * probabilities.clamp_min(1e-8).log()
    ).sum(-1).mean(1)
    mask = batch.evidence_mask.to(probabilities.dtype)
    return ((mean_entropy - particle_entropy) * mask).sum() / mask.sum().clamp_min(1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def metrics(
    logits: torch.Tensor, trajectory: ParticleTrajectory, batch: PluralBatch
) -> dict[str, float]:
    prediction = logits.argmax(-1)
    exact = prediction.eq(batch.answer)
    result = {"answer_accuracy": exact.float().mean().item()}
    for family, name in enumerate(FAMILIES):
        selected = batch.family.eq(family)
        result[f"{name}_accuracy"] = (
            exact[selected].float().mean().item() if selected.any() else float("nan")
        )
    behavior = trajectory.rounds[-1].behavior_logits.argmax(-1)
    unique_counts = []
    for row in behavior.detach().cpu():
        unique_counts.append(len({tuple(candidate.tolist()) for candidate in row}))
    result["mean_unique_behaviors"] = sum(unique_counts) / len(unique_counts)
    result["mean_final_log_weight"] = trajectory.final_log_weight.mean().item()
    return result


@torch.inference_mode()
def evaluate(
    model: PluralReasoner,
    *,
    family: int,
    depth: int,
    count: int,
    seed: int,
    config: BoardConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    batch = generate_batch(
        count, depth, config, seed=seed, family=family, device=device
    )
    logits, trajectory = model(batch)
    return {
        "family": FAMILIES[family],
        "depth": depth,
        "count": count,
        "seed": seed,
        "batch_sha256": batch_sha256(batch),
        **metrics(logits, trajectory, batch),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise PluralReasoningError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise PluralReasoningError("CUDA is required unless --allow-cpu is explicit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    board_config = BoardConfig(width=args.width)
    cohort_manifest_sha256 = None
    cohort_hashes: dict[tuple[int, int, int, int], str] = {}
    if args.cohort_manifest is not None:
        cohort_bytes = args.cohort_manifest.read_bytes()
        cohort_manifest_sha256 = hashlib.sha256(cohort_bytes).hexdigest()
        cohort_manifest = json.loads(cohort_bytes)
        if cohort_manifest.get("schema") != "shohin-fcpt-plural-cohort-v1":
            raise PluralReasoningError("cohort manifest schema differs")
        if cohort_manifest.get("status") != "frozen":
            raise PluralReasoningError("cohort manifest is not frozen")
        if cohort_manifest.get("config") != asdict(board_config):
            raise PluralReasoningError("cohort manifest config differs")
        cohort_hashes = {
            (
                int(row["family_id"]),
                int(row["depth"]),
                int(row["count"]),
                int(row["seed"]),
            ): str(row["sha256"])
            for row in cohort_manifest.get("batches") or []
            if row.get("split") == "development"
        }
    particle_config = ParticleConfig(
        width=args.width,
        slots=args.slots,
        particles=args.particles,
        branches=args.branches,
        rounds=args.rounds,
        heads=args.heads,
        ff_multiplier=args.ff_multiplier,
        outcome_classes=MODULUS,
        answer_classes=MODULUS,
        probes_per_round=args.probes_per_round,
    )
    model = PluralReasoner(board_config, particle_config, args.arm).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    started = time.monotonic()
    train_log = []
    model.train()
    for update in range(1, args.updates + 1):
        depth = 2 + ((update * 997 + args.seed) % (args.train_depth_max - 1))
        batch = generate_batch(
            args.batch_size,
            depth,
            board_config,
            seed=args.data_seed + update,
            device=device,
        )
        logits, trajectory = model(batch)
        answer_loss = F.cross_entropy(logits, batch.answer)
        prediction_loss = behavior_loss(trajectory, batch)
        diversity = behavioral_diversity(trajectory, batch)
        loss = answer_loss + args.behavior_weight * prediction_loss - args.diversity_weight * diversity
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update == 1 or update % args.log_every == 0 or update == args.updates:
            row = {
                "update": update,
                "depth": depth,
                "loss": loss.item(),
                "answer_loss": answer_loss.item(),
                "behavior_loss": prediction_loss.item(),
                "behavioral_diversity": diversity.item(),
                **metrics(logits, trajectory, batch),
            }
            train_log.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    elapsed = time.monotonic() - started
    evaluations = []
    for family in range(len(FAMILIES)):
        for depth in (5, 7):
            evaluations.append(
                evaluate(
                    model,
                    family=family,
                    depth=depth,
                    count=args.eval_count,
                    seed=args.eval_seed + family * 100 + depth,
                    config=board_config,
                    device=device,
                )
            )
    if cohort_hashes:
        observed = {
            (
                FAMILIES.index(row["family"]),
                int(row["depth"]),
                int(row["count"]),
                int(row["seed"]),
            ): str(row["batch_sha256"])
            for row in evaluations
        }
        if set(observed) != set(cohort_hashes) or any(
            observed[key] != cohort_hashes[key] for key in cohort_hashes
        ):
            raise PluralReasoningError("generated development cohort hash differs")
    checkpoint = args.output.with_suffix(".pt")
    if checkpoint.exists():
        raise PluralReasoningError(f"refusing existing checkpoint: {checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": SCHEMA,
            "arm": args.arm,
            "board_config": asdict(board_config),
            "particle_config": asdict(particle_config),
            "model": model.state_dict(),
            "seed": args.seed,
            "data_seed": args.data_seed,
            "updates": args.updates,
        },
        checkpoint,
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return {
        "schema": SCHEMA,
        "status": "complete",
        "arm": args.arm,
        "board_config": asdict(board_config),
        "particle_config": asdict(particle_config),
        "seed": args.seed,
        "data_seed": args.data_seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "train_depth_max": args.train_depth_max,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "behavior_weight": args.behavior_weight,
        "diversity_weight": args.diversity_weight,
        "parameters": parameter_count(model),
        "charged_examples": args.updates * args.batch_size,
        "elapsed_seconds": elapsed,
        "examples_per_second": args.updates * args.batch_size / elapsed,
        "train_log": train_log,
        "evaluations": evaluations,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "cohort_manifest": (
            str(args.cohort_manifest.resolve()) if args.cohort_manifest else None
        ),
        "cohort_manifest_sha256": cohort_manifest_sha256,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm", choices=("fcpt", "independent", "soft", "selection"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--data-seed", type=int, default=20260850)
    parser.add_argument("--eval-seed", type=int, default=51000)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-count", type=int, default=1024)
    parser.add_argument("--train-depth-max", type=int, default=4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--branches", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-multiplier", type=int, default=2)
    parser.add_argument("--probes-per-round", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--behavior-weight", type=float, default=0.5)
    parser.add_argument("--diversity-weight", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--cohort-manifest", type=Path)
    args = parser.parse_args()
    if not 2 <= args.train_depth_max <= 4:
        parser.error("pilot training depth must be between 2 and 4")
    if min(args.updates, args.batch_size, args.eval_count, args.log_every) <= 0:
        parser.error("training counts must be positive")
    return args


def main() -> int:
    args = parse_args()
    report = run(args)
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"train_log", "evaluations"}
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
