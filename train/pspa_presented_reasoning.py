#!/usr/bin/env python3
"""Depth-shift gate for prompt-selected presented algebra."""

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

from prompt_selected_presented_algebra import (
    PresentedAlgebraConfig,
    PresentedAlgebraResult,
    PromptSelectedPresentedAlgebra,
)


SCHEMA = "shohin-pspa-presented-reasoning-v1"
FAMILIES = ("cyclic", "dihedral", "permutation")


class PresentedReasoningError(RuntimeError):
    """The PSPA reasoning gate violated its fixed contract."""


@dataclass(frozen=True, slots=True)
class PresentedBatch:
    family: torch.Tensor
    observation_generator: torch.Tensor
    observation_input: torch.Tensor
    observation_output: torch.Tensor
    observation_mask: torch.Tensor
    generator_mask: torch.Tensor
    challenge_start: torch.Tensor
    challenge_word: torch.Tensor
    challenge_word_mask: torch.Tensor
    challenge_outcome: torch.Tensor
    challenge_mask: torch.Tensor
    query_start: torch.Tensor
    query_word: torch.Tensor
    query_word_mask: torch.Tensor
    answer: torch.Tensor
    true_tables: torch.Tensor

    def to(self, device: torch.device) -> PresentedBatch:
        return PresentedBatch(
            **{
                field.name: getattr(self, field.name).to(device)
                for field in fields(self)
            }
        )


def _conjugate(permutation: list[int], action: list[int]) -> list[int]:
    public = [0] * len(action)
    for internal, target in enumerate(action):
        public[permutation[internal]] = permutation[target]
    return public


def _apply_python(tables: list[list[int]], start: int, word: list[int]) -> int:
    state = start
    for generator in word:
        state = tables[generator][state]
    return state


def _actions(family: int, rng: random.Random, carrier: int) -> list[list[int]]:
    renaming = list(range(carrier))
    rng.shuffle(renaming)
    if family == 0:
        cycle = [(value + 1) % carrier for value in range(carrier)]
        return [_conjugate(renaming, cycle)]
    if family == 1:
        rotation = [(value + 1) % carrier for value in range(carrier)]
        reflection = [(-value) % carrier for value in range(carrier)]
        return [
            _conjugate(renaming, rotation),
            _conjugate(renaming, reflection),
        ]
    actions = []
    for _ in range(3):
        action = list(range(carrier))
        rng.shuffle(action)
        actions.append(action)
    return actions


def _distinguishing_challenge(
    true_tables: list[list[int]],
    false_tables: list[list[int]],
    active_generators: int,
    rng: random.Random,
    maximum_length: int,
) -> tuple[int, list[int], int]:
    carrier = len(true_tables[0])
    for _ in range(512):
        length = rng.randint(2, min(6, maximum_length))
        word = [rng.randrange(active_generators) for _ in range(length)]
        start = rng.randrange(carrier)
        true_outcome = _apply_python(true_tables, start, word)
        false_outcome = _apply_python(false_tables, start, word)
        if true_outcome != false_outcome:
            return start, word, true_outcome
    raise PresentedReasoningError("failed to construct distinguishing challenge")


def _row(
    family: int,
    query_length: int,
    rng: random.Random,
    config: PresentedAlgebraConfig,
) -> dict[str, Any]:
    carrier = config.carrier_size
    tables = _actions(family, rng, carrier)
    active = len(tables)
    missing: list[tuple[list[int], list[int]]] = []
    observations = []
    for generator, table in enumerate(tables):
        missing_inputs = rng.sample(range(carrier), 2)
        missing_outputs = [table[index] for index in missing_inputs]
        missing.append((missing_inputs, missing_outputs))
        for source, target in enumerate(table):
            if source not in missing_inputs:
                observations.append((generator, source, target))
    rng.shuffle(observations)

    challenges = []
    for generator in range(active):
        false_tables = [list(action) for action in tables]
        inputs, outputs = missing[generator]
        false_tables[generator][inputs[0]] = outputs[1]
        false_tables[generator][inputs[1]] = outputs[0]
        challenges.append(
            _distinguishing_challenge(
                tables,
                false_tables,
                active,
                rng,
                config.maximum_word_length,
            )
        )
    while len(challenges) < config.maximum_challenges:
        length = rng.randint(2, min(6, config.maximum_word_length))
        word = [rng.randrange(active) for _ in range(length)]
        start = rng.randrange(carrier)
        challenges.append((start, word, _apply_python(tables, start, word)))
    rng.shuffle(challenges)

    query_word = [rng.randrange(active) for _ in range(query_length)]
    query_start = rng.randrange(carrier)
    answer = _apply_python(tables, query_start, query_word)

    def padded_word(word: list[int]) -> tuple[list[int], list[bool]]:
        padding = config.maximum_word_length - len(word)
        return word + [0] * padding, [True] * len(word) + [False] * padding

    challenge_words = []
    challenge_word_masks = []
    for _, word, _ in challenges:
        padded, mask = padded_word(word)
        challenge_words.append(padded)
        challenge_word_masks.append(mask)
    padded_query, query_mask = padded_word(query_word)

    observation_padding = config.maximum_observations - len(observations)
    observations.extend([(0, 0, 0)] * observation_padding)
    padded_tables = tables + [list(range(carrier))] * (
        config.maximum_generators - active
    )
    return {
        "family": family,
        "observation_generator": [row[0] for row in observations],
        "observation_input": [row[1] for row in observations],
        "observation_output": [row[2] for row in observations],
        "observation_mask": [True] * (len(observations) - observation_padding)
        + [False] * observation_padding,
        "generator_mask": [True] * active
        + [False] * (config.maximum_generators - active),
        "challenge_start": [row[0] for row in challenges],
        "challenge_word": challenge_words,
        "challenge_word_mask": challenge_word_masks,
        "challenge_outcome": [row[2] for row in challenges],
        "challenge_mask": [True] * len(challenges),
        "query_start": query_start,
        "query_word": padded_query,
        "query_word_mask": query_mask,
        "answer": answer,
        "true_tables": padded_tables,
    }


def generate_batch(
    batch_size: int,
    query_length: int,
    config: PresentedAlgebraConfig,
    *,
    seed: int,
    family: int | None = None,
    device: torch.device | None = None,
) -> PresentedBatch:
    config.validate()
    if batch_size <= 0 or not 1 <= query_length <= config.maximum_word_length:
        raise PresentedReasoningError("batch size or query length differs")
    if family is not None and family not in range(len(FAMILIES)):
        raise PresentedReasoningError("unknown presentation family")
    rows = []
    for index in range(batch_size):
        row_family = family if family is not None else (seed + index) % len(FAMILIES)
        rng = random.Random(seed * 1_000_003 + index * 97 + row_family * 7919)
        rows.append(_row(row_family, query_length, rng, config))
    target = device or torch.device("cpu")

    def tensor(name: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor([row[name] for row in rows], dtype=dtype, device=target)

    return PresentedBatch(
        family=tensor("family", torch.long),
        observation_generator=tensor("observation_generator", torch.long),
        observation_input=tensor("observation_input", torch.long),
        observation_output=tensor("observation_output", torch.long),
        observation_mask=tensor("observation_mask", torch.bool),
        generator_mask=tensor("generator_mask", torch.bool),
        challenge_start=tensor("challenge_start", torch.long),
        challenge_word=tensor("challenge_word", torch.long),
        challenge_word_mask=tensor("challenge_word_mask", torch.bool),
        challenge_outcome=tensor("challenge_outcome", torch.long),
        challenge_mask=tensor("challenge_mask", torch.bool),
        query_start=tensor("query_start", torch.long),
        query_word=tensor("query_word", torch.long),
        query_word_mask=tensor("query_word_mask", torch.bool),
        answer=tensor("answer", torch.long),
        true_tables=tensor("true_tables", torch.float32),
    )


def batch_sha256(batch: PresentedBatch) -> str:
    digest = hashlib.sha256()
    for field in fields(batch):
        value = getattr(batch, field.name)
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class NeuralConfig:
    width: int = 64
    heads: int = 4
    layers: int = 2
    ff_multiplier: int = 2


class SourceEncoder(nn.Module):
    def __init__(self, algebra: PresentedAlgebraConfig, config: NeuralConfig):
        super().__init__()
        width = config.width
        self.algebra = algebra
        self.generator = nn.Embedding(algebra.maximum_generators, width)
        self.state = nn.Embedding(algebra.carrier_size, width)
        self.family = nn.Embedding(len(FAMILIES), width)
        self.word_position = nn.Embedding(algebra.maximum_word_length, width)
        self.observation_type = nn.Parameter(torch.randn(width) * 0.02)
        self.challenge_type = nn.Parameter(torch.randn(width) * 0.02)
        self.observation_projection = nn.Sequential(
            nn.Linear(width * 3, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.challenge_projection = nn.Sequential(
            nn.Linear(width * 3, width), nn.SiLU(), nn.Linear(width, width)
        )

    def word(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(values.shape[-1], device=values.device)
        encoded = self.generator(values) + self.word_position(positions)
        weights = mask.to(encoded.dtype).unsqueeze(-1)
        return (encoded * weights).sum(-2) / weights.sum(-2).clamp_min(1)

    def forward(self, batch: PresentedBatch) -> tuple[torch.Tensor, torch.Tensor]:
        family = self.family(batch.family)
        observations = self.observation_projection(
            torch.cat(
                (
                    self.generator(batch.observation_generator),
                    self.state(batch.observation_input),
                    self.state(batch.observation_output),
                ),
                -1,
            )
        )
        observations = observations + family[:, None] + self.observation_type
        challenge_word = self.word(
            batch.challenge_word, batch.challenge_word_mask
        )
        challenges = self.challenge_projection(
            torch.cat(
                (
                    challenge_word,
                    self.state(batch.challenge_start),
                    self.state(batch.challenge_outcome),
                ),
                -1,
            )
        )
        challenges = challenges + family[:, None] + self.challenge_type
        source = torch.cat((observations, challenges), 1)
        mask = torch.cat((batch.observation_mask, batch.challenge_mask), 1)
        return source, mask


class RecurrentPresentationControl(nn.Module):
    def __init__(self, algebra: PresentedAlgebraConfig, config: NeuralConfig):
        super().__init__()
        self.encoder = SourceEncoder(algebra, config)
        self.generator = self.encoder.generator
        self.state = self.encoder.state
        self.position = self.encoder.word_position
        self.attention = nn.MultiheadAttention(
            config.width, config.heads, batch_first=True
        )
        self.recurrence = nn.GRUCell(config.width, config.width)
        self.answer = nn.Linear(config.width, algebra.carrier_size)

    def forward(self, batch: PresentedBatch) -> torch.Tensor:
        source, source_mask = self.encoder(batch)
        hidden = self.state(batch.query_start)
        for position in range(batch.query_word.shape[-1]):
            request = (
                hidden
                + self.generator(batch.query_word[:, position])
                + self.position.weight[position]
            )
            read, _ = self.attention(
                request[:, None],
                source,
                source,
                key_padding_mask=~source_mask,
                need_weights=False,
            )
            proposal = self.recurrence(read[:, 0], hidden)
            hidden = torch.where(
                batch.query_word_mask[:, position, None], proposal, hidden
            )
        return self.answer(hidden)


class TransformerPresentationControl(nn.Module):
    def __init__(self, algebra: PresentedAlgebraConfig, config: NeuralConfig):
        super().__init__()
        self.encoder = SourceEncoder(algebra, config)
        self.query_start = self.encoder.state
        self.query_generator = self.encoder.generator
        self.position = self.encoder.word_position
        self.cls = nn.Parameter(torch.randn(config.width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            config.width,
            config.heads,
            config.width * config.ff_multiplier,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, config.layers)
        self.answer = nn.Linear(config.width, algebra.carrier_size)

    def forward(self, batch: PresentedBatch) -> torch.Tensor:
        source, source_mask = self.encoder(batch)
        positions = torch.arange(
            batch.query_word.shape[-1], device=batch.query_word.device
        )
        query = (
            self.query_generator(batch.query_word)
            + self.position(positions)[None]
            + self.query_start(batch.query_start)[:, None]
        )
        cls = self.cls[None, None].expand(batch.family.shape[0], 1, -1)
        tokens = torch.cat((cls, source, query), 1)
        mask = torch.cat(
            (
                torch.ones(
                    batch.family.shape[0], 1, dtype=torch.bool, device=tokens.device
                ),
                source_mask,
                batch.query_word_mask,
            ),
            1,
        )
        encoded = self.transformer(tokens, src_key_padding_mask=~mask)
        return self.answer(encoded[:, 0])


class PresentedReasoner(nn.Module):
    def __init__(self, algebra: PresentedAlgebraConfig, neural: NeuralConfig):
        super().__init__()
        self.algebra_config = algebra
        self.pspa = PromptSelectedPresentedAlgebra(algebra)
        self.recurrent = RecurrentPresentationControl(algebra, neural)
        self.transformer = TransformerPresentationControl(algebra, neural)

    def structured(
        self,
        batch: PresentedBatch,
        *,
        shuffle_challenges: bool = False,
        lineage_swap: bool = False,
    ) -> PresentedAlgebraResult:
        return self.pspa(
            batch.observation_generator,
            batch.observation_input,
            batch.observation_output,
            batch.observation_mask,
            batch.generator_mask,
            batch.challenge_start,
            batch.challenge_word,
            batch.challenge_word_mask,
            batch.challenge_outcome,
            batch.challenge_mask,
            batch.query_start,
            batch.query_word,
            batch.query_word_mask,
            shuffle_challenges=shuffle_challenges,
            lineage_swap=lineage_swap,
        )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise PresentedReasoningError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@torch.inference_mode()
def evaluate(
    model: PresentedReasoner,
    *,
    family: int,
    length: int,
    count: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    batch = generate_batch(
        count,
        length,
        model.algebra_config,
        seed=seed,
        family=family,
        device=device,
    )
    structured = model.structured(batch)
    shuffled = model.structured(batch, shuffle_challenges=True)
    swapped = model.structured(batch, lineage_swap=True)
    recurrent = model.recurrent(batch)
    transformer = model.transformer(batch)
    return {
        "family": FAMILIES[family],
        "length": length,
        "count": count,
        "seed": seed,
        "batch_sha256": batch_sha256(batch),
        "pspa_accuracy": structured.answer.eq(batch.answer).float().mean().item(),
        "recurrent_accuracy": recurrent.argmax(-1)
        .eq(batch.answer)
        .float()
        .mean()
        .item(),
        "transformer_accuracy": transformer.argmax(-1)
        .eq(batch.answer)
        .float()
        .mean()
        .item(),
        "shuffle_challenge_accuracy": shuffled.answer.eq(batch.answer).float().mean().item(),
        "lineage_swap_accuracy": swapped.answer.eq(batch.answer).float().mean().item(),
        "challenge_exact": structured.challenge_exact.float().mean().item(),
        "selected_table_exact": structured.selected_tables.argmax(-1)
        .eq(batch.true_tables.long())
        .all((-1, -2))
        .float()
        .mean()
        .item(),
        "selection_margin": structured.selection_margin.float().mean().item(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise PresentedReasoningError("CUDA is required unless --allow-cpu is explicit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    algebra = PresentedAlgebraConfig()
    neural = NeuralConfig()
    model = PresentedReasoner(algebra, neural).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    started = time.monotonic()
    train_log = []
    model.train()
    for update in range(1, args.updates + 1):
        length = 1 + ((update * 997 + args.seed) % args.train_max_length)
        batch = generate_batch(
            args.batch_size,
            length,
            algebra,
            seed=args.data_seed + update,
            device=device,
        )
        recurrent = model.recurrent(batch)
        transformer = model.transformer(batch)
        recurrent_loss = F.cross_entropy(recurrent, batch.answer)
        transformer_loss = F.cross_entropy(transformer, batch.answer)
        loss = recurrent_loss + transformer_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update == 1 or update % args.log_every == 0 or update == args.updates:
            row = {
                "update": update,
                "length": length,
                "loss": loss.item(),
                "recurrent_accuracy": recurrent.argmax(-1).eq(batch.answer).float().mean().item(),
                "transformer_accuracy": transformer.argmax(-1).eq(batch.answer).float().mean().item(),
            }
            train_log.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    elapsed = time.monotonic() - started
    evaluations = []
    for family in range(len(FAMILIES)):
        for length in (8, 12):
            evaluations.append(
                evaluate(
                    model,
                    family=family,
                    length=length,
                    count=args.eval_count,
                    seed=args.eval_seed + family * 100 + length,
                    device=device,
                )
            )
    checkpoint = args.output.with_suffix(".pt")
    if checkpoint.exists():
        raise PresentedReasoningError(f"refusing existing checkpoint: {checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": SCHEMA,
            "seed": args.seed,
            "algebra_config": asdict(algebra),
            "neural_config": asdict(neural),
            "model": model.state_dict(),
        },
        checkpoint,
    )
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "seed": args.seed,
        "data_seed": args.data_seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "charged_examples": args.updates * args.batch_size,
        "train_max_length": args.train_max_length,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "algebra_config": asdict(algebra),
        "neural_config": asdict(neural),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": elapsed,
        "examples_per_second": args.updates * args.batch_size / elapsed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "train_log": train_log,
        "evaluations": evaluations,
    }
    _atomic_json(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--data-seed", type=int, default=20260902)
    parser.add_argument("--eval-seed", type=int, default=61000)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-count", type=int, default=1024)
    parser.add_argument("--train-max-length", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    completed = run(parse_args())
    print(
        json.dumps(
            {
                "status": completed["status"],
                "parameters": completed["parameters"],
                "examples_per_second": completed["examples_per_second"],
            },
            sort_keys=True,
        )
    )
