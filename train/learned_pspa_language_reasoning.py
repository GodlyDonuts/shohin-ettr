#!/usr/bin/env python3
"""Learn a source-language compiler into whole executable presentations."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
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

from prompt_selected_presented_algebra import PresentedAlgebraConfig
from pspa_presented_reasoning import (
    FAMILIES,
    PresentedBatch,
    PresentedReasoningError,
    batch_sha256,
    generate_batch,
)


SCHEMA = "shohin-learned-pspa-language-v1"
PAD = 0
SEP = 1
OBSERVATION_A = 2
OBSERVATION_B = 3
CHALLENGE_A = 4
CHALLENGE_B = 5
FROM = 6
TO = 7
USING = 8
START = 9
RESULT = 10
AFTER = 11
MAP = 12
BECOMES = 13
GENERATOR_BASE = 32


@dataclass(frozen=True, slots=True)
class RenderedSource:
    tokens: torch.Tensor
    token_mask: torch.Tensor
    record_mask: torch.Tensor

    def to(self, device: torch.device) -> RenderedSource:
        return RenderedSource(
            tokens=self.tokens.to(device),
            token_mask=self.token_mask.to(device),
            record_mask=self.record_mask.to(device),
        )


@dataclass(frozen=True, slots=True)
class LanguageConfig:
    width: int = 64
    heads: int = 4
    layers: int = 2
    ff_multiplier: int = 2
    sinkhorn_rounds: int = 8


def state_token(config: PresentedAlgebraConfig, value: int) -> int:
    return GENERATOR_BASE + config.maximum_generators + value


def generator_token(value: int) -> int:
    return GENERATOR_BASE + value


def vocabulary_size(config: PresentedAlgebraConfig) -> int:
    return GENERATOR_BASE + config.maximum_generators + config.carrier_size


def render_source(
    batch: PresentedBatch,
    config: PresentedAlgebraConfig,
    *,
    seed: int,
) -> RenderedSource:
    """Render source-only evidence; query and answer are intentionally absent."""

    if batch.family.device.type != "cpu":
        raise PresentedReasoningError("rendering requires a CPU batch")
    maximum_records = config.maximum_observations + config.maximum_challenges
    maximum_tokens = config.maximum_word_length + 7
    all_records: list[list[list[int]]] = []
    for row in range(batch.family.shape[0]):
        rng = random.Random(seed * 1_000_003 + row * 8191)
        records: list[list[int]] = []
        for index in range(config.maximum_observations):
            if not bool(batch.observation_mask[row, index]):
                continue
            generator = int(batch.observation_generator[row, index])
            source = int(batch.observation_input[row, index])
            target = int(batch.observation_output[row, index])
            if rng.randrange(2) == 0:
                record = [
                    OBSERVATION_A,
                    generator_token(generator),
                    FROM,
                    state_token(config, source),
                    TO,
                    state_token(config, target),
                    SEP,
                ]
            else:
                record = [
                    OBSERVATION_B,
                    MAP,
                    state_token(config, source),
                    USING,
                    generator_token(generator),
                    BECOMES,
                    state_token(config, target),
                    SEP,
                ]
            records.append(record)
        for index in range(config.maximum_challenges):
            if not bool(batch.challenge_mask[row, index]):
                continue
            word = [
                generator_token(int(value))
                for value, keep in zip(
                    batch.challenge_word[row, index],
                    batch.challenge_word_mask[row, index],
                    strict=True,
                )
                if bool(keep)
            ]
            source = state_token(config, int(batch.challenge_start[row, index]))
            target = state_token(config, int(batch.challenge_outcome[row, index]))
            if rng.randrange(2) == 0:
                record = [CHALLENGE_A, START, source, AFTER, *word, RESULT, target, SEP]
            else:
                record = [CHALLENGE_B, source, USING, *word, TO, target, SEP]
            records.append(record)
        rng.shuffle(records)
        if len(records) > maximum_records or any(
            len(record) > maximum_tokens for record in records
        ):
            raise PresentedReasoningError("rendered source exceeds fixed geometry")
        all_records.append(records)

    tokens = torch.full(
        (batch.family.shape[0], maximum_records, maximum_tokens),
        PAD,
        dtype=torch.long,
    )
    token_mask = torch.zeros_like(tokens, dtype=torch.bool)
    record_mask = torch.zeros(
        batch.family.shape[0], maximum_records, dtype=torch.bool
    )
    for row, records in enumerate(all_records):
        for index, record in enumerate(records):
            tokens[row, index, : len(record)] = torch.tensor(record)
            token_mask[row, index, : len(record)] = True
            record_mask[row, index] = True
    return RenderedSource(tokens=tokens, token_mask=token_mask, record_mask=record_mask)


class RecordLanguageEncoder(nn.Module):
    def __init__(self, algebra: PresentedAlgebraConfig, config: LanguageConfig):
        super().__init__()
        self.algebra = algebra
        self.token = nn.Embedding(vocabulary_size(algebra), config.width)
        self.token_position = nn.Embedding(
            algebra.maximum_word_length + 7, config.width
        )
        self.token_mlp = nn.Sequential(
            nn.Linear(config.width, config.width * 2),
            nn.SiLU(),
            nn.Linear(config.width * 2, config.width),
        )
        layer = nn.TransformerEncoderLayer(
            config.width,
            config.heads,
            config.width * config.ff_multiplier,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.records = nn.TransformerEncoder(layer, config.layers)

    def forward(self, source: RenderedSource) -> torch.Tensor:
        positions = torch.arange(source.tokens.shape[-1], device=source.tokens.device)
        encoded = self.token(source.tokens) + self.token_position(positions)
        encoded = self.token_mlp(encoded)
        weights = source.token_mask.to(encoded.dtype).unsqueeze(-1)
        records = (encoded * weights).sum(-2) / weights.sum(-2).clamp_min(1)
        return self.records(records, src_key_padding_mask=~source.record_mask)


def sinkhorn(logits: torch.Tensor, rounds: int) -> torch.Tensor:
    log_probabilities = logits
    for _ in range(rounds):
        log_probabilities = log_probabilities - torch.logsumexp(
            log_probabilities, -1, keepdim=True
        )
        log_probabilities = log_probabilities - torch.logsumexp(
            log_probabilities, -2, keepdim=True
        )
    return log_probabilities.exp()


def greedy_permutation(scores: torch.Tensor) -> torch.Tensor:
    """Project batched square scores to whole permutations without field mixing."""

    if scores.shape[-1] != scores.shape[-2]:
        raise PresentedReasoningError("permutation scores must be square")
    carrier = scores.shape[-1]
    flat = scores.reshape(-1, carrier, carrier)
    count = flat.shape[0]
    rows = torch.ones(count, carrier, dtype=torch.bool, device=scores.device)
    columns = torch.ones_like(rows)
    result = torch.zeros_like(flat)
    batch_index = torch.arange(count, device=scores.device)
    for _ in range(carrier):
        allowed = rows[:, :, None] & columns[:, None, :]
        index = flat.masked_fill(~allowed, -torch.inf).flatten(1).argmax(-1)
        row = index // carrier
        column = index % carrier
        result[batch_index, row, column] = 1
        rows[batch_index, row] = False
        columns[batch_index, column] = False
    return result.reshape_as(scores)


def execute_word(
    tables: torch.Tensor,
    start: torch.Tensor,
    word: torch.Tensor,
    word_mask: torch.Tensor,
) -> torch.Tensor:
    """Execute a word against probabilistic generator action tables."""

    state = F.one_hot(start, tables.shape[-1]).to(tables.dtype)
    batch_index = torch.arange(tables.shape[0], device=tables.device)
    for position in range(word.shape[-1]):
        generator = word[..., position]
        leading = generator.shape[1:]
        index = batch_index.reshape(-1, *([1] * len(leading))).expand_as(generator)
        action = tables[index, generator]
        proposal = torch.einsum("...i,...ij->...j", state, action)
        state = torch.where(word_mask[..., position, None], proposal, state)
    return state


class PresentationCompiler(nn.Module):
    def __init__(
        self,
        algebra: PresentedAlgebraConfig,
        config: LanguageConfig,
        *,
        projection: str,
    ):
        super().__init__()
        if projection not in {"presented", "row_soft"}:
            raise PresentedReasoningError("unknown compiler projection")
        self.algebra = algebra
        self.config = config
        self.projection = projection
        self.encoder = RecordLanguageEncoder(algebra, config)
        self.generator_slot = nn.Embedding(algebra.maximum_generators, config.width)
        self.input_slot = nn.Embedding(algebra.carrier_size, config.width)
        self.cross_attention = nn.MultiheadAttention(
            config.width, config.heads, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(config.width)
        self.ff = nn.Sequential(
            nn.Linear(config.width, config.width * config.ff_multiplier),
            nn.GELU(),
            nn.Linear(config.width * config.ff_multiplier, config.width),
        )
        self.ff_norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, algebra.carrier_size)

    def logits(self, source: RenderedSource) -> torch.Tensor:
        records = self.encoder(source)
        generators = torch.arange(
            self.algebra.maximum_generators, device=records.device
        )
        inputs = torch.arange(self.algebra.carrier_size, device=records.device)
        slots = (
            self.generator_slot(generators)[:, None]
            + self.input_slot(inputs)[None]
        ).reshape(1, -1, self.config.width)
        slots = slots.expand(records.shape[0], -1, -1)
        for _ in range(2):
            read, _ = self.cross_attention(
                slots,
                records,
                records,
                key_padding_mask=~source.record_mask,
                need_weights=False,
            )
            slots = self.attention_norm(slots + read)
            slots = self.ff_norm(slots + self.ff(slots))
        return self.output(slots).reshape(
            records.shape[0],
            self.algebra.maximum_generators,
            self.algebra.carrier_size,
            self.algebra.carrier_size,
        )

    def forward(
        self,
        source: RenderedSource,
        generator_mask: torch.Tensor,
        *,
        hard: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits(source)
        if self.projection == "presented":
            tables = sinkhorn(logits, self.config.sinkhorn_rounds)
            if hard:
                tables = greedy_permutation(tables)
        else:
            tables = logits.softmax(-1)
            if hard:
                tables = F.one_hot(
                    tables.argmax(-1), self.algebra.carrier_size
                ).to(tables.dtype)
        identity = torch.eye(
            self.algebra.carrier_size, device=tables.device, dtype=tables.dtype
        )
        tables = torch.where(
            generator_mask[..., None, None], tables, identity[None, None]
        )
        return logits, tables


class DirectLanguageControl(nn.Module):
    def __init__(self, algebra: PresentedAlgebraConfig, config: LanguageConfig):
        super().__init__()
        self.algebra = algebra
        self.encoder = RecordLanguageEncoder(algebra, config)
        self.query_position = nn.Embedding(algebra.maximum_word_length + 1, config.width)
        self.query_recurrence = nn.GRUCell(config.width, config.width)
        self.cross_attention = nn.MultiheadAttention(
            config.width, config.heads, batch_first=True
        )
        self.answer = nn.Linear(config.width, algebra.carrier_size)

    def forward(self, batch: PresentedBatch, source: RenderedSource) -> torch.Tensor:
        records = self.encoder(source)
        hidden = self.encoder.token(
            batch.query_start + GENERATOR_BASE + self.algebra.maximum_generators
        )
        for position in range(batch.query_word.shape[-1]):
            token = self.encoder.token(batch.query_word[:, position] + GENERATOR_BASE)
            token = token + self.query_position.weight[position + 1]
            proposal = self.query_recurrence(token, hidden)
            hidden = torch.where(
                batch.query_word_mask[:, position, None], proposal, hidden
            )
        read, _ = self.cross_attention(
            hidden[:, None],
            records,
            records,
            key_padding_mask=~source.record_mask,
            need_weights=False,
        )
        return self.answer(hidden + read[:, 0])


class LearnedPSPAGate(nn.Module):
    def __init__(self, algebra: PresentedAlgebraConfig, config: LanguageConfig):
        super().__init__()
        self.algebra = algebra
        self.presented = PresentationCompiler(
            algebra, config, projection="presented"
        )
        self.row_soft = PresentationCompiler(algebra, config, projection="row_soft")
        self.direct = DirectLanguageControl(algebra, config)


def compiler_loss(tables: torch.Tensor, batch: PresentedBatch) -> torch.Tensor:
    batch_index = torch.arange(tables.shape[0], device=tables.device)[:, None]
    observed = tables[
        batch_index,
        batch.observation_generator,
        batch.observation_input,
        batch.observation_output,
    ]
    observation_loss = -observed.clamp_min(1e-8).log()[batch.observation_mask].mean()
    challenge = execute_word(
        tables,
        batch.challenge_start,
        batch.challenge_word,
        batch.challenge_word_mask,
    )
    outcome = challenge.gather(-1, batch.challenge_outcome[..., None]).squeeze(-1)
    challenge_loss = -outcome.clamp_min(1e-8).log()[batch.challenge_mask].mean()
    return observation_loss + challenge_loss


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
    model: LearnedPSPAGate,
    *,
    family: int,
    length: int,
    count: int,
    seed: int,
    renderer_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    cpu_batch = generate_batch(count, length, model.algebra, seed=seed, family=family)
    source = render_source(cpu_batch, model.algebra, seed=renderer_seed)
    shuffled_batch = replace(
        cpu_batch, challenge_outcome=cpu_batch.challenge_outcome.roll(1, 0)
    )
    shuffled_source = render_source(
        shuffled_batch, model.algebra, seed=renderer_seed
    )
    batch = cpu_batch.to(device)
    source = source.to(device)
    shuffled_source = shuffled_source.to(device)
    _, presented = model.presented(source, batch.generator_mask, hard=True)
    _, row_soft = model.row_soft(source, batch.generator_mask, hard=True)
    _, shuffled = model.presented(
        shuffled_source, batch.generator_mask, hard=True
    )
    presented_answer = execute_word(
        presented, batch.query_start, batch.query_word, batch.query_word_mask
    ).argmax(-1)
    row_answer = execute_word(
        row_soft, batch.query_start, batch.query_word, batch.query_word_mask
    ).argmax(-1)
    shuffled_answer = execute_word(
        shuffled, batch.query_start, batch.query_word, batch.query_word_mask
    ).argmax(-1)
    swapped_answer = execute_word(
        presented.roll(1, 0),
        batch.query_start,
        batch.query_word,
        batch.query_word_mask,
    ).argmax(-1)
    direct_answer = model.direct(batch, source).argmax(-1)
    challenge = execute_word(
        presented,
        batch.challenge_start,
        batch.challenge_word,
        batch.challenge_word_mask,
    ).argmax(-1)
    table_exact = presented.argmax(-1).eq(batch.true_tables.long()).all((-1, -2))
    return {
        "family": FAMILIES[family],
        "length": length,
        "count": count,
        "seed": seed,
        "renderer_seed": renderer_seed,
        "batch_sha256": batch_sha256(cpu_batch),
        "presented_accuracy": presented_answer.eq(batch.answer).float().mean().item(),
        "row_soft_accuracy": row_answer.eq(batch.answer).float().mean().item(),
        "direct_accuracy": direct_answer.eq(batch.answer).float().mean().item(),
        "shuffle_challenge_accuracy": shuffled_answer.eq(batch.answer).float().mean().item(),
        "lineage_swap_accuracy": swapped_answer.eq(batch.answer).float().mean().item(),
        "challenge_exact": (
            challenge.eq(batch.challenge_outcome) & batch.challenge_mask
        ).sum().div(batch.challenge_mask.sum()).item(),
        "selected_table_exact": table_exact.float().mean().item(),
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
    language = LanguageConfig()
    model = LearnedPSPAGate(algebra, language).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    started = time.monotonic()
    train_log = []
    for update in range(1, args.updates + 1):
        length = 1 + ((update * 997 + args.seed) % args.train_max_length)
        cpu_batch = generate_batch(
            args.batch_size, length, algebra, seed=args.data_seed + update
        )
        rendered = render_source(
            cpu_batch, algebra, seed=args.renderer_seed + update
        )
        batch = cpu_batch.to(device)
        rendered = rendered.to(device)
        model.train()
        _, presented = model.presented(rendered, batch.generator_mask, hard=False)
        _, row_soft = model.row_soft(rendered, batch.generator_mask, hard=False)
        presented_loss = compiler_loss(presented, batch)
        row_soft_loss = compiler_loss(row_soft, batch)
        direct_logits = model.direct(batch, rendered)
        direct_loss = F.cross_entropy(direct_logits, batch.answer)
        loss = presented_loss + row_soft_loss + direct_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update == 1 or update % args.log_every == 0 or update == args.updates:
            row = {
                "update": update,
                "length": length,
                "loss": loss.item(),
                "presented_source_loss": presented_loss.item(),
                "row_soft_source_loss": row_soft_loss.item(),
                "direct_accuracy": direct_logits.argmax(-1)
                .eq(batch.answer)
                .float()
                .mean()
                .item(),
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
                    renderer_seed=args.eval_renderer_seed + family * 100 + length,
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
            "language_config": asdict(language),
            "model": model.state_dict(),
        },
        checkpoint,
    )
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "seed": args.seed,
        "data_seed": args.data_seed,
        "renderer_seed": args.renderer_seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "charged_examples": args.updates * args.batch_size,
        "train_max_length": args.train_max_length,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "algebra_config": asdict(algebra),
        "language_config": asdict(language),
        "parameters": {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in (
                ("presented", model.presented),
                ("row_soft", model.row_soft),
                ("direct", model.direct),
            )
        },
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
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--data-seed", type=int, default=20260920)
    parser.add_argument("--renderer-seed", type=int, default=71000)
    parser.add_argument("--eval-seed", type=int, default=72000)
    parser.add_argument("--eval-renderer-seed", type=int, default=73000)
    parser.add_argument("--updates", type=int, default=2000)
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
