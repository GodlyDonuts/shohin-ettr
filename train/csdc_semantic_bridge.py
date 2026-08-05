#!/usr/bin/env python3
"""Learn rendered source challenges and feed them to frozen CSDC."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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

from evaluate_counterexample_selected_closure import (
    binary_completion_candidates,
    select_with_challenges,
)
from learned_pspa_language_reasoning import (
    AFTER,
    BECOMES,
    CHALLENGE_A,
    CHALLENGE_B,
    FROM,
    MAP,
    OBSERVATION_A,
    OBSERVATION_B,
    PAD,
    RESULT,
    SEP,
    START,
    TO,
    USING,
    LanguageConfig,
    LearnedPSPAGate,
    RenderedSource,
    execute_word,
    generator_token,
    state_token,
    vocabulary_size,
)
from prompt_selected_presented_algebra import PresentedAlgebraConfig
from pspa_presented_reasoning import (
    FAMILIES,
    PresentedBatch,
    PresentedReasoningError,
    batch_sha256,
    generate_batch,
)


SCHEMA = "shohin-csdc-semantic-challenge-bridge-v1"


@dataclass(frozen=True, slots=True)
class SemanticSource:
    rendered: RenderedSource
    challenge_record: torch.Tensor
    challenge_start: torch.Tensor
    challenge_outcome: torch.Tensor
    challenge_length: torch.Tensor
    challenge_word: torch.Tensor

    def to(self, device: torch.device) -> SemanticSource:
        return SemanticSource(
            rendered=self.rendered.to(device),
            challenge_record=self.challenge_record.to(device),
            challenge_start=self.challenge_start.to(device),
            challenge_outcome=self.challenge_outcome.to(device),
            challenge_length=self.challenge_length.to(device),
            challenge_word=self.challenge_word.to(device),
        )


@dataclass(frozen=True, slots=True)
class SemanticBridgeConfig:
    width: int = 64
    heads: int = 4
    layers: int = 2
    ff_multiplier: int = 2


@dataclass(frozen=True, slots=True)
class SemanticLogits:
    kind: torch.Tensor
    start: torch.Tensor
    outcome: torch.Tensor
    length: torch.Tensor
    word: torch.Tensor


@dataclass(frozen=True, slots=True)
class DecodedChallenges:
    record_index: torch.Tensor
    start: torch.Tensor
    outcome: torch.Tensor
    length: torch.Tensor
    word: torch.Tensor
    word_mask: torch.Tensor


def render_semantic_source(
    batch: PresentedBatch,
    config: PresentedAlgebraConfig,
    *,
    seed: int,
    templates: tuple[int, ...],
) -> SemanticSource:
    """Render records and retain aligned source-only semantic labels."""

    if batch.family.device.type != "cpu":
        raise PresentedReasoningError("semantic rendering requires a CPU batch")
    if not templates or any(template not in range(4) for template in templates):
        raise PresentedReasoningError("unknown challenge renderer template")
    maximum_records = config.maximum_observations + config.maximum_challenges
    maximum_tokens = config.maximum_word_length + 7
    rendered_rows: list[list[tuple[list[int], bool, int, int, list[int]]]] = []
    for row in range(batch.family.shape[0]):
        rng = random.Random(seed * 1_000_003 + row * 8191)
        records: list[tuple[list[int], bool, int, int, list[int]]] = []
        for index in range(config.maximum_observations):
            if not bool(batch.observation_mask[row, index]):
                continue
            generator = int(batch.observation_generator[row, index])
            source = int(batch.observation_input[row, index])
            target = int(batch.observation_output[row, index])
            if rng.randrange(2) == 0:
                tokens = [
                    OBSERVATION_A,
                    generator_token(generator),
                    FROM,
                    state_token(config, source),
                    TO,
                    state_token(config, target),
                    SEP,
                ]
            else:
                tokens = [
                    OBSERVATION_B,
                    MAP,
                    state_token(config, source),
                    USING,
                    generator_token(generator),
                    BECOMES,
                    state_token(config, target),
                    SEP,
                ]
            records.append((tokens, False, 0, 0, []))
        for index in range(config.maximum_challenges):
            if not bool(batch.challenge_mask[row, index]):
                continue
            word_values = [
                int(value)
                for value, keep in zip(
                    batch.challenge_word[row, index],
                    batch.challenge_word_mask[row, index],
                    strict=True,
                )
                if bool(keep)
            ]
            word_tokens = [generator_token(value) for value in word_values]
            source_value = int(batch.challenge_start[row, index])
            outcome_value = int(batch.challenge_outcome[row, index])
            source = state_token(config, source_value)
            outcome = state_token(config, outcome_value)
            template = templates[rng.randrange(len(templates))]
            if template == 0:
                tokens = [
                    CHALLENGE_A,
                    START,
                    source,
                    AFTER,
                    *word_tokens,
                    RESULT,
                    outcome,
                    SEP,
                ]
            elif template == 1:
                tokens = [
                    CHALLENGE_B,
                    source,
                    USING,
                    *word_tokens,
                    TO,
                    outcome,
                    SEP,
                ]
            elif template == 2:
                tokens = [
                    CHALLENGE_A,
                    RESULT,
                    outcome,
                    START,
                    source,
                    AFTER,
                    *word_tokens,
                    SEP,
                ]
            else:
                tokens = [
                    CHALLENGE_B,
                    RESULT,
                    outcome,
                    START,
                    source,
                    USING,
                    *word_tokens,
                    SEP,
                ]
            records.append(
                (tokens, True, source_value, outcome_value, word_values)
            )
        rng.shuffle(records)
        if len(records) > maximum_records or any(
            len(record[0]) > maximum_tokens for record in records
        ):
            raise PresentedReasoningError("semantic source exceeds fixed geometry")
        rendered_rows.append(records)

    geometry = (batch.family.shape[0], maximum_records)
    tokens = torch.full((*geometry, maximum_tokens), PAD, dtype=torch.long)
    token_mask = torch.zeros_like(tokens, dtype=torch.bool)
    record_mask = torch.zeros(geometry, dtype=torch.bool)
    challenge_record = torch.zeros(geometry, dtype=torch.bool)
    challenge_start = torch.zeros(geometry, dtype=torch.long)
    challenge_outcome = torch.zeros(geometry, dtype=torch.long)
    challenge_length = torch.ones(geometry, dtype=torch.long)
    challenge_word = torch.zeros(
        *geometry, config.maximum_word_length, dtype=torch.long
    )
    for row, records in enumerate(rendered_rows):
        for index, (record, is_challenge, source, outcome, word) in enumerate(records):
            tokens[row, index, : len(record)] = torch.tensor(record)
            token_mask[row, index, : len(record)] = True
            record_mask[row, index] = True
            challenge_record[row, index] = is_challenge
            if is_challenge:
                challenge_start[row, index] = source
                challenge_outcome[row, index] = outcome
                challenge_length[row, index] = len(word)
                challenge_word[row, index, : len(word)] = torch.tensor(word)
    return SemanticSource(
        rendered=RenderedSource(
            tokens=tokens, token_mask=token_mask, record_mask=record_mask
        ),
        challenge_record=challenge_record,
        challenge_start=challenge_start,
        challenge_outcome=challenge_outcome,
        challenge_length=challenge_length,
        challenge_word=challenge_word,
    )


class SemanticChallengeParser(nn.Module):
    def __init__(
        self,
        algebra: PresentedAlgebraConfig,
        config: SemanticBridgeConfig,
    ):
        super().__init__()
        self.algebra = algebra
        self.config = config
        maximum_tokens = algebra.maximum_word_length + 7
        self.token = nn.Embedding(vocabulary_size(algebra), config.width)
        self.position = nn.Embedding(maximum_tokens + 1, config.width)
        self.cls = nn.Parameter(torch.randn(config.width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            config.width,
            config.heads,
            config.width * config.ff_multiplier,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, config.layers)
        self.kind = nn.Linear(config.width, 2)
        self.start = nn.Linear(config.width, algebra.carrier_size)
        self.outcome = nn.Linear(config.width, algebra.carrier_size)
        self.length = nn.Linear(config.width, algebra.maximum_word_length)
        self.word = nn.Linear(
            config.width,
            algebra.maximum_word_length * algebra.maximum_generators,
        )

    def forward(self, source: RenderedSource) -> SemanticLogits:
        batch, records, tokens = source.tokens.shape
        flat_tokens = source.tokens.reshape(batch * records, tokens)
        flat_mask = source.token_mask.reshape(batch * records, tokens)
        valid = source.record_mask.reshape(-1)
        summary = torch.zeros(
            batch * records,
            self.config.width,
            dtype=self.cls.dtype,
            device=source.tokens.device,
        )
        selected_tokens = flat_tokens[valid]
        selected_mask = flat_mask[valid]
        positions = torch.arange(tokens + 1, device=source.tokens.device)
        cls = self.cls[None, None].expand(selected_tokens.shape[0], 1, -1)
        encoded = torch.cat((cls, self.token(selected_tokens)), 1)
        encoded = encoded + self.position(positions)[None]
        mask = torch.cat(
            (
                torch.ones(
                    selected_mask.shape[0],
                    1,
                    dtype=torch.bool,
                    device=selected_mask.device,
                ),
                selected_mask,
            ),
            1,
        )
        encoded = self.encoder(encoded, src_key_padding_mask=~mask)
        summary[valid] = encoded[:, 0]
        summary = summary.reshape(batch, records, self.config.width)
        return SemanticLogits(
            kind=self.kind(summary),
            start=self.start(summary),
            outcome=self.outcome(summary),
            length=self.length(summary),
            word=self.word(summary).reshape(
                batch,
                records,
                self.algebra.maximum_word_length,
                self.algebra.maximum_generators,
            ),
        )

    def decode(
        self,
        logits: SemanticLogits,
        source: RenderedSource,
    ) -> DecodedChallenges:
        score = logits.kind.softmax(-1)[..., 1].masked_fill(
            ~source.record_mask, -torch.inf
        )
        record_index = score.topk(self.algebra.maximum_challenges, -1).indices

        def gather(value: torch.Tensor) -> torch.Tensor:
            index = record_index
            for _ in value.shape[2:]:
                index = index.unsqueeze(-1)
            return value.gather(
                1, index.expand(*record_index.shape, *value.shape[2:])
            )

        start = gather(logits.start).argmax(-1)
        outcome = gather(logits.outcome).argmax(-1)
        length = gather(logits.length).argmax(-1) + 1
        word = gather(logits.word).argmax(-1)
        positions = torch.arange(
            self.algebra.maximum_word_length, device=word.device
        )
        word_mask = positions[None, None] < length[..., None]
        return DecodedChallenges(
            record_index=record_index,
            start=start,
            outcome=outcome,
            length=length,
            word=word,
            word_mask=word_mask,
        )


def semantic_loss(
    logits: SemanticLogits,
    source: SemanticSource,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = source.rendered.record_mask
    challenge = source.challenge_record
    kind_loss = F.cross_entropy(
        logits.kind[valid], challenge[valid].long()
    )
    start_loss = F.cross_entropy(logits.start[challenge], source.challenge_start[challenge])
    outcome_loss = F.cross_entropy(
        logits.outcome[challenge], source.challenge_outcome[challenge]
    )
    length_loss = F.cross_entropy(
        logits.length[challenge], source.challenge_length[challenge] - 1
    )
    challenge_lengths = source.challenge_length[challenge]
    positions = torch.arange(
        source.challenge_word.shape[-1], device=challenge.device
    )
    word_mask = positions[None] < challenge_lengths[:, None]
    word_loss = F.cross_entropy(
        logits.word[challenge][word_mask], source.challenge_word[challenge][word_mask]
    )
    loss = kind_loss + start_loss + outcome_loss + length_loss + word_loss
    metrics = {
        "kind_loss": kind_loss.item(),
        "start_loss": start_loss.item(),
        "outcome_loss": outcome_loss.item(),
        "length_loss": length_loss.item(),
        "word_loss": word_loss.item(),
    }
    return loss, metrics


def gather_targets(
    source: SemanticSource,
    record_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    def gather(value: torch.Tensor) -> torch.Tensor:
        index = record_index
        for _ in value.shape[2:]:
            index = index.unsqueeze(-1)
        return value.gather(
            1, index.expand(*record_index.shape, *value.shape[2:])
        )

    return (
        gather(source.challenge_record),
        gather(source.challenge_start),
        gather(source.challenge_outcome),
        gather(source.challenge_length),
        gather(source.challenge_word),
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
def evaluate_cohort(
    parser: SemanticChallengeParser,
    reasoner: LearnedPSPAGate,
    *,
    family: int,
    length: int,
    count: int,
    seed: int,
    renderer_seed: int,
    templates: tuple[int, ...],
    device: torch.device,
) -> dict[str, Any]:
    cpu_batch = generate_batch(count, length, reasoner.algebra, seed=seed, family=family)
    cpu_source = render_semantic_source(
        cpu_batch, reasoner.algebra, seed=renderer_seed, templates=templates
    )
    batch = cpu_batch.to(device)
    source = cpu_source.to(device)
    logits = parser(source.rendered)
    decoded = parser.decode(logits, source.rendered)
    _, row_probabilities = reasoner.row_soft(
        source.rendered, batch.generator_mask, hard=False
    )
    candidates = binary_completion_candidates(row_probabilities, batch.generator_mask)
    selected, _, _ = select_with_challenges(
        candidates,
        decoded.start,
        decoded.word,
        decoded.word_mask,
        decoded.outcome,
        torch.ones_like(decoded.start, dtype=torch.bool),
    )
    oracle, _, _ = select_with_challenges(
        candidates,
        batch.challenge_start,
        batch.challenge_word,
        batch.challenge_word_mask,
        batch.challenge_outcome,
        batch.challenge_mask,
    )
    shuffled, _, _ = select_with_challenges(
        candidates,
        decoded.start,
        decoded.word,
        decoded.word_mask,
        decoded.outcome.roll(1, 0),
        torch.ones_like(decoded.start, dtype=torch.bool),
    )

    def answer(tables: torch.Tensor) -> torch.Tensor:
        return execute_word(
            tables, batch.query_start, batch.query_word, batch.query_word_mask
        ).argmax(-1)

    true_record, true_start, true_outcome, true_length, true_word = gather_targets(
        source, decoded.record_index
    )
    positions = torch.arange(reasoner.algebra.maximum_word_length, device=device)
    true_word_mask = positions[None, None] < true_length[..., None]
    word_exact = (
        decoded.word.eq(true_word) | ~true_word_mask
    ).all(-1)
    tuple_exact = (
        true_record
        & decoded.start.eq(true_start)
        & decoded.outcome.eq(true_outcome)
        & decoded.length.eq(true_length)
        & word_exact
    )
    table_exact = selected.argmax(-1).eq(batch.true_tables.long()).all((-1, -2))
    return {
        "family": FAMILIES[family],
        "length": length,
        "count": count,
        "seed": seed,
        "renderer_seed": renderer_seed,
        "templates": list(templates),
        "batch_sha256": batch_sha256(cpu_batch),
        "learned_accuracy": answer(selected).eq(batch.answer).float().mean().item(),
        "oracle_accuracy": answer(oracle).eq(batch.answer).float().mean().item(),
        "shuffle_outcome_accuracy": answer(shuffled)
        .eq(batch.answer)
        .float()
        .mean()
        .item(),
        "lineage_swap_accuracy": answer(selected.roll(1, 0))
        .eq(batch.answer)
        .float()
        .mean()
        .item(),
        "challenge_record_exact": true_record.float().mean().item(),
        "challenge_start_exact": decoded.start.eq(true_start).float().mean().item(),
        "challenge_outcome_exact": decoded.outcome.eq(true_outcome).float().mean().item(),
        "challenge_length_exact": decoded.length.eq(true_length).float().mean().item(),
        "challenge_word_exact": word_exact.float().mean().item(),
        "challenge_tuple_exact": tuple_exact.float().mean().item(),
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
    reasoner_checkpoint_bytes = args.reasoner_checkpoint.read_bytes()
    reasoner_checkpoint = torch.load(
        args.reasoner_checkpoint, map_location="cpu", weights_only=True
    )
    algebra = PresentedAlgebraConfig(**reasoner_checkpoint["algebra_config"])
    language = LanguageConfig(**reasoner_checkpoint["language_config"])
    reasoner = LearnedPSPAGate(algebra, language).to(device)
    reasoner.load_state_dict(reasoner_checkpoint["model"])
    reasoner.eval()
    for parameter in reasoner.parameters():
        parameter.requires_grad_(False)

    parser_config = SemanticBridgeConfig()
    parser = SemanticChallengeParser(algebra, parser_config).to(device)
    optimizer = torch.optim.AdamW(
        parser.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_templates = tuple(int(value) for value in args.train_templates.split(","))
    started = time.monotonic()
    train_log = []
    for update in range(1, args.updates + 1):
        cpu_batch = generate_batch(
            args.batch_size,
            1 + ((update * 997 + args.seed) % 4),
            algebra,
            seed=args.data_seed + update,
        )
        source = render_semantic_source(
            cpu_batch,
            algebra,
            seed=args.renderer_seed + update,
            templates=train_templates,
        ).to(device)
        parser.train()
        logits = parser(source.rendered)
        loss, loss_metrics = semantic_loss(logits, source)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parser.parameters(), 1.0)
        optimizer.step()
        if update == 1 or update % args.log_every == 0 or update == args.updates:
            decoded = parser.decode(logits, source.rendered)
            true_record, true_start, true_outcome, true_length, true_word = gather_targets(
                source, decoded.record_index
            )
            positions = torch.arange(algebra.maximum_word_length, device=device)
            true_word_mask = positions[None, None] < true_length[..., None]
            row = {
                "update": update,
                "loss": loss.item(),
                **loss_metrics,
                "record_exact": true_record.float().mean().item(),
                "start_exact": decoded.start.eq(true_start).float().mean().item(),
                "outcome_exact": decoded.outcome.eq(true_outcome).float().mean().item(),
                "length_exact": decoded.length.eq(true_length).float().mean().item(),
                "word_exact": (
                    decoded.word.eq(true_word) | ~true_word_mask
                ).all(-1).float().mean().item(),
            }
            train_log.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    elapsed = time.monotonic() - started
    parser.eval()
    evaluations = []
    for split, templates, seed_offset in (
        ("development", train_templates, 0),
        ("renderer_shift", (3,), 10000),
    ):
        for family in range(len(FAMILIES)):
            for length in (8, 12):
                row = evaluate_cohort(
                    parser,
                    reasoner,
                    family=family,
                    length=length,
                    count=args.eval_count,
                    seed=args.eval_seed + seed_offset + family * 100 + length,
                    renderer_seed=(
                        args.eval_renderer_seed
                        + seed_offset
                        + family * 100
                        + length
                    ),
                    templates=templates,
                    device=device,
                )
                row["split"] = split
                evaluations.append(row)
    checkpoint = args.output.with_suffix(".pt")
    if checkpoint.exists():
        raise PresentedReasoningError(f"refusing existing checkpoint: {checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": SCHEMA,
            "seed": args.seed,
            "algebra_config": asdict(algebra),
            "parser_config": asdict(parser_config),
            "parser": parser.state_dict(),
            "reasoner_checkpoint_sha256": hashlib.sha256(
                reasoner_checkpoint_bytes
            ).hexdigest(),
        },
        checkpoint,
    )
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "seed": args.seed,
        "data_seed": args.data_seed,
        "renderer_seed": args.renderer_seed,
        "train_templates": list(train_templates),
        "updates": args.updates,
        "batch_size": args.batch_size,
        "charged_examples": args.updates * args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "parser_config": asdict(parser_config),
        "parser_parameters": sum(parameter.numel() for parameter in parser.parameters()),
        "reasoner_checkpoint": str(args.reasoner_checkpoint),
        "reasoner_checkpoint_sha256": hashlib.sha256(
            reasoner_checkpoint_bytes
        ).hexdigest(),
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
    parser.add_argument("--reasoner-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--data-seed", type=int, default=20261001)
    parser.add_argument("--renderer-seed", type=int, default=91000)
    parser.add_argument("--eval-seed", type=int, default=92000)
    parser.add_argument("--eval-renderer-seed", type=int, default=93000)
    parser.add_argument("--train-templates", default="0,1,2")
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-count", type=int, default=1024)
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
                "parser_parameters": completed["parser_parameters"],
                "examples_per_second": completed["examples_per_second"],
            },
            sort_keys=True,
        )
    )
