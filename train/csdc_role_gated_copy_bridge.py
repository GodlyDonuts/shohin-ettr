#!/usr/bin/env python3
"""Copy model-selected semantic source tokens into frozen CSDC."""

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

from csdc_semantic_bridge import (
    DecodedChallenges,
    SemanticSource,
    evaluate_cohort,
    gather_targets,
    render_semantic_source,
)
from learned_pspa_language_reasoning import (
    CHALLENGE_A,
    CHALLENGE_B,
    GENERATOR_BASE,
    RESULT,
    START,
    LanguageConfig,
    LearnedPSPAGate,
    RenderedSource,
    state_token,
    vocabulary_size,
)
from prompt_selected_presented_algebra import PresentedAlgebraConfig
from pspa_presented_reasoning import (
    PresentedReasoningError,
    generate_batch,
)


SCHEMA = "shohin-csdc-role-gated-copy-bridge-v1"
OTHER_ROLE = 0
START_ROLE = 1
OUTCOME_ROLE = 2
WORD_ROLE = 3
ROLE_COUNT = 4


@dataclass(frozen=True, slots=True)
class CopyBridgeConfig:
    width: int = 64
    heads: int = 4
    layers: int = 2
    ff_multiplier: int = 2


@dataclass(frozen=True, slots=True)
class CopyLogits:
    kind: torch.Tensor
    role: torch.Tensor


@dataclass(frozen=True, slots=True)
class RoleLabeledSource:
    source: SemanticSource
    token_role: torch.Tensor

    def to(self, device: torch.device) -> RoleLabeledSource:
        return RoleLabeledSource(
            source=self.source.to(device),
            token_role=self.token_role.to(device),
        )


def label_source_roles(
    source: SemanticSource,
    algebra: PresentedAlgebraConfig,
) -> RoleLabeledSource:
    """Attach source-token roles without exposing a query or answer."""

    tokens = source.rendered.tokens
    roles = torch.full_like(tokens, OTHER_ROLE)
    generator_end = GENERATOR_BASE + algebra.maximum_generators
    generator = (tokens >= GENERATOR_BASE) & (tokens < generator_end)
    roles[generator & source.challenge_record[..., None]] = WORD_ROLE
    for row, record in source.challenge_record.nonzero(as_tuple=False).tolist():
        active = int(source.rendered.token_mask[row, record].sum().item())
        values = tokens[row, record]
        first = int(values[0])
        second = int(values[1])
        if first == CHALLENGE_A and second == START:
            start_position = 2
            outcome_position = active - 2
        elif first == CHALLENGE_A and second == RESULT:
            outcome_position = 2
            start_position = 4
        elif first == CHALLENGE_B and second == RESULT:
            outcome_position = 2
            start_position = 4
        elif first == CHALLENGE_B:
            start_position = 1
            outcome_position = active - 2
        else:
            raise PresentedReasoningError("unknown challenge record layout")
        roles[row, record, start_position] = START_ROLE
        roles[row, record, outcome_position] = OUTCOME_ROLE
    return RoleLabeledSource(source=source, token_role=roles)


class RoleGatedCopyParser(nn.Module):
    def __init__(
        self,
        algebra: PresentedAlgebraConfig,
        config: CopyBridgeConfig,
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
        self.role = nn.Linear(config.width, ROLE_COUNT)

    def forward(self, source: RenderedSource) -> CopyLogits:
        batch, records, tokens = source.tokens.shape
        flat_tokens = source.tokens.reshape(batch * records, tokens)
        flat_mask = source.token_mask.reshape(batch * records, tokens)
        valid = source.record_mask.reshape(-1)
        summaries = torch.zeros(
            batch * records,
            self.config.width,
            dtype=self.cls.dtype,
            device=source.tokens.device,
        )
        token_states = torch.zeros(
            batch * records,
            tokens,
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
        summaries[valid] = encoded[:, 0]
        token_states[valid] = encoded[:, 1:]
        summaries = summaries.reshape(batch, records, self.config.width)
        token_states = token_states.reshape(
            batch, records, tokens, self.config.width
        )
        return CopyLogits(
            kind=self.kind(summaries),
            role=self.role(token_states),
        )

    def decode(
        self,
        logits: CopyLogits,
        source: RenderedSource,
    ) -> DecodedChallenges:
        challenge_score = logits.kind.softmax(-1)[..., 1].masked_fill(
            ~source.record_mask, -torch.inf
        )
        record_index = challenge_score.topk(
            self.algebra.maximum_challenges, -1
        ).indices

        def gather_records(value: torch.Tensor) -> torch.Tensor:
            index = record_index
            for _ in value.shape[2:]:
                index = index.unsqueeze(-1)
            return value.gather(
                1, index.expand(*record_index.shape, *value.shape[2:])
            )

        tokens = gather_records(source.tokens)
        token_mask = gather_records(source.token_mask)
        role_logits = gather_records(logits.role)

        def copy_state(role: int) -> torch.Tensor:
            score = role_logits[..., role].masked_fill(~token_mask, -torch.inf)
            position = score.argmax(-1)
            selected = tokens.gather(-1, position[..., None]).squeeze(-1)
            base = state_token(self.algebra, 0)
            return (selected - base).clamp(0, self.algebra.carrier_size - 1)

        start = copy_state(START_ROLE)
        outcome = copy_state(OUTCOME_ROLE)
        predicted_role = role_logits.argmax(-1)
        selected_word = predicted_role.eq(WORD_ROLE) & token_mask
        token_positions = torch.arange(tokens.shape[-1], device=tokens.device)
        order_key = torch.where(
            selected_word,
            token_positions[None, None],
            tokens.shape[-1],
        )
        ordered_position = order_key.argsort(-1)[
            ..., : self.algebra.maximum_word_length
        ]
        copied = tokens.gather(-1, ordered_position)
        word = (copied - GENERATOR_BASE).clamp(
            0, self.algebra.maximum_generators - 1
        )
        length = selected_word.sum(-1).clamp(
            1, self.algebra.maximum_word_length
        )
        word_position = torch.arange(
            self.algebra.maximum_word_length, device=tokens.device
        )
        word_mask = word_position[None, None] < length[..., None]
        return DecodedChallenges(
            record_index=record_index,
            start=start,
            outcome=outcome,
            length=length,
            word=word,
            word_mask=word_mask,
        )


def copy_loss(
    logits: CopyLogits,
    labeled: RoleLabeledSource,
) -> tuple[torch.Tensor, dict[str, float]]:
    source = labeled.source
    valid_records = source.rendered.record_mask
    kind_loss = F.cross_entropy(
        logits.kind[valid_records], source.challenge_record[valid_records].long()
    )
    valid_tokens = (
        source.rendered.token_mask & source.challenge_record[..., None]
    )
    role_loss = F.cross_entropy(
        logits.role[valid_tokens], labeled.token_role[valid_tokens]
    )
    loss = kind_loss + role_loss
    return loss, {
        "kind_loss": kind_loss.item(),
        "role_loss": role_loss.item(),
    }


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

    parser_config = CopyBridgeConfig()
    parser = RoleGatedCopyParser(algebra, parser_config).to(device)
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
        )
        labeled = label_source_roles(source, algebra).to(device)
        parser.train()
        logits = parser(labeled.source.rendered)
        loss, loss_metrics = copy_loss(logits, labeled)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parser.parameters(), 1.0)
        optimizer.step()
        if update == 1 or update % args.log_every == 0 or update == args.updates:
            parser.eval()
            with torch.inference_mode():
                fresh_logits = parser(labeled.source.rendered)
                decoded = parser.decode(fresh_logits, labeled.source.rendered)
                true_record, true_start, true_outcome, true_length, true_word = (
                    gather_targets(labeled.source, decoded.record_index)
                )
                positions = torch.arange(
                    algebra.maximum_word_length, device=device
                )
                word_mask = positions[None, None] < true_length[..., None]
                row = {
                    "update": update,
                    "loss": loss.item(),
                    **loss_metrics,
                    "record_exact": true_record.float().mean().item(),
                    "start_exact": decoded.start.eq(true_start).float().mean().item(),
                    "outcome_exact": decoded.outcome.eq(true_outcome)
                    .float()
                    .mean()
                    .item(),
                    "length_exact": decoded.length.eq(true_length)
                    .float()
                    .mean()
                    .item(),
                    "word_exact": (decoded.word.eq(true_word) | ~word_mask)
                    .all(-1)
                    .float()
                    .mean()
                    .item(),
                    "token_role_exact": fresh_logits.role.argmax(-1)
                    .eq(labeled.token_role)[
                        labeled.source.rendered.token_mask
                        & labeled.source.challenge_record[..., None]
                    ]
                    .float()
                    .mean()
                    .item(),
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
        for family in range(3):
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
    parser.add_argument("--seed", type=int, default=59)
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
