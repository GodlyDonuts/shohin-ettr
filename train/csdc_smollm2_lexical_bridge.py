#!/usr/bin/env python3
"""Feed model-decoded natural-language challenges into frozen CSDC."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from csdc_role_gated_copy_bridge import (
    OTHER_ROLE,
    OUTCOME_ROLE,
    ROLE_COUNT,
    START_ROLE,
    WORD_ROLE,
)
from csdc_semantic_bridge import DecodedChallenges, render_semantic_source
from evaluate_counterexample_selected_closure import (
    binary_completion_candidates,
    select_with_challenges,
)
from frozen_pointer_backbone import load_frozen_pointer_backbone
from learned_pspa_language_reasoning import (
    LanguageConfig,
    LearnedPSPAGate,
    execute_word,
)
from prompt_selected_presented_algebra import PresentedAlgebraConfig
from pspa_presented_reasoning import (
    FAMILIES,
    PresentedBatch,
    PresentedReasoningError,
    batch_sha256,
    generate_batch,
)


SCHEMA = "shohin-csdc-smollm2-lexical-bridge-v1"
EXPECTED_REASONER_SHA256 = (
    "c374e3b566808cb317ffcd2725653c9073d2e7aebeb75e93ed7ea2a7e2e27044"
)
EXPECTED_WARM_ADAPTER_SHA256 = (
    "abd22528da0d8dc4718c7a89d9c94520540a2f38b7f0b1d9a9e623d0af23cf4d"
)

TRAIN_STATE_ALIASES = (
    "amber", "cedar", "cobalt", "coral", "elm", "flint", "granite",
    "hazel", "indigo", "jade", "lilac", "maple", "ochre", "pearl",
    "quartz", "ruby", "saffron", "teal", "umber", "willow",
)
SHIFT_STATE_ALIASES = (
    "acorn", "birch", "clover", "dahlia", "ember", "fir", "ginger",
    "heather", "iris", "juniper", "kelp", "lotus", "moss", "nectar",
    "opal", "poppy", "reed", "spruce", "thyme", "violet",
)
TRAIN_GENERATOR_ALIASES = (
    "arc", "bend", "drift", "flip", "glide", "pivot", "roll", "turn",
)
SHIFT_GENERATOR_ALIASES = (
    "curl", "hinge", "orbit", "reflect", "rotate", "sweep", "tilt", "twist",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FieldSpan:
    start: int
    end: int
    role: int
    value: int


@dataclass(frozen=True, slots=True)
class AnnotatedRecord:
    text: str
    challenge: bool
    spans: tuple[FieldSpan, ...]
    start: int = 0
    outcome: int = 0
    word: tuple[int, ...] = ()


class RecordBuilder:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.length = 0
        self.spans: list[FieldSpan] = []

    def add(
        self,
        text: str,
        *,
        role: int | None = None,
        value: int = -1,
    ) -> None:
        start = self.length
        self.parts.append(text)
        self.length += len(text)
        if role is not None:
            if not text or role not in range(ROLE_COUNT) or value < 0:
                raise PresentedReasoningError("invalid annotated lexical span")
            self.spans.append(FieldSpan(start, self.length, role, value))

    def finish(
        self,
        *,
        challenge: bool,
        start: int = 0,
        outcome: int = 0,
        word: Iterable[int] = (),
    ) -> AnnotatedRecord:
        return AnnotatedRecord(
            text="".join(self.parts),
            challenge=challenge,
            spans=tuple(self.spans),
            start=start,
            outcome=outcome,
            word=tuple(word),
        )


@dataclass(frozen=True, slots=True)
class LexicalSource:
    ids: torch.Tensor
    valid_mask: torch.Tensor
    token_record: torch.Tensor
    record_mask: torch.Tensor
    challenge_record: torch.Tensor
    token_role: torch.Tensor
    token_value: torch.Tensor
    challenge_start: torch.Tensor
    challenge_outcome: torch.Tensor
    challenge_length: torch.Tensor
    challenge_word: torch.Tensor

    def to(self, device: torch.device | str) -> LexicalSource:
        return LexicalSource(
            **{
                field: getattr(self, field).to(device)
                for field in self.__dataclass_fields__
            }
        )


@dataclass(frozen=True, slots=True)
class LexicalBridgeConfig:
    layer: int
    width: int
    heads: int
    encoder_layers: int
    ff: int


@dataclass(frozen=True, slots=True)
class LexicalLogits:
    kind: torch.Tensor
    role: torch.Tensor


def _alias_assignment(
    count: int,
    pool: tuple[str, ...],
    rng: random.Random,
) -> tuple[str, ...]:
    if count > len(pool):
        raise PresentedReasoningError("alias pool is too small")
    return tuple(rng.sample(pool, count))


def _add_word(
    builder: RecordBuilder,
    word: list[int],
    generator_aliases: tuple[str, ...],
) -> None:
    for index, generator in enumerate(word):
        if index:
            builder.add(" then ")
        builder.add(
            generator_aliases[generator], role=WORD_ROLE, value=generator
        )


def _observation_record(
    template: int,
    generator: int,
    source: int,
    target: int,
    state_aliases: tuple[str, ...],
    generator_aliases: tuple[str, ...],
) -> AnnotatedRecord:
    builder = RecordBuilder()
    if template == 0:
        builder.add("Observation: applying ")
        builder.add(generator_aliases[generator])
        builder.add(" to ")
        builder.add(state_aliases[source])
        builder.add(" gives ")
        builder.add(state_aliases[target])
        builder.add(".")
    else:
        builder.add("When ")
        builder.add(generator_aliases[generator])
        builder.add(" is used on ")
        builder.add(state_aliases[source])
        builder.add(", it becomes ")
        builder.add(state_aliases[target])
        builder.add(".")
    return builder.finish(challenge=False)


def _challenge_record(
    template: int,
    start: int,
    word: list[int],
    outcome: int,
    state_aliases: tuple[str, ...],
    generator_aliases: tuple[str, ...],
) -> AnnotatedRecord:
    builder = RecordBuilder()
    if template == 0:
        builder.add("Challenge: starting at ")
        builder.add(state_aliases[start], role=START_ROLE, value=start)
        builder.add(", follow ")
        _add_word(builder, word, generator_aliases)
        builder.add("; the result is ")
        builder.add(state_aliases[outcome], role=OUTCOME_ROLE, value=outcome)
        builder.add(".")
    elif template == 1:
        builder.add("Challenge: ")
        builder.add(state_aliases[outcome], role=OUTCOME_ROLE, value=outcome)
        builder.add(" is reached from ")
        builder.add(state_aliases[start], role=START_ROLE, value=start)
        builder.add(" by applying ")
        _add_word(builder, word, generator_aliases)
        builder.add(".")
    elif template == 2:
        builder.add("Challenge result ")
        builder.add(state_aliases[outcome], role=OUTCOME_ROLE, value=outcome)
        builder.add(": begin with ")
        builder.add(state_aliases[start], role=START_ROLE, value=start)
        builder.add(", then apply ")
        _add_word(builder, word, generator_aliases)
        builder.add(".")
    elif template == 3:
        builder.add("Using ")
        _add_word(builder, word, generator_aliases)
        builder.add(", one arrives at ")
        builder.add(state_aliases[outcome], role=OUTCOME_ROLE, value=outcome)
        builder.add(" when the initial state was ")
        builder.add(state_aliases[start], role=START_ROLE, value=start)
        builder.add(".")
    else:
        raise PresentedReasoningError("unknown lexical challenge template")
    return builder.finish(
        challenge=True, start=start, outcome=outcome, word=word
    )


def _token_positions_for_span(
    offsets: list[tuple[int, int]],
    span: FieldSpan,
) -> list[int]:
    return [
        index
        for index, (start, end) in enumerate(offsets)
        if end > span.start and start < span.end
    ]


def render_lexical_source(
    batch: PresentedBatch,
    algebra: PresentedAlgebraConfig,
    tokenizer: Tokenizer,
    *,
    seed: int,
    templates: tuple[int, ...],
    shifted_aliases: bool,
    seq_len: int,
) -> LexicalSource:
    """Render source-only natural records and retain source-position labels."""

    if batch.family.device.type != "cpu":
        raise PresentedReasoningError("lexical rendering requires a CPU batch")
    if not templates or any(template not in range(4) for template in templates):
        raise PresentedReasoningError("invalid lexical templates")
    state_pool = SHIFT_STATE_ALIASES if shifted_aliases else TRAIN_STATE_ALIASES
    generator_pool = (
        SHIFT_GENERATOR_ALIASES if shifted_aliases else TRAIN_GENERATOR_ALIASES
    )
    maximum_records = algebra.maximum_observations + algebra.maximum_challenges
    maximum_word = algebra.maximum_word_length
    rows: list[dict[str, Any]] = []
    maximum_tokens = 0
    for row in range(batch.family.shape[0]):
        rng = random.Random(seed * 1_000_003 + row * 8191)
        states = _alias_assignment(algebra.carrier_size, state_pool, rng)
        generators = _alias_assignment(
            algebra.maximum_generators, generator_pool, rng
        )
        records: list[AnnotatedRecord] = []
        for index in range(algebra.maximum_observations):
            if not bool(batch.observation_mask[row, index]):
                continue
            records.append(
                _observation_record(
                    rng.randrange(2),
                    int(batch.observation_generator[row, index]),
                    int(batch.observation_input[row, index]),
                    int(batch.observation_output[row, index]),
                    states,
                    generators,
                )
            )
        for index in range(algebra.maximum_challenges):
            if not bool(batch.challenge_mask[row, index]):
                continue
            word = [
                int(value)
                for value, keep in zip(
                    batch.challenge_word[row, index],
                    batch.challenge_word_mask[row, index],
                    strict=True,
                )
                if bool(keep)
            ]
            records.append(
                _challenge_record(
                    templates[rng.randrange(len(templates))],
                    int(batch.challenge_start[row, index]),
                    word,
                    int(batch.challenge_outcome[row, index]),
                    states,
                    generators,
                )
            )
        rng.shuffle(records)
        if len(records) > maximum_records:
            raise PresentedReasoningError("lexical record count exceeds geometry")

        ids: list[int] = []
        token_record: list[int] = []
        token_role: list[int] = []
        token_value: list[int] = []
        record_targets: list[AnnotatedRecord] = []
        for record_index, record in enumerate(records):
            encoding = tokenizer.encode(record.text + "\n", add_special_tokens=False)
            if not encoding.ids:
                raise PresentedReasoningError("tokenizer emitted an empty record")
            roles = [OTHER_ROLE] * len(encoding.ids)
            values = [-1] * len(encoding.ids)
            for span in record.spans:
                positions = _token_positions_for_span(encoding.offsets, span)
                if not positions:
                    raise PresentedReasoningError("annotated span has no source token")
                position = positions[0]
                if roles[position] != OTHER_ROLE:
                    raise PresentedReasoningError("lexical spans collide on one token")
                roles[position] = span.role
                values[position] = span.value
            ids.extend(encoding.ids)
            token_record.extend([record_index] * len(encoding.ids))
            token_role.extend(roles)
            token_value.extend(values)
            record_targets.append(record)
        if len(ids) > seq_len:
            raise PresentedReasoningError(
                f"lexical source length {len(ids)} exceeds sequence length {seq_len}"
            )
        maximum_tokens = max(maximum_tokens, len(ids))
        rows.append(
            {
                "ids": ids,
                "token_record": token_record,
                "token_role": token_role,
                "token_value": token_value,
                "records": record_targets,
            }
        )

    geometry = (len(rows), maximum_tokens)
    ids_tensor = torch.zeros(geometry, dtype=torch.long)
    valid_mask = torch.zeros(geometry, dtype=torch.bool)
    token_record_tensor = torch.full(geometry, -1, dtype=torch.long)
    token_role_tensor = torch.full(geometry, OTHER_ROLE, dtype=torch.long)
    token_value_tensor = torch.full(geometry, -1, dtype=torch.long)
    record_mask = torch.zeros(len(rows), maximum_records, dtype=torch.bool)
    challenge_record = torch.zeros_like(record_mask)
    challenge_start = torch.zeros_like(record_mask, dtype=torch.long)
    challenge_outcome = torch.zeros_like(record_mask, dtype=torch.long)
    challenge_length = torch.ones_like(record_mask, dtype=torch.long)
    challenge_word = torch.zeros(
        len(rows), maximum_records, maximum_word, dtype=torch.long
    )
    for row_index, row in enumerate(rows):
        length = len(row["ids"])
        ids_tensor[row_index, :length] = torch.tensor(row["ids"])
        valid_mask[row_index, :length] = True
        token_record_tensor[row_index, :length] = torch.tensor(row["token_record"])
        token_role_tensor[row_index, :length] = torch.tensor(row["token_role"])
        token_value_tensor[row_index, :length] = torch.tensor(row["token_value"])
        for record_index, record in enumerate(row["records"]):
            record_mask[row_index, record_index] = True
            challenge_record[row_index, record_index] = record.challenge
            if record.challenge:
                challenge_start[row_index, record_index] = record.start
                challenge_outcome[row_index, record_index] = record.outcome
                challenge_length[row_index, record_index] = len(record.word)
                challenge_word[row_index, record_index, : len(record.word)] = (
                    torch.tensor(record.word)
                )
    return LexicalSource(
        ids=ids_tensor,
        valid_mask=valid_mask,
        token_record=token_record_tensor,
        record_mask=record_mask,
        challenge_record=challenge_record,
        token_role=token_role_tensor,
        token_value=token_value_tensor,
        challenge_start=challenge_start,
        challenge_outcome=challenge_outcome,
        challenge_length=challenge_length,
        challenge_word=challenge_word,
    )


class LexicalChallengeParser(nn.Module):
    """Frozen Smol residuals plus a warm-started lexical record tagger."""

    def __init__(self, model: nn.Module, config: LexicalBridgeConfig):
        super().__init__()
        if model.cfg.n_loop != 1 or not 0 <= config.layer < len(model.blocks):
            raise ValueError("invalid frozen lexical backbone")
        if config.width % config.heads or config.encoder_layers <= 0:
            raise ValueError("invalid lexical adapter geometry")
        self.model = model
        self.config = config
        self.model.requires_grad_(False)
        self.memory_norm = nn.LayerNorm(model.cfg.d_model)
        self.memory_projection = nn.Linear(
            model.cfg.d_model, config.width, bias=False
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.width,
            nhead=config.heads,
            dim_feedforward=config.ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.memory_encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.encoder_layers,
            enable_nested_tensor=False,
        )
        self.kind_head = nn.Linear(config.width, 2)
        self.role_head = nn.Linear(config.width, ROLE_COUNT)

    def adapter_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("model."):
                yield parameter

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            hidden = self.model.tok(ids)
            cos = self.model.cos[: ids.shape[1]].to(hidden.device)
            sin = self.model.sin[: ids.shape[1]].to(hidden.device)
            for block in self.model.blocks[: self.config.layer + 1]:
                hidden, _ = block(hidden, cos, sin)
        return hidden.detach()

    def forward(self, source: LexicalSource) -> LexicalLogits:
        hidden = self.encode(source.ids)
        memory = self.memory_projection(self.memory_norm(hidden))
        memory = self.memory_encoder(
            memory, src_key_padding_mask=~source.valid_mask
        )
        records = source.record_mask.shape[1]
        record_index = source.token_record.clamp_min(0)
        membership = F.one_hot(record_index, records).to(memory.dtype)
        membership = membership * source.valid_mask[..., None]
        summaries = torch.einsum("blr,blh->brh", membership, memory)
        counts = membership.sum(1).clamp_min(1).unsqueeze(-1)
        summaries = summaries / counts
        return LexicalLogits(
            kind=self.kind_head(summaries).float(),
            role=self.role_head(memory).float(),
        )

    def decode(
        self,
        logits: LexicalLogits,
        source: LexicalSource,
        algebra: PresentedAlgebraConfig,
    ) -> tuple[DecodedChallenges, torch.Tensor]:
        challenge_score = logits.kind.softmax(-1)[..., 1].masked_fill(
            ~source.record_mask, -torch.inf
        )
        record_index = challenge_score.topk(
            algebra.maximum_challenges, -1
        ).indices
        selected_record = (
            source.token_record[:, None, :].eq(record_index[..., None])
            & source.valid_mask[:, None, :]
        )

        def copy_value(role: int) -> tuple[torch.Tensor, torch.Tensor]:
            score = logits.role[:, None, :, role].masked_fill(
                ~selected_record, -torch.inf
            )
            position = score.argmax(-1)
            value = source.token_value[:, None, :].expand(
                -1, algebra.maximum_challenges, -1
            ).gather(-1, position[..., None]).squeeze(-1)
            return value.clamp_min(0), value.ge(0)

        start, start_valid = copy_value(START_ROLE)
        outcome, outcome_valid = copy_value(OUTCOME_ROLE)
        predicted_role = logits.role.argmax(-1)
        selected_word = (
            predicted_role[:, None, :].eq(WORD_ROLE) & selected_record
        )
        positions = torch.arange(source.ids.shape[1], device=source.ids.device)
        order_key = torch.where(
            selected_word,
            positions[None, None],
            source.ids.shape[1],
        )
        ordered_position = order_key.argsort(-1)[..., : algebra.maximum_word_length]
        values = source.token_value[:, None, :].expand(
            -1, algebra.maximum_challenges, -1
        ).gather(-1, ordered_position)
        length = selected_word.sum(-1).clamp(1, algebra.maximum_word_length)
        word_position = torch.arange(
            algebra.maximum_word_length, device=source.ids.device
        )
        word_mask = word_position[None, None] < length[..., None]
        word_valid = (values.ge(0) | ~word_mask).all(-1)
        valid = start_valid & outcome_valid & word_valid
        return (
            DecodedChallenges(
                record_index=record_index,
                start=start.clamp_max(algebra.carrier_size - 1),
                outcome=outcome.clamp_max(algebra.carrier_size - 1),
                length=length,
                word=values.clamp(0, algebra.maximum_generators - 1),
                word_mask=word_mask,
            ),
            valid,
        )


def lexical_loss(
    logits: LexicalLogits,
    source: LexicalSource,
) -> tuple[torch.Tensor, dict[str, float]]:
    kind_loss = F.cross_entropy(
        logits.kind[source.record_mask],
        source.challenge_record[source.record_mask].long(),
    )
    token_record = source.token_record.clamp_min(0)
    challenge_token = source.challenge_record.gather(1, token_record)
    challenge_token = challenge_token & source.valid_mask
    role_loss = F.cross_entropy(
        logits.role[challenge_token], source.token_role[challenge_token]
    )
    return kind_loss + role_loss, {
        "kind_loss": float(kind_loss.item()),
        "role_loss": float(role_loss.item()),
    }


def load_warm_adapter(
    parser: LexicalChallengeParser,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = payload.get("compiler")
    state = payload.get("adapter_state")
    if not isinstance(metadata, dict) or not isinstance(state, dict):
        raise ValueError("unsupported lexical warm-start checkpoint")
    expected = {
        "layer": parser.config.layer,
        "width": parser.config.width,
        "heads": parser.config.heads,
        "encoder_layers": parser.config.encoder_layers,
        "ff": parser.config.ff,
        "ordinary_tagger": True,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"warm-start {key} mismatch: {metadata.get(key)!r} != {value!r}"
            )

    def load_prefix(module: nn.Module, prefix: str) -> None:
        selected = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not selected:
            raise ValueError(f"warm-start lacks {prefix}")
        module.load_state_dict(selected, strict=True)

    load_prefix(parser.memory_norm, "memory_norm.")
    load_prefix(parser.memory_projection, "memory_projection.")
    load_prefix(parser.memory_encoder, "memory_encoder.")
    return metadata


def gather_lexical_targets(
    source: LexicalSource,
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


@torch.inference_mode()
def evaluate_cohort(
    parser: LexicalChallengeParser,
    reasoner: LearnedPSPAGate,
    tokenizer: Tokenizer,
    *,
    family: int,
    length: int,
    count: int,
    batch_size: int,
    seed: int,
    renderer_seed: int,
    templates: tuple[int, ...],
    shifted_aliases: bool,
    device: torch.device,
) -> dict[str, Any]:
    totals = {
        "learned": 0,
        "oracle": 0,
        "shuffled": 0,
        "lineage": 0,
        "tuple": 0,
        "table": 0,
        "decoded_valid": 0,
    }
    batch_hashes = []
    processed = 0
    while processed < count:
        current = min(batch_size, count - processed)
        cpu_batch = generate_batch(
            current,
            length,
            reasoner.algebra,
            seed=seed + processed * 1009,
            family=family,
        )
        batch_hashes.append(batch_sha256(cpu_batch))
        cpu_lexical = render_lexical_source(
            cpu_batch,
            reasoner.algebra,
            tokenizer,
            seed=renderer_seed + processed * 1013,
            templates=templates,
            shifted_aliases=shifted_aliases,
            seq_len=parser.model.cfg.seq_len,
        )
        cpu_typed = render_semantic_source(
            cpu_batch,
            reasoner.algebra,
            seed=renderer_seed + processed * 1019,
            templates=(0, 1, 2, 3),
        )
        batch = cpu_batch.to(device)
        lexical = cpu_lexical.to(device)
        typed = cpu_typed.to(device)
        logits = parser(lexical)
        decoded, decoded_valid = parser.decode(
            logits, lexical, reasoner.algebra
        )
        _, row_probabilities = reasoner.row_soft(
            typed.rendered, batch.generator_mask, hard=False
        )
        candidates = binary_completion_candidates(
            row_probabilities, batch.generator_mask
        )
        selected, _, _ = select_with_challenges(
            candidates,
            decoded.start,
            decoded.word,
            decoded.word_mask,
            decoded.outcome,
            decoded_valid,
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
            decoded_valid,
        )

        def answer(tables: torch.Tensor) -> torch.Tensor:
            return execute_word(
                tables,
                batch.query_start,
                batch.query_word,
                batch.query_word_mask,
            ).argmax(-1)

        true_record, true_start, true_outcome, true_length, true_word = (
            gather_lexical_targets(lexical, decoded.record_index)
        )
        positions = torch.arange(
            reasoner.algebra.maximum_word_length, device=device
        )
        true_word_mask = positions[None, None] < true_length[..., None]
        word_exact = (decoded.word.eq(true_word) | ~true_word_mask).all(-1)
        tuple_exact = (
            true_record
            & decoded.start.eq(true_start)
            & decoded.outcome.eq(true_outcome)
            & decoded.length.eq(true_length)
            & word_exact
            & decoded_valid
        ).all(-1)
        table_exact = selected.argmax(-1).eq(batch.true_tables.long()).all((-1, -2))
        totals["learned"] += int(answer(selected).eq(batch.answer).sum().item())
        totals["oracle"] += int(answer(oracle).eq(batch.answer).sum().item())
        totals["shuffled"] += int(answer(shuffled).eq(batch.answer).sum().item())
        totals["lineage"] += int(
            answer(selected.roll(1, 0)).eq(batch.answer).sum().item()
        )
        totals["tuple"] += int(tuple_exact.sum().item())
        totals["table"] += int(table_exact.sum().item())
        totals["decoded_valid"] += int(decoded_valid.all(-1).sum().item())
        processed += current

    return {
        "family": FAMILIES[family],
        "length": length,
        "count": count,
        "seed": seed,
        "renderer_seed": renderer_seed,
        "templates": list(templates),
        "shifted_aliases": shifted_aliases,
        "batch_sha256": hashlib.sha256(
            "\n".join(batch_hashes).encode("ascii")
        ).hexdigest(),
        "learned_accuracy": totals["learned"] / count,
        "oracle_accuracy": totals["oracle"] / count,
        "shuffle_outcome_accuracy": totals["shuffled"] / count,
        "lineage_swap_accuracy": totals["lineage"] / count,
        "challenge_tuple_exact": totals["tuple"] / count,
        "selected_table_exact": totals["table"] / count,
        "decoded_all_valid": totals["decoded_valid"] / count,
    }


def aggregate_evaluations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    metrics = (
        "learned_accuracy",
        "oracle_accuracy",
        "shuffle_outcome_accuracy",
        "lineage_swap_accuracy",
        "challenge_tuple_exact",
        "selected_table_exact",
        "decoded_all_valid",
    )
    for split in ("development", "lexical_shift"):
        selected = [row for row in rows if row["split"] == split]
        total = sum(row["count"] for row in selected)
        summary = {
            metric: sum(row[metric] * row["count"] for row in selected) / total
            for metric in metrics
        }
        summary["count"] = total
        summary["minimum_cohort_learned_accuracy"] = min(
            row["learned_accuracy"] for row in selected
        )
        summary["shuffle_drop_points"] = 100 * (
            summary["learned_accuracy"] - summary["shuffle_outcome_accuracy"]
        )
        summary["lineage_drop_points"] = 100 * (
            summary["learned_accuracy"] - summary["lineage_swap_accuracy"]
        )
        summaries[split] = summary
    return summaries


def assess_gate(summary: dict[str, Any]) -> dict[str, bool]:
    development = summary["development"]
    shift = summary["lexical_shift"]
    return {
        "development_answer_ge_95": development["learned_accuracy"] >= 0.95,
        "development_tuple_ge_95": development["challenge_tuple_exact"] >= 0.95,
        "development_table_ge_95": development["selected_table_exact"] >= 0.95,
        "shift_answer_ge_90": shift["learned_accuracy"] >= 0.90,
        "shift_tuple_ge_90": shift["challenge_tuple_exact"] >= 0.90,
        "shift_table_ge_90": shift["selected_table_exact"] >= 0.90,
        "every_shift_cohort_answer_ge_90": (
            shift["minimum_cohort_learned_accuracy"] >= 0.90
        ),
        "development_shuffle_drop_ge_20": development["shuffle_drop_points"] >= 20,
        "shift_shuffle_drop_ge_20": shift["shuffle_drop_points"] >= 20,
        "development_lineage_drop_ge_20": development["lineage_drop_points"] >= 20,
        "shift_lineage_drop_ge_20": shift["lineage_drop_points"] >= 20,
        "development_oracle_ge_98": development["oracle_accuracy"] >= 0.98,
        "shift_oracle_ge_98": shift["oracle_accuracy"] >= 0.98,
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


def _adapter_state(parser: LexicalChallengeParser) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in parser.state_dict().items()
        if not name.startswith("model.")
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise PresentedReasoningError("CUDA is required unless --allow-cpu is explicit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    reasoner_sha256 = sha256_file(args.reasoner_checkpoint)
    warm_sha256 = sha256_file(args.warm_adapter)
    if reasoner_sha256 != args.expected_reasoner_sha256:
        raise PresentedReasoningError("reasoner checkpoint hash differs")
    if warm_sha256 != args.expected_warm_sha256:
        raise PresentedReasoningError("warm lexical adapter hash differs")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    model, model_config, backbone_receipt = load_frozen_pointer_backbone(
        args.base, device=device
    )
    warm_payload = torch.load(
        args.warm_adapter, map_location="cpu", weights_only=True
    )
    warm_metadata = warm_payload["compiler"]
    parser_config = LexicalBridgeConfig(
        layer=int(warm_metadata["layer"]),
        width=int(warm_metadata["width"]),
        heads=int(warm_metadata["heads"]),
        encoder_layers=int(warm_metadata["encoder_layers"]),
        ff=int(warm_metadata["ff"]),
    )
    del warm_payload
    parser = LexicalChallengeParser(model, parser_config).to(device)
    load_warm_adapter(parser, args.warm_adapter)

    reasoner_payload = torch.load(
        args.reasoner_checkpoint, map_location="cpu", weights_only=True
    )
    algebra = PresentedAlgebraConfig(**reasoner_payload["algebra_config"])
    language = LanguageConfig(**reasoner_payload["language_config"])
    reasoner = LearnedPSPAGate(algebra, language).to(device)
    reasoner.load_state_dict(reasoner_payload["model"])
    reasoner.eval().requires_grad_(False)
    del reasoner_payload

    trainable = list(parser.adapter_parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
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
        source = render_lexical_source(
            cpu_batch,
            algebra,
            tokenizer,
            seed=args.renderer_seed + update,
            templates=train_templates,
            shifted_aliases=False,
            seq_len=model_config.seq_len,
        ).to(device)
        parser.train()
        parser.model.eval()
        optimizer.zero_grad(set_to_none=True)
        autocast_enabled = device.type == "cuda"
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            logits = parser(source)
            loss, loss_metrics = lexical_loss(logits, source)
        if not torch.isfinite(loss):
            raise PresentedReasoningError(f"non-finite loss at update {update}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(grad_norm):
            raise PresentedReasoningError(f"non-finite gradient at update {update}")
        optimizer.step()
        if update == 1 or update % args.log_every == 0 or update == args.updates:
            elapsed = max(time.monotonic() - started, 1e-6)
            row = {
                "update": update,
                "loss": float(loss.item()),
                **loss_metrics,
                "grad_norm": float(grad_norm.item()),
                "examples_per_second": update * args.batch_size / elapsed,
            }
            train_log.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    elapsed = time.monotonic() - started
    parser.eval()
    evaluations = []
    for split, templates, shifted, offset in (
        ("development", train_templates, False, 0),
        ("lexical_shift", (3,), True, 100_000),
    ):
        for family in range(len(FAMILIES)):
            for length in (8, 12):
                row = evaluate_cohort(
                    parser,
                    reasoner,
                    tokenizer,
                    family=family,
                    length=length,
                    count=args.eval_count,
                    batch_size=args.eval_batch_size,
                    seed=args.eval_seed + offset + family * 100 + length,
                    renderer_seed=(
                        args.eval_renderer_seed + offset + family * 100 + length
                    ),
                    templates=templates,
                    shifted_aliases=shifted,
                    device=device,
                )
                row["split"] = split
                evaluations.append(row)
                print(json.dumps({"evaluation": row}, sort_keys=True), flush=True)
    summary = aggregate_evaluations(evaluations)
    gates = assess_gate(summary)

    checkpoint = args.output.with_suffix(".pt")
    if checkpoint.exists():
        raise PresentedReasoningError(f"refusing existing checkpoint: {checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary_checkpoint = checkpoint.with_suffix(".pt.partial")
    torch.save(
        {
            "schema": SCHEMA,
            "seed": args.seed,
            "parser_config": asdict(parser_config),
            "adapter_state": _adapter_state(parser),
            "base_sha256": sha256_file(args.base),
            "tokenizer_sha256": sha256_file(args.tokenizer),
            "warm_adapter_sha256": warm_sha256,
            "reasoner_checkpoint_sha256": reasoner_sha256,
        },
        temporary_checkpoint,
    )
    os.replace(temporary_checkpoint, checkpoint)
    report = {
        "schema": SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "all_gates_pass": all(gates.values()),
        "gates": gates,
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
        "adapter_parameters": sum(parameter.numel() for parameter in trainable),
        "base_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "base_checkpoint_format": backbone_receipt.checkpoint_format,
        "base_import": backbone_receipt.base_import,
        "base_sha256": sha256_file(args.base),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "warm_adapter_sha256": warm_sha256,
        "reasoner_checkpoint_sha256": reasoner_sha256,
        "elapsed_seconds": elapsed,
        "examples_per_second": args.updates * args.batch_size / elapsed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "train_log": train_log,
        "evaluations": evaluations,
        "summary": summary,
        "claim_boundary": (
            "Frozen typed observation compiler plus model-decoded natural-language "
            "challenges into frozen CSDC; not unrestricted natural-language reasoning."
        ),
    }
    _atomic_json(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--warm-adapter", type=Path, required=True)
    parser.add_argument("--reasoner-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-reasoner-sha256", default=EXPECTED_REASONER_SHA256)
    parser.add_argument("--expected-warm-sha256", default=EXPECTED_WARM_ADAPTER_SHA256)
    parser.add_argument("--seed", type=int, default=2026080507)
    parser.add_argument("--data-seed", type=int, default=202608051000)
    parser.add_argument("--renderer-seed", type=int, default=202608052000)
    parser.add_argument("--eval-seed", type=int, default=202608053000)
    parser.add_argument("--eval-renderer-seed", type=int, default=202608054000)
    parser.add_argument("--train-templates", default="0,1,2")
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-count", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=64)
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
                "all_gates_pass": completed["all_gates_pass"],
                "summary": completed["summary"],
            },
            sort_keys=True,
        )
    )
