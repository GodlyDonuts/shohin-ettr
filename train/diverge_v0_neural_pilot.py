#!/usr/bin/env python3
"""Minimal learned source compiler/refiner pilot for DIVERGE-v0.

This is a component-island gate, not the frozen A--G promotion result. Candidate
record boundaries and two option spans are syntactic scaffolds. The network must
decide which records are real fault lines, compile each option's ordered program
and support prior, and bind delayed evidence to one source-sealed option key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from version_space_accounting import canonical_json_bytes

from diverge_v0 import (
    ANSWER,
    FaultLine,
    Guard,
    GuardedPatch,
    Literal,
    PacketCaps,
    Query,
    SupportFactor,
    TypedCell,
    TypedState,
    TypedTransaction,
    append_verified_nogood,
    account_packet,
    build_packet,
    enumerate_assignments,
    execute_packet,
    named_commitment,
    query_execution,
    read_query,
)
from diverge_v0_reference import verify_nogood
from frozen_pointer_backbone import load_frozen_pointer_backbone


SCHEMA = "shohin-diverge-v0-neural-component-pilot-v1"
WORD_BUCKETS = 4096
CHARACTERS = "abcdefghijklmnopqrstuvwxyz0123456789-_"
CHAR_TO_ID = {character: index + 2 for index, character in enumerate(CHARACTERS)}
WORD_PATTERN = re.compile(r"[a-z]+|[0-9]+|[^\w\s]", re.IGNORECASE)


PROGRAMS: dict[int, tuple[TypedTransaction, ...]] = {
    0: (
        TypedTransaction("ADD_VALUE", (0, 3)),
        TypedTransaction("SWAP_VALUE", (0, 1)),
    ),
    1: (
        TypedTransaction("SWAP_VALUE", (0, 1)),
        TypedTransaction("ADD_VALUE", (0, 3)),
    ),
    2: (TypedTransaction("SWAP_VALUE", (2, 3)),),
    3: (TypedTransaction("SWAP_VALUE", (3, 4)),),
}

ACTION_TEXT = {
    "add": "increase slot zero by three",
    "swap01": "exchange slot zero with slot one",
    "swap23": "exchange slot two with slot three",
    "swap34": "exchange slot three with slot four",
}

TRAIN_ALIAS_STEMS = (
    "amber", "cedar", "cobalt", "coral", "flint", "hazel", "indigo",
    "jade", "lilac", "maple", "ochre", "pearl", "quartz", "ruby",
)
SHIFT_ALIAS_STEMS = (
    "acorn", "birch", "clover", "dahlia", "ember", "ginger", "heather",
    "iris", "juniper", "lotus", "nectar", "poppy", "spruce", "violet",
)


@dataclass(frozen=True)
class OptionExample:
    alias: str
    text: str
    program: int
    prior_class: int  # 0 = favored mass 3, 1 = reserve mass 1


@dataclass(frozen=True)
class RecordExample:
    record_id: str
    text: str
    is_fault_line: bool
    options: tuple[OptionExample, OptionExample]
    gold_option: int


@dataclass(frozen=True)
class PilotEpisode:
    episode_id: str
    split: str
    ontology: str
    renderer: int
    source_text: str
    records: tuple[RecordExample, ...]
    primary_record_id: str
    evidence_alias: str
    evidence_text: str


@dataclass(frozen=True)
class CompilerPrediction:
    selected: tuple[bool, ...]
    programs: tuple[tuple[int, int], ...]
    priors: tuple[tuple[int, int], ...]
    evidence_record: int
    evidence_option: int


def _digest(domain: str, payload: object) -> str:
    body = canonical_json_bytes(payload)
    digest = hashlib.sha256()
    for part in (domain.encode("ascii"), body):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _alias(rng: random.Random, split: str, used: set[str]) -> str:
    stems = TRAIN_ALIAS_STEMS if split == "train" else SHIFT_ALIAS_STEMS
    while True:
        value = f"{rng.choice(stems)}-{rng.randrange(1000, 9999)}"
        if value not in used:
            used.add(value)
            return value


def _program_actions(program: int) -> tuple[str, ...]:
    return {
        0: (ACTION_TEXT["add"], ACTION_TEXT["swap01"]),
        1: (ACTION_TEXT["swap01"], ACTION_TEXT["add"]),
        2: (ACTION_TEXT["swap23"],),
        3: (ACTION_TEXT["swap34"],),
    }[program]


def _render_option(alias: str, program: int, prior: int, renderer: int) -> str:
    actions = _program_actions(program)
    status = "favored" if prior == 0 else "reserve"
    if len(actions) == 1:
        if renderer == 0:
            return f"{status} key {alias} means perform {actions[0]}"
        if renderer == 1:
            return f"under {alias} perform {actions[0]} with {status} status"
        if renderer == 2:
            return f"perform {actions[0]} for {status} key {alias}"
        return f"the {status} rule for key {alias} is perform {actions[0]}"
    first, second = actions
    if renderer == 0:
        return f"{status} key {alias} means first {first} then {second}"
    if renderer == 1:
        return f"under {alias} perform {first} before {second} with {status} status"
    if renderer == 2:
        return f"perform {first} before {second} for {status} key {alias}"
    return f"the {status} rule for key {alias} is first {first} then {second}"


def _render_record(
    ontology: str,
    option_a: OptionExample,
    option_b: OptionExample,
    *,
    is_fault_line: bool,
    renderer: int,
) -> str:
    if is_fault_line:
        wrappers = (
            "candidate alternatives in {ontology}: {a}; versus {b}.",
            "in {ontology}, candidate alternatives are {b}; compared with {a}.",
            "candidate alternatives for {ontology} list {a}; while {b}.",
            "for {ontology}, the candidate alternatives are {b}; and {a}.",
        )
    else:
        wrappers = (
            "background example in {ontology}, not a candidate: {a}; also {b}.",
            "descriptive archive only, not candidate alternatives: {b}; then {a}.",
            "ignore this background record for {ontology}: {a}; besides {b}.",
            "this {ontology} archive is descriptive only: {b}; alongside {a}.",
        )
    return wrappers[renderer].format(ontology=ontology, a=option_a.text, b=option_b.text)


def generate_episode(
    *,
    seed: int,
    split: str,
    width: int,
    renderer: int,
    ontology: str,
) -> PilotEpisode:
    if split not in {"train", "development", "confirmation"}:
        raise ValueError("invalid pilot split")
    if width < 1 or width > 6:
        raise ValueError("invalid pilot width")
    rng = random.Random(seed)
    episode_id = _digest(
        "diverge-neural-episode",
        {"seed": seed, "split": split, "width": width, "renderer": renderer, "ontology": ontology},
    )[:20]
    used: set[str] = set()
    records = []
    primary_record_id = ""
    evidence_alias = ""
    for index in range(width):
        programs = [0, 1] if index == 0 else [2, 3]
        rng.shuffle(programs)
        if index == 0:
            priors = [0 if program == 0 else 1 for program in programs]
        else:
            priors = [0, 1]
            rng.shuffle(priors)
        aliases = [_alias(rng, split, used), _alias(rng, split, used)]
        options = tuple(
            OptionExample(
                aliases[option],
                _render_option(aliases[option], programs[option], priors[option], renderer),
                programs[option],
                priors[option],
            )
            for option in range(2)
        )
        record_id = _digest("diverge-neural-record", {"episode": episode_id, "index": index})[:16]
        gold_option = programs.index(1) if index == 0 else rng.randrange(2)
        record = RecordExample(
            record_id,
            _render_record(ontology, options[0], options[1], is_fault_line=True, renderer=renderer),
            True,
            options,
            gold_option,
        )
        records.append(record)
        if index == 0:
            primary_record_id = record_id
            evidence_alias = options[gold_option].alias
    for distractor in range(rng.randrange(1, 3)):
        programs = [rng.randrange(4), rng.randrange(4)]
        priors = [rng.randrange(2), rng.randrange(2)]
        aliases = [_alias(rng, split, used), _alias(rng, split, used)]
        options = tuple(
            OptionExample(
                aliases[option],
                _render_option(aliases[option], programs[option], priors[option], renderer),
                programs[option],
                priors[option],
            )
            for option in range(2)
        )
        record_id = _digest(
            "diverge-neural-distractor", {"episode": episode_id, "index": distractor}
        )[:16]
        records.append(
            RecordExample(
                record_id,
                _render_record(ontology, options[0], options[1], is_fault_line=False, renderer=renderer),
                False,
                options,
                0,
            )
        )
    rng.shuffle(records)
    source_text = "\n".join(record.text for record in records)
    evidence_text = f"delayed diagnostic confirms active key {evidence_alias}."
    return PilotEpisode(
        episode_id,
        split,
        ontology,
        renderer,
        source_text,
        tuple(records),
        primary_record_id,
        evidence_alias,
        evidence_text,
    )


def _word_id(token: str) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return 2 + int.from_bytes(digest[:4], "big") % WORD_BUCKETS


def word_ids(text: str, aliases: Iterable[str] = ()) -> list[int]:
    normalized = text.lower()
    for alias in sorted(aliases, key=len, reverse=True):
        normalized = normalized.replace(alias.lower(), " alias ")
    tokens = WORD_PATTERN.findall(normalized)
    return [_word_id(token) for token in tokens] or [1]


def char_ids(text: str) -> list[int]:
    return [CHAR_TO_ID.get(character, 1) for character in text.lower()][:32] or [1]


def _pad(rows: list[list[int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(row) for row in rows], dtype=torch.long, device=device)
    width = int(lengths.max().item())
    tensor = torch.zeros(len(rows), width, dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        tensor[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
    return tensor, lengths


class SequenceEncoder(nn.Module):
    def __init__(self, vocabulary: int, embedding: int, hidden: int):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, embedding, padding_idx=0)
        self.gru = nn.GRU(
            embedding,
            hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(2 * hidden)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return self.norm(torch.cat((hidden[-2], hidden[-1]), dim=-1))


class DivergePilotCompiler(nn.Module):
    def __init__(
        self,
        width: int = 96,
        char_width: int = 48,
        selection_policy: str = "adaptive-gap",
    ):
        super().__init__()
        if selection_policy not in {"hard-threshold", "adaptive-gap"}:
            raise ValueError("unknown fault-line selection policy")
        self.selection_policy = selection_policy
        self.text = SequenceEncoder(WORD_BUCKETS + 2, width // 2, width // 2)
        self.alias = SequenceEncoder(len(CHAR_TO_ID) + 2, char_width // 2, char_width // 2)
        local_width = width // 4
        self.kind_convolutions = nn.ModuleList(
            nn.Conv1d(width // 2, local_width, kernel_size=kernel)
            for kernel in (1, 2, 3)
        )
        self.kind_head = nn.Linear(3 * local_width, 1)
        self.program = nn.Linear(width, len(PROGRAMS))
        self.prior = nn.Linear(width, 2)
        self.alias_projection = nn.Linear(char_width, char_width, bias=False)

    def encode_texts(self, rows: list[list[int]], device: torch.device) -> torch.Tensor:
        ids, lengths = _pad(rows, device)
        return self.text(ids, lengths)

    def tokenize(self, text: str, aliases: Iterable[str] = ()) -> list[int]:
        return word_ids(text, aliases)

    def adapter_parameters(self) -> Iterable[nn.Parameter]:
        return self.parameters()

    def adapter_state(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu() for name, value in self.state_dict().items()
        }

    def encode_aliases(self, rows: list[list[int]], device: torch.device) -> torch.Tensor:
        ids, lengths = _pad(rows, device)
        encoded = self.alias_projection(self.alias(ids, lengths))
        return F.normalize(encoded, dim=-1)

    def classify_records(
        self,
        rows: list[list[int]],
        device: torch.device,
    ) -> torch.Tensor:
        """Pool local lexical evidence without making record position semantic."""

        ids, lengths = _pad(rows, device)
        embedded = self.text.embedding(ids).transpose(1, 2)
        pooled = []
        for convolution in self.kind_convolutions:
            kernel = convolution.kernel_size[0]
            evidence = F.gelu(convolution(embedded))
            valid_lengths = (lengths - kernel + 1).clamp_min(1)
            positions = torch.arange(evidence.shape[-1], device=device)
            valid = positions[None, :] < valid_lengths[:, None]
            evidence = evidence.masked_fill(~valid[:, None, :], -torch.inf)
            pooled.append(evidence.max(-1).values)
        return self.kind_head(torch.cat(pooled, dim=-1)).squeeze(-1)


class SmolDivergePilotCompiler(nn.Module):
    """Frozen Smol residuals plus the same bounded DIVERGE compiler heads."""

    def __init__(
        self,
        backbone: nn.Module,
        tokenizer: Tokenizer,
        *,
        layer: int,
        width: int,
        char_width: int,
        selection_policy: str,
    ):
        super().__init__()
        if selection_policy not in {"hard-threshold", "adaptive-gap"}:
            raise ValueError("unknown fault-line selection policy")
        if not 0 <= layer < len(backbone.blocks):
            raise ValueError("Smol compiler layer is outside the backbone")
        if width % 4:
            raise ValueError("Smol compiler width must be divisible by four")
        self.backbone = backbone.requires_grad_(False)
        self.tokenizer = tokenizer
        self.layer = layer
        self.selection_policy = selection_policy
        self.memory_norm = nn.LayerNorm(backbone.cfg.d_model)
        self.memory_projection = nn.Linear(backbone.cfg.d_model, width, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=4 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.memory_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
            enable_nested_tensor=False,
        )
        local_width = width // 4
        self.kind_convolutions = nn.ModuleList(
            nn.Conv1d(width, local_width, kernel_size=kernel)
            for kernel in (1, 2, 3)
        )
        self.kind_head = nn.Linear(3 * local_width, 1)
        self.program = nn.Linear(width, len(PROGRAMS))
        self.prior = nn.Linear(width, 2)
        self.alias = SequenceEncoder(
            len(CHAR_TO_ID) + 2,
            char_width // 2,
            char_width // 2,
        )
        self.alias_projection = nn.Linear(char_width, char_width, bias=False)

    def tokenize(self, text: str, aliases: Iterable[str] = ()) -> list[int]:
        normalized = text.lower()
        for alias in sorted(aliases, key=len, reverse=True):
            normalized = normalized.replace(alias.lower(), " alias ")
        ids = self.tokenizer.encode(normalized, add_special_tokens=False).ids
        if not ids or len(ids) > self.backbone.cfg.seq_len:
            raise ValueError("Smol compiler source length is invalid")
        return ids

    def _memory(
        self,
        rows: list[list[int]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids, lengths = _pad(rows, device)
        self.backbone.eval()
        with torch.no_grad():
            hidden = self.backbone.tok(ids)
            cosine = self.backbone.cos[: ids.shape[1]].to(hidden.device)
            sine = self.backbone.sin[: ids.shape[1]].to(hidden.device)
            for block in self.backbone.blocks[: self.layer + 1]:
                hidden, _ = block(hidden, cosine, sine)
        hidden = hidden.detach().to(self.memory_projection.weight.dtype)
        memory = self.memory_projection(self.memory_norm(hidden))
        positions = torch.arange(ids.shape[1], device=device)
        valid = positions[None, :] < lengths[:, None]
        memory = self.memory_encoder(memory, src_key_padding_mask=~valid)
        return memory, lengths

    def encode_texts(self, rows: list[list[int]], device: torch.device) -> torch.Tensor:
        memory, lengths = self._memory(rows, device)
        positions = torch.arange(memory.shape[1], device=device)
        valid = positions[None, :] < lengths[:, None]
        return (memory * valid[..., None]).sum(1) / lengths[:, None].to(memory.dtype)

    def classify_records(
        self,
        rows: list[list[int]],
        device: torch.device,
    ) -> torch.Tensor:
        memory, lengths = self._memory(rows, device)
        transposed = memory.transpose(1, 2)
        pooled = []
        for convolution in self.kind_convolutions:
            kernel = convolution.kernel_size[0]
            evidence = F.gelu(convolution(transposed))
            valid_lengths = (lengths - kernel + 1).clamp_min(1)
            positions = torch.arange(evidence.shape[-1], device=device)
            valid = positions[None, :] < valid_lengths[:, None]
            evidence = evidence.masked_fill(~valid[:, None, :], -torch.inf)
            pooled.append(evidence.max(-1).values)
        return self.kind_head(torch.cat(pooled, dim=-1)).squeeze(-1)

    def encode_aliases(self, rows: list[list[int]], device: torch.device) -> torch.Tensor:
        ids, lengths = _pad(rows, device)
        encoded = self.alias_projection(self.alias(ids, lengths))
        return F.normalize(encoded, dim=-1)

    def adapter_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if not name.startswith("backbone."):
                yield parameter

    def adapter_state(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if not name.startswith("backbone.")
        }


def _training_batch(
    episodes: list[PilotEpisode],
    model: DivergePilotCompiler,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    records = [record for episode in episodes for record in episode.records]
    record_rows = []
    kind_values = []
    for episode in episodes:
        for record in episode.records:
            aliases = tuple(option.alias for option in record.options)
            record_rows.append(model.tokenize(record.text, aliases))
            kind_values.append(float(record.is_fault_line))
            counterfactual = _render_record(
                episode.ontology,
                record.options[0],
                record.options[1],
                is_fault_line=not record.is_fault_line,
                renderer=episode.renderer,
            )
            record_rows.append(model.tokenize(counterfactual, aliases))
            kind_values.append(float(not record.is_fault_line))
    kind_logits = model.classify_records(record_rows, device)
    kind_target = torch.tensor(kind_values, device=device)
    kind_loss = F.binary_cross_entropy_with_logits(
        kind_logits, kind_target
    )

    fault_options = [
        option
        for record in records
        if record.is_fault_line
        for option in record.options
    ]
    option_hidden = model.encode_texts(
        [model.tokenize(option.text, (option.alias,)) for option in fault_options],
        device,
    )
    program_target = torch.tensor([option.program for option in fault_options], device=device)
    prior_target = torch.tensor([option.prior_class for option in fault_options], device=device)
    program_loss = F.cross_entropy(model.program(option_hidden), program_target)
    prior_loss = F.cross_entropy(model.prior(option_hidden), prior_target)

    alias_losses = []
    alias_correct = 0
    for episode in episodes:
        source_aliases = [option.alias for record in episode.records for option in record.options]
        keys = model.encode_aliases([char_ids(value) for value in source_aliases], device)
        evidence = model.encode_aliases([char_ids(episode.evidence_alias)], device)
        logits = 16.0 * evidence @ keys.T
        target = torch.tensor([source_aliases.index(episode.evidence_alias)], device=device)
        alias_losses.append(F.cross_entropy(logits, target))
        alias_correct += int(logits.argmax(-1).item() == target.item())
    alias_loss = torch.stack(alias_losses).mean()
    loss = kind_loss + program_loss + prior_loss + alias_loss
    with torch.no_grad():
        metrics = {
            "loss": float(loss.item()),
            "kind_loss": float(kind_loss.item()),
            "program_loss": float(program_loss.item()),
            "prior_loss": float(prior_loss.item()),
            "alias_loss": float(alias_loss.item()),
            "kind_accuracy": float(
                (kind_logits.gt(0) == kind_target.bool())
                .float()
                .mean()
                .item()
            ),
            "program_accuracy": float(
                (model.program(option_hidden).argmax(-1) == program_target)
                .float()
                .mean()
                .item()
            ),
            "prior_accuracy": float(
                (model.prior(option_hidden).argmax(-1) == prior_target)
                .float()
                .mean()
                .item()
            ),
            "alias_accuracy": alias_correct / len(episodes),
        }
    return loss, metrics


def predict_episode(
    model: DivergePilotCompiler,
    episode: PilotEpisode,
    device: torch.device,
) -> CompilerPrediction:
    model.eval()
    with torch.no_grad():
        kind_score = model.classify_records(
            [
                model.tokenize(record.text, (option.alias for option in record.options))
                for record in episode.records
            ],
            device,
        )
        if model.selection_policy == "hard-threshold":
            selected = tuple(bool(value) for value in kind_score.gt(0).tolist())
        else:
            order = kind_score.argsort(descending=True)
            if len(order) == 1:
                selected_count = 1
            else:
                maximum = min(6, len(order) - 1)
                gaps = kind_score[order[:maximum]] - kind_score[order[1 : maximum + 1]]
                selected_count = 1 + int(gaps.argmax().item())
            selected_indices = set(order[:selected_count].tolist())
            selected = tuple(index in selected_indices for index in range(len(episode.records)))
        option_rows = [
            model.tokenize(option.text, (option.alias,))
            for record in episode.records
            for option in record.options
        ]
        option_hidden = model.encode_texts(option_rows, device)
        program_values = model.program(option_hidden).argmax(-1).tolist()
        prior_values = model.prior(option_hidden).argmax(-1).tolist()
        programs = [
            tuple(program_values[index : index + 2])
            for index in range(0, len(program_values), 2)
        ]
        priors = [
            tuple(prior_values[index : index + 2])
            for index in range(0, len(prior_values), 2)
        ]
        aliases = [option.alias for record in episode.records for option in record.options]
        keys = model.encode_aliases([char_ids(value) for value in aliases], device)
        evidence = model.encode_aliases([char_ids(episode.evidence_alias)], device)
        selected_alias = int((evidence @ keys.T).argmax(-1).item())
    return CompilerPrediction(
        selected,
        tuple(programs),
        tuple(priors),
        selected_alias // 2,
        selected_alias % 2,
    )


def _build_predicted_packet(
    episode: PilotEpisode,
    prediction: CompilerPrediction,
):
    selected_indices = [index for index, keep in enumerate(prediction.selected) if keep]
    if not selected_indices:
        return None, {}, {}
    variables = []
    supports = []
    patches = []
    provenance_to_record: dict[str, RecordExample] = {}
    old_id_by_record: dict[str, int] = {}
    for variable_id, record_index in enumerate(selected_indices):
        record = episode.records[record_index]
        provenance = named_commitment("diverge-neural-variable", record.record_id)
        provenance_to_record[provenance] = record
        old_id_by_record[record.record_id] = variable_id
        variables.append(
            FaultLine(
                variable_id,
                tuple(
                    named_commitment("diverge-neural-option", option.alias)
                    for option in record.options
                ),
                provenance,
            )
        )
        supports.append(
            SupportFactor(
                (variable_id,),
                tuple(
                    ((option,), 3 if prediction.priors[record_index][option] == 0 else 1)
                    for option in range(2)
                ),
                named_commitment("diverge-neural-support", record.record_id),
            )
        )
        for option in range(2):
            program = prediction.programs[record_index][option]
            for transaction in PROGRAMS[program]:
                patch_index = len(patches)
                patches.append(
                    GuardedPatch(
                        patch_index,
                        Guard((Literal(variable_id, option),)),
                        transaction,
                        named_commitment(
                            "diverge-neural-patch",
                            f"{record.record_id}:{option}:{patch_index}",
                        ),
                    )
                )
    packet = build_packet(
        source_commitment=_digest("diverge-neural-source", episode.source_text),
        shared_state=TypedState(
            tuple(
                TypedCell(slot, 0, value)
                for slot, value in enumerate((1, 10, 20, 30, 40))
            )
        ),
        variables=variables,
        support_factors=supports,
        patches=patches,
        caps=PacketCaps(max_patches=32),
    )
    canonical_by_record = {
        provenance_to_record[variable.provenance].record_id: variable.variable_id
        for variable in packet.variables
    }
    return packet, canonical_by_record, old_id_by_record


def _top_world_answer(packet, query: Query) -> int | None:
    receipt = execute_packet(packet)
    if receipt.overflow or not receipt.worlds:
        return None
    world = min(receipt.worlds, key=lambda item: (-item.mass, item.assignment))
    if world.state is None:
        return None
    return read_query(world.state, query)


def score_episode(
    model: DivergePilotCompiler,
    episode: PilotEpisode,
    device: torch.device,
    *,
    prediction: CompilerPrediction | None = None,
) -> dict[str, int]:
    if prediction is None:
        prediction = predict_episode(model, episode, device)
    true_fault = {record.record_id for record in episode.records if record.is_fault_line}
    selected_fault = {
        record.record_id
        for record, selected in zip(episode.records, prediction.selected, strict=True)
        if selected
    }
    all_gold_selected = true_fault.issubset(selected_fault)
    kind_exact = selected_fault == true_fault
    program_correct = 0
    program_total = 0
    prior_correct = 0
    for index, record in enumerate(episode.records):
        if not record.is_fault_line:
            continue
        for option in range(2):
            program_total += 1
            program_correct += int(prediction.programs[index][option] == record.options[option].program)
            prior_correct += int(prediction.priors[index][option] == record.options[option].prior_class)
    evidence_correct = (
        episode.records[prediction.evidence_record].record_id == episode.primary_record_id
        and episode.records[prediction.evidence_record].options[prediction.evidence_option].alias
        == episode.evidence_alias
    )
    packet, canonical, _ = _build_predicted_packet(episode, prediction)
    result = {
        "episodes": 1,
        "fault_records": len(true_fault),
        "fault_records_selected": len(true_fault & selected_fault),
        "distractor_records": len(episode.records) - len(true_fault),
        "distractor_records_selected": len(selected_fault - true_fault),
        "gold_support_recalled": int(all_gold_selected),
        "record_set_exact": int(kind_exact),
        "program_fields": program_total,
        "program_fields_correct": program_correct,
        "prior_fields_correct": prior_correct,
        "evidence_binding_exact": int(evidence_correct),
        "packet_exact": int(kind_exact and program_correct == program_total and prior_correct == program_total),
        "A_single": 0,
        "B_full_particles": 0,
        "C_independent": 0,
        "D_recurrent_single": 0,
        "E_soft": 0,
        "F_no_conflict": 0,
        "G_diverge": 0,
        "G_joint_packet_answer": 0,
    }
    if packet is None or packet.overflow or episode.primary_record_id not in canonical:
        return result
    query = Query("READ_VALUE", (0,))
    target = 13
    top_answer = _top_world_answer(packet, query)
    for arm in ("A_single", "B_full_particles", "C_independent", "D_recurrent_single"):
        result[arm] = int(top_answer == target)
    initial = query_execution(execute_packet(packet), query)
    result["F_no_conflict"] = int(initial.disposition == ANSWER and initial.answer == target)
    if initial.total_mass:
        weighted = sum(answer * mass for answer, mass in initial.marginals) / initial.total_mass
        result["E_soft"] = int(round(weighted) == target)

    predicted_record = episode.records[prediction.evidence_record]
    if predicted_record.record_id not in canonical:
        return result
    variable = canonical[predicted_record.record_id]
    confirmed = prediction.evidence_option
    primary_record = next(
        record for record in episode.records if record.record_id == episode.primary_record_id
    )
    primary_variable = canonical[episode.primary_record_id]
    valid = tuple(
        assignment
        for assignment in enumerate_assignments(packet)
        if assignment[primary_variable] == primary_record.gold_option
    )
    verification = verify_nogood(
        packet,
        guard=Guard((Literal(variable, 1 - confirmed),)),
        evidence_commitment=_digest("diverge-neural-evidence", episode.evidence_text),
        valid_assignments=valid,
    )
    if not verification.accepted or verification.nogood is None:
        return result
    refined = append_verified_nogood(packet, verification.nogood)
    decision = query_execution(execute_packet(refined), query)
    result["G_diverge"] = int(decision.disposition == ANSWER and decision.answer == target)
    result["G_joint_packet_answer"] = int(result["packet_exact"] and result["G_diverge"])
    return result


def evaluate(
    model: DivergePilotCompiler,
    *,
    split: str,
    count: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    configuration = {
        "development": (5, 2, "parcel-relation"),
        "confirmation": (6, 3, "signal-routing"),
    }[split]
    width, renderer, ontology = configuration
    totals: dict[str, int] = {}
    packets = []
    for index in range(count):
        episode = generate_episode(
            seed=seed + index,
            split=split,
            width=width,
            renderer=renderer,
            ontology=ontology,
        )
        row = score_episode(model, episode, device)
        for key, value in row.items():
            totals[key] = totals.get(key, 0) + value
        prediction = predict_episode(model, episode, device)
        packet, _, _ = _build_predicted_packet(episode, prediction)
        if packet is not None and not packet.overflow:
            packets.append(account_packet(packet, execute_packet(packet)))
    episodes = totals["episodes"]
    count_fields = {
        "episodes",
        "program_fields",
        "program_fields_correct",
        "prior_fields_correct",
        "fault_records",
        "fault_records_selected",
        "distractor_records",
        "distractor_records_selected",
    }
    rates = {
        key: value / episodes
        for key, value in totals.items()
        if key not in count_fields
    }
    rates["program_accuracy"] = totals["program_fields_correct"] / max(1, totals["program_fields"])
    rates["prior_accuracy"] = totals["prior_fields_correct"] / max(1, totals["program_fields"])
    rates["fault_line_recall"] = totals["fault_records_selected"] / max(
        1, totals["fault_records"]
    )
    rates["distractor_false_positive_rate"] = totals[
        "distractor_records_selected"
    ] / max(1, totals["distractor_records"])
    return {
        "split": split,
        "count": count,
        "width": width,
        "renderer": renderer,
        "ontology": ontology,
        "rates": rates,
        "mean_packet_bytes": (
            sum(item.packet_bytes for item in packets) / len(packets) if packets else 0
        ),
        "mean_materialized_particle_bytes": (
            sum(item.materialized_world_bytes for item in packets) / len(packets)
            if packets
            else 0
        ),
    }


def train(args: argparse.Namespace) -> tuple[nn.Module, dict[str, object]]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    backbone_summary: dict[str, object]
    if args.base is None:
        model = DivergePilotCompiler(
            args.width,
            args.char_width,
            selection_policy=args.selection_policy,
        ).to(device)
        backbone_summary = {"kind": "tiny_hash_gru"}
    else:
        if args.tokenizer is None:
            raise ValueError("--tokenizer is required with --base")
        backbone, _, receipt = load_frozen_pointer_backbone(args.base, device=device)
        model = SmolDivergePilotCompiler(
            backbone,
            Tokenizer.from_file(str(args.tokenizer)),
            layer=args.layer,
            width=args.width,
            char_width=args.char_width,
            selection_policy=args.selection_policy,
        ).to(device)
        backbone_summary = {
            "kind": "frozen_smollm2",
            "layer": args.layer,
            "base_sha256": _file_sha256(args.base),
            "tokenizer_sha256": _file_sha256(args.tokenizer),
            "base_import": receipt.base_import,
            "checkpoint_format": receipt.checkpoint_format,
        }
    trainable = list(model.adapter_parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    started = time.perf_counter()
    log = []
    for update in range(1, args.updates + 1):
        model.train()
        episodes = [
            generate_episode(
                seed=args.data_seed + update * args.batch_size + index,
                split="train",
                width=1 + ((update + index) % 4),
                renderer=(update + index) % 2,
                ontology="register-workshop",
            )
            for index in range(args.batch_size)
        ]
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = _training_batch(episodes, model, device)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite pilot loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if update == 1 or update % args.log_every == 0 or update == args.updates:
            row = {"update": update, **metrics, "grad_norm": float(grad_norm.item())}
            log.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    elapsed = time.perf_counter() - started
    summary = {
        "updates": args.updates,
        "batch_size": args.batch_size,
        "charged_episodes": args.updates * args.batch_size,
        "elapsed_seconds": elapsed,
        "episodes_per_second": args.updates * args.batch_size / elapsed,
        "parameters": sum(parameter.numel() for parameter in trainable),
        "backbone": backbone_summary,
        "train_log": log,
    }
    return model, summary


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--seed", type=int, default=2026080517)
    parser.add_argument("--data-seed", type=int, default=202608052000)
    parser.add_argument("--eval-seed", type=int, default=202608053000)
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-count", type=int, default=256)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--char-width", type=int, default=48)
    parser.add_argument(
        "--selection-policy",
        choices=("hard-threshold", "adaptive-gap"),
        default="adaptive-gap",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, training = train(args)
    evaluations = [
        evaluate(
            model,
            split=split,
            count=args.eval_count,
            seed=args.eval_seed + offset,
            device=torch.device(args.device),
        )
        for split, offset in (("development", 0), ("confirmation", 100_000))
    ]
    report = {
        "schema": SCHEMA,
        "status": "component_pilot_only",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "training": training,
        "evaluations": evaluations,
        "model_state_sha256": _digest(
            "diverge-neural-model-state",
            {
                name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
                for name, value in sorted(model.adapter_state().items())
            },
        ),
        "claim_boundary": (
            "Learned candidate-record detection, finite program/prior compilation, "
            "and delayed exact-alias binding on a syntactically scaffolded board. "
            "Not the frozen A-G promotion, unrestricted language compilation, "
            "public reasoning, or a DIVERGE architecture win."
        ),
    }
    checkpoint = args.output.with_suffix(".pt")
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    temporary = checkpoint.with_suffix(".pt.partial")
    torch.save(
        {
            "schema": SCHEMA,
            "arguments": report["arguments"],
            "state_dict": {
                name: value.detach().cpu() for name, value in model.adapter_state().items()
            },
            "model_state_sha256": report["model_state_sha256"],
        },
        temporary,
    )
    os.replace(temporary, checkpoint)
    report["checkpoint"] = str(checkpoint)
    report["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "evaluations": evaluations}, sort_keys=True))


if __name__ == "__main__":
    main()
