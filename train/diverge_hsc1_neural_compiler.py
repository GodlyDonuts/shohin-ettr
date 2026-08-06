#!/usr/bin/env python3
"""Train the frozen DIVERGE-HSC1 hierarchical structured source compiler."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import random
import time
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from diverge_hsc1_structured_compiler import (
    HierarchicalScores,
    OptionRoleScores,
    RecordStructuredScores,
    cut_viterbi,
    decode_hierarchical,
    exact,
    gold_option_path,
    option_markers,
    semantic_templates,
    shuffle_cut_channels,
    shuffle_semantic_roles,
)
from diverge_sc1_neural_compiler import (
    RawSourceCompiler,
    SourceEncoding,
    encode_source,
    sha256_file,
)
from diverge_sc1_source_compiler import (
    ALIAS_BEGIN,
    ALIAS_INSIDE,
    OTHER,
    ROLE_COUNT,
    RawSourceEpisode,
    generate_episode,
)
from diverge_wra1_neural_compiler import (
    EXPECTED_SC1_SHA256,
    _atomic_json,
    _atomic_torch,
    _state_sha256,
    load_frozen_sc1,
)
from diverge_wra1_whole_record import detect_segments, seal_source_packet
from frozen_pointer_backbone import load_frozen_pointer_backbone

SCHEMA = "shohin-diverge-hsc1-neural-structured-compiler-v1"
CPU_SCHEMA = "shohin-diverge-hsc1-structured-cpu-v1"


@dataclass(frozen=True, slots=True)
class SegmentRef:
    episode: int
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class OptionRef:
    segment: int
    episode: int
    option_index: int
    start: int
    end: int


@dataclass(slots=True)
class StructuredCompilerOutput:
    boundary: torch.Tensor
    word_lengths: torch.Tensor
    segments: tuple[SegmentRef, ...]
    episode_failures: dict[int, str]
    cut_logits: torch.Tensor
    cue_logits: torch.Tensor
    option_refs: tuple[OptionRef, ...]
    option_role_logits: torch.Tensor


def _negative_infinity(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_tensor(float("-inf"))


def torch_cut_log_partition(cuts: torch.Tensor) -> torch.Tensor:
    """Exact differentiable log-partition over 0 < a < b < t < width."""

    if cuts.ndim != 2 or cuts.shape[0] != 3 or cuts.shape[1] < 4:
        return _negative_infinity(cuts)
    # Keep only reachable absolute positions. This avoids differentiating
    # logcumsumexp through an all-negative-infinity prefix state.
    previous = cuts[0, 1:]
    for channel in (1, 2):
        first_absolute_position = channel + 1
        predecessor = torch.logcumsumexp(previous, dim=0)[:-1]
        previous = predecessor + cuts[channel, first_absolute_position:]
    return torch.logsumexp(previous, dim=0)


@lru_cache(maxsize=8)
def _template_groups(device: torch.device) -> tuple[torch.Tensor, ...]:
    groups = []
    templates = semantic_templates()
    for length in sorted({len(template.labels) for template in templates}):
        labels = [
            template.labels for template in templates if len(template.labels) == length
        ]
        groups.append(torch.tensor(labels, dtype=torch.long, device=device))
    return tuple(groups)


def torch_batched_option_log_partition(role_logits: torch.Tensor) -> torch.Tensor:
    """Exact global option partitions for a same-width batch."""

    if role_logits.ndim != 3 or role_logits.shape[2] != ROLE_COUNT:
        raise ValueError("batched option role logits have the wrong shape")
    if role_logits.shape[0] == 0 or role_logits.shape[1] == 0:
        return role_logits.new_full((role_logits.shape[0],), float("-inf"))
    margins = role_logits - role_logits[:, :, OTHER].unsqueeze(-1)
    by_role = margins.permute(0, 2, 1)
    partitions = []
    negative = _negative_infinity(role_logits)
    for labels in _template_groups(role_logits.device):
        if labels.shape[1] > role_logits.shape[1]:
            continue
        previous = by_role[:, labels[:, 0], :]
        for index in range(1, labels.shape[1]):
            monotonic = torch.logcumsumexp(previous, dim=2)[:, :, :-1]
            adjacent = previous[:, :, :-1]
            prior_labels = labels[:, index - 1]
            current_labels = labels[:, index]
            requires_adjacency = (current_labels == ALIAS_INSIDE) & (
                (prior_labels == ALIAS_BEGIN) | (prior_labels == ALIAS_INSIDE)
            )
            predecessor = torch.where(
                requires_adjacency[None, :, None], adjacent, monotonic
            )
            selected = by_role[:, current_labels, index:]
            previous = predecessor + selected
        partitions.append(torch.logsumexp(previous, dim=2))
    if not partitions:
        return negative.expand(role_logits.shape[0])
    return torch.logsumexp(torch.cat(partitions, dim=1), dim=1)


def torch_option_log_partition(role_logits: torch.Tensor) -> torch.Tensor:
    """Exact global log-partition over all templates and monotonic paths."""

    if role_logits.ndim != 2 or role_logits.shape[1] != ROLE_COUNT:
        raise ValueError("option role logits have the wrong shape")
    return torch_batched_option_log_partition(role_logits.unsqueeze(0))[0]


def torch_gold_option_score(
    role_logits: torch.Tensor, labels: Sequence[int], path: Sequence[int]
) -> torch.Tensor:
    if len(labels) != len(path):
        raise ValueError("gold option labels and path differ")
    margins = role_logits - role_logits[:, OTHER].unsqueeze(-1)
    positions = torch.tensor(path, dtype=torch.long, device=role_logits.device)
    roles = torch.tensor(labels, dtype=torch.long, device=role_logits.device)
    return margins[positions, roles].sum()


class HierarchicalStructuredCompiler(nn.Module):
    def __init__(
        self,
        source: RawSourceCompiler,
        *,
        width: int,
        local_layers: int,
        local_heads: int,
    ) -> None:
        super().__init__()
        if width != source.memory_projection.out_features:
            raise ValueError("HSC1 width must equal the frozen SC1 encoder width")
        if width % local_heads or local_layers <= 0:
            raise ValueError("invalid HSC1 local encoder geometry")
        self.source = source.eval().requires_grad_(False)
        self.width = width
        self.cut_head = nn.Linear(width, 3)
        self.cue_head = nn.Linear(width, 3)
        layer = nn.TransformerEncoderLayer(
            width,
            local_heads,
            4 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.local_encoder = nn.TransformerEncoder(
            layer,
            num_layers=local_layers,
            enable_nested_tensor=False,
        )
        self.local_norm = nn.LayerNorm(width)
        self.role_head = nn.Linear(width, ROLE_COUNT)

    def stage_a_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.cut_head.parameters()
        yield from self.cue_head.parameters()

    def stage_b_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.local_encoder.parameters()
        yield from self.local_norm.parameters()
        yield from self.role_head.parameters()

    def compiler_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.stage_a_parameters()
        yield from self.stage_b_parameters()

    def compiler_state(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if not name.startswith("source.")
        }

    def freeze_stage_a(self) -> None:
        for parameter in self.stage_a_parameters():
            parameter.requires_grad_(False)

    def _frozen_source(
        self, encodings: Sequence[SourceEncoding], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            words, lengths = self.source._encode_words(encodings, device)
            batch, _, hidden = words.shape
            bos = self.source.boundary_bos.view(1, 1, hidden).expand(batch, 1, hidden)
            eos = self.source.boundary_eos.view(1, 1, hidden).expand(batch, 1, hidden)
            left = torch.cat((bos, words), dim=1)
            right = torch.cat((words, eos), dim=1)
            boundary = (
                self.source.boundary_head(torch.cat((left, right), dim=-1))
                .squeeze(-1)
                .float()
            )
        return words, lengths, boundary

    def forward(
        self,
        encodings: Sequence[SourceEncoding],
        device: torch.device,
        *,
        include_options: bool,
    ) -> StructuredCompilerOutput:
        words, lengths, boundary = self._frozen_source(encodings, device)
        refs: list[SegmentRef] = []
        memories = []
        failures: dict[int, str] = {}
        for episode_index, length_tensor in enumerate(lengths):
            length = int(length_tensor.item())
            segments, reason, _ = detect_segments(
                boundary[episode_index, : length + 1].detach().cpu().tolist(), length
            )
            if reason is not None:
                failures[episode_index] = reason
                continue
            for start, end in segments:
                refs.append(SegmentRef(episode_index, start, end))
                memories.append(words[episode_index, start:end])
        if not memories:
            return StructuredCompilerOutput(
                boundary,
                lengths,
                tuple(refs),
                failures,
                words.new_zeros((0, 3, 0), dtype=torch.float32),
                words.new_zeros((0, 0, 3), dtype=torch.float32),
                (),
                words.new_zeros((0, 0, ROLE_COUNT), dtype=torch.float32),
            )

        widths = torch.tensor([memory.shape[0] for memory in memories], device=device)
        maximum = int(widths.max().item())
        memory = words.new_zeros((len(memories), maximum, self.width))
        valid = torch.arange(maximum, device=device)[None, :] < widths[:, None]
        for row, value in enumerate(memories):
            memory[row, : value.shape[0]] = value
        cut_logits = self.cut_head(memory).transpose(1, 2).float()
        cue_logits = self.cue_head(memory).float()
        cut_logits = cut_logits.masked_fill(~valid[:, None, :], float("-inf"))
        cue_logits = cue_logits.masked_fill(~valid[:, :, None], float("-inf"))

        if not include_options:
            return StructuredCompilerOutput(
                boundary,
                lengths,
                tuple(refs),
                failures,
                cut_logits,
                cue_logits,
                (),
                words.new_zeros((0, 0, ROLE_COUNT), dtype=torch.float32),
            )

        option_refs: list[OptionRef] = []
        option_memories = []
        for segment_index, ref in enumerate(refs):
            width = ref.end - ref.start
            _, cuts = cut_viterbi(
                cut_logits[segment_index, :, :width].detach().cpu().tolist()
            )
            if len(cuts) != 3:
                continue
            left, middle, trailer = cuts
            for option_index, (start, end) in enumerate(
                ((left, middle), (middle, trailer))
            ):
                option_refs.append(
                    OptionRef(
                        segment_index,
                        ref.episode,
                        option_index,
                        ref.start + start,
                        ref.start + end,
                    )
                )
                option_memories.append(memory[segment_index, start:end])
        if not option_memories:
            return StructuredCompilerOutput(
                boundary,
                lengths,
                tuple(refs),
                failures,
                cut_logits,
                cue_logits,
                tuple(option_refs),
                words.new_zeros((0, 0, ROLE_COUNT), dtype=torch.float32),
            )

        option_widths = torch.tensor(
            [value.shape[0] for value in option_memories], device=device
        )
        maximum_option = int(option_widths.max().item())
        options = words.new_zeros((len(option_memories), maximum_option, self.width))
        option_valid = (
            torch.arange(maximum_option, device=device)[None, :]
            < option_widths[:, None]
        )
        for row, value in enumerate(option_memories):
            options[row, : value.shape[0]] = value
        options = self.local_encoder(options, src_key_padding_mask=~option_valid)
        role_logits = self.role_head(self.local_norm(options)).float()
        role_logits = role_logits.masked_fill(~option_valid[:, :, None], float("-inf"))
        return StructuredCompilerOutput(
            boundary,
            lengths,
            tuple(refs),
            failures,
            cut_logits,
            cue_logits,
            tuple(option_refs),
            role_logits,
        )


def _episode_encodings(
    model: HierarchicalStructuredCompiler, episodes: Sequence[RawSourceEpisode]
) -> list[SourceEncoding]:
    return [
        encode_source(model.source.tokenizer, episode.tokens) for episode in episodes
    ]


def stage_a_loss(
    model: HierarchicalStructuredCompiler,
    episodes: Sequence[RawSourceEpisode],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(_episode_encodings(model, episodes), device, include_options=False)
    gold = {
        (episode_index, record.start, record.end): (episode, record)
        for episode_index, episode in enumerate(episodes)
        for record in episode.records
    }
    losses = []
    exact_phases = 0
    supervised = 0
    predicted_by_episode = {index: [] for index in range(len(episodes))}
    for ref in output.segments:
        predicted_by_episode[ref.episode].append((ref.start, ref.end))
    exact_segmentation = sum(
        predicted_by_episode[index]
        == [(record.start, record.end) for record in episode.records]
        for index, episode in enumerate(episodes)
    )
    for segment_index, ref in enumerate(output.segments):
        row = gold.get((ref.episode, ref.start, ref.end))
        if row is None:
            continue
        episode, record = row
        targets = tuple(value - ref.start for value in option_markers(episode, record))
        width = ref.end - ref.start
        cuts = output.cut_logits[segment_index, :, :width]
        log_partition = torch_cut_log_partition(cuts)
        gold_score = sum(
            cuts[channel, target] for channel, target in enumerate(targets)
        )
        cue = output.cue_logits[segment_index, :width]
        cue_margins = cue[:, 1:3] - cue[:, OTHER].unsqueeze(-1)
        cue_target = (record.cue_position - ref.start) * 2 + int(
            not record.is_fault_line
        )
        cue_loss = F.cross_entropy(
            cue_margins.reshape(1, -1),
            torch.tensor([cue_target], dtype=torch.long, device=device),
        )
        losses.append(log_partition - gold_score + cue_loss)
        _, predicted = cut_viterbi(cuts.detach().cpu().tolist())
        exact_phases += int(predicted == targets)
        supervised += 1
    if not losses:
        raise RuntimeError("frozen boundary detector produced no HSC1 records")
    loss = torch.stack(losses).mean()
    return loss, {
        "loss": float(loss.detach().item()),
        "supervised_records": float(supervised),
        "predicted_records": float(len(output.segments)),
        "segmentation_exact": exact_segmentation / len(episodes),
        "phase_exact": exact_phases / supervised,
        "boundary_failures": float(len(output.episode_failures)),
    }


def stage_b_loss(
    model: HierarchicalStructuredCompiler,
    episodes: Sequence[RawSourceEpisode],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(_episode_encodings(model, episodes), device, include_options=True)
    records = {
        (episode_index, record.start, record.end): (episode, record)
        for episode_index, episode in enumerate(episodes)
        for record in episode.records
    }
    grouped: dict[int, list[tuple[torch.Tensor, tuple[int, ...], tuple[int, ...]]]] = {}
    eligible = 0
    predicted = len(output.option_refs)
    phase_exact = 0
    for option_row, ref in enumerate(output.option_refs):
        segment = output.segments[ref.segment]
        item = records.get((ref.episode, segment.start, segment.end))
        if item is None:
            continue
        episode, record = item
        markers = option_markers(episode, record)
        expected = (
            (markers[0], markers[1]),
            (markers[1], markers[2]),
        )[ref.option_index]
        if (ref.start, ref.end) != expected:
            continue
        phase_exact += 1
        option = record.options[ref.option_index]
        template, path = gold_option_path(option, ref.start)
        width = ref.end - ref.start
        role_logits = output.option_role_logits[option_row, :width]
        grouped.setdefault(width, []).append((role_logits, template.labels, path))
        eligible += 1
    if not grouped:
        raise RuntimeError("stage-A HSC1 parse produced no exact option spans")
    losses = []
    for rows in grouped.values():
        logits = torch.stack([row[0] for row in rows])
        partitions = torch_batched_option_log_partition(logits)
        gold = torch.stack(
            [torch_gold_option_score(role, labels, path) for role, labels, path in rows]
        )
        losses.extend(partitions - gold)
    loss = torch.stack(losses).mean()
    return loss, {
        "loss": float(loss.detach().item()),
        "eligible_options": float(eligible),
        "predicted_options": float(predicted),
        "phase_exact": phase_exact / max(1, predicted),
        "boundary_failures": float(len(output.episode_failures)),
    }


def output_to_scores(
    output: StructuredCompilerOutput, episode_index: int, token_count: int
) -> HierarchicalScores:
    boundary = tuple(
        float(value)
        for value in output.boundary[episode_index, : token_count + 1].detach().cpu()
    )
    option_lookup = {
        (ref.segment, ref.option_index): (row, ref)
        for row, ref in enumerate(output.option_refs)
        if ref.episode == episode_index
    }
    rows = []
    for segment_index, ref in enumerate(output.segments):
        if ref.episode != episode_index:
            continue
        width = ref.end - ref.start
        options = []
        for option_index in range(2):
            item = option_lookup.get((segment_index, option_index))
            if item is None:
                break
            option_row, option_ref = item
            option_width = option_ref.end - option_ref.start
            options.append(
                OptionRoleScores(
                    option_ref.start,
                    option_ref.end,
                    tuple(
                        tuple(float(value) for value in role.detach().cpu())
                        for role in output.option_role_logits[option_row, :option_width]
                    ),
                )
            )
        if len(options) != 2:
            continue
        rows.append(
            RecordStructuredScores(
                ref.start,
                ref.end,
                tuple(
                    tuple(
                        float(value)
                        for value in output.cut_logits[segment_index, channel, :width]
                        .detach()
                        .cpu()
                    )
                    for channel in range(3)
                ),
                tuple(
                    tuple(float(value) for value in values.detach().cpu())
                    for values in output.cue_logits[segment_index, :width]
                ),
                (options[0], options[1]),
            )
        )
    return HierarchicalScores(boundary, tuple(rows))


def _record_signature(record) -> tuple[object, ...]:
    return (
        record.start,
        record.end,
        record.cue_position,
        record.is_fault_line,
        tuple(
            (
                option.alias_span,
                option.prior_position,
                option.prior_class,
                option.action_positions,
                option.program,
            )
            for option in record.options
        ),
    )


def _support_recalled(episode: RawSourceEpisode, receipt) -> bool:
    expected = {
        (
            record.start,
            record.end,
            record.cue_position,
            record.is_fault_line,
            tuple(
                (
                    option.alias_span,
                    option.prior_position,
                    option.prior_class,
                    option.action_positions,
                    option.program,
                )
                for option in record.options
            ),
        )
        for record in episode.records
        if record.is_fault_line
    }
    observed = {
        _record_signature(record) for record in receipt.records if record.is_fault_line
    }
    return expected.issubset(observed)


def _accepted_overlap(receipt) -> bool:
    if receipt.failed:
        return False
    for record in receipt.records:
        occupied = []
        for option in record.options:
            fields = set(range(*option.alias_span)) | {
                option.prior_position,
                *option.action_positions,
            }
            occupied.append(fields)
        if (
            occupied[0] & occupied[1]
            or record.cue_position in occupied[0] | occupied[1]
        ):
            return True
    return False


def _phase_exact(episode: RawSourceEpisode, scores: HierarchicalScores) -> bool:
    segments, reason, _ = detect_segments(scores.boundary, len(episode.tokens))
    expected_segments = tuple((record.start, record.end) for record in episode.records)
    if reason is not None or segments != expected_segments:
        return False
    rows = {(row.start, row.end): row for row in scores.records}
    if set(rows) != set(expected_segments):
        return False
    for record in episode.records:
        row = rows[record.start, record.end]
        _, cuts = cut_viterbi(row.cuts)
        expected = tuple(
            position - record.start for position in option_markers(episode, record)
        )
        if cuts != expected:
            return False
    return True


def evaluate(
    model: HierarchicalStructuredCompiler,
    *,
    cohort: str,
    count: int,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    totals = {
        "episodes": 0,
        "segmentation_exact": 0,
        "phase_exact": 0,
        "support_recalled": 0,
        "exact_packet": 0,
        "semantic_shuffle_exact": 0,
        "cut_shuffle_exact": 0,
        "source_poison_invariant": 0,
        "failed_closed": 0,
        "overflow": 0,
        "accepted_overlap": 0,
        "source_words": 0,
        "predicted_records": 0,
        "predicted_options": 0,
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, count, batch_size):
            episodes = [
                generate_episode(seed=seed + index, cohort=cohort)
                for index in range(start, min(count, start + batch_size))
            ]
            output = model(
                _episode_encodings(model, episodes), device, include_options=True
            )
            for row, episode in enumerate(episodes):
                scores = output_to_scores(output, row, len(episode.tokens))
                segments, reason, _ = detect_segments(
                    scores.boundary, len(episode.tokens)
                )
                expected_segments = tuple(
                    (record.start, record.end) for record in episode.records
                )
                receipt = decode_hierarchical(episode.tokens, scores)
                semantic_shuffled = decode_hierarchical(
                    episode.tokens, shuffle_semantic_roles(scores)
                )
                cut_shuffled = decode_hierarchical(
                    episode.tokens, shuffle_cut_channels(scores)
                )
                packet = seal_source_packet(episode.tokens, receipt)
                poisoned = seal_source_packet(
                    tuple("poison" for _ in episode.tokens), receipt
                )
                totals["episodes"] += 1
                totals["segmentation_exact"] += int(
                    reason is None and segments == expected_segments
                )
                totals["phase_exact"] += int(_phase_exact(episode, scores))
                totals["support_recalled"] += int(_support_recalled(episode, receipt))
                totals["exact_packet"] += int(exact(episode, receipt))
                totals["semantic_shuffle_exact"] += int(
                    exact(episode, semantic_shuffled)
                )
                totals["cut_shuffle_exact"] += int(exact(episode, cut_shuffled))
                totals["source_poison_invariant"] += int(
                    packet.records == poisoned.records
                )
                totals["failed_closed"] += int(receipt.failed)
                totals["overflow"] += int(receipt.overflow)
                totals["accepted_overlap"] += int(_accepted_overlap(receipt))
                totals["source_words"] += len(episode.tokens)
                totals["predicted_records"] += receipt.record_objects
                totals["predicted_options"] += receipt.option_objects
    episodes = totals["episodes"]
    rates = {
        key: value / episodes
        for key, value in totals.items()
        if key
        not in {
            "episodes",
            "source_words",
            "predicted_records",
            "predicted_options",
        }
    }
    rates["mean_source_words"] = totals["source_words"] / episodes
    rates["mean_predicted_records"] = totals["predicted_records"] / episodes
    rates["mean_predicted_options"] = totals["predicted_options"] / episodes
    return {"cohort": cohort, "count": count, "totals": totals, "rates": rates}


def _train_stage(
    model: HierarchicalStructuredCompiler,
    *,
    stage: str,
    updates: int,
    batch_size: int,
    data_seed: int,
    learning_rate: float,
    log_every: int,
    device: torch.device,
) -> tuple[dict[str, float], int, float, int]:
    if stage == "a":
        parameters = list(model.stage_a_parameters())
        loss_function = stage_a_loss
    elif stage == "b":
        parameters = list(model.stage_b_parameters())
        loss_function = stage_b_loss
    else:
        raise ValueError("unknown HSC1 training stage")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=0.01,
        fused=device.type == "cuda",
    )
    started = time.monotonic()
    charged = 0
    final_metrics: dict[str, float] = {}
    model.train()
    for update in range(1, updates + 1):
        episodes = [
            generate_episode(
                seed=data_seed + (update - 1) * batch_size + index,
                cohort="train",
            )
            for index in range(batch_size)
        ]
        optimizer.zero_grad(set_to_none=True)
        loss, final_metrics = loss_function(model, episodes, device)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite HSC1 stage-{stage} loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        charged += len(episodes)
        if update == 1 or update % log_every == 0:
            print(
                json.dumps(
                    {
                        "stage": stage,
                        "update": update,
                        "charged_episodes": charged,
                        "gradient_norm": float(gradient_norm),
                        **final_metrics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return (
        final_metrics,
        charged,
        time.monotonic() - started,
        sum(parameter.numel() for parameter in parameters),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--sc1-checkpoint", type=Path, required=True)
    parser.add_argument("--cpu-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202608056900)
    parser.add_argument("--data-seed", type=int, default=202608057000)
    parser.add_argument("--eval-seed", type=int, default=202608057100)
    parser.add_argument("--stage-a-updates", type=int, default=200)
    parser.add_argument("--stage-b-updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-count", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--pair-width", type=int, default=64)
    parser.add_argument("--local-layers", type=int, default=2)
    parser.add_argument("--local-heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    arguments = parser.parse_args()
    if (
        arguments.stage_a_updates <= 0
        or arguments.stage_b_updates <= 0
        or arguments.batch_size <= 0
        or arguments.eval_count <= 0
    ):
        raise ValueError("training and evaluation sizes must be positive")
    if sha256_file(arguments.sc1_checkpoint) != EXPECTED_SC1_SHA256:
        raise ValueError("frozen SC1 checkpoint hash differs")
    cpu_report = json.loads(arguments.cpu_report.read_text(encoding="utf-8"))
    if not cpu_report.get("passed") or cpu_report.get("schema") != CPU_SCHEMA:
        raise ValueError("HSC1 CPU gate is absent or failed")
    if arguments.output.exists() or arguments.output.with_suffix(".pt").exists():
        raise FileExistsError("refusing to overwrite an HSC1 artifact")

    torch.set_num_threads(arguments.threads)
    torch.manual_seed(arguments.seed)
    random.seed(arguments.seed)
    device = torch.device(arguments.device)
    backbone, _, backbone_receipt = load_frozen_pointer_backbone(
        arguments.base, device=device
    )
    tokenizer = Tokenizer.from_file(str(arguments.tokenizer))
    source = load_frozen_sc1(
        backbone,
        tokenizer,
        arguments.sc1_checkpoint,
        layer=arguments.layer,
        width=arguments.width,
        pair_width=arguments.pair_width,
    )
    model = HierarchicalStructuredCompiler(
        source,
        width=arguments.width,
        local_layers=arguments.local_layers,
        local_heads=arguments.local_heads,
    ).to(device)

    stage_a_metrics, stage_a_charge, stage_a_seconds, stage_a_parameters = _train_stage(
        model,
        stage="a",
        updates=arguments.stage_a_updates,
        batch_size=arguments.batch_size,
        data_seed=arguments.data_seed,
        learning_rate=arguments.learning_rate,
        log_every=arguments.log_every,
        device=device,
    )
    model.freeze_stage_a()
    stage_b_metrics, stage_b_charge, stage_b_seconds, stage_b_parameters = _train_stage(
        model,
        stage="b",
        updates=arguments.stage_b_updates,
        batch_size=arguments.batch_size,
        data_seed=arguments.data_seed + stage_a_charge,
        learning_rate=arguments.learning_rate,
        log_every=arguments.log_every,
        device=device,
    )
    peak_memory = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    evaluations = {
        cohort: evaluate(
            model,
            cohort=cohort,
            count=arguments.eval_count,
            seed=arguments.eval_seed + offset,
            batch_size=arguments.eval_batch_size,
            device=device,
        )
        for cohort, offset in (
            ("train", 0),
            ("lexical_shift", 100_000),
            ("renderer_shift", 200_000),
            ("composition_shift", 300_000),
        )
    }
    rates = {name: row["rates"] for name, row in evaluations.items()}
    gates = {
        "segmentation": min(row["segmentation_exact"] for row in rates.values())
        >= 0.99,
        "phase_parse": min(row["phase_exact"] for row in rates.values()) >= 0.99,
        "support": min(row["support_recalled"] for row in rates.values()) >= 0.95,
        "exact_packet": min(row["exact_packet"] for row in rates.values()) >= 0.95,
        "no_invalid": max(row["accepted_overlap"] for row in rates.values()) == 0.0,
        "no_overflow": max(row["overflow"] for row in rates.values()) == 0.0,
        "source_poison": min(row["source_poison_invariant"] for row in rates.values())
        == 1.0,
        "semantic_causality": min(
            row["exact_packet"] - row["semantic_shuffle_exact"]
            for row in rates.values()
        )
        >= 0.20,
        "cut_causality": min(
            row["exact_packet"] - row["cut_shuffle_exact"] for row in rates.values()
        )
        >= 0.20,
        "closed_compiler_advantage": min(
            rates[name]["exact_packet"]
            for name in ("lexical_shift", "renderer_shift", "composition_shift")
        )
        >= 0.15,
    }
    state = model.compiler_state()
    state_hash = _state_sha256(state)
    checkpoint = arguments.output.with_suffix(".pt")
    _atomic_torch(
        checkpoint,
        {
            "schema": SCHEMA,
            "arguments": vars(arguments),
            "state_dict": state,
            "model_state_sha256": state_hash,
            "source_checkpoint_sha256": EXPECTED_SC1_SHA256,
        },
    )
    report = {
        "schema": SCHEMA,
        "seed": arguments.seed,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(arguments).items()
        },
        "input_hashes": {
            "base": sha256_file(arguments.base),
            "tokenizer": sha256_file(arguments.tokenizer),
            "sc1_checkpoint": sha256_file(arguments.sc1_checkpoint),
            "cpu_report": sha256_file(arguments.cpu_report),
        },
        "backbone_receipt": {
            "checkpoint_format": backbone_receipt.checkpoint_format,
            "base_step": backbone_receipt.base_step,
            "initialization": backbone_receipt.initialization,
        },
        "training": {
            "updates": arguments.stage_a_updates + arguments.stage_b_updates,
            "charged_episodes": stage_a_charge + stage_b_charge,
            "stage_a": {
                "updates": arguments.stage_a_updates,
                "charged_episodes": stage_a_charge,
                "seconds": stage_a_seconds,
                "episodes_per_second": stage_a_charge / stage_a_seconds,
                "parameters": stage_a_parameters,
                "final_metrics": stage_a_metrics,
            },
            "stage_b": {
                "updates": arguments.stage_b_updates,
                "charged_episodes": stage_b_charge,
                "seconds": stage_b_seconds,
                "episodes_per_second": stage_b_charge / stage_b_seconds,
                "parameters": stage_b_parameters,
                "final_metrics": stage_b_metrics,
            },
            "peak_memory_bytes": peak_memory,
        },
        "parameters": {
            "complete": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_total": sum(
                parameter.numel() for parameter in model.compiler_parameters()
            ),
        },
        "evaluations": evaluations,
        "gates": gates,
        "passed_seed_gate": all(gates.values()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_state_sha256": state_hash,
    }
    _atomic_json(arguments.output, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["passed_seed_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
