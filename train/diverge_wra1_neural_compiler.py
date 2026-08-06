#!/usr/bin/env python3
"""Train DIVERGE-WRA1 complete-object slots over a frozen SC1 source pass."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from diverge_sc1_neural_compiler import (
    RawSourceCompiler,
    SourceEncoding,
    encode_source,
    sha256_file,
)
from diverge_sc1_source_compiler import (
    GoldOption,
    GoldRecord,
    RawSourceEpisode,
    generate_episode,
)
from diverge_wra1_whole_record import (
    MAX_ALIAS_LENGTH,
    PROGRAM_ROLES,
    SegmentScores,
    SlotScores,
    WholeRecordScores,
    decode_whole_records,
    detect_segments,
    exact,
    seal_source_packet,
    shuffle_lineage,
)
from frozen_pointer_backbone import load_frozen_pointer_backbone

SCHEMA = "shohin-diverge-wra1-neural-whole-record-v1"
EXPECTED_SC1_SHA256 = "7b5348cacb1772bf45e34442e94010db71a6be20bd8d689477d037ac5fee2ffd"


@dataclass(frozen=True, slots=True)
class SegmentRef:
    episode: int
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class OptionTargets:
    alias_start: int
    alias_length: int
    prior_class: int
    program_class: int
    prior_pointer: int
    action_1_pointer: int
    action_2_pointer_or_halt: int


@dataclass(slots=True)
class WholeCompilerOutput:
    boundary: torch.Tensor
    word_lengths: torch.Tensor
    segments: tuple[SegmentRef, ...]
    episode_failures: dict[int, str]
    record_kind: torch.Tensor
    cue_pointer: torch.Tensor
    alias_start: torch.Tensor
    alias_length: torch.Tensor
    prior_class: torch.Tensor
    program_class: torch.Tensor
    prior_pointer: torch.Tensor
    action_1_pointer: torch.Tensor
    action_2_pointer_or_halt: torch.Tensor


def option_targets(
    record: GoldRecord, option: GoldOption, *, halt_index: int
) -> OptionTargets:
    action_2 = (
        option.action_positions[1] - record.start
        if len(option.action_positions) == 2
        else halt_index
    )
    return OptionTargets(
        option.alias_span[0] - record.start,
        option.alias_span[1] - option.alias_span[0] - 1,
        option.prior_class,
        option.program,
        option.prior_position - record.start,
        option.action_positions[0] - record.start,
        action_2,
    )


class WholeRecordAssignmentCompiler(nn.Module):
    def __init__(
        self,
        source: RawSourceCompiler,
        *,
        width: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        if width != source.memory_projection.out_features:
            raise ValueError("WRA width must equal the frozen SC1 encoder width")
        if width % heads or layers <= 0:
            raise ValueError("invalid whole-record slot geometry")
        self.source = source.eval().requires_grad_(False)
        self.width = width
        self.slot_seeds = nn.Parameter(torch.empty(2, width))
        nn.init.normal_(self.slot_seeds, std=0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            width,
            heads,
            4 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.slot_decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        self.segment_norm = nn.LayerNorm(width)
        self.record_kind_head = nn.Linear(width, 2)
        self.cue_query = nn.Linear(width, width, bias=False)
        self.pointer_key = nn.Linear(width, width, bias=False)
        self.alias_query = nn.Linear(width, width, bias=False)
        self.prior_query = nn.Linear(width, width, bias=False)
        self.action_1_query = nn.Linear(width, width, bias=False)
        self.action_2_query = nn.Linear(width, width, bias=False)
        self.alias_length_head = nn.Linear(width, MAX_ALIAS_LENGTH)
        self.prior_class_head = nn.Linear(width, 2)
        self.program_class_head = nn.Linear(width, len(PROGRAM_ROLES))
        self.action_2_halt_head = nn.Linear(width, 1)

    def compiler_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if not name.startswith("source."):
                yield parameter

    def compiler_state(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if not name.startswith("source.")
        }

    def _frozen_source(
        self,
        encodings: Sequence[SourceEncoding],
        device: torch.device,
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

    @staticmethod
    def _pointer(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bsd,bwd->bsw", query, key) / (query.shape[-1] ** 0.5)

    def forward(
        self,
        encodings: Sequence[SourceEncoding],
        device: torch.device,
    ) -> WholeCompilerOutput:
        words, lengths, boundary = self._frozen_source(encodings, device)
        refs: list[SegmentRef] = []
        memories = []
        failures: dict[int, str] = {}
        for row, length_tensor in enumerate(lengths):
            length = int(length_tensor.item())
            segments, reason, _ = detect_segments(
                boundary[row, : length + 1].detach().cpu().tolist(), length
            )
            if reason is not None:
                failures[row] = reason
                continue
            for start, end in segments:
                refs.append(SegmentRef(row, start, end))
                memories.append(words[row, start:end])
        if not memories:
            return WholeCompilerOutput(
                boundary,
                lengths,
                tuple(refs),
                failures,
                words.new_zeros((0, 2)),
                words.new_zeros((0, 0)),
                words.new_zeros((0, 2, 0)),
                words.new_zeros((0, 2, MAX_ALIAS_LENGTH)),
                words.new_zeros((0, 2, 2)),
                words.new_zeros((0, 2, len(PROGRAM_ROLES))),
                words.new_zeros((0, 2, 0)),
                words.new_zeros((0, 2, 0)),
                words.new_zeros((0, 2, 1)),
            )

        widths = torch.tensor([memory.shape[0] for memory in memories], device=device)
        maximum = int(widths.max().item())
        memory = words.new_zeros((len(memories), maximum, self.width))
        valid = torch.arange(maximum, device=device)[None, :] < widths[:, None]
        for row, value in enumerate(memories):
            memory[row, : value.shape[0]] = value
        summary = (memory * valid.unsqueeze(-1)).sum(1) / widths.clamp_min(1).unsqueeze(
            -1
        )
        summary = self.segment_norm(summary)
        queries = self.slot_seeds.unsqueeze(0) + summary.unsqueeze(1)
        slots = self.slot_decoder(
            queries,
            memory,
            memory_key_padding_mask=~valid,
        )
        key = self.pointer_key(memory)
        alias_start = self._pointer(self.alias_query(slots), key)
        prior_pointer = self._pointer(self.prior_query(slots), key)
        action_1 = self._pointer(self.action_1_query(slots), key)
        action_2_values = self._pointer(self.action_2_query(slots), key)
        invalid = ~valid[:, None, :]
        alias_start = alias_start.masked_fill(invalid, float("-inf"))
        prior_pointer = prior_pointer.masked_fill(invalid, float("-inf"))
        action_1 = action_1.masked_fill(invalid, float("-inf"))
        action_2_values = action_2_values.masked_fill(invalid, float("-inf"))
        action_2 = torch.cat((action_2_values, self.action_2_halt_head(slots)), dim=-1)
        cue = torch.einsum("sd,swd->sw", self.cue_query(summary), key) / (
            self.width**0.5
        )
        cue = cue.masked_fill(~valid, float("-inf"))
        return WholeCompilerOutput(
            boundary,
            lengths,
            tuple(refs),
            failures,
            self.record_kind_head(summary).float(),
            cue.float(),
            alias_start.float(),
            self.alias_length_head(slots).float(),
            self.prior_class_head(slots).float(),
            self.program_class_head(slots).float(),
            prior_pointer.float(),
            action_1.float(),
            action_2.float(),
        )


def _ce(logits: torch.Tensor, target: int) -> torch.Tensor:
    return F.cross_entropy(
        logits.unsqueeze(0),
        torch.tensor([target], dtype=torch.long, device=logits.device),
    )


def _slot_cost(
    output: WholeCompilerOutput,
    segment_index: int,
    slot_index: int,
    target: OptionTargets,
) -> torch.Tensor:
    return sum(
        (
            _ce(output.alias_start[segment_index, slot_index], target.alias_start),
            _ce(output.alias_length[segment_index, slot_index], target.alias_length),
            _ce(output.prior_class[segment_index, slot_index], target.prior_class),
            _ce(output.program_class[segment_index, slot_index], target.program_class),
            _ce(output.prior_pointer[segment_index, slot_index], target.prior_pointer),
            _ce(
                output.action_1_pointer[segment_index, slot_index],
                target.action_1_pointer,
            ),
            _ce(
                output.action_2_pointer_or_halt[segment_index, slot_index],
                target.action_2_pointer_or_halt,
            ),
        )
    )


def _training_loss(
    model: WholeRecordAssignmentCompiler,
    episodes: Sequence[RawSourceEpisode],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(
        [encode_source(model.source.tokenizer, row.tokens) for row in episodes], device
    )
    gold = {
        (episode_index, record.start, record.end): record
        for episode_index, episode in enumerate(episodes)
        for record in episode.records
    }
    segment_losses = []
    matched = 0
    exact_segmentation = 0
    predicted_by_episode: dict[int, list[tuple[int, int]]] = {
        index: [] for index in range(len(episodes))
    }
    for ref in output.segments:
        predicted_by_episode[ref.episode].append((ref.start, ref.end))
    for index, episode in enumerate(episodes):
        expected = [(record.start, record.end) for record in episode.records]
        exact_segmentation += int(predicted_by_episode[index] == expected)

    halt_index = output.action_2_pointer_or_halt.shape[-1] - 1
    for segment_index, ref in enumerate(output.segments):
        record = gold.get((ref.episode, ref.start, ref.end))
        if record is None:
            continue
        targets = [
            option_targets(record, option, halt_index=halt_index)
            for option in record.options
        ]
        costs = [
            [_slot_cost(output, segment_index, slot, target) for target in targets]
            for slot in range(2)
        ]
        assignment = torch.minimum(costs[0][0] + costs[1][1], costs[0][1] + costs[1][0])
        kind = _ce(output.record_kind[segment_index], int(record.is_fault_line))
        cue = _ce(output.cue_pointer[segment_index], record.cue_position - record.start)
        segment_losses.append(kind + cue + 0.5 * assignment)
        matched += 1
    if not segment_losses:
        raise RuntimeError("frozen boundary detector produced no supervised segments")
    loss = torch.stack(segment_losses).mean()
    return loss, {
        "loss": float(loss.detach().item()),
        "matched_segments": float(matched),
        "predicted_segments": float(len(output.segments)),
        "segmentation_exact": exact_segmentation / len(episodes),
        "boundary_failures": float(len(output.episode_failures)),
    }


def output_to_scores(
    output: WholeCompilerOutput,
    episode_index: int,
    token_count: int,
) -> WholeRecordScores:
    boundary = tuple(
        float(value)
        for value in output.boundary[episode_index, : token_count + 1].detach().cpu()
    )
    rows = []
    halt_index = output.action_2_pointer_or_halt.shape[-1] - 1
    for index, ref in enumerate(output.segments):
        if ref.episode != episode_index:
            continue
        width = ref.end - ref.start
        slots = []
        for slot in range(2):
            action_2 = output.action_2_pointer_or_halt[index, slot]
            slots.append(
                SlotScores(
                    tuple(
                        float(value)
                        for value in output.alias_start[index, slot, :width]
                        .detach()
                        .cpu()
                    ),
                    tuple(
                        float(value)
                        for value in output.alias_length[index, slot].detach().cpu()
                    ),
                    tuple(
                        float(value)
                        for value in output.prior_class[index, slot].detach().cpu()
                    ),
                    tuple(
                        float(value)
                        for value in output.program_class[index, slot].detach().cpu()
                    ),
                    tuple(
                        float(value)
                        for value in output.prior_pointer[index, slot, :width]
                        .detach()
                        .cpu()
                    ),
                    tuple(
                        float(value)
                        for value in output.action_1_pointer[index, slot, :width]
                        .detach()
                        .cpu()
                    ),
                    tuple(
                        [float(value) for value in action_2[:width].detach().cpu()]
                        + [float(action_2[halt_index].detach().cpu())]
                    ),
                )
            )
        rows.append(
            SegmentScores(
                ref.start,
                ref.end,
                tuple(
                    float(value) for value in output.record_kind[index].detach().cpu()
                ),
                tuple(
                    float(value)
                    for value in output.cue_pointer[index, :width].detach().cpu()
                ),
                (slots[0], slots[1]),
            )
        )
    return WholeRecordScores(boundary, tuple(rows))


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


def evaluate(
    model: WholeRecordAssignmentCompiler,
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
        "support_recalled": 0,
        "exact_packet": 0,
        "lineage_shuffle_exact": 0,
        "source_poison_invariant": 0,
        "failed_closed": 0,
        "overflow": 0,
        "accepted_duplicate_fields": 0,
        "source_words": 0,
        "predicted_segments": 0,
        "predicted_slots": 0,
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, count, batch_size):
            episodes = [
                generate_episode(seed=seed + index, cohort=cohort)
                for index in range(start, min(count, start + batch_size))
            ]
            output = model(
                [
                    encode_source(model.source.tokenizer, episode.tokens)
                    for episode in episodes
                ],
                device,
            )
            for row, episode in enumerate(episodes):
                scores = output_to_scores(output, row, len(episode.tokens))
                segments, reason, _ = detect_segments(
                    scores.boundary, len(episode.tokens)
                )
                expected_segments = tuple(
                    (record.start, record.end) for record in episode.records
                )
                receipt = decode_whole_records(episode.tokens, scores)
                shuffled = decode_whole_records(episode.tokens, shuffle_lineage(scores))
                packet = seal_source_packet(episode.tokens, receipt)
                poisoned = seal_source_packet(
                    tuple("poison" for _ in episode.tokens), receipt
                )
                totals["episodes"] += 1
                totals["segmentation_exact"] += int(
                    reason is None and segments == expected_segments
                )
                totals["support_recalled"] += int(_support_recalled(episode, receipt))
                totals["exact_packet"] += int(exact(episode, receipt))
                totals["lineage_shuffle_exact"] += int(exact(episode, shuffled))
                totals["source_poison_invariant"] += int(
                    packet.records == poisoned.records
                )
                totals["failed_closed"] += int(receipt.failed)
                totals["overflow"] += int(receipt.overflow)
                totals["source_words"] += len(episode.tokens)
                totals["predicted_segments"] += receipt.record_objects
                totals["predicted_slots"] += receipt.option_objects
    episodes = totals["episodes"]
    rates = {
        key: value / episodes
        for key, value in totals.items()
        if key
        not in {"episodes", "source_words", "predicted_segments", "predicted_slots"}
    }
    rates["mean_source_words"] = totals["source_words"] / episodes
    rates["mean_predicted_segments"] = totals["predicted_segments"] / episodes
    rates["mean_predicted_slots"] = totals["predicted_slots"] / episodes
    return {"cohort": cohort, "count": count, "totals": totals, "rates": rates}


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _atomic_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


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


def load_frozen_sc1(
    backbone: nn.Module,
    tokenizer: Tokenizer,
    checkpoint: Path,
    *,
    layer: int,
    width: int,
    pair_width: int,
) -> RawSourceCompiler:
    source = RawSourceCompiler(
        backbone,
        tokenizer,
        layer=layer,
        width=width,
        pair_width=pair_width,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = source.load_state_dict(payload["state_dict"], strict=False)
    if unexpected or any(not name.startswith("backbone.") for name in missing):
        raise ValueError("SC1 checkpoint does not match frozen source architecture")
    return source.eval().requires_grad_(False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--sc1-checkpoint", type=Path, required=True)
    parser.add_argument("--cpu-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202608056500)
    parser.add_argument("--data-seed", type=int, default=202608056600)
    parser.add_argument("--eval-seed", type=int, default=202608056700)
    parser.add_argument("--updates", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-count", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--pair-width", type=int, default=64)
    parser.add_argument("--slot-layers", type=int, default=2)
    parser.add_argument("--slot-heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    arguments = parser.parse_args()
    if arguments.updates <= 0 or arguments.batch_size <= 0 or arguments.eval_count <= 0:
        raise ValueError("training and evaluation sizes must be positive")
    if sha256_file(arguments.sc1_checkpoint) != EXPECTED_SC1_SHA256:
        raise ValueError("frozen SC1 checkpoint hash differs")
    cpu_report = json.loads(arguments.cpu_report.read_text(encoding="utf-8"))
    if (
        not cpu_report.get("passed")
        or cpu_report.get("schema") != "shohin-diverge-wra1-whole-record-cpu-v1"
    ):
        raise ValueError("WRA1 CPU gate is absent or failed")

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
    model = WholeRecordAssignmentCompiler(
        source,
        width=arguments.width,
        layers=arguments.slot_layers,
        heads=arguments.slot_heads,
    ).to(device)
    parameters = list(model.compiler_parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=arguments.learning_rate,
        weight_decay=0.01,
        fused=device.type == "cuda",
    )
    started = time.monotonic()
    charged_episodes = 0
    final_metrics: dict[str, float] = {}
    peak_memory = 0
    model.train()
    for update in range(1, arguments.updates + 1):
        episodes = [
            generate_episode(
                seed=arguments.data_seed + (update - 1) * arguments.batch_size + index,
                cohort="train",
            )
            for index in range(arguments.batch_size)
        ]
        optimizer.zero_grad(set_to_none=True)
        loss, final_metrics = _training_loss(model, episodes, device)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite WRA1 compiler loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        charged_episodes += len(episodes)
        if device.type == "cuda":
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))
        if update == 1 or update % arguments.log_every == 0:
            print(
                json.dumps(
                    {
                        "update": update,
                        "charged_episodes": charged_episodes,
                        "gradient_norm": float(gradient_norm),
                        **final_metrics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    training_seconds = time.monotonic() - started
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
        "support": min(row["support_recalled"] for row in rates.values()) >= 0.95,
        "exact_packet": min(row["exact_packet"] for row in rates.values()) >= 0.95,
        "no_duplicates": max(row["accepted_duplicate_fields"] for row in rates.values())
        == 0.0,
        "no_overflow": max(row["overflow"] for row in rates.values()) == 0.0,
        "source_poison": min(row["source_poison_invariant"] for row in rates.values())
        == 1.0,
        "lineage_causality": min(
            row["exact_packet"] - row["lineage_shuffle_exact"] for row in rates.values()
        )
        >= 0.20,
        "sc1_shift_advantage": min(
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
            "updates": arguments.updates,
            "charged_episodes": charged_episodes,
            "seconds": training_seconds,
            "episodes_per_second": charged_episodes / training_seconds,
            "peak_memory_bytes": peak_memory,
            "final_metrics": final_metrics,
        },
        "parameters": {
            "complete": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(parameter.numel() for parameter in parameters),
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
