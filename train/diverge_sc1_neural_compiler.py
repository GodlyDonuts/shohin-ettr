#!/usr/bin/env python3
"""Train the bounded DIVERGE-SC1 raw-source compiler on frozen SmolLM2."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
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

from diverge_sc1_source_compiler import (
    ALIAS_BEGIN,
    ALIAS_INSIDE,
    BACKGROUND_CUE,
    CANDIDATE_CUE,
    CompilerScores,
    OTHER,
    PRIOR_FAVORED,
    PROGRAM_ROLES,
    ROLE_COUNT,
    RawSourceEpisode,
    _fused_occurrence_exact,
    _shuffle_boundaries,
    _zero_pairs,
    alpha_rename_episode,
    decode_independent,
    decode_joint,
    exact,
    generate_episode,
    seal_source,
)
from frozen_pointer_backbone import load_frozen_pointer_backbone


SCHEMA = "shohin-diverge-sc1-neural-raw-source-compiler-v1"
EXPECTED_WARM_SHA256 = "d614690f6446bf1635fc474d4ae941677b01ac465c0c60381c4b11bbe189826f"
ROLE_SCALE = 4.0
BOUNDARY_SCALE = 4.0
PAIR_SCALE = 8.0


@dataclass(frozen=True, slots=True)
class SourceEncoding:
    text: str
    ids: tuple[int, ...]
    token_to_word: tuple[int, ...]
    word_count: int


@dataclass(frozen=True, slots=True)
class CompilerOutput:
    role: torch.Tensor
    boundary: torch.Tensor
    pair: torch.Tensor
    word_lengths: torch.Tensor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _word_character_spans(tokens: Sequence[str]) -> tuple[str, tuple[tuple[int, int], ...]]:
    pieces = []
    spans = []
    cursor = 0
    for index, token in enumerate(tokens):
        if index:
            pieces.append(" ")
            cursor += 1
        start = cursor
        pieces.append(token)
        cursor += len(token)
        spans.append((start, cursor))
    return "".join(pieces), tuple(spans)


def encode_source(tokenizer: Tokenizer, tokens: Sequence[str]) -> SourceEncoding:
    text, word_spans = _word_character_spans(tokens)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if not encoded.ids:
        raise ValueError("raw source tokenized to an empty sequence")
    token_to_word = []
    represented = set()
    for token_start, token_end in encoded.offsets:
        overlaps = [
            (min(token_end, word_end) - max(token_start, word_start), word)
            for word, (word_start, word_end) in enumerate(word_spans)
            if token_end > word_start and token_start < word_end
        ]
        if not overlaps:
            token_to_word.append(-1)
            continue
        _, word = max(overlaps)
        token_to_word.append(word)
        represented.add(word)
    if represented != set(range(len(tokens))):
        raise ValueError("tokenizer failed to represent every raw-source word")
    return SourceEncoding(
        text,
        tuple(encoded.ids),
        tuple(token_to_word),
        len(tokens),
    )


def gold_role_targets(episode: RawSourceEpisode) -> tuple[int, ...]:
    target = [OTHER] * len(episode.tokens)

    def assign(position: int, role: int) -> None:
        if target[position] != OTHER and target[position] != role:
            raise ValueError("gold source roles overlap")
        target[position] = role

    for record in episode.records:
        assign(
            record.cue_position,
            CANDIDATE_CUE if record.is_fault_line else BACKGROUND_CUE,
        )
        for option in record.options:
            assign(option.alias_span[0], ALIAS_BEGIN)
            for position in range(option.alias_span[0] + 1, option.alias_span[1]):
                assign(position, ALIAS_INSIDE)
            assign(option.prior_position, PRIOR_FAVORED + option.prior_class)
            for position, role in zip(
                option.action_positions,
                PROGRAM_ROLES[option.program],
                strict=True,
            ):
                assign(position, role)
    return tuple(target)


def gold_boundaries(episode: RawSourceEpisode) -> tuple[int, ...]:
    target = [0] * (len(episode.tokens) + 1)
    for record in episode.records:
        target[record.start] = 1
        target[record.end] = 1
    return tuple(target)


def gold_pairs(episode: RawSourceEpisode) -> tuple[set[tuple[int, int]], set[int]]:
    positive: set[tuple[int, int]] = set()
    active = {position for position, _ in episode.decoy_roles}

    def add(left: int, right: int) -> None:
        positive.add(tuple(sorted((left, right))))
        active.update((left, right))

    for record in episode.records:
        active.add(record.cue_position)
        for option in record.options:
            active.update(range(option.alias_span[0], option.alias_span[1]))
            active.add(option.prior_position)
            active.update(option.action_positions)
            for position in (option.prior_position, *option.action_positions):
                add(option.alias_span[0], position)
            add(record.cue_position, option.alias_span[0])
        add(record.options[0].alias_span[0], record.options[1].alias_span[0])
    return positive, active


class RawSourceCompiler(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        tokenizer: Tokenizer,
        *,
        layer: int,
        width: int,
        pair_width: int,
    ) -> None:
        super().__init__()
        if not 0 <= layer < len(backbone.blocks) or width % 4 or pair_width <= 0:
            raise ValueError("invalid raw-source compiler geometry")
        self.backbone = backbone.requires_grad_(False)
        self.tokenizer = tokenizer
        self.layer = layer
        self.memory_norm = nn.LayerNorm(backbone.cfg.d_model)
        self.memory_projection = nn.Linear(backbone.cfg.d_model, width, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(
            width,
            4,
            4 * width,
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
        self.role_head = nn.Linear(width, ROLE_COUNT)
        self.boundary_bos = nn.Parameter(torch.zeros(width))
        self.boundary_eos = nn.Parameter(torch.zeros(width))
        self.boundary_head = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.pair_query = nn.Linear(width, pair_width, bias=False)
        self.pair_key = nn.Linear(width, pair_width, bias=False)
        self.pair_norm = nn.LayerNorm(width)
        self.pair_width = pair_width

    def load_warm_encoder(self, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload["state_dict"]
        prefixes = ("memory_norm.", "memory_projection.", "memory_encoder.")
        selected = {key: value for key, value in state.items() if key.startswith(prefixes)}
        missing, unexpected = self.load_state_dict(selected, strict=False)
        expected_missing = {
            key
            for key in self.state_dict()
            if not key.startswith("backbone.") and not key.startswith(prefixes)
        }
        actual_missing = {key for key in missing if not key.startswith("backbone.")}
        if actual_missing != expected_missing or unexpected:
            raise ValueError("warm raw-source encoder state differs")

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

    def _encode_words(
        self,
        encodings: Sequence[SourceEncoding],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_lengths = torch.tensor(
            [len(row.ids) for row in encodings], dtype=torch.long, device=device
        )
        maximum_tokens = int(token_lengths.max().item())
        if maximum_tokens > self.backbone.cfg.seq_len:
            raise ValueError("raw source exceeds frozen backbone context")
        ids = torch.zeros(len(encodings), maximum_tokens, dtype=torch.long, device=device)
        for row, encoding in enumerate(encodings):
            ids[row, : len(encoding.ids)] = torch.tensor(encoding.ids, device=device)
        self.backbone.eval()
        with torch.no_grad():
            hidden = self.backbone.tok(ids)
            cosine = self.backbone.cos[:maximum_tokens].to(hidden.device)
            sine = self.backbone.sin[:maximum_tokens].to(hidden.device)
            for block in self.backbone.blocks[: self.layer + 1]:
                hidden, _ = block(hidden, cosine, sine)
        hidden = self.memory_projection(
            self.memory_norm(hidden.detach().to(self.memory_projection.weight.dtype))
        )
        word_lengths = torch.tensor(
            [row.word_count for row in encodings], dtype=torch.long, device=device
        )
        maximum_words = int(word_lengths.max().item())
        words = torch.zeros(
            len(encodings), maximum_words, hidden.shape[-1],
            dtype=hidden.dtype, device=device,
        )
        counts = torch.zeros(
            len(encodings), maximum_words, 1, dtype=hidden.dtype, device=device
        )
        for row, encoding in enumerate(encodings):
            mapping = torch.tensor(encoding.token_to_word, device=device)
            valid = mapping.ge(0)
            words[row].index_add_(0, mapping[valid], hidden[row, : len(mapping)][valid])
            counts[row].index_add_(
                0,
                mapping[valid],
                torch.ones(int(valid.sum().item()), 1, device=device, dtype=hidden.dtype),
            )
        words = words / counts.clamp_min(1)
        valid_words = (
            torch.arange(maximum_words, device=device)[None, :] < word_lengths[:, None]
        )
        words = self.memory_encoder(words, src_key_padding_mask=~valid_words)
        return words, word_lengths

    def forward(
        self,
        encodings: Sequence[SourceEncoding],
        device: torch.device,
    ) -> CompilerOutput:
        words, lengths = self._encode_words(encodings, device)
        role = self.role_head(words).float()
        batch, width, hidden = words.shape
        bos = self.boundary_bos.view(1, 1, hidden).expand(batch, 1, hidden)
        eos = self.boundary_eos.view(1, 1, hidden).expand(batch, 1, hidden)
        left = torch.cat((bos, words), dim=1)
        right = torch.cat((words, eos), dim=1)
        boundary = self.boundary_head(torch.cat((left, right), dim=-1)).squeeze(-1).float()
        normalized = self.pair_norm(words)
        query = self.pair_query(normalized)
        key = self.pair_key(normalized)
        pair = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.pair_width)
        pair = 0.5 * (pair + pair.transpose(-1, -2))
        return CompilerOutput(role, boundary, pair.float(), lengths)


def _training_loss(
    model: RawSourceCompiler,
    episodes: Sequence[RawSourceEpisode],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    encodings = [encode_source(model.tokenizer, episode.tokens) for episode in episodes]
    output = model(encodings, device)
    maximum_words = output.role.shape[1]
    role_target = torch.full(
        (len(episodes), maximum_words), -100, dtype=torch.long, device=device
    )
    maximum_gaps = output.boundary.shape[1]
    boundary_target = torch.zeros(
        len(episodes), maximum_gaps, dtype=torch.float32, device=device
    )
    boundary_mask = torch.zeros_like(boundary_target, dtype=torch.bool)
    pair_losses = []
    pair_correct = 0
    pair_count = 0
    for row, episode in enumerate(episodes):
        roles = gold_role_targets(episode)
        role_target[row, : len(roles)] = torch.tensor(roles, device=device)
        boundaries = gold_boundaries(episode)
        boundary_target[row, : len(boundaries)] = torch.tensor(
            boundaries, device=device, dtype=torch.float32
        )
        boundary_mask[row, : len(boundaries)] = True
        positives, active = gold_pairs(episode)
        active = sorted(active)
        pairs = [
            (left, right)
            for offset, left in enumerate(active)
            for right in active[offset + 1 :]
        ]
        if not pairs or not positives:
            raise ValueError("raw-source pair supervision is empty")
        pair_index = torch.tensor(pairs, device=device, dtype=torch.long)
        logits = output.pair[row, pair_index[:, 0], pair_index[:, 1]]
        targets = torch.tensor(
            [float(tuple(sorted(pair)) in positives) for pair in pairs],
            device=device,
        )
        positive_count = int(targets.sum().item())
        negative_count = len(pairs) - positive_count
        weight = torch.tensor(
            [max(1.0, negative_count / max(1, positive_count))], device=device
        )
        pair_losses.append(F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight))
        pair_correct += int((logits.gt(0) == targets.bool()).sum().item())
        pair_count += len(pairs)
    role_weight = torch.tensor([0.05] + [1.0] * (ROLE_COUNT - 1), device=device)
    role_loss = F.cross_entropy(
        output.role.reshape(-1, ROLE_COUNT), role_target.reshape(-1), weight=role_weight
    )
    active_boundary = output.boundary[boundary_mask]
    active_target = boundary_target[boundary_mask]
    positives = float(active_target.sum().item())
    negatives = float(active_target.numel() - positives)
    boundary_loss = F.binary_cross_entropy_with_logits(
        active_boundary,
        active_target,
        pos_weight=torch.tensor([max(1.0, negatives / max(1.0, positives))], device=device),
    )
    pair_loss = torch.stack(pair_losses).mean()
    loss = role_loss + boundary_loss + pair_loss
    with torch.no_grad():
        valid_roles = role_target.ne(-100)
        role_accuracy = (
            output.role.argmax(-1)[valid_roles].eq(role_target[valid_roles]).float().mean()
        )
        boundary_accuracy = active_boundary.gt(0).eq(active_target.bool()).float().mean()
    return loss, {
        "loss": float(loss.item()),
        "role_loss": float(role_loss.item()),
        "boundary_loss": float(boundary_loss.item()),
        "pair_loss": float(pair_loss.item()),
        "role_accuracy": float(role_accuracy.item()),
        "boundary_accuracy": float(boundary_accuracy.item()),
        "pair_accuracy": pair_correct / pair_count,
    }


def output_to_scores(
    output: CompilerOutput,
    row: int,
    length: int,
) -> CompilerScores:
    role_logits = output.role[row, :length].detach().cpu()
    role = tuple(tuple(float(value * ROLE_SCALE) for value in values) for values in role_logits)
    boundary = tuple(
        float(value * BOUNDARY_SCALE)
        for value in output.boundary[row, : length + 1].detach().cpu()
    )
    margin = role_logits[:, 1:].max(-1).values - role_logits[:, OTHER]
    active = margin.gt(0).nonzero().flatten().tolist()
    pair_matrix = output.pair[row, :length, :length].detach().cpu()
    pair = tuple(
        (left, right, float(pair_matrix[left, right].item() * PAIR_SCALE))
        for offset, left in enumerate(active)
        for right in active[offset + 1 :]
    )
    return CompilerScores(role, boundary, pair)


def _support_recalled(episode: RawSourceEpisode, receipt) -> bool:
    expected = {
        (
            record.start,
            record.end,
            tuple(option.alias_span for option in record.options),
        )
        for record in episode.records
        if record.is_fault_line
    }
    observed = {
        (
            record.start,
            record.end,
            tuple(option.alias_span for option in record.options),
        )
        for record in receipt.records
        if record.is_fault_line
    }
    return expected.issubset(observed)


def evaluate(
    model: RawSourceCompiler,
    *,
    cohort: str,
    count: int,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    totals = {
        "episodes": 0,
        "joint_exact": 0,
        "independent_exact": 0,
        "no_pair_exact": 0,
        "shuffled_boundary_exact": 0,
        "support_recalled": 0,
        "fused_occurrence_exact": 0,
        "alpha_rename_exact": 0,
        "source_poison_invariant": 0,
        "overflow": 0,
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, count, batch_size):
            episodes = [
                generate_episode(seed=seed + index, cohort=cohort)
                for index in range(start, min(count, start + batch_size))
            ]
            encodings = [encode_source(model.tokenizer, episode.tokens) for episode in episodes]
            output = model(encodings, device)
            renamed_episodes = [alpha_rename_episode(episode) for episode in episodes]
            renamed_output = model(
                [encode_source(model.tokenizer, episode.tokens) for episode in renamed_episodes],
                device,
            )
            for row, episode in enumerate(episodes):
                scores = output_to_scores(output, row, len(episode.tokens))
                joint = decode_joint(episode.tokens, scores)
                independent = decode_independent(episode.tokens, scores)
                no_pair = decode_joint(episode.tokens, _zero_pairs(scores))
                shuffled = decode_joint(
                    episode.tokens,
                    _shuffle_boundaries(scores, seed=seed * 31 + start + row),
                )
                packet = seal_source(episode.tokens, joint)
                renamed = renamed_episodes[row]
                renamed_scores = output_to_scores(
                    renamed_output, row, len(renamed.tokens)
                )
                renamed_joint = decode_joint(renamed.tokens, renamed_scores)
                poisoned = seal_source(tuple("poison" for _ in episode.tokens), joint)
                values = {
                    "episodes": 1,
                    "joint_exact": int(exact(episode, joint)),
                    "independent_exact": int(exact(episode, independent)),
                    "no_pair_exact": int(exact(episode, no_pair)),
                    "shuffled_boundary_exact": int(exact(episode, shuffled)),
                    "support_recalled": int(_support_recalled(episode, joint)),
                    "fused_occurrence_exact": int(
                        exact(episode, joint) and _fused_occurrence_exact(packet)
                    ),
                    "alpha_rename_exact": int(exact(renamed, renamed_joint)),
                    "source_poison_invariant": int(packet.records == poisoned.records),
                    "overflow": int(joint.overflow),
                }
                for key, value in values.items():
                    totals[key] += value
    episodes = totals["episodes"]
    return {
        "cohort": cohort,
        "count": count,
        "rates": {
            key: value / episodes for key, value in totals.items() if key != "episodes"
        },
        "totals": totals,
    }


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--warm-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202608056200)
    parser.add_argument("--data-seed", type=int, default=202608056300)
    parser.add_argument("--eval-seed", type=int, default=202608056400)
    parser.add_argument("--updates", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-count", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--pair-width", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    arguments = parser.parse_args()
    if arguments.updates <= 0 or arguments.batch_size <= 0 or arguments.eval_count <= 0:
        raise ValueError("training and evaluation sizes must be positive")
    if arguments.warm_checkpoint.exists() and sha256_file(arguments.warm_checkpoint) != EXPECTED_WARM_SHA256:
        raise ValueError("warm role-copy checkpoint hash differs")

    torch.set_num_threads(arguments.threads)
    torch.manual_seed(arguments.seed)
    random.seed(arguments.seed)
    device = torch.device(arguments.device)
    backbone, _, backbone_receipt = load_frozen_pointer_backbone(
        arguments.base, device=device
    )
    tokenizer = Tokenizer.from_file(str(arguments.tokenizer))
    model = RawSourceCompiler(
        backbone,
        tokenizer,
        layer=arguments.layer,
        width=arguments.width,
        pair_width=arguments.pair_width,
    ).to(device)
    model.load_warm_encoder(arguments.warm_checkpoint)
    parameters = list(model.adapter_parameters())
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
            raise RuntimeError("nonfinite raw-source compiler loss")
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
    state = model.adapter_state()
    state_hash = _state_sha256(state)
    checkpoint = arguments.output.with_suffix(".pt")
    checkpoint_payload = {
        "schema": SCHEMA,
        "arguments": vars(arguments),
        "state_dict": state,
        "model_state_sha256": state_hash,
    }
    _atomic_torch(checkpoint, checkpoint_payload)
    trainable = sum(parameter.numel() for parameter in parameters)
    complete = sum(parameter.numel() for parameter in model.parameters())
    shifted = [evaluations[name]["rates"] for name in evaluations if name != "train"]
    gates = {
        "development_support": evaluations["train"]["rates"]["support_recalled"] >= 0.99,
        "shift_support": min(row["support_recalled"] for row in shifted) >= 0.99,
        "development_packet": evaluations["train"]["rates"]["joint_exact"] >= 0.95,
        "shift_packet": min(row["joint_exact"] for row in shifted) >= 0.90,
        "independent_advantage": min(
            row["joint_exact"] - row["independent_exact"] for row in shifted
        ) >= 0.15,
        "source_poison": min(row["source_poison_invariant"] for row in shifted) == 1.0,
        "no_overflow": max(row["overflow"] for row in shifted) == 0.0,
    }
    report = {
        "schema": SCHEMA,
        "seed": arguments.seed,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()},
        "input_hashes": {
            "base": sha256_file(arguments.base),
            "tokenizer": sha256_file(arguments.tokenizer),
            "warm_checkpoint": sha256_file(arguments.warm_checkpoint),
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
        "parameters": {"complete": complete, "trainable": trainable},
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
