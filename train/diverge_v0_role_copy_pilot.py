#!/usr/bin/env python3
"""Token-role/source-copy compiler gate for the exact DIVERGE-v0 packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

import diverge_v0_neural_pilot as pilot
from diverge_v0 import account_packet, execute_packet
from frozen_pointer_backbone import load_frozen_pointer_backbone


SCHEMA = "shohin-diverge-v0-token-role-copy-component-pilot-v1"
OTHER = 0
CANDIDATE_CUE = 1
BACKGROUND_CUE = 2
RECORD_ROLE_COUNT = 3
PRIOR_FAVORED = 1
PRIOR_RESERVE = 2
ACTION_BASE = 3
OPTION_ROLE_COUNT = 7

ACTION_ROLE = {
    pilot.ACTION_TEXT["add"]: ACTION_BASE,
    pilot.ACTION_TEXT["swap01"]: ACTION_BASE + 1,
    pilot.ACTION_TEXT["swap23"]: ACTION_BASE + 2,
    pilot.ACTION_TEXT["swap34"]: ACTION_BASE + 3,
}
ACTION_SEQUENCE_TO_PROGRAM = {
    (0, 1): 0,
    (1, 0): 1,
    (2,): 2,
    (3,): 3,
}
BACKGROUND_CUES = (
    "background example",
    "descriptive archive only",
    "ignore this background record",
    "archive is descriptive only",
)


@dataclass(frozen=True)
class EncodedText:
    ids: list[int]
    offsets: tuple[tuple[int, int], ...]
    normalized: str


def _normalize(text: str, aliases: Iterable[str]) -> str:
    normalized = text.lower()
    for alias in sorted(aliases, key=len, reverse=True):
        normalized = normalized.replace(alias.lower(), "alias")
    return normalized


def _span_labels(
    offsets: Sequence[tuple[int, int]],
    spans: Sequence[tuple[int, int, int]],
) -> list[int]:
    labels = [OTHER] * len(offsets)
    for token, (start, end) in enumerate(offsets):
        if start == end:
            continue
        matches = [role for left, right, role in spans if start < right and end > left]
        if len(set(matches)) > 1:
            raise ValueError("overlapping semantic role spans")
        if matches:
            labels[token] = matches[0]
    return labels


def _find_once(text: str, phrase: str) -> tuple[int, int]:
    start = text.find(phrase)
    if start < 0 or text.find(phrase, start + 1) >= 0:
        raise ValueError(f"expected one semantic phrase: {phrase!r}")
    return start, start + len(phrase)


def record_role_labels(
    text: str,
    offsets: Sequence[tuple[int, int]],
    *,
    is_fault_line: bool,
) -> list[int]:
    if is_fault_line:
        phrase = "candidate alternatives"
        role = CANDIDATE_CUE
    else:
        matches = [phrase for phrase in BACKGROUND_CUES if phrase in text]
        if len(matches) != 1:
            raise ValueError("background record has no unique cue")
        phrase = matches[0]
        role = BACKGROUND_CUE
    left, right = _find_once(text, phrase)
    return _span_labels(offsets, ((left, right, role),))


def option_role_labels(
    option: pilot.OptionExample,
    text: str,
    offsets: Sequence[tuple[int, int]],
) -> list[int]:
    spans = []
    status = "favored" if option.prior_class == 0 else "reserve"
    left, right = _find_once(text, status)
    spans.append((left, right, PRIOR_FAVORED + option.prior_class))
    for action in pilot._program_actions(option.program):
        left, right = _find_once(text, action)
        spans.append((left, right, ACTION_ROLE[action]))
    return _span_labels(offsets, spans)


def _valid_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
    return torch.arange(width, device=lengths.device)[None, :] < lengths[:, None]


def decode_record_roles(logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    valid = _valid_mask(lengths, logits.shape[1])
    candidate = logits[..., CANDIDATE_CUE].masked_fill(~valid, -torch.inf)
    background = logits[..., BACKGROUND_CUE].masked_fill(~valid, -torch.inf)
    return candidate.logsumexp(-1) > background.logsumexp(-1)


def decode_option_roles(logits: torch.Tensor, lengths: torch.Tensor) -> tuple[list[int], list[int]]:
    programs = []
    priors = []
    for row, length in zip(logits, lengths.tolist(), strict=True):
        active = row[:length]
        prior_score = active[:, PRIOR_FAVORED : PRIOR_RESERVE + 1].amax(0)
        priors.append(int(prior_score.argmax().item()))
        predicted = active[:, ACTION_BASE:].argmax(-1)
        confidence = active[:, ACTION_BASE:].amax(-1) - active[:, :ACTION_BASE].amax(-1)
        sequence = []
        for role, confident in zip(predicted.tolist(), confidence.gt(0).tolist(), strict=True):
            if confident and (not sequence or role != sequence[-1]):
                sequence.append(role)
        compact = tuple(sequence)
        if compact in ACTION_SEQUENCE_TO_PROGRAM:
            programs.append(ACTION_SEQUENCE_TO_PROGRAM[compact])
            continue
        presence = active[:, ACTION_BASE:].amax(0)
        pair_score = torch.minimum(presence[0], presence[1])
        single_score, single = presence[2:].max(0)
        if pair_score > single_score:
            positions = torch.arange(length, device=active.device, dtype=active.dtype)
            add_position = (active[:, ACTION_BASE].softmax(0) * positions).sum()
            swap_position = (active[:, ACTION_BASE + 1].softmax(0) * positions).sum()
            programs.append(0 if add_position < swap_position else 1)
        else:
            programs.append(2 + int(single.item()))
    return programs, priors


class SmolDivergeRoleCopyCompiler(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        tokenizer: Tokenizer,
        *,
        layer: int,
        width: int,
        char_width: int,
    ):
        super().__init__()
        if not 0 <= layer < len(backbone.blocks) or width % 4:
            raise ValueError("invalid token-role compiler geometry")
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
        self.record_role_head = nn.Linear(width, RECORD_ROLE_COUNT)
        self.option_role_head = nn.Linear(width, OPTION_ROLE_COUNT)
        self.alias = pilot.SequenceEncoder(
            len(pilot.CHAR_TO_ID) + 2,
            char_width // 2,
            char_width // 2,
        )
        self.alias_projection = nn.Linear(char_width, char_width, bias=False)

    def encode(self, text: str, aliases: Iterable[str]) -> EncodedText:
        normalized = _normalize(text, aliases)
        encoded = self.tokenizer.encode(normalized, add_special_tokens=False)
        if not encoded.ids or len(encoded.ids) > self.backbone.cfg.seq_len:
            raise ValueError("invalid token-role compiler source length")
        return EncodedText(list(encoded.ids), tuple(encoded.offsets), normalized)

    def _memory(
        self,
        rows: list[list[int]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids, lengths = pilot._pad(rows, device)
        self.backbone.eval()
        with torch.no_grad():
            hidden = self.backbone.tok(ids)
            cosine = self.backbone.cos[: ids.shape[1]].to(hidden.device)
            sine = self.backbone.sin[: ids.shape[1]].to(hidden.device)
            for block in self.backbone.blocks[: self.layer + 1]:
                hidden, _ = block(hidden, cosine, sine)
        hidden = hidden.detach().to(self.memory_projection.weight.dtype)
        memory = self.memory_projection(self.memory_norm(hidden))
        valid = _valid_mask(lengths, ids.shape[1])
        memory = self.memory_encoder(memory, src_key_padding_mask=~valid)
        return memory, lengths

    def record_logits(
        self,
        rows: list[list[int]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory, lengths = self._memory(rows, device)
        return self.record_role_head(memory).float(), lengths

    def option_logits(
        self,
        rows: list[list[int]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory, lengths = self._memory(rows, device)
        return self.option_role_head(memory).float(), lengths

    def encode_aliases(self, rows: list[list[int]], device: torch.device) -> torch.Tensor:
        ids, lengths = pilot._pad(rows, device)
        return F.normalize(self.alias_projection(self.alias(ids, lengths)), dim=-1)

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


def _pad_labels(rows: list[list[int]], width: int, device: torch.device) -> torch.Tensor:
    target = torch.full((len(rows), width), -100, dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        target[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
    return target


def training_batch(
    episodes: list[pilot.PilotEpisode],
    model: SmolDivergeRoleCopyCompiler,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    record_encodings = []
    record_targets = []
    record_truth = []
    for episode in episodes:
        for record in episode.records:
            aliases = tuple(option.alias for option in record.options)
            for is_fault, text in (
                (record.is_fault_line, record.text),
                (
                    not record.is_fault_line,
                    pilot._render_record(
                        episode.ontology,
                        record.options[0],
                        record.options[1],
                        is_fault_line=not record.is_fault_line,
                        renderer=episode.renderer,
                    ),
                ),
            ):
                encoded = model.encode(text, aliases)
                record_encodings.append(encoded)
                record_targets.append(
                    record_role_labels(
                        encoded.normalized,
                        encoded.offsets,
                        is_fault_line=is_fault,
                    )
                )
                record_truth.append(is_fault)
    record_logits, record_lengths = model.record_logits(
        [row.ids for row in record_encodings], device
    )
    record_target = _pad_labels(record_targets, record_logits.shape[1], device)
    record_weight = torch.tensor((0.1, 1.0, 1.0), device=device)
    record_loss = F.cross_entropy(
        record_logits.reshape(-1, RECORD_ROLE_COUNT),
        record_target.reshape(-1),
        weight=record_weight,
    )

    options = [
        option
        for episode in episodes
        for record in episode.records
        if record.is_fault_line
        for option in record.options
    ]
    option_encodings = [model.encode(option.text, (option.alias,)) for option in options]
    option_targets = [
        option_role_labels(option, encoded.normalized, encoded.offsets)
        for option, encoded in zip(options, option_encodings, strict=True)
    ]
    option_logits, option_lengths = model.option_logits(
        [row.ids for row in option_encodings], device
    )
    option_target = _pad_labels(option_targets, option_logits.shape[1], device)
    option_weight = torch.tensor((0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0), device=device)
    option_loss = F.cross_entropy(
        option_logits.reshape(-1, OPTION_ROLE_COUNT),
        option_target.reshape(-1),
        weight=option_weight,
    )

    alias_losses = []
    alias_correct = 0
    for episode in episodes:
        source_aliases = [option.alias for record in episode.records for option in record.options]
        keys = model.encode_aliases([pilot.char_ids(value) for value in source_aliases], device)
        evidence = model.encode_aliases([pilot.char_ids(episode.evidence_alias)], device)
        logits = 16.0 * evidence @ keys.T
        target = torch.tensor([source_aliases.index(episode.evidence_alias)], device=device)
        alias_losses.append(F.cross_entropy(logits, target))
        alias_correct += int(logits.argmax(-1).item() == target.item())
    alias_loss = torch.stack(alias_losses).mean()
    loss = record_loss + option_loss + alias_loss

    with torch.no_grad():
        program_values, prior_values = decode_option_roles(option_logits, option_lengths)
        metrics = {
            "loss": float(loss.item()),
            "record_role_loss": float(record_loss.item()),
            "option_role_loss": float(option_loss.item()),
            "alias_loss": float(alias_loss.item()),
            "record_accuracy": float(
                (
                    decode_record_roles(record_logits, record_lengths)
                    == torch.tensor(record_truth, device=device)
                ).float().mean().item()
            ),
            "program_accuracy": sum(
                predicted == option.program
                for predicted, option in zip(program_values, options, strict=True)
            ) / len(options),
            "prior_accuracy": sum(
                predicted == option.prior_class
                for predicted, option in zip(prior_values, options, strict=True)
            ) / len(options),
            "alias_accuracy": alias_correct / len(episodes),
        }
    return loss, metrics


def predict_episode(
    model: SmolDivergeRoleCopyCompiler,
    episode: pilot.PilotEpisode,
    device: torch.device,
) -> pilot.CompilerPrediction:
    model.eval()
    with torch.no_grad():
        records = [
            model.encode(record.text, (option.alias for option in record.options))
            for record in episode.records
        ]
        record_logits, record_lengths = model.record_logits(
            [record.ids for record in records], device
        )
        selected = tuple(decode_record_roles(record_logits, record_lengths).tolist())
        options = [option for record in episode.records for option in record.options]
        encoded_options = [model.encode(option.text, (option.alias,)) for option in options]
        option_logits, option_lengths = model.option_logits(
            [option.ids for option in encoded_options], device
        )
        program_values, prior_values = decode_option_roles(option_logits, option_lengths)
        programs = tuple(
            tuple(program_values[index : index + 2])
            for index in range(0, len(program_values), 2)
        )
        priors = tuple(
            tuple(prior_values[index : index + 2])
            for index in range(0, len(prior_values), 2)
        )
        aliases = [option.alias for option in options]
        keys = model.encode_aliases([pilot.char_ids(value) for value in aliases], device)
        evidence = model.encode_aliases([pilot.char_ids(episode.evidence_alias)], device)
        selected_alias = int((evidence @ keys.T).argmax(-1).item())
    return pilot.CompilerPrediction(
        selected,
        programs,
        priors,
        selected_alias // 2,
        selected_alias % 2,
    )


def evaluate(
    model: SmolDivergeRoleCopyCompiler,
    *,
    split: str,
    count: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    width, renderer, ontology = {
        "development": (5, 2, "parcel-relation"),
        "confirmation": (6, 3, "signal-routing"),
    }[split]
    totals: dict[str, int] = {}
    packets = []
    for index in range(count):
        episode = pilot.generate_episode(
            seed=seed + index,
            split=split,
            width=width,
            renderer=renderer,
            ontology=ontology,
        )
        prediction = predict_episode(model, episode, device)
        row = pilot.score_episode(model, episode, device, prediction=prediction)
        for key, value in row.items():
            totals[key] = totals.get(key, 0) + value
        packet, _, _ = pilot._build_predicted_packet(episode, prediction)
        if packet is not None and not packet.overflow:
            packets.append(account_packet(packet, execute_packet(packet)))
    episodes = totals["episodes"]
    count_fields = {
        "episodes", "program_fields", "program_fields_correct", "prior_fields_correct",
        "fault_records", "fault_records_selected", "distractor_records",
        "distractor_records_selected",
    }
    rates = {
        key: value / episodes for key, value in totals.items() if key not in count_fields
    }
    rates["program_accuracy"] = totals["program_fields_correct"] / totals["program_fields"]
    rates["prior_accuracy"] = totals["prior_fields_correct"] / totals["program_fields"]
    rates["fault_line_recall"] = totals["fault_records_selected"] / totals["fault_records"]
    rates["distractor_false_positive_rate"] = (
        totals["distractor_records_selected"] / totals["distractor_records"]
    )
    return {
        "split": split,
        "count": count,
        "width": width,
        "renderer": renderer,
        "ontology": ontology,
        "rates": rates,
        "mean_packet_bytes": sum(row.packet_bytes for row in packets) / max(1, len(packets)),
        "mean_materialized_particle_bytes": (
            sum(row.materialized_world_bytes for row in packets) / max(1, len(packets))
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026080522)
    parser.add_argument("--data-seed", type=int, default=202608054000)
    parser.add_argument("--eval-seed", type=int, default=202608055000)
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-count", type=int, default=256)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--char-width", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.output.with_suffix(".pt").exists():
        raise FileExistsError(args.output)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    backbone, _, receipt = load_frozen_pointer_backbone(args.base, device=device)
    model = SmolDivergeRoleCopyCompiler(
        backbone,
        Tokenizer.from_file(str(args.tokenizer)),
        layer=args.layer,
        width=args.width,
        char_width=args.char_width,
    ).to(device)
    trainable = list(model.adapter_parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    started = time.perf_counter()
    log = []
    for update in range(1, args.updates + 1):
        model.train()
        episodes = [
            pilot.generate_episode(
                seed=args.data_seed + update * args.batch_size + index,
                split="train",
                width=1 + ((update + index) % 4),
                renderer=(update + index) % 2,
                ontology="register-workshop",
            )
            for index in range(args.batch_size)
        ]
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = training_batch(episodes, model, device)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite token-role loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if update == 1 or update % args.log_every == 0 or update == args.updates:
            row = {"update": update, **metrics, "grad_norm": float(grad_norm.item())}
            log.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    elapsed = time.perf_counter() - started
    evaluations = [
        evaluate(
            model,
            split=split,
            count=args.eval_count,
            seed=args.eval_seed + offset,
            device=device,
        )
        for split, offset in (("development", 0), ("confirmation", 100_000))
    ]
    state = model.adapter_state()
    state_hash = pilot._digest(
        "diverge-role-copy-state",
        {
            name: hashlib.sha256(value.numpy().tobytes()).hexdigest()
            for name, value in sorted(state.items())
        },
    )
    checkpoint = args.output.with_suffix(".pt")
    temporary = checkpoint.with_suffix(".pt.partial")
    torch.save(
        {
            "schema": SCHEMA,
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "state_dict": state,
            "model_state_sha256": state_hash,
        },
        temporary,
    )
    os.replace(temporary, checkpoint)
    report = {
        "schema": SCHEMA,
        "status": "component_pilot_only",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "training": {
            "updates": args.updates,
            "batch_size": args.batch_size,
            "charged_episodes": args.updates * args.batch_size,
            "elapsed_seconds": elapsed,
            "episodes_per_second": args.updates * args.batch_size / elapsed,
            "parameters": sum(parameter.numel() for parameter in trainable),
            "base_sha256": _sha256(args.base),
            "tokenizer_sha256": _sha256(args.tokenizer),
            "base_import": receipt.base_import,
            "train_log": log,
        },
        "evaluations": evaluations,
        "model_state_sha256": state_hash,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "claim_boundary": (
            "Token-role tagging and ordered source copy into the exact DIVERGE packet "
            "on syntactically scaffolded candidate options; not the frozen A-G promotion, "
            "unrestricted language compilation, or general reasoning."
        ),
    }
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "evaluations": evaluations}, sort_keys=True))


if __name__ == "__main__":
    main()
