#!/usr/bin/env python3
"""Train and gate the frozen DIVERGE-JET1 joint epistemic trajectory."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from diverge_jet1_data import (
    MAX_PROGRAM_ACTIONS,
    PROGRAM_ACTIONS,
    JET1Episode,
    generate_jet1_episode,
    renderer_parity,
)
from diverge_jet1_runtime import (
    FIELD_COUNT,
    REGISTER_COUNT,
    VALUE_COUNT,
    JET1Config,
    JET1Output,
    JointEpistemicTrajectory,
    architecture_receipt,
)
from diverge_mei1_data import EVIDENCE_COHORTS, exact_action_batch
from hf_product_reasoning_train import (
    install_lora,
    load_product_backbone,
    resolve_product_backbone_layout,
)


SCHEMA = "shohin-diverge-jet1-joint-training-v1"
SEED = 202608058800
EVIDENCE_PREFIX = (
    "Read this delayed five-register audit. Recover every value before and after: "
)
COHORT_OFFSETS = {
    "train": 0,
    "lexical_shift": 100_000_000,
    "renderer_shift": 200_000_000,
    "composition_shift": 300_000_000,
}
EVAL_DEPTHS = (4, 8, 16, 24)


class JET1TrainingError(RuntimeError):
    pass


@dataclass(frozen=True)
class JET1Batch:
    words: tuple[tuple[str, ...], ...]
    batch_size: int
    depth: int
    initial_values: torch.Tensor
    evidence_targets: torch.Tensor
    program_actions: torch.Tensor
    program_action_mask: torch.Tensor
    prior_logits: torch.Tensor
    gold_candidates: torch.Tensor
    terminal_values: torch.Tensor
    query_slots: torch.Tensor
    answers: torch.Tensor


class QwenJET1(nn.Module):
    """Pinned Qwen text path plus one jointly optimized JET1 trajectory."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        lora_layers: int,
        lora_rank: int,
        lora_alpha: float,
        trajectory_config: JET1Config,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        (
            self.text_model,
            self.lm_head,
            hidden_size,
            self.backbone_layout,
        ) = resolve_product_backbone_layout(backbone)
        if hidden_size != trajectory_config.input_width:
            raise JET1TrainingError("JET1/Qwen hidden width differs")
        if not 0 < lora_layers <= len(self.text_model.layers):
            raise JET1TrainingError("JET1 LoRA layer count differs")
        self.lora_projection_count = 0
        for layer in self.text_model.layers[-lora_layers:]:
            self.lora_projection_count += install_lora(
                layer, lora_rank, lora_alpha
            )
        if self.lora_projection_count == 0:
            raise JET1TrainingError("JET1 installed no LoRA projections")
        self.trajectory = JointEpistemicTrajectory(trajectory_config)

    def lora_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for name, parameter in self.backbone.named_parameters()
            if parameter.requires_grad and ("lora_a" in name or "lora_b" in name)
        ]

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def encode_evidence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_sha256(
    module: nn.Module,
    *,
    exclude_lora: bool = False,
) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if exclude_lora and ("lora_a" in name or "lora_b" in name):
            continue
        raw = tensor.detach().cpu().contiguous()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name.encode("utf-8"))
        digest.update(str(raw.dtype).encode("ascii"))
        digest.update(str(tuple(raw.shape)).encode("ascii"))
        digest.update(raw.view(torch.uint8).numpy().tobytes())
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


def tensorize_episodes(
    episodes: list[JET1Episode], device: torch.device
) -> JET1Batch:
    if not episodes or len({episode.depth for episode in episodes}) != 1:
        raise JET1TrainingError("JET1 batch must have one positive depth")
    batch_size = len(episodes)
    depth = episodes[0].depth
    words: list[tuple[str, ...]] = []
    initial = torch.empty(batch_size, REGISTER_COUNT, dtype=torch.long)
    evidence = torch.empty(batch_size, depth, FIELD_COUNT, dtype=torch.long)
    actions = torch.zeros(
        batch_size,
        depth,
        2,
        MAX_PROGRAM_ACTIONS,
        dtype=torch.long,
    )
    action_mask = torch.zeros_like(actions, dtype=torch.bool)
    priors = torch.empty(batch_size, depth, 2, dtype=torch.float32)
    gold = torch.empty(batch_size, depth, dtype=torch.long)
    terminal = torch.empty(batch_size, REGISTER_COUNT, dtype=torch.long)
    query = torch.empty(batch_size, dtype=torch.long)
    answer = torch.empty(batch_size, dtype=torch.long)
    for row, episode in enumerate(episodes):
        initial[row] = torch.tensor(episode.initial_state)
        terminal[row] = torch.tensor(episode.terminal_state)
        query[row] = episode.query_slot
        answer[row] = episode.answer
        for step_index, step in enumerate(episode.steps):
            words.append(step.words)
            evidence[row, step_index] = torch.tensor((*step.before, *step.after))
            priors[row, step_index] = torch.tensor(step.prior_logits)
            gold[row, step_index] = step.gold_candidate
            for candidate, sequence in enumerate(step.candidate_actions):
                actions[row, step_index, candidate, : len(sequence)] = torch.tensor(
                    sequence
                )
                action_mask[row, step_index, candidate, : len(sequence)] = True
    return JET1Batch(
        words=tuple(words),
        batch_size=batch_size,
        depth=depth,
        initial_values=initial.to(device),
        evidence_targets=evidence.to(device),
        program_actions=actions.to(device),
        program_action_mask=action_mask.to(device),
        prior_logits=priors.to(device),
        gold_candidates=gold.to(device),
        terminal_values=terminal.to(device),
        query_slots=query.to(device),
        answers=answer.to(device),
    )


def tokenize_evidence(
    tokenizer: Any,
    words: tuple[tuple[str, ...], ...],
    *,
    maximum_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    texts = [EVIDENCE_PREFIX + " ".join(row) for row in words]
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    mask = encoded["attention_mask"]
    longest = int(mask.sum(-1).max())
    if longest > maximum_tokens:
        raise JET1TrainingError(
            f"JET1 evidence exceeds {maximum_tokens} tokens: {longest}"
        )
    return (
        encoded["input_ids"].to(device),
        mask.to(device),
        int(mask.sum()),
    )


def forward_batch(
    model: QwenJET1,
    tokenizer: Any,
    batch: JET1Batch,
    *,
    maximum_tokens: int,
    hard_forward: bool,
) -> tuple[JET1Output, int, torch.Tensor, torch.Tensor]:
    input_ids, attention_mask, charged_tokens = tokenize_evidence(
        tokenizer,
        batch.words,
        maximum_tokens=maximum_tokens,
        device=batch.initial_values.device,
    )
    features = model.encode_evidence(input_ids, attention_mask)
    token_width = features.shape[1]
    features = features.reshape(
        batch.batch_size,
        batch.depth,
        token_width,
        features.shape[-1],
    )
    mask = attention_mask.bool().reshape(
        batch.batch_size, batch.depth, token_width
    )
    output = model.trajectory(
        features,
        mask,
        batch.initial_values,
        batch.program_actions,
        batch.program_action_mask,
        batch.prior_logits,
        batch.query_slots,
        hard_forward=hard_forward,
    )
    return output, charged_tokens, features, mask


def probability_nll(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    selected = probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return -(selected + 1e-6).log().mean()


def operator_examples(
    *, seed: int, count: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if count < len(PROGRAM_ACTIONS):
        raise JET1TrainingError("JET1 operator batch is not action-complete")
    rng = random.Random(seed)
    states = [
        tuple(rng.randrange(4, 121) for _ in range(REGISTER_COUNT))
        for _ in range(count)
    ]
    actions = tuple(index % len(PROGRAM_ACTIONS) for index in range(count))
    targets = exact_action_batch(tuple(states), actions)
    return (
        torch.tensor(states, dtype=torch.long, device=device),
        torch.tensor(actions, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def training_loss(
    model: QwenJET1,
    tokenizer: Any,
    batch: JET1Batch,
    *,
    maximum_tokens: int,
    operator_seed: int,
) -> tuple[torch.Tensor, dict[str, float], int]:
    output, charged_tokens, _, _ = forward_batch(
        model,
        tokenizer,
        batch,
        maximum_tokens=maximum_tokens,
        hard_forward=True,
    )
    evidence_loss = probability_nll(
        output.evidence_probabilities, batch.evidence_targets
    )
    choice_loss = F.cross_entropy(
        output.choice_logits.reshape(-1, 2), batch.gold_candidates.reshape(-1)
    )
    terminal_loss = probability_nll(
        output.terminal_probabilities, batch.terminal_values
    )
    answer_loss = probability_nll(output.answer_probabilities, batch.answers)
    operator_state, operator_action, operator_target = operator_examples(
        seed=operator_seed,
        count=batch.batch_size * len(PROGRAM_ACTIONS),
        device=batch.initial_values.device,
    )
    operator_probabilities, operator_invalid = model.trajectory.executor.step(
        F.one_hot(operator_state, VALUE_COUNT).float(),
        operator_action,
        hard_forward=True,
    )
    operator_loss = probability_nll(operator_probabilities, operator_target)
    loss = (
        evidence_loss
        + choice_loss
        + terminal_loss
        + answer_loss
        + operator_loss
    )
    with torch.no_grad():
        evidence_exact = output.evidence_probabilities.argmax(-1).eq(
            batch.evidence_targets
        ).all(-1).float().mean()
        choice_exact = output.selected_candidates.eq(
            batch.gold_candidates
        ).all(-1).float().mean()
        terminal_exact = output.terminal_probabilities.argmax(-1).eq(
            batch.terminal_values
        ).all(-1).float().mean()
        answer_exact = output.answer_probabilities.argmax(-1).eq(
            batch.answers
        ).float().mean()
    metrics = {
        "loss": float(loss.detach()),
        "evidence_loss": float(evidence_loss.detach()),
        "choice_loss": float(choice_loss.detach()),
        "terminal_loss": float(terminal_loss.detach()),
        "answer_loss": float(answer_loss.detach()),
        "operator_loss": float(operator_loss.detach()),
        "evidence_pair_exact": float(evidence_exact),
        "choice_sequence_exact": float(choice_exact),
        "terminal_exact": float(terminal_exact),
        "answer_exact": float(answer_exact),
        "invalid_mass": float(
            torch.maximum(output.invalid_mass.max(), operator_invalid.max()).detach()
        ),
    }
    return loss, metrics, charged_tokens


def cosine_scale(update: int, updates: int, warmup: int) -> float:
    if not 1 <= update <= updates or not 0 < warmup < updates:
        raise JET1TrainingError("JET1 LR schedule differs")
    if update <= warmup:
        return update / warmup
    progress = (update - warmup) / (updates - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate_cell(
    model: QwenJET1,
    tokenizer: Any,
    *,
    cohort: str,
    depth: int,
    count: int,
    batch_size: int,
    seed: int,
    maximum_tokens: int,
    device: torch.device,
) -> dict[str, float | int]:
    totals = {
        "episodes": 0,
        "records": 0,
        "evidence_pairs": 0,
        "choice_sequences": 0,
        "terminal_states": 0,
        "answers": 0,
        "shuffle_answers": 0,
        "reset_answers": 0,
        "invalid_episodes": 0,
        "encoded_tokens": 0,
    }
    for start in range(0, count, batch_size):
        episodes = [
            generate_jet1_episode(
                seed=seed + index,
                cohort=cohort,
                depth=depth,
            )
            for index in range(start, min(start + batch_size, count))
        ]
        batch = tensorize_episodes(episodes, device)
        output, tokens, features, mask = forward_batch(
            model,
            tokenizer,
            batch,
            maximum_tokens=maximum_tokens,
            hard_forward=True,
        )
        if batch.batch_size < 2:
            raise JET1TrainingError("JET1 shuffle control needs two episodes")
        shuffled = model.trajectory(
            features.roll(1, dims=0),
            mask.roll(1, dims=0),
            batch.initial_values,
            batch.program_actions,
            batch.program_action_mask,
            batch.prior_logits,
            batch.query_slots,
            hard_forward=True,
        )
        reset = model.trajectory.read_query(
            F.one_hot(batch.initial_values, VALUE_COUNT).float(),
            batch.query_slots,
            hard_forward=True,
        )
        evidence_ok = output.evidence_probabilities.argmax(-1).eq(
            batch.evidence_targets
        ).all(-1)
        totals["episodes"] += batch.batch_size
        totals["records"] += batch.batch_size * depth
        totals["evidence_pairs"] += int(evidence_ok.sum())
        totals["choice_sequences"] += int(
            output.selected_candidates.eq(batch.gold_candidates).all(-1).sum()
        )
        totals["terminal_states"] += int(
            output.terminal_probabilities.argmax(-1)
            .eq(batch.terminal_values)
            .all(-1)
            .sum()
        )
        totals["answers"] += int(
            output.answer_probabilities.argmax(-1).eq(batch.answers).sum()
        )
        totals["shuffle_answers"] += int(
            shuffled.answer_probabilities.argmax(-1).eq(batch.answers).sum()
        )
        totals["reset_answers"] += int(reset.argmax(-1).eq(batch.answers).sum())
        totals["invalid_episodes"] += int(
            output.invalid_mass.gt(0).any(-1).sum()
        )
        totals["encoded_tokens"] += tokens
    episodes = totals["episodes"]
    records = totals["records"]
    return {
        **totals,
        "evidence_pair_exact": totals["evidence_pairs"] / records,
        "choice_sequence_exact": totals["choice_sequences"] / episodes,
        "terminal_state_exact": totals["terminal_states"] / episodes,
        "answer_exact": totals["answers"] / episodes,
        "wrong_prior_recovery": totals["answers"] / episodes,
        "evidence_shuffle_answer_exact": totals["shuffle_answers"] / episodes,
        "state_reset_answer_exact": totals["reset_answers"] / episodes,
    }


@torch.no_grad()
def evaluate_primitive(
    model: QwenJET1,
    *,
    count: int,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    complete = 0
    invalid = 0
    for start in range(0, count, batch_size):
        size = min(batch_size, count - start)
        states, actions, targets = operator_examples(
            seed=seed + start,
            count=size,
            device=device,
        )
        probabilities, invalid_mass = model.trajectory.executor.step(
            F.one_hot(states, VALUE_COUNT).float(),
            actions,
            hard_forward=True,
        )
        complete += int(probabilities.argmax(-1).eq(targets).all(-1).sum())
        invalid += int(invalid_mass.gt(0).any(-1).sum())
    return {
        "examples": count,
        "complete_state_exact": complete / count,
        "invalid_examples": invalid,
    }


def build_gate(
    cells: dict[str, dict[str, float | int]],
    primitive: dict[str, float | int],
    *,
    frozen_before: str,
    frozen_after: str,
    source_audit_pass: bool,
) -> dict[str, bool]:
    shifted = [
        row
        for key, row in cells.items()
        if not key.startswith("train/depth")
    ]
    gate = {
        "primitive_99_9pct": primitive["complete_state_exact"] >= 0.999,
        "evidence_pair_95pct_every_cell": min(
            row["evidence_pair_exact"] for row in cells.values()
        )
        >= 0.95,
        "choice_sequence_90pct_every_cell": min(
            row["choice_sequence_exact"] for row in cells.values()
        )
        >= 0.90,
        "terminal_state_90pct_every_cell": min(
            row["terminal_state_exact"] for row in cells.values()
        )
        >= 0.90,
        "answer_90pct_every_cell": min(
            row["answer_exact"] for row in cells.values()
        )
        >= 0.90,
        "wrong_prior_recovery_90pct_every_cell": min(
            row["wrong_prior_recovery"] for row in cells.values()
        )
        >= 0.90,
        "evidence_shuffle_drops_shifted_20pp_every_cell": min(
            row["answer_exact"] - row["evidence_shuffle_answer_exact"]
            for row in shifted
        )
        >= 0.20,
        "state_reset_drops_shifted_20pp_every_cell": min(
            row["answer_exact"] - row["state_reset_answer_exact"]
            for row in shifted
        )
        >= 0.20,
        "zero_invalid_acceptance": (
            primitive["invalid_examples"] == 0
            and sum(row["invalid_episodes"] for row in cells.values()) == 0
        ),
        "frozen_non_lora_backbone_unchanged": frozen_before == frozen_after,
        "candidate_source_audit": source_audit_pass,
    }
    gate["pass"] = all(gate.values())
    return gate


def _trainable_state(model: QwenJET1) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    from transformers import AutoTokenizer

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_root, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone, loader = load_product_backbone(
        args.model_root,
        args.model_loader,
        dtype=torch.bfloat16,
        device_map={"": device.index or 0},
    )
    trajectory_config = JET1Config(
        input_width=args.input_width,
        reader_width=args.reader_width,
        reader_heads=args.reader_heads,
        reader_layers=args.reader_layers,
    )
    model = QwenJET1(
        backbone,
        lora_layers=args.lora_layers,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        trajectory_config=trajectory_config,
    ).to(device)
    lora_parameters = model.lora_parameters()
    trajectory_parameters = list(model.trajectory.parameters())
    if not lora_parameters or not trajectory_parameters:
        raise JET1TrainingError("JET1 trainable groups are empty")
    unexpected = [
        name
        for name, parameter in model.backbone.named_parameters()
        if parameter.requires_grad and "lora_a" not in name and "lora_b" not in name
    ]
    if unexpected:
        raise JET1TrainingError(f"unexpected trainable backbone tensors: {unexpected[:3]}")
    frozen_before = _state_sha256(model.backbone, exclude_lora=True)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_parameters,
                "lr": args.lora_lr,
                "base_lr": args.lora_lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": trajectory_parameters,
                "lr": args.trajectory_lr,
                "base_lr": args.trajectory_lr,
                "weight_decay": args.weight_decay,
            },
        ],
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    parity = renderer_parity(seed=args.seed + 17, count=args.parity_count)
    if not parity["pass"]:
        raise JET1TrainingError("JET1 renderer parity failed")
    started = time.time()
    charged_tokens = 0
    records = 0
    last_metrics: dict[str, float] = {}
    model.train()
    for update in range(1, args.updates + 1):
        depth = (update - 1) % 8 + 1
        episode_seed = args.seed + update * 1_000_003
        episodes = [
            generate_jet1_episode(
                seed=episode_seed + index,
                cohort="train",
                depth=depth,
            )
            for index in range(args.batch_size)
        ]
        batch = tensorize_episodes(episodes, device)
        scale = cosine_scale(update, args.updates, args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * scale
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            loss, last_metrics, tokens = training_loss(
                model,
                tokenizer,
                batch,
                maximum_tokens=args.maximum_tokens,
                operator_seed=args.seed + 900_000_000 + update,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("JET1 training loss became non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [*lora_parameters, *trajectory_parameters], args.grad_clip
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("JET1 gradient norm became non-finite")
        optimizer.step()
        charged_tokens += tokens
        records += args.batch_size * depth
        last_metrics = {
            **last_metrics,
            "gradient_norm": float(gradient_norm),
            "lr_scale": scale,
            "depth": float(depth),
        }
        if update == 1 or update % args.log_every == 0 or update == args.updates:
            print(
                json.dumps(
                    {
                        "update": update,
                        "records": records,
                        "encoded_tokens": charged_tokens,
                        **last_metrics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    model.eval()
    primitive = evaluate_primitive(
        model,
        count=args.primitive_count,
        seed=args.seed + 700_000_000,
        batch_size=args.primitive_batch_size,
        device=device,
    )
    cells: dict[str, dict[str, float | int]] = {}
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        for cohort in EVIDENCE_COHORTS:
            for depth in EVAL_DEPTHS:
                key = f"{cohort}/depth{depth}"
                cells[key] = evaluate_cell(
                    model,
                    tokenizer,
                    cohort=cohort,
                    depth=depth,
                    count=args.eval_count,
                    batch_size=args.eval_batch_size,
                    seed=(
                        args.seed
                        + 500_000_000
                        + COHORT_OFFSETS[cohort]
                        + depth * 1_000_000
                    ),
                    maximum_tokens=args.maximum_tokens,
                    device=device,
                )
                print(json.dumps({"cell": key, **cells[key]}, sort_keys=True), flush=True)
    frozen_after = _state_sha256(model.backbone, exclude_lora=True)
    architecture = architecture_receipt(model.trajectory)
    gate = build_gate(
        cells,
        primitive,
        frozen_before=frozen_before,
        frozen_after=frozen_after,
        source_audit_pass=bool(architecture["source_audit"]["pass"]),
    )
    trainable_state = _trainable_state(model)
    elapsed = time.time() - started
    report = {
        "schema": SCHEMA,
        "status": "bounded-joint-epistemic-trajectory-complete",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "architecture": {
            **architecture,
            "backbone_layout": model.backbone_layout,
            "model_loader": loader,
            "lora_projection_count": model.lora_projection_count,
            "lora_trainable_parameters": sum(
                parameter.numel() for parameter in lora_parameters
            ),
            "trajectory_trainable_parameters": sum(
                parameter.numel() for parameter in trajectory_parameters
            ),
            "total_trainable_parameters": model.trainable_parameter_count(),
        },
        "inputs": {
            "model_root": str(args.model_root),
            "config_sha256": _sha256(args.model_root / "config.json"),
        },
        "renderer_parity": parity,
        "training": {
            "updates": args.updates,
            "episodes": args.updates * args.batch_size,
            "source_records": records,
            "encoded_tokens": charged_tokens,
            "elapsed_seconds": elapsed,
            "records_per_second": records / elapsed,
            "encoded_tokens_per_second": charged_tokens / elapsed,
            "last_batch": last_metrics,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated())
            if device.type == "cuda"
            else 0,
            "trainable_state_sha256": _state_sha256_from_dict(trainable_state),
        },
        "primitive": primitive,
        "evaluations": cells,
        "frozen_non_lora_backbone_before": frozen_before,
        "frozen_non_lora_backbone_after": frozen_after,
        "gate": gate,
        "claim_boundary": (
            "Synthetic joint source-state-execution-query gate only. A pass authorizes "
            "one HSC1 integration and one matched dense control; it is not a public "
            "reasoning or continuation-pretraining result."
        ),
    }
    checkpoint = {
        "schema": SCHEMA,
        "trajectory_config": asdict(trajectory_config),
        "trainable_state_dict": trainable_state,
        "report": report,
    }
    _atomic_torch(args.output, checkpoint)
    _atomic_json(args.report, report)
    print(
        json.dumps(
            {
                "checkpoint": str(args.output),
                "checkpoint_sha256": _sha256(args.output),
                "report": str(args.report),
                "report_sha256": _sha256(args.report),
                "gate": gate,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def _state_sha256_from_dict(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        raw = tensor.detach().cpu().contiguous()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name.encode("utf-8"))
        digest.update(str(raw.dtype).encode("ascii"))
        digest.update(str(tuple(raw.shape)).encode("ascii"))
        digest.update(raw.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-loader", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--updates", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-count", type=int, default=512)
    parser.add_argument("--primitive-count", type=int, default=20000)
    parser.add_argument("--primitive-batch-size", type=int, default=512)
    parser.add_argument("--parity-count", type=int, default=1000)
    parser.add_argument("--maximum-tokens", type=int, default=128)
    parser.add_argument("--input-width", type=int, default=1024)
    parser.add_argument("--reader-width", type=int, default=256)
    parser.add_argument("--reader-heads", type=int, default=8)
    parser.add_argument("--reader-layers", type=int, default=1)
    parser.add_argument("--lora-layers", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-lr", type=float, default=1e-5)
    parser.add_argument("--trajectory-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=80)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite JET1 output")
    if not (args.model_root / "config.json").is_file():
        raise FileNotFoundError("JET1 model root lacks config.json")
    if args.eval_batch_size < 2 or args.eval_count % args.eval_batch_size:
        raise ValueError("JET1 evaluation must use complete shuffle-control batches")
    if not args.smoke:
        observed = (
            args.seed,
            args.updates,
            args.batch_size,
            args.eval_count,
            args.primitive_count,
            args.parity_count,
            args.maximum_tokens,
            args.input_width,
            args.reader_width,
            args.reader_heads,
            args.reader_layers,
            args.lora_layers,
            args.lora_rank,
            args.lora_alpha,
            args.lora_lr,
            args.trajectory_lr,
            args.weight_decay,
            args.grad_clip,
            args.warmup,
        )
        expected = (
            SEED,
            1600,
            8,
            512,
            20000,
            1000,
            128,
            1024,
            256,
            8,
            1,
            4,
            8,
            16.0,
            1e-5,
            3e-4,
            0.01,
            1.0,
            80,
        )
        if observed != expected:
            raise ValueError("JET1 scientific contract differs")
    elif not 0 < args.warmup < args.updates:
        raise ValueError("JET1 smoke LR schedule differs")
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    report = run(args)
    if not args.smoke and not report["gate"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
