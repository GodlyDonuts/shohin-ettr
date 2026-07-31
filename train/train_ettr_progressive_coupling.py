#!/usr/bin/env python3
"""Progressively couple trained ETTR components under self-generated state.

The component islands establish local learnability behind exact interfaces.
This trainer starts from those hash-bound components, keeps the protected
Shohin base frozen, and gradually replaces exact packet states with the
model's own hard compiler/reactor states. Offline packet and transaction
labels remain training-only targets. They are never inputs to the autonomous
assembly evaluator.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Mapping, Sequence

from safetensors.torch import save_file
import torch
import torch.distributed as dist
import torch.nn.functional as F

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TypedTheoryState,
)
from ettr_checkpoint import load_ettr_checkpoint
from ettr_data_contract import ETTRContinuationBatch
from ettr_distributed import (
    ETTRDistributedCursor,
    ETTRDistributedGradientAverager,
)
from ettr_objectives import (
    ETTRCausalQueryPair,
    ETTRObjectiveConfig,
    _causal_query_binding_loss,
)
from ettr_optimization import ETTROptimizerBundle
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_v3_streaming import ETTRV3StreamingRelease, move_continuation_batch
from eval_ettr_v3 import (
    _build_model,
    _parameter_sha256,
    _read_hash_bound_json,
    _validate_checkpoint_cursor,
    _validate_run_contract,
)
from probe_ettr_oracle_interfaces import (
    packet_targets_to_state,
    policy_masks,
    target_policy,
)
from train_ettr_component_island import (
    _component_state,
    _evaluate_interfaces,
    _reactor_policy_logits,
    _reader_logits,
    _reader_pairs_from_logits,
    _sha256_file,
    _write_no_replace,
    compiler_packet_loss,
    load_component_warm_start,
)


RUN_SCHEMA = "shohin-ettr-progressive-coupling-run-v3"
REPORT_SCHEMA = "shohin-ettr-progressive-coupling-report-v3"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_COMPONENTS = ("compiler", "reactor", "reader")
_POLICY_FIELDS = (
    "opcode",
    "source",
    "target",
    "relation",
    "type_index",
    "value_code",
)
_READER_CAUSAL_BALANCE_MODES = ("population", "factor")


class ETTRProgressiveCouplingError(RuntimeError):
    """A progressive-coupling custody or training contract failed."""


def _distributed_environment() -> tuple[int, int, int]:
    values = tuple(
        int(os.environ.get(name, default))
        for name, default in (
            ("RANK", "0"),
            ("WORLD_SIZE", "1"),
            ("LOCAL_RANK", "0"),
        )
    )
    rank, world_size, local_rank = values
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ETTRProgressiveCouplingError(
            "distributed environment differs"
        )
    if not torch.cuda.is_available():
        raise ETTRProgressiveCouplingError(
            "progressive coupling requires CUDA"
        )
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=15),
        )
        if dist.get_rank() != rank or dist.get_world_size() != world_size:
            raise ETTRProgressiveCouplingError(
                "distributed process group differs"
            )
    return rank, world_size, local_rank


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _all_reduce_sum(value: torch.Tensor, world_size: int) -> None:
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)


def _broadcast_rank_zero_error(
    error: str | None,
    *,
    rank: int,
    world_size: int,
) -> None:
    if world_size == 1:
        if error is not None:
            raise ETTRProgressiveCouplingError(error)
        return
    values = [error if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    if values[0] is not None:
        raise ETTRProgressiveCouplingError(str(values[0]))


def progressive_coupling_probability(
    update: int,
    *,
    warmup_updates: int,
    ramp_updates: int,
) -> float:
    """Return the scheduled probability of consuming self-generated state."""

    if (
        update < 1
        or warmup_updates < 0
        or ramp_updates < 1
    ):
        raise ETTRProgressiveCouplingError(
            "progressive coupling schedule differs"
        )
    if update <= warmup_updates:
        return 0.0
    if ramp_updates == 1:
        return 1.0
    progress = update - warmup_updates
    return min(1.0, (progress - 1) / (ramp_updates - 1))


def deterministic_autonomous_choice(
    probability: float,
    *,
    coupling_seed: int,
    update: int,
    stage: int,
) -> bool:
    """Select one state source for a complete causal-factorial batch."""

    if (
        not math.isfinite(probability)
        or not 0.0 <= probability <= 1.0
        or not 0 <= coupling_seed < 2**63
        or update < 1
        or stage < -1
    ):
        raise ETTRProgressiveCouplingError(
            "progressive coupling decision differs"
        )
    if probability == 0.0:
        return False
    if probability == 1.0:
        return True
    payload = f"{coupling_seed}:{update}:{stage}".encode("ascii")
    draw = int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        "big",
    ) / 2**64
    return draw < probability


def deterministic_exact_anchor_steps(
    eligible_steps: Sequence[int],
    anchor_count: int,
    *,
    coupling_seed: int,
    update: int,
) -> tuple[int, ...]:
    """Select evenly spaced exact-state anchors with a rotating offset."""

    if (
        not eligible_steps
        or tuple(eligible_steps) != tuple(sorted(set(eligible_steps)))
        or any(step < 0 for step in eligible_steps)
        or not 1 <= anchor_count <= len(eligible_steps)
        or not 0 <= coupling_seed < 2**63
        or update < 1
    ):
        raise ETTRProgressiveCouplingError(
            "exact reactor anchor schedule differs"
        )
    payload = f"{coupling_seed}:{update}:exact-anchors".encode("ascii")
    offset = int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        "big",
    ) % len(eligible_steps)
    return tuple(
        sorted(
            {
                eligible_steps[
                    (
                        offset
                        + (index * len(eligible_steps)) // anchor_count
                    )
                    % len(eligible_steps)
                ]
                for index in range(anchor_count)
            }
        )
    )


def truncate_state_credit(
    state: TypedTheoryState,
    *,
    completed_steps: int,
    total_steps: int,
    credit_horizon: int,
    use_autonomous: bool,
) -> tuple[TypedTheoryState, bool]:
    """Detach recurrent credit at fixed boundaries without changing state."""

    if (
        completed_steps < 1
        or total_steps < completed_steps
        or not 1 <= credit_horizon <= total_steps
        or not isinstance(use_autonomous, bool)
    ):
        raise ETTRProgressiveCouplingError(
            "recurrent credit horizon differs"
        )
    should_truncate = (
        use_autonomous
        and completed_steps < total_steps
        and completed_steps % credit_horizon == 0
    )
    return (
        state.detached_clone() if should_truncate else state,
        should_truncate,
    )


def select_state_source(
    exact: TypedTheoryState,
    autonomous: TypedTheoryState,
    *,
    use_autonomous: bool,
) -> TypedTheoryState:
    """Choose an entire exact or autonomous batch without factor leakage."""

    if (
        not isinstance(use_autonomous, bool)
        or exact.step != autonomous.step
    ):
        raise ETTRProgressiveCouplingError(
            "progressive coupling state source differs"
        )
    exact_values = tuple(
        getattr(exact, name)
        for name in (
            "value_probabilities",
            "type_probabilities",
            "relations",
            "active",
            "root",
            "committed",
            "halted",
        )
    )
    autonomous_values = tuple(
        getattr(autonomous, name)
        for name in (
            "value_probabilities",
            "type_probabilities",
            "relations",
            "active",
            "root",
            "committed",
            "halted",
        )
    )
    if any(
        exact_value.shape != autonomous_value.shape
        or exact_value.device != autonomous_value.device
        for exact_value, autonomous_value in zip(
            exact_values,
            autonomous_values,
            strict=True,
        )
    ):
        raise ETTRProgressiveCouplingError(
            "progressive coupling state geometry differs"
        )
    return autonomous if use_autonomous else exact


def factorial_delta_matching_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    rectangle_rows: torch.Tensor,
    *,
    row_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match finite WORLD/COMMAND differences at one discrete interface."""

    if (
        prediction.shape != target.shape
        or prediction.ndim < 1
        or rectangle_rows.ndim != 3
        or rectangle_rows.shape[1:] != (2, 2)
        or rectangle_rows.dtype != torch.long
        or rectangle_rows.device != prediction.device
        or target.device != prediction.device
        or prediction.shape[0] != rectangle_rows.numel()
        or (
            row_mask is not None
            and (
                row_mask.shape != (prediction.shape[0],)
                or row_mask.dtype != torch.bool
                or row_mask.device != prediction.device
            )
        )
    ):
        raise ETTRProgressiveCouplingError(
            "factorial delta geometry differs"
        )
    r00 = rectangle_rows[:, 0, 0]
    r01 = rectangle_rows[:, 0, 1]
    r10 = rectangle_rows[:, 1, 0]
    r11 = rectangle_rows[:, 1, 1]
    left = torch.cat((r00, r01, r00, r10))
    right = torch.cat((r10, r11, r01, r11))
    pair_mask = torch.ones_like(left, dtype=torch.bool)
    if row_mask is not None:
        pair_mask = row_mask.index_select(0, left) & row_mask.index_select(
            0,
            right,
        )
    predicted_delta = (
        prediction.index_select(0, left)
        - prediction.index_select(0, right)
    ).reshape(left.numel(), -1)
    target_delta = (
        target.index_select(0, left)
        - target.index_select(0, right)
    ).reshape(left.numel(), -1)
    valid = pair_mask[:, None].expand_as(predicted_delta)
    changed = target_delta.ne(0) & valid
    unchanged = ~target_delta.ne(0) & valid
    squared_error = (predicted_delta.float() - target_delta.float()).square()

    changed_count = changed.sum()
    unchanged_count = unchanged.sum()
    changed_mean = (
        (squared_error * changed).sum()
        / changed_count.clamp_min(1).to(squared_error.dtype)
    )
    unchanged_mean = (
        (squared_error * unchanged).sum()
        / unchanged_count.clamp_min(1).to(squared_error.dtype)
    )
    supported_parts = (
        changed_count.gt(0).to(squared_error.dtype)
        + unchanged_count.gt(0).to(squared_error.dtype)
    )
    loss = (changed_mean + unchanged_mean) / supported_parts.clamp_min(1)
    return loss, pair_mask.any().to(squared_error.dtype)


def _masked_categorical_cross_entropy_device(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        logits.ndim != 2
        or targets.shape != (logits.shape[0],)
        or mask.shape != targets.shape
        or targets.dtype != torch.long
        or mask.dtype != torch.bool
        or targets.device != logits.device
        or mask.device != logits.device
    ):
        raise ETTRProgressiveCouplingError(
            "masked categorical geometry differs"
        )
    losses = F.cross_entropy(
        logits.float(),
        targets,
        reduction="none",
    )
    support = mask.sum()
    loss = (
        (losses * mask).sum()
        / support.clamp_min(1).to(losses.dtype)
    )
    return loss, support.gt(0).to(losses.dtype)


def _supported_field_means(
    values: Mapping[str, Sequence[tuple[torch.Tensor, torch.Tensor]]],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    means = {}
    supports = {}
    for name, observations in values.items():
        losses = torch.stack(tuple(value[0] for value in observations))
        indicators = torch.stack(tuple(value[1] for value in observations))
        support = indicators.sum()
        means[name] = (
            (losses * indicators).sum() / support.clamp_min(1)
        )
        supports[name] = support.gt(0).to(losses.dtype)
    return means, supports


def _supported_mean(
    values: Mapping[str, torch.Tensor],
    supports: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    losses = torch.stack(tuple(values.values()))
    indicators = torch.stack(
        tuple(supports[name] for name in values)
    )
    return (losses * indicators).sum() / indicators.sum().clamp_min(1)


def _state_factorial_delta_loss(
    prediction: TypedTheoryState,
    target: TypedTheoryState,
    rectangle_rows: torch.Tensor,
) -> torch.Tensor:
    losses = tuple(
        factorial_delta_matching_loss(
            getattr(prediction, name),
            getattr(target, name),
            rectangle_rows,
        )
        for name in (
            "value_probabilities",
            "type_probabilities",
            "relations",
            "active",
            "root",
            "committed",
            "halted",
        )
    )
    values = torch.stack(tuple(loss for loss, _support in losses))
    supports = torch.stack(
        tuple(support for _loss, support in losses)
    )
    return (values * supports).sum() / supports.sum().clamp_min(1)


def select_trainable_architecture(
    model: EndogenousTypedTheoryReactorGPT,
    *,
    freeze_reader: bool = False,
) -> dict[str, object]:
    """Freeze the base and train exactly the three ETTR architecture modules."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = {
        "compiler": model.compiler,
        "reactor": model.reactor,
        "reader": model.query_reader,
    }
    selected_ids: set[int] = set()
    component_parameters: dict[str, int] = {}
    trainable_component_parameters: dict[str, int] = {}
    for name, module in modules.items():
        for parameter in module.parameters():
            trainable = not (name == "reader" and freeze_reader)
            parameter.requires_grad_(trainable)
            if trainable:
                selected_ids.add(id(parameter))
        component_parameters[name] = sum(
            parameter.numel() for parameter in module.parameters()
        )
        trainable_component_parameters[name] = sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        )
    trainable_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if not selected_ids or selected_ids != trainable_ids:
        raise ETTRProgressiveCouplingError(
            "progressive coupling parameter ownership differs"
        )
    return {
        "component_parameters": component_parameters,
        "frozen_base": True,
        "frozen_reader_anchor": freeze_reader,
        "trainable_component_parameters": (
            trainable_component_parameters
        ),
        "trainable_parameters": sum(
            trainable_component_parameters.values()
        ),
        "trainable_tensors": len(trainable_ids),
    }


def _reader_state_loss(
    model: EndogenousTypedTheoryReactorGPT,
    batch: ETTRContinuationBatch,
    state: TypedTheoryState,
    *,
    causal_balance_mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    read_logits = _reader_logits(
        model,
        batch,
        state,
        injection="stage",
    )
    targets = batch.episodes.query.targets.gather(
        1,
        batch.episodes.query_read_index[:, None],
    ).squeeze(1)
    factual = F.cross_entropy(read_logits.float(), targets)
    pairs: Mapping[str, ETTRCausalQueryPair] = _reader_pairs_from_logits(
        read_logits,
        batch,
    )
    world = reader_causal_binding_loss(
        pairs["world"],
        margin=1.0,
        balance_mode=causal_balance_mode,
    )
    command = reader_causal_binding_loss(
        pairs["command"],
        margin=1.0,
        balance_mode=causal_balance_mode,
    )
    losses = {
        "factual": factual,
        "world_binding": world,
        "command_binding": command,
    }
    return torch.stack(tuple(losses.values())).mean(), {
        name: float(value.detach().cpu())
        for name, value in losses.items()
    }


def reader_causal_binding_loss(
    pair: ETTRCausalQueryPair,
    *,
    margin: float,
    balance_mode: str,
) -> torch.Tensor:
    """Optionally balance answer-changing and invariant causal pairs.

    The immutable factorial release intentionally contains both kinds of
    intervention. Population averaging preserves their natural frequency,
    while factor balancing prevents a rare answer-changing factor from being
    overwhelmed by many valid invariance pairs.
    """

    if balance_mode == "population":
        return _causal_query_binding_loss(pair, margin=margin)[0]
    if balance_mode != "factor":
        raise ETTRProgressiveCouplingError(
            "reader causal balance mode differs"
        )

    classification = 0.5 * (
        F.cross_entropy(pair.correct_logits, pair.correct_target)
        + F.cross_entropy(pair.foil_logits, pair.foil_target)
    )
    correct_for_correct = pair.correct_logits.gather(
        1,
        pair.correct_target[:, None],
    ).squeeze(1)
    correct_for_foil = pair.correct_logits.gather(
        1,
        pair.foil_target[:, None],
    ).squeeze(1)
    foil_for_correct = pair.foil_logits.gather(
        1,
        pair.correct_target[:, None],
    ).squeeze(1)
    foil_for_foil = pair.foil_logits.gather(
        1,
        pair.foil_target[:, None],
    ).squeeze(1)
    difference_in_differences = (
        correct_for_correct
        - correct_for_foil
        - foil_for_correct
        + foil_for_foil
    )
    effect_mask = pair.correct_target.ne(pair.foil_target)
    invariant_mask = ~effect_mask
    contrast = F.softplus(margin - difference_in_differences)

    correct_log_probabilities = F.log_softmax(
        pair.correct_logits,
        dim=-1,
    )
    foil_log_probabilities = F.log_softmax(pair.foil_logits, dim=-1)
    correct_probabilities = correct_log_probabilities.exp()
    foil_probabilities = foil_log_probabilities.exp()
    mean_probabilities = 0.5 * (
        correct_probabilities + foil_probabilities
    )
    mean_log_probabilities = mean_probabilities.clamp_min(
        torch.finfo(mean_probabilities.dtype).tiny
    ).log()
    invariance = 0.5 * (
        (
            correct_probabilities
            * (correct_log_probabilities - mean_log_probabilities)
        ).sum(dim=-1)
        + (
            foil_probabilities
            * (foil_log_probabilities - mean_log_probabilities)
        ).sum(dim=-1)
    )

    effect_support = effect_mask.sum()
    invariant_support = invariant_mask.sum()
    effect_mean = (
        contrast * effect_mask.to(contrast.dtype)
    ).sum() / effect_support.clamp_min(1)
    invariant_mean = (
        invariance * invariant_mask.to(invariance.dtype)
    ).sum() / invariant_support.clamp_min(1)
    supported = torch.stack(
        (
            effect_support.gt(0),
            invariant_support.gt(0),
        )
    ).to(classification.dtype)
    structural = (
        torch.stack((effect_mean, invariant_mean)) * supported
    ).sum() / supported.sum().clamp_min(1)
    return classification + structural


def progressive_coupling_loss(
    model: EndogenousTypedTheoryReactorGPT,
    batch: ETTRContinuationBatch,
    *,
    coupling_probability: float,
    coupling_seed: int,
    counterfactual_delta_weight: float,
    credit_horizon: int,
    exact_anchor_steps: int,
    profile_phase_timing: bool,
    reader_causal_balance_mode: str,
    update: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Train local interfaces while progressively consuming predicted state."""

    phase_seconds: dict[str, float] = {}
    phase_started = time.perf_counter()

    def finish_phase(name: str) -> None:
        nonlocal phase_started
        if not profile_phase_timing:
            return
        torch.cuda.synchronize()
        now = time.perf_counter()
        phase_seconds[name] = now - phase_started
        phase_started = now

    if profile_phase_timing:
        torch.cuda.synchronize()
        phase_started = time.perf_counter()
    with torch.no_grad():
        world_hidden = model._encode_to_stage(
            batch.episodes.world.tokens,
            pos=0,
        )
        command_hidden = model._encode_to_stage(
            batch.episodes.command.tokens,
            pos=0,
        )
    finish_phase("base_encoding")
    soft_initial = model.compiler(
        world_hidden.detach(),
        attention_mask=batch.episodes.world.attention_mask,
        hard=False,
    )
    compiler_loss, compiler_parts = compiler_packet_loss(
        soft_initial,
        batch.packet_targets,
    )
    exact_state = packet_targets_to_state(
        batch.packet_targets,
        model.config,
        step=0,
        dtype=soft_initial.value_probabilities.dtype,
    )
    compiler_delta_loss = _state_factorial_delta_loss(
        soft_initial,
        exact_state,
        batch.causal_rectangles.rows,
    )
    finish_phase("compiler_and_delta")
    autonomous_state = model.compiler(
        world_hidden.detach(),
        attention_mask=batch.episodes.world.attention_mask,
        hard=True,
    )
    autonomous_decisions = 0
    decisions = 1
    state_is_autonomous = deterministic_autonomous_choice(
        coupling_probability,
        coupling_seed=coupling_seed,
        update=update,
        stage=-1,
    )
    autonomous_decisions += int(state_is_autonomous)
    state = select_state_source(
        exact_state,
        autonomous_state,
        use_autonomous=state_is_autonomous,
    )

    targets = batch.transaction_targets
    masks = policy_masks(targets)
    coupled_field_losses: dict[
        str,
        list[tuple[torch.Tensor, torch.Tensor]],
    ] = {
        name: [] for name in _POLICY_FIELDS
    }
    exact_field_losses: dict[
        str,
        list[tuple[torch.Tensor, torch.Tensor]],
    ] = {
        name: [] for name in _POLICY_FIELDS
    }
    reactor_delta_losses: dict[
        str,
        list[tuple[torch.Tensor, torch.Tensor]],
    ] = {
        name: [] for name in _POLICY_FIELDS
    }
    supported_steps = (
        torch.stack(tuple(masks.values()))
        .any(dim=0)
        .any(dim=0)
        .nonzero(as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .tolist()
    )
    exact_anchor_indices = set(
        deterministic_exact_anchor_steps(
            supported_steps,
            min(exact_anchor_steps, len(supported_steps)),
            coupling_seed=coupling_seed,
            update=update,
        )
    )
    credit_truncations = 0
    for step in range(targets.opcode.shape[1]):
        policy, logits = _reactor_policy_logits(
            model.reactor,
            state,
            command_hidden=command_hidden.detach(),
            command_attention_mask=batch.episodes.command.attention_mask,
            hard=True,
        )
        exact_target_policy = target_policy(
            targets,
            model.config,
            step,
            dtype=exact_state.active.dtype,
        )
        exact_logits = None
        if step in exact_anchor_indices:
            if state_is_autonomous:
                _exact_policy, exact_logits = _reactor_policy_logits(
                    model.reactor,
                    exact_state,
                    command_hidden=command_hidden.detach(),
                    command_attention_mask=(
                        batch.episodes.command.attention_mask
                    ),
                    hard=True,
                )
            else:
                exact_logits = logits
        for name in _POLICY_FIELDS:
            coupled_loss = _masked_categorical_cross_entropy_device(
                logits[name],
                getattr(targets, name)[:, step],
                masks[name][:, step],
            )
            coupled_field_losses[name].append(coupled_loss)
            if exact_logits is not None:
                exact_loss = _masked_categorical_cross_entropy_device(
                    exact_logits[name],
                    getattr(targets, name)[:, step],
                    masks[name][:, step],
                )
                exact_field_losses[name].append(exact_loss)
                delta_loss = factorial_delta_matching_loss(
                    exact_logits[name].float().softmax(-1),
                    getattr(exact_target_policy, name).float(),
                    batch.causal_rectangles.rows,
                    row_mask=masks[name][:, step],
                )
                reactor_delta_losses[name].append(delta_loss)
        autonomous_state = model.reactor.apply(
            state,
            policy,
            hard=True,
            validate=False,
        )
        with torch.no_grad():
            exact_state = model.reactor.apply(
                exact_state,
                exact_target_policy,
                hard=True,
                validate=False,
            ).detached_clone()
        state_is_autonomous = deterministic_autonomous_choice(
            coupling_probability,
            coupling_seed=coupling_seed,
            update=update,
            stage=step,
        )
        autonomous_decisions += int(state_is_autonomous)
        decisions += 1
        state = select_state_source(
            exact_state,
            autonomous_state,
            use_autonomous=state_is_autonomous,
        )
        state, truncated = truncate_state_credit(
            state,
            completed_steps=step + 1,
            total_steps=targets.opcode.shape[1],
            credit_horizon=credit_horizon,
            use_autonomous=state_is_autonomous,
        )
        credit_truncations += int(truncated)
    finish_phase("recurrent_reactor_and_anchors")

    coupled_reactor_means, coupled_reactor_supports = (
        _supported_field_means(coupled_field_losses)
    )
    exact_reactor_means, exact_reactor_supports = (
        _supported_field_means(exact_field_losses)
    )
    reactor_delta_means, reactor_delta_supports = (
        _supported_field_means(reactor_delta_losses)
    )
    coupled_reactor_loss = _supported_mean(
        coupled_reactor_means,
        coupled_reactor_supports,
    )
    exact_reactor_loss = _supported_mean(
        exact_reactor_means,
        exact_reactor_supports,
    )
    reactor_delta_loss = _supported_mean(
        reactor_delta_means,
        reactor_delta_supports,
    )

    exact_terminal = packet_targets_to_state(
        batch.terminal_packet_targets,
        model.config,
        step=targets.opcode.shape[1],
        dtype=next(model.query_reader.parameters()).dtype,
    )
    exact_reader_loss, exact_reader_parts = _reader_state_loss(
        model,
        batch,
        exact_terminal,
        causal_balance_mode=reader_causal_balance_mode,
    )
    coupled_reader_loss, coupled_reader_parts = _reader_state_loss(
        model,
        batch,
        state,
        causal_balance_mode=reader_causal_balance_mode,
    )
    finish_phase("readers")
    base_losses = {
        "compiler": compiler_loss,
        "coupled_reactor": coupled_reactor_loss,
        "exact_reactor": exact_reactor_loss,
        "exact_reader": exact_reader_loss,
        "coupled_reader": coupled_reader_loss,
    }
    delta_losses = {
        "compiler_delta": compiler_delta_loss,
        "reactor_delta": reactor_delta_loss,
    }
    total = (
        torch.stack(tuple(base_losses.values())).sum()
        + counterfactual_delta_weight
        * torch.stack(tuple(delta_losses.values())).sum()
    ) / (
        len(base_losses)
        + counterfactual_delta_weight * len(delta_losses)
    )
    high_level = {**base_losses, **delta_losses}
    return total, {
        "autonomous_decisions": autonomous_decisions,
        "coupling_probability": coupling_probability,
        "credit_horizon": credit_horizon,
        "credit_truncations": credit_truncations,
        "decision_count": decisions,
        "exact_anchor_steps": sorted(exact_anchor_indices),
        "phase_seconds": phase_seconds,
        "reader_causal_balance_mode": reader_causal_balance_mode,
        "high_level": {
            name: float(value.detach().cpu())
            for name, value in high_level.items()
        },
        "compiler": compiler_parts,
        "coupled_reactor": {
            name: float(value.detach().cpu())
            for name, value in coupled_reactor_means.items()
        },
        "exact_reactor": {
            name: float(value.detach().cpu())
            for name, value in exact_reactor_means.items()
        },
        "reactor_delta": {
            name: float(value.detach().cpu())
            for name, value in reactor_delta_means.items()
        },
        "exact_reader": exact_reader_parts,
        "coupled_reader": coupled_reader_parts,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--protected-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--run-contract-sha256", required=True)
    for component in _COMPONENTS:
        parser.add_argument(
            f"--initial-{component}",
            type=Path,
            required=True,
        )
        parser.add_argument(
            f"--initial-{component}-sha256",
            required=True,
        )
        parser.add_argument(
            f"--{component}-learning-rate",
            type=float,
            required=True,
        )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--architecture-seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--coupling-seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=1_000)
    parser.add_argument("--start-position", type=int, default=20_000)
    parser.add_argument("--warmup-updates", type=int, default=100)
    parser.add_argument("--ramp-updates", type=int, default=700)
    parser.add_argument(
        "--counterfactual-delta-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument("--exact-anchor-steps", type=int, default=4)
    parser.add_argument("--credit-horizon", type=int, default=4)
    parser.add_argument(
        "--reader-causal-balance-mode",
        choices=_READER_CAUSAL_BALANCE_MODES,
        default="population",
    )
    parser.add_argument("--freeze-reader", action="store_true")
    parser.add_argument("--profile-phase-timing", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    hashes = (
        args.release_sha256,
        args.checkpoint_sha256,
        args.run_contract_sha256,
        args.initial_compiler_sha256,
        args.initial_reactor_sha256,
        args.initial_reader_sha256,
    )
    rates = (
        args.compiler_learning_rate,
        args.reactor_learning_rate,
        args.reader_learning_rate,
    )
    paths = (
        args.release_root,
        args.data_root,
        args.tokenizer,
        args.protected_checkpoint,
        args.checkpoint,
        args.run_contract,
        args.initial_compiler,
        args.initial_reactor,
        args.initial_reader,
        args.output,
    )
    if (
        any(_HEX64.fullmatch(value) is None for value in hashes)
        or _HEX40.fullmatch(args.source_commit) is None
        or any(not path.is_absolute() for path in paths)
        or not 0 <= args.architecture_seed < 2**63
        or not 0 <= args.data_seed < 2**63
        or not 0 <= args.coupling_seed < 2**63
        or args.updates < 1
        or args.start_position < 0
        or args.warmup_updates < 0
        or args.ramp_updates < 1
        or args.warmup_updates + args.ramp_updates > args.updates
        or not math.isfinite(args.counterfactual_delta_weight)
        or args.counterfactual_delta_weight <= 0.0
        or not 1 <= args.exact_anchor_steps <= 64
        or not 1 <= args.credit_horizon <= 64
        or args.reader_causal_balance_mode
        not in _READER_CAUSAL_BALANCE_MODES
        or any(
            not math.isfinite(rate) or not 0.0 < rate < 1.0
            for rate in rates
        )
        or not math.isfinite(args.weight_decay)
        or args.weight_decay < 0.0
        or not math.isfinite(args.gradient_clip)
        or args.gradient_clip <= 0.0
        or args.eval_batches < 2
        or args.log_every < 1
    ):
        raise ETTRProgressiveCouplingError(
            "progressive coupling arguments differ"
        )


def _save_components(
    model: EndogenousTypedTheoryReactorGPT,
    output: Path,
    *,
    suffix: str,
) -> dict[str, str]:
    hashes = {}
    for component in _COMPONENTS:
        path = output / f"{component}-{suffix}.safetensors"
        save_file(_component_state(model, component), path)
        os.chmod(path, 0o400)
        hashes[component] = _sha256_file(path)
    return hashes


def _distributed_parameter_sha256(
    model: EndogenousTypedTheoryReactorGPT,
    *,
    rank: int,
    world_size: int,
) -> str:
    digest = _parameter_sha256(model)
    if world_size == 1:
        return digest
    gathered: list[str | None] = [None] * world_size
    dist.all_gather_object(gathered, digest)
    if any(value != digest for value in gathered):
        raise ETTRProgressiveCouplingError(
            "distributed parameter identity differs"
        )
    return digest


def _distributed_mean(
    value: torch.Tensor,
    *,
    device: torch.device,
    world_size: int,
) -> float:
    reduced = value.detach().float().to(device).reshape(())
    _all_reduce_sum(reduced, world_size)
    reduced.div_(world_size)
    return float(reduced.cpu())


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    rank, world_size, local_rank = _distributed_environment()
    device = torch.device("cuda", local_rank)
    try:
        if "H100" not in torch.cuda.get_device_name(device).upper():
            raise ETTRProgressiveCouplingError(
                "progressive coupling requires H100"
            )
        output_error = None
        if rank == 0:
            try:
                if args.output.exists() or args.output.is_symlink():
                    raise ETTRProgressiveCouplingError(
                        "refusing an existing progressive coupling output"
                    )
                args.output.mkdir(mode=0o700, parents=True)
            except BaseException as exc:
                output_error = f"{type(exc).__name__}: {exc}"
        _broadcast_rank_zero_error(
            output_error,
            rank=rank,
            world_size=world_size,
        )
        _barrier(world_size)

        stream = ETTRV3StreamingRelease(
            args.release_root,
            expected_release_sha256=args.release_sha256,
            data_root=args.data_root,
            tokenizer_path=args.tokenizer,
        )
        source_verification = None
        source_error = None
        if rank == 0:
            try:
                source_verification = stream.verify_source_shards()
            except BaseException as exc:
                source_error = (
                    f"rank-zero source verification failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        _broadcast_rank_zero_error(
            source_error,
            rank=rank,
            world_size=world_size,
        )
        run_contract = _read_hash_bound_json(
            args.run_contract,
            expected_sha256=args.run_contract_sha256,
            label="ETTR run contract",
        )
        model_config, optimizer_config = _validate_run_contract(
            run_contract,
            release_sha256=args.release_sha256,
            release_source_commit=stream.release["source_commit"],
            architecture_seed=args.architecture_seed,
        )
        model, protected = _build_model(
            args.protected_checkpoint,
            architecture_seed=args.architecture_seed,
            model_config=model_config,
            device=device,
        )
        resume_optimizer = ETTROptimizerBundle(model, optimizer_config)
        resumed = load_ettr_checkpoint(
            args.checkpoint,
            expected_sha256=args.checkpoint_sha256,
            model=model,
            protected_base=protected,
            optimizer=resume_optimizer,
            scheduler=None,
        )
        _validate_checkpoint_cursor(
            resumed.progress,
            resumed.data_stream,
            run_contract=run_contract,
            stream=stream,
            release_sha256=args.release_sha256,
            protected_step=protected.step,
        )
        del resume_optimizer

        ownership = select_trainable_architecture(
            model,
            freeze_reader=args.freeze_reader,
        )
        loaded_component_sha256 = {}
        for component in _COMPONENTS:
            loaded_component_sha256[component] = load_component_warm_start(
                model,
                component,
                getattr(args, f"initial_{component}"),
                expected_sha256=getattr(
                    args,
                    f"initial_{component}_sha256",
                ),
            )
        optimizer_groups = [
            {
                "params": tuple(model.compiler.parameters()),
                "lr": args.compiler_learning_rate,
            },
            {
                "params": tuple(model.reactor.parameters()),
                "lr": args.reactor_learning_rate,
            },
        ]
        if not args.freeze_reader:
            optimizer_groups.append(
                {
                    "params": tuple(model.query_reader.parameters()),
                    "lr": args.reader_learning_rate,
                }
            )
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
        trainable = tuple(
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        averager = ETTRDistributedGradientAverager(
            world_size=world_size,
            all_reduce_sum=lambda value: _all_reduce_sum(
                value,
                world_size,
            )
        )
        packet_index = ETTRDiskPacketSufficiencyIndex(
            stream.packet_index_root
        )
        try:
            initial_parameter_sha256 = _distributed_parameter_sha256(
                model,
                rank=rank,
                world_size=world_size,
            )
            initial_component_sha256 = None
            before = None
            if rank == 0:
                initial_component_sha256 = _save_components(
                    model,
                    args.output,
                    suffix="initial",
                )
                before = _evaluate_interfaces(
                    model,
                    stream=stream,
                    packet_index=packet_index,
                    device=device,
                    data_seed=args.data_seed,
                    max_batches=args.eval_batches,
                    reader_injection="stage",
                )
                contract = {
                    "architecture_seed": args.architecture_seed,
                    "checkpoint_sha256": args.checkpoint_sha256,
                    "component_learning_rates": {
                        name: getattr(args, f"{name}_learning_rate")
                        for name in _COMPONENTS
                    },
                    "coupling": {
                        "batch_factorial_source_is_atomic": True,
                        "credit_horizon": args.credit_horizon,
                        "exact_anchor_steps_per_update": (
                            args.exact_anchor_steps
                        ),
                        "profile_phase_timing": args.profile_phase_timing,
                        "reader_causal_balance_mode": (
                            args.reader_causal_balance_mode
                        ),
                        "reader_is_frozen_semantic_anchor": (
                            args.freeze_reader
                        ),
                        "ramp_updates": args.ramp_updates,
                        "seed": args.coupling_seed,
                        "warmup_updates": args.warmup_updates,
                    },
                    "data_seed": args.data_seed,
                    "distributed": {
                        "gradient_reduction": "dense_mean",
                        "rank_zero_only_writer": True,
                        "world_size": world_size,
                    },
                    "eval_batches": args.eval_batches,
                    "gradient_clip": args.gradient_clip,
                    "high_level_loss_weights": {
                        "compiler": 1.0,
                        "coupled_reader": 1.0,
                        "coupled_reactor": 1.0,
                        "exact_reader": 1.0,
                        "exact_reactor": 1.0,
                        "compiler_delta": (
                            args.counterfactual_delta_weight
                        ),
                        "reactor_delta": (
                            args.counterfactual_delta_weight
                        ),
                    },
                    "loaded_component_sha256": loaded_component_sha256,
                    "oracle_at_autonomous_inference": False,
                    "oracle_training_boundary": {
                        "compiler": "direct_initial_packet_target",
                        "reactor": (
                            "target_transaction_loss_and_scheduled_exact_state"
                        ),
                        "reader": (
                            "exact_terminal_auxiliary_and_scheduled_terminal_state"
                        ),
                    },
                    "ownership": ownership,
                    "protected_checkpoint_sha256": (
                        protected.checkpoint_sha256
                    ),
                    "release_file_sha256": args.release_sha256,
                    "run_contract_sha256": args.run_contract_sha256,
                    "schema": RUN_SCHEMA,
                    "source_commit": args.source_commit,
                    "start_position": args.start_position,
                    "updates": args.updates,
                    "weight_decay": args.weight_decay,
                }
                _write_no_replace(
                    args.output / "coupling-contract.json",
                    (
                        json.dumps(
                            contract,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=True,
                            allow_nan=False,
                        ).encode("ascii")
                        + b"\n"
                    ),
                )
                _write_no_replace(
                    args.output / "train.jsonl",
                    b"",
                    mode=0o600,
                )
            _barrier(world_size)

            model.train()
            model.base.eval()
            cursor = ETTRDistributedCursor(
                epoch=0,
                position=args.start_position,
            )
            cursor.validate(
                core_batches=len(stream.records["train"]),
                world_size=world_size,
                accumulation=1,
            )
            iterator = stream.iter_positioned_batches(
                "train",
                rank=rank,
                world_size=world_size,
                epoch=cursor.epoch,
                seed=args.data_seed,
                start_position=cursor.position,
            )
            observed_rows = 0
            observed_token_positions = 0
            last_loss = None
            for update in range(1, args.updates + 1):
                active_epoch = cursor.epoch
                try:
                    local_position, cpu_batch = next(iterator)
                except StopIteration as exc:
                    raise ETTRProgressiveCouplingError(
                        "distributed stream ended before cursor boundary"
                    ) from exc
                if local_position != cursor.position + rank:
                    raise ETTRProgressiveCouplingError(
                        "distributed stream position differs"
                    )
                packet_index.verify_train((cpu_batch,))
                batch = move_continuation_batch(cpu_batch, device)
                batch.validate(
                    model.config,
                    ETTRObjectiveConfig(
                        vocab_size=model.base.cfg.vocab_size
                    ),
                )
                probability = progressive_coupling_probability(
                    update,
                    warmup_updates=args.warmup_updates,
                    ramp_updates=args.ramp_updates,
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    loss, parts = progressive_coupling_loss(
                        model,
                        batch,
                        coupling_probability=probability,
                        coupling_seed=args.coupling_seed,
                        counterfactual_delta_weight=(
                            args.counterfactual_delta_weight
                        ),
                        credit_horizon=args.credit_horizon,
                        exact_anchor_steps=args.exact_anchor_steps,
                        profile_phase_timing=args.profile_phase_timing,
                        reader_causal_balance_mode=(
                            args.reader_causal_balance_mode
                        ),
                        update=update,
                    )
                finite = torch.tensor(
                    int(bool(torch.isfinite(loss))),
                    dtype=torch.int32,
                    device=device,
                )
                _all_reduce_sum(finite, world_size)
                if int(finite) != world_size:
                    raise ETTRProgressiveCouplingError(
                        "progressive coupling loss is non-finite"
                    )
                backward_started = time.perf_counter()
                loss.backward()
                if args.profile_phase_timing:
                    torch.cuda.synchronize()
                    parts["phase_seconds"]["backward"] = (
                        time.perf_counter() - backward_started
                    )
                averager(trainable)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable,
                    args.gradient_clip,
                    error_if_nonfinite=True,
                )
                optimizer_started = time.perf_counter()
                optimizer.step()
                if args.profile_phase_timing:
                    torch.cuda.synchronize()
                    parts["phase_seconds"]["optimizer"] = (
                        time.perf_counter() - optimizer_started
                    )
                observed_rows += batch.episodes.world.tokens.shape[0]
                observed_token_positions += int(
                    batch.episodes.world.attention_mask.sum().detach().cpu()
                    + batch.episodes.command.attention_mask.sum().detach().cpu()
                    + batch.episodes.query.attention_mask.sum().detach().cpu()
                )
                cursor = cursor.advance(
                    core_batches=len(stream.records["train"]),
                    world_size=world_size,
                    accumulation=1,
                )
                if cursor.epoch != active_epoch:
                    iterator = stream.iter_positioned_batches(
                        "train",
                        rank=rank,
                        world_size=world_size,
                        epoch=cursor.epoch,
                        seed=args.data_seed,
                        start_position=cursor.position,
                    )
                if (
                    update % args.log_every == 0
                    or update == args.updates
                ):
                    last_loss = _distributed_mean(
                        loss,
                        device=device,
                        world_size=world_size,
                    )
                    if rank == 0:
                        with (args.output / "train.jsonl").open(
                            "ab",
                            buffering=0,
                        ) as log:
                            log.write(
                                (
                                    json.dumps(
                                        {
                                            "epoch": cursor.epoch,
                                            "gradient_norm_pre_clip": float(
                                                gradient_norm.detach()
                                                .float()
                                                .cpu()
                                            ),
                                            "loss": last_loss,
                                            "loss_parts_rank_zero": parts,
                                            "next_position": (
                                                cursor.position
                                            ),
                                            "schema": (
                                                "shohin-ettr-progressive-"
                                                "coupling-metric-v2"
                                            ),
                                            "update": update,
                                            "world_size": world_size,
                                        },
                                        sort_keys=True,
                                        separators=(",", ":"),
                                        ensure_ascii=True,
                                        allow_nan=False,
                                    ).encode("ascii")
                                    + b"\n"
                                )
                            )
            if rank == 0:
                os.chmod(args.output / "train.jsonl", 0o400)
            totals = torch.tensor(
                [observed_rows, observed_token_positions],
                dtype=torch.int64,
                device=device,
            )
            _all_reduce_sum(totals, world_size)
            final_parameter_sha256 = _distributed_parameter_sha256(
                model,
                rank=rank,
                world_size=world_size,
            )
            _barrier(world_size)

            if rank == 0:
                after = _evaluate_interfaces(
                    model,
                    stream=stream,
                    packet_index=packet_index,
                    device=device,
                    data_seed=args.data_seed,
                    max_batches=args.eval_batches,
                    reader_injection="stage",
                )
                final_component_sha256 = _save_components(
                    model,
                    args.output,
                    suffix="final",
                )
                report = {
                    "architecture_seed": args.architecture_seed,
                    "checkpoint_sha256": args.checkpoint_sha256,
                    "coupling": contract["coupling"],
                    "data_seed": args.data_seed,
                    "device": {
                        "bf16": torch.cuda.is_bf16_supported(),
                        "name": torch.cuda.get_device_name(device),
                        "world_size": world_size,
                    },
                    "evaluation": {"after": after, "before": before},
                    "final_component_sha256": final_component_sha256,
                    "final_parameter_sha256": (
                        final_parameter_sha256
                    ),
                    "initial_component_sha256": (
                        initial_component_sha256
                    ),
                    "initial_parameter_sha256": (
                        initial_parameter_sha256
                    ),
                    "last_loss": last_loss,
                    "loaded_component_sha256": (
                        loaded_component_sha256
                    ),
                    "observed_rows": int(totals[0].cpu()),
                    "observed_token_positions": int(totals[1].cpu()),
                    "oracle_at_autonomous_inference": False,
                    "ownership": ownership,
                    "protected_checkpoint_sha256": (
                        protected.checkpoint_sha256
                    ),
                    "release_file_sha256": args.release_sha256,
                    "release_manifest_sha256": stream.manifest.sha256(),
                    "schema": REPORT_SCHEMA,
                    "source_commit": args.source_commit,
                    "source_verification": source_verification,
                    "start_position": args.start_position,
                    "updates": args.updates,
                    "world_size": world_size,
                }
                _write_no_replace(
                    args.output / "report.json",
                    (
                        json.dumps(
                            report,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=True,
                            allow_nan=False,
                        ).encode("ascii")
                        + b"\n"
                    ),
                )
            _barrier(world_size)
        finally:
            packet_index.close()
        return 0
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
