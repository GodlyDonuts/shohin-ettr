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
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Mapping, Sequence

from safetensors.torch import save_file
import torch
import torch.nn.functional as F

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TypedTheoryState,
)
from ettr_checkpoint import load_ettr_checkpoint
from ettr_data_contract import ETTRContinuationBatch
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
    _masked_categorical_cross_entropy,
    _reactor_policy_logits,
    _reader_logits,
    _reader_pairs_from_logits,
    _sha256_file,
    _write_no_replace,
    compiler_packet_loss,
    load_component_warm_start,
)


RUN_SCHEMA = "shohin-ettr-progressive-coupling-run-v1"
REPORT_SCHEMA = "shohin-ettr-progressive-coupling-report-v1"
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


class ETTRProgressiveCouplingError(RuntimeError):
    """A progressive-coupling custody or training contract failed."""


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


def select_trainable_architecture(
    model: EndogenousTypedTheoryReactorGPT,
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
    for name, module in modules.items():
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            selected_ids.add(id(parameter))
        component_parameters[name] = sum(
            parameter.numel() for parameter in module.parameters()
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
        "trainable_parameters": sum(component_parameters.values()),
        "trainable_tensors": len(trainable_ids),
    }


def _reader_state_loss(
    model: EndogenousTypedTheoryReactorGPT,
    batch: ETTRContinuationBatch,
    state: TypedTheoryState,
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
    world = _causal_query_binding_loss(pairs["world"], margin=1.0)[0]
    command = _causal_query_binding_loss(
        pairs["command"],
        margin=1.0,
    )[0]
    losses = {
        "factual": factual,
        "world_binding": world,
        "command_binding": command,
    }
    return torch.stack(tuple(losses.values())).mean(), {
        name: float(value.detach().cpu())
        for name, value in losses.items()
    }


def progressive_coupling_loss(
    model: EndogenousTypedTheoryReactorGPT,
    batch: ETTRContinuationBatch,
    *,
    coupling_probability: float,
    coupling_seed: int,
    update: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Train local interfaces while progressively consuming predicted state."""

    with torch.no_grad():
        world_hidden = model._encode_to_stage(
            batch.episodes.world.tokens,
            pos=0,
        )
        command_hidden = model._encode_to_stage(
            batch.episodes.command.tokens,
            pos=0,
        )
    soft_initial = model.compiler(
        world_hidden.detach(),
        attention_mask=batch.episodes.world.attention_mask,
        hard=False,
    )
    compiler_loss, compiler_parts = compiler_packet_loss(
        soft_initial,
        batch.packet_targets,
    )
    autonomous_state = model.compiler(
        world_hidden.detach(),
        attention_mask=batch.episodes.world.attention_mask,
        hard=True,
    )
    exact_state = packet_targets_to_state(
        batch.packet_targets,
        model.config,
        step=0,
        dtype=next(model.reactor.parameters()).dtype,
    )
    autonomous_decisions = 0
    decisions = 1
    use_autonomous = deterministic_autonomous_choice(
        coupling_probability,
        coupling_seed=coupling_seed,
        update=update,
        stage=-1,
    )
    autonomous_decisions += int(use_autonomous)
    state = select_state_source(
        exact_state,
        autonomous_state,
        use_autonomous=use_autonomous,
    )

    targets = batch.transaction_targets
    masks = policy_masks(targets)
    field_losses: dict[str, list[torch.Tensor]] = {
        name: [] for name in _POLICY_FIELDS
    }
    for step in range(targets.opcode.shape[1]):
        policy, logits = _reactor_policy_logits(
            model.reactor,
            state,
            command_hidden=command_hidden.detach(),
            command_attention_mask=batch.episodes.command.attention_mask,
            hard=True,
        )
        for name in _POLICY_FIELDS:
            loss = _masked_categorical_cross_entropy(
                logits[name],
                getattr(targets, name)[:, step],
                masks[name][:, step],
            )
            if loss is not None:
                field_losses[name].append(loss)
        autonomous_state = model.reactor.apply(
            state,
            policy,
            hard=True,
            validate=False,
        )
        with torch.no_grad():
            exact_state = model.reactor.apply(
                exact_state,
                target_policy(
                    targets,
                    model.config,
                    step,
                    dtype=exact_state.active.dtype,
                ),
                hard=True,
                validate=False,
            ).detached_clone()
        use_autonomous = deterministic_autonomous_choice(
            coupling_probability,
            coupling_seed=coupling_seed,
            update=update,
            stage=step,
        )
        autonomous_decisions += int(use_autonomous)
        decisions += 1
        state = select_state_source(
            exact_state,
            autonomous_state,
            use_autonomous=use_autonomous,
        )

    reactor_means = {
        name: torch.stack(values).mean()
        for name, values in field_losses.items()
        if values
    }
    if not reactor_means:
        raise ETTRProgressiveCouplingError(
            "progressive coupling reactor loss has no support"
        )
    reactor_loss = torch.stack(tuple(reactor_means.values())).mean()

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
    )
    coupled_reader_loss, coupled_reader_parts = _reader_state_loss(
        model,
        batch,
        state,
    )
    high_level = {
        "compiler": compiler_loss,
        "reactor": reactor_loss,
        "exact_reader": exact_reader_loss,
        "coupled_reader": coupled_reader_loss,
    }
    total = torch.stack(tuple(high_level.values())).mean()
    return total, {
        "autonomous_decisions": autonomous_decisions,
        "coupling_probability": coupling_probability,
        "decision_count": decisions,
        "high_level": {
            name: float(value.detach().cpu())
            for name, value in high_level.items()
        },
        "compiler": compiler_parts,
        "reactor": {
            name: float(value.detach().cpu())
            for name, value in reactor_means.items()
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    if not torch.cuda.is_available():
        raise ETTRProgressiveCouplingError(
            "progressive coupling requires CUDA"
        )
    device = torch.device("cuda", 0)
    if "H100" not in torch.cuda.get_device_name(device).upper():
        raise ETTRProgressiveCouplingError(
            "progressive coupling requires H100"
        )
    if args.output.exists() or args.output.is_symlink():
        raise ETTRProgressiveCouplingError(
            "refusing an existing progressive coupling output"
        )
    args.output.mkdir(mode=0o700, parents=True)

    stream = ETTRV3StreamingRelease(
        args.release_root,
        expected_release_sha256=args.release_sha256,
        data_root=args.data_root,
        tokenizer_path=args.tokenizer,
    )
    source_verification = stream.verify_source_shards()
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

    ownership = select_trainable_architecture(model)
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
    optimizer = torch.optim.AdamW(
        [
            {
                "params": tuple(model.compiler.parameters()),
                "lr": args.compiler_learning_rate,
            },
            {
                "params": tuple(model.reactor.parameters()),
                "lr": args.reactor_learning_rate,
            },
            {
                "params": tuple(model.query_reader.parameters()),
                "lr": args.reader_learning_rate,
            },
        ],
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    trainable = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    packet_index = ETTRDiskPacketSufficiencyIndex(stream.packet_index_root)
    try:
        initial_parameter_sha256 = _parameter_sha256(model)
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
                "ramp_updates": args.ramp_updates,
                "seed": args.coupling_seed,
                "warmup_updates": args.warmup_updates,
            },
            "data_seed": args.data_seed,
            "eval_batches": args.eval_batches,
            "gradient_clip": args.gradient_clip,
            "high_level_loss_weights": {
                "compiler": 1.0,
                "coupled_reader": 1.0,
                "exact_reader": 1.0,
                "reactor": 1.0,
            },
            "loaded_component_sha256": loaded_component_sha256,
            "oracle_at_autonomous_inference": False,
            "oracle_training_boundary": {
                "compiler": "direct_initial_packet_target",
                "reactor": "target_transaction_loss_and_scheduled_exact_state",
                "reader": "exact_terminal_auxiliary_and_scheduled_terminal_state",
            },
            "ownership": ownership,
            "protected_checkpoint_sha256": protected.checkpoint_sha256,
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
        _write_no_replace(args.output / "train.jsonl", b"", mode=0o600)

        model.train()
        model.base.eval()
        epoch = 0
        position = args.start_position
        iterator = stream.iter_positioned_batches(
            "train",
            rank=0,
            world_size=1,
            epoch=epoch,
            seed=args.data_seed,
            start_position=position,
        )
        observed_rows = 0
        observed_token_positions = 0
        last_loss = None
        for update in range(1, args.updates + 1):
            try:
                position, cpu_batch = next(iterator)
            except StopIteration:
                epoch += 1
                position = 0
                iterator = stream.iter_positioned_batches(
                    "train",
                    rank=0,
                    world_size=1,
                    epoch=epoch,
                    seed=args.data_seed,
                )
                position, cpu_batch = next(iterator)
            packet_index.verify_train((cpu_batch,))
            batch = move_continuation_batch(cpu_batch, device)
            batch.validate(
                model.config,
                ETTRObjectiveConfig(vocab_size=model.base.cfg.vocab_size),
            )
            probability = progressive_coupling_probability(
                update,
                warmup_updates=args.warmup_updates,
                ramp_updates=args.ramp_updates,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, parts = progressive_coupling_loss(
                    model,
                    batch,
                    coupling_probability=probability,
                    coupling_seed=args.coupling_seed,
                    update=update,
                )
            if not bool(torch.isfinite(loss)):
                raise ETTRProgressiveCouplingError(
                    "progressive coupling loss is non-finite"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                args.gradient_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
            observed_rows += batch.episodes.world.tokens.shape[0]
            observed_token_positions += int(
                batch.episodes.world.attention_mask.sum().detach().cpu()
                + batch.episodes.command.attention_mask.sum().detach().cpu()
                + batch.episodes.query.attention_mask.sum().detach().cpu()
            )
            last_loss = float(loss.detach().cpu())
            if update % args.log_every == 0 or update == args.updates:
                with (args.output / "train.jsonl").open(
                    "ab",
                    buffering=0,
                ) as log:
                    log.write(
                        (
                            json.dumps(
                                {
                                    "epoch": epoch,
                                    "gradient_norm_pre_clip": float(
                                        gradient_norm.detach().float().cpu()
                                    ),
                                    "loss": last_loss,
                                    "loss_parts": parts,
                                    "position": position,
                                    "schema": "shohin-ettr-progressive-coupling-metric-v1",
                                    "update": update,
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                                ensure_ascii=True,
                                allow_nan=False,
                            ).encode("ascii")
                            + b"\n"
                        )
                    )
        os.chmod(args.output / "train.jsonl", 0o400)

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
            },
            "evaluation": {"after": after, "before": before},
            "final_component_sha256": final_component_sha256,
            "final_parameter_sha256": _parameter_sha256(model),
            "initial_component_sha256": initial_component_sha256,
            "initial_parameter_sha256": initial_parameter_sha256,
            "last_loss": last_loss,
            "loaded_component_sha256": loaded_component_sha256,
            "observed_rows": observed_rows,
            "observed_token_positions": observed_token_positions,
            "oracle_at_autonomous_inference": False,
            "ownership": ownership,
            "protected_checkpoint_sha256": protected.checkpoint_sha256,
            "release_file_sha256": args.release_sha256,
            "release_manifest_sha256": stream.manifest.sha256(),
            "schema": REPORT_SCHEMA,
            "source_commit": args.source_commit,
            "source_verification": source_verification,
            "start_position": args.start_position,
            "updates": args.updates,
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
    finally:
        packet_index.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
