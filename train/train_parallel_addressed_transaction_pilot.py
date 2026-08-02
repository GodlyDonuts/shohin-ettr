#!/usr/bin/env python3
"""Fit and evaluate the parallel addressed ETTR COMMAND compiler."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Sequence

from safetensors.torch import save_file
import torch

from endogenous_typed_theory_reactor import TRANSACTION_COUNT, TypedTheoryState
from eval_algebraic_query_joint_state import (
    _evaluate,
    _load_compiler,
    _strict_load_joint_model,
)
from ettr_objectives import ETTRObjectiveConfig
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_query_supervision import iter_batches_with_query_specs
from ettr_v3_streaming import ETTRV3StreamingRelease, move_continuation_batch
from opcode_program_registry import load_opcode_program_registry
from parallel_addressed_transaction_compiler import (
    ParallelAddressedTransactionCompiler,
    ParallelScheduledReactor,
    RegistryProjectedAddressedScheduleCompiler,
)
from probe_ettr_oracle_interfaces import (
    _packet_batch_counts,
    packet_targets_to_state,
    policy_masks,
    target_policy,
)
from train_ettr_component_island import (
    _canonical_bytes,
    _sha256_file,
    _write_no_replace,
)


CONTRACT_SCHEMA = "shohin-ettr-parallel-addressed-transaction-contract-v9"
REPORT_SCHEMA = "shohin-ettr-parallel-addressed-transaction-report-v9"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_FIELDS = (
    "opcode",
    "source",
    "target",
    "relation",
    "type_index",
    "value_code",
)


class ParallelTransactionPilotError(RuntimeError):
    """The parallel transaction pilot violated its sealed contract."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--protected-checkpoint", type=Path, required=True)
    parser.add_argument("--joint-model", type=Path, required=True)
    parser.add_argument("--joint-model-sha256", required=True)
    parser.add_argument("--joint-run-contract", type=Path, required=True)
    parser.add_argument("--joint-run-contract-sha256", required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--compiler-sha256", required=True)
    parser.add_argument("--compiler-contract", type=Path, required=True)
    parser.add_argument("--compiler-contract-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--architecture-seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--start-position", type=int, default=13_200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--semantic-prefix-weight", type=float, default=0.0)
    parser.add_argument(
        "--training-initial-state",
        choices=("oracle", "autonomous"),
        default="oracle",
    )
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--grounded-pointers", action="store_true")
    parser.add_argument("--valid-pointer-masks", action="store_true")
    parser.add_argument("--token-native-command-mask", action="store_true")
    parser.add_argument("--cover-verified-command-mask", action="store_true")
    parser.add_argument(
        "--token-native-occurrence-command",
        action="store_true",
    )
    parser.add_argument(
        "--token-native-syntax-graph-command",
        action="store_true",
    )
    parser.add_argument(
        "--token-native-declaration-binding-command",
        action="store_true",
    )
    parser.add_argument(
        "--required-device-class",
        choices=("h100", "cuda"),
        default="h100",
    )
    parser.add_argument("--opcode-program-registry", type=Path)
    parser.add_argument("--opcode-program-registry-sha256")
    parser.add_argument(
        "--registry-projected-opcode-training",
        action="store_true",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    paths = (
        args.release_root,
        args.data_root,
        args.tokenizer,
        args.protected_checkpoint,
        args.joint_model,
        args.joint_run_contract,
        args.compiler,
        args.compiler_contract,
        args.output,
    ) + ((args.opcode_program_registry,) if args.opcode_program_registry else ())
    hashes = (
        args.release_sha256,
        args.joint_model_sha256,
        args.joint_run_contract_sha256,
        args.compiler_sha256,
        args.compiler_contract_sha256,
    ) + (
        (args.opcode_program_registry_sha256,)
        if args.opcode_program_registry_sha256
        else ()
    )
    if (
        any(not path.is_absolute() for path in paths)
        or any(_HEX64.fullmatch(value) is None for value in hashes)
        or _HEX40.fullmatch(args.source_commit) is None
        or not 0 <= args.architecture_seed < 2**63
        or not 0 <= args.data_seed < 2**63
        or args.updates < 1
        or args.start_position < 0
        or args.eval_batches < 2
        or args.log_every < 1
        or (args.valid_pointer_masks and not args.grounded_pointers)
        or (args.cover_verified_command_mask and not args.token_native_command_mask)
        or (args.token_native_occurrence_command and not args.token_native_command_mask)
        or (
            args.token_native_syntax_graph_command
            and not args.token_native_command_mask
        )
        or (
            args.token_native_occurrence_command
            and args.token_native_syntax_graph_command
        )
        or (
            args.token_native_declaration_binding_command
            and not args.token_native_syntax_graph_command
        )
        or (
            (args.opcode_program_registry is None)
            != (args.opcode_program_registry_sha256 is None)
        )
        or (
            args.registry_projected_opcode_training
            and args.opcode_program_registry is None
        )
        or not math.isfinite(args.learning_rate)
        or not 0.0 < args.learning_rate < 1.0
        or not math.isfinite(args.gradient_clip)
        or args.gradient_clip <= 0.0
        or not math.isfinite(args.semantic_prefix_weight)
        or args.semantic_prefix_weight < 0.0
        or args.output.exists()
        or args.output.is_symlink()
        or not args.output.parent.is_dir()
    ):
        raise ParallelTransactionPilotError(
            "parallel transaction pilot arguments differ"
        )


def _balanced_categorical_loss(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    if (
        probabilities.ndim != 3
        or targets.shape != probabilities.shape[:2]
        or mask.shape != targets.shape
        or targets.dtype != torch.long
        or mask.dtype != torch.bool
    ):
        raise ParallelTransactionPilotError(
            "parallel transaction target geometry differs"
        )
    selected_targets = targets[mask]
    if selected_targets.numel() == 0:
        return None
    selected = (
        probabilities[mask]
        .gather(
            1,
            selected_targets[:, None],
        )
        .squeeze(1)
    )
    losses = -selected.float().clamp_min(torch.finfo(torch.float32).eps).log()
    class_means = [
        losses[selected_targets.eq(class_index)].mean()
        for class_index in selected_targets.unique(sorted=True)
    ]
    return torch.stack(class_means).mean()


def _opcode_program_matches(targets, compiler) -> torch.Tensor:
    if (
        compiler is None
        or compiler.opcode_program_table is None
        or compiler.opcode_program_step_mask is None
    ):
        raise ParallelTransactionPilotError("opcode program loss geometry differs")
    steps = targets.opcode.shape[1]
    if (
        compiler.opcode_program_table.ndim != 2
        or compiler.opcode_program_step_mask.shape
        != compiler.opcode_program_table.shape
        or compiler.opcode_program_table.shape[1] < steps
    ):
        raise ParallelTransactionPilotError("opcode program loss geometry differs")
    return (
        compiler.opcode_program_step_mask[:, :steps].unsqueeze(0)
        == targets.step_mask.unsqueeze(1)
    ).all(-1) & (
        (
            compiler.opcode_program_table[:, :steps].unsqueeze(0)
            == targets.opcode.unsqueeze(1)
        )
        | ~targets.step_mask.unsqueeze(1)
    ).all(-1)


def _schedule_loss(
    schedule,
    targets,
    compiler: ParallelAddressedTransactionCompiler | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    masks = policy_masks(targets)
    losses = {}
    for name in _FIELDS:
        loss = _balanced_categorical_loss(
            getattr(schedule, name),
            getattr(targets, name),
            masks[name],
        )
        if loss is not None:
            losses[name] = loss
    program_probabilities = getattr(schedule, "program_probabilities", None)
    if program_probabilities is not None:
        if (
            compiler is None
            or compiler.opcode_program_table is None
            or program_probabilities.ndim != 2
            or program_probabilities.shape[1] != compiler.opcode_program_table.shape[0]
        ):
            raise ParallelTransactionPilotError("opcode program loss geometry differs")
        matches = _opcode_program_matches(targets, compiler)
        if not bool(matches.sum(-1).eq(1).all()):
            raise ParallelTransactionPilotError(
                "opcode program target is absent or ambiguous"
            )
        selected = program_probabilities.gather(
            1,
            matches.long().argmax(-1, keepdim=True),
        ).squeeze(1)
        losses["program"] = (
            -selected.float().clamp_min(torch.finfo(torch.float32).eps).log().mean()
        )
    if not losses:
        raise ParallelTransactionPilotError("parallel transaction loss has no support")
    return torch.stack(tuple(losses.values())).mean(), {
        name: float(value.detach().cpu()) for name, value in losses.items()
    }


def _balanced_binary_brier(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Score sparse binary state coordinates without majority-zero collapse."""

    if (
        predicted.shape != target.shape
        or mask.shape != target.shape
        or mask.dtype != torch.bool
    ):
        raise ParallelTransactionPilotError(
            "parallel transaction binary state geometry differs"
        )
    predicted = predicted.float()
    target = target.float()
    if not bool(
        ((predicted >= 0.0) & (predicted <= 1.0)).all()
        and ((target == 0.0) | (target == 1.0)).all()
    ):
        raise ParallelTransactionPilotError(
            "parallel transaction binary state probabilities differ"
        )
    losses = (predicted - target).square()
    positive = mask & target.bool()
    negative = mask & ~target.bool()
    class_means = []
    if bool(positive.any()):
        class_means.append(losses[positive].mean())
    if bool(negative.any()):
        class_means.append(losses[negative].mean())
    if not class_means:
        raise ParallelTransactionPilotError(
            "parallel transaction binary state mask is empty"
        )
    return torch.stack(class_means).mean()


def _categorical_brier(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if (
        predicted.shape != target.shape
        or predicted.ndim != 3
        or mask.shape != predicted.shape[:2]
        or mask.dtype != torch.bool
        or not bool(mask.any())
    ):
        raise ParallelTransactionPilotError(
            "parallel transaction categorical state geometry differs"
        )
    return (predicted.float() - target.float()).square().sum(-1)[mask].mean()


def _state_brier(
    predicted,
    target,
    *,
    slot_mask: torch.Tensor,
    relation_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    categorical_mask = slot_mask & target.active.bool()
    parts = {
        "active": _balanced_binary_brier(
            predicted.active,
            target.active,
            slot_mask,
        ),
        "committed": _balanced_binary_brier(
            predicted.committed,
            target.committed,
            torch.ones_like(target.committed, dtype=torch.bool),
        ),
        "halted": _balanced_binary_brier(
            predicted.halted,
            target.halted,
            torch.ones_like(target.halted, dtype=torch.bool),
        ),
        "relations": _balanced_binary_brier(
            predicted.relations,
            target.relations,
            relation_mask,
        ),
        "root": _balanced_binary_brier(
            predicted.root,
            target.root,
            slot_mask,
        ),
        "type_index": _categorical_brier(
            predicted.type_probabilities,
            target.type_probabilities,
            categorical_mask,
        ),
        "value_code": _categorical_brier(
            predicted.value_probabilities,
            target.value_probabilities,
            categorical_mask,
        ),
    }
    return torch.stack(tuple(parts.values())).mean(), parts


def _semantic_prefix_loss(
    schedule,
    executor,
    initial,
    transaction_targets,
    packet_targets,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Align the schedule with every exact state prefix it creates."""

    predicted = initial
    oracle = initial
    losses = []
    totals: dict[str, torch.Tensor] = {}
    steps = transaction_targets.opcode.shape[1]
    for step in range(steps):
        predicted = executor.apply(
            predicted,
            schedule.policy(step),
            hard=False,
            validate=False,
        )
        with torch.no_grad():
            oracle = executor.apply(
                oracle,
                target_policy(
                    transaction_targets,
                    executor.config,
                    step,
                    dtype=initial.active.dtype,
                ),
                hard=True,
                validate=False,
            )
        loss, parts = _state_brier(
            predicted,
            oracle,
            slot_mask=packet_targets.slot_mask,
            relation_mask=packet_targets.relation_mask,
        )
        losses.append(loss)
        for name, value in parts.items():
            totals[name] = totals.get(name, value * 0.0) + value
    return torch.stack(losses).mean(), {
        name: float((value / steps).detach().cpu())
        for name, value in sorted(totals.items())
    }


def _merge_counts(destination, source) -> None:
    for name, (correct, total) in source.items():
        values = destination.setdefault(name, [0, 0])
        values[0] += int(correct)
        values[1] += int(total)


def _summarize_counts(counts) -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "correct": values[0],
            "total": values[1],
            "rate": values[0] / values[1] if values[1] else 0.0,
        }
        for name, values in sorted(counts.items())
    }


def _schedule_counts(schedule, targets) -> dict[str, tuple[int, int]]:
    masks = policy_masks(targets)
    counts = {}
    joint = torch.ones_like(targets.step_mask)
    for name in _FIELDS:
        mask = masks[name]
        prediction = getattr(schedule, f"applied_{name}").argmax(-1)
        correct = prediction.eq(getattr(targets, name))
        counts[name] = (
            int((correct & mask).sum().detach().cpu()),
            int(mask.sum().detach().cpu()),
        )
        joint &= correct | ~mask
    counts["joint"] = (
        int((joint & targets.step_mask).sum().detach().cpu()),
        int(targets.step_mask.sum().detach().cpu()),
    )
    return counts


def _program_statistics(schedule, targets, compiler) -> dict[str, object] | None:
    probabilities = getattr(schedule, "program_probabilities", None)
    if probabilities is None:
        return None
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] != targets.opcode.shape[0]
        or compiler.opcode_program_table is None
        or probabilities.shape[1] != compiler.opcode_program_table.shape[0]
    ):
        raise ParallelTransactionPilotError("opcode program metric geometry differs")
    matches = _opcode_program_matches(targets, compiler)
    if not bool(matches.sum(-1).le(1).all()):
        raise ParallelTransactionPilotError("opcode program target is ambiguous")
    known = matches.any(-1)
    predicted = probabilities.argmax(-1)
    target = matches.long().argmax(-1)
    entropy = -(
        probabilities.float()
        * probabilities.float().clamp_min(torch.finfo(torch.float32).eps).log()
    ).sum(-1)
    return {
        "correct": int((predicted.eq(target) & known).sum().detach().cpu()),
        "known": int(known.sum().detach().cpu()),
        "probability_entropy_sum": float(entropy.sum().detach().cpu()),
        "rows": int(probabilities.shape[0]),
        "selected_counts": torch.bincount(
            predicted,
            minlength=probabilities.shape[1],
        )
        .detach()
        .cpu(),
    }


def _evaluate_interfaces(
    schedule_compiler,
    executor,
    model,
    *,
    stream,
    packet_index,
    device,
    data_seed: int,
    max_batches: int,
) -> dict[str, object]:
    schedule_compiler.eval()
    schedule_counts = {}
    terminal_counts = {}
    program_correct = 0
    program_known = 0
    program_entropy_sum = 0.0
    program_rows = 0
    program_selected_counts = None
    iterator = stream.iter_positioned_batches(
        "development",
        rank=0,
        world_size=1,
        epoch=0,
        seed=data_seed,
    )
    observed = 0
    for _position, cpu_batch in iterator:
        if observed >= max_batches:
            break
        packet_index.verify_validation((cpu_batch,))
        batch = move_continuation_batch(cpu_batch, device)
        initial = packet_targets_to_state(
            batch.packet_targets,
            model.config,
            step=0,
            dtype=next(schedule_compiler.parameters()).dtype,
        )
        with torch.inference_mode():
            command_hidden = model._encode_to_stage(
                batch.episodes.command.tokens,
                pos=0,
            )
            schedule = schedule_compiler(
                initial,
                command_hidden=command_hidden,
                command_tokens=(
                    batch.episodes.command.tokens
                    if schedule_compiler.token_native_command_mask
                    else None
                ),
                command_attention_mask=batch.episodes.command.attention_mask.bool(),
                steps=batch.transaction_targets.opcode.shape[1],
                hard=True,
            )
            terminal = initial
            for step in range(batch.transaction_targets.opcode.shape[1]):
                terminal = executor.apply(
                    terminal,
                    schedule.policy(step),
                    hard=True,
                    validate=False,
                )
        _merge_counts(
            schedule_counts,
            _schedule_counts(schedule, batch.transaction_targets),
        )
        _merge_counts(
            terminal_counts,
            _packet_batch_counts(terminal, batch.terminal_packet_targets),
        )
        statistics = _program_statistics(
            schedule,
            batch.transaction_targets,
            schedule_compiler,
        )
        if statistics is not None:
            program_correct += statistics["correct"]
            program_known += statistics["known"]
            program_entropy_sum += statistics["probability_entropy_sum"]
            program_rows += statistics["rows"]
            selected_counts = statistics["selected_counts"]
            program_selected_counts = (
                selected_counts
                if program_selected_counts is None
                else program_selected_counts + selected_counts
            )
        observed += 1
    if observed != max_batches:
        raise ParallelTransactionPilotError(
            "parallel transaction development split is too short"
        )
    result = {
        "batches": observed,
        "oracle_initial_hard_schedule": _summarize_counts(schedule_counts),
        "oracle_initial_terminal_packet": _summarize_counts(terminal_counts),
    }
    if program_selected_counts is not None:
        selected_probabilities = program_selected_counts.float() / program_rows
        positive = selected_probabilities > 0
        selected_entropy = -(
            selected_probabilities[positive] * selected_probabilities[positive].log()
        ).sum()
        result["opcode_program_selector"] = {
            "classes_used": int(positive.sum()),
            "exact_class": {
                "correct": program_correct,
                "rate": program_correct / program_known if program_known else 0.0,
                "total": program_known,
            },
            "known_target_coverage": {
                "covered": program_known,
                "rate": program_known / program_rows,
                "total": program_rows,
            },
            "mean_probability_entropy": program_entropy_sum / program_rows,
            "selected_class_entropy": float(selected_entropy),
        }
    return result


def _module_sha256(module: torch.nn.Module, path: Path) -> str:
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in module.state_dict().items()
        },
        path,
    )
    os.chmod(path, 0o400)
    return _sha256_file(path)


def _state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        raw = value.detach().cpu().contiguous().view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def _precision_context(is_h100: bool):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if is_h100
        else nullcontext()
    )


def _training_initial_state(
    model,
    batch,
    *,
    source: str,
    dtype: torch.dtype,
) -> TypedTheoryState:
    if source == "oracle":
        return packet_targets_to_state(
            batch.packet_targets,
            model.config,
            step=0,
            dtype=dtype,
        )
    if source == "autonomous":
        with torch.no_grad():
            return model.compile_world(
                batch.episodes.world.tokens,
                attention_mask=batch.episodes.world.attention_mask,
                hard=True,
            ).detached_clone()
    raise ParallelTransactionPilotError(
        "parallel transaction training initial state differs"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    if not torch.cuda.is_available():
        raise ParallelTransactionPilotError("parallel transaction pilot requires CUDA")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    is_h100 = "H100" in torch.cuda.get_device_name(device).upper()
    if args.required_device_class == "h100" and not is_h100:
        raise ParallelTransactionPilotError(
            "parallel transaction pilot requires an H100"
        )

    stream = ETTRV3StreamingRelease(
        args.release_root,
        expected_release_sha256=args.release_sha256,
        data_root=args.data_root,
        tokenizer_path=args.tokenizer,
    )
    source_verification = stream.verify_source_shards()
    model, _joint_payload, provenance, _joint_contract = _strict_load_joint_model(
        args,
        device=device,
    )
    reader, _compiler_contract, reader_parameters, replacement_parameters = (
        _load_compiler(
            args,
            model=model,
            stream=stream,
            device=device,
        )
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    executor = model.reactor
    opcode_registry = (
        load_opcode_program_registry(
            args.opcode_program_registry,
            expected_sha256=args.opcode_program_registry_sha256,
            max_steps=model.config.max_steps,
            opcode_classes=TRANSACTION_COUNT,
        )
        if args.opcode_program_registry is not None
        else None
    )
    torch.manual_seed(args.architecture_seed)
    torch.cuda.manual_seed_all(args.architecture_seed)
    schedule_compiler = ParallelAddressedTransactionCompiler(
        model.config,
        width=args.width,
        layers=args.layers,
        num_heads=args.num_heads,
        grounded_pointers=args.grounded_pointers,
        valid_pointer_masks=args.valid_pointer_masks,
        token_native_command_mask=args.token_native_command_mask,
        cover_verified_command_mask=args.cover_verified_command_mask,
        token_native_occurrence_command=(args.token_native_occurrence_command),
        token_native_syntax_graph_command=(args.token_native_syntax_graph_command),
        token_native_declaration_binding_command=(
            args.token_native_declaration_binding_command
        ),
        token_native_codebook_ids=(
            stream.codec.codebook.token_ids if args.token_native_command_mask else None
        ),
        token_native_codebook_atoms=(
            stream.codec.codebook.atoms if args.cover_verified_command_mask else None
        ),
        token_native_vocab_size=(
            model.base.cfg.vocab_size if args.token_native_command_mask else None
        ),
        opcode_program_sequences=(
            opcode_registry.programs
            if opcode_registry is not None
            and not args.registry_projected_opcode_training
            else None
        ),
    ).to(device=device, dtype=next(model.parameters()).dtype)
    if args.registry_projected_opcode_training:
        if opcode_registry is None:
            raise ParallelTransactionPilotError(
                "registry-projected opcode training lacks a registry"
            )
        schedule_compiler = RegistryProjectedAddressedScheduleCompiler(
            schedule_compiler,
            opcode_registry.programs,
        )
    schedule_parameters = sum(
        parameter.numel() for parameter in schedule_compiler.parameters()
    )
    removed_reactor_parameters = sum(
        parameter.numel() for parameter in executor.parameters()
    )
    complete_parameters = (
        replacement_parameters - removed_reactor_parameters + schedule_parameters
    )
    optimizer = torch.optim.AdamW(
        schedule_compiler.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )
    packet_index = ETTRDiskPacketSufficiencyIndex(stream.packet_index_root)
    objective_config = ETTRObjectiveConfig(vocab_size=model.base.cfg.vocab_size)
    try:
        args.output.mkdir(mode=0o700)
        if opcode_registry is not None:
            _write_no_replace(
                args.output / "opcode-program-registry.json",
                opcode_registry.payload,
            )
        initial_sha256 = _module_sha256(
            schedule_compiler,
            args.output / "schedule-initial.safetensors",
        )
        before_interface = _evaluate_interfaces(
            schedule_compiler,
            executor,
            model,
            stream=stream,
            packet_index=packet_index,
            device=device,
            data_seed=args.data_seed,
            max_batches=args.eval_batches,
        )
        parallel_reactor = ParallelScheduledReactor(
            schedule_compiler,
            model.config,
        )
        model.reactor = parallel_reactor
        deployed_parameters = (
            sum(parameter.numel() for parameter in model.parameters())
            - sum(parameter.numel() for parameter in model.query_reader.parameters())
            + reader_parameters
        )
        if deployed_parameters != complete_parameters:
            raise ParallelTransactionPilotError(
                "parallel transaction deployed parameter count differs"
            )
        before_end_to_end = _evaluate(
            reader,
            model,
            stream=stream,
            packet_index=packet_index,
            device=device,
            data_seed=args.data_seed,
            max_batches=args.eval_batches,
        )
        contract = {
            "architecture": {
                "layers": args.layers,
                "grounded_pointers": args.grounded_pointers,
                "num_heads": args.num_heads,
                "opcode_program_classes": (
                    opcode_registry.classes if opcode_registry is not None else None
                ),
                "opcode_program_development_instance_coverage": (
                    opcode_registry.development_instance_coverage
                    if opcode_registry is not None
                    else None
                ),
                "opcode_program_registry_payload_sha256": (
                    opcode_registry.payload_sha256
                    if opcode_registry is not None
                    else None
                ),
                "opcode_program_registry_sha256": (
                    opcode_registry.file_sha256 if opcode_registry is not None else None
                ),
                "registry_projected_opcode_training": (
                    args.registry_projected_opcode_training
                ),
                "parameterless_exact_algebra": True,
                "removed_recurrent_policy_parameters": (removed_reactor_parameters),
                "seed": args.architecture_seed,
                "sticky_schedule": True,
                "token_native_command_mask": (args.token_native_command_mask),
                "cover_verified_command_mask": (
                    args.cover_verified_command_mask
                ),
                "token_native_occurrence_command": (
                    args.token_native_occurrence_command
                ),
                "token_native_syntax_graph_command": (
                    args.token_native_syntax_graph_command
                ),
                "token_native_declaration_binding_command": (
                    args.token_native_declaration_binding_command
                ),
                "token_native_codebook_sha256": (
                    stream.codec.codebook_sha256
                    if args.token_native_command_mask
                    else None
                ),
                "valid_pointer_masks": args.valid_pointer_masks,
                "width": args.width,
            },
            "compiler_contract_sha256": args.compiler_contract_sha256,
            "compiler_sha256": args.compiler_sha256,
            "complete_system_parameters": complete_parameters,
            "data_seed": args.data_seed,
            "eval_batches": args.eval_batches,
            "gradient_clip": args.gradient_clip,
            "joint_model_sha256": args.joint_model_sha256,
            "joint_run_contract_sha256": args.joint_run_contract_sha256,
            "learning_rate": args.learning_rate,
            "protected_checkpoint_sha256": provenance.checkpoint_sha256,
            "reader_parameters": reader_parameters,
            "release_file_sha256": args.release_sha256,
            "required_device_class": args.required_device_class,
            "schedule_parameters": schedule_parameters,
            "schema": CONTRACT_SCHEMA,
            "semantic_prefix_weight": args.semantic_prefix_weight,
            "source_commit": args.source_commit,
            "start_position": args.start_position,
            "training_initial_state": args.training_initial_state,
            "updates": args.updates,
        }
        contract_sha256 = _write_no_replace(
            args.output / "pilot-contract.json",
            _canonical_bytes(contract),
        )
        _write_no_replace(args.output / "train.jsonl", b"", mode=0o600)

        iterator = iter_batches_with_query_specs(
            stream,
            "train",
            epoch=0,
            seed=args.data_seed,
            start_position=args.start_position,
        )
        schedule_compiler.train()
        last_loss = None
        last_schedule_loss = None
        last_semantic_prefix_loss = None
        last_position = args.start_position
        for update in range(1, args.updates + 1):
            try:
                last_position, cpu_batch, _cpu_specs = next(iterator)
            except StopIteration as exc:
                raise ParallelTransactionPilotError(
                    "parallel transaction train stream exhausted"
                ) from exc
            packet_index.verify_train((cpu_batch,))
            batch = move_continuation_batch(cpu_batch, device)
            batch.validate(model.config, objective_config)
            initial = _training_initial_state(
                model,
                batch,
                source=args.training_initial_state,
                dtype=next(schedule_compiler.parameters()).dtype,
            )
            with torch.no_grad():
                command_hidden = model._encode_to_stage(
                    batch.episodes.command.tokens,
                    pos=0,
                )
            optimizer.zero_grad(set_to_none=True)
            with _precision_context(is_h100):
                schedule = schedule_compiler(
                    initial,
                    command_hidden=command_hidden.detach(),
                    command_tokens=(
                        batch.episodes.command.tokens
                        if args.token_native_command_mask
                        else None
                    ),
                    command_attention_mask=(
                        batch.episodes.command.attention_mask.bool()
                    ),
                    steps=batch.transaction_targets.opcode.shape[1],
                    hard=False,
                )
                schedule_loss, parts = _schedule_loss(
                    schedule,
                    batch.transaction_targets,
                    schedule_compiler,
                )
                if args.semantic_prefix_weight > 0.0:
                    semantic_prefix_loss, semantic_prefix_parts = _semantic_prefix_loss(
                        schedule,
                        executor,
                        initial,
                        batch.transaction_targets,
                        batch.terminal_packet_targets,
                    )
                else:
                    semantic_prefix_loss = schedule_loss * 0.0
                    semantic_prefix_parts = {}
                loss = (
                    schedule_loss + args.semantic_prefix_weight * semantic_prefix_loss
                )
            if not bool(torch.isfinite(loss)):
                raise ParallelTransactionPilotError(
                    "parallel transaction loss is nonfinite"
                )
            last_loss = float(loss.detach().cpu())
            last_schedule_loss = float(schedule_loss.detach().cpu())
            last_semantic_prefix_loss = float(semantic_prefix_loss.detach().cpu())
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                schedule_compiler.parameters(),
                args.gradient_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
            if update % args.log_every == 0 or update == args.updates:
                metric = {
                    "gradient_norm_pre_clip": float(
                        gradient_norm.detach().float().cpu()
                    ),
                    "loss": last_loss,
                    "parts": parts,
                    "position": last_position,
                    "schema": "shohin-ettr-parallel-addressed-transaction-metric-v2",
                    "semantic_prefix_loss": last_semantic_prefix_loss,
                    "semantic_prefix_parts": semantic_prefix_parts,
                    "schedule_loss": last_schedule_loss,
                    "update": update,
                }
                with (args.output / "train.jsonl").open("ab", buffering=0) as log:
                    log.write(_canonical_bytes(metric))
        os.chmod(args.output / "train.jsonl", 0o400)

        schedule_compiler.eval()
        after_interface = _evaluate_interfaces(
            schedule_compiler,
            executor,
            model,
            stream=stream,
            packet_index=packet_index,
            device=device,
            data_seed=args.data_seed,
            max_batches=args.eval_batches,
        )
        after_end_to_end = _evaluate(
            reader,
            model,
            stream=stream,
            packet_index=packet_index,
            device=device,
            data_seed=args.data_seed,
            max_batches=args.eval_batches,
        )
        final_sha256 = _module_sha256(
            schedule_compiler,
            args.output / "schedule-final.safetensors",
        )
        report = {
            "after_end_to_end": after_end_to_end,
            "after_interface": after_interface,
            "before_end_to_end": before_end_to_end,
            "before_interface": before_interface,
            "contract_sha256": contract_sha256,
            "final_schedule_sha256": final_sha256,
            "initial_schedule_sha256": initial_sha256,
            "last_loss": last_loss,
            "last_position": last_position,
            "last_schedule_loss": last_schedule_loss,
            "last_semantic_prefix_loss": last_semantic_prefix_loss,
            "protected_checkpoint_sha256": provenance.checkpoint_sha256,
            "runtime_precision": str(next(schedule_compiler.parameters()).dtype),
            "schema": REPORT_SCHEMA,
            "source_verification": source_verification,
            "status": "pass",
            "updates_completed": args.updates,
        }
        _write_no_replace(
            args.output / "report.json",
            _canonical_bytes(report),
        )
        sums = []
        for name in (
            "pilot-contract.json",
            "report.json",
            "schedule-final.safetensors",
            "schedule-initial.safetensors",
            "train.jsonl",
        ) + (("opcode-program-registry.json",) if opcode_registry is not None else ()):
            sums.append(f"{_sha256_file(args.output / name)}  {name}\n")
        _write_no_replace(
            args.output / "SHA256SUMS",
            "".join(sums).encode("ascii"),
        )
        for path in args.output.iterdir():
            os.chmod(path, 0o400)
        os.chmod(args.output, 0o500)
    except BaseException:
        if args.output.exists():
            shutil.rmtree(args.output, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "complete_system_parameters": complete_parameters,
                "final_schedule_state_sha256": _state_sha256(schedule_compiler),
                "output": str(args.output),
                "schedule_parameters": schedule_parameters,
                "status": "pass",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
