#!/usr/bin/env python3
"""Fit ETTR state construction through the fixed query algebra.

The pretrained language backbone and the successful source-query compiler are
frozen.  Only the ETTR WORLD compiler and COMMAND reactor are optimized.  A
fully autonomous source-deleted causal answer loss flows through hard
straight-through states and the fixed algebraic executor, while packet and
transaction supervision retain the intended typed-state semantics.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Sequence

from safetensors.torch import save_file
import torch
import torch.nn.functional as F

from eval_algebraic_query_joint_state import (
    _evaluate,
    _load_compiler,
    _reader_forward,
    _strict_load_joint_model,
)
from ettr_objectives import ETTRObjectiveConfig, _causal_query_binding_loss
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_query_supervision import iter_batches_with_query_specs
from ettr_v3_streaming import ETTRV3StreamingRelease, move_continuation_batch
from eval_ettr_v3 import _parameter_sha256
from probe_ettr_oracle_interfaces import packet_targets_to_state, target_policy
from train_ettr_component_island import (
    _canonical_bytes,
    _component_state,
    _reader_pairs_from_logits,
    _sha256_file,
    _write_no_replace,
    component_loss,
)
from train_typed_query_state_reader_pilot import _truth_loss


CONTRACT_SCHEMA = "shohin-ettr-algebraic-state-semantic-contract-v1"
REPORT_SCHEMA = "shohin-ettr-algebraic-state-semantic-report-v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class AlgebraicStateSemanticError(RuntimeError):
    """The algebraic state-semantic pilot violated its sealed contract."""


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
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--start-position", type=int, default=13_200)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--compiler-aux-weight", type=float, default=0.25)
    parser.add_argument("--reactor-aux-weight", type=float, default=0.25)
    parser.add_argument(
        "--optimization-mode",
        choices=("joint-global", "causal-owner-alternating"),
        default="joint-global",
    )
    parser.add_argument(
        "--semantic-program-source",
        choices=("predicted", "oracle"),
        default="predicted",
    )
    parser.add_argument(
        "--owner-state-bridge",
        choices=("autonomous", "oracle-factors"),
        default="autonomous",
    )
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--reactor-reduction",
        choices=("decision-mean", "head-class-balanced"),
        default="head-class-balanced",
    )
    parser.add_argument(
        "--required-device-class",
        choices=("h100", "cuda"),
        default="h100",
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
    )
    hashes = (
        args.release_sha256,
        args.joint_model_sha256,
        args.joint_run_contract_sha256,
        args.compiler_sha256,
        args.compiler_contract_sha256,
    )
    if (
        any(not path.is_absolute() for path in paths)
        or any(_HEX64.fullmatch(value) is None for value in hashes)
        or _HEX40.fullmatch(args.source_commit) is None
        or not 0 <= args.data_seed < 2**63
        or args.updates < 1
        or args.start_position < 0
        or args.eval_batches < 2
        or args.log_every < 1
        or not math.isfinite(args.learning_rate)
        or not 0.0 < args.learning_rate < 1.0
        or not math.isfinite(args.gradient_clip)
        or args.gradient_clip <= 0.0
        or not math.isfinite(args.compiler_aux_weight)
        or not math.isfinite(args.reactor_aux_weight)
        or args.compiler_aux_weight < 0.0
        or args.reactor_aux_weight < 0.0
        or args.compiler_aux_weight + args.reactor_aux_weight <= 0.0
        or (
            args.owner_state_bridge == "oracle-factors"
            and args.optimization_mode != "causal-owner-alternating"
        )
        or args.output.exists()
        or args.output.is_symlink()
        or not args.output.parent.is_dir()
    ):
        raise AlgebraicStateSemanticError(
            "algebraic state-semantic arguments differ"
        )


def _set_training_ownership(model, reader) -> tuple[list[torch.Tensor], int]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in reader.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for module in (model.compiler, model.reactor):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    if not trainable:
        raise AlgebraicStateSemanticError(
            "algebraic state-semantic ownership is empty"
        )
    return trainable, sum(parameter.numel() for parameter in trainable)


def _owner_parameters(model, owner: str) -> list[torch.Tensor]:
    if owner not in {"compiler", "reactor"}:
        raise AlgebraicStateSemanticError(
            "algebraic state-semantic owner differs"
        )
    module = getattr(model, owner)
    return list(module.parameters())


def _set_active_owner(model, owner: str | None) -> None:
    for name in ("compiler", "reactor"):
        active = owner is None or name == owner
        for parameter in getattr(model, name).parameters():
            parameter.requires_grad_(active)


def _factor_truth_loss(
    logits: torch.Tensor,
    target_batch,
    *,
    factor: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if factor not in {"world", "command"}:
        raise AlgebraicStateSemanticError(
            "algebraic state-semantic factor differs"
        )
    targets = target_batch.episodes.query.targets.gather(
        1,
        target_batch.episodes.query_read_index[:, None],
    ).squeeze(1)
    factual = F.cross_entropy(logits.float(), targets)
    pairs = _reader_pairs_from_logits(logits, target_batch)
    binding = _causal_query_binding_loss(
        pairs[factor],
        margin=1.0,
        reduction="balanced-logmeanexp",
        classification_weight=0.25,
        effect_weight=1.0,
        invariance_weight=0.25,
        risk_temperature=0.25,
    )[0]
    return 0.125 * factual + 0.5 * binding, {
        f"{factor}_binding": binding,
        "factual": factual,
    }


def _semantic_loss(
    model,
    reader,
    batch,
    specs,
    *,
    factor: str | None = None,
    oracle_program: bool = False,
    owner_state_bridge: str = "autonomous",
):
    initial_state, terminal_state = _semantic_states(
        model,
        batch,
        factor=factor,
        owner_state_bridge=owner_state_bridge,
    )
    output = _reader_forward(
        reader,
        batch,
        specs,
        initial_state,
        terminal_state,
        oracle_program=oracle_program,
    )
    if factor is None:
        return _truth_loss(output.vocab_logits, batch)
    return _factor_truth_loss(output.vocab_logits, batch, factor=factor)


def _semantic_states(
    model,
    batch,
    *,
    factor: str | None,
    owner_state_bridge: str,
):
    if owner_state_bridge not in {"autonomous", "oracle-factors"}:
        raise AlgebraicStateSemanticError(
            "algebraic state-semantic owner state bridge differs"
        )
    use_oracle_initial = (
        owner_state_bridge == "oracle-factors" and factor == "command"
    )
    if use_oracle_initial:
        initial_state = packet_targets_to_state(
            batch.packet_targets,
            model.config,
            step=0,
            dtype=next(model.reactor.parameters()).dtype,
        )
    else:
        compiler_context = (
            torch.no_grad() if factor == "command" else nullcontext()
        )
        with compiler_context:
            initial_state = model.compile_world(
                batch.episodes.world.tokens,
                attention_mask=batch.episodes.world.attention_mask,
                hard=True,
            )

    use_oracle_actions = (
        owner_state_bridge == "oracle-factors" and factor == "world"
    )
    if use_oracle_actions:
        terminal_state = initial_state
        for step in range(batch.transaction_targets.opcode.shape[1]):
            terminal_state = model.reactor.apply(
                terminal_state,
                target_policy(
                    batch.transaction_targets,
                    model.config,
                    step,
                    dtype=terminal_state.active.dtype,
                ),
                hard=True,
                validate=False,
            )
    else:
        terminal_state, _trace = model.execute(
            initial_state,
            steps=batch.transaction_targets.opcode.shape[1],
            hard=True,
            command_idx=batch.episodes.command.tokens,
            command_attention_mask=batch.episodes.command.attention_mask,
        )
    return initial_state, terminal_state


def _precision_context(is_h100: bool):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if is_h100
        else nullcontext()
    )


def _require_finite(loss: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(loss)):
        raise AlgebraicStateSemanticError(
            f"algebraic state-semantic {label} loss is nonfinite"
        )


def _joint_update(
    model,
    reader,
    batch,
    specs,
    *,
    optimizer,
    trainable,
    args,
    is_h100: bool,
) -> dict[str, object]:
    optimizer.zero_grad(set_to_none=True)
    with _precision_context(is_h100):
        semantic, semantic_parts = _semantic_loss(
            model,
            reader,
            batch,
            specs,
            oracle_program=args.semantic_program_source == "oracle",
        )
    _require_finite(semantic, "answer")
    semantic_value = float(semantic.detach().cpu())
    semantic_part_values = {
        name: float(value.detach().cpu())
        for name, value in semantic_parts.items()
    }
    semantic.backward()
    del semantic, semantic_parts

    with _precision_context(is_h100):
        compiler_aux, compiler_parts = component_loss(model, batch, "compiler")
    _require_finite(compiler_aux, "compiler")
    compiler_value = float(compiler_aux.detach().cpu())
    (args.compiler_aux_weight * compiler_aux).backward()
    del compiler_aux

    with _precision_context(is_h100):
        reactor_aux, reactor_parts = component_loss(
            model,
            batch,
            "reactor",
            reactor_reduction=args.reactor_reduction,
        )
    _require_finite(reactor_aux, "reactor")
    reactor_value = float(reactor_aux.detach().cpu())
    (args.reactor_aux_weight * reactor_aux).backward()
    del reactor_aux

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        trainable,
        args.gradient_clip,
        error_if_nonfinite=True,
    )
    optimizer.step()
    return {
        "compiler_aux": compiler_value,
        "compiler_parts": compiler_parts,
        "gradient_norm_pre_clip": float(
            gradient_norm.detach().float().cpu()
        ),
        "loss": (
            semantic_value
            + args.compiler_aux_weight * compiler_value
            + args.reactor_aux_weight * reactor_value
        ),
        "reactor_aux": reactor_value,
        "reactor_parts": reactor_parts,
        "semantic": semantic_value,
        "semantic_parts": semantic_part_values,
    }


def _causal_owner_update(
    model,
    reader,
    batch,
    specs,
    *,
    optimizers,
    owner_parameters,
    args,
    is_h100: bool,
) -> dict[str, object]:
    _set_active_owner(model, "compiler")
    optimizers["compiler"].zero_grad(set_to_none=True)
    with _precision_context(is_h100):
        world_semantic, world_parts = _semantic_loss(
            model,
            reader,
            batch,
            specs,
            factor="world",
            oracle_program=args.semantic_program_source == "oracle",
            owner_state_bridge=args.owner_state_bridge,
        )
    _require_finite(world_semantic, "WORLD answer")
    world_value = float(world_semantic.detach().cpu())
    world_part_values = {
        f"world_owner_{name}": float(value.detach().cpu())
        for name, value in world_parts.items()
    }
    world_semantic.backward()
    del world_semantic, world_parts
    with _precision_context(is_h100):
        compiler_aux, compiler_parts = component_loss(model, batch, "compiler")
    _require_finite(compiler_aux, "compiler")
    compiler_value = float(compiler_aux.detach().cpu())
    (args.compiler_aux_weight * compiler_aux).backward()
    del compiler_aux
    compiler_norm = torch.nn.utils.clip_grad_norm_(
        owner_parameters["compiler"],
        args.gradient_clip,
        error_if_nonfinite=True,
    )
    optimizers["compiler"].step()

    _set_active_owner(model, "reactor")
    optimizers["reactor"].zero_grad(set_to_none=True)
    with _precision_context(is_h100):
        command_semantic, command_parts = _semantic_loss(
            model,
            reader,
            batch,
            specs,
            factor="command",
            oracle_program=args.semantic_program_source == "oracle",
            owner_state_bridge=args.owner_state_bridge,
        )
    _require_finite(command_semantic, "COMMAND answer")
    command_value = float(command_semantic.detach().cpu())
    command_part_values = {
        f"command_owner_{name}": float(value.detach().cpu())
        for name, value in command_parts.items()
    }
    command_semantic.backward()
    del command_semantic, command_parts
    with _precision_context(is_h100):
        reactor_aux, reactor_parts = component_loss(
            model,
            batch,
            "reactor",
            reactor_reduction=args.reactor_reduction,
        )
    _require_finite(reactor_aux, "reactor")
    reactor_value = float(reactor_aux.detach().cpu())
    (args.reactor_aux_weight * reactor_aux).backward()
    del reactor_aux
    reactor_norm = torch.nn.utils.clip_grad_norm_(
        owner_parameters["reactor"],
        args.gradient_clip,
        error_if_nonfinite=True,
    )
    optimizers["reactor"].step()
    _set_active_owner(model, None)

    semantic_value = world_value + command_value
    return {
        "compiler_aux": compiler_value,
        "compiler_parts": compiler_parts,
        "gradient_norm_pre_clip": {
            "compiler": float(compiler_norm.detach().float().cpu()),
            "reactor": float(reactor_norm.detach().float().cpu()),
        },
        "loss": (
            semantic_value
            + args.compiler_aux_weight * compiler_value
            + args.reactor_aux_weight * reactor_value
        ),
        "reactor_aux": reactor_value,
        "reactor_parts": reactor_parts,
        "semantic": semantic_value,
        "semantic_parts": world_part_values | command_part_values,
    }


def _save_component(model, component: str, path: Path) -> str:
    save_file(_component_state(model, component), path)
    os.chmod(path, 0o400)
    return _sha256_file(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    if not torch.cuda.is_available():
        raise AlgebraicStateSemanticError(
            "algebraic state-semantic pilot requires CUDA"
        )
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    device_name = torch.cuda.get_device_name(device)
    is_h100 = "H100" in device_name.upper()
    if args.required_device_class == "h100" and not is_h100:
        raise AlgebraicStateSemanticError(
            "algebraic state-semantic pilot requires an H100"
        )

    stream = ETTRV3StreamingRelease(
        args.release_root,
        expected_release_sha256=args.release_sha256,
        data_root=args.data_root,
        tokenizer_path=args.tokenizer,
    )
    source_verification = stream.verify_source_shards()
    model, joint_payload, provenance, joint_contract = _strict_load_joint_model(
        args,
        device=device,
    )
    if not is_h100:
        model.to(dtype=torch.float32)
    (
        reader,
        compiler_contract,
        reader_parameters,
        replacement_system_parameters,
    ) = _load_compiler(
        args,
        model=model,
        stream=stream,
        device=device,
    )
    trainable, trainable_parameters = _set_training_ownership(model, reader)
    owner_parameters = {
        owner: _owner_parameters(model, owner)
        for owner in ("compiler", "reactor")
    }
    optimizer = None
    optimizers = None
    if args.optimization_mode == "joint-global":
        optimizer = torch.optim.AdamW(
            trainable,
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01,
            fused=True,
        )
    else:
        optimizers = {
            owner: torch.optim.AdamW(
                parameters,
                lr=args.learning_rate,
                betas=(0.9, 0.95),
                weight_decay=0.01,
                fused=True,
            )
            for owner, parameters in owner_parameters.items()
        }
    packet_index = ETTRDiskPacketSufficiencyIndex(stream.packet_index_root)
    objective_config = ETTRObjectiveConfig(vocab_size=model.base.cfg.vocab_size)
    try:
        args.output.mkdir(mode=0o700)
        initial_compiler_sha256 = _save_component(
            model,
            "compiler",
            args.output / "compiler-initial.safetensors",
        )
        initial_reactor_sha256 = _save_component(
            model,
            "reactor",
            args.output / "reactor-initial.safetensors",
        )
        before_parameter_sha256 = _parameter_sha256(model)
        before = _evaluate(
            reader,
            model,
            stream=stream,
            packet_index=packet_index,
            device=device,
            data_seed=args.data_seed,
            max_batches=args.eval_batches,
        )
        contract = {
            "compiler_aux_weight": args.compiler_aux_weight,
            "compiler_contract_sha256": args.compiler_contract_sha256,
            "compiler_sha256": args.compiler_sha256,
            "data_seed": args.data_seed,
            "eval_batches": args.eval_batches,
            "gradient_clip": args.gradient_clip,
            "joint_model_sha256": args.joint_model_sha256,
            "joint_run_contract_sha256": args.joint_run_contract_sha256,
            "learning_rate": args.learning_rate,
            "optimization_mode": args.optimization_mode,
            "owner_state_bridge": args.owner_state_bridge,
            "reader_parameters": reader_parameters,
            "reactor_aux_weight": args.reactor_aux_weight,
            "reactor_reduction": args.reactor_reduction,
            "release_file_sha256": args.release_sha256,
            "replacement_system_parameters": replacement_system_parameters,
            "required_device_class": args.required_device_class,
            "runtime_precision": str(next(model.parameters()).dtype),
            "schema": CONTRACT_SCHEMA,
            "semantic_program_source": args.semantic_program_source,
            "source_commit": args.source_commit,
            "start_position": args.start_position,
            "trainable_components": ["compiler", "reactor"],
            "trainable_parameters": trainable_parameters,
            "updates": args.updates,
        }
        contract_sha256 = _write_no_replace(
            args.output / "pilot-contract.json",
            _canonical_bytes(contract),
        )
        _write_no_replace(args.output / "train.jsonl", b"", mode=0o600)

        model.eval()
        model.compiler.train()
        model.reactor.train()
        reader.eval()
        iterator = iter_batches_with_query_specs(
            stream,
            "train",
            epoch=0,
            seed=args.data_seed,
            start_position=args.start_position,
        )
        last_loss = None
        last_position = args.start_position
        for update in range(1, args.updates + 1):
            try:
                last_position, cpu_batch, cpu_specs = next(iterator)
            except StopIteration as exc:
                raise AlgebraicStateSemanticError(
                    "algebraic state-semantic train stream exhausted"
                ) from exc
            packet_index.verify_train((cpu_batch,))
            batch = move_continuation_batch(cpu_batch, device)
            specs = cpu_specs.to(device)
            batch.validate(model.config, objective_config)
            if args.optimization_mode == "joint-global":
                assert optimizer is not None
                metric = _joint_update(
                    model,
                    reader,
                    batch,
                    specs,
                    optimizer=optimizer,
                    trainable=trainable,
                    args=args,
                    is_h100=is_h100,
                )
            else:
                assert optimizers is not None
                metric = _causal_owner_update(
                    model,
                    reader,
                    batch,
                    specs,
                    optimizers=optimizers,
                    owner_parameters=owner_parameters,
                    args=args,
                    is_h100=is_h100,
                )
            last_loss = float(metric["loss"])
            if update % args.log_every == 0 or update == args.updates:
                metric |= {
                    "position": last_position,
                    "schema": "shohin-ettr-algebraic-state-semantic-metric-v1",
                    "update": update,
                }
                with (args.output / "train.jsonl").open("ab", buffering=0) as log:
                    log.write(_canonical_bytes(metric))
        os.chmod(args.output / "train.jsonl", 0o400)

        model.eval()
        after = _evaluate(
            reader,
            model,
            stream=stream,
            packet_index=packet_index,
            device=device,
            data_seed=args.data_seed,
            max_batches=args.eval_batches,
        )
        final_compiler_sha256 = _save_component(
            model,
            "compiler",
            args.output / "compiler-final.safetensors",
        )
        final_reactor_sha256 = _save_component(
            model,
            "reactor",
            args.output / "reactor-final.safetensors",
        )
        report = {
            "after_parameter_sha256": _parameter_sha256(model),
            "before_parameter_sha256": before_parameter_sha256,
            "compiler_contract_source_commit": compiler_contract["source_commit"],
            "contract_sha256": contract_sha256,
            "device": device_name,
            "evaluation": {"after": after, "before": before},
            "final_compiler_sha256": final_compiler_sha256,
            "final_reactor_sha256": final_reactor_sha256,
            "initial_compiler_sha256": initial_compiler_sha256,
            "initial_reactor_sha256": initial_reactor_sha256,
            "joint_model_optimizer_step": joint_payload["optimizer_step"],
            "joint_training_source_commit": joint_contract["source_commit"],
            "last_loss": last_loss,
            "last_position": last_position,
            "protected_checkpoint_sha256": provenance.checkpoint_sha256,
            "reader_parameters": reader_parameters,
            "replacement_system_parameters": replacement_system_parameters,
            "runtime_precision": str(next(model.parameters()).dtype),
            "schema": REPORT_SCHEMA,
            "source_verification": source_verification,
            "status": "pass",
            "trainable_parameters": trainable_parameters,
            "updates_completed": args.updates,
        }
        _write_no_replace(args.output / "report.json", _canonical_bytes(report))
        files = (
            "compiler-final.safetensors",
            "compiler-initial.safetensors",
            "pilot-contract.json",
            "reactor-final.safetensors",
            "reactor-initial.safetensors",
            "report.json",
            "train.jsonl",
        )
        _write_no_replace(
            args.output / "SHA256SUMS",
            "".join(
                f"{_sha256_file(args.output / name)}  {name}\n"
                for name in files
            ).encode("ascii"),
        )
        for path in args.output.iterdir():
            path.chmod(0o400)
        args.output.chmod(0o500)
    except BaseException:
        if args.output.exists():
            shutil.rmtree(args.output)
        raise
    finally:
        packet_index.close()
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
