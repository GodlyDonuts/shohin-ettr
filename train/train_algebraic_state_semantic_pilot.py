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

from eval_algebraic_query_joint_state import (
    _evaluate,
    _load_compiler,
    _reader_forward,
    _strict_load_joint_model,
)
from ettr_objectives import ETTRObjectiveConfig
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_query_supervision import iter_batches_with_query_specs
from ettr_v3_streaming import ETTRV3StreamingRelease, move_continuation_batch
from eval_ettr_v3 import _parameter_sha256
from train_ettr_component_island import (
    _canonical_bytes,
    _component_state,
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


def _semantic_loss(model, reader, batch, specs):
    initial_state = model.compile_world(
        batch.episodes.world.tokens,
        attention_mask=batch.episodes.world.attention_mask,
        hard=True,
    )
    terminal_state, _trace = model.execute(
        initial_state,
        steps=batch.transaction_targets.opcode.shape[1],
        hard=True,
        command_idx=batch.episodes.command.tokens,
        command_attention_mask=batch.episodes.command.attention_mask,
    )
    output = _reader_forward(
        reader,
        batch,
        specs,
        initial_state,
        terminal_state,
        oracle_program=False,
    )
    return _truth_loss(output.vocab_logits, batch)


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
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )
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
            "reader_parameters": reader_parameters,
            "reactor_aux_weight": args.reactor_aux_weight,
            "reactor_reduction": args.reactor_reduction,
            "release_file_sha256": args.release_sha256,
            "replacement_system_parameters": replacement_system_parameters,
            "required_device_class": args.required_device_class,
            "runtime_precision": str(next(model.parameters()).dtype),
            "schema": CONTRACT_SCHEMA,
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
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if is_h100
                else nullcontext()
            )
            with autocast:
                semantic, semantic_parts = _semantic_loss(
                    model,
                    reader,
                    batch,
                    specs,
                )
                compiler_aux, compiler_parts = component_loss(
                    model,
                    batch,
                    "compiler",
                )
                reactor_aux, reactor_parts = component_loss(
                    model,
                    batch,
                    "reactor",
                    reactor_reduction=args.reactor_reduction,
                )
                loss = (
                    semantic
                    + args.compiler_aux_weight * compiler_aux
                    + args.reactor_aux_weight * reactor_aux
                )
            if not bool(torch.isfinite(loss)):
                raise AlgebraicStateSemanticError(
                    "algebraic state-semantic loss is nonfinite"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                args.gradient_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            if update % args.log_every == 0 or update == args.updates:
                metric = {
                    "compiler_aux": float(compiler_aux.detach().cpu()),
                    "compiler_parts": compiler_parts,
                    "gradient_norm_pre_clip": float(
                        gradient_norm.detach().float().cpu()
                    ),
                    "loss": last_loss,
                    "position": last_position,
                    "reactor_aux": float(reactor_aux.detach().cpu()),
                    "reactor_parts": reactor_parts,
                    "schema": "shohin-ettr-algebraic-state-semantic-metric-v1",
                    "semantic": float(semantic.detach().cpu()),
                    "semantic_parts": {
                        name: float(value.detach().cpu())
                        for name, value in semantic_parts.items()
                    },
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
