#!/usr/bin/env python3
"""Fit one ETTR component against a hash-bound co-adapted joint model.

The parent base transformer and the two unselected ETTR components remain
frozen. The output is a component artifact for an explicit later composition
step; this trainer never silently rewrites the parent joint checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
from typing import Mapping, Sequence

from safetensors.torch import save_file
import torch

from ettr_objectives import ETTRObjectiveConfig
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_v3_streaming import ETTRV3StreamingRelease, move_continuation_batch
from eval_ettr_v3 import _parameter_sha256, _read_hash_bound_json
from train_ettr_component_island import (
    _REACTOR_REDUCTIONS,
    _canonical_bytes,
    _component_state,
    _evaluate_interfaces,
    _sha256_file,
    _write_no_replace,
    component_loss,
    select_trainable_component,
)
from train_ettr_joint_instruction_canary import (
    MODEL_SCHEMA,
    RUN_SCHEMA,
    _load_parent,
)


REPORT_SCHEMA = "shohin-ettr-joint-component-island-report-v1"
CONTRACT_SCHEMA = "shohin-ettr-joint-component-island-contract-v1"
_COMPONENTS = ("compiler", "reactor", "reader")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ETTRJointComponentIslandError(RuntimeError):
    """A joint component-island custody or optimization contract failed."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=_COMPONENTS, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--parent-joint-model", type=Path, required=True)
    parser.add_argument("--parent-joint-model-sha256", required=True)
    parser.add_argument("--parent-run-contract", type=Path, required=True)
    parser.add_argument("--parent-run-contract-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=5_000)
    parser.add_argument("--start-position", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--reader-injection",
        choices=("stage", "late", "postnorm", "postnorm-scaled"),
        default="stage",
    )
    parser.add_argument(
        "--reactor-reduction",
        choices=_REACTOR_REDUCTIONS,
        default="decision-mean",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    paths = (
        args.release_root,
        args.data_root,
        args.tokenizer,
        args.parent_joint_model,
        args.parent_run_contract,
        args.output,
    )
    hashes = (
        args.release_sha256,
        args.parent_joint_model_sha256,
        args.parent_run_contract_sha256,
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
        or not math.isfinite(args.weight_decay)
        or args.weight_decay < 0.0
        or not math.isfinite(args.gradient_clip)
        or args.gradient_clip <= 0.0
        or (args.component != "reader" and args.reader_injection != "stage")
        or (
            args.component != "reactor"
            and getattr(
                args,
                "reactor_reduction",
                "decision-mean",
            )
            != "decision-mean"
        )
        or args.output.exists()
        or args.output.is_symlink()
        or not args.output.parent.is_dir()
    ):
        raise ETTRJointComponentIslandError(
            "joint component-island arguments differ"
        )


def _validate_parent_lineage(
    parent_contract: Mapping[str, object],
    parent_payload: Mapping[str, object],
    *,
    release_sha256: str,
    parent_run_contract_sha256: str,
) -> None:
    if (
        parent_contract.get("schema") != RUN_SCHEMA
        or parent_payload.get("schema") != MODEL_SCHEMA
        or parent_payload.get("run_contract_sha256")
        != parent_run_contract_sha256
        or parent_contract.get("ettr_release_sha256")
        != release_sha256
        or parent_contract.get("model_config")
        != parent_payload.get("ettr_config")
    ):
        raise ETTRJointComponentIslandError(
            "joint component parent lineage differs"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ETTRJointComponentIslandError(
            "joint component island requires one process"
        )
    if not torch.cuda.is_available():
        raise ETTRJointComponentIslandError(
            "joint component island requires CUDA"
        )
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    if "H100" not in torch.cuda.get_device_name(device).upper():
        raise ETTRJointComponentIslandError(
            "joint component island requires an H100"
        )

    parent_contract = _read_hash_bound_json(
        args.parent_run_contract,
        expected_sha256=args.parent_run_contract_sha256,
        label="parent tri-stream run contract",
    )
    model, parent_payload = _load_parent(
        args.parent_joint_model,
        expected_sha256=args.parent_joint_model_sha256,
    )
    _validate_parent_lineage(
        parent_contract,
        parent_payload,
        release_sha256=args.release_sha256,
        parent_run_contract_sha256=args.parent_run_contract_sha256,
    )
    stream = ETTRV3StreamingRelease(
        args.release_root,
        expected_release_sha256=args.release_sha256,
        data_root=args.data_root,
        tokenizer_path=args.tokenizer,
    )
    source_verification = stream.verify_source_shards()
    model.to(device=device, dtype=torch.bfloat16)
    ownership = select_trainable_component(model, args.component)
    trainable = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    packet_index = ETTRDiskPacketSufficiencyIndex(
        stream.packet_index_root
    )
    try:
        args.output.mkdir(mode=0o700)
        parent_parameter_sha256 = _parameter_sha256(model)
        frozen_parameter_sha256 = {
            "base": _parameter_sha256(model.base),
            "compiler": _parameter_sha256(model.compiler),
            "reactor": _parameter_sha256(model.reactor),
            "reader": _parameter_sha256(model.query_reader),
        }
        initial_component = _component_state(model, args.component)
        initial_path = args.output / "component-initial.safetensors"
        save_file(initial_component, initial_path)
        os.chmod(initial_path, 0o400)
        initial_component_sha256 = _sha256_file(initial_path)
        before = _evaluate_interfaces(
            model,
            stream=stream,
            packet_index=packet_index,
            device=device,
            data_seed=args.data_seed,
            max_batches=args.eval_batches,
            reader_injection=args.reader_injection,
        )
        contract = {
            "component": args.component,
            "data_seed": args.data_seed,
            "eval_batches": args.eval_batches,
            "gradient_clip": args.gradient_clip,
            "learning_rate": args.learning_rate,
            "oracle_at_autonomous_inference": False,
            "oracle_training_boundary": {
                "compiler": "initial_packet_targets",
                "reactor": (
                    "initial_packet_and_prior_transactions_teacher_forced"
                ),
                "reader": "exact_terminal_packet",
            }[args.component],
            "parent_joint_model": str(args.parent_joint_model),
            "parent_joint_model_sha256": (
                args.parent_joint_model_sha256
            ),
            "parent_run_contract": str(args.parent_run_contract),
            "parent_run_contract_sha256": (
                args.parent_run_contract_sha256
            ),
            "release_file_sha256": args.release_sha256,
            "reader_injection": args.reader_injection,
            "reactor_reduction": args.reactor_reduction,
            "schema": CONTRACT_SCHEMA,
            "source_commit": args.source_commit,
            "start_position": args.start_position,
            "updates": args.updates,
            "weight_decay": args.weight_decay,
        }
        contract_sha256 = _write_no_replace(
            args.output / "island-contract.json",
            _canonical_bytes(contract),
        )
        _write_no_replace(
            args.output / "train.jsonl",
            b"",
            mode=0o600,
        )

        model.train()
        epoch = 0
        iterator = stream.iter_positioned_batches(
            "train",
            rank=0,
            world_size=1,
            epoch=epoch,
            seed=args.data_seed,
            start_position=args.start_position,
        )
        observed_rows = 0
        observed_token_positions = 0
        last_loss = None
        last_position = args.start_position
        for update in range(1, args.updates + 1):
            try:
                last_position, cpu_batch = next(iterator)
            except StopIteration:
                epoch += 1
                iterator = stream.iter_positioned_batches(
                    "train",
                    rank=0,
                    world_size=1,
                    epoch=epoch,
                    seed=args.data_seed,
                )
                last_position, cpu_batch = next(iterator)
            packet_index.verify_train((cpu_batch,))
            batch = move_continuation_batch(cpu_batch, device)
            batch.validate(
                model.config,
                ETTRObjectiveConfig(
                    vocab_size=model.base.cfg.vocab_size
                ),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                loss, parts = component_loss(
                    model,
                    batch,
                    args.component,
                    reader_injection=args.reader_injection,
                    reactor_reduction=args.reactor_reduction,
                )
            if not bool(torch.isfinite(loss)):
                raise ETTRJointComponentIslandError(
                    "joint component loss is non-finite"
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
                        _canonical_bytes(
                            {
                                "component": args.component,
                                "epoch": epoch,
                                "gradient_norm_pre_clip": float(
                                    gradient_norm.detach().float().cpu()
                                ),
                                "loss": last_loss,
                                "loss_parts": parts,
                                "position": last_position,
                                "schema": (
                                    "shohin-ettr-joint-component-metric-v1"
                                ),
                                "update": update,
                            }
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
            reader_injection=args.reader_injection,
        )
        final_component = _component_state(model, args.component)
        final_path = args.output / "component-final.safetensors"
        save_file(final_component, final_path)
        os.chmod(final_path, 0o400)
        final_component_sha256 = _sha256_file(final_path)
        final_parameter_sha256 = _parameter_sha256(model)
        final_module_sha256 = {
            "base": _parameter_sha256(model.base),
            "compiler": _parameter_sha256(model.compiler),
            "reactor": _parameter_sha256(model.reactor),
            "reader": _parameter_sha256(model.query_reader),
        }
        unchanged_modules = {
            name: (
                final_module_sha256[name]
                == frozen_parameter_sha256[name]
            )
            for name in frozen_parameter_sha256
            if name != args.component
        }
        if not all(unchanged_modules.values()):
            raise ETTRJointComponentIslandError(
                "a frozen parent module changed"
            )
        report = {
            "component": args.component,
            "contract_sha256": contract_sha256,
            "data_seed": args.data_seed,
            "device": {
                "bf16": torch.cuda.is_bf16_supported(),
                "name": torch.cuda.get_device_name(device),
            },
            "evaluation": {"after": after, "before": before},
            "final_component_sha256": final_component_sha256,
            "final_module_sha256": final_module_sha256,
            "final_parameter_sha256": final_parameter_sha256,
            "initial_component_sha256": initial_component_sha256,
            "last_loss": last_loss,
            "observed_rows": observed_rows,
            "observed_token_positions": observed_token_positions,
            "oracle_at_autonomous_inference": False,
            "ownership": ownership,
            "parent_joint_model_sha256": (
                args.parent_joint_model_sha256
            ),
            "parent_parameter_sha256": parent_parameter_sha256,
            "parent_run_contract_sha256": (
                args.parent_run_contract_sha256
            ),
            "release_file_sha256": args.release_sha256,
            "release_manifest_sha256": stream.manifest.sha256(),
            "reader_injection": args.reader_injection,
            "reactor_reduction": args.reactor_reduction,
            "schema": REPORT_SCHEMA,
            "source_commit": args.source_commit,
            "source_verification": source_verification,
            "start_position": args.start_position,
            "unchanged_frozen_modules": unchanged_modules,
            "updates": args.updates,
        }
        _write_no_replace(
            args.output / "report.json",
            _canonical_bytes(report),
        )
        files = (
            "component-final.safetensors",
            "component-initial.safetensors",
            "island-contract.json",
            "report.json",
            "train.jsonl",
        )
        manifest = "".join(
            f"{_sha256_file(args.output / name)}  {name}\n"
            for name in files
        ).encode("ascii")
        _write_no_replace(
            args.output / "SHA256SUMS",
            manifest,
        )
    finally:
        packet_index.close()

    print(
        json.dumps(
            {
                "component": args.component,
                "final_component_sha256": final_component_sha256,
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
