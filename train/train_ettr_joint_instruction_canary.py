#!/usr/bin/env python3
"""Three-stream ETTR post-training canary.

The complete ETTR model, its language base, and completion-masked instruction
behavior remain under one optimizer. The experiment is deliberately bounded
and non-resumable; production continuation still requires a cursor-complete
checkpoint contract.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Mapping, Sequence

import torch
from tokenizers import Tokenizer

from data import ShardLoader
from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TheoryReactorConfig,
)
from ettr_distributed import ETTRDistributedCursor
from ettr_instruction_stream import (
    WeightedPackedInstructionStream,
    to_device_batch,
)
from ettr_joint_stream import (
    ETTRTriPositionScheduler,
    ETTRTriScheduleConfig,
    GeneralLanguageStepConfig,
    GeneralLanguageUpdateStep,
)
from ettr_objectives import ETTRObjectiveConfig, ETTRObjectiveWeights
from ettr_optimization import ETTROptimizerBundle, ETTROptimizerConfig
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_train_step import ETTRTrainStep, ETTRTrainStepConfig
from ettr_v3_streaming import ETTRV3StreamingRelease
from model import GPT, GPTConfig
from sft import DEFAULT_Q_FIELDS, DEFAULT_R_FIELDS, build_packed
from train_ettr_joint_stream_canary import (
    _canonical_bytes,
    _dataclass_contract,
    _ettr_metric_payload,
    _legacy_general_resolution,
    _seed_update,
    _torch_save_no_replace,
    _write_no_replace,
)
from workspace_checkpoint import file_sha256


_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE = _ROOT / "pipeline"
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from ettr_il_v3_protocol import CHARGED_POSITIONS_PER_ROW  # noqa: E402


RUN_SCHEMA = "shohin-ettr-tri-stream-canary-v1"
MODEL_SCHEMA = "shohin-ettr-joint-model-canary-v1"
REPORT_SCHEMA = "shohin-ettr-tri-stream-report-v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ETTRTriCanaryError(RuntimeError):
    """The bounded native post-training contract differs."""


def _parse_weight(value: str) -> tuple[str, float]:
    name, separator, raw_weight = value.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError(
            "instruction weights must be NAME=WEIGHT"
        )
    try:
        weight = float(raw_weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "instruction weight is not numeric"
        ) from exc
    if not math.isfinite(weight) or weight <= 0:
        raise argparse.ArgumentTypeError(
            "instruction weight must be finite and positive"
        )
    return name, weight


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--ettr-data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--legacy-general-shard-dir",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--legacy-general-weight",
        type=float,
        action="append",
        required=True,
    )
    parser.add_argument("--parent-joint-model", type=Path, required=True)
    parser.add_argument("--parent-joint-model-sha256", required=True)
    parser.add_argument("--instruction-data", type=Path, required=True)
    parser.add_argument("--instruction-data-sha256", required=True)
    parser.add_argument(
        "--instruction-sample-weight",
        action="append",
        type=_parse_weight,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--general-batch-size", type=int, default=16)
    parser.add_argument("--instruction-batch-size", type=int, default=16)
    parser.add_argument("--general-position-weight", type=int, required=True)
    parser.add_argument(
        "--instruction-position-weight",
        type=int,
        required=True,
    )
    parser.add_argument("--ettr-position-weight", type=int, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--total-updates", type=int, required=True)
    parser.add_argument("--warmup-updates", type=int, default=200)
    parser.add_argument("--base-lr-muon", type=float, default=0.0015)
    parser.add_argument("--base-lr-adam", type=float, default=0.00035)
    parser.add_argument(
        "--architecture-lr-muon",
        type=float,
        default=0.003,
    )
    parser.add_argument(
        "--architecture-lr-adam",
        type=float,
        default=0.0006,
    )
    parser.add_argument("--nll-gradient-cap", type=float, default=4.0)
    parser.add_argument("--query-binding-weight", type=float, default=1.0)
    parser.add_argument(
        "--gradient-clip-mode",
        choices=("global", "owner"),
        default="owner",
    )
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    paths = (
        args.release_root,
        args.ettr_data_root,
        args.tokenizer,
        args.parent_joint_model,
        args.instruction_data,
        args.output,
        *args.legacy_general_shard_dir,
    )
    weights = dict(args.instruction_sample_weight)
    if (
        _HEX64.fullmatch(args.release_sha256) is None
        or _HEX64.fullmatch(args.parent_joint_model_sha256) is None
        or _HEX64.fullmatch(args.instruction_data_sha256) is None
        or _HEX40.fullmatch(args.source_commit) is None
        or any(not path.is_absolute() for path in paths)
        or len(weights) != len(args.instruction_sample_weight)
        or len(args.legacy_general_shard_dir)
        != len(args.legacy_general_weight)
        or any(
            not math.isfinite(weight) or weight <= 0
            for weight in args.legacy_general_weight
        )
        or args.updates < 2
        or args.total_updates < args.updates
        or not 0 <= args.warmup_updates < args.total_updates
        or args.general_batch_size < 1
        or args.instruction_batch_size < 1
        or min(
            args.general_position_weight,
            args.instruction_position_weight,
            args.ettr_position_weight,
        )
        < 1
        or not 0 <= args.data_seed < 2**63
        or args.log_every < 1
    ):
        raise ETTRTriCanaryError("tri-stream canary arguments differ")


def _regular_file_identity(path: Path) -> dict[str, int]:
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size <= 0
    ):
        raise ETTRTriCanaryError("instruction input file differs")
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mtime_ns": metadata.st_mtime_ns,
        "size": metadata.st_size,
    }


def _load_parent(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[EndogenousTypedTheoryReactorGPT, Mapping[str, object]]:
    if file_sha256(path) != expected_sha256:
        raise ETTRTriCanaryError("parent joint-model hash differs")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ETTRTriCanaryError("parent joint-model is unreadable") from exc
    required = {
        "base_config",
        "ettr_config",
        "initialization",
        "model",
        "optimizer_step",
        "run_contract_sha256",
        "schedule",
        "schema",
        "source_commit",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != required
        or payload.get("schema") != MODEL_SCHEMA
        or not isinstance(payload.get("model"), Mapping)
    ):
        raise ETTRTriCanaryError("parent joint-model contract differs")
    model = EndogenousTypedTheoryReactorGPT(
        GPT(GPTConfig(**payload["base_config"])),
        TheoryReactorConfig(**payload["ettr_config"]),
    )
    try:
        incompatibility = model.load_state_dict(payload["model"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ETTRTriCanaryError(
            "parent joint-model strict load differs"
        ) from exc
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ETTRTriCanaryError("parent joint-model strict load differs")
    return model, payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ETTRTriCanaryError("tri-stream canary requires one H100")
    if not torch.cuda.is_available():
        raise ETTRTriCanaryError("tri-stream canary requires CUDA")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    if "H100" not in torch.cuda.get_device_name(device).upper():
        raise ETTRTriCanaryError("tri-stream canary requires an H100")
    if args.output.exists() or args.output.is_symlink():
        raise ETTRTriCanaryError("refusing an existing tri-stream output")

    stream = ETTRV3StreamingRelease(
        args.release_root,
        expected_release_sha256=args.release_sha256,
        data_root=args.ettr_data_root,
        tokenizer_path=args.tokenizer,
    )
    source_verification = stream.verify_source_shards()
    general = _legacy_general_resolution(
        args.legacy_general_shard_dir,
        args.legacy_general_weight,
        tokenizer_sha256=stream.manifest.tokenizer_sha256,
    )
    instruction_identity = _regular_file_identity(args.instruction_data)
    if file_sha256(args.instruction_data) != args.instruction_data_sha256:
        raise ETTRTriCanaryError("instruction data hash differs")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise ETTRTriCanaryError("instruction tokenizer lacks EOS")
    model, parent_payload = _load_parent(
        args.parent_joint_model,
        expected_sha256=args.parent_joint_model_sha256,
    )
    if (
        model.base.cfg.vocab_size != tokenizer.get_vocab_size()
        or general["tokenizer_sha256"]
        != stream.manifest.tokenizer_sha256
    ):
        raise ETTRTriCanaryError("tri-stream tokenizer identities differ")
    packed_inputs, packed_targets, groups, packing = build_packed(
        [args.instruction_data],
        tokenizer,
        model.base.cfg.seq_len,
        DEFAULT_Q_FIELDS,
        DEFAULT_R_FIELDS,
        eos_id,
        group_field="training_group",
        prompt_override_field="completion_prompt",
        return_stats=True,
    )
    if (
        packing["skipped"]["blank_lines"]
        or packing["skipped"]["invalid_fields"]
        or not len(packed_inputs)
    ):
        raise ETTRTriCanaryError(
            "instruction corpus packing differs"
        )
    instruction_weights = dict(args.instruction_sample_weight)
    instruction = WeightedPackedInstructionStream(
        packed_inputs,
        packed_targets,
        groups,
        batch_size=args.instruction_batch_size,
        sample_weights=instruction_weights,
        seed=args.data_seed,
    )

    model.to(device=device, dtype=torch.bfloat16)
    optimizer = ETTROptimizerBundle(
        model,
        ETTROptimizerConfig(
            train_base=True,
            base_lr_muon=args.base_lr_muon,
            base_lr_adam=args.base_lr_adam,
            architecture_lr_muon=args.architecture_lr_muon,
            architecture_lr_adam=args.architecture_lr_adam,
            warmup_updates=args.warmup_updates,
            total_updates=args.total_updates,
        ),
    )
    general_step_config = GeneralLanguageStepConfig()
    language_step = GeneralLanguageUpdateStep(
        model,
        optimizer,
        step_config=general_step_config,
    )
    packet_index = ETTRDiskPacketSufficiencyIndex(
        stream.packet_index_root
    )
    objective_config = ETTRObjectiveConfig(
        vocab_size=model.base.cfg.vocab_size,
        nll_gradient_cap=args.nll_gradient_cap,
    )
    objective_weights = ETTRObjectiveWeights(
        world_query_binding=args.query_binding_weight,
        command_query_binding=args.query_binding_weight,
    )
    ettr_step_config = ETTRTrainStepConfig(
        gradient_clip_mode=args.gradient_clip_mode,
        hard_transactions=True,
    )
    ettr_step = ETTRTrainStep(
        model,
        optimizer,
        objective_config,
        manifest=stream.manifest,
        packet_sufficiency=packet_index,
        manifest_sha256=stream.manifest.sha256(),
        objective_weights=objective_weights,
        step_config=ettr_step_config,
    )
    schedule = ETTRTriPositionScheduler(
        ETTRTriScheduleConfig(
            args.general_position_weight,
            args.instruction_position_weight,
            args.ettr_position_weight,
        )
    )
    language_loader = ShardLoader(
        general["shard_dirs"],
        seq_len=model.base.cfg.seq_len,
        batch_size=args.general_batch_size,
        rank=0,
        world=1,
        seed=args.data_seed,
        domain_weights=general["domain_weights"],
    )
    general_positions = args.general_batch_size * model.base.cfg.seq_len
    ettr_positions = (
        int(stream.release["training_rows_per_batch"])
        * CHARGED_POSITIONS_PER_ROW
    )
    cursor = ETTRDistributedCursor(epoch=0, position=0)

    args.output.mkdir(mode=0o700, parents=True)
    run_contract = {
        "data_seed": args.data_seed,
        "ettr_manifest_sha256": stream.manifest.sha256(),
        "ettr_positions_per_update": ettr_positions,
        "ettr_release_sha256": args.release_sha256,
        "ettr_step_config": _dataclass_contract(ettr_step_config),
        "general_corpora": general["corpora"],
        "general_inventory_sha256": general["inventory_sha256"],
        "general_legacy_scientific_control": True,
        "general_positions_per_update": general_positions,
        "general_step_config": _dataclass_contract(
            general_step_config
        ),
        "instruction_data": str(args.instruction_data),
        "instruction_data_sha256": args.instruction_data_sha256,
        "instruction_file_identity": instruction_identity,
        "instruction_packing": packing,
        "instruction_sample_weights": dict(
            sorted(instruction_weights.items())
        ),
        "model_config": asdict(model.config),
        "objective_config": asdict(objective_config),
        "objective_weights": asdict(objective_weights),
        "optimizer_config": asdict(optimizer.config),
        "parameter_receipt": asdict(model.parameter_receipt()),
        "parent_joint_model": str(args.parent_joint_model),
        "parent_joint_model_sha256": args.parent_joint_model_sha256,
        "parent_optimizer_step": parent_payload["optimizer_step"],
        "parent_run_contract_sha256": parent_payload[
            "run_contract_sha256"
        ],
        "schedule_config": asdict(schedule.config),
        "schema": RUN_SCHEMA,
        "scientific_canary_non_resumable": True,
        "source_commit": args.source_commit,
        "source_verification": source_verification,
        "target_updates": args.updates,
        "tokenizer_sha256": stream.manifest.tokenizer_sha256,
    }
    run_contract_sha256 = _write_no_replace(
        args.output / "run-contract.json",
        _canonical_bytes(run_contract),
    )
    log_path = args.output / "train.jsonl"
    _write_no_replace(log_path, b"", mode=0o600)
    log_handle = log_path.open("ab", buffering=0)
    final_losses: dict[str, float | None] = {
        "general": None,
        "instruction": None,
        "ettr": None,
    }
    model.train()
    try:
        while optimizer.next_update < args.updates:
            instruction_x, instruction_y, instruction_positions = (
                instruction.peek()
            )
            selected = schedule.select(
                general_positions=general_positions,
                instruction_positions=instruction_positions,
                ettr_positions=ettr_positions,
            )
            update_seed = _seed_update(
                seed=args.data_seed,
                optimizer_step=optimizer.next_update,
                stream=selected,
            )
            if selected == "general":
                batch = language_loader.next_batch(device)
                receipt = language_step.update((batch,))
                charged = int(receipt.supervised_token_count)
                final_losses["general"] = float(
                    receipt.loss.detach().float().cpu()
                )
                metrics = {
                    "gradient_norm": float(
                        receipt.gradient_norm.detach().float().cpu()
                    ),
                    "language_loss": final_losses["general"],
                }
            elif selected == "instruction":
                batch = to_device_batch(
                    instruction_x,
                    instruction_y,
                    device,
                )
                receipt = language_step.update((batch,))
                charged = int(receipt.supervised_token_count)
                if charged != instruction_positions:
                    raise ETTRTriCanaryError(
                        "instruction charged positions differ"
                    )
                instruction.advance()
                final_losses["instruction"] = float(
                    receipt.loss.detach().float().cpu()
                )
                metrics = {
                    "gradient_norm": float(
                        receipt.gradient_norm.detach().float().cpu()
                    ),
                    "instruction_loss": final_losses["instruction"],
                }
            else:
                usable = cursor.validate(
                    core_batches=len(stream.records["train"]),
                    world_size=1,
                    accumulation=1,
                )
                if usable == 0:
                    raise ETTRTriCanaryError(
                        "ETTR stream has no usable training batch"
                    )
                iterator = stream.iter_positioned_batches(
                    "train",
                    rank=0,
                    world_size=1,
                    epoch=cursor.epoch,
                    seed=args.data_seed,
                    start_position=cursor.position,
                    device=device,
                )
                position, batch = next(iterator)
                if position != cursor.position:
                    raise ETTRTriCanaryError(
                        "ETTR stream position differs"
                    )
                receipt = ettr_step.update((batch,))
                cursor = cursor.advance(
                    core_batches=len(stream.records["train"]),
                    world_size=1,
                    accumulation=1,
                )
                charged = ettr_positions
                final_losses["ettr"] = float(
                    receipt.total_loss.detach().float().cpu()
                )
                metrics = _ettr_metric_payload(receipt)
            position_receipt = schedule.record(
                stream=selected,
                positions=charged,
            )
            if (
                receipt.optimizer_step % args.log_every == 0
                or receipt.optimizer_step == args.updates
            ):
                log_handle.write(
                    _canonical_bytes(
                        {
                            **metrics,
                            **asdict(position_receipt),
                            "learning_rate_scale": (
                                receipt.learning_rate_scale
                            ),
                            "optimizer_step": receipt.optimizer_step,
                            "schema": (
                                "shohin-ettr-tri-stream-metric-v1"
                            ),
                            "stream": selected,
                            "update_seed": update_seed,
                        }
                    )
                )
    finally:
        log_handle.close()
        packet_index.close()

    if (
        _regular_file_identity(args.instruction_data)
        != instruction_identity
        or file_sha256(args.instruction_data)
        != args.instruction_data_sha256
        or _legacy_general_resolution(
            args.legacy_general_shard_dir,
            args.legacy_general_weight,
            tokenizer_sha256=stream.manifest.tokenizer_sha256,
        )["inventory_sha256"]
        != general["inventory_sha256"]
    ):
        raise ETTRTriCanaryError("tri-stream input changed during training")

    model.eval()
    full_model_path = args.output / "joint-model-final.pt"
    full_model_sha256 = _torch_save_no_replace(
        full_model_path,
        {
            "base_config": asdict(model.base.cfg),
            "ettr_config": asdict(model.config),
            "initialization": {
                "initialization": "parent-joint-model",
                "parent_joint_model_sha256": (
                    args.parent_joint_model_sha256
                ),
            },
            "model": model.state_dict(),
            "optimizer_step": optimizer.next_update,
            "run_contract_sha256": run_contract_sha256,
            "schedule": schedule.state_dict(),
            "schema": MODEL_SCHEMA,
            "source_commit": args.source_commit,
        },
    )
    base_path = args.output / "base-eval-final.pt"
    base_sha256 = _torch_save_no_replace(
        base_path,
        {
            "cfg": asdict(model.base.cfg),
            "data_seed": args.data_seed,
            "data_stream_generation": 0,
            "data_stream_seed": args.data_seed,
            "initialization": "parent-joint-tri-stream",
            "model": model.base.state_dict(),
            "schema": "shohin-joint-canary-base-eval-v1",
            "step": optimizer.next_update,
        },
    )
    final = schedule.receipt
    report = {
        "base_eval_checkpoint": base_path.name,
        "base_eval_checkpoint_sha256": base_sha256,
        "final_losses": final_losses,
        "full_model_checkpoint": full_model_path.name,
        "full_model_checkpoint_sha256": full_model_sha256,
        "instruction_stream_receipt": asdict(instruction.receipt),
        "optimizer_step": optimizer.next_update,
        "parent_joint_model_sha256": args.parent_joint_model_sha256,
        "position_fractions": {
            "general": final.general_positions / final.total_positions,
            "instruction": (
                final.instruction_positions / final.total_positions
            ),
            "ettr": final.ettr_positions / final.total_positions,
        },
        "run_contract_sha256": run_contract_sha256,
        "schedule_receipt": asdict(final),
        "schema": REPORT_SCHEMA,
    }
    _write_no_replace(
        args.output / "final-report.json",
        _canonical_bytes(report),
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
