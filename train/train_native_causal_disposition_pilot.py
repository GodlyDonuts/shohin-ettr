#!/usr/bin/env python3
"""Fit the native ETTR disposition motor on oracle terminal states.

This is an interface diagnostic, not an autonomous reasoning claim. The
backbone, compiler, and reactor remain frozen. The experiment can isolate a
dedicated linear or nonlinear truth motor, or jointly adapt the warm-started
query/state reader. A later source-deleted composition is admissible only if
both held-out causal factors improve here.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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

from ettr_objectives import ETTRObjectiveConfig, _causal_query_binding_loss
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_token_transcode import (
    TokenNativeETTRTranscoder,
    receipt_value as transcode_receipt_value,
)
from ettr_v3_streaming import ETTRV3StreamingRelease, move_continuation_batch
from eval_ettr_v3 import _parameter_sha256, _read_hash_bound_json
from native_causal_disposition_reader import (
    NativeCausalDispositionReader,
    answer_token_ids_from_tokenizer,
)
from train_ettr_component_island import (
    _canonical_bytes,
    _pair_rows,
    _reader_pairs_from_logits,
    _sha256_file,
    _summary,
    _write_no_replace,
    load_component_warm_start,
)
from train_ettr_joint_component_island import _validate_parent_lineage
from train_ettr_joint_instruction_canary import _load_parent
from probe_ettr_oracle_interfaces import packet_targets_to_state
from probe_ettr_causal_queries import _depth_bucket


CONTRACT_SCHEMA = "shohin-ettr-native-disposition-pilot-contract-v1"
REPORT_SCHEMA = "shohin-ettr-native-disposition-pilot-report-v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class NativeDispositionPilotError(RuntimeError):
    """The native answer-motor pilot violated its sealed contract."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--target-tokenizer", type=Path, required=True)
    parser.add_argument("--parent-joint-model", type=Path, required=True)
    parser.add_argument("--parent-joint-model-sha256", required=True)
    parser.add_argument("--parent-run-contract", type=Path, required=True)
    parser.add_argument("--parent-run-contract-sha256", required=True)
    parser.add_argument("--initial-reader", type=Path, required=True)
    parser.add_argument("--initial-reader-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=2_000)
    parser.add_argument("--start-position", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--truth-motor-hidden", type=int, default=0)
    parser.add_argument("--train-reader", action="store_true")
    parser.add_argument("--reader-slot-addresses", action="store_true")
    parser.add_argument("--reader-initial-state", action="store_true")
    parser.add_argument("--reader-direct-output", action="store_true")
    parser.add_argument("--reader-state-bottleneck", action="store_true")
    parser.add_argument("--state-only-motor", action="store_true")
    parser.add_argument(
        "--reader-parameter-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument(
        "--motor-query-geometry",
        choices=("stage", "late"),
        default="stage",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    paths = (
        args.release_root,
        args.data_root,
        args.tokenizer,
        args.target_tokenizer,
        args.parent_joint_model,
        args.parent_run_contract,
        args.initial_reader,
        args.output,
    )
    hashes = (
        args.release_sha256,
        args.parent_joint_model_sha256,
        args.parent_run_contract_sha256,
        args.initial_reader_sha256,
    )
    if (
        any(not path.is_absolute() for path in paths)
        or any(_HEX64.fullmatch(value) is None for value in hashes)
        or _HEX40.fullmatch(args.source_commit) is None
        or not 0 <= args.data_seed < 2**63
        or not 0 <= args.model_seed < 2**63
        or args.updates < 1
        or args.start_position < 0
        or args.eval_batches < 2
        or args.log_every < 1
        or not 1 <= args.gradient_accumulation <= 64
        or not 0 <= args.truth_motor_hidden <= 16_384
        or not math.isfinite(args.learning_rate)
        or not 0.0 < args.learning_rate < 1.0
        or not math.isfinite(args.gradient_clip)
        or args.gradient_clip <= 0.0
        or (args.reader_slot_addresses and not args.train_reader)
        or (
            args.reader_initial_state
            and (not args.train_reader or not args.reader_slot_addresses)
        )
        or args.output.exists()
        or args.output.is_symlink()
        or not args.output.parent.is_dir()
    ):
        raise NativeDispositionPilotError("native disposition pilot arguments differ")


def _gather_read_logits(
    logits: torch.Tensor,
    read_index: torch.Tensor,
) -> torch.Tensor:
    return logits.gather(
        1,
        read_index[:, None, None].expand(-1, 1, logits.shape[-1]),
    ).squeeze(1)


def _annotate_pair_rows(pair, depths: torch.Tensor) -> list[dict[str, object]]:
    pair_rows = _pair_rows(pair)
    if depths.ndim != 1 or depths.numel() != len(pair_rows):
        raise NativeDispositionPilotError("native disposition depth support differs")
    return [
        row
        | {
            "depth": int(depth.detach().cpu()),
            "depth_bucket": _depth_bucket(int(depth.detach().cpu())),
        }
        for row, depth in zip(pair_rows, depths, strict=True)
    ]


def _forward_logits(
    model,
    reader: NativeCausalDispositionReader,
    batch,
    *,
    motor_query_geometry: str,
) -> torch.Tensor:
    with torch.no_grad():
        stage_hidden = model._encode_to_stage(
            batch.episodes.query.tokens,
            pos=0,
        )
        if motor_query_geometry == "stage":
            motor_hidden = stage_hidden
        elif motor_query_geometry == "late":
            motor_hidden = model.base.norm(
                model._decode_from_stage(stage_hidden, pos=0)
            )
        else:
            raise NativeDispositionPilotError("motor query geometry differs")
        state = packet_targets_to_state(
            batch.terminal_packet_targets,
            model.config,
            step=batch.transaction_targets.opcode.shape[1],
            dtype=stage_hidden.dtype,
        )
        initial_state = None
        if reader.config.reader_initial_state:
            initial_state = packet_targets_to_state(
                batch.packet_targets,
                model.config,
                step=0,
                dtype=stage_hidden.dtype,
            )
    return reader(
        stage_hidden,
        state,
        initial_state=initial_state,
        motor_hidden=motor_hidden,
        attention_mask=batch.episodes.query.attention_mask,
    )


def _loss(
    model,
    reader: NativeCausalDispositionReader,
    batch,
    *,
    motor_query_geometry: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = _forward_logits(
        model,
        reader,
        batch,
        motor_query_geometry=motor_query_geometry,
    )
    read_logits = _gather_read_logits(
        logits,
        batch.episodes.query_read_index,
    )
    targets = batch.episodes.query.targets.gather(
        1,
        batch.episodes.query_read_index[:, None],
    ).squeeze(1)
    factual = F.cross_entropy(read_logits.float(), targets)
    pairs = _reader_pairs_from_logits(read_logits, batch)
    world = _causal_query_binding_loss(
        pairs["world"],
        margin=1.0,
        reduction="balanced-logmeanexp",
        classification_weight=0.25,
        effect_weight=1.0,
        invariance_weight=0.25,
        risk_temperature=0.25,
    )[0]
    command = _causal_query_binding_loss(
        pairs["command"],
        margin=1.0,
        reduction="balanced-logmeanexp",
        classification_weight=0.25,
        effect_weight=1.0,
        invariance_weight=0.25,
        risk_temperature=0.25,
    )[0]
    parts = {
        "command_binding": command,
        "factual": factual,
        "world_binding": world,
    }
    loss = 0.25 * factual + 0.5 * (world + command)
    return loss, {name: float(value.detach().cpu()) for name, value in parts.items()}


def _evaluate(
    model,
    reader: NativeCausalDispositionReader,
    *,
    stream: ETTRV3StreamingRelease,
    packet_index: ETTRDiskPacketSufficiencyIndex,
    transcoder: TokenNativeETTRTranscoder,
    device: torch.device,
    data_seed: int,
    max_batches: int,
    motor_query_geometry: str,
) -> dict[str, object]:
    reader.eval()
    rows: dict[str, list[dict[str, object]]] = {
        "world": [],
        "command": [],
    }
    factual_correct = 0
    factual_count = 0
    iterator = stream.iter_positioned_batches(
        "development",
        rank=0,
        world_size=1,
        epoch=0,
        seed=data_seed,
    )
    for observed, (_, cpu_batch) in enumerate(iterator):
        if observed >= max_batches:
            break
        packet_index.verify_validation((cpu_batch,))
        batch = move_continuation_batch(
            transcoder.transcode_batch(cpu_batch),
            device,
        )
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ),
        ):
            logits = _forward_logits(
                model,
                reader,
                batch,
                motor_query_geometry=motor_query_geometry,
            )
            read_logits = _gather_read_logits(
                logits,
                batch.episodes.query_read_index,
            )
            targets = batch.episodes.query.targets.gather(
                1,
                batch.episodes.query_read_index[:, None],
            ).squeeze(1)
            factual_correct += int(read_logits.argmax(-1).eq(targets).sum())
            factual_count += targets.numel()
            pairs = _reader_pairs_from_logits(read_logits, batch)
            (
                _world_packet,
                _world_command,
                world_target,
                _command_packet,
                _command_command,
                command_target,
            ) = batch.causal_rectangles.intervention_indices()
            depths = batch.transaction_targets.step_mask.sum(-1)
            for factor, pair in pairs.items():
                target_index = world_target if factor == "world" else command_target
                rows[factor].extend(
                    _annotate_pair_rows(
                        pair,
                        depths.index_select(0, target_index),
                    )
                )
    if factual_count != max_batches * 16:
        raise NativeDispositionPilotError(
            "native disposition evaluation support differs"
        )
    return {
        "batches": max_batches,
        "factual_top1": factual_correct / factual_count,
        "oracle_terminal_reader": {
            factor: _summary(value) for factor, value in rows.items()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise NativeDispositionPilotError("pilot requires one process")
    if not torch.cuda.is_available():
        raise NativeDispositionPilotError("pilot requires CUDA")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    if "H100" not in torch.cuda.get_device_name(device).upper():
        raise NativeDispositionPilotError("pilot requires an H100")

    parent_contract = _read_hash_bound_json(
        args.parent_run_contract,
        expected_sha256=args.parent_run_contract_sha256,
        label="parent run contract",
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
    warm_sha256 = load_component_warm_start(
        model,
        "reader",
        args.initial_reader,
        expected_sha256=args.initial_reader_sha256,
    )
    stream = ETTRV3StreamingRelease(
        args.release_root,
        expected_release_sha256=args.release_sha256,
        data_root=args.data_root,
        tokenizer_path=args.tokenizer,
    )
    source_verification = stream.verify_source_shards()
    transcoder = TokenNativeETTRTranscoder(
        args.tokenizer,
        args.target_tokenizer,
    )
    if model.base.cfg.vocab_size != transcoder.target_vocab_size:
        raise NativeDispositionPilotError("target vocabulary differs")
    answer_token_ids = answer_token_ids_from_tokenizer(args.target_tokenizer)
    model.requires_grad_(False)
    model.to(device=device, dtype=torch.bfloat16).eval()
    torch.manual_seed(args.model_seed)
    torch.cuda.manual_seed_all(args.model_seed)
    reader_config = replace(
        model.config,
        reader_slot_addresses=args.reader_slot_addresses,
        reader_initial_state=args.reader_initial_state,
        reader_direct_output=args.reader_direct_output,
        reader_state_bottleneck=args.reader_state_bottleneck,
    )
    reader = NativeCausalDispositionReader(
        reader_config,
        vocab_size=model.base.cfg.vocab_size,
        answer_token_ids=answer_token_ids,
        truth_motor_hidden=args.truth_motor_hidden,
        state_only_motor=args.state_only_motor,
    )
    reader.load_reader_state(model.query_reader)
    reader.reader.requires_grad_(args.train_reader)
    reader_parameter_dtype = (
        torch.float32 if args.reader_parameter_dtype == "float32" else torch.bfloat16
    )
    reader.to(device=device, dtype=reader_parameter_dtype)
    trainable = tuple(
        parameter for parameter in reader.parameters() if parameter.requires_grad
    )
    trainable_parameter_count = sum(parameter.numel() for parameter in trainable)
    if trainable_parameter_count < 2_306:
        raise NativeDispositionPilotError(
            "truth-motor trainable parameter count differs"
        )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    packet_index = ETTRDiskPacketSufficiencyIndex(stream.packet_index_root)
    try:
        args.output.mkdir(mode=0o700)
        initial_path = args.output / "native-reader-initial.safetensors"
        save_file(
            {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in reader.state_dict().items()
            },
            initial_path,
        )
        os.chmod(initial_path, 0o400)
        initial_sha256 = _sha256_file(initial_path)
        before = _evaluate(
            model,
            reader,
            stream=stream,
            packet_index=packet_index,
            transcoder=transcoder,
            device=device,
            data_seed=args.data_seed,
            max_batches=args.eval_batches,
            motor_query_geometry=args.motor_query_geometry,
        )
        contract: dict[str, object] = {
            "answer_token_ids": list(answer_token_ids),
            "causal_loss": {
                "classification_weight": 0.25,
                "effect_weight": 1.0,
                "invariance_weight": 0.25,
                "margin": 1.0,
                "risk_temperature": 0.25,
            },
            "data_seed": args.data_seed,
            "eval_batches": args.eval_batches,
            "gradient_clip": args.gradient_clip,
            "gradient_accumulation": args.gradient_accumulation,
            "initial_reader_sha256": warm_sha256,
            "learning_rate": args.learning_rate,
            "model_seed": args.model_seed,
            "motor_query_geometry": args.motor_query_geometry,
            "oracle_at_autonomous_inference": False,
            "oracle_training_boundary": (
                "exact_initial_and_terminal_packet"
                if args.reader_initial_state
                else "exact_terminal_packet"
            ),
            "parent_joint_model_sha256": args.parent_joint_model_sha256,
            "parent_run_contract_sha256": args.parent_run_contract_sha256,
            "release_file_sha256": args.release_sha256,
            "schema": CONTRACT_SCHEMA,
            "source_commit": args.source_commit,
            "start_position": args.start_position,
            "token_transcode": transcode_receipt_value(transcoder.receipt),
            "train_reader": args.train_reader,
            "reader_slot_addresses": args.reader_slot_addresses,
            "reader_initial_state": args.reader_initial_state,
            "reader_direct_output": args.reader_direct_output,
            "reader_parameter_dtype": args.reader_parameter_dtype,
            "reader_state_bottleneck": args.reader_state_bottleneck,
            "state_only_motor": args.state_only_motor,
            "trainable_parameters": trainable_parameter_count,
            "truth_motor_hidden": args.truth_motor_hidden,
            "updates": args.updates,
        }
        contract_sha256 = _write_no_replace(
            args.output / "pilot-contract.json",
            _canonical_bytes(contract),
        )
        _write_no_replace(args.output / "train.jsonl", b"", mode=0o600)

        reader.train()
        iterator = stream.iter_positioned_batches(
            "train",
            rank=0,
            world_size=1,
            epoch=0,
            seed=args.data_seed,
            start_position=args.start_position,
        )
        last_loss = None
        last_position = args.start_position
        for update in range(1, args.updates + 1):
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            accumulated_parts = {
                "command_binding": 0.0,
                "factual": 0.0,
                "world_binding": 0.0,
            }
            for _microstep in range(args.gradient_accumulation):
                try:
                    last_position, cpu_batch = next(iterator)
                except StopIteration as exc:
                    raise NativeDispositionPilotError(
                        "pilot exhausted the admitted train stream"
                    ) from exc
                packet_index.verify_train((cpu_batch,))
                batch = move_continuation_batch(
                    transcoder.transcode_batch(cpu_batch),
                    device,
                )
                batch.validate(
                    model.config,
                    ETTRObjectiveConfig(vocab_size=model.base.cfg.vocab_size),
                )
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    loss, parts = _loss(
                        model,
                        reader,
                        batch,
                        motor_query_geometry=args.motor_query_geometry,
                    )
                if not bool(torch.isfinite(loss)):
                    raise NativeDispositionPilotError("pilot loss is nonfinite")
                (loss / args.gradient_accumulation).backward()
                accumulated_loss += float(loss.detach().cpu())
                for name, value in parts.items():
                    accumulated_parts[name] += value
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                args.gradient_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
            last_loss = accumulated_loss / args.gradient_accumulation
            parts = {
                name: value / args.gradient_accumulation
                for name, value in accumulated_parts.items()
            }
            if update % args.log_every == 0 or update == args.updates:
                with (args.output / "train.jsonl").open("ab", buffering=0) as log:
                    log.write(
                        _canonical_bytes(
                            {
                                "gradient_norm_pre_clip": float(
                                    gradient_norm.detach().float().cpu()
                                ),
                                "loss": last_loss,
                                "loss_parts": parts,
                                "position": last_position,
                                "schema": ("shohin-ettr-native-disposition-metric-v1"),
                                "update": update,
                            }
                        )
                    )
        os.chmod(args.output / "train.jsonl", 0o400)
        after = _evaluate(
            model,
            reader,
            stream=stream,
            packet_index=packet_index,
            transcoder=transcoder,
            device=device,
            data_seed=args.data_seed,
            max_batches=args.eval_batches,
            motor_query_geometry=args.motor_query_geometry,
        )
        final_path = args.output / "native-reader-final.safetensors"
        save_file(
            {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in reader.state_dict().items()
            },
            final_path,
        )
        os.chmod(final_path, 0o400)
        final_sha256 = _sha256_file(final_path)
        report = {
            "answer_token_ids": list(answer_token_ids),
            "contract_sha256": contract_sha256,
            "device": torch.cuda.get_device_name(device),
            "evaluation": {"after": after, "before": before},
            "final_native_reader_sha256": final_sha256,
            "initial_native_reader_sha256": initial_sha256,
            "last_loss": last_loss,
            "model_seed": args.model_seed,
            "motor_query_geometry": args.motor_query_geometry,
            "native_reader_parameter_sha256": _parameter_sha256(reader),
            "parent_joint_model_sha256": args.parent_joint_model_sha256,
            "release_file_sha256": args.release_sha256,
            "schema": REPORT_SCHEMA,
            "source_commit": args.source_commit,
            "source_verification": source_verification,
            "gradient_accumulation": args.gradient_accumulation,
            "train_reader": args.train_reader,
            "reader_slot_addresses": args.reader_slot_addresses,
            "reader_initial_state": args.reader_initial_state,
            "reader_direct_output": args.reader_direct_output,
            "reader_parameter_dtype": args.reader_parameter_dtype,
            "reader_state_bottleneck": args.reader_state_bottleneck,
            "state_only_motor": args.state_only_motor,
            "trainable_parameters": trainable_parameter_count,
            "truth_motor_hidden": args.truth_motor_hidden,
            "updates": args.updates,
        }
        _write_no_replace(
            args.output / "report.json",
            _canonical_bytes(report),
        )
        files = (
            "native-reader-final.safetensors",
            "native-reader-initial.safetensors",
            "pilot-contract.json",
            "report.json",
            "train.jsonl",
        )
        _write_no_replace(
            args.output / "SHA256SUMS",
            "".join(
                f"{_sha256_file(args.output / name)}  {name}\n" for name in files
            ).encode("ascii"),
        )
    except BaseException:
        shutil.rmtree(args.output, ignore_errors=True)
        raise
    finally:
        packet_index.close()
    print(
        json.dumps(
            {
                "final_native_reader_sha256": final_sha256,
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
