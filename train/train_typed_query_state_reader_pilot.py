#!/usr/bin/env python3
"""Fit and gate the typed query compiler/state executor on oracle states.

This is an interface diagnostic.  It does not claim autonomous ETTR
composition because packet states are supplied from offline targets.  The
claim-bearing number is nevertheless source-deleted and query-autonomous:
the model receives original query tokens and predicts its own program before
answering.  A separately reported oracle-program arm isolates executor
capacity and is never promotable as reasoning.
"""

from __future__ import annotations

import argparse
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

from endogenous_typed_theory_reactor import SYSTEM_PARAMETER_CAP
from ettr_objectives import _causal_query_binding_loss
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_query_supervision import (
    ETTRQuerySpecBatch,
    QUERY_OPERATIONS,
    iter_batches_with_query_specs,
)
from ettr_token_transcode import (
    TokenNativeETTRTranscoder,
    receipt_value as transcode_receipt_value,
)
from ettr_v3_streaming import ETTRV3StreamingRelease, move_continuation_batch
from eval_ettr_v3 import _read_hash_bound_json
from native_causal_disposition_reader import answer_token_ids_from_tokenizer
from probe_ettr_causal_queries import _depth_bucket
from probe_ettr_oracle_interfaces import packet_targets_to_state
from train_ettr_component_island import (
    _canonical_bytes,
    _pair_rows,
    _reader_pairs_from_logits,
    _sha256_file,
    _summary,
    _write_no_replace,
)
from train_ettr_joint_component_island import _validate_parent_lineage
from train_ettr_joint_instruction_canary import _load_parent
from typed_query_state_reader import TypedQueryReaderOutput, TypedQueryStateReader


CONTRACT_SCHEMA = "shohin-ettr-typed-query-state-reader-contract-v1"
REPORT_SCHEMA = "shohin-ettr-typed-query-state-reader-report-v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class TypedQueryPilotError(RuntimeError):
    """The typed query/state pilot violated its sealed contract."""


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=4_000)
    parser.add_argument("--start-position", type=int, default=12_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--query-layers", type=int, default=3)
    parser.add_argument("--state-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--teacher-decay-updates", type=int, default=2_000)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    paths = (
        args.release_root,
        args.data_root,
        args.tokenizer,
        args.target_tokenizer,
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
        or not 0 <= args.model_seed < 2**63
        or args.updates < 1
        or args.start_position < 0
        or args.eval_batches < 2
        or args.log_every < 1
        or not 64 <= args.width <= 1_024
        or args.width % args.num_heads
        or not 1 <= args.query_layers <= 12
        or not 1 <= args.state_layers <= 12
        or not 1 <= args.teacher_decay_updates <= args.updates
        or not math.isfinite(args.learning_rate)
        or not 0.0 < args.learning_rate < 1.0
        or not math.isfinite(args.gradient_clip)
        or args.gradient_clip <= 0.0
        or args.output.exists()
        or args.output.is_symlink()
        or not args.output.parent.is_dir()
    ):
        raise TypedQueryPilotError("typed query pilot arguments differ")


def _annotate_pair_rows(pair, depths: torch.Tensor) -> list[dict[str, object]]:
    rows = _pair_rows(pair)
    if depths.ndim != 1 or depths.numel() != len(rows):
        raise TypedQueryPilotError("typed query depth support differs")
    return [
        row
        | {
            "depth": int(depth.detach().cpu()),
            "depth_bucket": _depth_bucket(int(depth.detach().cpu())),
        }
        for row, depth in zip(rows, depths, strict=True)
    ]


def _states(batch, config, *, dtype: torch.dtype):
    initial = packet_targets_to_state(
        batch.packet_targets,
        config,
        step=0,
        dtype=dtype,
    )
    terminal = packet_targets_to_state(
        batch.terminal_packet_targets,
        config,
        step=batch.transaction_targets.opcode.shape[1],
        dtype=dtype,
    )
    return initial, terminal


def _forward(
    reader: TypedQueryStateReader,
    source_batch,
    specs: ETTRQuerySpecBatch,
    *,
    teacher: bool,
) -> TypedQueryReaderOutput:
    initial, terminal = _states(source_batch, reader.config, dtype=torch.float32)
    return reader(
        source_batch.episodes.query.tokens,
        source_batch.episodes.query.attention_mask.bool(),
        source_batch.episodes.query_read_index,
        initial,
        terminal,
        teacher_program=specs if teacher else None,
    )


def _truth_loss(logits: torch.Tensor, target_batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    targets = target_batch.episodes.query.targets.gather(
        1,
        target_batch.episodes.query_read_index[:, None],
    ).squeeze(1)
    factual = F.cross_entropy(logits.float(), targets)
    pairs = _reader_pairs_from_logits(logits, target_batch)
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
    return 0.25 * factual + 0.5 * (world + command), {
        "command_binding": command,
        "factual": factual,
        "world_binding": world,
    }


def _compiler_loss(
    output: TypedQueryReaderOutput,
    specs: ETTRQuerySpecBatch,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    operation = F.cross_entropy(output.operation_logits.float(), specs.operation)
    present = F.cross_entropy(
        output.argument_present_logits.float().flatten(0, 1),
        specs.argument_mask.long().flatten(),
    )
    if not bool(specs.argument_mask.any()):
        raise TypedQueryPilotError("typed query compiler has no argument support")
    arguments = F.cross_entropy(
        output.argument_logits.float()[specs.argument_mask],
        specs.arguments[specs.argument_mask],
    )
    return operation + 0.5 * (present + arguments), {
        "argument": arguments,
        "argument_present": present,
        "operation": operation,
    }


def _loss(
    reader: TypedQueryStateReader,
    source_batch,
    target_batch,
    specs: ETTRQuerySpecBatch,
    *,
    teacher_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    autonomous = _forward(reader, source_batch, specs, teacher=False)
    autonomous_loss, autonomous_parts = _truth_loss(
        autonomous.vocab_logits,
        target_batch,
    )
    compiler, compiler_parts = _compiler_loss(autonomous, specs)
    teacher_loss = torch.zeros_like(autonomous_loss)
    if teacher_weight > 0.0:
        teacher_output = _forward(reader, source_batch, specs, teacher=True)
        teacher_loss = _truth_loss(teacher_output.vocab_logits, target_batch)[0]
    loss = autonomous_loss + 0.5 * compiler + teacher_weight * teacher_loss
    parts = {
        "autonomous": autonomous_loss,
        "compiler": compiler,
        "teacher": teacher_loss,
        **{f"autonomous_{name}": value for name, value in autonomous_parts.items()},
        **{f"compiler_{name}": value for name, value in compiler_parts.items()},
    }
    return loss, {
        name: float(value.detach().cpu()) for name, value in parts.items()
    }


def _compiler_counts(
    output: TypedQueryReaderOutput,
    specs: ETTRQuerySpecBatch,
) -> dict[str, int]:
    operation = output.operation_logits.argmax(dim=-1).eq(specs.operation)
    present = output.argument_present_logits.argmax(dim=-1).bool()
    arguments = output.argument_logits.argmax(dim=-1)
    argument_correct = arguments.eq(specs.arguments) | ~specs.argument_mask
    present_exact = present.eq(specs.argument_mask).all(dim=-1)
    argument_exact = argument_correct.all(dim=-1)
    return {
        "argument_correct": int(argument_correct[specs.argument_mask].sum()),
        "argument_total": int(specs.argument_mask.sum()),
        "exact_program": int((operation & present_exact & argument_exact).sum()),
        "operation_correct": int(operation.sum()),
        "rows": specs.operation.numel(),
    }


def _evaluate(
    reader: TypedQueryStateReader,
    *,
    stream: ETTRV3StreamingRelease,
    packet_index: ETTRDiskPacketSufficiencyIndex,
    transcoder: TokenNativeETTRTranscoder,
    device: torch.device,
    data_seed: int,
    max_batches: int,
) -> dict[str, object]:
    reader.eval()
    rows = {
        "autonomous": {"world": [], "command": []},
        "oracle_program": {"world": [], "command": []},
    }
    factual = {"autonomous": 0, "oracle_program": 0}
    compiler = {
        "argument_correct": 0,
        "argument_total": 0,
        "exact_program": 0,
        "operation_correct": 0,
        "rows": 0,
    }
    iterator = iter_batches_with_query_specs(
        stream,
        "development",
        epoch=0,
        seed=data_seed,
    )
    for observed, (_position, cpu_batch, cpu_specs) in enumerate(iterator):
        if observed >= max_batches:
            break
        packet_index.verify_validation((cpu_batch,))
        source_batch = move_continuation_batch(cpu_batch, device)
        target_batch = move_continuation_batch(
            transcoder.transcode_batch(cpu_batch),
            device,
        )
        specs = cpu_specs.to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = {
                "autonomous": _forward(reader, source_batch, specs, teacher=False),
                "oracle_program": _forward(reader, source_batch, specs, teacher=True),
            }
        targets = target_batch.episodes.query.targets.gather(
            1,
            target_batch.episodes.query_read_index[:, None],
        ).squeeze(1)
        counts = _compiler_counts(outputs["autonomous"], specs)
        for name, value in counts.items():
            compiler[name] += value
        (
            _world_packet,
            _world_command,
            world_target,
            _command_packet,
            _command_command,
            command_target,
        ) = target_batch.causal_rectangles.intervention_indices()
        depths = target_batch.transaction_targets.step_mask.sum(-1)
        for mode, output in outputs.items():
            logits = output.vocab_logits
            factual[mode] += int(logits.argmax(-1).eq(targets).sum())
            pairs = _reader_pairs_from_logits(logits, target_batch)
            for factor, pair in pairs.items():
                target_index = world_target if factor == "world" else command_target
                rows[mode][factor].extend(
                    _annotate_pair_rows(pair, depths.index_select(0, target_index))
                )
    expected = max_batches * 16
    if compiler["rows"] != expected:
        raise TypedQueryPilotError("typed query evaluation support differs")
    return {
        "batches": max_batches,
        "compiler": {
            "argument_accuracy": (
                compiler["argument_correct"] / compiler["argument_total"]
            ),
            "exact_program_accuracy": compiler["exact_program"] / expected,
            "operation_accuracy": compiler["operation_correct"] / expected,
            "rows": expected,
        },
        **{
            mode: {
                "factual_top1": factual[mode] / expected,
                "oracle_terminal_reader": {
                    factor: _summary(values)
                    for factor, values in rows[mode].items()
                },
            }
            for mode in ("autonomous", "oracle_program")
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise TypedQueryPilotError("typed query pilot requires one process")
    if not torch.cuda.is_available():
        raise TypedQueryPilotError("typed query pilot requires CUDA")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    if "H100" not in torch.cuda.get_device_name(device).upper():
        raise TypedQueryPilotError("typed query pilot requires an H100")

    parent_contract = _read_hash_bound_json(
        args.parent_run_contract,
        expected_sha256=args.parent_run_contract_sha256,
        label="parent run contract",
    )
    parent, parent_payload = _load_parent(
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
    transcoder = TokenNativeETTRTranscoder(args.tokenizer, args.target_tokenizer)
    if parent.base.cfg.vocab_size != transcoder.target_vocab_size:
        raise TypedQueryPilotError("typed query target vocabulary differs")
    answer_token_ids = answer_token_ids_from_tokenizer(args.target_tokenizer)
    torch.manual_seed(args.model_seed)
    torch.cuda.manual_seed_all(args.model_seed)
    reader = TypedQueryStateReader(
        parent.config,
        source_vocab_size=stream.tokenizer.get_vocab_size(),
        target_vocab_size=parent.base.cfg.vocab_size,
        answer_token_ids=answer_token_ids,
        width=args.width,
        query_layers=args.query_layers,
        state_layers=args.state_layers,
        num_heads=args.num_heads,
    )
    parent_parameters = sum(parameter.numel() for parameter in parent.parameters())
    old_reader_parameters = sum(
        parameter.numel() for parameter in parent.query_reader.parameters()
    )
    reader_parameters = sum(parameter.numel() for parameter in reader.parameters())
    replacement_system_parameters = (
        parent_parameters - old_reader_parameters + reader_parameters
    )
    if replacement_system_parameters > SYSTEM_PARAMETER_CAP:
        raise TypedQueryPilotError("typed query reader exceeds system parameter cap")
    del parent
    reader.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        reader.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )
    packet_index = ETTRDiskPacketSufficiencyIndex(stream.packet_index_root)
    try:
        args.output.mkdir(mode=0o700)
        initial_path = args.output / "typed-query-reader-initial.safetensors"
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in reader.state_dict().items()
            },
            initial_path,
        )
        os.chmod(initial_path, 0o400)
        initial_sha256 = _sha256_file(initial_path)
        before = _evaluate(
            reader,
            stream=stream,
            packet_index=packet_index,
            transcoder=transcoder,
            device=device,
            data_seed=args.data_seed,
            max_batches=args.eval_batches,
        )
        contract = {
            "answer_token_ids": list(answer_token_ids),
            "architecture": {
                "num_heads": args.num_heads,
                "query_layers": args.query_layers,
                "state_layers": args.state_layers,
                "width": args.width,
            },
            "data_seed": args.data_seed,
            "eval_batches": args.eval_batches,
            "gradient_clip": args.gradient_clip,
            "learning_rate": args.learning_rate,
            "model_seed": args.model_seed,
            "oracle_at_autonomous_inference": False,
            "oracle_program_report_is_promotable": False,
            "parent_joint_model_sha256": args.parent_joint_model_sha256,
            "parent_run_contract_sha256": args.parent_run_contract_sha256,
            "query_operations": list(QUERY_OPERATIONS),
            "reader_parameters": reader_parameters,
            "release_file_sha256": args.release_sha256,
            "replacement_system_parameters": replacement_system_parameters,
            "schema": CONTRACT_SCHEMA,
            "source_commit": args.source_commit,
            "start_position": args.start_position,
            "teacher_decay_updates": args.teacher_decay_updates,
            "token_transcode": transcode_receipt_value(transcoder.receipt),
            "updates": args.updates,
        }
        contract_sha256 = _write_no_replace(
            args.output / "pilot-contract.json",
            _canonical_bytes(contract),
        )
        _write_no_replace(args.output / "train.jsonl", b"", mode=0o600)

        reader.train()
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
                raise TypedQueryPilotError("typed query train stream exhausted") from exc
            packet_index.verify_train((cpu_batch,))
            source_batch = move_continuation_batch(cpu_batch, device)
            target_batch = move_continuation_batch(
                transcoder.transcode_batch(cpu_batch),
                device,
            )
            specs = cpu_specs.to(device)
            teacher_weight = max(
                0.0,
                1.0 - (update - 1) / args.teacher_decay_updates,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = _loss(
                    reader,
                    source_batch,
                    target_batch,
                    specs,
                    teacher_weight=teacher_weight,
                )
            if not bool(torch.isfinite(loss)):
                raise TypedQueryPilotError("typed query loss is nonfinite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                reader.parameters(),
                args.gradient_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
            last_loss = float(loss.detach().cpu())
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
                                "schema": "shohin-ettr-typed-query-metric-v1",
                                "teacher_weight": teacher_weight,
                                "update": update,
                            }
                        )
                    )
        os.chmod(args.output / "train.jsonl", 0o400)
        after = _evaluate(
            reader,
            stream=stream,
            packet_index=packet_index,
            transcoder=transcoder,
            device=device,
            data_seed=args.data_seed,
            max_batches=args.eval_batches,
        )
        final_path = args.output / "typed-query-reader-final.safetensors"
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in reader.state_dict().items()
            },
            final_path,
        )
        os.chmod(final_path, 0o400)
        final_sha256 = _sha256_file(final_path)
        report = {
            "contract_sha256": contract_sha256,
            "device": torch.cuda.get_device_name(device),
            "evaluation": {"after": after, "before": before},
            "final_reader_sha256": final_sha256,
            "initial_reader_sha256": initial_sha256,
            "last_loss": last_loss,
            "last_position": last_position,
            "oracle_boundary": (
                "initial_and_terminal_states_only; query program targets are "
                "training-only and omitted from autonomous evaluation"
            ),
            "reader_parameters": reader_parameters,
            "replacement_system_parameters": replacement_system_parameters,
            "schema": REPORT_SCHEMA,
            "source_verification": source_verification,
            "status": "pass",
            "updates_completed": args.updates,
        }
        _write_no_replace(
            args.output / "report.json",
            _canonical_bytes(report),
        )
        files = (
            "pilot-contract.json",
            "report.json",
            "train.jsonl",
            "typed-query-reader-final.safetensors",
            "typed-query-reader-initial.safetensors",
        )
        _write_no_replace(
            args.output / "SHA256SUMS",
            "".join(
                f"{_sha256_file(args.output / name)}  {name}\n" for name in files
            ).encode("ascii"),
        )
        os.chmod(args.output, 0o500)
    except BaseException:
        if args.output.exists():
            shutil.rmtree(args.output)
        raise
    finally:
        packet_index.close()
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
