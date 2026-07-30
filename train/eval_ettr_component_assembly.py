#!/usr/bin/env python3
"""Evaluate a hash-bound ETTR component assembly end to end.

Component-island metrics use training-only oracle interfaces. This evaluator
instead loads one compiler, reactor, and query reader into the ordinary
source-deleted model and scores their autonomous composition on the immutable
development stream.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Mapping, Sequence

from safetensors.torch import load as load_safetensors
import torch
from torch import nn

from ettr_data_contract import continuation_batch_payload_sha256
from ettr_objectives import ETTRObjectiveConfig
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_train_step import ETTRCompositeTrainingSubject
from ettr_v3_streaming import ETTRV3StreamingRelease, move_continuation_batch
from eval_ettr_v3 import (
    ETTRV3EvaluationError,
    _arm_summary,
    _build_model,
    _canonical_bytes,
    _evaluate,
    _paired_loss_summary,
    _parameter_sha256,
    _physical_file,
    _read_hash_bound_json,
    _validate_run_contract,
    _write_no_replace,
)


REPORT_SCHEMA = "shohin-ettr-component-assembly-development-evaluation-v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_COMPONENTS = ("compiler", "reactor", "query_reader")


class ETTRComponentAssemblyError(ETTRV3EvaluationError):
    """A component assembly cannot preserve its evaluation contract."""


def _read_hash_bound_bytes(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> bytes:
    if _HEX64.fullmatch(expected_sha256) is None:
        raise ETTRComponentAssemblyError(f"{label} SHA-256 differs")
    before = _physical_file(path, label)
    if before.st_mode & 0o222:
        raise ETTRComponentAssemblyError(f"{label} is mutable")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size != before.st_size
        ):
            raise ETTRComponentAssemblyError(f"{label} identity changed")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            payload = source.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        any(getattr(opened, name) != getattr(after, name) for name in identity)
        or len(payload) != opened.st_size
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise ETTRComponentAssemblyError(f"{label} changed or hash differs")
    return payload


def load_hash_bound_component(
    module: nn.Module,
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> str:
    """Strictly load one immutable component and return its verified hash."""

    payload = _read_hash_bound_bytes(
        path,
        expected_sha256=expected_sha256,
        label=label,
    )
    try:
        state = load_safetensors(payload)
        incompatibility = module.load_state_dict(state, strict=True)
    except Exception as exc:
        raise ETTRComponentAssemblyError(
            f"{label} strict load differs"
        ) from exc
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ETTRComponentAssemblyError(
            f"{label} strict load differs"
        )
    return expected_sha256


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--protected-checkpoint", type=Path, required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--run-contract-sha256", required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--compiler-sha256", required=True)
    parser.add_argument("--reactor", type=Path, required=True)
    parser.add_argument("--reactor-sha256", required=True)
    parser.add_argument("--query-reader", type=Path, required=True)
    parser.add_argument("--query-reader-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--architecture-seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--max-batches", type=int, default=64)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    hashes = (
        args.release_sha256,
        args.run_contract_sha256,
        args.compiler_sha256,
        args.reactor_sha256,
        args.query_reader_sha256,
    )
    paths = (
        args.release_root,
        args.data_root,
        args.tokenizer,
        args.protected_checkpoint,
        args.run_contract,
        args.compiler,
        args.reactor,
        args.query_reader,
        args.output,
    )
    if (
        any(_HEX64.fullmatch(value) is None for value in hashes)
        or _HEX40.fullmatch(args.source_commit) is None
        or not 0 <= args.architecture_seed < 2**63
        or not 0 <= args.data_seed < 2**63
        or args.max_batches < 2
        or any(not path.is_absolute() for path in paths)
    ):
        raise ETTRComponentAssemblyError(
            "component assembly arguments differ"
        )


def _gates(
    raw: Mapping[str, object],
    candidate: Mapping[str, object],
    paired: Mapping[str, object],
) -> dict[str, bool]:
    raw_rates = raw["query_binding_margin_rates"]
    candidate_rates = candidate["query_binding_margin_rates"]
    if not isinstance(raw_rates, Mapping) or not isinstance(
        candidate_rates,
        Mapping,
    ):
        raise ETTRComponentAssemblyError(
            "component assembly margin summary differs"
        )

    def rate_gain(name: str) -> bool:
        raw_rate = raw_rates[name]
        candidate_rate = candidate_rates[name]
        return (
            isinstance(raw_rate, (float, int))
            and isinstance(candidate_rate, (float, int))
            and candidate_rate > raw_rate
        )

    total = paired["total"]
    if not isinstance(total, Mapping):
        raise ETTRComponentAssemblyError(
            "component assembly paired summary differs"
        )
    return {
        "all_metrics_finite": True,
        "command_query_margin_rate_increased": rate_gain("command"),
        "paired_total_loss_upper_95_below_zero": bool(
            total["improved_with_upper_95_below_zero"]
        ),
        "world_query_margin_rate_increased": rate_gain("world"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    if not torch.cuda.is_available():
        raise ETTRComponentAssemblyError(
            "component assembly evaluation requires CUDA"
        )
    device = torch.device("cuda", 0)
    if "H100" not in torch.cuda.get_device_name(device).upper():
        raise ETTRComponentAssemblyError(
            "component assembly evaluation requires H100"
        )
    if args.output.exists() or args.output.is_symlink():
        raise ETTRComponentAssemblyError(
            "component assembly output already exists"
        )

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
    model_config, _optimizer_config = _validate_run_contract(
        run_contract,
        release_sha256=args.release_sha256,
        release_source_commit=stream.release["source_commit"],
        architecture_seed=args.architecture_seed,
    )
    raw_model, raw_provenance = _build_model(
        args.protected_checkpoint,
        architecture_seed=args.architecture_seed,
        model_config=model_config,
        device=device,
    )
    candidate_model, candidate_provenance = _build_model(
        args.protected_checkpoint,
        architecture_seed=args.architecture_seed,
        model_config=model_config,
        device=device,
    )
    if (
        raw_provenance != candidate_provenance
        or raw_provenance.checkpoint_sha256
        != stream.manifest.protected_checkpoint_sha256
        or run_contract["parameter_receipt"]
        != asdict(raw_model.parameter_receipt())
        or run_contract["parameter_receipt"]
        != asdict(candidate_model.parameter_receipt())
    ):
        raise ETTRComponentAssemblyError(
            "component assembly base contract differs"
        )

    component_paths = {
        "compiler": args.compiler,
        "reactor": args.reactor,
        "query_reader": args.query_reader,
    }
    component_hashes = {
        "compiler": args.compiler_sha256,
        "reactor": args.reactor_sha256,
        "query_reader": args.query_reader_sha256,
    }
    for name in _COMPONENTS:
        load_hash_bound_component(
            getattr(candidate_model, name),
            component_paths[name],
            expected_sha256=component_hashes[name],
            label=name.replace("_", " "),
        )
    candidate_model.eval()
    raw_parameter_sha256 = _parameter_sha256(raw_model)
    candidate_parameter_sha256 = _parameter_sha256(candidate_model)
    if candidate_parameter_sha256 == raw_parameter_sha256:
        raise ETTRComponentAssemblyError(
            "component assembly parameters did not change"
        )

    objective_config = ETTRObjectiveConfig(
        vocab_size=raw_model.base.cfg.vocab_size
    )
    raw_subject = ETTRCompositeTrainingSubject(
        raw_model,
        objective_config,
        None,
        hard_transactions=True,
    )
    candidate_subject = ETTRCompositeTrainingSubject(
        candidate_model,
        objective_config,
        None,
        hard_transactions=True,
    )
    raw_losses: list[dict[str, float]] = []
    raw_counts: list[dict[str, int]] = []
    candidate_losses: list[dict[str, float]] = []
    candidate_counts: list[dict[str, int]] = []
    batches = []
    packet_index = ETTRDiskPacketSufficiencyIndex(stream.packet_index_root)
    try:
        iterator = stream.iter_positioned_batches(
            "development",
            rank=0,
            world_size=1,
            epoch=0,
            seed=args.data_seed,
        )
        for position, cpu_batch in iterator:
            if len(raw_losses) >= args.max_batches:
                break
            packet_index.verify_validation((cpu_batch,))
            payload_sha256 = continuation_batch_payload_sha256(cpu_batch)
            batch = move_continuation_batch(cpu_batch, device)
            batch.validate(raw_model.config, objective_config)
            raw_loss, raw_count = _evaluate(raw_subject, batch)
            candidate_loss, candidate_count = _evaluate(
                candidate_subject,
                batch,
            )
            raw_losses.append(raw_loss)
            raw_counts.append(raw_count)
            candidate_losses.append(candidate_loss)
            candidate_counts.append(candidate_count)
            batches.append(
                {
                    "batch_payload_sha256": payload_sha256,
                    "losses": {
                        "component_assembly": candidate_loss,
                        "raw": raw_loss,
                    },
                    "position": position,
                }
            )
            del batch
    finally:
        packet_index.close()
    if len(raw_losses) != args.max_batches:
        raise ETTRComponentAssemblyError(
            "development split contains fewer batches than requested"
        )

    raw_summary = _arm_summary(raw_losses, raw_counts)
    candidate_summary = _arm_summary(
        candidate_losses,
        candidate_counts,
    )
    paired = _paired_loss_summary(raw_losses, candidate_losses)
    gates = _gates(raw_summary, candidate_summary, paired)
    gates["component_parameters_changed"] = (
        candidate_parameter_sha256 != raw_parameter_sha256
    )
    gates["strict_learning_signal"] = all(gates.values())
    report = {
        "architecture_seed": args.architecture_seed,
        "arms": {
            "component_assembly": {
                **candidate_summary,
                "component_sha256": component_hashes,
                "parameter_sha256": candidate_parameter_sha256,
            },
            "raw": {
                **raw_summary,
                "parameter_sha256": raw_parameter_sha256,
            },
        },
        "batches": batches,
        "data_seed": args.data_seed,
        "device": {
            "bf16": torch.cuda.is_bf16_supported(),
            "name": torch.cuda.get_device_name(device),
        },
        "gates": gates,
        "paired_component_assembly_minus_raw": paired,
        "protected_checkpoint_sha256": raw_provenance.checkpoint_sha256,
        "release_file_sha256": args.release_sha256,
        "release_manifest_sha256": stream.manifest.sha256(),
        "run_contract_sha256": args.run_contract_sha256,
        "schema": REPORT_SCHEMA,
        "source_commit": args.source_commit,
        "source_verification": source_verification,
        "split": "development",
    }
    digest = _write_no_replace(args.output, _canonical_bytes(report))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_sha256": digest,
                "strict_learning_signal": gates["strict_learning_signal"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
