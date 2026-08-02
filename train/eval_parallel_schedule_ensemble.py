#!/usr/bin/env python3
"""Evaluate a deterministic mean ensemble of trained ETTR schedules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
from types import SimpleNamespace
from typing import Sequence

import torch

from endogenous_typed_theory_reactor import SYSTEM_PARAMETER_CAP
from ettr_checkpoint import load_protected_base_model
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_v3_streaming import ETTRV3StreamingRelease
from eval_algebraic_query_joint_state import (
    _ARMS,
    _evaluate,
    _load_compiler,
    _load_parallel_schedule,
    _strict_load_joint_model,
)
from eval_ettr_v3 import _parameter_sha256
from parallel_addressed_transaction_compiler import (
    MeanParallelAddressedScheduleCompiler,
    ParallelScheduledReactor,
)
from train_ettr_component_island import (
    _canonical_bytes,
    _sha256_file,
    _write_no_replace,
)


CONTRACT_SCHEMA = "shohin-ettr-parallel-schedule-ensemble-contract-v1"
REPORT_SCHEMA = "shohin-ettr-parallel-schedule-ensemble-report-v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ParallelScheduleEnsembleError(RuntimeError):
    """The deterministic schedule ensemble violated its sealed contract."""


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
    parser.add_argument(
        "--schedule-run-dir",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--schedule-run-sha256s-sha256",
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--max-batches", type=int, default=32)
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
        *args.schedule_run_dir,
    )
    hashes = (
        args.release_sha256,
        args.joint_model_sha256,
        args.joint_run_contract_sha256,
        args.compiler_sha256,
        args.compiler_contract_sha256,
        *args.schedule_run_sha256s_sha256,
    )
    if (
        any(not path.is_absolute() for path in paths)
        or any(_HEX64.fullmatch(value) is None for value in hashes)
        or _HEX40.fullmatch(args.source_commit) is None
        or not 0 <= args.data_seed < 2**63
        or args.max_batches < 2
        or not 2 <= len(args.schedule_run_dir) <= 6
        or len(args.schedule_run_dir)
        != len(args.schedule_run_sha256s_sha256)
        or len(set(args.schedule_run_dir)) != len(args.schedule_run_dir)
        or args.output.exists()
        or args.output.is_symlink()
        or not args.output.parent.is_dir()
    ):
        raise ParallelScheduleEnsembleError(
            "parallel schedule ensemble arguments differ"
        )


def _load_ensemble(
    args: argparse.Namespace,
    *,
    model,
    provenance,
    replacement_system_parameters: int,
):
    original_reactor = model.reactor
    removed_reactor_parameters = sum(
        parameter.numel() for parameter in original_reactor.parameters()
    )
    compilers = []
    receipts = []
    for path, digest in zip(
        args.schedule_run_dir,
        args.schedule_run_sha256s_sha256,
        strict=True,
    ):
        model.reactor = original_reactor
        member_args = SimpleNamespace(
            **(
                vars(args)
                | {
                    "schedule_run_dir": path,
                    "schedule_run_sha256s_sha256": digest,
                }
            )
        )
        receipt, _complete_parameters = _load_parallel_schedule(
            member_args,
            model=model,
            provenance=provenance,
            replacement_system_parameters=replacement_system_parameters,
        )
        if receipt is None:
            raise ParallelScheduleEnsembleError(
                "parallel schedule ensemble member is absent"
            )
        compilers.append(model.reactor.compiler)
        receipts.append(receipt)
    ensemble = MeanParallelAddressedScheduleCompiler(compilers)
    ensemble_parameters = sum(
        parameter.numel() for parameter in ensemble.parameters()
    )
    complete_parameters = (
        replacement_system_parameters
        - removed_reactor_parameters
        + ensemble_parameters
    )
    if complete_parameters > SYSTEM_PARAMETER_CAP:
        raise ParallelScheduleEnsembleError(
            "parallel schedule ensemble exceeds parameter cap"
        )
    model.reactor = ParallelScheduledReactor(ensemble, model.config)
    model.eval()
    return ensemble, receipts, complete_parameters


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    if not torch.cuda.is_available():
        raise ParallelScheduleEnsembleError(
            "parallel schedule ensemble requires CUDA"
        )
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    if (
        args.required_device_class == "h100"
        and "H100" not in torch.cuda.get_device_name(device).upper()
    ):
        raise ParallelScheduleEnsembleError(
            "parallel schedule ensemble requires an H100"
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
    protected_provenance = load_protected_base_model(
        args.protected_checkpoint
    )[1]
    if (
        provenance.checkpoint_sha256
        != stream.manifest.protected_checkpoint_sha256
        or provenance.checkpoint_sha256
        != protected_provenance.checkpoint_sha256
    ):
        raise ParallelScheduleEnsembleError(
            "parallel schedule ensemble protected checkpoint differs"
        )
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
    ensemble, schedule_receipts, complete_parameters = _load_ensemble(
        args,
        model=model,
        provenance=provenance,
        replacement_system_parameters=replacement_system_parameters,
    )
    contract = {
        "aggregation": "uniform_probability_mean_then_single_hard_schedule",
        "compiler_contract_sha256": args.compiler_contract_sha256,
        "compiler_sha256": args.compiler_sha256,
        "complete_system_parameters": complete_parameters,
        "data_seed": args.data_seed,
        "fully_autonomous_arm": "autonomous_program_autonomous_state",
        "joint_model_sha256": args.joint_model_sha256,
        "joint_run_contract_sha256": args.joint_run_contract_sha256,
        "max_batches": args.max_batches,
        "member_count": len(ensemble.compilers),
        "non_promotable_diagnostic_arms": list(_ARMS[1:]),
        "protected_checkpoint_sha256": provenance.checkpoint_sha256,
        "reader_parameters": reader_parameters,
        "release_file_sha256": args.release_sha256,
        "required_device_class": args.required_device_class,
        "schedule_receipts": schedule_receipts,
        "schema": CONTRACT_SCHEMA,
        "source_commit": args.source_commit,
    }
    packet_index = ETTRDiskPacketSufficiencyIndex(stream.packet_index_root)
    try:
        args.output.mkdir(mode=0o700)
        contract_sha256 = _write_no_replace(
            args.output / "evaluation-contract.json",
            _canonical_bytes(contract),
        )
        evaluation = _evaluate(
            reader,
            model,
            stream=stream,
            packet_index=packet_index,
            device=device,
            data_seed=args.data_seed,
            max_batches=args.max_batches,
        )
        report = {
            "compiler_contract_source_commit": compiler_contract["source_commit"],
            "complete_system_parameters": complete_parameters,
            "contract_sha256": contract_sha256,
            "device": torch.cuda.get_device_name(device),
            "evaluation": evaluation,
            "joint_model_optimizer_step": joint_payload["optimizer_step"],
            "joint_model_parameter_sha256": _parameter_sha256(model),
            "joint_training_source_commit": joint_contract["source_commit"],
            "member_count": len(ensemble.compilers),
            "reader_parameters": reader_parameters,
            "runtime_precision": str(next(model.parameters()).dtype),
            "schedule_receipts": schedule_receipts,
            "schema": REPORT_SCHEMA,
            "source_verification": source_verification,
            "status": "pass",
        }
        _write_no_replace(
            args.output / "report.json",
            _canonical_bytes(report),
        )
        files = ("evaluation-contract.json", "report.json")
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
