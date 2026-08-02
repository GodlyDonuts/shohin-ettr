#!/usr/bin/env python3
"""Evaluate a sealed terminal-state quotient compiler on a fresh ordering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
from typing import Mapping, Sequence

from safetensors.torch import load_file
import torch

from endogenous_typed_theory_reactor import SYSTEM_PARAMETER_CAP
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_v3_streaming import ETTRV3StreamingRelease
from eval_algebraic_query_joint_state import (
    _evaluate,
    _load_compiler,
    _strict_load_joint_model,
)
from eval_ettr_v3 import _parameter_sha256, _read_hash_bound_json
from parallel_terminal_state_compiler import (
    ParallelTerminalStateCompiler,
    ParallelTerminalStateReactor,
)
from train_ettr_component_island import (
    _canonical_bytes,
    _sha256_file,
    _write_no_replace,
)
from train_parallel_terminal_state_pilot import (
    ATOMIC_CONTRACT_SCHEMA as PILOT_ATOMIC_CONTRACT_SCHEMA,
    ATOMIC_REPORT_SCHEMA as PILOT_ATOMIC_REPORT_SCHEMA,
    CAUSAL_DELTA_CONTRACT_SCHEMA as PILOT_CAUSAL_DELTA_CONTRACT_SCHEMA,
    CAUSAL_DELTA_REPORT_SCHEMA as PILOT_CAUSAL_DELTA_REPORT_SCHEMA,
    CONTRACT_SCHEMA as PILOT_CONTRACT_SCHEMA,
    LEGACY_CONTRACT_SCHEMA as PILOT_LEGACY_CONTRACT_SCHEMA,
    LEGACY_REPORT_SCHEMA as PILOT_LEGACY_REPORT_SCHEMA,
    REPORT_SCHEMA as PILOT_REPORT_SCHEMA,
)


CONTRACT_SCHEMA = "shohin-ettr-parallel-terminal-state-eval-contract-v1"
REPORT_SCHEMA = "shohin-ettr-parallel-terminal-state-eval-report-v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_RUN_FILES = (
    "pilot-contract.json",
    "report.json",
    "terminal-compiler-final.safetensors",
    "terminal-compiler-initial.safetensors",
    "train.jsonl",
)


class ParallelTerminalStateEvaluationError(RuntimeError):
    """The terminal-state evaluation violated its sealed contract."""


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
    parser.add_argument("--terminal-run-dir", type=Path, required=True)
    parser.add_argument("--terminal-run-sha256s-sha256", required=True)
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
        args.terminal_run_dir,
        args.output,
    )
    hashes = (
        args.release_sha256,
        args.joint_model_sha256,
        args.joint_run_contract_sha256,
        args.compiler_sha256,
        args.compiler_contract_sha256,
        args.terminal_run_sha256s_sha256,
    )
    if (
        any(not path.is_absolute() for path in paths)
        or any(_HEX64.fullmatch(value) is None for value in hashes)
        or _HEX40.fullmatch(args.source_commit) is None
        or not 0 <= args.data_seed < 2**63
        or args.max_batches < 2
        or args.output.exists()
        or args.output.is_symlink()
        or not args.output.parent.is_dir()
    ):
        raise ParallelTerminalStateEvaluationError(
            "terminal-state evaluation arguments differ"
        )


def _load_state_file(path: Path) -> dict[str, torch.Tensor]:
    try:
        return dict(load_file(str(path), device="cpu"))
    except Exception as exc:
        raise ParallelTerminalStateEvaluationError(
            "terminal-state component is unreadable"
        ) from exc


def _require_module_state(
    module: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
) -> None:
    current = module.state_dict()
    if current.keys() != state.keys() or any(
        not torch.equal(current[name].detach().cpu(), state[name])
        for name in current
    ):
        raise ParallelTerminalStateEvaluationError(
            "terminal-state initial component differs"
        )


def _run_receipt(run_dir: Path, expected_sha256: str) -> dict[str, str]:
    sums_path = run_dir / "SHA256SUMS"
    if _sha256_file(sums_path) != expected_sha256:
        raise ParallelTerminalStateEvaluationError(
            "terminal-state run receipt differs"
        )
    expected: dict[str, str] = {}
    try:
        lines = sums_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ParallelTerminalStateEvaluationError(
            "terminal-state run receipt is unreadable"
        ) from exc
    for line in lines:
        fields = line.split("  ")
        if (
            len(fields) != 2
            or _HEX64.fullmatch(fields[0]) is None
            or fields[1] not in _RUN_FILES
            or fields[1] in expected
        ):
            raise ParallelTerminalStateEvaluationError(
                "terminal-state run receipt differs"
            )
        expected[fields[1]] = fields[0]
    if tuple(sorted(expected)) != tuple(sorted(_RUN_FILES)):
        raise ParallelTerminalStateEvaluationError(
            "terminal-state run receipt differs"
        )
    for name, digest in expected.items():
        if _sha256_file(run_dir / name) != digest:
            raise ParallelTerminalStateEvaluationError(
                "terminal-state run file differs"
            )
    return expected


def _load_terminal_compiler(
    args,
    *,
    model,
    provenance,
    replacement_system_parameters: int,
) -> tuple[dict[str, object], int]:
    expected = _run_receipt(
        args.terminal_run_dir,
        args.terminal_run_sha256s_sha256,
    )
    contract = _read_hash_bound_json(
        args.terminal_run_dir / "pilot-contract.json",
        expected_sha256=expected["pilot-contract.json"],
        label="terminal-state contract",
    )
    report = _read_hash_bound_json(
        args.terminal_run_dir / "report.json",
        expected_sha256=expected["report.json"],
        label="terminal-state report",
    )
    architecture = contract.get("architecture")
    if (
        contract.get("schema")
        not in (
            PILOT_ATOMIC_CONTRACT_SCHEMA,
            PILOT_CONTRACT_SCHEMA,
            PILOT_CAUSAL_DELTA_CONTRACT_SCHEMA,
            PILOT_LEGACY_CONTRACT_SCHEMA,
        )
        or report.get("schema")
        not in (
            PILOT_ATOMIC_REPORT_SCHEMA,
            PILOT_REPORT_SCHEMA,
            PILOT_CAUSAL_DELTA_REPORT_SCHEMA,
            PILOT_LEGACY_REPORT_SCHEMA,
        )
        or (
            (contract.get("schema") == PILOT_ATOMIC_CONTRACT_SCHEMA)
            != (report.get("schema") == PILOT_ATOMIC_REPORT_SCHEMA)
        )
        or (
            (contract.get("schema") == PILOT_CONTRACT_SCHEMA)
            != (report.get("schema") == PILOT_REPORT_SCHEMA)
        )
        or (
            (contract.get("schema") == PILOT_CAUSAL_DELTA_CONTRACT_SCHEMA)
            != (report.get("schema") == PILOT_CAUSAL_DELTA_REPORT_SCHEMA)
        )
        or report.get("status") != "pass"
        or not isinstance(architecture, Mapping)
        or report.get("contract_sha256") != expected["pilot-contract.json"]
        or contract.get("release_file_sha256") != args.release_sha256
        or contract.get("joint_model_sha256") != args.joint_model_sha256
        or contract.get("joint_run_contract_sha256")
        != args.joint_run_contract_sha256
        or contract.get("compiler_sha256") != args.compiler_sha256
        or contract.get("compiler_contract_sha256")
        != args.compiler_contract_sha256
        or report.get("protected_checkpoint_sha256")
        != provenance.checkpoint_sha256
        or report.get("initial_compiler_sha256")
        != expected["terminal-compiler-initial.safetensors"]
        or report.get("final_compiler_sha256")
        != expected["terminal-compiler-final.safetensors"]
        or architecture.get("direct_terminal_quotient") is not True
        or architecture.get("no_query_input") is not True
        or architecture.get("no_transaction_trace_claim") is not True
    ):
        raise ParallelTerminalStateEvaluationError(
            "terminal-state run lineage differs"
        )
    if contract.get("schema") in (
        PILOT_ATOMIC_CONTRACT_SCHEMA,
        PILOT_CONTRACT_SCHEMA,
        PILOT_CAUSAL_DELTA_CONTRACT_SCHEMA,
    ):
        objective = contract.get("objective")
        if (
            architecture.get("causal_rectangle_delta_credit") is not True
            or not isinstance(objective, Mapping)
            or objective.get("causal_pairing")
            != "complete-2x2-terminal-state-edges"
            or not isinstance(objective.get("causal_delta_weight"), (int, float))
            or float(objective["causal_delta_weight"]) <= 0.0
        ):
            raise ParallelTerminalStateEvaluationError(
                "terminal-state causal delta contract differs"
            )
    residual_edits = contract.get("schema") == PILOT_CONTRACT_SCHEMA
    atomic_edits = contract.get("schema") == PILOT_ATOMIC_CONTRACT_SCHEMA
    if residual_edits != (architecture.get("sparse_residual_edits") is True):
        raise ParallelTerminalStateEvaluationError(
            "terminal-state residual edit contract differs"
        )
    if atomic_edits != (architecture.get("atomic_typed_edits") is True):
        raise ParallelTerminalStateEvaluationError(
            "terminal-state atomic edit contract differs"
        )
    try:
        seed = int(architecture["seed"])
        width = int(architecture["width"])
        layers = int(architecture["layers"])
        num_heads = int(architecture["num_heads"])
        relation_width = int(architecture["relation_width"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ParallelTerminalStateEvaluationError(
            "terminal-state architecture differs"
        ) from exc
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    compiler = ParallelTerminalStateCompiler(
        model.config,
        width=width,
        layers=layers,
        num_heads=num_heads,
        relation_width=relation_width,
        residual_edits=residual_edits,
        atomic_edits=atomic_edits,
    ).to(
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
    )
    _require_module_state(
        compiler,
        _load_state_file(
            args.terminal_run_dir / "terminal-compiler-initial.safetensors"
        ),
    )
    try:
        incompatibility = compiler.load_state_dict(
            _load_state_file(
                args.terminal_run_dir / "terminal-compiler-final.safetensors"
            ),
            strict=True,
        )
    except RuntimeError as exc:
        raise ParallelTerminalStateEvaluationError(
            "terminal-state final component differs"
        ) from exc
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ParallelTerminalStateEvaluationError(
            "terminal-state final component differs"
        )
    compiler_parameters = sum(
        parameter.numel() for parameter in compiler.parameters()
    )
    removed_reactor_parameters = sum(
        parameter.numel() for parameter in model.reactor.parameters()
    )
    complete_parameters = (
        replacement_system_parameters
        - removed_reactor_parameters
        + compiler_parameters
    )
    if (
        contract.get("compiler_parameters") != compiler_parameters
        or contract.get("complete_system_parameters") != complete_parameters
        or complete_parameters > SYSTEM_PARAMETER_CAP
    ):
        raise ParallelTerminalStateEvaluationError(
            "terminal-state parameter receipt differs"
        )
    model.reactor = ParallelTerminalStateReactor(compiler, model.config)
    compiler.eval()
    return (
        {
            "complete_system_parameters": complete_parameters,
            "compiler_parameters": compiler_parameters,
            "contract_sha256": expected["pilot-contract.json"],
            "report_sha256": expected["report.json"],
            "run_sha256s_sha256": args.terminal_run_sha256s_sha256,
            "source_commit": contract["source_commit"],
            "training_initial_state": contract["training_initial_state"],
            "updates_completed": report["updates_completed"],
        },
        complete_parameters,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    if not torch.cuda.is_available():
        raise ParallelTerminalStateEvaluationError(
            "terminal-state evaluation requires CUDA"
        )
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    if (
        args.required_device_class == "h100"
        and "H100" not in torch.cuda.get_device_name(device).upper()
    ):
        raise ParallelTerminalStateEvaluationError(
            "terminal-state evaluation requires an H100"
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
    reader, compiler_contract, reader_parameters, replacement_parameters = (
        _load_compiler(args, model=model, stream=stream, device=device)
    )
    receipt, complete_parameters = _load_terminal_compiler(
        args,
        model=model,
        provenance=provenance,
        replacement_system_parameters=replacement_parameters,
    )
    contract = {
        "compiler_contract_sha256": args.compiler_contract_sha256,
        "compiler_sha256": args.compiler_sha256,
        "data_seed": args.data_seed,
        "fully_autonomous_arm": "autonomous_program_autonomous_state",
        "joint_model_sha256": args.joint_model_sha256,
        "joint_run_contract_sha256": args.joint_run_contract_sha256,
        "max_batches": args.max_batches,
        "non_promotable_diagnostic_arms": [
            "oracle_program_autonomous_state",
            "autonomous_program_oracle_state",
            "oracle_program_oracle_state",
        ],
        "protected_checkpoint_sha256": provenance.checkpoint_sha256,
        "reader_parameters": reader_parameters,
        "release_file_sha256": args.release_sha256,
        "replacement_system_parameters": complete_parameters,
        "required_device_class": args.required_device_class,
        "schema": CONTRACT_SCHEMA,
        "source_commit": args.source_commit,
        "terminal_state_receipt": receipt,
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
            "contract_sha256": contract_sha256,
            "device": torch.cuda.get_device_name(device),
            "evaluation": evaluation,
            "joint_model_optimizer_step": joint_payload["optimizer_step"],
            "joint_model_parameter_sha256": _parameter_sha256(model),
            "joint_training_source_commit": joint_contract["source_commit"],
            "reader_parameters": reader_parameters,
            "replacement_system_parameters": complete_parameters,
            "runtime_precision": str(next(model.parameters()).dtype),
            "schema": REPORT_SCHEMA,
            "source_verification": source_verification,
            "status": "pass",
            "terminal_state_receipt": receipt,
        }
        _write_no_replace(
            args.output / "report.json",
            _canonical_bytes(report),
        )
        names = ("evaluation-contract.json", "report.json")
        _write_no_replace(
            args.output / "SHA256SUMS",
            "".join(
                f"{_sha256_file(args.output / name)}  {name}\n"
                for name in names
            ).encode("ascii"),
        )
        for path in args.output.iterdir():
            path.chmod(0o400)
        args.output.chmod(0o500)
    except BaseException:
        if args.output.exists():
            shutil.rmtree(args.output, ignore_errors=True)
        raise
    finally:
        packet_index.close()
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
