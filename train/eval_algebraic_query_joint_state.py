#!/usr/bin/env python3
"""Cross a trained query compiler with oracle and autonomous ETTR state.

The four evaluation arms form a strict 2x2 localization board:

* predicted query program x architecture-produced state (fully autonomous),
* oracle query program x architecture-produced state,
* predicted query program x oracle state, and
* oracle query program x oracle state.

Only the first arm is an end-to-end candidate.  The other three arms are
non-promotable interface diagnostics.  No query target, operation label, or
packet target enters the fully autonomous forward.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import re
import shutil
from typing import Mapping, Sequence

from safetensors.torch import load_file
import torch

from algebraic_query_state_reader import AlgebraicQueryStateReader
from endogenous_typed_theory_reactor import SYSTEM_PARAMETER_CAP, TRANSACTION_COUNT
from ettr_checkpoint import load_protected_base_model
from ettr_objectives import ETTRObjectiveConfig
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_query_supervision import iter_batches_with_query_specs
from ettr_v3_streaming import ETTRV3StreamingRelease, move_continuation_batch
from eval_ettr_joint_model import (
    MODEL_SCHEMA,
    _build_initial_model,
    _load_joint_payload,
)
from eval_ettr_joint_instruction_model import (
    RUN_SCHEMA,
    _load_initialization_contract,
    _validate_model_lineage,
    _validate_run_lineage,
)
from eval_ettr_v3 import _parameter_sha256, _read_hash_bound_json
from native_causal_disposition_reader import answer_token_ids_from_tokenizer
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
)
from train_ettr_component_island import (
    _canonical_bytes,
    _reader_pairs_from_logits,
    _sha256_file,
    _summary,
    _write_no_replace,
)
from train_typed_query_state_reader_pilot import (
    CONTRACT_SCHEMA as COMPILER_CONTRACT_SCHEMA,
    _annotate_pair_rows,
    _compiler_counts,
    _states,
)


CONTRACT_SCHEMA = "shohin-ettr-algebraic-joint-state-eval-contract-v2"
REPORT_SCHEMA = "shohin-ettr-algebraic-joint-state-eval-report-v2"
PROJECTION_CONTRACT_SCHEMA = (
    "shohin-ettr-algebraic-joint-state-opcode-projection-contract-v1"
)
PROJECTION_REPORT_SCHEMA = (
    "shohin-ettr-algebraic-joint-state-opcode-projection-report-v1"
)
STATE_CONTRACT_SCHEMA = "shohin-ettr-algebraic-state-semantic-contract-v1"
STATE_REPORT_SCHEMA = "shohin-ettr-algebraic-state-semantic-report-v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_STATE_RUN_FILES = (
    "compiler-final.safetensors",
    "compiler-initial.safetensors",
    "pilot-contract.json",
    "reactor-final.safetensors",
    "reactor-initial.safetensors",
    "report.json",
    "train.jsonl",
)
_SCHEDULE_RUN_FILES = (
    "pilot-contract.json",
    "report.json",
    "schedule-final.safetensors",
    "schedule-initial.safetensors",
    "train.jsonl",
)
_SCHEDULE_OPTIONAL_FILES = ("opcode-program-registry.json",)
_SCHEDULE_SCHEMAS = {
    "shohin-ettr-parallel-addressed-transaction-contract-v1": (
        "shohin-ettr-parallel-addressed-transaction-report-v1"
    ),
    "shohin-ettr-parallel-addressed-transaction-contract-v2": (
        "shohin-ettr-parallel-addressed-transaction-report-v2"
    ),
    "shohin-ettr-parallel-addressed-transaction-contract-v3": (
        "shohin-ettr-parallel-addressed-transaction-report-v3"
    ),
    "shohin-ettr-parallel-addressed-transaction-contract-v4": (
        "shohin-ettr-parallel-addressed-transaction-report-v4"
    ),
    "shohin-ettr-parallel-addressed-transaction-contract-v5": (
        "shohin-ettr-parallel-addressed-transaction-report-v5"
    ),
    "shohin-ettr-parallel-addressed-transaction-contract-v6": (
        "shohin-ettr-parallel-addressed-transaction-report-v6"
    ),
    "shohin-ettr-parallel-addressed-transaction-contract-v7": (
        "shohin-ettr-parallel-addressed-transaction-report-v7"
    ),
    "shohin-ettr-parallel-addressed-transaction-contract-v8": (
        "shohin-ettr-parallel-addressed-transaction-report-v8"
    ),
}
_ARMS = (
    "autonomous_program_autonomous_state",
    "oracle_program_autonomous_state",
    "autonomous_program_oracle_state",
    "oracle_program_oracle_state",
)


class AlgebraicJointStateEvaluationError(RuntimeError):
    """The compiler/state integration violated its sealed contract."""


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
    parser.add_argument("--state-run-dir", type=Path)
    parser.add_argument("--state-run-sha256s-sha256")
    parser.add_argument("--schedule-run-dir", type=Path)
    parser.add_argument("--schedule-run-sha256s-sha256")
    parser.add_argument("--opcode-projection-registry", type=Path)
    parser.add_argument("--opcode-projection-registry-sha256")
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
    )
    hashes = (
        args.release_sha256,
        args.joint_model_sha256,
        args.joint_run_contract_sha256,
        args.compiler_sha256,
        args.compiler_contract_sha256,
    )
    state_run_supplied = args.state_run_dir is not None
    schedule_run_supplied = args.schedule_run_dir is not None
    projection_registry = getattr(args, "opcode_projection_registry", None)
    projection_registry_sha256 = getattr(
        args,
        "opcode_projection_registry_sha256",
        None,
    )
    projection_supplied = projection_registry is not None
    if (
        any(not path.is_absolute() for path in paths)
        or any(_HEX64.fullmatch(value) is None for value in hashes)
        or _HEX40.fullmatch(args.source_commit) is None
        or not 0 <= args.data_seed < 2**63
        or args.max_batches < 2
        or state_run_supplied != (args.state_run_sha256s_sha256 is not None)
        or schedule_run_supplied != (args.schedule_run_sha256s_sha256 is not None)
        or projection_supplied
        != (projection_registry_sha256 is not None)
        or (projection_supplied and not schedule_run_supplied)
        or (
            state_run_supplied
            and (
                not args.state_run_dir.is_absolute()
                or _HEX64.fullmatch(args.state_run_sha256s_sha256) is None
            )
        )
        or (
            schedule_run_supplied
            and (
                not args.schedule_run_dir.is_absolute()
                or _HEX64.fullmatch(args.schedule_run_sha256s_sha256) is None
            )
        )
        or (
            projection_supplied
            and (
                not projection_registry.is_absolute()
                or _HEX64.fullmatch(projection_registry_sha256) is None
            )
        )
        or args.output.exists()
        or args.output.is_symlink()
        or not args.output.parent.is_dir()
    ):
        raise AlgebraicJointStateEvaluationError(
            "algebraic joint-state evaluation arguments differ"
        )


def _load_state_file(path: Path) -> dict[str, torch.Tensor]:
    try:
        return dict(load_file(str(path), device="cpu"))
    except Exception as exc:
        raise AlgebraicJointStateEvaluationError(
            "state-semantic component is unreadable"
        ) from exc


def _require_module_state(module, state: Mapping[str, torch.Tensor]) -> None:
    current = module.state_dict()
    if current.keys() != state.keys() or any(
        not torch.equal(current[name].detach().cpu(), state[name]) for name in current
    ):
        raise AlgebraicJointStateEvaluationError(
            "state-semantic initial component differs"
        )


def _load_state_semantic_components(
    args,
    *,
    model,
    provenance,
) -> dict[str, object] | None:
    if args.state_run_dir is None:
        return None
    sums_path = args.state_run_dir / "SHA256SUMS"
    if _sha256_file(sums_path) != args.state_run_sha256s_sha256:
        raise AlgebraicJointStateEvaluationError("state-semantic run receipt differs")
    expected: dict[str, str] = {}
    try:
        lines = sums_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AlgebraicJointStateEvaluationError(
            "state-semantic run receipt is unreadable"
        ) from exc
    for line in lines:
        fields = line.split("  ")
        if (
            len(fields) != 2
            or _HEX64.fullmatch(fields[0]) is None
            or fields[1] not in _STATE_RUN_FILES
            or fields[1] in expected
        ):
            raise AlgebraicJointStateEvaluationError(
                "state-semantic run receipt differs"
            )
        expected[fields[1]] = fields[0]
    if tuple(sorted(expected)) != tuple(sorted(_STATE_RUN_FILES)):
        raise AlgebraicJointStateEvaluationError("state-semantic run receipt differs")
    for name, digest in expected.items():
        if _sha256_file(args.state_run_dir / name) != digest:
            raise AlgebraicJointStateEvaluationError("state-semantic run file differs")
    contract = _read_hash_bound_json(
        args.state_run_dir / "pilot-contract.json",
        expected_sha256=expected["pilot-contract.json"],
        label="state-semantic contract",
    )
    report = _read_hash_bound_json(
        args.state_run_dir / "report.json",
        expected_sha256=expected["report.json"],
        label="state-semantic report",
    )
    before_sha256 = _parameter_sha256(model)
    if (
        contract.get("schema") != STATE_CONTRACT_SCHEMA
        or report.get("schema") != STATE_REPORT_SCHEMA
        or report.get("status") != "pass"
        or report.get("contract_sha256") != expected["pilot-contract.json"]
        or contract.get("release_file_sha256") != args.release_sha256
        or contract.get("joint_model_sha256") != args.joint_model_sha256
        or contract.get("joint_run_contract_sha256") != args.joint_run_contract_sha256
        or contract.get("compiler_sha256") != args.compiler_sha256
        or contract.get("compiler_contract_sha256") != args.compiler_contract_sha256
        or report.get("protected_checkpoint_sha256") != provenance.checkpoint_sha256
        or report.get("before_parameter_sha256") != before_sha256
        or report.get("initial_compiler_sha256")
        != expected["compiler-initial.safetensors"]
        or report.get("initial_reactor_sha256")
        != expected["reactor-initial.safetensors"]
        or report.get("final_compiler_sha256") != expected["compiler-final.safetensors"]
        or report.get("final_reactor_sha256") != expected["reactor-final.safetensors"]
    ):
        raise AlgebraicJointStateEvaluationError("state-semantic lineage differs")
    _require_module_state(
        model.compiler,
        _load_state_file(args.state_run_dir / "compiler-initial.safetensors"),
    )
    _require_module_state(
        model.reactor,
        _load_state_file(args.state_run_dir / "reactor-initial.safetensors"),
    )
    try:
        compiler_result = model.compiler.load_state_dict(
            _load_state_file(args.state_run_dir / "compiler-final.safetensors"),
            strict=True,
        )
        reactor_result = model.reactor.load_state_dict(
            _load_state_file(args.state_run_dir / "reactor-final.safetensors"),
            strict=True,
        )
    except RuntimeError as exc:
        raise AlgebraicJointStateEvaluationError(
            "state-semantic final component differs"
        ) from exc
    if (
        compiler_result.missing_keys
        or compiler_result.unexpected_keys
        or reactor_result.missing_keys
        or reactor_result.unexpected_keys
        or _parameter_sha256(model) != report.get("after_parameter_sha256")
    ):
        raise AlgebraicJointStateEvaluationError(
            "state-semantic final component differs"
        )
    model.eval()
    return {
        "contract_sha256": expected["pilot-contract.json"],
        "report_sha256": expected["report.json"],
        "run_sha256s_sha256": args.state_run_sha256s_sha256,
        "source_commit": contract["source_commit"],
        "updates_completed": report["updates_completed"],
    }


def _load_parallel_schedule(
    args,
    *,
    model,
    provenance,
    stream,
    replacement_system_parameters: int,
) -> tuple[dict[str, object] | None, int]:
    if args.schedule_run_dir is None:
        return None, replacement_system_parameters
    sums_path = args.schedule_run_dir / "SHA256SUMS"
    if _sha256_file(sums_path) != args.schedule_run_sha256s_sha256:
        raise AlgebraicJointStateEvaluationError(
            "parallel schedule run receipt differs"
        )
    expected: dict[str, str] = {}
    try:
        lines = sums_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AlgebraicJointStateEvaluationError(
            "parallel schedule run receipt is unreadable"
        ) from exc
    for line in lines:
        fields = line.split("  ")
        if (
            len(fields) != 2
            or _HEX64.fullmatch(fields[0]) is None
            or fields[1] not in _SCHEDULE_RUN_FILES + _SCHEDULE_OPTIONAL_FILES
            or fields[1] in expected
        ):
            raise AlgebraicJointStateEvaluationError(
                "parallel schedule run receipt differs"
            )
        expected[fields[1]] = fields[0]
    if tuple(sorted(expected)) not in {
        tuple(sorted(_SCHEDULE_RUN_FILES)),
        tuple(sorted(_SCHEDULE_RUN_FILES + _SCHEDULE_OPTIONAL_FILES)),
    }:
        raise AlgebraicJointStateEvaluationError(
            "parallel schedule run receipt differs"
        )
    for name, digest in expected.items():
        if _sha256_file(args.schedule_run_dir / name) != digest:
            raise AlgebraicJointStateEvaluationError(
                "parallel schedule run file differs"
            )
    contract = _read_hash_bound_json(
        args.schedule_run_dir / "pilot-contract.json",
        expected_sha256=expected["pilot-contract.json"],
        label="parallel schedule contract",
    )
    report = _read_hash_bound_json(
        args.schedule_run_dir / "report.json",
        expected_sha256=expected["report.json"],
        label="parallel schedule report",
    )
    architecture = contract.get("architecture")
    expected_report_schema = _SCHEDULE_SCHEMAS.get(contract.get("schema"))
    if (
        expected_report_schema is None
        or report.get("schema") != expected_report_schema
        or report.get("status") != "pass"
        or not isinstance(architecture, Mapping)
        or report.get("contract_sha256") != expected["pilot-contract.json"]
        or contract.get("release_file_sha256") != args.release_sha256
        or contract.get("joint_model_sha256") != args.joint_model_sha256
        or contract.get("joint_run_contract_sha256") != args.joint_run_contract_sha256
        or contract.get("compiler_sha256") != args.compiler_sha256
        or contract.get("compiler_contract_sha256") != args.compiler_contract_sha256
        or report.get("protected_checkpoint_sha256") != provenance.checkpoint_sha256
        or report.get("initial_schedule_sha256")
        != expected["schedule-initial.safetensors"]
        or report.get("final_schedule_sha256") != expected["schedule-final.safetensors"]
    ):
        raise AlgebraicJointStateEvaluationError("parallel schedule lineage differs")
    try:
        seed = int(architecture["seed"])
        width = int(architecture["width"])
        layers = int(architecture["layers"])
        num_heads = int(architecture["num_heads"])
        grounded_pointers = architecture.get("grounded_pointers", False)
        valid_pointer_masks = architecture.get("valid_pointer_masks", False)
        token_native_command_mask = architecture.get(
            "token_native_command_mask",
            False,
        )
        cover_verified_command_mask = architecture.get(
            "cover_verified_command_mask",
            False,
        )
        token_native_occurrence_command = architecture.get(
            "token_native_occurrence_command",
            False,
        )
        token_native_syntax_graph_command = architecture.get(
            "token_native_syntax_graph_command",
            False,
        )
        token_native_declaration_binding_command = architecture.get(
            "token_native_declaration_binding_command",
            False,
        )
        opcode_program_registry_sha256 = architecture.get(
            "opcode_program_registry_sha256"
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AlgebraicJointStateEvaluationError(
            "parallel schedule architecture differs"
        ) from exc
    if (
        not isinstance(grounded_pointers, bool)
        or not isinstance(valid_pointer_masks, bool)
        or (valid_pointer_masks and not grounded_pointers)
        or not isinstance(token_native_command_mask, bool)
        or not isinstance(cover_verified_command_mask, bool)
        or not isinstance(token_native_occurrence_command, bool)
        or not isinstance(token_native_syntax_graph_command, bool)
        or not isinstance(token_native_declaration_binding_command, bool)
        or (token_native_occurrence_command and not token_native_command_mask)
        or (token_native_syntax_graph_command and not token_native_command_mask)
        or (cover_verified_command_mask and not token_native_command_mask)
        or (token_native_occurrence_command and token_native_syntax_graph_command)
        or (
            token_native_declaration_binding_command
            and not token_native_syntax_graph_command
        )
        or (
            token_native_command_mask
            and architecture.get("token_native_codebook_sha256")
            != stream.codec.codebook_sha256
        )
        or (
            (opcode_program_registry_sha256 is None)
            != ("opcode-program-registry.json" not in expected)
        )
    ):
        raise AlgebraicJointStateEvaluationError(
            "parallel schedule architecture differs"
        )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    opcode_registry = (
        load_opcode_program_registry(
            (args.schedule_run_dir / "opcode-program-registry.json").resolve(),
            expected_sha256=opcode_program_registry_sha256,
            max_steps=model.config.max_steps,
            opcode_classes=TRANSACTION_COUNT,
        )
        if opcode_program_registry_sha256 is not None
        else None
    )
    if opcode_registry is not None and (
        architecture.get("opcode_program_classes") != opcode_registry.classes
        or architecture.get("opcode_program_registry_payload_sha256")
        != opcode_registry.payload_sha256
        or architecture.get("opcode_program_development_instance_coverage")
        != opcode_registry.development_instance_coverage
    ):
        raise AlgebraicJointStateEvaluationError(
            "parallel schedule opcode registry differs"
        )
    schedule = ParallelAddressedTransactionCompiler(
        model.config,
        width=width,
        layers=layers,
        num_heads=num_heads,
        grounded_pointers=grounded_pointers,
        valid_pointer_masks=valid_pointer_masks,
        token_native_command_mask=token_native_command_mask,
        cover_verified_command_mask=cover_verified_command_mask,
        token_native_occurrence_command=token_native_occurrence_command,
        token_native_syntax_graph_command=(token_native_syntax_graph_command),
        token_native_declaration_binding_command=(
            token_native_declaration_binding_command
        ),
        token_native_codebook_ids=(
            stream.codec.codebook.token_ids if token_native_command_mask else None
        ),
        token_native_codebook_atoms=(
            stream.codec.codebook.atoms if cover_verified_command_mask else None
        ),
        token_native_vocab_size=(
            model.base.cfg.vocab_size if token_native_command_mask else None
        ),
        opcode_program_sequences=(
            opcode_registry.programs if opcode_registry is not None else None
        ),
    ).to(
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
    )
    _require_module_state(
        schedule,
        _load_state_file(args.schedule_run_dir / "schedule-initial.safetensors"),
    )
    try:
        incompatibility = schedule.load_state_dict(
            _load_state_file(args.schedule_run_dir / "schedule-final.safetensors"),
            strict=True,
        )
    except RuntimeError as exc:
        raise AlgebraicJointStateEvaluationError(
            "parallel schedule final component differs"
        ) from exc
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise AlgebraicJointStateEvaluationError(
            "parallel schedule final component differs"
        )
    schedule_parameters = sum(parameter.numel() for parameter in schedule.parameters())
    removed_reactor_parameters = sum(
        parameter.numel() for parameter in model.reactor.parameters()
    )
    complete_parameters = (
        replacement_system_parameters - removed_reactor_parameters + schedule_parameters
    )
    legacy_reported_parameters = replacement_system_parameters + schedule_parameters
    reported_parameters = contract.get("complete_system_parameters")
    if (
        contract.get("schedule_parameters") != schedule_parameters
        or complete_parameters > SYSTEM_PARAMETER_CAP
        or (
            contract.get("schema")
            == "shohin-ettr-parallel-addressed-transaction-contract-v2"
            and reported_parameters != complete_parameters
        )
        or (
            contract.get("schema")
            == "shohin-ettr-parallel-addressed-transaction-contract-v1"
            and reported_parameters != legacy_reported_parameters
        )
    ):
        raise AlgebraicJointStateEvaluationError(
            "parallel schedule parameter receipt differs"
        )
    model.reactor = ParallelScheduledReactor(schedule, model.config)
    schedule.eval()
    return (
        {
            "complete_system_parameters": complete_parameters,
            "contract_sha256": expected["pilot-contract.json"],
            "legacy_reported_parameters": reported_parameters,
            "report_sha256": expected["report.json"],
            "run_sha256s_sha256": args.schedule_run_sha256s_sha256,
            "schedule_parameters": schedule_parameters,
            "source_commit": contract["source_commit"],
            "updates_completed": report["updates_completed"],
        },
        complete_parameters,
    )


def _strict_load_joint_model(args, *, device: torch.device):
    run_contract = _read_hash_bound_json(
        args.joint_run_contract,
        expected_sha256=args.joint_run_contract_sha256,
        label="joint run contract",
    )
    composition_hint = run_contract.get("component_composition")
    if (
        run_contract.get("schema") != RUN_SCHEMA
        or run_contract.get("ettr_release_sha256") != args.release_sha256
        or not isinstance(composition_hint, Mapping)
    ):
        raise AlgebraicJointStateEvaluationError("joint run contract differs")
    try:
        parent_contract_path = Path(composition_hint["parent_run_contract"])
        parent_contract_sha256 = str(composition_hint["parent_run_contract_sha256"])
        parent_model_path = Path(composition_hint["parent_joint_model"])
        parent_model_sha256 = str(composition_hint["parent_joint_model_sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AlgebraicJointStateEvaluationError(
            "joint composition parent receipt differs"
        ) from exc
    parent_contract = _read_hash_bound_json(
        parent_contract_path,
        expected_sha256=parent_contract_sha256,
        label="composition parent run contract",
    )
    composition = _validate_run_lineage(
        parent_contract,
        run_contract,
        release_sha256=args.release_sha256,
        parent_run_contract_sha256=parent_contract_sha256,
        parent_joint_model_sha256=parent_model_sha256,
    )
    if composition is None:
        raise AlgebraicJointStateEvaluationError("joint composition receipt is missing")
    parent_payload = _load_joint_payload(
        parent_model_path,
        expected_sha256=parent_model_sha256,
    )
    payload = _load_joint_payload(
        args.joint_model,
        expected_sha256=args.joint_model_sha256,
    )
    if (
        payload.get("schema") != MODEL_SCHEMA
        or payload.get("run_contract_sha256") != args.joint_run_contract_sha256
        or payload.get("source_commit") != run_contract.get("source_commit")
        or payload.get("ettr_config") != run_contract.get("model_config")
    ):
        raise AlgebraicJointStateEvaluationError("joint model lineage differs")
    _parent_config, candidate_config = _validate_model_lineage(
        parent_payload,
        payload,
        parent_run_contract_sha256=parent_contract_sha256,
        run_contract_sha256=args.joint_run_contract_sha256,
        parent_contract=parent_contract,
        run_contract=run_contract,
        composition=composition,
    )
    initialization_contract = _load_initialization_contract(
        parent_contract,
        composition=composition,
    )
    model, provenance = _build_initial_model(
        args.protected_checkpoint,
        run_contract=initialization_contract,
        device=device,
    )
    model.set_open_state_read_floor(
        float(candidate_config.get("open_state_read_floor", 0.0))
    )
    model.set_execution_trace_read_scale(
        float(candidate_config.get("execution_trace_read_scale", 0.0))
    )
    model.set_valid_pointer_masks(
        bool(candidate_config.get("valid_pointer_masks", False))
    )
    model.set_query_readout_geometry(
        str(payload.get("query_readout_geometry", "stage"))
    )
    protected_provenance = load_protected_base_model(args.protected_checkpoint)[1]
    parameter_receipt = asdict(model.parameter_receipt())
    if (
        provenance.checkpoint_sha256 != protected_provenance.checkpoint_sha256
        or payload.get("base_config") != protected_provenance.base_config
        or parent_contract.get("parameter_receipt") != parameter_receipt
        or run_contract.get("parameter_receipt") != parameter_receipt
    ):
        raise AlgebraicJointStateEvaluationError(
            "joint protected-model receipt differs"
        )
    try:
        incompatibility = model.load_state_dict(payload["model"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise AlgebraicJointStateEvaluationError(
            "joint model strict load differs"
        ) from exc
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise AlgebraicJointStateEvaluationError("joint model strict load differs")
    if not torch.cuda.is_bf16_supported():
        model.to(dtype=torch.float32)
    model.eval()
    return model, payload, provenance, run_contract


def _load_compiler(
    args,
    *,
    model,
    stream: ETTRV3StreamingRelease,
    device: torch.device,
):
    compiler_contract = _read_hash_bound_json(
        args.compiler_contract,
        expected_sha256=args.compiler_contract_sha256,
        label="compiler contract",
    )
    architecture = compiler_contract.get("architecture")
    if (
        compiler_contract.get("schema") != COMPILER_CONTRACT_SCHEMA
        or compiler_contract.get("executor_mode") != "algebraic"
        or compiler_contract.get("release_file_sha256") != args.release_sha256
        or not isinstance(architecture, Mapping)
        or architecture.get("executor_mode") != "algebraic"
    ):
        raise AlgebraicJointStateEvaluationError("compiler contract differs")
    answer_token_ids = answer_token_ids_from_tokenizer(args.tokenizer)
    try:
        reader = AlgebraicQueryStateReader(
            model.config,
            source_vocab_size=stream.tokenizer.get_vocab_size(),
            target_vocab_size=model.base.cfg.vocab_size,
            answer_token_ids=answer_token_ids,
            width=int(architecture["width"]),
            query_layers=int(architecture["query_layers"]),
            num_heads=int(architecture["num_heads"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AlgebraicJointStateEvaluationError(
            "compiler architecture differs"
        ) from exc
    if _sha256_file(args.compiler) != args.compiler_sha256:
        raise AlgebraicJointStateEvaluationError("compiler hash differs")
    try:
        state = dict(load_file(str(args.compiler), device="cpu"))
    except Exception as exc:
        raise AlgebraicJointStateEvaluationError(
            "compiler checkpoint is unreadable"
        ) from exc
    source_answer_ids = state.pop("answer_token_ids", None)
    if not isinstance(source_answer_ids, torch.Tensor) or source_answer_ids.shape != (
        4,
    ):
        raise AlgebraicJointStateEvaluationError(
            "compiler answer-token receipt differs"
        )
    try:
        incompatibility = reader.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        raise AlgebraicJointStateEvaluationError(
            "compiler strict transplant differs"
        ) from exc
    if incompatibility.missing_keys != ["answer_token_ids"] or (
        incompatibility.unexpected_keys
    ):
        raise AlgebraicJointStateEvaluationError("compiler strict transplant differs")
    reader_parameters = sum(parameter.numel() for parameter in reader.parameters())
    old_reader_parameters = sum(
        parameter.numel() for parameter in model.query_reader.parameters()
    )
    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    replacement_system_parameters = (
        model_parameters - old_reader_parameters + reader_parameters
    )
    if replacement_system_parameters > SYSTEM_PARAMETER_CAP:
        raise AlgebraicJointStateEvaluationError(
            "algebraic joint-state system exceeds parameter cap"
        )
    reader.to(device=device, dtype=torch.float32)
    reader.eval()
    return (
        reader,
        compiler_contract,
        reader_parameters,
        replacement_system_parameters,
    )


def _reader_forward(
    reader,
    batch,
    specs,
    initial_state,
    terminal_state,
    *,
    oracle_program: bool,
):
    return reader(
        batch.episodes.query.tokens,
        batch.episodes.query.attention_mask.bool(),
        batch.episodes.query_read_index,
        initial_state,
        terminal_state,
        teacher_program=specs if oracle_program else None,
    )


def _evaluate(
    reader,
    model,
    *,
    stream: ETTRV3StreamingRelease,
    packet_index: ETTRDiskPacketSufficiencyIndex,
    device: torch.device,
    data_seed: int,
    max_batches: int,
    collect_state_diagnostics: bool = False,
) -> dict[str, object]:
    rows = {arm: {"world": [], "command": []} for arm in _ARMS}
    factual = {arm: 0 for arm in _ARMS}
    compiler = {
        "argument_correct": 0,
        "argument_total": 0,
        "exact_program": 0,
        "operation_correct": 0,
        "rows": 0,
    }
    diagnostic_schedule: dict[str, list[int]] = {}
    diagnostic_oracle_terminal: dict[str, list[int]] = {}
    diagnostic_autonomous_terminal: dict[str, list[int]] = {}
    objective_config = ETTRObjectiveConfig(vocab_size=model.base.cfg.vocab_size)
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
        batch = move_continuation_batch(cpu_batch, device)
        specs = cpu_specs.to(device)
        batch.validate(model.config, objective_config)
        oracle_initial, oracle_terminal = _states(
            batch,
            model.config,
            dtype=torch.float32,
        )
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if torch.cuda.is_bf16_supported()
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            autonomous_initial = model.compile_world(
                batch.episodes.world.tokens,
                attention_mask=batch.episodes.world.attention_mask,
                hard=True,
            )
            autonomous_terminal, _trace = model.execute(
                autonomous_initial.detached_clone(),
                steps=batch.transaction_targets.opcode.shape[1],
                hard=True,
                command_idx=batch.episodes.command.tokens,
                command_attention_mask=(batch.episodes.command.attention_mask),
            )
            if collect_state_diagnostics:
                oracle_runtime_initial = packet_targets_to_state(
                    batch.packet_targets,
                    model.config,
                    step=0,
                    dtype=next(model.reactor.parameters()).dtype,
                )
                oracle_projected_terminal, oracle_trace = model.execute(
                    oracle_runtime_initial,
                    steps=batch.transaction_targets.opcode.shape[1],
                    hard=True,
                    command_idx=batch.episodes.command.tokens,
                    command_attention_mask=(
                        batch.episodes.command.attention_mask
                    ),
                )
            outputs = {
                "autonomous_program_autonomous_state": _reader_forward(
                    reader,
                    batch,
                    specs,
                    autonomous_initial,
                    autonomous_terminal,
                    oracle_program=False,
                ),
                "oracle_program_autonomous_state": _reader_forward(
                    reader,
                    batch,
                    specs,
                    autonomous_initial,
                    autonomous_terminal,
                    oracle_program=True,
                ),
                "autonomous_program_oracle_state": _reader_forward(
                    reader,
                    batch,
                    specs,
                    oracle_initial,
                    oracle_terminal,
                    oracle_program=False,
                ),
                "oracle_program_oracle_state": _reader_forward(
                    reader,
                    batch,
                    specs,
                    oracle_initial,
                    oracle_terminal,
                    oracle_program=True,
                ),
            }
        counts = _compiler_counts(
            outputs["autonomous_program_oracle_state"],
            specs,
        )
        for name, value in counts.items():
            compiler[name] += value
        targets = batch.episodes.query.targets.gather(
            1,
            batch.episodes.query_read_index[:, None],
        ).squeeze(1)
        (
            _world_packet,
            _world_command,
            world_target,
            _command_packet,
            _command_command,
            command_target,
        ) = batch.causal_rectangles.intervention_indices()
        depths = batch.transaction_targets.step_mask.sum(-1)
        if collect_state_diagnostics:
            masks = policy_masks(batch.transaction_targets)
            joint = torch.ones_like(batch.transaction_targets.step_mask)
            for name in (
                "opcode",
                "source",
                "target",
                "relation",
                "type_index",
                "value_code",
            ):
                mask = masks[name]
                predicted = getattr(oracle_trace, f"applied_{name}").argmax(-1)
                correct = predicted.eq(getattr(batch.transaction_targets, name))
                values = diagnostic_schedule.setdefault(name, [0, 0])
                values[0] += int((correct & mask).sum().detach().cpu())
                values[1] += int(mask.sum().detach().cpu())
                joint &= correct | ~mask
            values = diagnostic_schedule.setdefault("joint", [0, 0])
            values[0] += int(
                (joint & batch.transaction_targets.step_mask).sum().detach().cpu()
            )
            values[1] += int(batch.transaction_targets.step_mask.sum().detach().cpu())
            for destination, state in (
                (diagnostic_oracle_terminal, oracle_projected_terminal),
                (diagnostic_autonomous_terminal, autonomous_terminal),
            ):
                for name, counts in _packet_batch_counts(
                    state,
                    batch.terminal_packet_targets,
                ).items():
                    values = destination.setdefault(name, [0, 0])
                    values[0] += counts[0]
                    values[1] += counts[1]
        for arm, output in outputs.items():
            logits = output.vocab_logits
            factual[arm] += int(logits.argmax(-1).eq(targets).sum())
            pairs = _reader_pairs_from_logits(logits, batch)
            for factor, pair in pairs.items():
                target_index = world_target if factor == "world" else command_target
                rows[arm][factor].extend(
                    _annotate_pair_rows(
                        pair,
                        depths.index_select(0, target_index),
                    )
                )
    expected = max_batches * 16
    if compiler["rows"] != expected:
        raise AlgebraicJointStateEvaluationError(
            "algebraic joint-state evaluation support differs"
        )
    result = {
        "arms": {
            arm: {
                "factual_top1": factual[arm] / expected,
                "source_deleted_causal": {
                    factor: _summary(values) for factor, values in rows[arm].items()
                },
            }
            for arm in _ARMS
        },
        "batches": max_batches,
        "compiler": {
            "argument_accuracy": (
                compiler["argument_correct"] / compiler["argument_total"]
            ),
            "exact_program_accuracy": compiler["exact_program"] / expected,
            "operation_accuracy": compiler["operation_correct"] / expected,
            "rows": expected,
        },
    }
    if collect_state_diagnostics:
        def summarize(values):
            return {
                name: {
                    "correct": counts[0],
                    "total": counts[1],
                    "rate": counts[0] / counts[1] if counts[1] else 0.0,
                }
                for name, counts in sorted(values.items())
            }

        result["opcode_projection_diagnostic"] = {
            "oracle_initial_hard_schedule": summarize(diagnostic_schedule),
            "oracle_initial_terminal_packet": summarize(
                diagnostic_oracle_terminal
            ),
            "autonomous_initial_terminal_packet": summarize(
                diagnostic_autonomous_terminal
            ),
        }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    if not torch.cuda.is_available():
        raise AlgebraicJointStateEvaluationError(
            "algebraic joint-state evaluation requires CUDA"
        )
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    if (
        args.required_device_class == "h100"
        and "H100" not in torch.cuda.get_device_name(device).upper()
    ):
        raise AlgebraicJointStateEvaluationError(
            "algebraic joint-state evaluation requires an H100"
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
    if provenance.checkpoint_sha256 != stream.manifest.protected_checkpoint_sha256:
        raise AlgebraicJointStateEvaluationError(
            "release protected-checkpoint receipt differs"
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
    state_semantic_receipt = _load_state_semantic_components(
        args,
        model=model,
        provenance=provenance,
    )
    schedule_receipt, replacement_system_parameters = _load_parallel_schedule(
        args,
        model=model,
        provenance=provenance,
        stream=stream,
        replacement_system_parameters=replacement_system_parameters,
    )
    projection_receipt = None
    if args.opcode_projection_registry is not None:
        if schedule_receipt is None or not isinstance(
            model.reactor,
            ParallelScheduledReactor,
        ):
            raise AlgebraicJointStateEvaluationError(
                "opcode projection requires a parallel schedule"
            )
        base_schedule = model.reactor.compiler
        if getattr(base_schedule, "opcode_program_table", None) is not None:
            raise AlgebraicJointStateEvaluationError(
                "opcode projection requires an unprojected schedule"
            )
        registry = load_opcode_program_registry(
            args.opcode_projection_registry,
            expected_sha256=args.opcode_projection_registry_sha256,
            max_steps=model.config.max_steps,
            opcode_classes=TRANSACTION_COUNT,
        )
        model.reactor = ParallelScheduledReactor(
            RegistryProjectedAddressedScheduleCompiler(
                base_schedule,
                registry.programs,
            ),
            model.config,
        )
        projection_receipt = {
            "classes": registry.classes,
            "development_instance_coverage": (
                registry.development_instance_coverage
            ),
            "parameter_delta": 0,
            "payload_sha256": registry.payload_sha256,
            "registry_sha256": registry.file_sha256,
            "score": "mean-per-step-log-probability-without-class-prior",
            "selection": "single-global-hard-valid-opcode-program",
        }
    schema = (
        PROJECTION_CONTRACT_SCHEMA
        if projection_receipt is not None
        else CONTRACT_SCHEMA
    )
    contract = {
        "compiler_contract_sha256": args.compiler_contract_sha256,
        "compiler_sha256": args.compiler_sha256,
        "data_seed": args.data_seed,
        "fully_autonomous_arm": "autonomous_program_autonomous_state",
        "joint_model_sha256": args.joint_model_sha256,
        "joint_run_contract_sha256": args.joint_run_contract_sha256,
        "max_batches": args.max_batches,
        "non_promotable_diagnostic_arms": list(_ARMS[1:]),
        "opcode_program_projection": projection_receipt,
        "protected_checkpoint_sha256": provenance.checkpoint_sha256,
        "reader_parameters": reader_parameters,
        "release_file_sha256": args.release_sha256,
        "replacement_system_parameters": replacement_system_parameters,
        "required_device_class": args.required_device_class,
        "runtime_precision": str(next(model.parameters()).dtype),
        "schedule_receipt": schedule_receipt,
        "schema": schema,
        "source_commit": args.source_commit,
        "state_semantic_receipt": state_semantic_receipt,
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
            collect_state_diagnostics=projection_receipt is not None,
        )
        report = {
            "compiler_contract_source_commit": compiler_contract["source_commit"],
            "contract_sha256": contract_sha256,
            "device": torch.cuda.get_device_name(device),
            "evaluation": evaluation,
            "joint_model_optimizer_step": joint_payload["optimizer_step"],
            "joint_model_parameter_sha256": _parameter_sha256(model),
            "joint_training_source_commit": joint_contract["source_commit"],
            "opcode_program_projection": projection_receipt,
            "reader_parameters": reader_parameters,
            "replacement_system_parameters": replacement_system_parameters,
            "runtime_precision": str(next(model.parameters()).dtype),
            "schedule_receipt": schedule_receipt,
            "schema": (
                PROJECTION_REPORT_SCHEMA
                if projection_receipt is not None
                else REPORT_SCHEMA
            ),
            "source_verification": source_verification,
            "state_semantic_receipt": state_semantic_receipt,
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
                f"{_sha256_file(args.output / name)}  {name}\n" for name in files
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
