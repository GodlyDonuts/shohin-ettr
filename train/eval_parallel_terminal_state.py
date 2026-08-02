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

from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_query_supervision import iter_batches_with_query_specs
from ettr_v3_streaming import ETTRV3StreamingRelease, move_continuation_batch
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
from operation_state_transition_compiler import (
    FactorizedOperationStateTransitionCompiler,
    OperationEffectSetCompiler,
    OperationStateTransitionCompiler,
)
from operation_effect_set_diagnostics import (
    effect_set_batch_counts,
    merge_effect_diagnostics,
    summarize_effect_diagnostics,
)
from operation_state_supervision import (
    index_atomic_edits,
    index_typed_state,
    oracle_operation_boundary_states,
)
from probe_ettr_oracle_interfaces import packet_targets_to_state
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
    DECLARATION_BOUND_ATOMIC_CONTRACT_SCHEMA as PILOT_DECLARATION_CONTRACT_SCHEMA,
    DECLARATION_BOUND_ATOMIC_REPORT_SCHEMA as PILOT_DECLARATION_REPORT_SCHEMA,
    FACTORIZED_OPERATION_STATE_CONTRACT_SCHEMA as PILOT_FACTORIZED_OPERATION_STATE_CONTRACT_SCHEMA,
    FACTORIZED_OPERATION_STATE_REPORT_SCHEMA as PILOT_FACTORIZED_OPERATION_STATE_REPORT_SCHEMA,
    LEGACY_CONTRACT_SCHEMA as PILOT_LEGACY_CONTRACT_SCHEMA,
    LEGACY_REPORT_SCHEMA as PILOT_LEGACY_REPORT_SCHEMA,
    LEXICAL_ATOMIC_CONTRACT_SCHEMA as PILOT_LEXICAL_CONTRACT_SCHEMA,
    LEXICAL_ATOMIC_REPORT_SCHEMA as PILOT_LEXICAL_REPORT_SCHEMA,
    OCCURRENCE_LINKED_ATOMIC_CONTRACT_SCHEMA as PILOT_OCCURRENCE_CONTRACT_SCHEMA,
    OCCURRENCE_LINKED_ATOMIC_REPORT_SCHEMA as PILOT_OCCURRENCE_REPORT_SCHEMA,
    OPERATION_RECURRENT_ATOMIC_CONTRACT_SCHEMA as PILOT_OPERATION_CONTRACT_SCHEMA,
    OPERATION_RECURRENT_ATOMIC_REPORT_SCHEMA as PILOT_OPERATION_REPORT_SCHEMA,
    OPERATION_EFFECT_SET_CONTRACT_SCHEMA as PILOT_OPERATION_EFFECT_SET_CONTRACT_SCHEMA,
    OPERATION_EFFECT_SET_REPORT_SCHEMA as PILOT_OPERATION_EFFECT_SET_REPORT_SCHEMA,
    OPERATION_STATE_ATOMIC_CONTRACT_SCHEMA as PILOT_OPERATION_STATE_CONTRACT_SCHEMA,
    OPERATION_STATE_ATOMIC_REPORT_SCHEMA as PILOT_OPERATION_STATE_REPORT_SCHEMA,
    REPORT_SCHEMA as PILOT_REPORT_SCHEMA,
    SYNTAX_ROUTED_ATOMIC_CONTRACT_SCHEMA as PILOT_SYNTAX_CONTRACT_SCHEMA,
    SYNTAX_ROUTED_ATOMIC_REPORT_SCHEMA as PILOT_SYNTAX_REPORT_SCHEMA,
    derive_atomic_edit_targets,
)
from train_parallel_addressed_transaction_pilot import _training_initial_state


CONTRACT_SCHEMA = "shohin-ettr-parallel-terminal-state-eval-contract-v2"
REPORT_SCHEMA = "shohin-ettr-parallel-terminal-state-eval-report-v2"
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


def _load_module_state(
    module: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
) -> None:
    """Strictly replace one measured component without changing its geometry."""

    try:
        incompatibility = module.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ParallelTerminalStateEvaluationError(
            "terminal-state measured component differs"
        ) from exc
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ParallelTerminalStateEvaluationError(
            "terminal-state measured component differs"
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
    stream,
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
    schema_pairs = {
        PILOT_LEGACY_CONTRACT_SCHEMA: PILOT_LEGACY_REPORT_SCHEMA,
        PILOT_CAUSAL_DELTA_CONTRACT_SCHEMA: PILOT_CAUSAL_DELTA_REPORT_SCHEMA,
        PILOT_CONTRACT_SCHEMA: PILOT_REPORT_SCHEMA,
        PILOT_ATOMIC_CONTRACT_SCHEMA: PILOT_ATOMIC_REPORT_SCHEMA,
        PILOT_LEXICAL_CONTRACT_SCHEMA: PILOT_LEXICAL_REPORT_SCHEMA,
        PILOT_SYNTAX_CONTRACT_SCHEMA: PILOT_SYNTAX_REPORT_SCHEMA,
        PILOT_OCCURRENCE_CONTRACT_SCHEMA: PILOT_OCCURRENCE_REPORT_SCHEMA,
        PILOT_DECLARATION_CONTRACT_SCHEMA: PILOT_DECLARATION_REPORT_SCHEMA,
        PILOT_OPERATION_CONTRACT_SCHEMA: PILOT_OPERATION_REPORT_SCHEMA,
        PILOT_OPERATION_STATE_CONTRACT_SCHEMA: PILOT_OPERATION_STATE_REPORT_SCHEMA,
        PILOT_FACTORIZED_OPERATION_STATE_CONTRACT_SCHEMA: (
            PILOT_FACTORIZED_OPERATION_STATE_REPORT_SCHEMA
        ),
        PILOT_OPERATION_EFFECT_SET_CONTRACT_SCHEMA: (
            PILOT_OPERATION_EFFECT_SET_REPORT_SCHEMA
        ),
    }
    run_schema = contract.get("schema")
    if (
        run_schema not in schema_pairs
        or report.get("schema") != schema_pairs[run_schema]
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
        PILOT_LEXICAL_CONTRACT_SCHEMA,
        PILOT_SYNTAX_CONTRACT_SCHEMA,
        PILOT_OCCURRENCE_CONTRACT_SCHEMA,
        PILOT_DECLARATION_CONTRACT_SCHEMA,
        PILOT_OPERATION_CONTRACT_SCHEMA,
        PILOT_OPERATION_STATE_CONTRACT_SCHEMA,
        PILOT_FACTORIZED_OPERATION_STATE_CONTRACT_SCHEMA,
        PILOT_OPERATION_EFFECT_SET_CONTRACT_SCHEMA,
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
    atomic_edits = contract.get("schema") in {
        PILOT_ATOMIC_CONTRACT_SCHEMA,
        PILOT_LEXICAL_CONTRACT_SCHEMA,
        PILOT_SYNTAX_CONTRACT_SCHEMA,
        PILOT_OCCURRENCE_CONTRACT_SCHEMA,
        PILOT_DECLARATION_CONTRACT_SCHEMA,
        PILOT_OPERATION_CONTRACT_SCHEMA,
        PILOT_OPERATION_STATE_CONTRACT_SCHEMA,
        PILOT_FACTORIZED_OPERATION_STATE_CONTRACT_SCHEMA,
        PILOT_OPERATION_EFFECT_SET_CONTRACT_SCHEMA,
    }
    lexical_command = contract.get("schema") in {
        PILOT_LEXICAL_CONTRACT_SCHEMA,
        PILOT_SYNTAX_CONTRACT_SCHEMA,
        PILOT_OCCURRENCE_CONTRACT_SCHEMA,
        PILOT_DECLARATION_CONTRACT_SCHEMA,
        PILOT_OPERATION_CONTRACT_SCHEMA,
        PILOT_OPERATION_STATE_CONTRACT_SCHEMA,
        PILOT_FACTORIZED_OPERATION_STATE_CONTRACT_SCHEMA,
        PILOT_OPERATION_EFFECT_SET_CONTRACT_SCHEMA,
    }
    token_native_command_mask = contract.get("schema") in {
        PILOT_SYNTAX_CONTRACT_SCHEMA,
        PILOT_OCCURRENCE_CONTRACT_SCHEMA,
        PILOT_DECLARATION_CONTRACT_SCHEMA,
        PILOT_OPERATION_CONTRACT_SCHEMA,
        PILOT_OPERATION_STATE_CONTRACT_SCHEMA,
        PILOT_FACTORIZED_OPERATION_STATE_CONTRACT_SCHEMA,
        PILOT_OPERATION_EFFECT_SET_CONTRACT_SCHEMA,
    }
    token_native_occurrence_command = (
        contract.get("schema") == PILOT_OCCURRENCE_CONTRACT_SCHEMA
    )
    token_native_syntax_graph_command = (
        contract.get("schema")
        in {
            PILOT_DECLARATION_CONTRACT_SCHEMA,
            PILOT_OPERATION_CONTRACT_SCHEMA,
            PILOT_OPERATION_STATE_CONTRACT_SCHEMA,
            PILOT_FACTORIZED_OPERATION_STATE_CONTRACT_SCHEMA,
            PILOT_OPERATION_EFFECT_SET_CONTRACT_SCHEMA,
        }
    )
    token_native_declaration_binding_command = (
        contract.get("schema")
        in {
            PILOT_DECLARATION_CONTRACT_SCHEMA,
            PILOT_OPERATION_CONTRACT_SCHEMA,
            PILOT_OPERATION_STATE_CONTRACT_SCHEMA,
            PILOT_FACTORIZED_OPERATION_STATE_CONTRACT_SCHEMA,
            PILOT_OPERATION_EFFECT_SET_CONTRACT_SCHEMA,
        }
    )
    cover_verified_command_mask = (
        contract.get("schema")
        in {
            PILOT_DECLARATION_CONTRACT_SCHEMA,
            PILOT_OPERATION_CONTRACT_SCHEMA,
            PILOT_OPERATION_STATE_CONTRACT_SCHEMA,
            PILOT_FACTORIZED_OPERATION_STATE_CONTRACT_SCHEMA,
            PILOT_OPERATION_EFFECT_SET_CONTRACT_SCHEMA,
        }
    )
    token_native_operation_recurrence_command = (
        contract.get("schema")
        in {
            PILOT_OPERATION_CONTRACT_SCHEMA,
            PILOT_OPERATION_STATE_CONTRACT_SCHEMA,
            PILOT_FACTORIZED_OPERATION_STATE_CONTRACT_SCHEMA,
            PILOT_OPERATION_EFFECT_SET_CONTRACT_SCHEMA,
        }
    )
    token_native_operation_state_command = (
        contract.get("schema")
        in {
            PILOT_OPERATION_STATE_CONTRACT_SCHEMA,
            PILOT_FACTORIZED_OPERATION_STATE_CONTRACT_SCHEMA,
            PILOT_OPERATION_EFFECT_SET_CONTRACT_SCHEMA,
        }
    )
    factorized_operation_effect_command = (
        contract.get("schema")
        == PILOT_FACTORIZED_OPERATION_STATE_CONTRACT_SCHEMA
    )
    operation_effect_set_command = (
        contract.get("schema") == PILOT_OPERATION_EFFECT_SET_CONTRACT_SCHEMA
    )
    if residual_edits != (architecture.get("sparse_residual_edits") is True):
        raise ParallelTerminalStateEvaluationError(
            "terminal-state residual edit contract differs"
        )
    if atomic_edits != (architecture.get("atomic_typed_edits") is True):
        raise ParallelTerminalStateEvaluationError(
            "terminal-state atomic edit contract differs"
        )
    expected_flags = {
        "cover_verified_command_mask": cover_verified_command_mask,
        "lexical_command_rail": lexical_command,
        "token_native_command_mask": token_native_command_mask,
        "token_native_declaration_binding_command": (
            token_native_declaration_binding_command
        ),
        "token_native_occurrence_command": token_native_occurrence_command,
        "token_native_operation_recurrence_command": (
            token_native_operation_recurrence_command
        ),
        "token_native_operation_state_command": (
            token_native_operation_state_command
        ),
        "factorized_operation_effect_command": (
            factorized_operation_effect_command
        ),
        "operation_effect_set_command": operation_effect_set_command,
        "token_native_syntax_graph_command": token_native_syntax_graph_command,
    }
    if any(architecture.get(name, False) is not value for name, value in expected_flags.items()):
        raise ParallelTerminalStateEvaluationError(
            "terminal-state public syntax architecture differs"
        )
    if token_native_command_mask and (
        architecture.get("token_native_codebook_sha256")
        != stream.codec.codebook_sha256
    ):
        raise ParallelTerminalStateEvaluationError(
            "terminal-state token-native codebook differs"
        )
    try:
        seed = int(architecture["seed"])
        width = int(architecture["width"])
        layers = int(architecture["layers"])
        num_heads = int(architecture["num_heads"])
        relation_width = int(architecture["relation_width"])
        operation_effect_slots = int(
            architecture.get("operation_effect_slots", 0)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ParallelTerminalStateEvaluationError(
            "terminal-state architecture differs"
        ) from exc
    if operation_effect_set_command:
        objective = contract.get("objective")
        if (
            not 1 <= operation_effect_slots <= 64
            or not isinstance(objective, Mapping)
            or objective.get("unordered_typed_effect_set") is not True
            or objective.get("effect_set_matching")
            != "detached-sinkhorn-typed-bipartite"
        ):
            raise ParallelTerminalStateEvaluationError(
                "operation effect set contract differs"
            )
    elif operation_effect_slots != 0:
        raise ParallelTerminalStateEvaluationError(
            "operation effect set geometry differs"
        )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    compiler_class = (
        OperationEffectSetCompiler
        if operation_effect_set_command
        else FactorizedOperationStateTransitionCompiler
        if factorized_operation_effect_command
        else OperationStateTransitionCompiler
        if token_native_operation_state_command
        else ParallelTerminalStateCompiler
    )
    compiler = compiler_class(
        model.config,
        width=width,
        layers=layers,
        num_heads=num_heads,
        relation_width=relation_width,
        residual_edits=residual_edits,
        atomic_edits=atomic_edits,
        lexical_command=lexical_command,
        token_native_command_mask=token_native_command_mask,
        cover_verified_command_mask=cover_verified_command_mask,
        token_native_occurrence_command=token_native_occurrence_command,
        token_native_syntax_graph_command=token_native_syntax_graph_command,
        token_native_declaration_binding_command=(
            token_native_declaration_binding_command
        ),
        token_native_operation_recurrence_command=(
            token_native_operation_recurrence_command
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
        **(
            {"maximum_effects": operation_effect_slots}
            if operation_effect_set_command
            else {}
        ),
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


def _typed_state_exact_count(
    predicted,
    target,
    *,
    slot_mask: torch.Tensor,
    relation_mask: torch.Tensor,
) -> int:
    """Count rows with an exact typed state on every represented field."""

    target_active = target.active.gt(0.5)
    active_mask = slot_mask & target_active
    exact = (
        (
            predicted.active.gt(0.5).eq(target_active)
            | ~slot_mask
        ).all(-1)
        & (
            predicted.value_probabilities.argmax(-1).eq(
                target.value_probabilities.argmax(-1)
            )
            | ~active_mask
        ).all(-1)
        & (
            predicted.type_probabilities.argmax(-1).eq(
                target.type_probabilities.argmax(-1)
            )
            | ~active_mask
        ).all(-1)
        & (
            predicted.relations.gt(0.5).eq(target.relations.gt(0.5))
            | ~relation_mask
        ).flatten(1).all(-1)
        & predicted.root.gt(0.5).eq(target.root.gt(0.5)).all(-1)
        & predicted.committed.gt(0.5).eq(target.committed.gt(0.5))
        & predicted.halted.gt(0.5).eq(target.halted.gt(0.5))
    )
    return int(exact.sum().detach().cpu())


def _evaluate_operation_effects(
    compiler: OperationEffectSetCompiler,
    oracle_executor,
    model,
    *,
    stream,
    packet_index,
    device: torch.device,
    data_seed: int,
    max_batches: int,
    training_initial_state: str,
) -> dict[str, object]:
    """Measure held-out hard local effects without changing causal gates."""

    compiler.eval()
    oracle_executor.eval()
    aggregate: dict[str, object] = {}
    iterator = iter_batches_with_query_specs(
        stream,
        "development",
        epoch=0,
        seed=data_seed,
    )
    observed = 0
    for _position, cpu_batch, _cpu_specs in iterator:
        if observed >= max_batches:
            break
        packet_index.verify_validation((cpu_batch,))
        batch = move_continuation_batch(cpu_batch, device)
        with torch.inference_mode():
            initial = _training_initial_state(
                model,
                batch,
                source=training_initial_state,
                dtype=next(compiler.parameters()).dtype,
            )
            oracle = oracle_operation_boundary_states(
                oracle_executor,
                initial,
                batch.transaction_targets,
            )
            command_hidden = model._encode_to_stage(
                batch.episodes.command.tokens,
                pos=0,
            )
            command_lexical = model.base.tok(batch.episodes.command.tokens)
            terminal, trace = compiler.forward_with_operation_states(
                initial,
                command_hidden=command_hidden,
                command_lexical=command_lexical,
                command_tokens=batch.episodes.command.tokens,
                command_attention_mask=batch.episodes.command.attention_mask.bool(),
                steps=batch.transaction_targets.opcode.shape[1],
                hard=True,
            )
            previous = initial
            for rank, (predicted_state, edits, target_state) in enumerate(
                zip(
                    trace.operation_states,
                    trace.operation_edits,
                    oracle.states,
                    strict=True,
                )
            ):
                index = torch.nonzero(
                    oracle.mask[:, rank], as_tuple=False
                ).flatten()
                if index.numel() == 0:
                    continue
                predicted_selected = index_typed_state(predicted_state, index)
                target_selected = index_typed_state(target_state, index)
                previous_selected = index_typed_state(previous, index)
                slot_mask = batch.terminal_packet_targets.slot_mask.index_select(
                    0, index
                )
                relation_mask = (
                    batch.terminal_packet_targets.relation_mask.index_select(
                        0, index
                    )
                )
                labels = derive_atomic_edit_targets(
                    previous_selected, target_selected
                )
                values = effect_set_batch_counts(
                    index_atomic_edits(edits, index),
                    labels,
                    slot_mask=slot_mask,
                    relation_mask=relation_mask,
                )
                merge_effect_diagnostics(aggregate, values)
                counts = aggregate.setdefault("counts", {})
                assert isinstance(counts, dict)
                counts["operation_state_instances"] = int(
                    counts.get("operation_state_instances", 0)
                ) + int(index.numel())
                counts["operation_state_exact"] = int(
                    counts.get("operation_state_exact", 0)
                ) + _typed_state_exact_count(
                    predicted_selected,
                    target_selected,
                    slot_mask=slot_mask,
                    relation_mask=relation_mask,
                )
                previous = target_state

            target_terminal = packet_targets_to_state(
                batch.terminal_packet_targets,
                model.config,
                step=batch.transaction_targets.opcode.shape[1],
                dtype=next(compiler.parameters()).dtype,
            )
            final_labels = derive_atomic_edit_targets(
                oracle.last_state, target_terminal
            )
            merge_effect_diagnostics(
                aggregate,
                effect_set_batch_counts(
                    trace.final_edits,
                    final_labels,
                    slot_mask=batch.terminal_packet_targets.slot_mask,
                    relation_mask=batch.terminal_packet_targets.relation_mask,
                ),
            )
            counts = aggregate.setdefault("counts", {})
            assert isinstance(counts, dict)
            batch_size = int(batch.terminal_packet_targets.slot_mask.shape[0])
            counts["terminal_state_instances"] = int(
                counts.get("terminal_state_instances", 0)
            ) + batch_size
            counts["terminal_state_exact"] = int(
                counts.get("terminal_state_exact", 0)
            ) + _typed_state_exact_count(
                terminal,
                target_terminal,
                slot_mask=batch.terminal_packet_targets.slot_mask,
                relation_mask=batch.terminal_packet_targets.relation_mask,
            )
        observed += 1
    if observed != max_batches:
        raise ParallelTerminalStateEvaluationError(
            "operation effect development split is too short"
        )
    summary = summarize_effect_diagnostics(aggregate)
    counts = summary["counts"]
    assert isinstance(counts, dict)
    summary["batches"] = observed
    summary["operation_state_exact_rate"] = (
        int(counts["operation_state_exact"])
        / int(counts["operation_state_instances"])
    )
    summary["terminal_state_exact_rate"] = (
        int(counts["terminal_state_exact"])
        / int(counts["terminal_state_instances"])
    )
    return summary


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
    oracle_executor = model.reactor
    reader, compiler_contract, reader_parameters, replacement_parameters = (
        _load_compiler(args, model=model, stream=stream, device=device)
    )
    receipt, complete_parameters = _load_terminal_compiler(
        args,
        model=model,
        provenance=provenance,
        stream=stream,
        replacement_system_parameters=replacement_parameters,
    )
    contract = {
        "compiler_contract_sha256": args.compiler_contract_sha256,
        "compiler_sha256": args.compiler_sha256,
        "data_seed": args.data_seed,
        "fully_autonomous_arm": "autonomous_program_autonomous_state",
        "joint_model_sha256": args.joint_model_sha256,
        "joint_run_contract_sha256": args.joint_run_contract_sha256,
        "local_operation_effect_diagnostic": isinstance(
            model.reactor.compiler, OperationEffectSetCompiler
        ),
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
        operation_effect_diagnostics = None
        if isinstance(model.reactor.compiler, OperationEffectSetCompiler):
            training_initial_state = receipt.get("training_initial_state")
            if not isinstance(training_initial_state, str):
                raise ParallelTerminalStateEvaluationError(
                    "operation effect initial-state receipt differs"
                )
            measured_compiler = model.reactor.compiler
            initial_state = _load_state_file(
                args.terminal_run_dir / "terminal-compiler-initial.safetensors"
            )
            final_state = _load_state_file(
                args.terminal_run_dir / "terminal-compiler-final.safetensors"
            )
            local_by_phase = {}
            evaluation_by_phase = {}
            for phase, state in (
                ("before", initial_state),
                ("after", final_state),
            ):
                _load_module_state(measured_compiler, state)
                local_by_phase[phase] = _evaluate_operation_effects(
                    measured_compiler,
                    oracle_executor,
                    model,
                    stream=stream,
                    packet_index=packet_index,
                    device=device,
                    data_seed=args.data_seed,
                    max_batches=args.max_batches,
                    training_initial_state=training_initial_state,
                )
                evaluation_by_phase[phase] = _evaluate(
                    reader,
                    model,
                    stream=stream,
                    packet_index=packet_index,
                    device=device,
                    data_seed=args.data_seed,
                    max_batches=args.max_batches,
                )
            _load_module_state(measured_compiler, final_state)
            operation_effect_diagnostics = local_by_phase
            evaluation = evaluation_by_phase
        else:
            evaluation = {
                "after": _evaluate(
                    reader,
                    model,
                    stream=stream,
                    packet_index=packet_index,
                    device=device,
                    data_seed=args.data_seed,
                    max_batches=args.max_batches,
                )
            }
        report = {
            "compiler_contract_source_commit": compiler_contract["source_commit"],
            "contract_sha256": contract_sha256,
            "device": torch.cuda.get_device_name(device),
            "evaluation": evaluation,
            "joint_model_optimizer_step": joint_payload["optimizer_step"],
            "joint_model_parameter_sha256": _parameter_sha256(model),
            "joint_training_source_commit": joint_contract["source_commit"],
            "operation_effect_diagnostics": operation_effect_diagnostics,
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
