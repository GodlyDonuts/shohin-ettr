#!/usr/bin/env python3
"""Fit and gate direct ETTR terminal-state quotient transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Sequence

from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from eval_algebraic_query_joint_state import (
    _evaluate,
    _load_compiler,
    _strict_load_joint_model,
)
from ettr_objectives import ETTRObjectiveConfig
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_query_supervision import iter_batches_with_query_specs
from ettr_v3_streaming import ETTRV3StreamingRelease, move_continuation_batch
from parallel_terminal_state_compiler import (
    AtomicTypedEdits,
    ParallelTerminalStateCompiler,
    ParallelTerminalStateReactor,
)
from operation_state_supervision import (
    index_atomic_edits,
    index_typed_state,
    oracle_operation_boundary_states,
)
from operation_state_transition_compiler import (
    EFFECT_ALLOCATE,
    EFFECT_CLEAR,
    EFFECT_COMMIT,
    EFFECT_HALT,
    EFFECT_KIND_COUNT,
    EFFECT_LINK,
    EFFECT_NOOP,
    EFFECT_REJECT,
    EFFECT_REPLACE,
    EFFECT_ROOT_CLEAR,
    EFFECT_ROOT_SET,
    EFFECT_UNLINK,
    EFFECT_WRITE,
    FactorizedOperationStateTransitionCompiler,
    OperationEffectSetCompiler,
    OperationFamilyGatedWriteLinkCompiler,
    OperationStateBoundFamilyGatedWriteLinkCompiler,
    OperationPostWriteLinkRailCompiler,
    OperationStateTransitionCompiler,
    OperationWriteLinkRailCompiler,
    LINK_RAIL_EFFECT_SLOTS,
    ROLE_ANCHORED_EFFECT_MOTORS_PER_ROLE,
    ROLE_ANCHORED_EFFECT_ROLES,
    ROLE_ANCHORED_EFFECT_SLOTS,
    WRITE_LINK_RAIL_EFFECT_SLOTS,
    WRITE_RAIL_EFFECT_SLOTS,
    OPERATION_EFFECT_FAMILY_COUNT,
    OPERATION_EFFECT_FAMILY_LINK,
    OPERATION_EFFECT_FAMILY_NONE,
    OPERATION_EFFECT_FAMILY_WRITE,
)
from probe_ettr_oracle_interfaces import (
    _packet_batch_counts,
    packet_targets_to_state,
)
from train_ettr_component_island import (
    _canonical_bytes,
    _sha256_file,
    _write_no_replace,
)
from train_parallel_addressed_transaction_pilot import (
    _merge_counts,
    _precision_context,
    _state_brier,
    _summarize_counts,
    _training_initial_state,
)


ATOMIC_CONTRACT_SCHEMA = "shohin-ettr-parallel-terminal-state-contract-v4"
ATOMIC_REPORT_SCHEMA = "shohin-ettr-parallel-terminal-state-report-v4"
LEXICAL_ATOMIC_CONTRACT_SCHEMA = "shohin-ettr-parallel-terminal-state-contract-v5"
LEXICAL_ATOMIC_REPORT_SCHEMA = "shohin-ettr-parallel-terminal-state-report-v5"
SYNTAX_ROUTED_ATOMIC_CONTRACT_SCHEMA = "shohin-ettr-parallel-terminal-state-contract-v6"
SYNTAX_ROUTED_ATOMIC_REPORT_SCHEMA = "shohin-ettr-parallel-terminal-state-report-v6"
OCCURRENCE_LINKED_ATOMIC_CONTRACT_SCHEMA = (
    "shohin-ettr-parallel-terminal-state-contract-v7"
)
OCCURRENCE_LINKED_ATOMIC_REPORT_SCHEMA = "shohin-ettr-parallel-terminal-state-report-v7"
DECLARATION_BOUND_ATOMIC_CONTRACT_SCHEMA = (
    "shohin-ettr-parallel-terminal-state-contract-v8"
)
DECLARATION_BOUND_ATOMIC_REPORT_SCHEMA = "shohin-ettr-parallel-terminal-state-report-v8"
OPERATION_RECURRENT_ATOMIC_CONTRACT_SCHEMA = (
    "shohin-ettr-parallel-terminal-state-contract-v9"
)
OPERATION_RECURRENT_ATOMIC_REPORT_SCHEMA = (
    "shohin-ettr-parallel-terminal-state-report-v9"
)
OPERATION_STATE_ATOMIC_CONTRACT_SCHEMA = (
    "shohin-ettr-parallel-terminal-state-contract-v10"
)
OPERATION_STATE_ATOMIC_REPORT_SCHEMA = "shohin-ettr-parallel-terminal-state-report-v10"
FACTORIZED_OPERATION_STATE_CONTRACT_SCHEMA = (
    "shohin-ettr-parallel-terminal-state-contract-v11"
)
FACTORIZED_OPERATION_STATE_REPORT_SCHEMA = (
    "shohin-ettr-parallel-terminal-state-report-v11"
)
OPERATION_EFFECT_SET_CONTRACT_SCHEMA = (
    "shohin-ettr-parallel-terminal-state-contract-v12"
)
OPERATION_EFFECT_SET_REPORT_SCHEMA = "shohin-ettr-parallel-terminal-state-report-v12"
ROLE_ANCHORED_EFFECT_SET_CONTRACT_SCHEMA = (
    "shohin-ettr-parallel-terminal-state-contract-v13"
)
ROLE_ANCHORED_EFFECT_SET_REPORT_SCHEMA = (
    "shohin-ettr-parallel-terminal-state-report-v13"
)
CARDINALITY_GATED_EFFECT_SET_CONTRACT_SCHEMA = (
    "shohin-ettr-parallel-terminal-state-contract-v14"
)
CARDINALITY_GATED_EFFECT_SET_REPORT_SCHEMA = (
    "shohin-ettr-parallel-terminal-state-report-v14"
)
WRITE_LINK_RAIL_CONTRACT_SCHEMA = "shohin-ettr-parallel-terminal-state-contract-v15"
WRITE_LINK_RAIL_REPORT_SCHEMA = "shohin-ettr-parallel-terminal-state-report-v15"
RAIL_LOCAL_EFFECT_CONTRACT_SCHEMA = "shohin-ettr-parallel-terminal-state-contract-v16"
RAIL_LOCAL_EFFECT_REPORT_SCHEMA = "shohin-ettr-parallel-terminal-state-report-v16"
POST_WRITE_LINK_CONTRACT_SCHEMA = "shohin-ettr-parallel-terminal-state-contract-v17"
POST_WRITE_LINK_REPORT_SCHEMA = "shohin-ettr-parallel-terminal-state-report-v17"
OPERATION_FAMILY_GATE_CONTRACT_SCHEMA = (
    "shohin-ettr-parallel-terminal-state-contract-v18"
)
OPERATION_FAMILY_GATE_REPORT_SCHEMA = "shohin-ettr-parallel-terminal-state-report-v18"
OPERATION_FAMILY_ISLAND_CONTRACT_SCHEMA = (
    "shohin-ettr-parallel-terminal-state-contract-v19"
)
OPERATION_FAMILY_ISLAND_REPORT_SCHEMA = (
    "shohin-ettr-parallel-terminal-state-report-v19"
)
OPERATION_STATE_BOUND_FAMILY_CONTRACT_SCHEMA = (
    "shohin-ettr-parallel-terminal-state-contract-v20"
)
OPERATION_STATE_BOUND_FAMILY_REPORT_SCHEMA = (
    "shohin-ettr-parallel-terminal-state-report-v20"
)
OPERATION_STATE_BOUND_FAMILY_JOINT_CONTRACT_SCHEMA = (
    "shohin-ettr-parallel-terminal-state-contract-v21"
)
OPERATION_STATE_BOUND_FAMILY_JOINT_REPORT_SCHEMA = (
    "shohin-ettr-parallel-terminal-state-report-v21"
)
CONTRACT_SCHEMA = "shohin-ettr-parallel-terminal-state-contract-v3"
REPORT_SCHEMA = "shohin-ettr-parallel-terminal-state-report-v3"
CAUSAL_DELTA_CONTRACT_SCHEMA = "shohin-ettr-parallel-terminal-state-contract-v2"
CAUSAL_DELTA_REPORT_SCHEMA = "shohin-ettr-parallel-terminal-state-report-v2"
LEGACY_CONTRACT_SCHEMA = "shohin-ettr-parallel-terminal-state-contract-v1"
LEGACY_REPORT_SCHEMA = "shohin-ettr-parallel-terminal-state-report-v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_TERMINAL_RUN_FILES = (
    "pilot-contract.json",
    "report.json",
    "terminal-compiler-final.safetensors",
    "terminal-compiler-initial.safetensors",
    "train.jsonl",
)


class ParallelTerminalStatePilotError(RuntimeError):
    """The terminal-state quotient pilot violated its sealed contract."""


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
    parser.add_argument("--terminal-warm-start-dir", type=Path)
    parser.add_argument("--terminal-warm-start-sha256s-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--architecture-seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--start-position", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--causal-delta-weight", type=float, required=True)
    parser.add_argument(
        "--training-initial-state",
        choices=("oracle", "autonomous"),
        default="autonomous",
    )
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--relation-width", type=int, default=64)
    parser.add_argument("--residual-edits", action="store_true")
    parser.add_argument("--atomic-edits", action="store_true")
    parser.add_argument("--lexical-command", action="store_true")
    parser.add_argument("--token-native-command-mask", action="store_true")
    parser.add_argument("--cover-verified-command-mask", action="store_true")
    parser.add_argument(
        "--token-native-occurrence-command",
        action="store_true",
    )
    parser.add_argument(
        "--token-native-syntax-graph-command",
        action="store_true",
    )
    parser.add_argument(
        "--token-native-declaration-binding-command",
        action="store_true",
    )
    parser.add_argument(
        "--token-native-operation-recurrence-command",
        action="store_true",
    )
    parser.add_argument(
        "--token-native-operation-state-command",
        action="store_true",
    )
    parser.add_argument(
        "--factorized-operation-effect-command",
        action="store_true",
    )
    parser.add_argument(
        "--operation-effect-set-command",
        action="store_true",
    )
    parser.add_argument(
        "--operation-effect-role-anchors",
        action="store_true",
    )
    parser.add_argument(
        "--operation-effect-cardinality-gate",
        action="store_true",
    )
    parser.add_argument(
        "--operation-effect-write-link-rails",
        action="store_true",
    )
    parser.add_argument(
        "--operation-effect-rail-local-loss",
        action="store_true",
    )
    parser.add_argument(
        "--operation-effect-post-write-link-binding",
        action="store_true",
    )
    parser.add_argument(
        "--operation-effect-family-gate",
        action="store_true",
    )
    parser.add_argument(
        "--operation-effect-family-island",
        action="store_true",
    )
    parser.add_argument(
        "--operation-effect-family-state-binding",
        action="store_true",
    )
    parser.add_argument("--atomic-action-weight", type=float, default=1.0)
    parser.add_argument(
        "--required-device-class",
        choices=("h100", "cuda"),
        default="h100",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    operation_state = getattr(
        args,
        "token_native_operation_state_command",
        False,
    )
    factorized_operation_effect = getattr(
        args,
        "factorized_operation_effect_command",
        False,
    )
    operation_effect_set = getattr(
        args,
        "operation_effect_set_command",
        False,
    )
    operation_effect_role_anchors = getattr(
        args,
        "operation_effect_role_anchors",
        False,
    )
    operation_effect_cardinality_gate = getattr(
        args,
        "operation_effect_cardinality_gate",
        False,
    )
    operation_effect_write_link_rails = getattr(
        args,
        "operation_effect_write_link_rails",
        False,
    )
    operation_effect_rail_local_loss = getattr(
        args,
        "operation_effect_rail_local_loss",
        False,
    )
    operation_effect_post_write_link_binding = getattr(
        args,
        "operation_effect_post_write_link_binding",
        False,
    )
    operation_effect_family_gate = getattr(
        args,
        "operation_effect_family_gate",
        False,
    )
    operation_effect_family_island = getattr(
        args,
        "operation_effect_family_island",
        False,
    )
    operation_effect_family_state_binding = getattr(
        args,
        "operation_effect_family_state_binding",
        False,
    )
    terminal_warm_start_dir = getattr(args, "terminal_warm_start_dir", None)
    terminal_warm_start_sha256s_sha256 = getattr(
        args,
        "terminal_warm_start_sha256s_sha256",
        None,
    )
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
        or not 0 <= args.architecture_seed < 2**63
        or not 0 <= args.data_seed < 2**63
        or args.updates < 1
        or args.start_position < 0
        or args.eval_batches < 2
        or args.log_every < 1
        or not math.isfinite(args.learning_rate)
        or not 0.0 < args.learning_rate < 1.0
        or not math.isfinite(args.gradient_clip)
        or args.gradient_clip <= 0.0
        or not math.isfinite(args.causal_delta_weight)
        or args.causal_delta_weight <= 0.0
        or not math.isfinite(args.atomic_action_weight)
        or args.atomic_action_weight <= 0.0
        or (args.residual_edits and args.atomic_edits)
        or (args.lexical_command and not args.atomic_edits)
        or (
            args.token_native_command_mask
            and (not args.atomic_edits or not args.lexical_command)
        )
        or (args.token_native_occurrence_command and not args.token_native_command_mask)
        or (args.cover_verified_command_mask and not args.token_native_command_mask)
        or (
            args.token_native_syntax_graph_command
            and not args.token_native_command_mask
        )
        or (
            args.token_native_occurrence_command
            and args.token_native_syntax_graph_command
        )
        or (
            args.token_native_declaration_binding_command
            and not args.token_native_syntax_graph_command
        )
        or (
            args.token_native_operation_recurrence_command
            and not args.token_native_declaration_binding_command
        )
        or (operation_state and not args.token_native_operation_recurrence_command)
        or (operation_state and args.training_initial_state != "oracle")
        or (factorized_operation_effect and not operation_state)
        or (operation_effect_set and not operation_state)
        or (operation_effect_set and factorized_operation_effect)
        or (
            operation_effect_role_anchors
            and not (operation_effect_set or operation_effect_write_link_rails)
        )
        or (operation_effect_cardinality_gate and not operation_effect_role_anchors)
        or (operation_effect_write_link_rails and not operation_state)
        or (operation_effect_write_link_rails and operation_effect_set)
        or (operation_effect_write_link_rails and factorized_operation_effect)
        or (operation_effect_write_link_rails and not operation_effect_role_anchors)
        or (operation_effect_write_link_rails and operation_effect_cardinality_gate)
        or (operation_effect_rail_local_loss and not operation_effect_write_link_rails)
        or (
            operation_effect_post_write_link_binding
            and (
                not operation_effect_write_link_rails
                or operation_effect_rail_local_loss
            )
        )
        or (
            operation_effect_family_gate
            and (
                not operation_effect_write_link_rails
                or operation_effect_rail_local_loss
                or operation_effect_post_write_link_binding
            )
        )
        or (operation_effect_family_island and not operation_effect_family_gate)
        or (operation_effect_family_state_binding and not operation_effect_family_gate)
        or (
            operation_effect_family_state_binding
            and not operation_effect_family_island
            and terminal_warm_start_dir is None
        )
        or (
            terminal_warm_start_dir is not None
            and (
                not operation_effect_family_gate
                or operation_effect_family_island
            )
        )
        or (
            (terminal_warm_start_dir is None)
            != (terminal_warm_start_sha256s_sha256 is None)
        )
        or (
            terminal_warm_start_dir is not None
            and (
                not terminal_warm_start_dir.is_absolute()
                or terminal_warm_start_dir == args.output
                or terminal_warm_start_dir.is_symlink()
                or not terminal_warm_start_dir.is_dir()
                or _HEX64.fullmatch(terminal_warm_start_sha256s_sha256) is None
            )
        )
        or args.output.exists()
        or args.output.is_symlink()
        or not args.output.parent.is_dir()
    ):
        raise ParallelTerminalStatePilotError("terminal-state pilot arguments differ")


def _causal_edge_indices(
    rectangle_rows: torch.Tensor,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if (
        rectangle_rows.ndim != 3
        or rectangle_rows.shape[1:] != (2, 2)
        or rectangle_rows.dtype != torch.long
    ):
        raise ParallelTerminalStatePilotError(
            "terminal-state causal rectangle geometry differs"
        )
    r00 = rectangle_rows[:, 0, 0]
    r01 = rectangle_rows[:, 0, 1]
    r10 = rectangle_rows[:, 1, 0]
    r11 = rectangle_rows[:, 1, 1]
    return {
        "world": (torch.cat((r00, r01)), torch.cat((r10, r11))),
        "command": (torch.cat((r00, r10)), torch.cat((r01, r11))),
    }


def _changed_coordinate_delta_brier(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor,
    categorical: bool,
) -> tuple[torch.Tensor | None, int]:
    if (
        predicted.shape != target.shape
        or mask.shape != target.shape[: mask.ndim]
        or mask.dtype != torch.bool
        or predicted.shape[0] <= int(torch.stack((left, right)).max())
    ):
        raise ParallelTerminalStatePilotError(
            "terminal-state causal delta geometry differs"
        )
    predicted_delta = (
        predicted.index_select(0, right).float()
        - predicted.index_select(0, left).float()
    )
    target_delta = (
        target.index_select(0, right).float() - target.index_select(0, left).float()
    )
    support = mask.index_select(0, left) & mask.index_select(0, right)
    if categorical:
        if predicted.ndim != 3 or mask.ndim != 2:
            raise ParallelTerminalStatePilotError(
                "terminal-state categorical causal delta geometry differs"
            )
        changed = support & target_delta.abs().amax(dim=-1).gt(0.0)
        error = (predicted_delta - target_delta).square().sum(dim=-1)
    else:
        changed = support & target_delta.abs().gt(0.0)
        error = (predicted_delta - target_delta).square()
    changed_count = int(changed.sum().detach().cpu())
    if changed_count == 0:
        return None, 0
    return error[changed].mean(), changed_count


def causal_terminal_delta_brier(
    predicted,
    target,
    *,
    rectangle_rows: torch.Tensor,
    slot_mask: torch.Tensor,
    relation_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, int]]:
    """Credit only terminal coordinates changed by WORLD or COMMAND.

    Rectangle membership is a training-time grouping contract. The objective
    consumes terminal packet targets, never QUERY bytes or answer labels.
    """

    if (
        slot_mask.ndim != 2
        or relation_mask.ndim != 4
        or slot_mask.dtype != torch.bool
        or relation_mask.dtype != torch.bool
        or slot_mask.shape[0] != rectangle_rows.numel()
        or relation_mask.shape[0] != rectangle_rows.numel()
    ):
        raise ParallelTerminalStatePilotError(
            "terminal-state causal support geometry differs"
        )
    fields = {
        "active": (predicted.active, target.active, slot_mask, False),
        "root": (predicted.root, target.root, slot_mask, False),
        "relations": (
            predicted.relations,
            target.relations,
            relation_mask,
            False,
        ),
        "type_index": (
            predicted.type_probabilities,
            target.type_probabilities,
            slot_mask,
            True,
        ),
        "value_code": (
            predicted.value_probabilities,
            target.value_probabilities,
            slot_mask,
            True,
        ),
        "committed": (
            predicted.committed,
            target.committed,
            torch.ones_like(target.committed, dtype=torch.bool),
            False,
        ),
        "halted": (
            predicted.halted,
            target.halted,
            torch.ones_like(target.halted, dtype=torch.bool),
            False,
        ),
    }
    parts: dict[str, torch.Tensor] = {}
    changed_counts: dict[str, int] = {}
    for axis, (left, right) in _causal_edge_indices(rectangle_rows).items():
        for field, (field_predicted, field_target, mask, categorical) in fields.items():
            value, changed_count = _changed_coordinate_delta_brier(
                field_predicted,
                field_target,
                left=left,
                right=right,
                mask=mask,
                categorical=categorical,
            )
            key = f"{axis}.{field}"
            changed_counts[key] = changed_count
            if value is not None:
                parts[key] = value
    if not parts:
        raise ParallelTerminalStatePilotError(
            "terminal-state causal delta has no changed coordinates"
        )
    return torch.stack(tuple(parts.values())).mean(), parts, changed_counts


def derive_atomic_edit_targets(
    initial,
    target,
) -> dict[str, torch.Tensor]:
    """Canonicalize an initial/terminal state difference without a trace."""

    initial_active = initial.active.gt(0.5)
    target_active = target.active.gt(0.5)
    initial_value = initial.value_probabilities.argmax(-1)
    target_value = target.value_probabilities.argmax(-1)
    initial_type = initial.type_probabilities.argmax(-1)
    target_type = target.type_probabilities.argmax(-1)

    node_action = torch.zeros_like(initial_value)
    node_action = torch.where(
        ~initial_active & target_active,
        torch.ones_like(node_action),
        node_action,
    )
    node_action = torch.where(
        initial_active & ~target_active,
        torch.full_like(node_action, 3),
        node_action,
    )
    retained = initial_active & target_active
    node_action = torch.where(
        retained & initial_type.ne(target_type),
        torch.full_like(node_action, 4),
        node_action,
    )
    node_action = torch.where(
        retained & initial_type.eq(target_type) & initial_value.ne(target_value),
        torch.full_like(node_action, 2),
        node_action,
    )

    initial_relation = initial.relations.gt(0.5)
    target_relation = target.relations.gt(0.5)
    relation_action = torch.zeros_like(initial.relations, dtype=torch.long)
    relation_action = torch.where(
        ~initial_relation & target_relation,
        torch.ones_like(relation_action),
        relation_action,
    )
    relation_action = torch.where(
        initial_relation & ~target_relation,
        torch.full_like(relation_action, 2),
        relation_action,
    )

    root_action = torch.zeros(
        initial.root.shape[0],
        dtype=torch.long,
        device=initial.root.device,
    )
    root_changed = initial.root.ne(target.root).any(-1)
    target_has_root = target.root.gt(0.5).any(-1)
    root_action = torch.where(
        root_changed & ~target_has_root,
        torch.ones_like(root_action),
        root_action,
    )
    root_action = torch.where(
        root_changed & target_has_root,
        target.root.argmax(-1) + 2,
        root_action,
    )

    disposition_action = torch.zeros_like(root_action)
    status_changed = initial.committed.ne(target.committed) | initial.halted.ne(
        target.halted
    )
    disposition_action = torch.where(
        status_changed & target.committed.gt(0.5) & target.halted.le(0.5),
        torch.ones_like(disposition_action),
        disposition_action,
    )
    disposition_action = torch.where(
        status_changed & target.committed.le(0.5) & target.halted.gt(0.5),
        torch.full_like(disposition_action, 2),
        disposition_action,
    )
    disposition_action = torch.where(
        status_changed & target.committed.gt(0.5) & target.halted.gt(0.5),
        torch.full_like(disposition_action, 3),
        disposition_action,
    )
    return {
        "node_action": node_action,
        "value_code": target_value,
        "type_index": target_type,
        "relation_action": relation_action,
        "root_action": root_action,
        "disposition_action": disposition_action,
    }


def verify_atomic_edit_reconstruction(
    compiler: ParallelTerminalStateCompiler,
    initial,
    target,
    labels: dict[str, torch.Tensor],
    *,
    steps: int,
) -> None:
    """Fail before optimization if the canonical edit cannot express target."""

    edits = AtomicTypedEdits(
        node_action=F.one_hot(labels["node_action"], 5).float(),
        value_code=F.one_hot(
            labels["value_code"],
            compiler.config.num_value_codes,
        ).float(),
        type_index=F.one_hot(
            labels["type_index"],
            compiler.config.num_types,
        ).float(),
        relation_action=F.one_hot(labels["relation_action"], 3).float(),
        root_action=F.one_hot(
            labels["root_action"],
            2 + compiler.config.num_slots,
        ).float(),
        disposition_action=F.one_hot(
            labels["disposition_action"],
            4,
        ).float(),
    )
    reconstructed = compiler.apply_atomic_edits(
        initial,
        edits,
        steps=steps,
        hard=True,
    )
    for name in (
        "value_probabilities",
        "type_probabilities",
        "relations",
        "active",
        "root",
        "committed",
        "halted",
    ):
        if not torch.equal(getattr(reconstructed, name), getattr(target, name)):
            raise ParallelTerminalStatePilotError(
                f"canonical atomic edit reconstruction differs: {name}"
            )


def _class_balanced_nll(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, int]]:
    if probabilities.shape[:-1] != target.shape or target.shape != mask.shape:
        raise ParallelTerminalStatePilotError(
            "atomic typed-edit supervision geometry differs"
        )
    losses = []
    counts: dict[str, int] = {}
    log_probabilities = probabilities.float().clamp_min(1e-7).log()
    for index in range(probabilities.shape[-1]):
        selected = mask & target.eq(index)
        count = int(selected.sum().detach().cpu())
        counts[str(index)] = count
        if count:
            losses.append(-log_probabilities[..., index][selected].mean())
    if not losses:
        raise ParallelTerminalStatePilotError("atomic typed-edit supervision is empty")
    return torch.stack(losses).mean(), counts


def _operation_effect_targets(
    target: dict[str, torch.Tensor],
    *,
    maximum_effects: int,
    slot_mask: torch.Tensor,
    relation_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Canonicalize a dense atomic difference as an unordered effect set."""

    node_action = target["node_action"]
    relation_action = target["relation_action"]
    batch, slots = node_action.shape
    relations = relation_action.shape[1]
    device = node_action.device

    node_kinds = torch.tensor(
        [EFFECT_ALLOCATE, EFFECT_WRITE, EFFECT_CLEAR, EFFECT_REPLACE],
        device=device,
    )
    node_mask = torch.stack(
        tuple(node_action.eq(index) & slot_mask for index in range(1, 5)),
        dim=1,
    )
    node_kind = node_kinds[:, None].expand(-1, slots).reshape(-1)
    node_index = torch.arange(slots, device=device).repeat(4)
    node_value = (
        target["value_code"][:, None, :]
        .expand(-1, 4, -1)
        .reshape(
            batch,
            -1,
        )
    )
    node_type = (
        target["type_index"][:, None, :]
        .expand(-1, 4, -1)
        .reshape(
            batch,
            -1,
        )
    )

    relation_cells = relations * slots * slots
    relation_mask_by_action = torch.stack(
        (
            relation_action.eq(1) & relation_mask,
            relation_action.eq(2) & relation_mask,
        ),
        dim=1,
    )
    relation_kind = (
        torch.tensor(
            [EFFECT_LINK, EFFECT_UNLINK],
            device=device,
        )[:, None]
        .expand(-1, relation_cells)
        .reshape(-1)
    )
    relation_index = torch.arange(relation_cells, device=device).repeat(2)

    root_action = target["root_action"]
    root_mask = torch.cat(
        (
            root_action.eq(1).unsqueeze(-1),
            root_action[:, None].eq(torch.arange(slots, device=device)[None, :] + 2),
        ),
        dim=1,
    )
    root_kind = torch.tensor(
        [EFFECT_ROOT_CLEAR, *([EFFECT_ROOT_SET] * slots)],
        device=device,
    )
    root_index = torch.tensor([0, *range(slots)], device=device)

    disposition = target["disposition_action"]
    disposition_mask = torch.stack(
        tuple(disposition.eq(index) for index in range(1, 4)),
        dim=1,
    )
    disposition_kind = torch.tensor(
        [EFFECT_COMMIT, EFFECT_HALT, EFFECT_REJECT],
        device=device,
    )

    mask = torch.cat(
        (
            node_mask.reshape(batch, -1),
            relation_mask_by_action.reshape(batch, -1),
            root_mask,
            disposition_mask,
        ),
        dim=1,
    )
    zeros_node = torch.zeros(
        2 * relation_cells + 1 + slots + 3,
        dtype=torch.long,
        device=device,
    )
    zeros_relation = torch.zeros(
        4 * slots + 1 + slots + 3,
        dtype=torch.long,
        device=device,
    )
    kind_values = torch.cat((node_kind, relation_kind, root_kind, disposition_kind))
    node_values = torch.cat((node_index, zeros_node))
    relation_values = torch.cat(
        (
            zeros_relation[: 4 * slots],
            relation_index,
            zeros_relation[4 * slots :],
        )
    )
    root_values = torch.cat(
        (
            torch.zeros(
                4 * slots + 2 * relation_cells,
                dtype=torch.long,
                device=device,
            ),
            root_index,
            torch.zeros(3, dtype=torch.long, device=device),
        )
    )
    payload_zeros = torch.zeros(
        batch,
        2 * relation_cells + 1 + slots + 3,
        dtype=torch.long,
        device=device,
    )
    value_values = torch.cat((node_value, payload_zeros), dim=1)
    type_values = torch.cat((node_type, payload_zeros), dim=1)
    if not (
        mask.shape[1]
        == kind_values.shape[0]
        == node_values.shape[0]
        == relation_values.shape[0]
        == root_values.shape[0]
        == value_values.shape[1]
        == type_values.shape[1]
    ):
        raise ParallelTerminalStatePilotError(
            "operation effect target geometry differs"
        )
    count = mask.sum(-1)
    if bool(count.gt(maximum_effects).any()):
        maximum = int(count.max().detach().cpu())
        raise ParallelTerminalStatePilotError(
            f"operation effect target exceeds set capacity: {maximum}"
        )
    rank = mask.cumsum(-1) - 1
    outputs = {
        name: torch.zeros(
            batch,
            maximum_effects,
            dtype=torch.long,
            device=device,
        )
        for name in ("kind", "node", "relation", "root", "value", "type")
    }
    sources = {
        "kind": kind_values[None].expand(batch, -1),
        "node": node_values[None].expand(batch, -1),
        "relation": relation_values[None].expand(batch, -1),
        "root": root_values[None].expand(batch, -1),
        "value": value_values,
        "type": type_values,
    }
    for effect_rank in range(maximum_effects):
        selected = mask & rank.eq(effect_rank)
        for name, source in sources.items():
            outputs[name][:, effect_rank] = torch.where(
                selected,
                source,
                torch.zeros_like(source),
            ).sum(-1)
    outputs["count"] = count
    return outputs


def operation_effect_set_loss(
    edits: AtomicTypedEdits,
    target: dict[str, torch.Tensor],
    *,
    slot_mask: torch.Tensor,
    relation_mask: torch.Tensor,
) -> torch.Tensor:
    """Permutation-invariant typed-effect matching used only for training."""

    fields = (
        edits.effect_kind,
        edits.effect_node_pointer,
        edits.effect_value_code,
        edits.effect_type_index,
        edits.effect_relation_link,
        edits.effect_relation_unlink,
        edits.effect_root_pointer,
    )
    if any(value is None for value in fields):
        raise ParallelTerminalStatePilotError(
            "operation effect set predictions are incomplete"
        )
    kind = edits.effect_kind
    node_pointer = edits.effect_node_pointer
    value_code = edits.effect_value_code
    type_index = edits.effect_type_index
    relation_link = edits.effect_relation_link
    relation_unlink = edits.effect_relation_unlink
    root_pointer = edits.effect_root_pointer
    assert kind is not None
    assert node_pointer is not None
    assert value_code is not None
    assert type_index is not None
    assert relation_link is not None
    assert relation_unlink is not None
    assert root_pointer is not None
    if kind.ndim != 3 or kind.shape[-1] != EFFECT_KIND_COUNT:
        raise ParallelTerminalStatePilotError(
            "operation effect set kind geometry differs"
        )
    batch, effects, _kinds = kind.shape
    labels = _operation_effect_targets(
        target,
        maximum_effects=effects,
        slot_mask=slot_mask,
        relation_mask=relation_mask,
    )
    costs = []
    epsilon = 1e-7
    flat_link = relation_link.reshape(batch, effects, -1)
    flat_unlink = relation_unlink.reshape(batch, effects, -1)

    def selected_nll(
        probabilities: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        gathered = probabilities.gather(
            -1,
            label[:, None, None].expand(-1, effects, 1),
        ).squeeze(-1)
        return -gathered.clamp_min(epsilon).log()

    for target_rank in range(effects):
        target_kind = labels["kind"][:, target_rank]
        value = selected_nll(kind, target_kind)
        node_kind = target_kind.ge(EFFECT_ALLOCATE) & target_kind.le(EFFECT_REPLACE)
        pointer_channel = target_kind.ne(EFFECT_ALLOCATE).to(torch.long)
        selected_pointer = node_pointer.gather(
            2,
            pointer_channel[:, None, None, None].expand(
                -1,
                effects,
                1,
                node_pointer.shape[-1],
            ),
        ).squeeze(2)
        value = value + node_kind[:, None] * selected_nll(
            selected_pointer,
            labels["node"][:, target_rank],
        )
        writes_value = (
            target_kind.eq(EFFECT_ALLOCATE)
            | target_kind.eq(EFFECT_WRITE)
            | target_kind.eq(EFFECT_REPLACE)
        )
        value = value + writes_value[:, None] * selected_nll(
            value_code,
            labels["value"][:, target_rank],
        )
        writes_type = target_kind.eq(EFFECT_ALLOCATE) | target_kind.eq(EFFECT_REPLACE)
        value = value + writes_type[:, None] * selected_nll(
            type_index,
            labels["type"][:, target_rank],
        )
        value = value + target_kind.eq(EFFECT_LINK)[:, None] * selected_nll(
            flat_link,
            labels["relation"][:, target_rank],
        )
        value = value + target_kind.eq(EFFECT_UNLINK)[:, None] * selected_nll(
            flat_unlink,
            labels["relation"][:, target_rank],
        )
        value = value + target_kind.eq(EFFECT_ROOT_SET)[:, None] * selected_nll(
            root_pointer,
            labels["root"][:, target_rank],
        )
        costs.append(value)
    cost = torch.stack(costs, dim=-1)
    log_assignment = -cost.detach() / 0.25
    for _ in range(12):
        log_assignment = log_assignment - log_assignment.logsumexp(
            dim=2,
            keepdim=True,
        )
        log_assignment = log_assignment - log_assignment.logsumexp(
            dim=1,
            keepdim=True,
        )
    assignment = log_assignment.exp().detach()
    real = labels["kind"].ne(EFFECT_NOOP)
    real_count = real.sum(-1, keepdim=True).clamp_min(1)
    noop_count = (~real).sum(-1, keepdim=True).clamp_min(1)
    column_weight = torch.where(
        real,
        real_count.reciprocal().to(cost.dtype),
        noop_count.reciprocal().to(cost.dtype),
    )
    weighted = assignment * column_weight[:, None, :]
    matching_loss = (weighted * cost).sum() / weighted.sum().clamp_min(1e-7)
    if edits.effect_count is None:
        return matching_loss
    batch_mask = torch.ones(batch, dtype=torch.bool, device=kind.device)
    count_loss, _counts = _class_balanced_nll(
        edits.effect_count,
        labels["count"],
        batch_mask,
    )
    return torch.stack((matching_loss, count_loss)).mean()


def operation_write_link_rail_loss(
    edits: AtomicTypedEdits,
    target: dict[str, torch.Tensor],
    *,
    slot_mask: torch.Tensor,
    relation_mask: torch.Tensor,
) -> torch.Tensor:
    """Supervise canonical WRITE/LINK motors without cross-kind matching.

    The deployed object remains an unordered typed set.  Training assigns
    WRITE targets in ascending state-slot order and LINK targets in ascending
    flattened relation order.  This fixed convention removes the detached
    cross-kind Sinkhorn assignment, so activity, pointer, and payload heads
    receive independent gradients even when one rail initially collapses.
    """

    fields = (
        edits.effect_kind,
        edits.effect_node_pointer,
        edits.effect_value_code,
        edits.effect_relation_link,
    )
    if any(value is None for value in fields):
        raise ParallelTerminalStatePilotError(
            "write/link rail predictions are incomplete"
        )
    kind = edits.effect_kind
    node_pointer = edits.effect_node_pointer
    value_code = edits.effect_value_code
    relation_link = edits.effect_relation_link
    assert kind is not None
    assert node_pointer is not None
    assert value_code is not None
    assert relation_link is not None
    batch = kind.shape[0]
    if (
        kind.shape != (batch, WRITE_LINK_RAIL_EFFECT_SLOTS, EFFECT_KIND_COUNT)
        or node_pointer.shape[:3] != (batch, WRITE_LINK_RAIL_EFFECT_SLOTS, 2)
        or value_code.shape[:2] != (batch, WRITE_LINK_RAIL_EFFECT_SLOTS)
        or relation_link.shape[:2] != (batch, WRITE_LINK_RAIL_EFFECT_SLOTS)
    ):
        raise ParallelTerminalStatePilotError(
            "write/link rail prediction geometry differs"
        )
    labels = _operation_effect_targets(
        target,
        maximum_effects=WRITE_LINK_RAIL_EFFECT_SLOTS,
        slot_mask=slot_mask,
        relation_mask=relation_mask,
    )
    target_kind = labels["kind"]
    write_mask = target_kind.eq(EFFECT_WRITE)
    link_mask = target_kind.eq(EFFECT_LINK)
    if (
        int(write_mask.sum(-1).max().detach().cpu()) > WRITE_RAIL_EFFECT_SLOTS
        or int(link_mask.sum(-1).max().detach().cpu()) > LINK_RAIL_EFFECT_SLOTS
        or bool((target_kind.ne(EFFECT_NOOP) & ~write_mask & ~link_mask).any())
    ):
        raise ParallelTerminalStatePilotError("write/link rail target support differs")

    write_active = torch.zeros(
        batch,
        WRITE_RAIL_EFFECT_SLOTS,
        dtype=torch.long,
        device=kind.device,
    )
    write_node = torch.zeros_like(write_active)
    write_value = torch.zeros_like(write_active)
    link_active = torch.zeros(
        batch,
        LINK_RAIL_EFFECT_SLOTS,
        dtype=torch.long,
        device=kind.device,
    )
    link_relation = torch.zeros_like(link_active)
    write_rank = write_mask.cumsum(-1) - 1
    link_rank = link_mask.cumsum(-1) - 1
    for target_rank in range(WRITE_LINK_RAIL_EFFECT_SLOTS):
        rows = torch.nonzero(write_mask[:, target_rank], as_tuple=False).flatten()
        if rows.numel():
            motors = write_rank[rows, target_rank]
            write_active[rows, motors] = 1
            write_node[rows, motors] = labels["node"][rows, target_rank]
            write_value[rows, motors] = labels["value"][rows, target_rank]
        rows = torch.nonzero(link_mask[:, target_rank], as_tuple=False).flatten()
        if rows.numel():
            motors = link_rank[rows, target_rank]
            link_active[rows, motors] = 1
            link_relation[rows, motors] = labels["relation"][rows, target_rank]

    write_probability = kind[:, :WRITE_RAIL_EFFECT_SLOTS, EFFECT_WRITE]
    link_probability = kind[:, WRITE_RAIL_EFFECT_SLOTS:, EFFECT_LINK]
    all_write = torch.ones_like(write_active, dtype=torch.bool)
    all_link = torch.ones_like(link_active, dtype=torch.bool)
    parts = [
        _class_balanced_nll(
            torch.stack((1.0 - write_probability, write_probability), dim=-1),
            write_active,
            all_write,
        )[0],
        _class_balanced_nll(
            torch.stack((1.0 - link_probability, link_probability), dim=-1),
            link_active,
            all_link,
        )[0],
    ]
    write_positive = write_active.bool()
    if bool(write_positive.any()):
        parts.extend(
            (
                _class_balanced_nll(
                    node_pointer[:, :WRITE_RAIL_EFFECT_SLOTS, 0],
                    write_node,
                    write_positive,
                )[0],
                _class_balanced_nll(
                    value_code[:, :WRITE_RAIL_EFFECT_SLOTS],
                    write_value,
                    write_positive,
                )[0],
            )
        )
    link_positive = link_active.bool()
    if bool(link_positive.any()):
        parts.append(
            _class_balanced_nll(
                relation_link[:, WRITE_RAIL_EFFECT_SLOTS:].flatten(2),
                link_relation,
                link_positive,
            )[0]
        )
    return torch.stack(parts).mean()


def operation_effect_family_targets(
    target: dict[str, torch.Tensor],
    *,
    slot_mask: torch.Tensor,
    relation_mask: torch.Tensor,
) -> torch.Tensor:
    """Derive the corpus-exact mutually exclusive NONE/WRITE/LINK family."""

    labels = _operation_effect_targets(
        target,
        maximum_effects=WRITE_LINK_RAIL_EFFECT_SLOTS,
        slot_mask=slot_mask,
        relation_mask=relation_mask,
    )["kind"]
    write = labels.eq(EFFECT_WRITE).any(-1)
    link = labels.eq(EFFECT_LINK).any(-1)
    unsupported = (
        labels.ne(EFFECT_NOOP) & ~labels.eq(EFFECT_WRITE) & ~labels.eq(EFFECT_LINK)
    )
    if bool((write & link).any()) or bool(unsupported.any()):
        raise ParallelTerminalStatePilotError(
            "operation effect family target support differs"
        )
    family = torch.full_like(
        write,
        OPERATION_EFFECT_FAMILY_NONE,
        dtype=torch.long,
    )
    family = torch.where(
        write,
        torch.full_like(family, OPERATION_EFFECT_FAMILY_WRITE),
        family,
    )
    family = torch.where(
        link,
        torch.full_like(family, OPERATION_EFFECT_FAMILY_LINK),
        family,
    )
    return family


def operation_effect_family_loss(
    edits: AtomicTypedEdits,
    target: dict[str, torch.Tensor],
    *,
    slot_mask: torch.Tensor,
    relation_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Supervise the corpus-exact mutually exclusive NONE/WRITE/LINK family."""

    if (
        edits.effect_family is None
        or edits.effect_family.ndim != 2
        or edits.effect_family.shape[-1] != OPERATION_EFFECT_FAMILY_COUNT
    ):
        raise ParallelTerminalStatePilotError(
            "operation effect family prediction differs"
        )
    family = operation_effect_family_targets(
        target,
        slot_mask=slot_mask,
        relation_mask=relation_mask,
    )
    mask = torch.ones_like(family, dtype=torch.bool)
    return _class_balanced_nll(edits.effect_family, family, mask)


def operation_family_island_objective(
    probabilities: tuple[torch.Tensor, ...],
    mask: torch.Tensor,
    oracle,
    initial,
    *,
    slot_mask: torch.Tensor,
    relation_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, dict[str, int]]]:
    """Fit only the public operation family against oracle preceding states."""

    if len(probabilities) != len(oracle.states) or not torch.equal(mask, oracle.mask):
        raise ParallelTerminalStatePilotError(
            "operation family island supervision geometry differs"
        )
    losses: list[torch.Tensor] = []
    parts: dict[str, torch.Tensor] = {}
    counts: dict[str, dict[str, int]] = {}
    previous = initial
    for rank, (family_probabilities, target) in enumerate(
        zip(probabilities, oracle.states, strict=True)
    ):
        index = torch.nonzero(mask[:, rank], as_tuple=False).flatten()
        if index.numel() == 0:
            continue
        previous_selected = index_typed_state(previous, index)
        target_selected = index_typed_state(target, index)
        labels = derive_atomic_edit_targets(previous_selected, target_selected)
        family_target = operation_effect_family_targets(
            labels,
            slot_mask=slot_mask.index_select(0, index),
            relation_mask=relation_mask.index_select(0, index),
        )
        loss, class_counts = _class_balanced_nll(
            family_probabilities.index_select(0, index),
            family_target,
            torch.ones_like(family_target, dtype=torch.bool),
        )
        losses.append(loss)
        parts[f"operation_{rank}.action.effect_family"] = loss
        counts[f"operation_{rank}.effect_family"] = class_counts
        previous = target
    if not losses:
        raise ParallelTerminalStatePilotError(
            "operation family island supervision is empty"
        )
    return torch.stack(losses).mean(), parts, counts


def atomic_typed_edit_loss(
    edits: AtomicTypedEdits,
    target: dict[str, torch.Tensor],
    *,
    slot_mask: torch.Tensor,
    relation_mask: torch.Tensor,
    rail_local_loss: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, dict[str, int]]]:
    """Supervise a canonical coherent state difference, class-balanced."""

    batch_mask = torch.ones(
        edits.root_action.shape[0],
        dtype=torch.bool,
        device=edits.root_action.device,
    )
    node_action = target["node_action"]
    parts: dict[str, torch.Tensor] = {}
    counts: dict[str, dict[str, int]] = {}
    specifications = {
        "node_action": (edits.node_action, node_action, slot_mask),
        "relation_action": (
            edits.relation_action,
            target["relation_action"],
            relation_mask,
        ),
        "root_action": (edits.root_action, target["root_action"], batch_mask),
        "disposition_action": (
            edits.disposition_action,
            target["disposition_action"],
            batch_mask,
        ),
        "value_code": (
            edits.value_code,
            target["value_code"],
            slot_mask & (node_action.eq(1) | node_action.eq(2) | node_action.eq(4)),
        ),
        "type_index": (
            edits.type_index,
            target["type_index"],
            slot_mask & (node_action.eq(1) | node_action.eq(4)),
        ),
    }
    effect_fields = (
        edits.effect_kind,
        edits.effect_node_pointer,
        edits.effect_value_code,
        edits.effect_type_index,
        edits.effect_relation_link,
        edits.effect_relation_unlink,
        edits.effect_root_pointer,
    )
    present_effect_fields = [value is not None for value in effect_fields]
    if any(present_effect_fields) and not all(present_effect_fields):
        raise ParallelTerminalStatePilotError("operation effect set heads differ")
    if all(present_effect_fields):
        parts["effect_set"] = (
            operation_write_link_rail_loss(
                edits,
                target,
                slot_mask=slot_mask,
                relation_mask=relation_mask,
            )
            if rail_local_loss
            else operation_effect_set_loss(
                edits,
                target,
                slot_mask=slot_mask,
                relation_mask=relation_mask,
            )
        )
    if edits.effect_family is not None:
        if not all(present_effect_fields):
            raise ParallelTerminalStatePilotError(
                "operation effect family requires typed effect rails"
            )
        family_loss, family_counts = operation_effect_family_loss(
            edits,
            target,
            slot_mask=slot_mask,
            relation_mask=relation_mask,
        )
        parts["effect_family"] = family_loss
        counts["effect_family"] = family_counts
    count_specifications = {
        "node_edit_count": (
            edits.node_edit_count,
            (node_action.ne(0) & slot_mask).sum(-1),
        ),
        "relation_link_count": (
            edits.relation_link_count,
            (target["relation_action"].eq(1) & relation_mask).sum(
                dim=(1, 2, 3)
            ),
        ),
        "relation_unlink_count": (
            edits.relation_unlink_count,
            (target["relation_action"].eq(2) & relation_mask).sum(
                dim=(1, 2, 3)
            ),
        ),
    }
    present_counts = [
        probabilities is not None
        for probabilities, _labels in count_specifications.values()
    ]
    if any(present_counts) and not all(present_counts):
        raise ParallelTerminalStatePilotError(
            "factorized operation effect count heads differ"
        )
    for name, (probabilities, labels) in count_specifications.items():
        if probabilities is None:
            continue
        if int(labels.max().detach().cpu()) >= probabilities.shape[-1]:
            raise ParallelTerminalStatePilotError(
                f"factorized operation effect count exceeds support: {name}"
            )
        value, class_counts = _class_balanced_nll(
            probabilities,
            labels,
            batch_mask,
        )
        parts[name] = value
        counts[name] = class_counts
    for name, (probabilities, labels, mask) in specifications.items():
        if not bool(mask.any()):
            counts[name] = {str(index): 0 for index in range(probabilities.shape[-1])}
            continue
        value, class_counts = _class_balanced_nll(
            probabilities,
            labels,
            mask,
        )
        parts[name] = value
        counts[name] = class_counts
    return torch.stack(tuple(parts.values())).mean(), parts, counts


def operation_boundary_objective(
    compiler: OperationStateTransitionCompiler,
    trace,
    oracle,
    initial,
    terminal_target,
    *,
    slot_mask: torch.Tensor,
    relation_mask: torch.Tensor,
    verify_reconstruction: bool,
    rail_local_loss: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, dict[str, int]],
]:
    """Credit every public operation at its exact cumulative state boundary."""

    if (
        len(trace.operation_states) != len(oracle.states)
        or len(trace.operation_edits) != len(oracle.states)
        or trace.operation_mask.shape != oracle.mask.shape
        or not torch.equal(trace.operation_mask, oracle.mask)
    ):
        raise ParallelTerminalStatePilotError(
            "operation-boundary supervision geometry differs"
        )
    state_losses = []
    action_losses = []
    parts: dict[str, torch.Tensor] = {}
    counts: dict[str, dict[str, int]] = {}
    previous = initial
    for rank, (predicted, edits, target) in enumerate(
        zip(
            trace.operation_states,
            trace.operation_edits,
            oracle.states,
            strict=True,
        )
    ):
        index = torch.nonzero(oracle.mask[:, rank], as_tuple=False).flatten()
        if index.numel() == 0:
            continue
        predicted_selected = index_typed_state(predicted, index)
        target_selected = index_typed_state(target, index)
        previous_selected = index_typed_state(previous, index)
        selected_slots = slot_mask.index_select(0, index)
        selected_relations = relation_mask.index_select(0, index)
        state_loss, state_parts = _state_brier(
            predicted_selected,
            target_selected,
            slot_mask=selected_slots,
            relation_mask=selected_relations,
        )
        state_losses.append(state_loss)
        for name, value in state_parts.items():
            parts[f"operation_{rank}.state.{name}"] = value
        labels = derive_atomic_edit_targets(previous_selected, target_selected)
        if verify_reconstruction:
            verify_atomic_edit_reconstruction(
                compiler,
                previous_selected,
                target_selected,
                labels,
                steps=1,
            )
        action_loss, action_parts, action_counts = atomic_typed_edit_loss(
            index_atomic_edits(edits, index),
            labels,
            slot_mask=selected_slots,
            relation_mask=selected_relations,
            rail_local_loss=rail_local_loss,
        )
        action_losses.append(action_loss)
        for name, value in action_parts.items():
            parts[f"operation_{rank}.action.{name}"] = value
        for name, value in action_counts.items():
            counts[f"operation_{rank}.{name}"] = value
        previous = target

    final_labels = derive_atomic_edit_targets(
        oracle.last_state,
        terminal_target,
    )
    if verify_reconstruction:
        verify_atomic_edit_reconstruction(
            compiler,
            oracle.last_state,
            terminal_target,
            final_labels,
            steps=1,
        )
    final_action, final_parts, final_counts = atomic_typed_edit_loss(
        trace.final_edits,
        final_labels,
        slot_mask=slot_mask,
        relation_mask=relation_mask,
        rail_local_loss=False,
    )
    action_losses.append(final_action)
    for name, value in final_parts.items():
        parts[f"final.action.{name}"] = value
    for name, value in final_counts.items():
        counts[f"final.{name}"] = value
    return (
        torch.stack(state_losses).mean(),
        torch.stack(action_losses).mean(),
        parts,
        counts,
    )


def _module_sha256(module: torch.nn.Module, path: Path) -> str:
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in module.state_dict().items()
        },
        path,
    )
    os.chmod(path, 0o400)
    return _sha256_file(path)


def _terminal_warm_start_receipt(
    run_dir: Path,
    expected_sha256s_sha256: str,
    *,
    successor_schema: str,
) -> dict[str, object]:
    """Verify and describe one immutable terminal-compiler predecessor."""

    sums_path = run_dir / "SHA256SUMS"
    if (
        sums_path.is_symlink()
        or not sums_path.is_file()
        or _sha256_file(sums_path) != expected_sha256s_sha256
    ):
        raise ParallelTerminalStatePilotError(
            "terminal warm-start receipt differs"
        )
    entries: dict[str, str] = {}
    for line in sums_path.read_text(encoding="ascii").splitlines():
        fields = line.split("  ", 1)
        if (
            len(fields) != 2
            or _HEX64.fullmatch(fields[0]) is None
            or fields[1] not in _TERMINAL_RUN_FILES
            or fields[1] in entries
        ):
            raise ParallelTerminalStatePilotError(
                "terminal warm-start manifest differs"
            )
        entries[fields[1]] = fields[0]
    if set(entries) != set(_TERMINAL_RUN_FILES):
        raise ParallelTerminalStatePilotError(
            "terminal warm-start file set differs"
        )
    for name, expected in entries.items():
        path = run_dir / name
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != expected:
            raise ParallelTerminalStatePilotError(
                "terminal warm-start file differs"
            )

    try:
        contract = json.loads((run_dir / "pilot-contract.json").read_text())
        report = json.loads((run_dir / "report.json").read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ParallelTerminalStatePilotError(
            "terminal warm-start metadata differs"
        ) from exc
    schema = contract.get("schema")
    allowed_predecessor = {
        OPERATION_FAMILY_GATE_CONTRACT_SCHEMA: (
            OPERATION_FAMILY_ISLAND_CONTRACT_SCHEMA,
            OPERATION_FAMILY_ISLAND_REPORT_SCHEMA,
        ),
        OPERATION_STATE_BOUND_FAMILY_JOINT_CONTRACT_SCHEMA: (
            OPERATION_STATE_BOUND_FAMILY_CONTRACT_SCHEMA,
            OPERATION_STATE_BOUND_FAMILY_REPORT_SCHEMA,
        ),
    }.get(successor_schema)
    if (
        allowed_predecessor is None
        or schema != allowed_predecessor[0]
        or report.get("schema") != allowed_predecessor[1]
        or report.get("status") != "pass"
        or report.get("contract_sha256") != entries["pilot-contract.json"]
        or report.get("final_compiler_sha256")
        != entries["terminal-compiler-final.safetensors"]
    ):
        raise ParallelTerminalStatePilotError(
            "terminal warm-start lineage differs"
        )
    return {
        "compiler_sha256": entries["terminal-compiler-final.safetensors"],
        "contract_schema": schema,
        "contract_sha256": entries["pilot-contract.json"],
        "report_sha256": entries["report.json"],
        "run_dir": str(run_dir),
        "sha256s_sha256": expected_sha256s_sha256,
    }


def _load_terminal_warm_start(
    compiler: torch.nn.Module,
    run_dir: Path,
    expected_sha256s_sha256: str,
    *,
    successor_schema: str,
) -> dict[str, object]:
    receipt = _terminal_warm_start_receipt(
        run_dir,
        expected_sha256s_sha256,
        successor_schema=successor_schema,
    )
    try:
        state = load_file(
            str(run_dir / "terminal-compiler-final.safetensors"),
            device="cpu",
        )
        incompatibility = compiler.load_state_dict(state, strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ParallelTerminalStatePilotError(
            "terminal warm-start compiler differs"
        ) from exc
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ParallelTerminalStatePilotError(
            "terminal warm-start compiler differs"
        )
    return receipt


def _run_schemas(
    residual_edits: bool,
    atomic_edits: bool = False,
    lexical_command: bool = False,
    token_native_command_mask: bool = False,
    token_native_occurrence_command: bool = False,
    token_native_syntax_graph_command: bool = False,
    token_native_declaration_binding_command: bool = False,
    token_native_operation_recurrence_command: bool = False,
    token_native_operation_state_command: bool = False,
    factorized_operation_effect_command: bool = False,
    operation_effect_set_command: bool = False,
    operation_effect_role_anchors: bool = False,
    operation_effect_cardinality_gate: bool = False,
    operation_effect_write_link_rails: bool = False,
    operation_effect_rail_local_loss: bool = False,
    operation_effect_post_write_link_binding: bool = False,
    operation_effect_family_gate: bool = False,
    operation_effect_family_island: bool = False,
    operation_effect_family_state_binding: bool = False,
) -> tuple[str, str, str]:
    if operation_effect_write_link_rails:
        if (
            not token_native_operation_state_command
            or factorized_operation_effect_command
            or operation_effect_set_command
            or not operation_effect_role_anchors
            or operation_effect_cardinality_gate
            or (
                operation_effect_post_write_link_binding
                and operation_effect_rail_local_loss
            )
            or (
                operation_effect_family_gate
                and (
                    operation_effect_post_write_link_binding
                    or operation_effect_rail_local_loss
                )
            )
            or (operation_effect_family_island and not operation_effect_family_gate)
            or (
                operation_effect_family_state_binding
                and not operation_effect_family_gate
            )
        ):
            raise ParallelTerminalStatePilotError(
                "write/link rail architecture schema differs"
            )
        return (
            (
                OPERATION_STATE_BOUND_FAMILY_CONTRACT_SCHEMA
                if operation_effect_family_state_binding
                and operation_effect_family_island
                else OPERATION_STATE_BOUND_FAMILY_JOINT_CONTRACT_SCHEMA
                if operation_effect_family_state_binding
                else OPERATION_FAMILY_ISLAND_CONTRACT_SCHEMA
                if operation_effect_family_island
                else OPERATION_FAMILY_GATE_CONTRACT_SCHEMA
                if operation_effect_family_gate
                else POST_WRITE_LINK_CONTRACT_SCHEMA
                if operation_effect_post_write_link_binding
                else RAIL_LOCAL_EFFECT_CONTRACT_SCHEMA
                if operation_effect_rail_local_loss
                else WRITE_LINK_RAIL_CONTRACT_SCHEMA
            ),
            (
                OPERATION_STATE_BOUND_FAMILY_REPORT_SCHEMA
                if operation_effect_family_state_binding
                and operation_effect_family_island
                else OPERATION_STATE_BOUND_FAMILY_JOINT_REPORT_SCHEMA
                if operation_effect_family_state_binding
                else OPERATION_FAMILY_ISLAND_REPORT_SCHEMA
                if operation_effect_family_island
                else OPERATION_FAMILY_GATE_REPORT_SCHEMA
                if operation_effect_family_gate
                else POST_WRITE_LINK_REPORT_SCHEMA
                if operation_effect_post_write_link_binding
                else RAIL_LOCAL_EFFECT_REPORT_SCHEMA
                if operation_effect_rail_local_loss
                else WRITE_LINK_RAIL_REPORT_SCHEMA
            ),
            (
                "shohin-ettr-parallel-terminal-state-metric-v20"
                if operation_effect_family_state_binding
                and operation_effect_family_island
                else "shohin-ettr-parallel-terminal-state-metric-v21"
                if operation_effect_family_state_binding
                else "shohin-ettr-parallel-terminal-state-metric-v19"
                if operation_effect_family_island
                else "shohin-ettr-parallel-terminal-state-metric-v18"
                if operation_effect_family_gate
                else "shohin-ettr-parallel-terminal-state-metric-v17"
                if operation_effect_post_write_link_binding
                else "shohin-ettr-parallel-terminal-state-metric-v16"
                if operation_effect_rail_local_loss
                else "shohin-ettr-parallel-terminal-state-metric-v15"
            ),
        )
    if (
        operation_effect_rail_local_loss
        or operation_effect_post_write_link_binding
        or operation_effect_family_gate
        or operation_effect_family_island
        or operation_effect_family_state_binding
    ):
        raise ParallelTerminalStatePilotError(
            "rail-local effect loss requires write/link rails"
        )
    if operation_effect_set_command:
        if (
            not token_native_operation_state_command
            or factorized_operation_effect_command
        ):
            raise ParallelTerminalStatePilotError(
                "operation effect set architecture schema differs"
            )
        if operation_effect_cardinality_gate and not operation_effect_role_anchors:
            raise ParallelTerminalStatePilotError(
                "cardinality-gated effect architecture schema differs"
            )
        return (
            (
                CARDINALITY_GATED_EFFECT_SET_CONTRACT_SCHEMA
                if operation_effect_cardinality_gate
                else ROLE_ANCHORED_EFFECT_SET_CONTRACT_SCHEMA
                if operation_effect_role_anchors
                else OPERATION_EFFECT_SET_CONTRACT_SCHEMA
            ),
            (
                CARDINALITY_GATED_EFFECT_SET_REPORT_SCHEMA
                if operation_effect_cardinality_gate
                else ROLE_ANCHORED_EFFECT_SET_REPORT_SCHEMA
                if operation_effect_role_anchors
                else OPERATION_EFFECT_SET_REPORT_SCHEMA
            ),
            (
                "shohin-ettr-parallel-terminal-state-metric-v14"
                if operation_effect_cardinality_gate
                else "shohin-ettr-parallel-terminal-state-metric-v13"
                if operation_effect_role_anchors
                else "shohin-ettr-parallel-terminal-state-metric-v12"
            ),
        )
    if operation_effect_role_anchors:
        raise ParallelTerminalStatePilotError(
            "operation effect role-anchor architecture schema differs"
        )
    if factorized_operation_effect_command:
        if not token_native_operation_state_command:
            raise ParallelTerminalStatePilotError(
                "factorized operation effect architecture schema differs"
            )
        return (
            FACTORIZED_OPERATION_STATE_CONTRACT_SCHEMA,
            FACTORIZED_OPERATION_STATE_REPORT_SCHEMA,
            "shohin-ettr-parallel-terminal-state-metric-v11",
        )
    if token_native_operation_state_command:
        if (
            residual_edits
            or not atomic_edits
            or not lexical_command
            or not token_native_command_mask
            or token_native_occurrence_command
            or not token_native_syntax_graph_command
            or not token_native_declaration_binding_command
            or not token_native_operation_recurrence_command
        ):
            raise ParallelTerminalStatePilotError(
                "operation-state terminal architecture schema differs"
            )
        return (
            OPERATION_STATE_ATOMIC_CONTRACT_SCHEMA,
            OPERATION_STATE_ATOMIC_REPORT_SCHEMA,
            "shohin-ettr-parallel-terminal-state-metric-v10",
        )
    if token_native_operation_recurrence_command:
        if (
            residual_edits
            or not atomic_edits
            or not lexical_command
            or not token_native_command_mask
            or token_native_occurrence_command
            or not token_native_syntax_graph_command
            or not token_native_declaration_binding_command
        ):
            raise ParallelTerminalStatePilotError(
                "operation-recurrent terminal-state architecture schema differs"
            )
        return (
            OPERATION_RECURRENT_ATOMIC_CONTRACT_SCHEMA,
            OPERATION_RECURRENT_ATOMIC_REPORT_SCHEMA,
            "shohin-ettr-parallel-terminal-state-metric-v9",
        )
    if token_native_declaration_binding_command:
        if (
            residual_edits
            or not atomic_edits
            or not lexical_command
            or not token_native_command_mask
            or token_native_occurrence_command
            or not token_native_syntax_graph_command
        ):
            raise ParallelTerminalStatePilotError(
                "declaration-bound terminal-state architecture schema differs"
            )
        return (
            DECLARATION_BOUND_ATOMIC_CONTRACT_SCHEMA,
            DECLARATION_BOUND_ATOMIC_REPORT_SCHEMA,
            "shohin-ettr-parallel-terminal-state-metric-v8",
        )
    if token_native_syntax_graph_command:
        raise ParallelTerminalStatePilotError(
            "syntax graph requires declaration-bound terminal-state architecture"
        )
    if token_native_occurrence_command:
        if (
            residual_edits
            or not atomic_edits
            or not lexical_command
            or not token_native_command_mask
        ):
            raise ParallelTerminalStatePilotError(
                "occurrence-linked terminal-state architecture schema differs"
            )
        return (
            OCCURRENCE_LINKED_ATOMIC_CONTRACT_SCHEMA,
            OCCURRENCE_LINKED_ATOMIC_REPORT_SCHEMA,
            "shohin-ettr-parallel-terminal-state-metric-v7",
        )
    if token_native_command_mask:
        if residual_edits or not atomic_edits or not lexical_command:
            raise ParallelTerminalStatePilotError(
                "syntax-routed terminal-state architecture schema differs"
            )
        return (
            SYNTAX_ROUTED_ATOMIC_CONTRACT_SCHEMA,
            SYNTAX_ROUTED_ATOMIC_REPORT_SCHEMA,
            "shohin-ettr-parallel-terminal-state-metric-v6",
        )
    if lexical_command:
        if residual_edits or not atomic_edits:
            raise ParallelTerminalStatePilotError(
                "lexical terminal-state architecture schema differs"
            )
        return (
            LEXICAL_ATOMIC_CONTRACT_SCHEMA,
            LEXICAL_ATOMIC_REPORT_SCHEMA,
            "shohin-ettr-parallel-terminal-state-metric-v5",
        )
    if atomic_edits:
        if residual_edits:
            raise ParallelTerminalStatePilotError(
                "terminal-state architecture schema differs"
            )
        return (
            ATOMIC_CONTRACT_SCHEMA,
            ATOMIC_REPORT_SCHEMA,
            "shohin-ettr-parallel-terminal-state-metric-v4",
        )
    if residual_edits:
        return (
            CONTRACT_SCHEMA,
            REPORT_SCHEMA,
            "shohin-ettr-parallel-terminal-state-metric-v3",
        )
    return (
        CAUSAL_DELTA_CONTRACT_SCHEMA,
        CAUSAL_DELTA_REPORT_SCHEMA,
        "shohin-ettr-parallel-terminal-state-metric-v2",
    )


def _state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        raw = value.detach().cpu().contiguous().view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def _evaluate_interfaces(
    compiler,
    model,
    *,
    stream,
    packet_index,
    device,
    data_seed: int,
    max_batches: int,
) -> dict[str, object]:
    compiler.eval()
    counts = {"oracle": {}, "autonomous": {}}
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
            command_hidden = model._encode_to_stage(
                batch.episodes.command.tokens,
                pos=0,
            )
            command_lexical = (
                model.base.tok(batch.episodes.command.tokens)
                if compiler.lexical_command
                else None
            )
            for source in counts:
                initial = _training_initial_state(
                    model,
                    batch,
                    source=source,
                    dtype=next(compiler.parameters()).dtype,
                )
                terminal = compiler(
                    initial,
                    command_hidden=command_hidden,
                    command_lexical=command_lexical,
                    command_tokens=(
                        batch.episodes.command.tokens
                        if compiler.token_native_command_mask
                        else None
                    ),
                    command_attention_mask=(
                        batch.episodes.command.attention_mask.bool()
                    ),
                    steps=batch.transaction_targets.opcode.shape[1],
                    hard=True,
                )
                _merge_counts(
                    counts[source],
                    _packet_batch_counts(
                        terminal,
                        batch.terminal_packet_targets,
                    ),
                )
        observed += 1
    if observed != max_batches:
        raise ParallelTerminalStatePilotError(
            "terminal-state development split is too short"
        )
    return {
        "batches": observed,
        "terminal_packet": {
            source: _summarize_counts(values) for source, values in counts.items()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    contract_schema, report_schema, metric_schema = _run_schemas(
        args.residual_edits,
        args.atomic_edits,
        args.lexical_command,
        args.token_native_command_mask,
        args.token_native_occurrence_command,
        args.token_native_syntax_graph_command,
        args.token_native_declaration_binding_command,
        args.token_native_operation_recurrence_command,
        args.token_native_operation_state_command,
        args.factorized_operation_effect_command,
        args.operation_effect_set_command,
        args.operation_effect_role_anchors,
        args.operation_effect_cardinality_gate,
        args.operation_effect_write_link_rails,
        args.operation_effect_rail_local_loss,
        args.operation_effect_post_write_link_binding,
        args.operation_effect_family_gate,
        args.operation_effect_family_island,
        args.operation_effect_family_state_binding,
    )
    if (
        contract_schema == OPERATION_STATE_BOUND_FAMILY_JOINT_CONTRACT_SCHEMA
        and args.terminal_warm_start_dir is None
    ):
        raise ParallelTerminalStatePilotError(
            "state-bound family joint release requires an exact warm start"
        )
    if (
        args.terminal_warm_start_dir is not None
        and contract_schema
        not in {
            OPERATION_FAMILY_GATE_CONTRACT_SCHEMA,
            OPERATION_STATE_BOUND_FAMILY_JOINT_CONTRACT_SCHEMA,
        }
    ):
        raise ParallelTerminalStatePilotError(
            "terminal warm start is not valid for this release schema"
        )
    if not torch.cuda.is_available():
        raise ParallelTerminalStatePilotError("terminal-state pilot requires CUDA")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    is_h100 = "H100" in torch.cuda.get_device_name(device).upper()
    if args.required_device_class == "h100" and not is_h100:
        raise ParallelTerminalStatePilotError("terminal-state pilot requires an H100")

    stream = ETTRV3StreamingRelease(
        args.release_root,
        expected_release_sha256=args.release_sha256,
        data_root=args.data_root,
        tokenizer_path=args.tokenizer,
    )
    source_verification = stream.verify_source_shards()
    model, _joint_payload, provenance, _joint_contract = _strict_load_joint_model(
        args,
        device=device,
    )
    reader, _compiler_contract, reader_parameters, replacement_parameters = (
        _load_compiler(
            args,
            model=model,
            stream=stream,
            device=device,
        )
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    oracle_executor = model.reactor
    removed_reactor_parameters = sum(
        parameter.numel() for parameter in model.reactor.parameters()
    )
    torch.manual_seed(args.architecture_seed)
    torch.cuda.manual_seed_all(args.architecture_seed)
    compiler_class = (
        OperationStateBoundFamilyGatedWriteLinkCompiler
        if args.operation_effect_family_state_binding
        else OperationFamilyGatedWriteLinkCompiler
        if args.operation_effect_family_gate
        else OperationPostWriteLinkRailCompiler
        if args.operation_effect_post_write_link_binding
        else OperationWriteLinkRailCompiler
        if args.operation_effect_write_link_rails
        else OperationEffectSetCompiler
        if args.operation_effect_set_command
        else FactorizedOperationStateTransitionCompiler
        if args.factorized_operation_effect_command
        else OperationStateTransitionCompiler
        if args.token_native_operation_state_command
        else ParallelTerminalStateCompiler
    )
    compiler = compiler_class(
        model.config,
        width=args.width,
        layers=args.layers,
        num_heads=args.num_heads,
        relation_width=args.relation_width,
        residual_edits=args.residual_edits,
        atomic_edits=args.atomic_edits,
        lexical_command=args.lexical_command,
        token_native_command_mask=args.token_native_command_mask,
        cover_verified_command_mask=args.cover_verified_command_mask,
        token_native_occurrence_command=(args.token_native_occurrence_command),
        token_native_syntax_graph_command=(args.token_native_syntax_graph_command),
        token_native_declaration_binding_command=(
            args.token_native_declaration_binding_command
        ),
        token_native_operation_recurrence_command=(
            args.token_native_operation_recurrence_command
        ),
        token_native_codebook_ids=(
            stream.codec.codebook.token_ids if args.token_native_command_mask else None
        ),
        token_native_codebook_atoms=(
            stream.codec.codebook.atoms if args.cover_verified_command_mask else None
        ),
        token_native_vocab_size=(
            model.base.cfg.vocab_size if args.token_native_command_mask else None
        ),
        **(
            {
                "maximum_effect_roles": ROLE_ANCHORED_EFFECT_ROLES,
            }
            if args.operation_effect_write_link_rails
            else {
                "maximum_effect_roles": ROLE_ANCHORED_EFFECT_ROLES,
                "maximum_effects": ROLE_ANCHORED_EFFECT_SLOTS,
                "public_role_anchors": True,
                "explicit_effect_cardinality": (args.operation_effect_cardinality_gate),
            }
            if args.operation_effect_role_anchors
            else {"public_role_anchors": False}
            if args.operation_effect_set_command
            else {}
        ),
    ).to(device=device, dtype=next(model.parameters()).dtype)
    warm_start_receipt = None
    if args.terminal_warm_start_dir is not None:
        warm_start_receipt = _load_terminal_warm_start(
            compiler,
            args.terminal_warm_start_dir,
            args.terminal_warm_start_sha256s_sha256,
            successor_schema=contract_schema,
        )
    compiler_parameters = sum(parameter.numel() for parameter in compiler.parameters())
    complete_parameters = (
        replacement_parameters - removed_reactor_parameters + compiler_parameters
    )
    optimizer = torch.optim.AdamW(
        compiler.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )
    packet_index = ETTRDiskPacketSufficiencyIndex(stream.packet_index_root)
    objective_config = ETTRObjectiveConfig(vocab_size=model.base.cfg.vocab_size)
    try:
        args.output.mkdir(mode=0o700)
        initial_sha256 = _module_sha256(
            compiler,
            args.output / "terminal-compiler-initial.safetensors",
        )
        if (
            warm_start_receipt is not None
            and initial_sha256 != warm_start_receipt["compiler_sha256"]
        ):
            raise ParallelTerminalStatePilotError(
                "terminal warm-start preservation differs"
            )
        before_interface = _evaluate_interfaces(
            compiler,
            model,
            stream=stream,
            packet_index=packet_index,
            device=device,
            data_seed=args.data_seed,
            max_batches=args.eval_batches,
        )
        model.reactor = ParallelTerminalStateReactor(compiler, model.config)
        deployed_parameters = (
            sum(parameter.numel() for parameter in model.parameters())
            - sum(parameter.numel() for parameter in model.query_reader.parameters())
            + reader_parameters
        )
        if deployed_parameters != complete_parameters:
            raise ParallelTerminalStatePilotError(
                "terminal-state deployed parameter count differs"
            )
        before_end_to_end = _evaluate(
            reader,
            model,
            stream=stream,
            packet_index=packet_index,
            device=device,
            data_seed=args.data_seed,
            max_batches=args.eval_batches,
        )
        contract = {
            "architecture": {
                "causal_rectangle_delta_credit": True,
                "atomic_typed_edits": args.atomic_edits,
                "fixed_atomic_edit_algebra": args.atomic_edits,
                "lexical_command_rail": args.lexical_command,
                "token_native_command_mask": (args.token_native_command_mask),
                "cover_verified_command_mask": (args.cover_verified_command_mask),
                "token_native_occurrence_command": (
                    args.token_native_occurrence_command
                ),
                "token_native_syntax_graph_command": (
                    args.token_native_syntax_graph_command
                ),
                "token_native_declaration_binding_command": (
                    args.token_native_declaration_binding_command
                ),
                "token_native_operation_recurrence_command": (
                    args.token_native_operation_recurrence_command
                ),
                "token_native_operation_state_command": (
                    args.token_native_operation_state_command
                ),
                "factorized_operation_effect_command": (
                    args.factorized_operation_effect_command
                ),
                "operation_effect_set_command": (args.operation_effect_set_command),
                "operation_effect_role_anchors": (args.operation_effect_role_anchors),
                "operation_effect_cardinality_gate": (
                    args.operation_effect_cardinality_gate
                ),
                "operation_effect_write_link_rails": (
                    args.operation_effect_write_link_rails
                ),
                "operation_effect_rail_local_loss": (
                    args.operation_effect_rail_local_loss
                ),
                "operation_effect_post_write_link_binding": (
                    args.operation_effect_post_write_link_binding
                ),
                "operation_effect_family_gate": (args.operation_effect_family_gate),
                "operation_effect_family_island": (
                    args.operation_effect_family_island
                ),
                "operation_effect_family_state_binding": (
                    args.operation_effect_family_state_binding
                ),
                "operation_effect_slots": (
                    compiler.maximum_effects
                    if isinstance(
                        compiler,
                        (OperationEffectSetCompiler, OperationWriteLinkRailCompiler),
                    )
                    else 0
                ),
                "operation_effect_role_count": (
                    compiler.effect_role_count
                    if isinstance(
                        compiler,
                        (OperationEffectSetCompiler, OperationWriteLinkRailCompiler),
                    )
                    else 0
                ),
                "operation_effect_motors_per_role": (
                    compiler.effect_motors_per_role
                    if isinstance(
                        compiler,
                        (OperationEffectSetCompiler, OperationWriteLinkRailCompiler),
                    )
                    else 0
                ),
                "token_native_codebook_sha256": (
                    stream.codec.codebook_sha256
                    if args.token_native_command_mask
                    else None
                ),
                "direct_terminal_quotient": True,
                "layers": args.layers,
                "no_query_input": True,
                "no_transaction_trace_claim": True,
                "operation_boundary_labels_training_only": (
                    args.token_native_operation_state_command
                ),
                "operation_public_ledger_deferred": (
                    args.operation_effect_write_link_rails
                ),
                "num_heads": args.num_heads,
                "relation_width": args.relation_width,
                "removed_recurrent_policy_parameters": (removed_reactor_parameters),
                "sparse_residual_edits": args.residual_edits,
                "seed": args.architecture_seed,
                "typed_hard_state_constraints": True,
                "width": args.width,
            },
            "compiler_contract_sha256": args.compiler_contract_sha256,
            "compiler_parameters": compiler_parameters,
            "compiler_sha256": args.compiler_sha256,
            "complete_system_parameters": complete_parameters,
            "data_seed": args.data_seed,
            "eval_batches": args.eval_batches,
            "gradient_clip": args.gradient_clip,
            "joint_model_sha256": args.joint_model_sha256,
            "joint_run_contract_sha256": args.joint_run_contract_sha256,
            "learning_rate": args.learning_rate,
            "objective": {
                "binary": "class-balanced-brier",
                "atomic_action_weight": (
                    args.atomic_action_weight if args.atomic_edits else 0.0
                ),
                "atomic_actions": (
                    "operation-boundary-and-final-class-balanced-state-difference"
                    if args.token_native_operation_state_command
                    else "canonical-class-balanced-state-difference"
                    if args.atomic_edits
                    else None
                ),
                "causal_delta_weight": args.causal_delta_weight,
                "causal_pairing": "complete-2x2-terminal-state-edges",
                "categorical": "categorical-brier",
                "target": "query-independent-terminal-packet",
                "operation_boundary_state_credit": (
                    args.token_native_operation_state_command
                ),
                "global_sparse_effect_cardinality": (
                    args.factorized_operation_effect_command
                ),
                "unordered_typed_effect_set": (
                    args.operation_effect_set_command
                    or args.operation_effect_write_link_rails
                ),
                "effect_set_matching": (
                    "canonical-write-link-rail-local"
                    if args.operation_effect_rail_local_loss
                    else "detached-sinkhorn-typed-bipartite"
                    if (
                        args.operation_effect_set_command
                        or args.operation_effect_write_link_rails
                    )
                    else None
                ),
                "effect_set_role_anchors": (
                    "shared-public-operation-root-and-semantic-children"
                    if args.operation_effect_write_link_rails
                    else "public-operation-root-and-semantic-children-five-motors-per-role"
                    if args.operation_effect_role_anchors
                    else None
                ),
                "effect_set_role_capacity": (
                    {
                        "effect_slots": WRITE_LINK_RAIL_EFFECT_SLOTS,
                        "link_slots": LINK_RAIL_EFFECT_SLOTS,
                        "maximum_roles": ROLE_ANCHORED_EFFECT_ROLES,
                        "write_slots": WRITE_RAIL_EFFECT_SLOTS,
                    }
                    if args.operation_effect_write_link_rails
                    else {
                        "effect_slots": ROLE_ANCHORED_EFFECT_SLOTS,
                        "maximum_roles": ROLE_ANCHORED_EFFECT_ROLES,
                        "motors_per_role": (ROLE_ANCHORED_EFFECT_MOTORS_PER_ROLE),
                    }
                    if args.operation_effect_role_anchors
                    else None
                ),
                "effect_set_cardinality": (
                    "operation-family-gated-separate-counts-plus-top-k"
                    if args.operation_effect_family_gate
                    else "separate-write-link-count-heads-plus-top-k-per-rail"
                    if args.operation_effect_write_link_rails
                    else "explicit-total-count-plus-top-k-motor-activity"
                    if args.operation_effect_cardinality_gate
                    else None
                ),
                "write_link_typed_rails": (args.operation_effect_write_link_rails),
                "operation_public_ledger": (
                    "command-cursor-outcome-disposition-final-suffix"
                    if args.operation_effect_write_link_rails
                    else None
                ),
                "operation_effect_family_control": (
                    "role-state-bilinear-none-write-link-before-rail-release"
                    if args.operation_effect_family_state_binding
                    else "exclusive-none-write-link-before-rail-release"
                    if args.operation_effect_family_gate
                    else None
                ),
                "operation_effect_family_training": (
                    "oracle-state-family-only-payload-rails-bypassed"
                    if args.operation_effect_family_island
                    else "joint-family-count-payload"
                    if args.operation_effect_family_gate
                    else None
                ),
                "operation_effect_family_binding": (
                    "multihead-role-query-state-key-value-product"
                    if args.operation_effect_family_state_binding
                    else None
                ),
                "write_link_binding_state": (
                    "post-write-differentiable-state"
                    if args.operation_effect_post_write_link_binding
                    else "pre-operation-state"
                    if args.operation_effect_write_link_rails
                    else None
                ),
            },
            "protected_checkpoint_sha256": provenance.checkpoint_sha256,
            "reader_parameters": reader_parameters,
            "release_file_sha256": args.release_sha256,
            "required_device_class": args.required_device_class,
            "schema": contract_schema,
            "source_commit": args.source_commit,
            "start_position": args.start_position,
            "training_initial_state": args.training_initial_state,
            "updates": args.updates,
            "warm_start": warm_start_receipt,
        }
        contract_sha256 = _write_no_replace(
            args.output / "pilot-contract.json",
            _canonical_bytes(contract),
        )
        _write_no_replace(args.output / "train.jsonl", b"", mode=0o600)

        iterator = iter_batches_with_query_specs(
            stream,
            "train",
            epoch=0,
            seed=args.data_seed,
            start_position=args.start_position,
        )
        compiler.train()
        last_loss = None
        last_position = args.start_position
        for update in range(1, args.updates + 1):
            try:
                last_position, cpu_batch, _cpu_specs = next(iterator)
            except StopIteration as exc:
                raise ParallelTerminalStatePilotError(
                    "terminal-state train stream exhausted"
                ) from exc
            packet_index.verify_train((cpu_batch,))
            batch = move_continuation_batch(cpu_batch, device)
            batch.validate(model.config, objective_config)
            initial = _training_initial_state(
                model,
                batch,
                source=args.training_initial_state,
                dtype=next(compiler.parameters()).dtype,
            )
            target = packet_targets_to_state(
                batch.terminal_packet_targets,
                model.config,
                step=batch.transaction_targets.opcode.shape[1],
                dtype=next(compiler.parameters()).dtype,
            )
            with torch.no_grad():
                command_hidden = model._encode_to_stage(
                    batch.episodes.command.tokens,
                    pos=0,
                )
                command_lexical = (
                    model.base.tok(batch.episodes.command.tokens)
                    if args.lexical_command
                    else None
                )
            operation_targets = (
                oracle_operation_boundary_states(
                    oracle_executor,
                    initial,
                    batch.transaction_targets,
                    defer_public_ledger=(
                        args.operation_effect_write_link_rails
                    ),
                )
                if args.token_native_operation_state_command
                else None
            )
            atomic_targets = None
            if args.atomic_edits and not args.token_native_operation_state_command:
                atomic_targets = derive_atomic_edit_targets(initial, target)
                if update == 1:
                    verify_atomic_edit_reconstruction(
                        compiler,
                        initial,
                        target,
                        atomic_targets,
                        steps=batch.transaction_targets.opcode.shape[1],
                    )
            optimizer.zero_grad(set_to_none=True)
            with _precision_context(is_h100):
                operation_prefix_loss = target.active.float().sum() * 0.0
                if args.operation_effect_family_island:
                    if (
                        not isinstance(
                            compiler,
                            OperationFamilyGatedWriteLinkCompiler,
                        )
                        or operation_targets is None
                        or command_lexical is None
                    ):
                        raise ParallelTerminalStatePilotError(
                            "operation family island inputs are absent"
                        )
                    family_probabilities, family_mask = (
                        compiler.forward_with_oracle_operation_families(
                            initial,
                            operation_targets.states,
                            operation_targets.mask,
                            command_hidden=command_hidden.detach(),
                            command_lexical=command_lexical.detach(),
                            command_tokens=batch.episodes.command.tokens,
                            command_attention_mask=(
                                batch.episodes.command.attention_mask.bool()
                            ),
                            steps=batch.transaction_targets.opcode.shape[1],
                            hard=False,
                        )
                    )
                    (
                        atomic_action_loss,
                        atomic_action_parts,
                        atomic_action_counts,
                    ) = operation_family_island_objective(
                        family_probabilities,
                        family_mask,
                        operation_targets,
                        initial,
                        slot_mask=batch.terminal_packet_targets.slot_mask,
                        relation_mask=batch.terminal_packet_targets.relation_mask,
                    )
                    predicted = initial
                elif args.token_native_operation_state_command:
                    if (
                        not isinstance(
                            compiler,
                            OperationStateTransitionCompiler,
                        )
                        or operation_targets is None
                        or command_lexical is None
                    ):
                        raise ParallelTerminalStatePilotError(
                            "operation-state training inputs are absent"
                        )
                    predicted, operation_trace = compiler.forward_with_operation_states(
                        initial,
                        command_hidden=command_hidden.detach(),
                        command_lexical=command_lexical.detach(),
                        command_tokens=batch.episodes.command.tokens,
                        command_attention_mask=(
                            batch.episodes.command.attention_mask.bool()
                        ),
                        steps=batch.transaction_targets.opcode.shape[1],
                        hard=False,
                    )
                    (
                        operation_prefix_loss,
                        atomic_action_loss,
                        atomic_action_parts,
                        atomic_action_counts,
                    ) = operation_boundary_objective(
                        compiler,
                        operation_trace,
                        operation_targets,
                        initial,
                        target,
                        slot_mask=batch.terminal_packet_targets.slot_mask,
                        relation_mask=(batch.terminal_packet_targets.relation_mask),
                        verify_reconstruction=update == 1,
                        rail_local_loss=args.operation_effect_rail_local_loss,
                    )
                elif args.atomic_edits:
                    predicted, atomic_edits = compiler.forward_with_atomic_edits(
                        initial,
                        command_hidden=command_hidden.detach(),
                        command_lexical=(
                            None
                            if command_lexical is None
                            else command_lexical.detach()
                        ),
                        command_tokens=(
                            batch.episodes.command.tokens
                            if args.token_native_command_mask
                            else None
                        ),
                        command_attention_mask=(
                            batch.episodes.command.attention_mask.bool()
                        ),
                        steps=batch.transaction_targets.opcode.shape[1],
                        hard=False,
                    )
                    if atomic_targets is None:
                        raise ParallelTerminalStatePilotError(
                            "atomic typed-edit targets are absent"
                        )
                    (
                        atomic_action_loss,
                        atomic_action_parts,
                        atomic_action_counts,
                    ) = atomic_typed_edit_loss(
                        atomic_edits,
                        atomic_targets,
                        slot_mask=batch.terminal_packet_targets.slot_mask,
                        relation_mask=(batch.terminal_packet_targets.relation_mask),
                    )
                else:
                    predicted = compiler(
                        initial,
                        command_hidden=command_hidden.detach(),
                        command_attention_mask=(
                            batch.episodes.command.attention_mask.bool()
                        ),
                        steps=batch.transaction_targets.opcode.shape[1],
                        hard=False,
                    )
                    atomic_action_loss = predicted.active.float().sum() * 0.0
                    atomic_action_parts = {}
                    atomic_action_counts = {}
                state_loss, state_parts = _state_brier(
                    predicted,
                    target,
                    slot_mask=batch.terminal_packet_targets.slot_mask,
                    relation_mask=(batch.terminal_packet_targets.relation_mask),
                )
                causal_delta_loss, delta_parts, changed_counts = (
                    causal_terminal_delta_brier(
                        predicted,
                        target,
                        rectangle_rows=batch.causal_rectangles.rows,
                        slot_mask=batch.terminal_packet_targets.slot_mask,
                        relation_mask=(batch.terminal_packet_targets.relation_mask),
                    )
                )
                loss = (
                    atomic_action_loss
                    if args.operation_effect_family_island
                    else (
                        state_loss
                        + args.causal_delta_weight * causal_delta_loss
                        + args.atomic_action_weight * atomic_action_loss
                        + operation_prefix_loss
                    )
                )
            if not bool(torch.isfinite(loss)):
                raise ParallelTerminalStatePilotError(
                    "terminal-state loss is nonfinite"
                )
            last_loss = float(loss.detach().cpu())
            loss.backward()
            if args.operation_effect_family_island:
                forbidden_gradients = tuple(
                    name
                    for name, parameter in compiler.named_parameters()
                    if name.startswith(("write_rail.", "link_rail."))
                    or name.startswith(("write_", "link_"))
                    if parameter.grad is not None
                )
                if forbidden_gradients:
                    raise ParallelTerminalStatePilotError(
                        "operation family island reached payload rails"
                    )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                compiler.parameters(),
                args.gradient_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
            if update % args.log_every == 0 or update == args.updates:
                metric = {
                    "gradient_norm_pre_clip": float(
                        gradient_norm.detach().float().cpu()
                    ),
                    "loss": last_loss,
                    "causal_delta_loss": float(causal_delta_loss.detach().cpu()),
                    "causal_delta_parts": {
                        name: float(value.detach().cpu())
                        for name, value in delta_parts.items()
                    },
                    "atomic_action_loss": float(atomic_action_loss.detach().cpu()),
                    "atomic_action_parts": {
                        name: float(value.detach().cpu())
                        for name, value in atomic_action_parts.items()
                    },
                    "atomic_action_counts": atomic_action_counts,
                    "operation_prefix_loss": float(
                        operation_prefix_loss.detach().cpu()
                    ),
                    "changed_coordinates": changed_counts,
                    "state_loss": float(state_loss.detach().cpu()),
                    "state_parts": {
                        name: float(value.detach().cpu())
                        for name, value in state_parts.items()
                    },
                    "position": last_position,
                    "schema": metric_schema,
                    "update": update,
                }
                with (args.output / "train.jsonl").open("ab", buffering=0) as log:
                    log.write(_canonical_bytes(metric))
        os.chmod(args.output / "train.jsonl", 0o400)

        compiler.eval()
        after_interface = _evaluate_interfaces(
            compiler,
            model,
            stream=stream,
            packet_index=packet_index,
            device=device,
            data_seed=args.data_seed,
            max_batches=args.eval_batches,
        )
        after_end_to_end = _evaluate(
            reader,
            model,
            stream=stream,
            packet_index=packet_index,
            device=device,
            data_seed=args.data_seed,
            max_batches=args.eval_batches,
        )
        final_sha256 = _module_sha256(
            compiler,
            args.output / "terminal-compiler-final.safetensors",
        )
        report = {
            "after_end_to_end": after_end_to_end,
            "after_interface": after_interface,
            "before_end_to_end": before_end_to_end,
            "before_interface": before_interface,
            "contract_sha256": contract_sha256,
            "final_compiler_sha256": final_sha256,
            "initial_compiler_sha256": initial_sha256,
            "last_loss": last_loss,
            "last_position": last_position,
            "protected_checkpoint_sha256": provenance.checkpoint_sha256,
            "runtime_precision": str(next(compiler.parameters()).dtype),
            "schema": report_schema,
            "source_verification": source_verification,
            "status": "pass",
            "updates_completed": args.updates,
        }
        _write_no_replace(
            args.output / "report.json",
            _canonical_bytes(report),
        )
        names = (
            "pilot-contract.json",
            "report.json",
            "terminal-compiler-final.safetensors",
            "terminal-compiler-initial.safetensors",
            "train.jsonl",
        )
        _write_no_replace(
            args.output / "SHA256SUMS",
            "".join(
                f"{_sha256_file(args.output / name)}  {name}\n" for name in names
            ).encode("ascii"),
        )
        for path in args.output.iterdir():
            os.chmod(path, 0o400)
        os.chmod(args.output, 0o500)
    except BaseException:
        if args.output.exists():
            shutil.rmtree(args.output, ignore_errors=True)
        raise
    finally:
        packet_index.close()
    print(
        json.dumps(
            {
                "complete_system_parameters": complete_parameters,
                "compiler_parameters": compiler_parameters,
                "final_compiler_state_sha256": _state_sha256(compiler),
                "output": str(args.output),
                "status": "pass",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
