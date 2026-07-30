"""Executable, no-fit arm mechanics for R12-ETTR-IL-v2.

This module contains only model-side causal transformations and read-only
accounting.  It deliberately has no optimizer, update loop, checkpoint,
filesystem writer, launcher, or scheduler surface.

The static operation convention is an exact count of scalar products whose
result is connected to the loss.  It is not a vendor-kernel FLOP estimate.
Every trainable matrix use contributes one scalar product per matrix element
and input position.  The A4 equalizer contributes the exact missing number of
fixed orthogonal scalar products in both the forward graph and its input
gradient graph.  This gives a deterministic, implementation-inspectable
operator ledger without pretending that nonlinear kernels have portable FLOP
definitions.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Literal, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from endogenous_typed_theory_reactor import (
    EndogenousTheoryCompiler,
    EndogenousTypedTheoryReactorGPT,
    GenericTransactionReactor,
    ReactorTrace,
    SourceDeletedQueryReader,
    TheoryReactorConfig,
    TheoryReactorError,
    TransactionPolicy,
    TypedTheoryState,
    _disposition_probabilities,
    _hard_one_hot,
    validate_state,
)
from ettr_data_contract import ETTRCausalRectangle
from ettr_objectives import (
    ETTRObjectiveBatch,
    ETTRPacketTargets,
    ETTRTransactionTargets,
    ETTRVariantAlignment,
)


_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE = _ROOT / "pipeline"
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from ettr_il_v2_controls import (  # noqa: E402
    BindingDerangement,
    canonical_json_bytes as control_json_bytes,
)


PROTOCOL = "R12-ETTR-IL-v2"
ARM_MECHANICS_SCHEMA = "r12-ettr-il-v2-arm-mechanics-v1"
OPERATION_LEDGER_SCHEMA = "r12-ettr-il-v2-static-operation-ledger-v1"
OPERATION_CONVENTION = (
    "loss_connected_linear_scalar_products_forward_and_input_gradient_v1"
)

ArmName = Literal[
    "treatment",
    "state_reset",
    "binding_deranged",
    "query_only",
    "dense_state",
]

PRIMARY_ARMS: tuple[ArmName, ...] = (
    "treatment",
    "state_reset",
    "binding_deranged",
    "query_only",
    "dense_state",
)

WORLD_POSITIONS = 192
COMMAND_POSITIONS = 96
QUERY_POSITIONS = 48
TRANSACTION_POSITIONS = 64
ROWS_PER_MICROSTEP = 32
MICROSTEPS_PER_UPDATE = 4
ROWS_PER_UPDATE = ROWS_PER_MICROSTEP * MICROSTEPS_PER_UPDATE
REACTOR_ROW_CALLS_PER_ROW = 3
READER_CALLS_PER_ROW = 3

PRODUCTION_ARCHITECTURE_PARAMETERS = 67_697_771
PRODUCTION_ARCHITECTURE_MUON = 67_024_896
PRODUCTION_ARCHITECTURE_ADAMW = 672_875
PRODUCTION_COMPLETE_PARAMETERS = 192_779_435
PRODUCTION_REACTOR_PARAMETERS = 29_757_217
PRODUCTION_RETAINED_REACTOR_PARAMETERS = 2_454_305
PRODUCTION_REMOVED_REACTOR_PARAMETERS = 27_302_912
PRODUCTION_REMOVED_MUON = 27_262_976
PRODUCTION_REMOVED_ADAMW = 39_936

DENSE_HIDDEN_WIDTH = 1_241
DENSE_MLP_WIDTH = 4_123
DENSE_ADAPTER_ROWS = 19
DENSE_ADAPTER_COLUMNS = 421
DENSE_ADAPTER_PARAMETERS = 7_999
DENSE_EQUALIZER_COEFFICIENT = 2.0**-20

_RETAINED_REACTOR_PARAMETERS = (
    "control_seed",
    "step_embedding",
    "type_embedding",
    "value_embedding",
    "active_projection",
    "root_projection",
    "status_projection",
    "command_projection",
    "command_norm",
    "command_attention",
    "output_norm",
    "opcode_head",
    "source_query",
    "target_query",
    "slot_key",
    "relation_head",
    "type_head",
    "value_head",
)
_REMOVED_PREFIXES = ("relation_projection.", "core.")


class ETTRILV2ArmError(ValueError):
    """An arm transformation or equality gate failed closed."""


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class ETTRILV2ArmConfig:
    """One immutable arm's only permitted causal differences."""

    arm: ArmName
    reactor_mode: str
    target_mode: str
    reader_mode: str
    hard_transactions: bool = True
    reactor_positions: int = TRANSACTION_POSITIONS

    def validate(self) -> None:
        expected = {
            "treatment": ("native_recurrence", "factual", "sealed_packet"),
            "state_reset": (
                "compiler_state_reset_each_position",
                "factual",
                "sealed_packet",
            ),
            "binding_deranged": (
                "native_recurrence",
                "bundle_deranged",
                "sealed_packet",
            ),
            "query_only": (
                "native_recurrence",
                "factual",
                "canonical_empty_packet",
            ),
            "dense_state": (
                "parameter_matched_dense_recurrence",
                "factual",
                "sealed_packet",
            ),
        }
        if (
            self.arm not in expected
            or (
                self.reactor_mode,
                self.target_mode,
                self.reader_mode,
            )
            != expected[self.arm]
            or self.hard_transactions is not True
            or self.reactor_positions != TRANSACTION_POSITIONS
        ):
            raise ETTRILV2ArmError("arm configuration differs")

    def sha256(self) -> str:
        self.validate()
        return _sha256(
            {
                "config": asdict(self),
                "protocol": PROTOCOL,
                "schema": ARM_MECHANICS_SCHEMA,
            }
        )


ARM_CONFIGS: Mapping[ArmName, ETTRILV2ArmConfig] = {
    arm: ETTRILV2ArmConfig(
        arm=arm,
        reactor_mode={
            "treatment": "native_recurrence",
            "state_reset": "compiler_state_reset_each_position",
            "binding_deranged": "native_recurrence",
            "query_only": "native_recurrence",
            "dense_state": "parameter_matched_dense_recurrence",
        }[arm],
        target_mode=("bundle_deranged" if arm == "binding_deranged" else "factual"),
        reader_mode=(
            "canonical_empty_packet" if arm == "query_only" else "sealed_packet"
        ),
    )
    for arm in PRIMARY_ARMS
}


@dataclass(frozen=True, slots=True)
class ParameterLedger:
    """Exact optimizer-family and active-parameter ownership."""

    muon: int
    adamw: int
    unique_trainable: int
    active_parameter_names_sha256: str

    def validate(self) -> None:
        if (
            self.muon < 1
            or self.adamw < 1
            or self.unique_trainable != self.muon + self.adamw
            or len(self.active_parameter_names_sha256) != 64
        ):
            raise ETTRILV2ArmError("parameter ledger differs")


@dataclass(frozen=True, slots=True)
class TokenPositionLedger:
    """Exact, arm-invariant encoded and supervised position geometry."""

    world_positions_per_row: int
    command_calls_per_row: int
    command_positions_per_call: int
    query_calls_per_row: int
    query_positions_per_call: int
    encoded_positions_per_row: int
    supervised_positions_per_row: int
    rows_per_microstep: int
    microsteps_per_update: int
    encoded_positions_per_update: int
    supervised_positions_per_update: int

    @classmethod
    def production(cls) -> "TokenPositionLedger":
        encoded = WORLD_POSITIONS + 2 * COMMAND_POSITIONS + 3 * QUERY_POSITIONS
        supervised = WORLD_POSITIONS - 1 + COMMAND_POSITIONS - 1 + QUERY_POSITIONS - 1
        return cls(
            world_positions_per_row=WORLD_POSITIONS,
            command_calls_per_row=2,
            command_positions_per_call=COMMAND_POSITIONS,
            query_calls_per_row=3,
            query_positions_per_call=QUERY_POSITIONS,
            encoded_positions_per_row=encoded,
            supervised_positions_per_row=supervised,
            rows_per_microstep=ROWS_PER_MICROSTEP,
            microsteps_per_update=MICROSTEPS_PER_UPDATE,
            encoded_positions_per_update=encoded * ROWS_PER_UPDATE,
            supervised_positions_per_update=supervised * ROWS_PER_UPDATE,
        )

    def validate(self) -> None:
        expected = type(self).production()
        if self != expected:
            raise ETTRILV2ArmError("token-position ledger differs")


@dataclass(frozen=True, slots=True)
class EqualizerStepPlan:
    """Exact factorization of one scalar-product remainder."""

    scalar_products: int
    carrier_length: int
    width: int
    full_replays: int
    full_rows: int
    output_columns: int
    partial_dot: int

    @classmethod
    def build(
        cls,
        scalar_products: int,
        *,
        carrier_length: int,
        width: int,
    ) -> "EqualizerStepPlan":
        if scalar_products < 0 or carrier_length < 1 or width < 1:
            raise ETTRILV2ArmError("equalizer remainder is inadmissible")
        full_unit = carrier_length * width * width
        full_replays, remainder = divmod(scalar_products, full_unit)
        full_rows, remainder = divmod(remainder, width * width)
        output_columns, partial_dot = divmod(remainder, width)
        result = cls(
            scalar_products=scalar_products,
            carrier_length=carrier_length,
            width=width,
            full_replays=full_replays,
            full_rows=full_rows,
            output_columns=output_columns,
            partial_dot=partial_dot,
        )
        if result.reconstructed_scalar_products != scalar_products:
            raise AssertionError("equalizer factorization differs")
        return result

    @property
    def reconstructed_scalar_products(self) -> int:
        return (
            self.full_replays * self.carrier_length * self.width * self.width
            + self.full_rows * self.width * self.width
            + self.output_columns * self.width
            + self.partial_dot
        )


@dataclass(frozen=True, slots=True)
class StaticOperationLedger:
    """Exact competitive-path scalar-product ledger for one update."""

    schema: str
    convention: str
    arm: ArmName
    rows_per_update: int
    reactor_row_calls_per_row: int
    reactor_positions: int
    native_replacement_forward_scalar_products: int
    arm_replacement_forward_scalar_products: int
    equalizer_forward_scalar_products: int
    total_forward_scalar_products: int
    arm_replacement_backward_scalar_products: int
    equalizer_backward_scalar_products: int
    total_backward_scalar_products: int
    total_training_scalar_products: int
    common_path_signature: str
    equalizer_plan_sha256: str

    def validate(self) -> None:
        if (
            self.schema != OPERATION_LEDGER_SCHEMA
            or self.convention != OPERATION_CONVENTION
            or self.arm not in PRIMARY_ARMS
            or self.rows_per_update != ROWS_PER_UPDATE
            or self.reactor_row_calls_per_row != REACTOR_ROW_CALLS_PER_ROW
            or self.reactor_positions != TRANSACTION_POSITIONS
            or self.native_replacement_forward_scalar_products < 1
            or self.arm_replacement_forward_scalar_products < 1
            or self.equalizer_forward_scalar_products < 0
            or self.total_forward_scalar_products
            != self.arm_replacement_forward_scalar_products
            + self.equalizer_forward_scalar_products
            or self.total_forward_scalar_products
            != self.native_replacement_forward_scalar_products
            or self.arm_replacement_backward_scalar_products
            != 2 * self.arm_replacement_forward_scalar_products
            or self.equalizer_backward_scalar_products
            != 2 * self.equalizer_forward_scalar_products
            or self.total_backward_scalar_products
            != self.arm_replacement_backward_scalar_products
            + self.equalizer_backward_scalar_products
            or self.total_backward_scalar_products
            != 2 * self.native_replacement_forward_scalar_products
            or self.total_training_scalar_products
            != self.total_forward_scalar_products + self.total_backward_scalar_products
            or len(self.common_path_signature) != 64
            or len(self.equalizer_plan_sha256) != 64
        ):
            raise ETTRILV2ArmError("static operation ledger differs")


@dataclass(frozen=True, slots=True)
class ArmEqualityReceipt:
    """All Phase-1 equal-budget mechanics gates for one five-arm set."""

    schema: str
    protocol: str
    parameter_ledgers: Mapping[str, ParameterLedger]
    token_ledgers: Mapping[str, TokenPositionLedger]
    operation_ledgers: Mapping[str, StaticOperationLedger]
    exact_parameter_equality: bool
    exact_token_position_equality: bool
    exact_static_operation_equality: bool
    weight_updates: int

    def validate(self) -> None:
        expected = set(PRIMARY_ARMS)
        if (
            self.schema != ARM_MECHANICS_SCHEMA
            or self.protocol != PROTOCOL
            or set(self.parameter_ledgers) != expected
            or set(self.token_ledgers) != expected
            or set(self.operation_ledgers) != expected
            or not self.exact_parameter_equality
            or not self.exact_token_position_equality
            or not self.exact_static_operation_equality
            or self.weight_updates != 0
        ):
            raise ETTRILV2ArmError("arm equality receipt differs")
        for value in self.parameter_ledgers.values():
            value.validate()
        for value in self.token_ledgers.values():
            value.validate()
        for value in self.operation_ledgers.values():
            value.validate()


def _optimizer_family(name: str, parameter: nn.Parameter) -> str:
    return (
        "muon"
        if parameter.ndim == 2 and "tok" not in name and "head" not in name
        else "adamw"
    )


def parameter_ledger(
    modules: Mapping[str, nn.Module],
) -> ParameterLedger:
    """Count unique active parameters under the frozen optimizer rule."""

    if not modules:
        raise ETTRILV2ArmError("parameter module inventory is empty")
    seen: set[int] = set()
    names: list[str] = []
    counts = {"muon": 0, "adamw": 0}
    for prefix in sorted(modules):
        module = modules[prefix]
        if not isinstance(module, nn.Module):
            raise ETTRILV2ArmError("parameter inventory contains a non-module")
        for local_name, parameter in module.named_parameters():
            identity = id(parameter)
            name = f"{prefix}.{local_name}"
            if identity in seen:
                raise ETTRILV2ArmError("parameter ownership is duplicated")
            if not parameter.requires_grad:
                raise ETTRILV2ArmError("architecture parameter is inactive")
            seen.add(identity)
            names.append(name)
            counts[_optimizer_family(name, parameter)] += parameter.numel()
    result = ParameterLedger(
        muon=counts["muon"],
        adamw=counts["adamw"],
        unique_trainable=sum(counts.values()),
        active_parameter_names_sha256=_sha256(sorted(names)),
    )
    result.validate()
    return result


def architecture_parameter_ledger(
    compiler: EndogenousTheoryCompiler,
    reactor: nn.Module,
    query_reader: SourceDeletedQueryReader,
) -> ParameterLedger:
    return parameter_ledger(
        {
            "compiler": compiler,
            "query_reader": query_reader,
            "reactor": reactor,
        }
    )


def _clone_reset_state(initial: TypedTheoryState) -> TypedTheoryState:
    """Fresh differentiable compiler-state clone with an open reset status."""

    return TypedTheoryState(
        value_probabilities=initial.value_probabilities.clone(),
        type_probabilities=initial.type_probabilities.clone(),
        relations=initial.relations.clone(),
        active=initial.active.clone(),
        root=initial.root.clone(),
        committed=torch.zeros_like(initial.committed),
        halted=torch.zeros_like(initial.halted),
        step=0,
    )


def _trace(
    policies: Sequence[TransactionPolicy],
    states: Sequence[TypedTheoryState],
) -> ReactorTrace:
    if not policies or len(policies) != len(states):
        raise ETTRILV2ArmError("reactor trace population differs")
    return ReactorTrace(
        opcode=torch.stack(
            [item.opcode_probabilities for item in policies],
            dim=1,
        ),
        source=torch.stack(
            [item.source_probabilities for item in policies],
            dim=1,
        ),
        target=torch.stack(
            [item.target_probabilities for item in policies],
            dim=1,
        ),
        relation=torch.stack(
            [item.relation_probabilities for item in policies],
            dim=1,
        ),
        type_index=torch.stack(
            [item.type_probabilities for item in policies],
            dim=1,
        ),
        value_code=torch.stack(
            [item.value_probabilities for item in policies],
            dim=1,
        ),
        applied_opcode=torch.stack([item.opcode for item in policies], dim=1),
        applied_source=torch.stack([item.source for item in policies], dim=1),
        applied_target=torch.stack([item.target for item in policies], dim=1),
        applied_relation=torch.stack(
            [item.relation for item in policies],
            dim=1,
        ),
        applied_type_index=torch.stack(
            [item.type_index for item in policies],
            dim=1,
        ),
        applied_value_code=torch.stack(
            [item.value_code for item in policies],
            dim=1,
        ),
        active=torch.stack([item.active for item in states], dim=1),
        committed=torch.stack([item.committed for item in states], dim=1),
        halted=torch.stack([item.halted for item in states], dim=1),
    )


def execute_state_reset(
    reactor: GenericTransactionReactor,
    initial_state: TypedTheoryState,
    *,
    steps: int,
    hard: bool,
    command_hidden: torch.Tensor | None = None,
    command_attention_mask: torch.Tensor | None = None,
) -> tuple[TypedTheoryState, ReactorTrace]:
    """Execute A1: every position sees only a fresh step-zero compiler state."""

    if not isinstance(reactor, GenericTransactionReactor):
        raise ETTRILV2ArmError("state-reset requires the native reactor")
    if not 1 <= steps <= reactor.config.max_steps:
        raise ETTRILV2ArmError("state-reset step count differs")
    validate_state(initial_state, reactor.config)
    command, command_padding = reactor._prepare_command(  # noqa: SLF001
        initial_state,
        command_hidden,
        command_attention_mask,
    )
    policies: list[TransactionPolicy] = []
    outputs: list[TypedTheoryState] = []
    for _position in range(steps):
        reset = _clone_reset_state(initial_state)
        policy = reactor._policy(  # noqa: SLF001
            reset,
            hard=hard,
            command=command,
            command_padding=command_padding,
            validate=False,
        )
        output = reactor.apply(
            reset,
            policy,
            hard=hard,
            validate=False,
        )
        policies.append(policy)
        outputs.append(output)
    terminal = outputs[-1]
    validate_state(terminal, reactor.config)
    return terminal, _trace(policies, outputs)


def execute_reactor_arm(
    config: ETTRILV2ArmConfig,
    reactor: nn.Module,
    initial_state: TypedTheoryState,
    *,
    steps: int,
    hard: bool,
    command_hidden: torch.Tensor | None = None,
    command_attention_mask: torch.Tensor | None = None,
) -> tuple[TypedTheoryState, ReactorTrace]:
    """Run the arm's real recurrent mechanic without computing any update."""

    config.validate()
    if hard is not config.hard_transactions:
        raise ETTRILV2ArmError("arm hard-transaction mode differs")
    if steps != config.reactor_positions:
        raise ETTRILV2ArmError("arm reactor horizon differs")
    if config.arm == "state_reset":
        return execute_state_reset(
            reactor,  # type: ignore[arg-type]
            initial_state,
            steps=steps,
            hard=hard,
            command_hidden=command_hidden,
            command_attention_mask=command_attention_mask,
        )
    if config.arm == "dense_state":
        if not isinstance(reactor, DenseStateReactor):
            raise ETTRILV2ArmError("dense arm requires the dense reactor")
    elif not isinstance(reactor, GenericTransactionReactor):
        raise ETTRILV2ArmError("native arm requires the native reactor")
    return reactor(
        initial_state,
        steps=steps,
        hard=hard,
        command_hidden=command_hidden,
        command_attention_mask=command_attention_mask,
    )


def canonical_empty_packet(
    reference: TypedTheoryState,
    config: TheoryReactorConfig,
) -> TypedTheoryState:
    """Build the one canonical A3 reader packet on the reference device."""

    validate_state(reference, config)
    batch = reference.active.shape[0]
    kwargs = {
        "device": reference.active.device,
        "dtype": reference.active.dtype,
    }
    result = TypedTheoryState(
        value_probabilities=torch.zeros(
            batch,
            config.num_slots,
            config.num_value_codes,
            **kwargs,
        ),
        type_probabilities=torch.zeros(
            batch,
            config.num_slots,
            config.num_types,
            **kwargs,
        ),
        relations=torch.zeros(
            batch,
            config.num_relations,
            config.num_slots,
            config.num_slots,
            **kwargs,
        ),
        active=torch.zeros(batch, config.num_slots, **kwargs),
        root=torch.zeros(batch, config.num_slots, **kwargs),
        committed=torch.zeros(batch, **kwargs),
        halted=torch.zeros(batch, **kwargs),
        step=config.max_steps,
    )
    validate_state(result, config)
    return result


def reader_packet_for_arm(
    config: ETTRILV2ArmConfig,
    packet: TypedTheoryState,
    reactor_config: TheoryReactorConfig,
) -> TypedTheoryState:
    config.validate()
    if config.arm == "query_only":
        return canonical_empty_packet(packet, reactor_config)
    return packet


def answer_query_for_arm(
    config: ETTRILV2ArmConfig,
    model: EndogenousTypedTheoryReactorGPT,
    packet: TypedTheoryState,
    query_tokens: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute one factual, intervention, or foil reader call for an arm."""

    reader_packet = reader_packet_for_arm(config, packet, model.config)
    logits, loss = model.answer_query(
        reader_packet,
        query_tokens,
        targets=None,
        attention_mask=attention_mask,
    )
    if loss is not None:
        raise AssertionError("read-only query unexpectedly returned a loss")
    return logits


def _index_fields(value: object, index: torch.Tensor) -> object:
    return type(value)(
        **{
            field.name: getattr(value, field.name).index_select(0, index)
            for field in fields(value)
        }
    )


@dataclass(frozen=True, slots=True)
class TargetBundleBank:
    """Dataset-wide offline target bundles keyed by semantic rectangle ID."""

    rectangle_ids: tuple[str, ...]
    rows_per_rectangle: int
    packet_targets: ETTRPacketTargets
    terminal_packet_targets: ETTRPacketTargets
    transaction_targets: ETTRTransactionTargets
    initial_committed: torch.Tensor
    initial_halted: torch.Tensor
    answer_labels: torch.Tensor
    equivariance: ETTRVariantAlignment

    def validate(self) -> None:
        rectangles = len(self.rectangle_ids)
        rows = rectangles * self.rows_per_rectangle
        if (
            rectangles < 2
            or len(set(self.rectangle_ids)) != rectangles
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in self.rectangle_ids
            )
            or self.rows_per_rectangle != 16
            or self.packet_targets.active.shape[0] != rows
            or self.terminal_packet_targets.active.shape[0] != rows
            or self.transaction_targets.opcode.shape[0] != rows
            or self.initial_committed.shape != (rows,)
            or self.initial_halted.shape != (rows,)
            or self.answer_labels.shape != (rows,)
            or self.answer_labels.dtype != torch.long
        ):
            raise ETTRILV2ArmError("target-bundle bank geometry differs")
        if (
            self.equivariance.left_index.numel() < 1
            or int(self.equivariance.left_index.max()) >= rows
            or int(self.equivariance.right_index.max()) >= rows
        ):
            raise ETTRILV2ArmError("target-bundle alignment leaves the bank")

    def rows(self, rectangle_id: str) -> torch.Tensor:
        try:
            position = self.rectangle_ids.index(rectangle_id)
        except ValueError as exc:
            raise ETTRILV2ArmError(
                "derangement donor is absent from the target bank"
            ) from exc
        start = position * self.rows_per_rectangle
        return torch.arange(
            start,
            start + self.rows_per_rectangle,
            device=self.answer_labels.device,
        )


@dataclass(frozen=True, slots=True)
class ArmSupervision:
    """Offline labels after the A2 bundle-level causal transformation."""

    packet_targets: ETTRPacketTargets
    terminal_packet_targets: ETTRPacketTargets
    transaction_targets: ETTRTransactionTargets
    initial_committed: torch.Tensor
    initial_halted: torch.Tensor
    answer_labels: torch.Tensor
    equivariance: ETTRVariantAlignment
    recipient_ids: tuple[str, ...]
    donor_ids: tuple[str, ...]


def _select_and_rebase_alignment(
    alignment: ETTRVariantAlignment,
    donor_rows: torch.Tensor,
) -> ETTRVariantAlignment:
    """Select donor pairs and map their global row IDs to recipient positions."""

    if donor_rows.ndim != 1 or donor_rows.dtype != torch.long:
        raise ETTRILV2ArmError("donor row index geometry differs")
    row_to_local = {
        int(row): local for local, row in enumerate(donor_rows.detach().cpu().tolist())
    }
    pair_indices = [
        index
        for index, (left, right) in enumerate(
            zip(
                alignment.left_index.detach().cpu().tolist(),
                alignment.right_index.detach().cpu().tolist(),
                strict=True,
            )
        )
        if int(left) in row_to_local and int(right) in row_to_local
    ]
    if not pair_indices:
        raise ETTRILV2ArmError("donor alignment has no selected support")
    pair_index = torch.tensor(
        pair_indices,
        dtype=torch.long,
        device=donor_rows.device,
    )
    left_global = alignment.left_index.index_select(0, pair_index)
    right_global = alignment.right_index.index_select(0, pair_index)
    left = torch.tensor(
        [row_to_local[int(value)] for value in left_global.cpu().tolist()],
        dtype=torch.long,
        device=donor_rows.device,
    )
    right = torch.tensor(
        [row_to_local[int(value)] for value in right_global.cpu().tolist()],
        dtype=torch.long,
        device=donor_rows.device,
    )
    return ETTRVariantAlignment(
        left_index=left,
        right_index=right,
        slot_permutation=alignment.slot_permutation.index_select(
            0,
            pair_index,
        ),
        type_permutation=alignment.type_permutation.index_select(
            0,
            pair_index,
        ),
        relation_permutation=alignment.relation_permutation.index_select(
            0,
            pair_index,
        ),
        value_permutation=alignment.value_permutation.index_select(
            0,
            pair_index,
        ),
        slot_mask=alignment.slot_mask.index_select(0, pair_index),
        relation_mask=alignment.relation_mask.index_select(0, pair_index),
        step_mask=alignment.step_mask.index_select(0, pair_index),
    )


def apply_binding_derangement(
    *,
    recipient_ids: Sequence[str],
    derangement: BindingDerangement,
    bank: TargetBundleBank,
) -> ArmSupervision:
    """Apply A2's complete donor bundle while leaving candidate bytes outside."""

    bank.validate()
    recipients = tuple(recipient_ids)
    assignments = derangement.assignments
    assignment_recipients = tuple(value.recipient_id for value in assignments)
    assignment_donors = tuple(value.donor_id for value in assignments)
    assignment_sha256 = hashlib.sha256(
        control_json_bytes([value.as_dict() for value in assignments])
    ).hexdigest()
    if any(value.recipient_id == value.donor_id for value in assignments):
        raise ETTRILV2ArmError("derangement contains a fixed point")
    if (
        derangement.fold not in (0, 1, 2)
        or not assignments
        or len(set(assignment_recipients)) != len(assignments)
        or len(set(assignment_donors)) != len(assignments)
        or any(
            len(value.donor_digest) != 64
            or any(
                character not in "0123456789abcdef" for character in value.donor_digest
            )
            or value.donor_rank < 0
            for value in assignments
        )
        or assignment_sha256 != derangement.assignment_sha256
        or derangement.receipt()["fixed_points"] != 0
    ):
        raise ETTRILV2ArmError("binding-derangement receipt differs")
    if (
        not recipients
        or len(set(recipients)) != len(recipients)
        or any(value not in bank.rectangle_ids for value in recipients)
    ):
        raise ETTRILV2ArmError("recipient rectangle inventory differs")
    assignment = {value.recipient_id: value.donor_id for value in assignments}
    if any(value not in assignment for value in recipients):
        raise ETTRILV2ArmError("derangement omits a recipient")
    donors = tuple(assignment[value] for value in recipients)
    if any(left == right for left, right in zip(recipients, donors, strict=True)):
        raise ETTRILV2ArmError("derangement contains a fixed point")
    donor_rows = torch.cat(tuple(bank.rows(value) for value in donors))
    result = ArmSupervision(
        packet_targets=_index_fields(bank.packet_targets, donor_rows),
        terminal_packet_targets=_index_fields(
            bank.terminal_packet_targets,
            donor_rows,
        ),
        transaction_targets=_index_fields(
            bank.transaction_targets,
            donor_rows,
        ),
        initial_committed=bank.initial_committed.index_select(0, donor_rows),
        initial_halted=bank.initial_halted.index_select(0, donor_rows),
        answer_labels=bank.answer_labels.index_select(0, donor_rows),
        equivariance=_select_and_rebase_alignment(
            bank.equivariance,
            donor_rows,
        ),
        recipient_ids=recipients,
        donor_ids=donors,
    )
    if result.answer_labels.shape != (16 * len(recipients),):
        raise ETTRILV2ArmError("deranged answer-label geometry differs")
    return result


def apply_supervision_to_objective(
    objective: ETTRObjectiveBatch,
    supervision: ArmSupervision,
    rectangles: ETTRCausalRectangle,
) -> ETTRObjectiveBatch:
    """Replace every A2 target family, never the candidate token bytes/logits."""

    (
        _world_packet,
        world_command,
        world_target,
        command_packet,
        _command_command,
        command_target,
    ) = rectangles.intervention_indices()
    labels = supervision.answer_labels
    world_query = replace(
        objective.world_query_binding,
        correct_target=labels.index_select(0, world_target),
        foil_target=labels.index_select(0, world_command),
    )
    command_query = replace(
        objective.command_query_binding,
        correct_target=labels.index_select(0, command_target),
        foil_target=labels.index_select(0, command_packet),
    )
    return replace(
        objective,
        packet_targets=supervision.packet_targets,
        terminal_packet_targets=supervision.terminal_packet_targets,
        world_intervention_targets=_index_fields(
            supervision.terminal_packet_targets,
            world_target,
        ),
        world_intervention_transaction_targets=_index_fields(
            supervision.transaction_targets,
            world_target,
        ),
        command_intervention_targets=_index_fields(
            supervision.terminal_packet_targets,
            command_target,
        ),
        command_intervention_transaction_targets=_index_fields(
            supervision.transaction_targets,
            command_target,
        ),
        world_query_binding=world_query,
        command_query_binding=command_query,
        transaction_targets=supervision.transaction_targets,
        initial_committed=supervision.initial_committed,
        initial_halted=supervision.initial_halted,
        equivariance=supervision.equivariance,
    )


class _DenseHeadAdapter(nn.Module):
    """The exact 19 x 421 optimizer-family reclassification block."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(DENSE_ADAPTER_ROWS, DENSE_ADAPTER_COLUMNS)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.weight)


class _SplitResidualProjection(nn.Module):
    """One full 4,123 -> 1,241 map with an active 7,999-weight subblock."""

    def __init__(self) -> None:
        super().__init__()
        self.dense_head_adapter = _DenseHeadAdapter()
        self.top_tail_weight = nn.Parameter(
            torch.empty(
                DENSE_ADAPTER_ROWS,
                DENSE_MLP_WIDTH - DENSE_ADAPTER_COLUMNS,
            )
        )
        self.bottom_weight = nn.Parameter(
            torch.empty(
                DENSE_HIDDEN_WIDTH - DENSE_ADAPTER_ROWS,
                DENSE_MLP_WIDTH,
            )
        )
        self.bias = nn.Parameter(torch.empty(DENSE_HIDDEN_WIDTH))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        top = self.dense_head_adapter(value[..., :DENSE_ADAPTER_COLUMNS])
        top = top + F.linear(
            value[..., DENSE_ADAPTER_COLUMNS:],
            self.top_tail_weight,
        )
        top = top + self.bias[:DENSE_ADAPTER_ROWS]
        bottom = F.linear(
            value,
            self.bottom_weight,
            self.bias[DENSE_ADAPTER_ROWS:],
        )
        return torch.cat((top, bottom), dim=-1)


class _TwoLayerDenseGRU(nn.Module):
    """Explicit two-layer GRU plus the frozen six active gate-offset vectors."""

    def __init__(self) -> None:
        super().__init__()
        for layer, input_width in ((0, 512), (1, DENSE_HIDDEN_WIDTH)):
            setattr(
                self,
                f"weight_ih_l{layer}",
                nn.Parameter(torch.empty(3 * DENSE_HIDDEN_WIDTH, input_width)),
            )
            setattr(
                self,
                f"weight_hh_l{layer}",
                nn.Parameter(torch.empty(3 * DENSE_HIDDEN_WIDTH, DENSE_HIDDEN_WIDTH)),
            )
            setattr(
                self,
                f"bias_ih_l{layer}",
                nn.Parameter(torch.empty(3 * DENSE_HIDDEN_WIDTH)),
            )
            setattr(
                self,
                f"bias_hh_l{layer}",
                nn.Parameter(torch.empty(3 * DENSE_HIDDEN_WIDTH)),
            )
        # Rank three intentionally remains AdamW under the frozen ownership
        # rule.  All six vectors enter the three gate equations.
        self.gate_offsets = nn.Parameter(torch.empty(2, 3, DENSE_HIDDEN_WIDTH))

    def forward(
        self,
        value: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs: list[torch.Tensor] = []
        current = value
        for layer, prior in enumerate(hidden):
            input_gates = F.linear(
                current,
                getattr(self, f"weight_ih_l{layer}"),
                getattr(self, f"bias_ih_l{layer}"),
            )
            hidden_gates = F.linear(
                prior,
                getattr(self, f"weight_hh_l{layer}"),
                getattr(self, f"bias_hh_l{layer}"),
            )
            input_reset, input_update, input_new = input_gates.chunk(3, -1)
            hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, -1)
            offsets = self.gate_offsets[layer]
            reset = torch.sigmoid(input_reset + hidden_reset + offsets[0])
            update = torch.sigmoid(input_update + hidden_update + offsets[1])
            candidate = torch.tanh(input_new + reset * hidden_new + offsets[2])
            current = (1.0 - update) * candidate + update * prior
            outputs.append(current)
        return outputs[0], outputs[1]


def _hadamard(width: int) -> torch.Tensor:
    if width < 1 or width & (width - 1):
        raise ETTRILV2ArmError("equalizer width is not a power of two")
    value = torch.ones(1, 1)
    while value.shape[0] < width:
        value = torch.cat(
            (
                torch.cat((value, value), dim=1),
                torch.cat((value, -value), dim=1),
            ),
            dim=0,
        )
    return value / math.sqrt(width)


def _orthogonal_signal(
    carrier: torch.Tensor,
    matrix: torch.Tensor,
    plan: EqualizerStepPlan,
) -> torch.Tensor:
    """Execute exactly ``plan.scalar_products`` fixed scalar products."""

    batch, length, width = carrier.shape
    if (
        length != plan.carrier_length
        or width != plan.width
        or matrix.shape != (width, width)
    ):
        raise TheoryReactorError("equalizer carrier geometry differs")
    matrix = matrix.to(device=carrier.device, dtype=carrier.dtype)
    signal = carrier.new_zeros(batch)
    remaining_replays = plan.full_replays
    # Chunking limits activation memory without changing the actual matmuls.
    while remaining_replays:
        chunk = min(remaining_replays, 8)
        replay = torch.matmul(
            carrier.unsqueeze(0).expand(chunk, -1, -1, -1),
            matrix,
        )
        signal = signal + replay.sum(dim=(0, 2, 3))
        remaining_replays -= chunk
    if plan.full_rows:
        signal = signal + torch.matmul(
            carrier[:, : plan.full_rows],
            matrix,
        ).sum(dim=(1, 2))
    if plan.output_columns:
        signal = signal + F.linear(
            carrier[:, 0],
            matrix[: plan.output_columns],
        ).sum(-1)
    if plan.partial_dot:
        signal = signal + (
            carrier[:, 0, : plan.partial_dot] * matrix[0, : plan.partial_dot]
        ).sum(-1)
    return signal / max(1, plan.scalar_products)


class _BackwardOrthogonalReplay(torch.autograd.Function):
    """Add the second fixed replay required by the matched backward ledger."""

    @staticmethod
    def forward(
        ctx: object,
        value: torch.Tensor,
        matrix: torch.Tensor,
        direction: torch.Tensor,
        scalar_products: int,
        carrier_length: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(matrix, direction)  # type: ignore[attr-defined]
        ctx.scalar_products = scalar_products  # type: ignore[attr-defined]
        ctx.carrier_length = carrier_length  # type: ignore[attr-defined]
        return value

    @staticmethod
    def backward(
        ctx: object,
        gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        matrix, direction = ctx.saved_tensors  # type: ignore[attr-defined]
        plan = EqualizerStepPlan.build(
            ctx.scalar_products,  # type: ignore[attr-defined]
            carrier_length=ctx.carrier_length,  # type: ignore[attr-defined]
            width=gradient.shape[-1],
        )
        carrier = gradient[:, None, :].expand(
            -1,
            plan.carrier_length,
            -1,
        )
        signal = _orthogonal_signal(carrier, matrix, plan)
        replayed = gradient + (
            DENSE_EQUALIZER_COEFFICIENT
            * torch.tanh(signal)[:, None]
            * direction.to(gradient)
        )
        return replayed, None, None, None, None


def _tagged_seed(fold: int, seed: int, name: str) -> int:
    payload = (f"{PROTOCOL}|dense|{fold}|{seed}|{name}").encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def _initialize_parameter(
    parameter: nn.Parameter,
    *,
    fold: int,
    seed: int,
    name: str,
) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_tagged_seed(fold, seed, name))
    fan_in = parameter.shape[-1] if parameter.ndim >= 2 else parameter.numel()
    bound = 1.0 / math.sqrt(max(1, fan_in))
    value = torch.empty(
        parameter.shape,
        dtype=torch.float32,
        device="cpu",
    ).uniform_(-bound, bound, generator=generator)
    parameter.data.copy_(value.to(parameter))


class DenseStateReactor(nn.Module):
    """A4's exact favorable dense controller and generic transaction shell."""

    def __init__(
        self,
        treatment: GenericTransactionReactor,
        *,
        fold: int,
        seed: int,
    ) -> None:
        super().__init__()
        if not isinstance(treatment, GenericTransactionReactor):
            raise ETTRILV2ArmError("dense control source reactor differs")
        if treatment.config != TheoryReactorConfig():
            raise ETTRILV2ArmError(
                "dense control is defined only for frozen production geometry"
            )
        if fold not in (0, 1, 2) or not 0 <= seed < 2**63:
            raise ETTRILV2ArmError("dense initializer identity differs")
        self.config = treatment.config
        self.fold = fold
        self.seed = seed
        self._native_replacement_step_scalar_products = (
            _native_replacement_step_applications(treatment)
        )
        for name in _RETAINED_REACTOR_PARAMETERS:
            setattr(self, name, copy.deepcopy(getattr(treatment, name)))

        # nn.Linear constructors initialize eagerly.  Forking prevents those
        # overwritten defaults from perturbing the arm-level/global RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.packet_initialization = nn.Linear(
                512,
                DENSE_HIDDEN_WIDTH,
            )
            self.initial_state_0 = nn.Parameter(torch.empty(DENSE_HIDDEN_WIDTH))
            self.initial_state_1 = nn.Parameter(torch.empty(DENSE_HIDDEN_WIDTH))
            self.dense_gru = _TwoLayerDenseGRU()
            self.residual_expand = nn.Linear(
                DENSE_HIDDEN_WIDTH,
                DENSE_MLP_WIDTH,
            )
            self.residual_project = _SplitResidualProjection()
            self.output_map = nn.Linear(DENSE_HIDDEN_WIDTH, 512)

        feature_count = (
            self.config.num_slots * self.config.num_value_codes
            + self.config.num_slots * self.config.num_types
            + self.config.num_relations * self.config.num_slots * self.config.num_slots
            + 2 * self.config.num_slots
            + 2
        )
        columns: list[int] = []
        signs: list[float] = []
        counts = [0] * 512
        for index in range(feature_count):
            digest = hashlib.sha256(
                (f"{PROTOCOL}|dense-sketch|{index}").encode("ascii")
            ).digest()
            column = int.from_bytes(digest[:8], "big") % 512
            sign = 1.0 if digest[8] & 1 else -1.0
            columns.append(column)
            signs.append(sign)
            counts[column] += 1
        normalizers = [1.0 / math.sqrt(max(1, counts[column])) for column in columns]
        self.register_buffer(
            "sketch_columns",
            torch.tensor(columns, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "sketch_weights",
            torch.tensor(signs, dtype=torch.float32)
            * torch.tensor(normalizers, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "equalizer_matrix",
            _hadamard(512),
            persistent=False,
        )
        direction = torch.tensor(
            [
                1.0
                if hashlib.sha256(
                    f"{PROTOCOL}|equalizer-direction|{index}".encode("ascii")
                ).digest()[0]
                & 1
                else -1.0
                for index in range(512)
            ]
        )
        self.register_buffer(
            "equalizer_direction",
            direction / math.sqrt(512),
            persistent=False,
        )

        for name, parameter in self.named_replacement_parameters():
            _initialize_parameter(
                parameter,
                fold=fold,
                seed=seed,
                name=name,
            )
        self._validate_parameter_contract(treatment)

    def named_replacement_parameters(
        self,
    ) -> tuple[tuple[str, nn.Parameter], ...]:
        retained = {
            id(parameter)
            for name in _RETAINED_REACTOR_PARAMETERS
            for parameter in (
                getattr(self, name).parameters()
                if isinstance(getattr(self, name), nn.Module)
                else (getattr(self, name),)
            )
        }
        return tuple(
            (name, parameter)
            for name, parameter in self.named_parameters()
            if id(parameter) not in retained
        )

    def _validate_parameter_contract(
        self,
        treatment: GenericTransactionReactor,
    ) -> None:
        treatment_count = sum(value.numel() for value in treatment.parameters())
        dense_count = sum(value.numel() for value in self.parameters())
        replacement = sum(
            value.numel() for _name, value in self.named_replacement_parameters()
        )
        retained = dense_count - replacement
        adapter = self.residual_project.dense_head_adapter.weight.numel()
        replacement_ledger = parameter_ledger({"replacement": _ReplacementView(self)})
        if (
            treatment_count != PRODUCTION_REACTOR_PARAMETERS
            or dense_count != PRODUCTION_REACTOR_PARAMETERS
            or replacement != PRODUCTION_REMOVED_REACTOR_PARAMETERS
            or retained != PRODUCTION_RETAINED_REACTOR_PARAMETERS
            or adapter != DENSE_ADAPTER_PARAMETERS
            or replacement_ledger.muon != PRODUCTION_REMOVED_MUON
            or replacement_ledger.adamw != PRODUCTION_REMOVED_ADAMW
        ):
            raise ETTRILV2ArmError("dense parameter contract differs")
        source = dict(treatment.named_parameters())
        dense = dict(self.named_parameters())
        for name in _RETAINED_REACTOR_PARAMETERS:
            source_items = {
                key: value
                for key, value in source.items()
                if key == name or key.startswith(f"{name}.")
            }
            dense_items = {
                key: value
                for key, value in dense.items()
                if key == name or key.startswith(f"{name}.")
            }
            if source_items.keys() != dense_items.keys() or any(
                not torch.equal(source_items[key], dense_items[key])
                for key in source_items
            ):
                raise ETTRILV2ArmError("dense retained parameter bytes differ")

    def _flat_state(self, state: TypedTheoryState) -> torch.Tensor:
        flat = torch.cat(
            (
                state.value_probabilities.flatten(1),
                state.type_probabilities.flatten(1),
                state.relations.flatten(1),
                state.active,
                state.root,
                state.committed[:, None],
                state.halted[:, None],
            ),
            dim=1,
        )
        if flat.shape[1] != self.sketch_columns.numel():
            raise TheoryReactorError("dense flat-state field geometry differs")
        return flat

    def _sketch(self, state: TypedTheoryState) -> torch.Tensor:
        flat = self._flat_state(state)
        weighted = flat * self.sketch_weights.to(flat)
        result = flat.new_zeros(flat.shape[0], 512)
        return result.scatter_add(
            1,
            self.sketch_columns.to(flat.device).expand(flat.shape[0], -1),
            weighted,
        )

    def _prepare_command(
        self,
        state: TypedTheoryState,
        command_hidden: torch.Tensor | None,
        command_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return GenericTransactionReactor._prepare_command(  # noqa: SLF001
            self,  # type: ignore[arg-type]
            state,
            command_hidden,
            command_attention_mask,
        )

    def _packet_control(
        self,
        state: TypedTheoryState,
        *,
        initial_sketch: torch.Tensor,
        command: torch.Tensor | None,
        command_padding: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        type_context = torch.einsum(
            "bst,tw->bsw",
            state.type_probabilities,
            self.type_embedding,
        )
        value_context = torch.einsum(
            "bsc,cw->bsw",
            state.value_probabilities,
            self.value_embedding,
        )
        slots = (
            value_context
            + type_context
            + self.active_projection(state.active.unsqueeze(-1))
            + self.root_projection(state.root.unsqueeze(-1))
        )
        pooled = (slots * state.active.unsqueeze(-1)).sum(1) / (
            state.active.sum(1, keepdim=True).clamp_min(1.0)
        )
        control = (
            self.control_seed.to(slots.dtype).unsqueeze(0)
            + self.step_embedding.weight[state.step].to(slots.dtype)
            + pooled
            + initial_sketch
            + self.status_projection(_disposition_probabilities(state))
        )
        if command is not None:
            command_read, _ = self.command_attention(
                control[:, None, :],
                self.command_norm(command),
                self.command_norm(command),
                key_padding_mask=command_padding,
                need_weights=False,
            )
            control = control + command_read[:, 0]
        return control, slots

    def _policy_from_control(
        self,
        state: TypedTheoryState,
        control: torch.Tensor,
        slots: torch.Tensor,
        *,
        hard: bool,
        equalizer_plan: EqualizerStepPlan,
    ) -> tuple[TransactionPolicy, torch.Tensor]:
        carrier = torch.cat((control[:, None, :], slots), dim=1)
        signal = _orthogonal_signal(
            carrier,
            self.equalizer_matrix,
            equalizer_plan,
        )
        control = control + (
            DENSE_EQUALIZER_COEFFICIENT
            * torch.tanh(signal)[:, None]
            * self.equalizer_direction.to(control)
        )
        control = _BackwardOrthogonalReplay.apply(
            control,
            self.equalizer_matrix,
            self.equalizer_direction,
            equalizer_plan.scalar_products,
            equalizer_plan.carrier_length,
        )
        return self._heads(state, control, slots, hard=hard), control

    def _heads(
        self,
        state: TypedTheoryState,
        control: torch.Tensor,
        slots: torch.Tensor,
        *,
        hard: bool,
    ) -> TransactionPolicy:
        control = self.output_norm(control)
        encoded_slots = self.output_norm(slots)
        keys = self.slot_key(encoded_slots)
        source_logits = torch.einsum(
            "bw,bsw->bs",
            self.source_query(control),
            keys,
        ).float()
        source_probabilities = source_logits.softmax(-1)
        target_logits = torch.einsum(
            "bw,bsw->bs",
            self.target_query(control),
            keys,
        ).float()
        target_probabilities = target_logits.softmax(-1)
        opcode_logits = self.opcode_head(control).float()
        relation_logits = self.relation_head(control).float()
        type_logits = self.type_head(control).float()
        value_logits = self.value_head(control).float()
        opcode_probabilities = opcode_logits.softmax(-1)
        relation_probabilities = relation_logits.softmax(-1)
        type_probabilities = type_logits.softmax(-1)
        value_probabilities = value_logits.softmax(-1)
        values = (
            opcode_probabilities,
            source_probabilities,
            target_probabilities,
            relation_probabilities,
            type_probabilities,
            value_probabilities,
        )
        applied = tuple(_hard_one_hot(value) for value in values) if hard else values
        dtype = state.value_probabilities.dtype
        return TransactionPolicy(
            opcode=applied[0].to(dtype),
            source=applied[1].to(dtype),
            target=applied[2].to(dtype),
            relation=applied[3].to(dtype),
            type_index=applied[4].to(dtype),
            value_code=applied[5].to(dtype),
            opcode_probabilities=values[0],
            source_probabilities=values[1],
            target_probabilities=values[2],
            relation_probabilities=values[3],
            type_probabilities=values[4],
            value_probabilities=values[5],
            opcode_logits=opcode_logits,
            source_logits=source_logits,
            target_logits=target_logits,
            relation_logits=relation_logits,
            type_logits=type_logits,
            value_logits=value_logits,
        )

    def apply(
        self,
        state: TypedTheoryState,
        policy: TransactionPolicy,
        *,
        hard: bool = False,
        validate: bool = True,
    ) -> TypedTheoryState:
        return GenericTransactionReactor.apply(
            self,  # type: ignore[arg-type]
            state,
            policy,
            hard=hard,
            validate=validate,
        )

    def forward(
        self,
        state: TypedTheoryState,
        *,
        steps: int,
        hard: bool = False,
        command_hidden: torch.Tensor | None = None,
        command_attention_mask: torch.Tensor | None = None,
    ) -> tuple[TypedTheoryState, ReactorTrace]:
        if (
            not 1 <= steps <= self.config.max_steps
            or state.step != 0
            or state.step + steps > self.config.max_steps
        ):
            raise TheoryReactorError(
                "dense forward requires a step-zero bounded reactor prefix"
            )
        validate_state(state, self.config)
        command, command_padding = self._prepare_command(
            state,
            command_hidden,
            command_attention_mask,
        )
        initial_sketch = self._sketch(state)
        initialized = self.packet_initialization(initial_sketch)
        hidden = (
            initialized + self.initial_state_0.to(initialized),
            initialized + self.initial_state_1.to(initialized),
        )
        plans = dense_equalizer_plans(
            self,
            steps=self.config.max_steps,
        )[:steps]
        policies: list[TransactionPolicy] = []
        states: list[TypedTheoryState] = []
        for position in range(steps):
            control, slots = self._packet_control(
                state,
                initial_sketch=initial_sketch,
                command=command,
                command_padding=command_padding,
            )
            hidden = self.dense_gru(control, hidden)
            expanded = F.gelu(self.residual_expand(hidden[1]))
            dense = hidden[1] + self.residual_project(expanded)
            control = self.output_map(dense)
            policy, _ = self._policy_from_control(
                state,
                control,
                slots,
                hard=hard,
                equalizer_plan=plans[position],
            )
            state = self.apply(
                state,
                policy,
                hard=hard,
                validate=False,
            )
            policies.append(policy)
            states.append(state)
        validate_state(state, self.config)
        return state, _trace(policies, states)


class _ReplacementView(nn.Module):
    """Register only A4 replacement parameters for an ownership receipt."""

    def __init__(self, dense: DenseStateReactor) -> None:
        super().__init__()
        self.packet_initialization = dense.packet_initialization
        self.initial_state_0 = dense.initial_state_0
        self.initial_state_1 = dense.initial_state_1
        self.dense_gru = dense.dense_gru
        self.residual_expand = dense.residual_expand
        self.residual_project = dense.residual_project
        self.output_map = dense.output_map


def _native_replacement_step_applications(
    reactor: GenericTransactionReactor,
) -> int:
    slots = reactor.config.num_slots
    carrier = slots + 1
    total = 0
    for name, parameter in reactor.named_parameters():
        if not name.startswith(_REMOVED_PREFIXES):
            continue
        multiplicity = slots if name.startswith("relation_projection.") else carrier
        total += parameter.numel() * multiplicity
    return total


def _dense_step_applications(
    dense: DenseStateReactor,
    *,
    position: int,
) -> int:
    total = 0
    for name, parameter in dense.named_replacement_parameters():
        initialization = name.startswith("packet_initialization.") or name.startswith(
            "initial_state_"
        )
        if not initialization or position == 0:
            total += parameter.numel()
    return total


def dense_equalizer_plans(
    dense: DenseStateReactor,
    *,
    steps: int,
) -> tuple[EqualizerStepPlan, ...]:
    if steps != TRANSACTION_POSITIONS:
        raise ETTRILV2ArmError("dense equalizer horizon differs")
    target = dense._native_replacement_step_scalar_products  # noqa: SLF001
    plans = tuple(
        EqualizerStepPlan.build(
            target - _dense_step_applications(dense, position=position),
            carrier_length=dense.config.num_slots + 1,
            width=dense.config.state_width,
        )
        for position in range(steps)
    )
    if any(value.scalar_products <= 0 for value in plans):
        raise ETTRILV2ArmError("dense operation remainder is nonpositive")
    return plans


def _common_path_signature(
    treatment: GenericTransactionReactor,
    dense: DenseStateReactor,
) -> str:
    treatment_shapes = {
        name: tuple(parameter.shape)
        for name, parameter in treatment.named_parameters()
        if not name.startswith(_REMOVED_PREFIXES)
    }
    dense_shapes = {
        name: tuple(parameter.shape)
        for name, parameter in dense.named_parameters()
        if name.split(".", 1)[0] in _RETAINED_REACTOR_PARAMETERS
    }
    if treatment_shapes != dense_shapes:
        raise ETTRILV2ArmError("common reactor path geometry differs")
    return _sha256(
        {
            "calls": {
                "reader_per_row": READER_CALLS_PER_ROW,
                "reactor_row_calls_per_row": REACTOR_ROW_CALLS_PER_ROW,
                "rows_per_update": ROWS_PER_UPDATE,
            },
            "retained_parameter_shapes": treatment_shapes,
            "transaction_positions": TRANSACTION_POSITIONS,
        }
    )


def operation_ledgers(
    treatment: GenericTransactionReactor,
    dense: DenseStateReactor,
) -> Mapping[str, StaticOperationLedger]:
    """Build all five exact competitive-path operation receipts."""

    if treatment.config != TheoryReactorConfig() or dense.config != treatment.config:
        raise ETTRILV2ArmError("operation ledger production geometry differs")
    call_multiplier = ROWS_PER_UPDATE * REACTOR_ROW_CALLS_PER_ROW
    native_per_call = (
        _native_replacement_step_applications(treatment) * TRANSACTION_POSITIONS
    )
    plans = dense_equalizer_plans(
        dense,
        steps=TRANSACTION_POSITIONS,
    )
    dense_per_call = sum(
        _dense_step_applications(dense, position=position)
        for position in range(TRANSACTION_POSITIONS)
    )
    equalizer_per_call = sum(value.scalar_products for value in plans)
    if dense_per_call + equalizer_per_call != native_per_call:
        raise ETTRILV2ArmError("dense forward operation equality differs")
    common = _common_path_signature(treatment, dense)
    plan_sha = _sha256([asdict(value) for value in plans])
    result: dict[str, StaticOperationLedger] = {}
    for arm in PRIMARY_ARMS:
        if arm == "dense_state":
            own = dense_per_call * call_multiplier
            equalizer = equalizer_per_call * call_multiplier
            equalizer_sha = plan_sha
        else:
            own = native_per_call * call_multiplier
            equalizer = 0
            equalizer_sha = _sha256([])
        target = native_per_call * call_multiplier
        ledger = StaticOperationLedger(
            schema=OPERATION_LEDGER_SCHEMA,
            convention=OPERATION_CONVENTION,
            arm=arm,
            rows_per_update=ROWS_PER_UPDATE,
            reactor_row_calls_per_row=REACTOR_ROW_CALLS_PER_ROW,
            reactor_positions=TRANSACTION_POSITIONS,
            native_replacement_forward_scalar_products=target,
            arm_replacement_forward_scalar_products=own,
            equalizer_forward_scalar_products=equalizer,
            total_forward_scalar_products=own + equalizer,
            arm_replacement_backward_scalar_products=2 * own,
            equalizer_backward_scalar_products=2 * equalizer,
            total_backward_scalar_products=2 * (own + equalizer),
            total_training_scalar_products=3 * (own + equalizer),
            common_path_signature=common,
            equalizer_plan_sha256=equalizer_sha,
        )
        ledger.validate()
        result[arm] = ledger
    return result


def build_arm_equality_receipt(
    compiler: EndogenousTheoryCompiler,
    treatment: GenericTransactionReactor,
    dense: DenseStateReactor,
    query_reader: SourceDeletedQueryReader,
) -> ArmEqualityReceipt:
    """Fail closed unless all five arms meet every equality ledger."""

    treatment_parameters = architecture_parameter_ledger(
        compiler,
        treatment,
        query_reader,
    )
    dense_parameters = architecture_parameter_ledger(
        compiler,
        dense,
        query_reader,
    )
    parameter_ledgers = {
        arm: (dense_parameters if arm == "dense_state" else treatment_parameters)
        for arm in PRIMARY_ARMS
    }
    token = TokenPositionLedger.production()
    token.validate()
    token_ledgers = {arm: token for arm in PRIMARY_ARMS}
    operations = operation_ledgers(treatment, dense)
    parameter_values = {
        (
            value.muon,
            value.adamw,
            value.unique_trainable,
        )
        for value in parameter_ledgers.values()
    }
    token_values = {tuple(asdict(value).values()) for value in token_ledgers.values()}
    operation_values = {
        (
            value.total_forward_scalar_products,
            value.total_backward_scalar_products,
            value.total_training_scalar_products,
            value.common_path_signature,
        )
        for value in operations.values()
    }
    result = ArmEqualityReceipt(
        schema=ARM_MECHANICS_SCHEMA,
        protocol=PROTOCOL,
        parameter_ledgers=parameter_ledgers,
        token_ledgers=token_ledgers,
        operation_ledgers=operations,
        exact_parameter_equality=len(parameter_values) == 1,
        exact_token_position_equality=len(token_values) == 1,
        exact_static_operation_equality=len(operation_values) == 1,
        weight_updates=0,
    )
    result.validate()
    if (
        treatment_parameters.muon != PRODUCTION_ARCHITECTURE_MUON
        or treatment_parameters.adamw != PRODUCTION_ARCHITECTURE_ADAMW
        or treatment_parameters.unique_trainable != PRODUCTION_ARCHITECTURE_PARAMETERS
        or (
            dense_parameters.muon,
            dense_parameters.adamw,
            dense_parameters.unique_trainable,
        )
        != (
            treatment_parameters.muon,
            treatment_parameters.adamw,
            treatment_parameters.unique_trainable,
        )
    ):
        raise ETTRILV2ArmError("production arm parameter equality differs")
    return result


__all__ = [
    "ARM_CONFIGS",
    "ARM_MECHANICS_SCHEMA",
    "ArmEqualityReceipt",
    "ArmName",
    "ArmSupervision",
    "DENSE_ADAPTER_PARAMETERS",
    "DenseStateReactor",
    "ETTRILV2ArmConfig",
    "ETTRILV2ArmError",
    "EqualizerStepPlan",
    "OPERATION_CONVENTION",
    "OPERATION_LEDGER_SCHEMA",
    "PRIMARY_ARMS",
    "ParameterLedger",
    "StaticOperationLedger",
    "TargetBundleBank",
    "TokenPositionLedger",
    "answer_query_for_arm",
    "apply_binding_derangement",
    "apply_supervision_to_objective",
    "architecture_parameter_ledger",
    "build_arm_equality_receipt",
    "canonical_empty_packet",
    "dense_equalizer_plans",
    "execute_reactor_arm",
    "execute_state_reset",
    "operation_ledgers",
    "parameter_ledger",
    "reader_packet_for_arm",
]
