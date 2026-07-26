"""Frozen continuation-data and runner-to-objective contract for ETTR.

Candidate-visible tensors contain token segments and generic categorical
supervision only.  No family identifier, semantic executor, parser product,
host callback, answer verifier, or continuous source payload is admitted.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import re

import torch

from endogenous_typed_theory_reactor import (
    TheoryReactorConfig,
    TheoryReactorError,
)
from ettr_episode import (
    ETTREpisodeBatch,
    ETTREpisodeOutput,
    ETTREpisodeSegment,
    ETTRInterventionOutput,
)
from ettr_objectives import (
    ETTRCausalQueryPair,
    ETTRObjectiveBatch,
    ETTRObjectiveConfig,
    ETTRPacketTargets,
    ETTRTokenTargets,
    ETTRTransactionPredictions,
    ETTRTransactionTargets,
    ETTRVariantAlignment,
)


ETTR_CONTINUATION_SCHEMA = "shohin-ettr-continuation-data-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _packet_target_rows_differ(
    targets: ETTRPacketTargets,
    left_index: torch.Tensor,
    right_index: torch.Tensor,
) -> torch.Tensor:
    left_active = targets.active.index_select(0, left_index)
    right_active = targets.active.index_select(0, right_index)
    left_slot_mask = targets.slot_mask.index_select(0, left_index)
    right_slot_mask = targets.slot_mask.index_select(0, right_index)
    slot_support = left_slot_mask & right_slot_mask
    categorical_support = slot_support & (left_active | right_active)
    relation_support = (
        targets.relation_mask.index_select(0, left_index)
        & targets.relation_mask.index_select(0, right_index)
    )
    return (
        ((left_active != right_active) & slot_support).any(dim=1)
        | (
            (
                targets.root.index_select(0, left_index)
                != targets.root.index_select(0, right_index)
            )
            & slot_support
        ).any(dim=1)
        | (
            (
                targets.value_code.index_select(0, left_index)
                != targets.value_code.index_select(0, right_index)
            )
            & categorical_support
        ).any(dim=1)
        | (
            (
                targets.type_index.index_select(0, left_index)
                != targets.type_index.index_select(0, right_index)
            )
            & categorical_support
        ).any(dim=1)
        | (
            (
                targets.relations.index_select(0, left_index)
                != targets.relations.index_select(0, right_index)
            )
            & relation_support
        ).flatten(1).any(dim=1)
        | (
            targets.committed.index_select(0, left_index)
            != targets.committed.index_select(0, right_index)
        )
        | (
            targets.halted.index_select(0, left_index)
            != targets.halted.index_select(0, right_index)
        )
    )


def _require_same_packet_target(
    targets: ETTRPacketTargets,
    left: torch.Tensor,
    right: torch.Tensor,
    name: str,
) -> None:
    torch._assert_async(
        torch.stack(
            tuple(
                getattr(targets, field.name)
                .index_select(0, left)
                .eq(getattr(targets, field.name).index_select(0, right))
                .all()
                for field in fields(targets)
            )
        ).all(),
        f"ETTR causal rectangle {name} packet target differs",
    )


@dataclass(frozen=True, slots=True)
class ETTRCausalRectangle:
    """Immutable factorial rows ``[rectangle, world, command]``."""

    rows: torch.Tensor

    def validate(
        self,
        episodes: ETTREpisodeBatch,
        factual_initial_targets: ETTRPacketTargets,
        factual_terminal_targets: ETTRPacketTargets,
    ) -> None:
        batch = episodes.world.tokens.shape[0]
        if (
            self.rows.ndim != 3
            or self.rows.shape[1:] != (2, 2)
            or self.rows.dtype != torch.long
            or self.rows.device != episodes.world.tokens.device
            or self.rows.shape[0] < 1
        ):
            raise TheoryReactorError("ETTR causal rectangle geometry differs")
        flat = self.rows.flatten()
        expected = torch.arange(
            batch,
            device=episodes.world.tokens.device,
        )
        if flat.numel() != batch:
            raise TheoryReactorError(
                "ETTR causal rectangles must partition the batch"
            )
        torch._assert_async(
            flat.sort().values.eq(expected).all(),
            "ETTR causal rectangles must partition the batch",
        )
        r00 = self.rows[:, 0, 0]
        r01 = self.rows[:, 0, 1]
        r10 = self.rows[:, 1, 0]
        r11 = self.rows[:, 1, 1]
        _require_same_packet_target(
            factual_initial_targets,
            r00,
            r01,
            "WORLD W0",
        )
        _require_same_packet_target(
            factual_initial_targets,
            r10,
            r11,
            "WORLD W1",
        )
        torch._assert_async(
            _packet_target_rows_differ(
                factual_initial_targets,
                r00,
                r10,
            ).all(),
            "ETTR causal rectangle WORLD factors have identical packet targets",
        )
        for segment, left, right, name in (
            (episodes.world, r00, r01, "WORLD W0"),
            (episodes.world, r10, r11, "WORLD W1"),
            (episodes.command, r00, r10, "COMMAND C0"),
            (episodes.command, r01, r11, "COMMAND C1"),
            (episodes.world, r00, r10, "WORLD factors"),
            (episodes.command, r00, r01, "COMMAND factors"),
        ):
            _require_distinct_source(segment, left, right, name)
        for field_name in ("slot_mask", "relation_mask"):
            reference = getattr(factual_terminal_targets, field_name).index_select(
                0,
                r00,
            )
            for index in (r01, r10, r11):
                torch._assert_async(
                    getattr(
                        factual_terminal_targets,
                        field_name,
                    ).index_select(0, index).eq(reference).all(),
                    "ETTR causal rectangle packet support differs",
                )
        for left, right, name in (
            (r00, r10, "WORLD/C0"),
            (r01, r11, "WORLD/C1"),
            (r00, r01, "COMMAND/W0"),
            (r10, r11, "COMMAND/W1"),
        ):
            torch._assert_async(
                _packet_target_rows_differ(
                    factual_terminal_targets,
                    left,
                    right,
                ).all(),
                f"ETTR causal rectangle {name} has no terminal consequence",
            )
        read_indices = torch.stack(
            tuple(
                episodes.query_read_index.index_select(0, index)
                for index in (r00, r01, r10, r11)
            ),
            dim=1,
        )
        torch._assert_async(
            read_indices.eq(read_indices[:, :1]).all(),
            "ETTR causal rectangle query read indices differ",
        )
        positions = torch.arange(
            episodes.query.tokens.shape[1],
            device=episodes.query.tokens.device,
        )[None, :]
        prefix_mask = positions.le(read_indices[:, :1])
        reference_tokens = episodes.query.tokens.index_select(0, r00)
        reference_mask = episodes.query.attention_mask.index_select(0, r00)
        for index in (r01, r10, r11):
            candidate_tokens = episodes.query.tokens.index_select(0, index)
            candidate_mask = episodes.query.attention_mask.index_select(0, index)
            torch._assert_async(
                (
                    (~prefix_mask | candidate_tokens.eq(reference_tokens))
                    & (~prefix_mask | candidate_mask.eq(reference_mask))
                ).all(),
                "ETTR causal rectangle query prefixes differ",
            )
        labels = tuple(
            episodes.query.targets.index_select(0, index)
            .gather(1, read_indices[:, :1])
            .squeeze(1)
            for index in (r00, r01, r10, r11)
        )
        for left, right, name in (
            (labels[0], labels[2], "WORLD/C0"),
            (labels[1], labels[3], "WORLD/C1"),
            (labels[0], labels[1], "COMMAND/W0"),
            (labels[2], labels[3], "COMMAND/W1"),
        ):
            torch._assert_async(
                left.ne(right).all(),
                f"ETTR causal rectangle {name} query labels are identical",
            )

    def intervention_indices(
        self,
    ) -> tuple[torch.Tensor, ...]:
        r00 = self.rows[:, 0, 0]
        r01 = self.rows[:, 0, 1]
        r10 = self.rows[:, 1, 0]
        r11 = self.rows[:, 1, 1]
        world_packet = torch.cat((r11, r10, r01, r00))
        world_command = torch.cat((r00, r01, r10, r11))
        world_target = torch.cat((r10, r11, r00, r01))
        command_packet = torch.cat((r00, r01, r10, r11))
        command_command = torch.cat((r11, r10, r01, r00))
        command_target = torch.cat((r01, r00, r11, r10))
        return (
            world_packet,
            world_command,
            world_target,
            command_packet,
            command_command,
            command_target,
        )


def _require_distinct_source(
    segment: ETTREpisodeSegment,
    left: torch.Tensor,
    right: torch.Tensor,
    name: str,
) -> None:
    left_tokens = segment.tokens.index_select(0, left)
    right_tokens = segment.tokens.index_select(0, right)
    left_mask = segment.attention_mask.index_select(0, left)
    right_mask = segment.attention_mask.index_select(0, right)
    differs = (
        (left_tokens.ne(right_tokens) & (left_mask | right_mask))
        | left_mask.ne(right_mask)
    ).any(dim=1)
    torch._assert_async(
        differs.all(),
        f"ETTR causal rectangle {name} raw renderings are identical",
    )


def _index_packet_targets(
    targets: ETTRPacketTargets,
    index: torch.Tensor,
) -> ETTRPacketTargets:
    return ETTRPacketTargets(
        **{
            field.name: getattr(targets, field.name).index_select(0, index)
            for field in fields(targets)
        }
    )


def _index_transaction_targets(
    targets: ETTRTransactionTargets,
    index: torch.Tensor,
) -> ETTRTransactionTargets:
    return ETTRTransactionTargets(
        **{
            field.name: getattr(targets, field.name).index_select(0, index)
            for field in fields(targets)
        }
    )


def _gather_query_rows(
    values: torch.Tensor,
    row_index: torch.Tensor,
    read_index: torch.Tensor,
) -> torch.Tensor:
    selected = values.index_select(0, row_index)
    selected_read = read_index.index_select(0, row_index)
    if selected.ndim == 3:
        return selected.gather(
            1,
            selected_read[:, None, None].expand(
                -1,
                1,
                selected.shape[-1],
            ),
        ).squeeze(1)
    if selected.ndim == 2:
        return selected.gather(1, selected_read[:, None]).squeeze(1)
    raise TheoryReactorError("ETTR causal query source rank differs")


def _validate_target_trajectory(
    initial: ETTRPacketTargets,
    transactions: ETTRTransactionTargets,
    terminal: ETTRPacketTargets,
    reactor_config: TheoryReactorConfig,
) -> None:
    """Replay generic labeled transactions without any ontology semantics."""

    batch, slots = initial.active.shape
    rows = torch.arange(batch, device=initial.active.device)
    active = initial.active.clone()
    root = initial.root.clone()
    values = initial.value_code.clone()
    types = initial.type_index.clone()
    relations = initial.relations.clone()
    committed = initial.committed.clone()
    halted = initial.halted.clone()
    for step in range(transactions.opcode.shape[1]):
        valid = transactions.step_mask[:, step]
        open_state = valid & ~committed & ~halted
        opcode = transactions.opcode[:, step]
        source = transactions.source[:, step]
        target = transactions.target[:, step]
        relation = transactions.relation[:, step]
        source_mask = torch.nn.functional.one_hot(
            source,
            slots,
        ).bool()
        alloc = source_mask & (
            open_state
            & opcode.eq(0)
            & ~active[rows, source]
        )[:, None]
        write = source_mask & (
            open_state
            & opcode.eq(1)
            & active[rows, source]
        )[:, None]
        clear = source_mask & (
            open_state
            & opcode.eq(2)
            & active[rows, source]
        )[:, None]
        active = (active | alloc) & ~clear
        values = torch.where(
            (alloc | write),
            transactions.value_code[:, step, None],
            values,
        )
        types = torch.where(
            alloc,
            transactions.type_index[:, step, None],
            types,
        )
        values = torch.where(clear, torch.zeros_like(values), values)
        types = torch.where(clear, torch.zeros_like(types), types)

        pair = (
            torch.nn.functional.one_hot(
                relation,
                reactor_config.num_relations,
            ).bool()[:, :, None, None]
            & source_mask[:, None, :, None]
            & torch.nn.functional.one_hot(
                target,
                slots,
            ).bool()[:, None, None, :]
        )
        link = (
            open_state
            & opcode.eq(3)
        )[:, None, None, None]
        unlink = (
            open_state
            & opcode.eq(4)
        )[:, None, None, None]
        relations = (relations | (link & pair)) & ~(unlink & pair)
        clear_pair = clear[:, None, :, None] | clear[:, None, None, :]
        relations = relations & ~clear_pair
        relations = (
            relations
            & active[:, None, :, None]
            & active[:, None, None, :]
        )
        torch._assert_async(
            relations.flatten(1).sum(-1).le(reactor_config.max_edges).all(),
            "ETTR target trajectory exceeds the relation-edge budget",
        )

        set_root = open_state & opcode.eq(5)
        requested_root = source_mask & active
        root = torch.where(set_root[:, None], requested_root, root)
        root = root & active
        committed = committed | (
            open_state & (opcode.eq(6) | opcode.eq(8))
        )
        halted = halted | (
            open_state & (opcode.eq(7) | opcode.eq(8))
        )
        torch._assert_async(
            (
                ~valid
                | (
                    transactions.committed[:, step].eq(committed)
                    & transactions.halted[:, step].eq(halted)
                )
            ).all(),
            "ETTR transaction disposition disagrees with labeled recurrence",
        )

    slot_mask = terminal.slot_mask
    categorical_mask = slot_mask & terminal.active
    relation_mask = terminal.relation_mask
    consistent = (
        ((active == terminal.active) | ~slot_mask).all()
        & ((root == terminal.root) | ~slot_mask).all()
        & ((values == terminal.value_code) | ~categorical_mask).all()
        & ((types == terminal.type_index) | ~categorical_mask).all()
        & ((relations == terminal.relations) | ~relation_mask).all()
        & committed.eq(terminal.committed).all()
        & halted.eq(terminal.halted).all()
    )
    torch._assert_async(
        consistent,
        "ETTR transaction labels do not realize the terminal packet target",
    )


@dataclass(frozen=True, slots=True)
class ETTRContinuationBatch:
    """One architecture batch plus its generic offline supervision."""

    manifest_sha256: str
    dataset_sha256: str
    episodes: ETTREpisodeBatch
    packet_targets: ETTRPacketTargets
    terminal_packet_targets: ETTRPacketTargets
    causal_rectangles: ETTRCausalRectangle
    transaction_targets: ETTRTransactionTargets
    initial_committed: torch.Tensor
    initial_halted: torch.Tensor
    equivariance: ETTRVariantAlignment | None

    def validate(
        self,
        reactor_config: TheoryReactorConfig,
        objective_config: ETTRObjectiveConfig,
    ) -> None:
        if (
            _SHA256.fullmatch(self.manifest_sha256) is None
            or _SHA256.fullmatch(self.dataset_sha256) is None
        ):
            raise TheoryReactorError("ETTR continuation snapshot receipt differs")
        self.episodes.validate()
        ETTRPacketTargets(
            **{
                field.name: getattr(self.packet_targets, field.name)
                for field in fields(self.packet_targets)
            }
        )
        ETTRPacketTargets(
            **{
                field.name: getattr(self.terminal_packet_targets, field.name)
                for field in fields(self.terminal_packet_targets)
            }
        )
        if not isinstance(self.causal_rectangles, ETTRCausalRectangle):
            raise TheoryReactorError("ETTR causal rectangle type differs")
        self.causal_rectangles.validate(
            self.episodes,
            self.packet_targets,
            self.terminal_packet_targets,
        )
        ETTRTransactionTargets(
            **{
                field.name: getattr(self.transaction_targets, field.name)
                for field in fields(self.transaction_targets)
            }
        )
        if self.equivariance is not None:
            ETTRVariantAlignment(
                **{
                    field.name: getattr(self.equivariance, field.name)
                    for field in fields(self.equivariance)
                }
            )
        batch = self.episodes.world.tokens.shape[0]
        steps = self.transaction_targets.opcode.shape[1]
        if (
            objective_config.num_slots != reactor_config.num_slots
            or objective_config.num_types != reactor_config.num_types
            or objective_config.num_relations != reactor_config.num_relations
            or objective_config.num_value_codes != reactor_config.num_value_codes
            or objective_config.relation_edge_budget != reactor_config.max_edges
            or steps > reactor_config.max_steps
            or self.packet_targets.active.shape != (batch, reactor_config.num_slots)
            or self.terminal_packet_targets.active.shape
            != (batch, reactor_config.num_slots)
            or self.transaction_targets.opcode.shape[0] != batch
            or self.initial_committed.shape != (batch,)
            or self.initial_halted.shape != (batch,)
            or self.initial_committed.dtype != torch.bool
            or self.initial_halted.dtype != torch.bool
        ):
            raise TheoryReactorError("ETTR continuation/objective geometry differs")
        torch._assert_async(
            self.packet_targets.committed.eq(self.initial_committed).all()
            & self.packet_targets.halted.eq(self.initial_halted).all()
            & ~self.initial_committed.any()
            & ~self.initial_halted.any(),
            "ETTR initial packet disposition differs from compiler reset state",
        )
        final_valid = self.transaction_targets.step_mask.sum(-1) - 1
        final_committed = self.transaction_targets.committed.gather(
            1,
            final_valid[:, None],
        ).squeeze(1)
        final_halted = self.transaction_targets.halted.gather(
            1,
            final_valid[:, None],
        ).squeeze(1)
        padded = ~self.transaction_targets.step_mask.all(dim=1)
        torch._assert_async(
            (~padded | final_committed | final_halted).all(),
            "ETTR padded transaction row remains open at its supervision boundary",
        )
        _validate_target_trajectory(
            self.packet_targets,
            self.transaction_targets,
            self.terminal_packet_targets,
            reactor_config,
        )
        devices = {
            self.episodes.world.tokens.device,
            self.packet_targets.active.device,
            self.terminal_packet_targets.active.device,
            self.causal_rectangles.rows.device,
            self.transaction_targets.opcode.device,
            self.initial_committed.device,
            self.initial_halted.device,
        }
        if self.equivariance is not None:
            devices.add(self.equivariance.left_index.device)
        if len(devices) != 1:
            raise TheoryReactorError("ETTR continuation tensors must share one device")
        declared = {
            field.name
            for value in (
                self,
                self.packet_targets,
                self.terminal_packet_targets,
                self.causal_rectangles,
                self.transaction_targets,
            )
            for field in fields(value)
        }
        if any("family" in name or "ontology" in name for name in declared):
            raise TheoryReactorError(
                "ETTR continuation contract exposes a family label"
            )

    def objective_batch(
        self,
        output: ETTREpisodeOutput,
        interventions: ETTRInterventionOutput,
    ) -> ETTRObjectiveBatch:
        """Join reset segment logits without creating boundary targets."""

        if not isinstance(output, ETTREpisodeOutput):
            raise TheoryReactorError("ETTR episode output type differs")
        if not isinstance(interventions, ETTRInterventionOutput):
            raise TheoryReactorError("ETTR intervention output type differs")
        segments = (
            self.episodes.world,
            self.episodes.command,
            self.episodes.query,
        )
        logits = torch.cat(
            (
                output.world_logits,
                output.command_logits,
                output.query_logits,
            ),
            dim=1,
        )
        token_ids = torch.cat(
            tuple(segment.tokens for segment in segments),
            dim=1,
        )
        mask = torch.cat(
            tuple(segment.attention_mask for segment in segments),
            dim=1,
        ).bool()
        reset_mask = torch.zeros_like(mask)
        cursor = 0
        for segment in segments:
            reset_mask[:, cursor] = True
            cursor += segment.tokens.shape[1]
        (
            _world_packet,
            world_command,
            world_target,
            command_packet,
            _command_command,
            command_target,
        ) = self.causal_rectangles.intervention_indices()
        world_query_binding = ETTRCausalQueryPair(
            correct_logits=interventions.world_query_logits,
            foil_logits=_gather_query_rows(
                output.query_logits,
                world_command,
                self.episodes.query_read_index,
            ),
            correct_target=_gather_query_rows(
                self.episodes.query.targets,
                world_target,
                self.episodes.query_read_index,
            ),
            foil_target=_gather_query_rows(
                self.episodes.query.targets,
                world_command,
                self.episodes.query_read_index,
            ),
        )
        command_query_binding = ETTRCausalQueryPair(
            correct_logits=interventions.command_query_logits,
            foil_logits=_gather_query_rows(
                output.query_logits,
                command_packet,
                self.episodes.query_read_index,
            ),
            correct_target=_gather_query_rows(
                self.episodes.query.targets,
                command_target,
                self.episodes.query_read_index,
            ),
            foil_target=_gather_query_rows(
                self.episodes.query.targets,
                command_packet,
                self.episodes.query_read_index,
            ),
        )
        return ETTRObjectiveBatch(
            token_logits=logits,
            token_targets=ETTRTokenTargets(
                token_ids=token_ids,
                mask=mask,
                reset_mask=reset_mask,
            ),
            packet_prediction=output.initial_state,
            packet_targets=self.packet_targets,
            terminal_packet_prediction=output.terminal_state,
            terminal_packet_targets=self.terminal_packet_targets,
            world_intervention_prediction=(
                interventions.world_terminal_state
            ),
            world_intervention_targets=_index_packet_targets(
                self.terminal_packet_targets,
                world_target,
            ),
            world_intervention_transactions=(
                ETTRTransactionPredictions.from_reactor_trace(
                    interventions.world_trace
                )
            ),
            world_intervention_transaction_targets=(
                _index_transaction_targets(
                    self.transaction_targets,
                    world_target,
                )
            ),
            command_intervention_prediction=(
                interventions.command_terminal_state
            ),
            command_intervention_targets=_index_packet_targets(
                self.terminal_packet_targets,
                command_target,
            ),
            command_intervention_transactions=(
                ETTRTransactionPredictions.from_reactor_trace(
                    interventions.command_trace
                )
            ),
            command_intervention_transaction_targets=(
                _index_transaction_targets(
                    self.transaction_targets,
                    command_target,
                )
            ),
            world_query_binding=world_query_binding,
            command_query_binding=command_query_binding,
            transactions=(ETTRTransactionPredictions.from_reactor_trace(output.trace)),
            transaction_targets=self.transaction_targets,
            initial_committed=self.initial_committed,
            initial_halted=self.initial_halted,
            equivariance=self.equivariance,
        )


@dataclass(frozen=True, slots=True)
class ETTRContinuationManifest:
    schema: str
    protected_checkpoint_sha256: str
    tokenizer_sha256: str
    qualification_payload_sha256: str
    hybrid_payload_sha256: str
    train_rows: int
    validation_rows: int
    train_payload_sha256: str
    validation_payload_sha256: str
    source_deleted: bool
    immutable_snapshot: bool
    live_writer_input: bool
    family_label_fields: tuple[str, ...]

    def validate(self) -> None:
        if self.schema != ETTR_CONTINUATION_SCHEMA:
            raise TheoryReactorError("ETTR continuation manifest schema differs")
        for name in (
            "protected_checkpoint_sha256",
            "tokenizer_sha256",
            "qualification_payload_sha256",
            "hybrid_payload_sha256",
            "train_payload_sha256",
            "validation_payload_sha256",
        ):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise TheoryReactorError(f"ETTR continuation manifest {name} differs")
        if (
            self.train_rows <= 0
            or self.validation_rows <= 0
            or not self.source_deleted
            or not self.immutable_snapshot
            or self.live_writer_input
            or self.family_label_fields
            or self.train_payload_sha256 == self.validation_payload_sha256
        ):
            raise TheoryReactorError("ETTR continuation data custody differs")


__all__ = [
    "ETTR_CONTINUATION_SCHEMA",
    "ETTRCausalRectangle",
    "ETTRContinuationBatch",
    "ETTRContinuationManifest",
]
