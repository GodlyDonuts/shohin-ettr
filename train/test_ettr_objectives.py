from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from endogenous_typed_theory_reactor import (
    TRANSACTION_COUNT,
    ReactorTrace,
    TypedTheoryState,
)
from ettr_objectives import (
    OBJECTIVE_SCHEMA,
    ETTRCompositeObjective,
    ETTRObjectiveBatch,
    ETTRObjectiveConfig,
    ETTRObjectiveError,
    ETTRObjectiveWeights,
    ETTRPacketTargets,
    ETTRTokenTargets,
    ETTRTransactionPredictions,
    ETTRTransactionTargets,
    ETTRVariantAlignment,
)


BATCH = 2
TOKENS = 5
VOCAB = 11
SLOTS = 3
TYPES = 2
RELATIONS = 2
VALUES = 4
STEPS = 3


def _leaf(value: torch.Tensor) -> torch.Tensor:
    return value.clone().detach().requires_grad_(True)


def _config(
    *,
    require_equivariance_pairs: bool = True,
) -> ETTRObjectiveConfig:
    return ETTRObjectiveConfig(
        vocab_size=VOCAB,
        num_slots=SLOTS,
        num_types=TYPES,
        num_relations=RELATIONS,
        num_value_codes=VALUES,
        active_slot_budget=2,
        relation_edge_budget=4,
        require_equivariance_pairs=require_equivariance_pairs,
    )


def _packet_labels() -> ETTRPacketTargets:
    active = torch.tensor([[True, True, False], [True, True, False]])
    root = torch.tensor([[True, False, False], [True, False, False]])
    relations = torch.zeros(
        BATCH,
        RELATIONS,
        SLOTS,
        SLOTS,
        dtype=torch.bool,
    )
    relations[:, 0, 0, 1] = True
    return ETTRPacketTargets(
        value_code=torch.tensor([[0, 1, 0], [0, 1, 0]]),
        type_index=torch.tensor([[0, 1, 0], [0, 1, 0]]),
        relations=relations,
        active=active,
        root=root,
        committed=torch.zeros(BATCH, dtype=torch.bool),
        halted=torch.zeros(BATCH, dtype=torch.bool),
        slot_mask=torch.ones(BATCH, SLOTS, dtype=torch.bool),
        relation_mask=torch.ones(
            BATCH,
            RELATIONS,
            SLOTS,
            SLOTS,
            dtype=torch.bool,
        ),
    )


def _hard_packet_state() -> TypedTheoryState:
    labels = _packet_labels()
    values = F.one_hot(labels.value_code, VALUES).float()
    values = values * labels.active.unsqueeze(-1)
    types = F.one_hot(labels.type_index, TYPES).float()
    types = types * labels.active.unsqueeze(-1)
    return TypedTheoryState(
        value_probabilities=_leaf(values),
        type_probabilities=_leaf(types),
        relations=_leaf(labels.relations.float()),
        active=_leaf(labels.active.float()),
        root=_leaf(labels.root.float()),
        committed=_leaf(torch.zeros(BATCH)),
        halted=_leaf(torch.zeros(BATCH)),
        step=0,
    )


def _transaction_labels() -> ETTRTransactionTargets:
    # ALLOC, LINK, HALT exercise every generic operand at least once.
    opcode = torch.tensor([[0, 3, 7], [0, 3, 7]])
    return ETTRTransactionTargets(
        opcode=opcode,
        source=torch.tensor([[0, 0, 2], [0, 0, 2]]),
        target=torch.tensor([[2, 1, 1], [2, 1, 1]]),
        relation=torch.tensor([[1, 0, 1], [1, 0, 1]]),
        type_index=torch.tensor([[0, 1, 1], [0, 1, 1]]),
        value_code=torch.tensor([[1, 2, 3], [1, 2, 3]]),
        committed=torch.zeros(BATCH, STEPS, dtype=torch.bool),
        halted=torch.tensor([[False, False, True], [False, False, True]]),
        step_mask=torch.ones(BATCH, STEPS, dtype=torch.bool),
    )


def _transaction_predictions() -> ETTRTransactionPredictions:
    labels = _transaction_labels()
    active = _packet_labels().active.float()
    return ETTRTransactionPredictions(
        opcode=_leaf(F.one_hot(labels.opcode, TRANSACTION_COUNT).float()),
        source=_leaf(F.one_hot(labels.source, SLOTS).float()),
        target=_leaf(F.one_hot(labels.target, SLOTS).float()),
        relation=_leaf(F.one_hot(labels.relation, RELATIONS).float()),
        type_index=_leaf(F.one_hot(labels.type_index, TYPES).float()),
        value_code=_leaf(F.one_hot(labels.value_code, VALUES).float()),
        active=_leaf(active[:, None, :].expand(-1, STEPS, -1)),
        committed=_leaf(labels.committed.float()),
        halted=_leaf(labels.halted.float()),
    )


def _identity_alignment() -> ETTRVariantAlignment:
    return ETTRVariantAlignment(
        left_index=torch.tensor([0]),
        right_index=torch.tensor([1]),
        slot_permutation=torch.arange(SLOTS)[None, :],
        type_permutation=torch.arange(TYPES)[None, :],
        relation_permutation=torch.arange(RELATIONS)[None, :],
        value_permutation=torch.arange(VALUES)[None, :],
        slot_mask=torch.ones(1, SLOTS, dtype=torch.bool),
        relation_mask=torch.ones(
            1,
            RELATIONS,
            SLOTS,
            SLOTS,
            dtype=torch.bool,
        ),
        step_mask=torch.ones(1, STEPS, dtype=torch.bool),
    )


def _batch() -> ETTRObjectiveBatch:
    torch.manual_seed(17)
    token_ids = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    return ETTRObjectiveBatch(
        token_logits=torch.randn(
            BATCH,
            TOKENS,
            VOCAB,
            requires_grad=True,
        ),
        token_targets=ETTRTokenTargets(
            token_ids=token_ids,
            mask=torch.ones(BATCH, TOKENS, dtype=torch.bool),
            reset_mask=torch.tensor([[True, False, False, False, False]] * BATCH),
        ),
        packet_prediction=_hard_packet_state(),
        packet_targets=_packet_labels(),
        terminal_packet_prediction=_hard_packet_state(),
        terminal_packet_targets=_packet_labels(),
        world_intervention_prediction=_hard_packet_state(),
        world_intervention_targets=_packet_labels(),
        world_intervention_transactions=_transaction_predictions(),
        world_intervention_transaction_targets=_transaction_labels(),
        command_intervention_prediction=_hard_packet_state(),
        command_intervention_targets=_packet_labels(),
        command_intervention_transactions=_transaction_predictions(),
        command_intervention_transaction_targets=_transaction_labels(),
        transactions=_transaction_predictions(),
        transaction_targets=_transaction_labels(),
        initial_committed=torch.zeros(BATCH, dtype=torch.bool),
        initial_halted=torch.zeros(BATCH, dtype=torch.bool),
        equivariance=_identity_alignment(),
    )


def test_public_target_records_expose_every_collator_shape() -> None:
    batch = _batch()
    assert batch.token_targets.token_ids.shape == (BATCH, TOKENS)
    assert batch.token_targets.mask.shape == (BATCH, TOKENS)
    assert batch.token_targets.reset_mask.shape == (BATCH, TOKENS)

    packet = batch.packet_targets
    assert packet.value_code.shape == (BATCH, SLOTS)
    assert packet.type_index.shape == (BATCH, SLOTS)
    assert packet.relations.shape == (
        BATCH,
        RELATIONS,
        SLOTS,
        SLOTS,
    )
    assert packet.active.shape == (BATCH, SLOTS)
    assert packet.root.shape == (BATCH, SLOTS)
    assert packet.committed.shape == (BATCH,)
    assert packet.halted.shape == (BATCH,)
    assert packet.slot_mask.shape == (BATCH, SLOTS)
    assert packet.relation_mask.shape == (
        BATCH,
        RELATIONS,
        SLOTS,
        SLOTS,
    )

    transaction = batch.transaction_targets
    for name in (
        "opcode",
        "source",
        "target",
        "relation",
        "type_index",
        "value_code",
        "committed",
        "halted",
        "step_mask",
    ):
        assert getattr(transaction, name).shape == (BATCH, STEPS)

    alignment = batch.equivariance
    assert alignment is not None
    assert alignment.left_index.shape == (1,)
    assert alignment.right_index.shape == (1,)
    assert alignment.slot_permutation.shape == (1, SLOTS)
    assert alignment.type_permutation.shape == (1, TYPES)
    assert alignment.relation_permutation.shape == (1, RELATIONS)
    assert alignment.value_permutation.shape == (1, VALUES)
    assert alignment.slot_mask.shape == (1, SLOTS)
    assert alignment.relation_mask.shape == (
        1,
        RELATIONS,
        SLOTS,
        SLOTS,
    )
    assert alignment.step_mask.shape == (1, STEPS)


def test_token_mask_allows_only_reset_authorized_segment_restarts() -> None:
    token_ids = torch.tensor([[1, 2, 0, 3, 4]])
    mask = torch.tensor([[True, True, False, True, True]])
    targets = ETTRTokenTargets(
        token_ids=token_ids,
        mask=mask,
        reset_mask=torch.tensor([[True, False, False, True, False]]),
    )
    assert targets.mask.tolist() == [[True, True, False, True, True]]
    with pytest.raises(
        RuntimeError,
        match="restart only at an explicit reset",
    ):
        ETTRTokenTargets(
            token_ids=token_ids,
            mask=mask,
            reset_mask=torch.tensor([[True, False, False, False, False]]),
        )


def test_composite_breakdown_is_finite_weighted_and_differentiable() -> None:
    batch = _batch()
    weights = ETTRObjectiveWeights(
        token_lm=1.3,
        packet=0.9,
        world_intervention=1.1,
        command_intervention=1.2,
        transaction=0.8,
        equivariance=0.7,
        commit_halt=0.6,
        sparsity=0.05,
        anti_bypass=0.2,
    )
    output = ETTRCompositeObjective(
        _config(),
        weights=weights,
    )(batch)
    expected = sum(
        getattr(weights, name) * getattr(output, name)
        for name in (
            "token_lm",
            "packet",
            "world_intervention",
            "command_intervention",
            "transaction",
            "equivariance",
            "commit_halt",
            "sparsity",
            "anti_bypass",
        )
    )
    torch.testing.assert_close(output.total, expected)
    output.total.backward()
    assert batch.token_logits.grad is not None
    assert batch.packet_prediction.value_probabilities.grad is not None
    assert batch.transactions.opcode.grad is not None
    assert torch.isfinite(output.total)


def test_receipt_counts_stay_device_resident_and_auditable() -> None:
    output = ETTRCompositeObjective(_config())(_batch())
    receipt = output.receipt
    assert receipt.schema == OBJECTIVE_SCHEMA
    assert receipt.batch_size == BATCH
    assert receipt.sequence_tokens == TOKENS
    assert receipt.equivariance_pairs == 1
    assert receipt.causal_lm_shift == 1
    assert receipt.weights == ETTRObjectiveWeights().items()
    for name in (
        "lm_target_tokens",
        "supervised_packet_slots",
        "supervised_relation_cells",
        "supervised_world_intervention_slots",
        "supervised_world_intervention_relation_cells",
        "supervised_world_intervention_transaction_decisions",
        "supervised_command_intervention_slots",
        "supervised_command_intervention_relation_cells",
        "supervised_command_intervention_transaction_decisions",
        "supervised_transaction_steps",
        "supervised_transaction_decisions",
        "supervised_opcode_decisions",
        "supervised_source_decisions",
        "supervised_target_decisions",
        "supervised_relation_decisions",
        "supervised_type_decisions",
        "supervised_value_code_decisions",
        "equivariance_packet_cells",
        "equivariance_transaction_cells",
    ):
        count = getattr(receipt, name)
        assert count.shape == ()
        assert count.dtype == torch.int64
        assert count.device == output.total.device
    torch.testing.assert_close(
        receipt.lm_target_tokens,
        torch.tensor(8),
    )
    torch.testing.assert_close(
        receipt.supervised_packet_slots,
        torch.tensor(12),
    )
    torch.testing.assert_close(
        receipt.supervised_relation_cells,
        torch.tensor(72),
    )
    torch.testing.assert_close(
        receipt.supervised_world_intervention_slots,
        torch.tensor(6),
    )
    torch.testing.assert_close(
        receipt.supervised_world_intervention_relation_cells,
        torch.tensor(36),
    )
    torch.testing.assert_close(
        receipt.supervised_world_intervention_transaction_decisions,
        torch.tensor(18),
    )
    torch.testing.assert_close(
        receipt.supervised_command_intervention_slots,
        torch.tensor(6),
    )
    torch.testing.assert_close(
        receipt.supervised_command_intervention_relation_cells,
        torch.tensor(36),
    )
    torch.testing.assert_close(
        receipt.supervised_command_intervention_transaction_decisions,
        torch.tensor(18),
    )
    torch.testing.assert_close(
        receipt.supervised_transaction_steps,
        torch.tensor(6),
    )
    torch.testing.assert_close(
        receipt.supervised_transaction_decisions,
        torch.tensor(18),
    )
    expected_head_support = {
        "supervised_opcode_decisions": 6,
        "supervised_source_decisions": 4,
        "supervised_target_decisions": 2,
        "supervised_relation_decisions": 2,
        "supervised_type_decisions": 2,
        "supervised_value_code_decisions": 2,
    }
    for name, expected in expected_head_support.items():
        torch.testing.assert_close(
            getattr(receipt, name),
            torch.tensor(expected),
        )


def test_hot_path_contains_no_explicit_host_sync_escape() -> None:
    source = Path(__file__).with_name("ettr_objectives.py").read_text()
    for forbidden in (
        ".item(",
        ".tolist(",
        ".cpu(",
        ".numpy(",
        "bool(torch.",
        "torch.equal(",
        "torch.allclose(",
    ):
        assert forbidden not in source


def test_token_loss_is_strictly_one_step_causal() -> None:
    batch = _batch()
    objective = ETTRCompositeObjective(
        _config(),
        weights=ETTRObjectiveWeights(
            token_lm=1.0,
            packet=0.0,
            world_intervention=0.0,
            command_intervention=0.0,
            transaction=0.0,
            equivariance=0.0,
            commit_halt=0.0,
            sparsity=0.0,
            anti_bypass=0.0,
        ),
    )
    baseline = objective(batch)
    changed_logits = batch.token_logits.detach().clone()
    changed_logits[:, -1] = 1_000.0
    changed_logits.requires_grad_(True)
    changed = objective(replace(batch, token_logits=changed_logits))
    torch.testing.assert_close(changed.token_lm, baseline.token_lm)
    changed.total.backward()
    torch.testing.assert_close(
        changed_logits.grad[:, -1],
        torch.zeros_like(changed_logits.grad[:, -1]),
    )
    assert changed_logits.grad[:, :-1].abs().sum() > 0


def test_token_loss_never_crosses_an_explicit_segment_reset() -> None:
    batch = _batch()
    reset = batch.token_targets.reset_mask.clone()
    reset[:, 3] = True
    targets = replace(
        batch.token_targets,
        reset_mask=reset,
    )
    objective = ETTRCompositeObjective(
        _config(),
        weights=ETTRObjectiveWeights(
            token_lm=1.0,
            packet=0.0,
            world_intervention=0.0,
            command_intervention=0.0,
            transaction=0.0,
            equivariance=0.0,
            commit_halt=0.0,
            sparsity=0.0,
            anti_bypass=0.0,
        ),
    )
    baseline = objective(replace(batch, token_targets=targets)).token_lm
    changed_ids = targets.token_ids.clone()
    changed_ids[:, 3] = (changed_ids[:, 3] + 3) % VOCAB
    changed = objective(
        replace(
            batch,
            token_targets=replace(
                targets,
                token_ids=changed_ids,
            ),
        )
    ).token_lm
    torch.testing.assert_close(changed, baseline)


def test_transaction_loss_ignores_operands_unused_by_opcode() -> None:
    batch = _batch()
    objective = ETTRCompositeObjective(_config())
    baseline = objective(batch).transaction

    irrelevant_source = batch.transactions.source.detach().clone()
    irrelevant_source[:, 2] = irrelevant_source[:, 2].roll(1, -1)
    irrelevant = replace(
        batch.transactions,
        source=irrelevant_source,
    )
    ignored_loss = objective(replace(batch, transactions=irrelevant)).transaction
    torch.testing.assert_close(ignored_loss, baseline)

    relevant_source = batch.transactions.source.detach().clone()
    relevant_source[:, 0] = relevant_source[:, 0].roll(1, -1)
    relevant = replace(batch.transactions, source=relevant_source)
    relevant_loss = objective(replace(batch, transactions=relevant)).transaction
    assert relevant_loss > baseline


def test_native_trace_bridge_preserves_and_supervises_value_code() -> None:
    batch = _batch()
    prediction = batch.transactions
    trace = ReactorTrace(
        opcode=prediction.opcode,
        source=prediction.source,
        target=prediction.target,
        relation=prediction.relation,
        type_index=prediction.type_index,
        value_code=prediction.value_code,
        applied_opcode=prediction.opcode,
        applied_source=prediction.source,
        applied_target=prediction.target,
        applied_relation=prediction.relation,
        applied_type_index=prediction.type_index,
        applied_value_code=prediction.value_code,
        active=prediction.active,
        committed=prediction.committed,
        halted=prediction.halted,
    )
    bridged = ETTRTransactionPredictions.from_reactor_trace(trace)
    assert bridged.value_code is trace.value_code
    assert bridged.value_code.shape == (BATCH, STEPS, VALUES)

    objective = ETTRCompositeObjective(_config())
    baseline = objective(replace(batch, transactions=bridged)).transaction
    wrong_value = bridged.value_code.detach().clone()
    wrong_value[:, 0] = wrong_value[:, 0].roll(1, -1)
    changed = objective(
        replace(
            batch,
            transactions=replace(
                bridged,
                value_code=wrong_value,
            ),
        )
    ).transaction
    assert changed > baseline


def _right_permuted_state(
    state: TypedTheoryState,
    *,
    slot: torch.Tensor,
    type_index: torch.Tensor,
    relation: torch.Tensor,
    value_code: torch.Tensor,
) -> TypedTheoryState:
    values = state.value_probabilities.detach().clone()
    types = state.type_probabilities.detach().clone()
    relations = state.relations.detach().clone()
    active = state.active.detach().clone()
    root = state.root.detach().clone()
    for left_slot in range(SLOTS):
        right_slot = int(slot[left_slot])
        active[1, right_slot] = state.active[0, left_slot]
        root[1, right_slot] = state.root[0, left_slot]
        for left_value in range(VALUES):
            values[1, right_slot, int(value_code[left_value])] = (
                state.value_probabilities[0, left_slot, left_value]
            )
        for left_type in range(TYPES):
            types[1, right_slot, int(type_index[left_type])] = state.type_probabilities[
                0, left_slot, left_type
            ]
    relations[1].zero_()
    for left_relation in range(RELATIONS):
        for left_source in range(SLOTS):
            for left_target in range(SLOTS):
                relations[
                    1,
                    int(relation[left_relation]),
                    int(slot[left_source]),
                    int(slot[left_target]),
                ] = state.relations[
                    0,
                    left_relation,
                    left_source,
                    left_target,
                ]
    return TypedTheoryState(
        value_probabilities=_leaf(values),
        type_probabilities=_leaf(types),
        relations=_leaf(relations),
        active=_leaf(active),
        root=_leaf(root),
        committed=_leaf(state.committed.detach()),
        halted=_leaf(state.halted.detach()),
        step=state.step,
    )


def _right_permuted_transactions(
    prediction: ETTRTransactionPredictions,
    *,
    slot: torch.Tensor,
    type_index: torch.Tensor,
    relation: torch.Tensor,
    value_code: torch.Tensor,
) -> ETTRTransactionPredictions:
    values = {
        field.name: getattr(prediction, field.name).detach().clone()
        for field in fields(ETTRTransactionPredictions)
    }
    values["opcode"][1] = prediction.opcode[0]
    values["committed"][1] = prediction.committed[0]
    values["halted"][1] = prediction.halted[0]
    for left_slot in range(SLOTS):
        right_slot = int(slot[left_slot])
        values["source"][1, :, right_slot] = prediction.source[0, :, left_slot]
        values["target"][1, :, right_slot] = prediction.target[0, :, left_slot]
        values["active"][1, :, right_slot] = prediction.active[0, :, left_slot]
    for left_type in range(TYPES):
        values["type_index"][1, :, int(type_index[left_type])] = prediction.type_index[
            0, :, left_type
        ]
    for left_relation in range(RELATIONS):
        values["relation"][1, :, int(relation[left_relation])] = prediction.relation[
            0, :, left_relation
        ]
    for left_value in range(VALUES):
        values["value_code"][1, :, int(value_code[left_value])] = prediction.value_code[
            0, :, left_value
        ]
    return ETTRTransactionPredictions(
        **{name: _leaf(value) for name, value in values.items()}
    )


def test_equivariance_uses_declared_variant_coordinate_transport() -> None:
    batch = _batch()
    slot = torch.tensor([1, 0, 2])
    type_index = torch.tensor([1, 0])
    relation = torch.tensor([1, 0])
    value_code = torch.tensor([1, 0, 3, 2])
    alignment = replace(
        _identity_alignment(),
        slot_permutation=slot[None, :],
        type_permutation=type_index[None, :],
        relation_permutation=relation[None, :],
        value_permutation=value_code[None, :],
    )
    state = _right_permuted_state(
        batch.packet_prediction,
        slot=slot,
        type_index=type_index,
        relation=relation,
        value_code=value_code,
    )
    transactions = _right_permuted_transactions(
        batch.transactions,
        slot=slot,
        type_index=type_index,
        relation=relation,
        value_code=value_code,
    )
    transported = replace(
        batch,
        packet_prediction=state,
        terminal_packet_prediction=state,
        transactions=transactions,
        equivariance=alignment,
    )
    objective = ETTRCompositeObjective(_config())
    exact = objective(transported).equivariance
    torch.testing.assert_close(exact, torch.zeros_like(exact))

    broken_values = state.value_probabilities.detach().clone()
    broken_values[1, slot[0]] = broken_values[1, slot[0]].roll(1)
    broken_state = replace(
        state,
        value_probabilities=_leaf(broken_values),
    )
    broken = objective(
        replace(transported, packet_prediction=broken_state)
    ).equivariance
    assert broken > exact


def test_terminal_packet_is_directly_supervised() -> None:
    batch = _batch()
    objective = ETTRCompositeObjective(_config())
    exact = objective(batch).packet
    values = batch.terminal_packet_prediction.value_probabilities.detach().clone()
    values[:, 0] = values[:, 0].roll(1, dims=-1)
    terminal = replace(
        batch.terminal_packet_prediction,
        value_probabilities=_leaf(values),
    )
    broken = objective(
        replace(batch, terminal_packet_prediction=terminal)
    )
    assert broken.packet > exact
    broken.packet.backward()
    assert terminal.value_probabilities.grad is not None


def test_terminal_disposition_is_part_of_packet_supervision() -> None:
    batch = _batch()
    objective = ETTRCompositeObjective(_config())
    exact = objective(batch).packet
    terminal = replace(
        batch.terminal_packet_prediction,
        committed=_leaf(torch.ones(BATCH)),
        halted=_leaf(torch.ones(BATCH)),
    )
    broken = objective(
        replace(batch, terminal_packet_prediction=terminal)
    )
    assert broken.packet > exact
    broken.packet.backward()
    assert terminal.committed.grad is not None
    assert terminal.halted.grad is not None


def test_factorial_world_and_command_arms_are_supervised_separately() -> None:
    batch = _batch()
    objective = ETTRCompositeObjective(_config())
    exact = objective(batch)
    values = (
        batch.world_intervention_prediction.value_probabilities.detach().clone()
    )
    values[:, 0] = values[:, 0].roll(1, dims=-1)
    world = replace(
        batch.world_intervention_prediction,
        value_probabilities=_leaf(values),
    )
    broken = objective(
        replace(batch, world_intervention_prediction=world)
    )
    assert broken.world_intervention > exact.world_intervention
    torch.testing.assert_close(
        broken.command_intervention,
        exact.command_intervention,
    )
    broken.world_intervention.backward()
    assert world.value_probabilities.grad is not None


def test_commit_halt_loss_checks_prefix_recurrence_and_labels() -> None:
    batch = _batch()
    objective = ETTRCompositeObjective(_config())
    exact = objective(batch).commit_halt
    wrong_status = replace(
        batch.transactions,
        halted=torch.zeros_like(batch.transactions.halted),
    )
    wrong = objective(replace(batch, transactions=wrong_status)).commit_halt
    assert wrong > exact


def test_anti_bypass_rejects_soft_dense_packet_channels() -> None:
    batch = _batch()
    hard = ETTRCompositeObjective(_config())(batch)
    dense_state = TypedTheoryState(
        value_probabilities=_leaf(torch.full((BATCH, SLOTS, VALUES), 0.5)),
        type_probabilities=_leaf(torch.full((BATCH, SLOTS, TYPES), 0.5)),
        relations=_leaf(
            torch.full(
                (BATCH, RELATIONS, SLOTS, SLOTS),
                0.9,
            )
        ),
        active=_leaf(torch.full((BATCH, SLOTS), 0.9)),
        root=_leaf(torch.full((BATCH, SLOTS), 0.5)),
        committed=_leaf(torch.full((BATCH,), 0.5)),
        halted=_leaf(torch.full((BATCH,), 0.5)),
        step=0,
    )
    dense = ETTRCompositeObjective(_config())(
        replace(batch, packet_prediction=dense_state)
    )
    assert dense.anti_bypass > hard.anti_bypass
    assert dense.sparsity > hard.sparsity


def test_config_targets_shapes_ranges_and_finiteness_fail_closed() -> None:
    with pytest.raises(ETTRObjectiveError, match="active-slot budget"):
        ETTRObjectiveConfig(
            vocab_size=VOCAB,
            num_slots=2,
            active_slot_budget=3,
        )
    with pytest.raises(ETTRObjectiveError, match="weights"):
        ETTRObjectiveWeights(token_lm=float("nan"))
    with pytest.raises(ETTRObjectiveError, match="relation geometry"):
        labels = _packet_labels()
        replace(
            labels,
            relations=torch.zeros(
                BATCH,
                RELATIONS,
                SLOTS,
                SLOTS + 1,
                dtype=torch.bool,
            ),
        )
    with pytest.raises(RuntimeError, match="row-wise permutation"):
        replace(
            _identity_alignment(),
            slot_permutation=torch.tensor([[0, 0, 2]]),
        )
    with pytest.raises(RuntimeError, match="frozen range"):
        replace(
            _transaction_labels(),
            opcode=torch.tensor(
                [[0, 3, TRANSACTION_COUNT], [0, 3, 7]]
            ),
        )
    with pytest.raises(RuntimeError, match="not monotone"):
        replace(
            _transaction_labels(),
            halted=torch.tensor([[True, False, True], [False, False, True]]),
        )

    batch = _batch()
    bad_targets = replace(
        batch.packet_targets,
        value_code=torch.full((BATCH, SLOTS), VALUES),
    )
    with pytest.raises(RuntimeError, match="class range"):
        ETTRCompositeObjective(_config())(replace(batch, packet_targets=bad_targets))
    nan_logits = batch.token_logits.detach().clone()
    nan_logits[0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        ETTRCompositeObjective(_config())(replace(batch, token_logits=nan_logits))
    with pytest.raises(ETTRObjectiveError, match="equivariance pairs"):
        ETTRCompositeObjective(_config())(replace(batch, equivariance=None))


def test_equivariance_can_be_explicitly_disabled_for_ablation() -> None:
    batch = replace(_batch(), equivariance=None)
    output = ETTRCompositeObjective(_config(require_equivariance_pairs=False))(batch)
    torch.testing.assert_close(
        output.equivariance,
        torch.zeros_like(output.equivariance),
    )
    assert output.receipt.equivariance_pairs == 0
