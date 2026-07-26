from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import inspect

import pytest
import torch

from endogenous_typed_theory_reactor import (
    EndogenousTheoryCompiler,
    EndogenousTypedTheoryReactorGPT,
    GenericTransactionReactor,
    SourceDeletedQueryReader,
    TheoryReactorConfig,
    TypedTheoryState,
    validate_state,
)
from ettr_data_contract import ETTRCausalRectangle
from ettr_il_v2_arms import (
    ARM_CONFIGS,
    DENSE_ADAPTER_PARAMETERS,
    PRIMARY_ARMS,
    DenseStateReactor,
    ETTRILV2ArmError,
    TargetBundleBank,
    TokenPositionLedger,
    answer_query_for_arm,
    apply_binding_derangement,
    apply_supervision_to_objective,
    build_arm_equality_receipt,
    canonical_empty_packet,
    dense_equalizer_plans,
    execute_state_reset,
    operation_ledgers,
    parameter_ledger,
)
from ettr_il_v2_controls import (
    BindingDerangement,
    DerangementAssignment,
    canonical_json_bytes as control_json_bytes,
)
from ettr_objectives import (
    ETTRCausalQueryPair,
    ETTRObjectiveBatch,
    ETTRVariantAlignment,
)
from model import GPT, GPTConfig
from test_ettr_data_contract import _packet, _transactions


MODEL_SEED = 827771697280926998


def _small_config() -> TheoryReactorConfig:
    return TheoryReactorConfig(
        d_model=32,
        state_width=32,
        num_slots=6,
        num_types=3,
        num_relations=3,
        num_value_codes=64,
        max_edges=96,
        num_heads=4,
        compiler_layers=1,
        reactor_layers=1,
        query_layers=1,
        ff_multiplier=2,
        max_steps=6,
        stage_after_block=1,
        parameter_cap=1_000_000,
    )


def _small_model() -> EndogenousTypedTheoryReactorGPT:
    torch.manual_seed(2026072601)
    base = GPT(
        GPTConfig(
            vocab_size=64,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            d_model=32,
            d_ff=64,
            seq_len=32,
            zloss=0.0,
        )
    )
    return EndogenousTypedTheoryReactorGPT(base, _small_config()).eval()


@pytest.fixture(scope="module")
def production_modules() -> tuple[
    EndogenousTheoryCompiler,
    GenericTransactionReactor,
    DenseStateReactor,
    SourceDeletedQueryReader,
]:
    config = TheoryReactorConfig()
    torch.manual_seed(2026072602)
    compiler = EndogenousTheoryCompiler(config)
    treatment = GenericTransactionReactor(config)
    query_reader = SourceDeletedQueryReader(config)
    global_before = torch.get_rng_state().clone()
    dense = DenseStateReactor(
        treatment,
        fold=0,
        seed=MODEL_SEED,
    )
    assert torch.equal(torch.get_rng_state(), global_before)
    return compiler, treatment, dense, query_reader


def test_all_five_arm_configs_are_distinct_and_exact() -> None:
    assert tuple(ARM_CONFIGS) == PRIMARY_ARMS
    assert len({value.sha256() for value in ARM_CONFIGS.values()}) == 5
    for arm, value in ARM_CONFIGS.items():
        value.validate()
        assert value.arm == arm
        assert value.hard_transactions
        assert value.reactor_positions == 64


def test_a1_resets_to_the_compiler_packet_before_every_position() -> None:
    model = _small_model()
    world = torch.randint(0, 64, (2, 9))
    command = torch.randint(0, 64, (2, 7))
    initial = model.compile_world(world, hard=True)
    command_hidden = model._encode_to_stage(command, pos=0)

    native_terminal, native_trace = model.reactor(
        initial,
        steps=4,
        hard=True,
        command_hidden=command_hidden,
    )
    reset_terminal, reset_trace = execute_state_reset(
        model.reactor,
        initial,
        steps=4,
        hard=True,
        command_hidden=command_hidden,
    )

    assert native_terminal.step == 4
    assert reset_terminal.step == 1
    assert native_trace.opcode.shape == reset_trace.opcode.shape == (2, 4, 9)
    for name in (
        "opcode",
        "source",
        "target",
        "relation",
        "type_index",
        "value_code",
    ):
        values = getattr(reset_trace, name)
        torch.testing.assert_close(
            values,
            values[:, :1].expand_as(values),
            rtol=0,
            atol=0,
        )
    # The reset arm did not skip any policy or transaction application.
    assert reset_trace.applied_opcode.shape[1] == 4
    assert reset_trace.active.shape[1] == 4


def test_a3_canonical_empty_packet_erases_every_reader_state_field() -> None:
    model = _small_model()
    first = model.compile_world(
        torch.randint(0, 64, (2, 8)),
        hard=True,
    )
    second = model.compile_world(
        torch.randint(0, 64, (2, 8)),
        hard=True,
    )
    empty = canonical_empty_packet(first, model.config)
    assert empty.step == model.config.max_steps
    for field in fields(empty):
        value = getattr(empty, field.name)
        if isinstance(value, torch.Tensor):
            assert not bool(value.any())

    query = torch.randint(0, 64, (2, 6))
    first_logits = answer_query_for_arm(
        ARM_CONFIGS["query_only"],
        model,
        first,
        query,
    )
    second_logits = answer_query_for_arm(
        ARM_CONFIGS["query_only"],
        model,
        second,
        query,
    )
    torch.testing.assert_close(first_logits, second_logits, rtol=0, atol=0)


def _alignment(rows: int) -> ETTRVariantAlignment:
    assert rows == 32
    left = torch.arange(0, rows, 2)
    right = left + 1
    pairs = left.numel()
    return ETTRVariantAlignment(
        left_index=left,
        right_index=right,
        slot_permutation=torch.arange(6).expand(pairs, -1).clone(),
        type_permutation=torch.arange(3).expand(pairs, -1).clone(),
        relation_permutation=torch.arange(3).expand(pairs, -1).clone(),
        value_permutation=torch.arange(64).expand(pairs, -1).clone(),
        slot_mask=torch.ones(pairs, 6, dtype=torch.bool),
        relation_mask=torch.ones(pairs, 3, 6, 6, dtype=torch.bool),
        step_mask=torch.ones(pairs, 3, dtype=torch.bool),
    )


def _target_bank() -> tuple[
    TargetBundleBank,
    BindingDerangement,
    tuple[str, str],
]:
    rectangle_ids = ("0" * 64, "1" * 64)
    packet = _packet(32)
    packet_values = packet.value_code.clone()
    packet_values[16:, 0] = 11
    packet = replace(packet, value_code=packet_values)
    terminal_values = packet.value_code.clone()
    terminal_values[:16, 0] = 13
    terminal_values[16:, 0] = 17
    terminal = replace(packet, value_code=terminal_values)
    transactions = _transactions(32)
    transaction_values = transactions.value_code.clone()
    transaction_values[:16] = 13
    transaction_values[16:] = 17
    transactions = replace(transactions, value_code=transaction_values)
    first_pattern = torch.tensor([0, 1, 1, 0]).repeat(4)
    labels = torch.cat((first_pattern, 1 - first_pattern))
    bank = TargetBundleBank(
        rectangle_ids=rectangle_ids,
        rows_per_rectangle=16,
        packet_targets=packet,
        terminal_packet_targets=terminal,
        transaction_targets=transactions,
        initial_committed=torch.zeros(32, dtype=torch.bool),
        initial_halted=torch.zeros(32, dtype=torch.bool),
        answer_labels=labels,
        equivariance=_alignment(32),
    )
    assignments = (
        DerangementAssignment(
            recipient_id=rectangle_ids[0],
            donor_id=rectangle_ids[1],
            donor_rank=0,
            donor_digest="a" * 64,
        ),
        DerangementAssignment(
            recipient_id=rectangle_ids[1],
            donor_id=rectangle_ids[0],
            donor_rank=0,
            donor_digest="b" * 64,
        ),
    )
    derangement = BindingDerangement(
        fold=0,
        assignments=assignments,
        assignment_sha256=hashlib.sha256(
            control_json_bytes([value.as_dict() for value in assignments])
        ).hexdigest(),
    )
    return bank, derangement, rectangle_ids


def test_a2_applies_the_entire_donor_bundle_not_candidate_bytes() -> None:
    bank, derangement, rectangle_ids = _target_bank()
    supervision = apply_binding_derangement(
        recipient_ids=rectangle_ids,
        derangement=derangement,
        bank=bank,
    )
    assert supervision.recipient_ids == rectangle_ids
    assert supervision.donor_ids == rectangle_ids[::-1]
    torch.testing.assert_close(
        supervision.packet_targets.value_code[:16],
        bank.packet_targets.value_code[16:],
    )
    torch.testing.assert_close(
        supervision.terminal_packet_targets.value_code[:16],
        bank.terminal_packet_targets.value_code[16:],
    )
    torch.testing.assert_close(
        supervision.transaction_targets.value_code[:16],
        bank.transaction_targets.value_code[16:],
    )
    torch.testing.assert_close(
        supervision.answer_labels[:16],
        bank.answer_labels[16:],
    )
    assert supervision.equivariance.left_index.numel() == 16
    assert int(supervision.equivariance.left_index.max()) < 32
    assert int(supervision.equivariance.right_index.max()) < 32


def test_a2_rebinds_every_objective_target_and_preserves_predictions() -> None:
    bank, derangement, rectangle_ids = _target_bank()
    supervision = apply_binding_derangement(
        recipient_ids=rectangle_ids,
        derangement=derangement,
        bank=bank,
    )
    rectangle_rows = torch.arange(32).view(8, 2, 2)
    rectangles = ETTRCausalRectangle(rows=rectangle_rows)
    (
        _world_packet,
        world_command,
        world_target,
        command_packet,
        _command_command,
        command_target,
    ) = rectangles.intervention_indices()
    logits = torch.randn(32, 64)
    factual_labels = bank.answer_labels
    objective = ETTRObjectiveBatch(
        token_logits=torch.randn(32, 8, 64),
        token_targets=object(),
        packet_prediction=object(),
        packet_targets=bank.packet_targets,
        terminal_packet_prediction=object(),
        terminal_packet_targets=bank.terminal_packet_targets,
        world_intervention_prediction=object(),
        world_intervention_targets=bank.terminal_packet_targets,
        world_intervention_transactions=object(),
        world_intervention_transaction_targets=bank.transaction_targets,
        command_intervention_prediction=object(),
        command_intervention_targets=bank.terminal_packet_targets,
        command_intervention_transactions=object(),
        command_intervention_transaction_targets=bank.transaction_targets,
        world_query_binding=ETTRCausalQueryPair(
            correct_logits=logits,
            foil_logits=logits.clone(),
            correct_target=factual_labels.index_select(0, world_target),
            foil_target=factual_labels.index_select(0, world_command),
        ),
        command_query_binding=ETTRCausalQueryPair(
            correct_logits=logits,
            foil_logits=logits.clone(),
            correct_target=factual_labels.index_select(0, command_target),
            foil_target=factual_labels.index_select(0, command_packet),
        ),
        transactions=object(),
        transaction_targets=bank.transaction_targets,
        initial_committed=bank.initial_committed,
        initial_halted=bank.initial_halted,
        equivariance=bank.equivariance,
    )
    transformed = apply_supervision_to_objective(
        objective,
        supervision,
        rectangles,
    )
    assert transformed.token_logits is objective.token_logits
    assert transformed.token_targets is objective.token_targets
    assert transformed.packet_prediction is objective.packet_prediction
    assert transformed.transactions is objective.transactions
    torch.testing.assert_close(
        transformed.world_query_binding.correct_target,
        supervision.answer_labels.index_select(0, world_target),
    )
    torch.testing.assert_close(
        transformed.command_query_binding.correct_target,
        supervision.answer_labels.index_select(0, command_target),
    )
    assert transformed.packet_targets is supervision.packet_targets
    assert transformed.transaction_targets is supervision.transaction_targets


def test_a2_rejects_a_fixed_point_before_target_application() -> None:
    bank, _derangement, rectangle_ids = _target_bank()
    assignments = (
        DerangementAssignment(
            recipient_id=rectangle_ids[0],
            donor_id=rectangle_ids[0],
            donor_rank=0,
            donor_digest="a" * 64,
        ),
        DerangementAssignment(
            recipient_id=rectangle_ids[1],
            donor_id=rectangle_ids[0],
            donor_rank=0,
            donor_digest="b" * 64,
        ),
    )
    attacked = BindingDerangement(
        fold=0,
        assignments=assignments,
        assignment_sha256=hashlib.sha256(
            control_json_bytes([value.as_dict() for value in assignments])
        ).hexdigest(),
    )
    with pytest.raises(ETTRILV2ArmError, match="fixed point"):
        apply_binding_derangement(
            recipient_ids=rectangle_ids,
            derangement=attacked,
            bank=bank,
        )


def test_a4_exact_replacement_and_optimizer_family_ownership(
    production_modules: tuple[
        EndogenousTheoryCompiler,
        GenericTransactionReactor,
        DenseStateReactor,
        SourceDeletedQueryReader,
    ],
) -> None:
    _compiler, treatment, dense, _query_reader = production_modules
    assert sum(value.numel() for value in treatment.parameters()) == 29_757_217
    assert sum(value.numel() for value in dense.parameters()) == 29_757_217
    replacement = dict(dense.named_replacement_parameters())
    assert sum(value.numel() for value in replacement.values()) == 27_302_912
    assert (
        dense.residual_project.dense_head_adapter.weight.numel()
        == DENSE_ADAPTER_PARAMETERS
    )
    assert dense.dense_gru.gate_offsets.shape == (2, 3, 1_241)
    treatment_ledger = parameter_ledger({"reactor": treatment})
    dense_ledger = parameter_ledger({"reactor": dense})
    assert (
        (
            dense_ledger.muon,
            dense_ledger.adamw,
            dense_ledger.unique_trainable,
        )
        == (
            treatment_ledger.muon,
            treatment_ledger.adamw,
            treatment_ledger.unique_trainable,
        )
        == (29_563_904, 193_313, 29_757_217)
    )


def test_a4_tagged_initialization_is_deterministic_and_rng_neutral(
    production_modules: tuple[
        EndogenousTheoryCompiler,
        GenericTransactionReactor,
        DenseStateReactor,
        SourceDeletedQueryReader,
    ],
) -> None:
    _compiler, treatment, dense, _query_reader = production_modules
    before = torch.get_rng_state().clone()
    replay = DenseStateReactor(treatment, fold=0, seed=MODEL_SEED)
    assert torch.equal(torch.get_rng_state(), before)
    first = dict(dense.named_replacement_parameters())
    second = dict(replay.named_replacement_parameters())
    assert first.keys() == second.keys()
    assert all(torch.equal(first[name], second[name]) for name in first)
    changed = DenseStateReactor(treatment, fold=0, seed=MODEL_SEED + 1)
    third = dict(changed.named_replacement_parameters())
    assert any(not torch.equal(first[name], third[name]) for name in first)


def test_a4_rejects_nonproduction_geometry() -> None:
    with pytest.raises(ETTRILV2ArmError, match="production geometry"):
        DenseStateReactor(
            GenericTransactionReactor(_small_config()),
            fold=0,
            seed=MODEL_SEED,
        )


def test_a4_dense_forward_executes_production_geometry(
    production_modules: tuple[
        EndogenousTheoryCompiler,
        GenericTransactionReactor,
        DenseStateReactor,
        SourceDeletedQueryReader,
    ],
) -> None:
    _compiler, _treatment, dense, _query_reader = production_modules
    config = dense.config
    batch = 1
    values = torch.zeros(batch, config.num_slots, config.num_value_codes)
    values[:, :, 0] = 1
    types = torch.zeros(batch, config.num_slots, config.num_types)
    types[:, :, 0] = 1
    active = torch.zeros(batch, config.num_slots)
    active[:, :2] = 1
    root = torch.zeros(batch, config.num_slots)
    root[:, 0] = 1
    initial = TypedTheoryState(
        value_probabilities=values,
        type_probabilities=types,
        relations=torch.zeros(
            batch,
            config.num_relations,
            config.num_slots,
            config.num_slots,
        ),
        active=active,
        root=root,
        committed=torch.zeros(batch),
        halted=torch.zeros(batch),
        step=0,
    )
    validate_state(initial, config)

    flat = dense._flat_state(initial)  # noqa: SLF001
    assert flat.shape == (batch, dense.sketch_columns.numel())
    assert flat.shape[1] == (
        config.num_slots * config.num_value_codes
        + config.num_slots * config.num_types
        + config.num_relations * config.num_slots * config.num_slots
        + 2 * config.num_slots
        + 2
    )
    torch.testing.assert_close(flat[:, -2], initial.committed)
    torch.testing.assert_close(flat[:, -1], initial.halted)

    with torch.inference_mode():
        terminal, trace = dense(initial, steps=1, hard=True)

    validate_state(terminal, config)
    assert terminal.step == 1
    expected_trace_shapes = {
        "opcode": (batch, 1, 9),
        "source": (batch, 1, config.num_slots),
        "target": (batch, 1, config.num_slots),
        "relation": (batch, 1, config.num_relations),
        "type_index": (batch, 1, config.num_types),
        "value_code": (batch, 1, config.num_value_codes),
        "applied_opcode": (batch, 1, 9),
        "applied_source": (batch, 1, config.num_slots),
        "applied_target": (batch, 1, config.num_slots),
        "applied_relation": (batch, 1, config.num_relations),
        "applied_type_index": (batch, 1, config.num_types),
        "applied_value_code": (batch, 1, config.num_value_codes),
        "active": (batch, 1, config.num_slots),
        "committed": (batch, 1),
        "halted": (batch, 1),
    }
    for name, shape in expected_trace_shapes.items():
        value = getattr(trace, name)
        assert value.shape == shape
        assert torch.isfinite(value).all()
    for name in (
        "applied_opcode",
        "applied_source",
        "applied_target",
        "applied_relation",
        "applied_type_index",
        "applied_value_code",
    ):
        value = getattr(trace, name)
        torch.testing.assert_close(
            value.sum(-1),
            torch.ones(batch, 1),
            rtol=0,
            atol=0,
        )


def test_equalizer_factorization_and_all_five_operation_ledgers_are_exact(
    production_modules: tuple[
        EndogenousTheoryCompiler,
        GenericTransactionReactor,
        DenseStateReactor,
        SourceDeletedQueryReader,
    ],
) -> None:
    _compiler, treatment, dense, _query_reader = production_modules
    plans = dense_equalizer_plans(dense, steps=64)
    assert len(plans) == 64
    assert all(
        value.scalar_products == value.reconstructed_scalar_products
        and value.scalar_products > 0
        for value in plans
    )
    ledgers = operation_ledgers(treatment, dense)
    assert set(ledgers) == set(PRIMARY_ARMS)
    assert {
        (
            value.total_forward_scalar_products,
            value.total_backward_scalar_products,
            value.total_training_scalar_products,
            value.common_path_signature,
        )
        for value in ledgers.values()
    }.__len__() == 1
    assert ledgers["dense_state"].equalizer_forward_scalar_products > 0
    assert ledgers["treatment"].equalizer_forward_scalar_products == 0
    assert (
        ledgers["dense_state"].equalizer_backward_scalar_products
        == 2 * ledgers["dense_state"].equalizer_forward_scalar_products
    )


def test_complete_production_receipt_closes_parameters_tokens_and_operations(
    production_modules: tuple[
        EndogenousTheoryCompiler,
        GenericTransactionReactor,
        DenseStateReactor,
        SourceDeletedQueryReader,
    ],
) -> None:
    compiler, treatment, dense, query_reader = production_modules
    receipt = build_arm_equality_receipt(
        compiler,
        treatment,
        dense,
        query_reader,
    )
    receipt.validate()
    assert receipt.weight_updates == 0
    assert receipt.exact_parameter_equality
    assert receipt.exact_token_position_equality
    assert receipt.exact_static_operation_equality
    assert {
        (
            value.muon,
            value.adamw,
            value.unique_trainable,
        )
        for value in receipt.parameter_ledgers.values()
    } == {(67_024_896, 672_875, 67_697_771)}
    token = TokenPositionLedger.production()
    assert token.encoded_positions_per_row == 528
    assert token.encoded_positions_per_update == 67_584
    assert token.supervised_positions_per_update == 42_624


def test_module_has_no_fit_update_checkpoint_or_launch_surface() -> None:
    import ettr_il_v2_arms as module

    source = inspect.getsource(module)
    for forbidden in (
        ".backward(",
        "optimizer.step",
        "torch.save",
        "subprocess",
        "sbatch",
        "submit",
        "save_file",
    ):
        assert forbidden not in source
    assert not any(
        name in module.__all__
        for name in (
            "fit",
            "train",
            "update",
            "save",
            "submit",
        )
    )
