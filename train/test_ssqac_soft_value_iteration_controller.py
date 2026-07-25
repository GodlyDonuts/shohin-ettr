from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from episode_functor_algebra_machine import (
    AlgebraMachineError,
    execute_program,
    verify_reduction_program,
)
import ssqac_soft_value_iteration_controller as soft


def _tiny_config(
    *,
    backup_iterations: int = 2,
    raw_matrix_features: bool = True,
    structural_action_scalars: bool = True,
    pair_relation_features: bool = True,
    message_passing: bool = True,
) -> soft.SoftValueIterationConfig:
    return soft.SoftValueIterationConfig(
        width=16,
        message_layers=1,
        action_hidden=24,
        transition_hidden=24,
        field_harmonics=1,
        coordinate_harmonics=1,
        backup_iterations=backup_iterations,
        temperature=0.5,
        discount=0.9,
        raw_matrix_features=raw_matrix_features,
        structural_action_scalars=structural_action_scalars,
        pair_relation_features=pair_relation_features,
        message_passing=message_passing,
    )


def _tiny_controller(
    *,
    backup_iterations: int = 2,
    raw_matrix_features: bool = True,
    structural_action_scalars: bool = True,
    pair_relation_features: bool = True,
    message_passing: bool = True,
) -> soft.SoftValueIterationController:
    torch.manual_seed(17)
    return soft.SoftValueIterationController(
        _tiny_config(
            backup_iterations=backup_iterations,
            raw_matrix_features=raw_matrix_features,
            structural_action_scalars=structural_action_scalars,
            pair_relation_features=pair_relation_features,
            message_passing=message_passing,
        )
    )


def _preparation_trace(
    matrix: tuple[tuple[int, ...], ...],
    *,
    maximum_steps: int = 64,
) -> tuple[soft.MacroAction, ...]:
    counter = soft.PreparationOracleCounter()
    current = matrix
    actions = []
    for _ in range(maximum_steps):
        action = soft.next_preparation_macro(current, counter=counter)
        assert action in soft.enumerate_legal_macro_actions(current)
        actions.append(action)
        if action.kind == soft.ACTION_HALT:
            return tuple(actions)
        current = soft.apply_macro_action(current, action)
    raise AssertionError("preparation trace did not halt")


def test_halt_is_always_legal_and_does_not_leak_endpoint() -> None:
    nonterminal = ((2, 1), (0, 1))
    terminal = ((1, 0), (0, 1))
    for matrix in (nonterminal, terminal):
        actions = soft.enumerate_legal_macro_actions(matrix)
        assert soft.MacroAction(soft.ACTION_HALT) in actions
        assert actions[-1] == soft.MacroAction(soft.ACTION_HALT)


def test_local_macros_compile_exactly_to_primitive_vm() -> None:
    matrix = ((2, 1, 0), (1, 3, 1))
    normalize = soft.MacroAction(
        soft.ACTION_NORMALIZE,
        row_a=0,
        column=0,
    )
    normalized = soft.apply_macro_action(matrix, normalize)
    normalized_vm = execute_program(
        matrix,
        soft.compile_macro_to_primitives(matrix, normalize),
    )
    assert normalized_vm.rows == normalized

    eliminate = soft.MacroAction(
        soft.ACTION_ELIMINATE,
        row_a=0,
        row_b=1,
        column=0,
    )
    eliminated = soft.apply_macro_action(matrix, eliminate)
    eliminated_vm = execute_program(
        matrix,
        soft.compile_macro_to_primitives(matrix, eliminate),
    )
    assert eliminated_vm.rows == eliminated

    swap = soft.MacroAction(soft.ACTION_SWAP, row_a=0, row_b=1)
    swapped = soft.apply_macro_action(matrix, swap)
    swapped_vm = execute_program(
        matrix,
        soft.compile_macro_to_primitives(matrix, swap),
    )
    assert swapped_vm.rows == swapped


def test_preparation_traces_reach_strict_canonical_rref() -> None:
    matrices = (
        ((0, 1, 2), (2, 0, 1)),
        ((0, 0, 3), (0, 2, 1), (1, 0, 4)),
        ((2, 1, 0, 4), (1, 3, 1, 0), (0, 1, 2, 1)),
        ((0, 0, 0), (4, 0, 2), (0, 5, 1)),
    )
    for matrix in matrices:
        actions = _preparation_trace(matrix)
        assert actions[-1].kind == soft.ACTION_HALT
        program = soft.compile_macro_trace_to_primitives(matrix, actions)
        state = execute_program(matrix, program)
        receipt = verify_reduction_program(matrix, state)
        assert receipt.passed


def test_preparation_dataset_is_exact_and_deduplicated() -> None:
    matrices = (
        ((2, 1, 0), (1, 3, 1)),
        ((4, 0, 1), (0, 5, 1)),
    )
    counter = soft.PreparationOracleCounter()
    states = soft.build_preparation_states(
        matrices,
        maximum_steps=32,
        counter=counter,
    )
    assert states
    assert counter.calls >= len(matrices)
    assert len({state.sha256 for state in states}) == len(states)
    for state in states:
        assert state.target_action in soft.enumerate_legal_macro_actions(state.rows)


def test_random_label_control_is_seeded_and_nonexpert() -> None:
    counter = soft.PreparationOracleCounter()
    states = soft.build_preparation_states(
        (((2, 1, 0), (1, 3, 1)),),
        maximum_steps=32,
        counter=counter,
    )
    first = soft.make_random_label_control(states, seed=23)
    second = soft.make_random_label_control(states, seed=23)
    assert first == second
    for expert, control in zip(states, first, strict=True):
        legal = soft.enumerate_legal_macro_actions(expert.rows)
        if len(legal) > 1:
            assert control.target_action != expert.target_action
        assert control.target_action in legal


def test_architecture_has_only_matrix_and_legal_action_inputs() -> None:
    controller = _tiny_controller()
    forward_parameters = set(inspect.signature(controller.forward).parameters)
    assert forward_parameters == {"rows", "actions"}
    forbidden = {
        "source",
        "query",
        "workspace",
        "oracle",
        "verifier",
        "frontier",
        "beam",
        "previous",
        "step",
    }
    names = tuple(name.lower() for name, _ in controller.named_parameters())
    assert not any(token in name for token in forbidden for name in names)
    assert len(controller.cell_layers) == controller.config.message_layers
    assert isinstance(controller.shared_backup_cell, torch.nn.GRUCell)


def test_parameter_count_is_geometry_independent_and_under_budget() -> None:
    controller = soft.SoftValueIterationController()
    direct = sum(parameter.numel() for parameter in controller.parameters())
    assert controller.parameter_count == direct
    assert controller.parameter_count_breakdown()["total"] == direct
    assert controller.complete_system_parameter_count == (
        soft.PROTECTED_FLAGSHIP_PARAMETERS + direct
    )
    assert controller.complete_system_parameter_count < soft.TOTAL_PARAMETER_BUDGET
    same_parameters = soft.SoftValueIterationController(
        replace_backup_iterations(controller.config, 0)
    )
    assert same_parameters.parameter_count == direct


def replace_backup_iterations(
    config: soft.SoftValueIterationConfig,
    iterations: int,
) -> soft.SoftValueIterationConfig:
    return soft.SoftValueIterationConfig(
        width=config.width,
        message_layers=config.message_layers,
        action_hidden=config.action_hidden,
        transition_hidden=config.transition_hidden,
        field_harmonics=config.field_harmonics,
        coordinate_harmonics=config.coordinate_harmonics,
        backup_iterations=iterations,
        temperature=config.temperature,
        discount=config.discount,
    )


def test_shared_backup_cell_runs_exactly_fixed_internal_iterations() -> None:
    matrix = ((2, 1, 0), (1, 3, 1))
    controller = _tiny_controller(backup_iterations=3)
    calls = 0

    def count_call(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        nonlocal calls
        calls += 1

    handle = controller.shared_backup_cell.register_forward_hook(count_call)
    scored = controller.score_actions(matrix)
    handle.remove()
    assert calls == 3
    assert scored.internal_backup_iterations == 3
    assert scored.action_value_backups == 3 * len(scored.actions)
    assert scored.logits.shape == (len(scored.actions),)
    assert torch.isfinite(scored.logits).all()

    zero = _tiny_controller(backup_iterations=0)
    zero.load_state_dict(controller.state_dict())
    zero_calls = 0

    def count_zero(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        nonlocal zero_calls
        zero_calls += 1

    handle = zero.shared_backup_cell.register_forward_hook(count_zero)
    zero_scores = zero.score_actions(matrix)
    handle.remove()
    assert zero_calls == 0
    assert zero_scores.internal_backup_iterations == 0
    torch.testing.assert_close(
        zero_scores.logits,
        zero_scores.local_reward,
    )


def test_controller_accepts_strictly_larger_geometry_without_new_tables() -> None:
    controller = _tiny_controller()
    small = ((2, 1, 0), (1, 3, 1))
    large = (
        (2, 1, 0, 4, 0, 1),
        (1, 3, 1, 0, 2, 0),
        (0, 1, 2, 1, 0, 3),
        (4, 0, 1, 0, 1, 2),
    )
    small_scores = controller.score_actions(small)
    large_scores = controller.score_actions(large)
    assert small_scores.logits.numel() == len(small_scores.actions)
    assert large_scores.logits.numel() == len(large_scores.actions)
    assert torch.isfinite(large_scores.logits).all()


def test_training_is_bounded_and_updates_the_model() -> None:
    counter = soft.PreparationOracleCounter()
    states = soft.build_preparation_states(
        (((2, 1, 0), (1, 3, 1)),),
        maximum_steps=32,
        counter=counter,
    )
    controller = _tiny_controller(backup_iterations=1)
    before = soft.model_state_sha256(controller)
    updates = soft.train_controller(
        controller,
        states,
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        maximum_updates=2,
        shuffle_seed=31,
    )
    after = soft.model_state_sha256(controller)
    assert updates == 2
    assert before != after
    assert 0.0 <= soft.label_accuracy(controller, states) <= 1.0


def test_candidate_rollout_has_no_oracle_search_or_verifier_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _tiny_controller(backup_iterations=2)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("forbidden evaluator leaked into candidate")

    monkeypatch.setattr(soft, "next_preparation_macro", forbidden)
    monkeypatch.setattr(soft, "verify_reduction_program", forbidden)
    monkeypatch.setattr(soft, "assess_candidate_rollout", forbidden)
    source = inspect.getsource(soft.candidate_matrix_only_rollout)
    assert "next_preparation_macro" not in source
    assert "verify_reduction_program" not in source
    assert "assess_candidate_rollout" not in source
    assert "beam" not in source
    assert "frontier" not in source
    rollout = soft.candidate_matrix_only_rollout(
        controller,
        ((2, 1, 0), (1, 3, 1)),
        maximum_steps=3,
    )
    assert rollout.audit.oracle_calls == 0
    assert rollout.audit.search_calls == 0
    assert rollout.audit.verifier_calls == 0
    assert rollout.audit.internal_backup_iterations == (
        rollout.audit.model_decisions * 2
    )


def test_separate_assessor_rejects_premature_and_noncanonical_halt() -> None:
    identity = ((1, 0), (0, 1))
    good = soft.CandidateRollout(
        halted=True,
        invalid=False,
        overlong=False,
        actions=(soft.MacroAction(soft.ACTION_HALT),),
        output_rows=identity,
        audit=soft.CandidateRuntimeAudit(1, 0, 0, 0, 0, 0, 0),
    )
    assert soft.assess_candidate_rollout(
        identity,
        good,
    ).strict_canonical_certified

    premature_matrix = ((2, 0), (0, 1))
    premature = soft.CandidateRollout(
        halted=True,
        invalid=False,
        overlong=False,
        actions=(soft.MacroAction(soft.ACTION_HALT),),
        output_rows=premature_matrix,
        audit=soft.CandidateRuntimeAudit(1, 0, 0, 0, 0, 0, 0),
    )
    rejected = soft.assess_candidate_rollout(
        premature_matrix,
        premature,
    )
    assert not rejected.strict_canonical_certified
    assert rejected.invalid

    unordered = ((0, 1), (1, 0))
    noncanonical = soft.CandidateRollout(
        halted=True,
        invalid=False,
        overlong=False,
        actions=(soft.MacroAction(soft.ACTION_HALT),),
        output_rows=unordered,
        audit=soft.CandidateRuntimeAudit(1, 0, 0, 0, 0, 0, 0),
    )
    rejected = soft.assess_candidate_rollout(unordered, noncanonical)
    assert not rejected.strict_canonical_certified
    assert rejected.invalid


def test_complete_trace_requires_exactly_one_final_halt() -> None:
    matrix = ((2, 0), (0, 1))
    normalize = soft.MacroAction(
        soft.ACTION_NORMALIZE,
        row_a=0,
        column=0,
    )
    with pytest.raises(soft.SoftValueIterationError, match="terminate"):
        soft.compile_macro_trace_to_primitives(matrix, (normalize,))
    with pytest.raises(soft.SoftValueIterationError, match="after HALT"):
        soft.compile_macro_trace_to_primitives(
            matrix,
            (
                soft.MacroAction(soft.ACTION_HALT),
                normalize,
                soft.MacroAction(soft.ACTION_HALT),
            ),
        )


def test_strict_larger_geometry_gate_rejects_overlap() -> None:
    with pytest.raises(
        soft.SoftValueIterationError,
        match="strictly larger",
    ):
        soft.SoftValueExperimentConfig(
            train_maximum_rows=3,
            train_maximum_columns=4,
            evaluation_minimum_rows=3,
            evaluation_minimum_columns=5,
        )
    with pytest.raises(
        soft.SoftValueIterationError,
        match="strictly larger",
    ):
        soft.SoftValueExperimentConfig(
            train_maximum_rows=3,
            train_maximum_columns=4,
            evaluation_minimum_rows=4,
            evaluation_minimum_columns=4,
        )


def test_tiny_experiment_is_deterministic_and_reports_all_controls() -> None:
    config = soft.SoftValueExperimentConfig(
        seed=41,
        train_matrices=2,
        evaluation_matrices=2,
        train_maximum_rows=2,
        train_maximum_columns=3,
        evaluation_minimum_rows=3,
        evaluation_minimum_columns=4,
        evaluation_maximum_rows=3,
        evaluation_maximum_columns=4,
        maximum_preparation_steps=32,
        maximum_rollout_steps=4,
        epochs=1,
        batch_size=16,
        learning_rate=1e-3,
        maximum_updates=1,
        controller=soft.SoftValueIterationConfig(
            width=8,
            message_layers=1,
            action_hidden=12,
            transition_hidden=12,
            field_harmonics=1,
            coordinate_harmonics=1,
            backup_iterations=1,
            temperature=0.5,
            discount=0.9,
        ),
    )
    first = soft.run_bounded_experiment(config)
    second = soft.run_bounded_experiment(config)
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.status == soft.STATUS
    assert first.outcome == soft.OUTCOME_INCONCLUSIVE
    assert first.device == "cpu"
    assert first.candidate_input_fields == (
        "matrix",
        "legal_local_action_features",
    )
    assert not first.candidate_has_host_search
    assert not first.candidate_has_verifier
    assert not first.candidate_has_oracle
    assert first.fixed_shared_weight_recurrence
    assert first.treatment_backup_iterations_per_decision == 1
    assert first.random_control_backup_iterations_per_decision == 1
    assert first.zero_control_backup_iterations_per_decision == 0
    assert first.parameter_budget_passed
    assert first.strict_geometry_disjoint
    assert first.final_candidate_oracle_calls == 0
    assert first.final_candidate_search_calls == 0
    assert first.final_candidate_verifier_calls == 0
    assert first.no_oracle_no_search_no_verifier_gate_passed
    assert first.treatment_internal_backup_iterations == (
        first.treatment_model_decisions
    )
    for certified, invalid, overlong in (
        (
            first.treatment_strict_canonical_certified,
            first.treatment_invalid,
            first.treatment_overlong,
        ),
        (
            first.random_control_strict_canonical_certified,
            first.random_control_invalid,
            first.random_control_overlong,
        ),
        (
            first.zero_control_strict_canonical_certified,
            first.zero_control_invalid,
            first.zero_control_overlong,
        ),
    ):
        assert certified + invalid + overlong == first.evaluation_matrices
    assert not first.material_gate_passed


def test_assessor_retains_existing_strict_endpoint_errors() -> None:
    unordered = ((0, 1), (1, 0))
    actions = (soft.MacroAction(soft.ACTION_HALT),)
    program = soft.compile_macro_trace_to_primitives(unordered, actions)
    state = execute_program(unordered, program)
    with pytest.raises(AlgebraMachineError, match="pivot order"):
        verify_reduction_program(unordered, state)


def test_structural_action_scalar_ablation_retains_only_type_and_operands() -> None:
    matrix = ((2, 0, 1), (1, 3, 0))
    full = _tiny_controller()
    removed = _tiny_controller(structural_action_scalars=False)
    removed.load_state_dict(full.state_dict())
    saw_structural_signal = False
    for action in soft.enumerate_legal_macro_actions(matrix):
        full_values = full._action_scalars(matrix, action)
        removed_values = removed._action_scalars(matrix, action)
        assert (
            removed_values[: soft.MINIMAL_ACTION_SCALAR_FEATURES]
            == full_values[: soft.MINIMAL_ACTION_SCALAR_FEATURES]
        )
        assert removed_values[soft.MINIMAL_ACTION_SCALAR_FEATURES :] == (0.0,) * (
            soft.ACTION_SCALAR_FEATURES - soft.MINIMAL_ACTION_SCALAR_FEATURES
        )
        saw_structural_signal |= any(full_values[soft.MINIMAL_ACTION_SCALAR_FEATURES :])
    assert saw_structural_signal
    assert full.parameter_count == removed.parameter_count
    assert removed.config.raw_matrix_features


def test_raw_matrix_ablation_removes_coefficients_but_retains_geometry() -> None:
    full = _tiny_controller()
    removed = _tiny_controller(raw_matrix_features=False)
    removed.load_state_dict(full.state_dict())
    first = torch.tensor(((2, 0, 1), (1, 3, 0)), dtype=torch.long)
    second = torch.tensor(((7, 8, 9), (10, 11, 12)), dtype=torch.long)
    with torch.no_grad():
        torch.testing.assert_close(
            removed.encode_matrix(first),
            removed.encode_matrix(second),
            rtol=0.0,
            atol=0.0,
        )
        assert not torch.equal(
            full.encode_matrix(first),
            full.encode_matrix(second),
        )
    assert removed.parameter_count == full.parameter_count


def test_pair_and_message_ablation_controls_are_disjoint() -> None:
    matrix = ((2, 1, 0), (1, 3, 1))
    actions = soft.enumerate_legal_macro_actions(matrix)
    full = _tiny_controller(backup_iterations=2)
    pair_removed = _tiny_controller(
        backup_iterations=2,
        pair_relation_features=False,
    )
    message_disabled = _tiny_controller(
        backup_iterations=2,
        message_passing=False,
    )
    for controller in (pair_removed, message_disabled):
        controller.load_state_dict(full.state_dict())
    full_relations, full_neighbors = full._pair_relations(
        actions,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    removed_relations, removed_neighbors = pair_removed._pair_relations(
        actions,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    disabled_relations, disabled_neighbors = message_disabled._pair_relations(
        actions,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.count_nonzero(full_relations)
    assert torch.count_nonzero(removed_relations) == 0
    torch.testing.assert_close(full_neighbors, removed_neighbors)
    torch.testing.assert_close(full_relations, disabled_relations)
    torch.testing.assert_close(
        disabled_neighbors,
        torch.eye(len(actions), dtype=torch.bool),
    )
    assert full.parameter_count == pair_removed.parameter_count
    assert full.parameter_count == message_disabled.parameter_count


def test_one_step_and_message_disabled_controls_have_exact_compute() -> None:
    matrix = ((2, 1, 0), (1, 3, 1))
    full = _tiny_controller(backup_iterations=3)
    one_step = _tiny_controller(backup_iterations=1)
    disabled = _tiny_controller(
        backup_iterations=3,
        message_passing=False,
    )
    for controller in (one_step, disabled):
        controller.load_state_dict(full.state_dict())
    one = one_step.score_actions(matrix)
    blocked = disabled.score_actions(matrix)
    action_count = len(one.actions)
    assert one.resources.internal_backup_iterations == 1
    assert one.resources.action_value_updates == action_count
    assert one.resources.transition_pairs_evaluated == action_count**2
    assert blocked.resources.internal_backup_iterations == 3
    assert blocked.resources.action_value_updates == 3 * action_count
    assert blocked.resources.active_message_edges == 3 * action_count
    assert blocked.resources.transition_pairs_evaluated == (3 * action_count**2)
    assert full.parameter_count == one_step.parameter_count
    assert full.parameter_count == disabled.parameter_count


def test_forward_resource_counts_are_exact_on_known_matrix() -> None:
    matrix = ((2, 1, 0), (1, 3, 1))
    controller = _tiny_controller(backup_iterations=2)
    scored = controller.score_actions(matrix)
    action_count = len(scored.actions)
    resources = scored.resources
    assert resources.model_forward_calls == 1
    assert resources.matrix_cells_encoded == 6
    assert resources.raw_matrix_feature_values == 6 * (5 + 4)
    assert resources.coordinate_feature_values == 6 * 2 * (2 + 2)
    assert resources.matrix_message_cell_updates == 6
    assert resources.action_nodes_scored == action_count
    assert resources.minimal_action_scalar_values == 7 * action_count
    assert resources.structural_action_scalar_values == 13 * action_count
    assert resources.transition_pairs_evaluated == (2 * action_count**2)
    assert resources.pair_relation_feature_values == (2 * action_count**2 * 12)
    assert resources.internal_backup_iterations == 2
    assert resources.action_value_updates == 2 * action_count


def test_training_receipt_counts_presentations_and_forwards_exactly() -> None:
    counter = soft.PreparationOracleCounter()
    states = soft.build_preparation_states(
        (((2, 1, 0), (1, 3, 1)),),
        maximum_steps=32,
        counter=counter,
    )
    controller = _tiny_controller(backup_iterations=1)
    receipt = soft.train_controller_with_receipt(
        controller,
        states,
        epochs=1,
        batch_size=len(states),
        learning_rate=1e-3,
        maximum_updates=1,
        shuffle_seed=101,
    )
    assert receipt.optimizer_updates == 1
    assert receipt.labeled_state_presentations == len(states)
    assert receipt.parameters == controller.parameter_count
    assert receipt.resources.model_forward_calls == len(states)
    assert receipt.resources.internal_backup_iterations == len(states)


def test_action_renderer_order_is_logit_equivariant() -> None:
    matrix = ((2, 1, 0), (1, 3, 1))
    controller = _tiny_controller(backup_iterations=2)
    canonical = controller.score_actions(matrix)
    reversed_actions = tuple(reversed(canonical.actions))
    recoded = controller.score_actions(matrix, reversed_actions)
    canonical_logits = {
        action: canonical.logits[index]
        for index, action in enumerate(canonical.actions)
    }
    recoded_logits = {
        action: recoded.logits[index] for index, action in enumerate(recoded.actions)
    }
    for action in canonical.actions:
        torch.testing.assert_close(
            canonical_logits[action],
            recoded_logits[action],
            rtol=1e-5,
            atol=1e-6,
        )


def test_field_representative_and_action_renderer_rollouts_match() -> None:
    matrix = ((2, 1, 0), (1, 3, 1))
    controller = _tiny_controller(backup_iterations=2)
    canonical = soft.candidate_matrix_only_rollout(
        controller,
        matrix,
        maximum_steps=4,
    )
    representative = soft.candidate_matrix_only_rollout(
        controller,
        soft.representative_recode_matrix(matrix, seed=103),
        maximum_steps=4,
    )
    renderer = soft.candidate_matrix_only_rollout(
        controller,
        matrix,
        maximum_steps=4,
        action_renderer_seed=107,
    )
    assert soft._same_rollout(canonical, representative)
    assert soft._same_rollout(canonical, renderer)


def test_legal_action_set_is_equivariant_under_matrix_permutation() -> None:
    matrix = ((2, 0, 1), (1, 3, 0), (0, 1, 4))
    row_order = (2, 0, 1)
    column_order = (1, 2, 0)
    permuted = soft.permute_matrix(
        matrix,
        row_order=row_order,
        column_order=column_order,
    )
    expected = {
        soft.remap_action_under_permutation(
            action,
            row_order=row_order,
            column_order=column_order,
        )
        for action in soft.enumerate_legal_macro_actions(matrix)
    }
    assert expected == set(soft.enumerate_legal_macro_actions(permuted))


def test_feature_leakage_audit_reports_conservative_lower_bounds() -> None:
    counter = soft.PreparationOracleCounter()
    states = soft.build_preparation_states(
        (((2, 0, 1), (1, 3, 0)),),
        maximum_steps=32,
        counter=counter,
    )
    audit = soft.audit_feature_leakage(states)
    assert audit.states == len(states)
    assert 0 < audit.legal_nonzero_cells_revealed <= audit.nonzero_cells
    assert audit.legal_unit_cells_revealed > 0
    assert audit.legal_nonunit_cells_revealed > 0
    assert audit.full_scalar_exact_coefficient_cells == (
        audit.legal_nonzero_cells_revealed
    )
    assert 0.0 < audit.legal_nonzero_recall <= 1.0
    assert audit.pair_relation_bits == sum(
        len(soft.enumerate_legal_macro_actions(state.rows)) ** 2
        * soft.PAIR_RELATION_FEATURES
        for state in states
    )
    assert audit.positive_pair_relation_bits > 0
    assert audit.message_graph_edges > 0


def test_tiny_hostile_audit_has_matched_arms_and_exact_recodings() -> None:
    config = soft.SoftValueExperimentConfig(
        seed=109,
        train_matrices=2,
        evaluation_matrices=2,
        train_maximum_rows=2,
        train_maximum_columns=3,
        evaluation_minimum_rows=3,
        evaluation_minimum_columns=4,
        evaluation_maximum_rows=3,
        evaluation_maximum_columns=4,
        maximum_preparation_steps=32,
        maximum_rollout_steps=4,
        epochs=1,
        batch_size=64,
        learning_rate=1e-3,
        maximum_updates=1,
        controller=_tiny_config(backup_iterations=2),
    )
    report = soft.run_hostile_audit_experiment(config)
    assert report.schema == soft.HOSTILE_AUDIT_SCHEMA
    assert report.status == soft.HOSTILE_AUDIT_STATUS
    assert report.outcome == soft.HOSTILE_AUDIT_OUTCOME
    assert report.all_arm_parameter_counts_equal
    assert report.all_arm_initializations_equal
    assert report.all_arm_optimizer_updates_equal
    assert report.all_arm_state_presentations_equal
    assert report.parameter_budget_passed
    assert report.no_oracle_no_search_no_verifier_gate_passed
    arms = {arm.name: arm for arm in report.arms}
    assert set(arms) == {
        "treatment",
        "structural_action_scalars_removed",
        "raw_matrix_removed",
        "pair_relation_features_removed",
        "legal_operands_types_only",
        "one_backup_iteration",
        "message_passing_disabled",
        "zero_backup_iterations",
        "random_labels",
    }
    assert not arms["structural_action_scalars_removed"].controller_config[
        "structural_action_scalars"
    ]
    assert not arms["raw_matrix_removed"].controller_config["raw_matrix_features"]
    assert arms["one_backup_iteration"].controller_config["backup_iterations"] == 1
    assert not arms["message_passing_disabled"].controller_config["message_passing"]
    for arm in arms.values():
        assert arm.evaluation.total == report.evaluation_matrices
        assert arm.no_oracle_no_search_no_verifier_gate_passed
        assert arm.training_resources.optimizer_updates == 1
    assert report.recoding.action_order_trace_matches == 2
    assert report.recoding.action_order_assessment_matches == 2
    assert report.recoding.representative_trace_matches == 2
    assert report.recoding.representative_assessment_matches == 2
    assert report.recoding.legal_action_permutation_matches == 2
    assert report.recoding.permuted_evaluation.total == 2


def test_hostile_job_is_isolated_and_requests_exact_resources() -> None:
    job = (
        Path(__file__).parent
        / "jobs"
        / "ssqac_soft_value_iteration_hostile_audit.sbatch"
    ).read_text(encoding="ascii")
    assert "#SBATCH --cpus-per-task=4" in job
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in job
    assert "#SBATCH --mem=96G" in job
    assert "#SBATCH --time=08:00:00" in job
    assert "--hostile-audit" in job
    assert "ssqac_soft_value_iteration_hostile" in job
    assert "train.py" not in job
    assert "ckpt_" not in job
    assert "flagship_out" not in job
    assert "sbatch " not in job
