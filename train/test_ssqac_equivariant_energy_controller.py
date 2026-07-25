from __future__ import annotations

import inspect

import pytest
import torch

from episode_functor_algebra_machine import (
    AlgebraMachineError,
    execute_program,
    verify_reduction_program,
)
import ssqac_equivariant_energy_controller as energy


def _tiny_controller() -> energy.EquivariantEnergyController:
    torch.manual_seed(7)
    return energy.EquivariantEnergyController(
        energy.EnergyControllerConfig(
            width=16,
            message_layers=1,
            residual_hidden=24,
            field_harmonics=2,
            residual_bound=0.49,
        )
    )


def _old_to_new(permutation: tuple[int, ...]) -> dict[int, int]:
    return {old: new for new, old in enumerate(permutation)}


def _permute_matrix(
    matrix: tuple[tuple[int, ...], ...],
    row_permutation: tuple[int, ...],
    column_permutation: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(matrix[old_row][old_column] for old_column in column_permutation)
        for old_row in row_permutation
    )


def _remap_action(
    action: energy.LocalEnergyAction,
    row_permutation: tuple[int, ...],
    column_permutation: tuple[int, ...],
) -> energy.LocalEnergyAction:
    if action.kind == energy.ACTION_HALT:
        return action
    rows = _old_to_new(row_permutation)
    columns = _old_to_new(column_permutation)
    return energy.LocalEnergyAction(
        action.kind,
        row_a=rows[action.row_a],
        row_b=rows[action.row_b] if action.kind == energy.ACTION_ELIMINATE else 0,
        column=columns[action.column],
    )


def test_explicit_energy_is_permutation_invariant_and_exact_at_terminal() -> None:
    unordered_basis = ((0, 1, 4), (1, 0, 3), (0, 0, 0))
    assert energy.defect_energy(unordered_basis) == 0
    assert energy.field_rank(unordered_basis) == 2
    recoded = _permute_matrix(
        unordered_basis,
        (2, 0, 1),
        (2, 0, 1),
    )
    assert energy.defect_energy(recoded) == 0

    defective = ((2, 1, 0), (1, 1, 1))
    row_recoded = _permute_matrix(defective, (1, 0), (0, 1, 2))
    column_recoded = _permute_matrix(defective, (0, 1), (2, 0, 1))
    assert energy.defect_energy(defective) > 0
    assert energy.defect_energy(row_recoded) == energy.defect_energy(defective)
    assert energy.defect_energy(column_recoded) == energy.defect_energy(defective)


def test_hard_actions_compile_exactly_to_primitive_vm() -> None:
    matrix = ((2, 1, 0), (1, 3, 1))
    normalize = energy.LocalEnergyAction(
        energy.ACTION_NORMALIZE,
        row_a=0,
        column=0,
    )
    normalized = energy.apply_local_action(matrix, normalize)
    vm_normalized = execute_program(
        matrix,
        energy.compile_action_to_vm_primitives(matrix, normalize),
    )
    assert vm_normalized.rows == normalized

    eliminate = energy.LocalEnergyAction(
        energy.ACTION_ELIMINATE,
        row_a=0,
        row_b=1,
        column=0,
    )
    eliminated = energy.apply_local_action(matrix, eliminate)
    vm_eliminated = execute_program(
        matrix,
        energy.compile_action_to_vm_primitives(matrix, eliminate),
    )
    assert vm_eliminated.rows == eliminated


def test_illegal_local_actions_fail_closed() -> None:
    matrix = ((1, 0), (0, 1))
    with pytest.raises(energy.EnergyControllerError, match="zero energy"):
        energy.apply_local_action(
            ((2, 0), (0, 1)),
            energy.LocalEnergyAction(energy.ACTION_HALT),
        )
    with pytest.raises(energy.EnergyControllerError, match="nonunit"):
        energy.apply_local_action(
            matrix,
            energy.LocalEnergyAction(
                energy.ACTION_NORMALIZE,
                row_a=0,
                column=0,
            ),
        )
    with pytest.raises(energy.EnergyControllerError, match="must differ"):
        energy.apply_local_action(
            ((1, 1), (1, 2)),
            energy.LocalEnergyAction(
                energy.ACTION_ELIMINATE,
                row_a=0,
                row_b=0,
                column=0,
            ),
        )


def test_architecture_has_no_forbidden_state_or_geometry_parameters() -> None:
    controller = _tiny_controller()
    forward_parameters = set(inspect.signature(controller.forward).parameters)
    forbidden = {
        "source",
        "query",
        "workspace",
        "hidden",
        "previous",
        "step",
        "row_positions",
        "column_positions",
    }
    assert not forward_parameters.intersection(forbidden)
    names = tuple(name.lower() for name, _ in controller.named_parameters())
    assert not any(
        token in name
        for name in names
        for token in ("position", "recurrent", "step", "workspace", "query")
    )


def test_parameter_count_is_exact_and_geometry_independent() -> None:
    controller = _tiny_controller()
    direct = sum(parameter.numel() for parameter in controller.parameters())
    breakdown = controller.parameter_count_breakdown()
    assert controller.parameter_count == direct
    assert breakdown["total"] == direct
    assert direct == 8_889

    same_architecture = energy.EquivariantEnergyController(controller.config)
    assert same_architecture.parameter_count == direct


def test_scores_are_row_and_column_permutation_equivariant() -> None:
    controller = _tiny_controller().eval()
    matrix = ((2, 1, 0, 4), (1, 3, 1, 0), (0, 1, 2, 1))
    row_permutation = (2, 0, 1)
    column_permutation = (3, 1, 0, 2)
    recoded = _permute_matrix(
        matrix,
        row_permutation,
        column_permutation,
    )
    original_scores = controller.score_actions(matrix)
    recoded_scores = controller.score_actions(recoded)
    recoded_lookup = {
        action: index for index, action in enumerate(recoded_scores.actions)
    }
    assert original_scores.energy_before == recoded_scores.energy_before
    for index, action in enumerate(original_scores.actions):
        mapped = _remap_action(
            action,
            row_permutation,
            column_permutation,
        )
        mapped_index = recoded_lookup[mapped]
        assert original_scores.energy_after[index] == (
            recoded_scores.energy_after[mapped_index]
        )
        torch.testing.assert_close(
            original_scores.learned_residual[index],
            recoded_scores.learned_residual[mapped_index],
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            original_scores.total_score[index],
            recoded_scores.total_score[mapped_index],
            rtol=1e-5,
            atol=1e-6,
        )


def test_bounded_residual_cannot_reverse_integer_energy_advantage() -> None:
    controller = _tiny_controller()
    matrix = ((2, 1, 0), (1, 3, 1))
    scores = controller.score_actions(matrix)
    for left in range(len(scores.actions)):
        for right in range(len(scores.actions)):
            explicit_gap = (
                scores.explicit_reduction[left]
                - scores.explicit_reduction[right]
            )
            if explicit_gap >= 1:
                assert scores.total_score[left] > scores.total_score[right]


def test_terminal_rollout_certifies_without_ordering_rows() -> None:
    controller = _tiny_controller()
    terminal = ((0, 1, 4), (1, 0, 3), (0, 0, 0))
    result = energy.final_oracle_free_rollout(
        controller,
        terminal,
        maximum_steps=2,
    )
    assert result.certified
    assert result.unordered_certified
    assert not result.strict_canonical_certified
    assert result.final_energy == 0
    assert result.actions == (energy.LocalEnergyAction(energy.ACTION_HALT),)
    assert result.oracle_calls == 0


def test_unordered_endpoint_is_explicitly_rejected_by_canonical_rref() -> None:
    matrix = ((0, 1, 4), (1, 0, 3), (0, 0, 0))
    actions = (energy.LocalEnergyAction(energy.ACTION_HALT),)
    provenance = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert energy.verify_unordered_reduced_basis(
        matrix,
        matrix,
        provenance,
    )
    program = energy.compile_action_trace_to_vm_primitives(matrix, actions)
    assert program[-1].opcode == energy.OP_HALT
    state = execute_program(matrix, program)
    with pytest.raises(AlgebraMachineError, match="pivot order"):
        verify_reduction_program(matrix, state)
    assert not energy.verify_strict_canonical_action_trace(matrix, actions)


def test_canonical_endpoint_passes_both_certificates() -> None:
    matrix = ((1, 0, 3), (0, 1, 4), (0, 0, 0))
    actions = (energy.LocalEnergyAction(energy.ACTION_HALT),)
    result = energy.final_oracle_free_rollout(
        _tiny_controller(),
        matrix,
        maximum_steps=2,
    )
    assert result.unordered_certified
    assert result.strict_canonical_certified
    assert energy.verify_strict_canonical_action_trace(matrix, actions)


def test_complete_vm_trace_requires_one_final_halt() -> None:
    matrix = ((2, 0), (0, 1))
    normalize = energy.LocalEnergyAction(
        energy.ACTION_NORMALIZE,
        row_a=0,
        column=0,
    )
    with pytest.raises(energy.EnergyControllerError, match="terminate"):
        energy.compile_action_trace_to_vm_primitives(matrix, (normalize,))
    with pytest.raises(energy.EnergyControllerError, match="after HALT"):
        energy.compile_action_trace_to_vm_primitives(
            matrix,
            (
                energy.LocalEnergyAction(energy.ACTION_HALT),
                normalize,
                energy.LocalEnergyAction(energy.ACTION_HALT),
            ),
        )


def test_final_rollout_has_no_oracle_call_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _tiny_controller()

    def forbidden(*_args: object, **_kwargs: object) -> tuple[int, ...]:
        raise AssertionError("preparation oracle leaked into final rollout")

    monkeypatch.setattr(energy, "expert_action_indices", forbidden)
    assert (
        "expert_action_indices"
        not in energy.final_oracle_free_rollout.__code__.co_names
    )
    result = energy.final_oracle_free_rollout(
        controller,
        ((2, 0), (0, 3)),
        maximum_steps=8,
        learned_residual=False,
    )
    assert result.certified
    assert result.strict_canonical_certified
    assert result.oracle_calls == 0


def test_expert_and_random_labels_are_bounded_controls() -> None:
    matrices = (
        ((2, 1, 0), (1, 3, 1)),
        ((4, 0, 1), (0, 5, 1)),
    )
    counter = energy.OracleCounter()
    expert = energy.build_expert_states(
        matrices,
        maximum_steps=8,
        counter=counter,
    )
    random_left = energy.make_random_label_control(expert, seed=11)
    random_right = energy.make_random_label_control(expert, seed=11)
    assert expert
    assert counter.calls > 0
    assert random_left == random_right
    for original, state in zip(expert, random_left, strict=True):
        transitions = energy.evaluate_transitions(state.rows)
        assert 0 <= state.target_indices[0] < len(transitions)
        if len(transitions) > len(original.target_indices):
            assert state.target_indices[0] not in original.target_indices


def test_training_is_bounded_and_changes_only_trainable_residual() -> None:
    controller = _tiny_controller()
    counter = energy.OracleCounter()
    states = energy.build_expert_states(
        (((2, 1, 0), (1, 3, 1)),),
        maximum_steps=6,
        counter=counter,
    )
    before = energy.model_state_sha256(controller)
    updates = energy.train_energy_controller(
        controller,
        states,
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        shuffle_seed=9,
        maximum_updates=2,
    )
    after = energy.model_state_sha256(controller)
    assert updates == 2
    assert before != after
    assert 0.0 <= energy.label_accuracy(controller, states) <= 1.0


def test_strict_larger_geometry_gate_rejects_overlap() -> None:
    with pytest.raises(energy.EnergyControllerError, match="strictly larger"):
        energy.EnergyExperimentConfig(
            train_maximum_rows=3,
            train_maximum_columns=4,
            evaluation_minimum_rows=3,
            evaluation_minimum_columns=5,
        )
    with pytest.raises(energy.EnergyControllerError, match="strictly larger"):
        energy.EnergyExperimentConfig(
            train_maximum_rows=3,
            train_maximum_columns=4,
            evaluation_minimum_rows=4,
            evaluation_minimum_columns=4,
        )


def test_tiny_experiment_is_deterministic_and_reports_all_controls() -> None:
    config = energy.EnergyExperimentConfig(
        seed=31,
        train_matrices=2,
        evaluation_matrices=2,
        train_maximum_rows=2,
        train_maximum_columns=3,
        evaluation_minimum_rows=3,
        evaluation_minimum_columns=4,
        evaluation_maximum_rows=3,
        evaluation_maximum_columns=4,
        maximum_expert_steps=4,
        maximum_rollout_steps=6,
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        maximum_updates=1,
        controller=energy.EnergyControllerConfig(
            width=8,
            message_layers=1,
            residual_hidden=12,
            field_harmonics=1,
            residual_bound=0.49,
        ),
    )
    first = energy.run_bounded_experiment(config)
    second = energy.run_bounded_experiment(config)
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.status == energy.STATUS
    assert first.final_rollout_oracle_calls == 0
    assert first.evaluation_minimum_rows > first.train_maximum_rows
    assert first.evaluation_minimum_columns > first.train_maximum_columns
    assert first.energy_only_unordered_certified + (
        first.energy_only_invalid
    ) + first.energy_only_overlong == first.evaluation_matrices
    assert first.expert_model_unordered_certified + (
        first.expert_model_invalid
    ) + first.expert_model_overlong == first.evaluation_matrices
    assert first.random_label_model_unordered_certified + (
        first.random_label_model_invalid
    ) + first.random_label_model_overlong == first.evaluation_matrices
    assert first.energy_only_strict_canonical_certified <= (
        first.energy_only_unordered_certified
    )
    assert first.expert_model_strict_canonical_certified <= (
        first.expert_model_unordered_certified
    )
    assert first.random_label_model_strict_canonical_certified <= (
        first.random_label_model_unordered_certified
    )
    assert first.controller_parameters == (
        first.parameter_count_breakdown["total"]
    )


def test_matrix_generation_and_manifests_are_deterministic_and_disjoint() -> None:
    train = energy.generate_matrices(
        seed=1,
        count=3,
        minimum_rows=2,
        maximum_rows=2,
        minimum_columns=2,
        maximum_columns=3,
    )
    evaluation = energy.generate_matrices(
        seed=2,
        count=3,
        minimum_rows=3,
        maximum_rows=3,
        minimum_columns=4,
        maximum_columns=4,
        excluded=set(train),
    )
    assert set(train).isdisjoint(evaluation)
    assert energy.matrix_manifest(train) == energy.matrix_manifest(train)
    assert energy.matrix_manifest(train) != energy.matrix_manifest(evaluation)


def test_config_and_matrix_bounds_fail_closed() -> None:
    with pytest.raises(energy.EnergyControllerError, match="residual_bound"):
        energy.EnergyControllerConfig(residual_bound=0.5)
    with pytest.raises(energy.EnergyControllerError, match="at most"):
        energy.defect_energy(((1,) * 13,))
    with pytest.raises(energy.EnergyControllerError, match="inconsistent"):
        energy.canonical_matrix(((1, 2), (3,)))


def test_random_control_model_starts_from_identical_parameters() -> None:
    config = energy.EnergyControllerConfig(
        width=8,
        message_layers=1,
        residual_hidden=12,
        field_harmonics=1,
    )
    first, second = energy._fresh_identical_controllers(config, seed=19)
    assert energy.model_state_sha256(first) == energy.model_state_sha256(second)
    with torch.no_grad():
        next(second.parameters()).add_(1.0)
    assert energy.model_state_sha256(first) != energy.model_state_sha256(second)


def test_replacing_geometry_does_not_change_parameter_count() -> None:
    controller = _tiny_controller()
    small = controller.score_actions(((2, 1), (1, 1)))
    large = controller.score_actions(
        ((2, 1, 0, 4), (1, 3, 1, 0), (0, 1, 2, 1))
    )
    assert small.actions
    assert large.actions
    assert controller.parameter_count == 8_889
