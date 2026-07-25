from __future__ import annotations

import inspect
import random

import pytest
import torch

from episode_functor_algebra_machine import (
    AlgebraMachineError,
    execute_program,
    verify_reduction_program,
)
import ssqac_successor_value_controller as successor


def _tiny_config(
    *,
    planner_iterations: int = 2,
) -> successor.SuccessorValueConfig:
    return successor.SuccessorValueConfig(
        field_width=8,
        width=16,
        cell_hidden=24,
        matrix_layers=1,
        planner_hidden=24,
        planner_iterations=planner_iterations,
        coordinate_harmonics=1,
        dropout=0.0,
    )


def _tiny_controller(
    *,
    planner_iterations: int = 2,
) -> successor.SuccessorValueController:
    torch.manual_seed(17)
    return successor.SuccessorValueController(
        _tiny_config(planner_iterations=planner_iterations)
    )


def _matrix() -> tuple[tuple[int, ...], ...]:
    return ((2, 1, 0), (1, 3, 1))


def _resource() -> successor.MutableResourceCounts:
    return successor.MutableResourceCounts()


def _unlock_preparation() -> None:
    successor._PREPARATION_LOCKED = False


def _map_action(
    action: successor.SuccessorAction,
    row_old_to_new: list[int],
    column_old_to_new: list[int],
) -> successor.SuccessorAction:
    if action.kind == successor.ACTION_HALT:
        return action
    row_a = row_old_to_new[action.row_a]
    row_b = (
        row_old_to_new[action.row_b]
        if action.kind
        in (successor.ACTION_ELIMINATE, successor.ACTION_SWAP)
        else 0
    )
    column = (
        column_old_to_new[action.column]
        if action.kind
        in (successor.ACTION_NORMALIZE, successor.ACTION_ELIMINATE)
        else 0
    )
    return successor.SuccessorAction(
        action.kind,
        row_a=row_a,
        row_b=row_b,
        column=column,
    )


def _inverse(order: list[int]) -> list[int]:
    result = [0] * len(order)
    for new, old in enumerate(order):
        result[old] = new
    return result


def _permute_matrix(
    matrix: tuple[tuple[int, ...], ...],
    row_order: list[int],
    column_order: list[int],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(matrix[old_row][old_column] for old_column in column_order)
        for old_row in row_order
    )


def _forward_inputs(
    controller: successor.SuccessorValueController,
    matrix: tuple[tuple[int, ...], ...],
    actions: tuple[successor.SuccessorAction, ...],
    visible_successors: tuple[tuple[tuple[int, ...], ...], ...],
    *,
    row_codes: list[int],
    column_codes: list[int],
    planner_iterations: int,
) -> torch.Tensor:
    device = next(controller.parameters()).device
    return controller(
        torch.tensor(matrix, dtype=torch.long, device=device),
        torch.tensor(visible_successors, dtype=torch.long, device=device),
        torch.tensor(
            [successor.ACTION_TO_INDEX[action.kind] for action in actions],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            [action.row_a for action in actions],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            [action.row_b for action in actions],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            [action.column for action in actions],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(row_codes, dtype=torch.float32, device=device),
        torch.tensor(column_codes, dtype=torch.float32, device=device),
        planner_iterations,
    )


def test_local_successors_compile_exactly_to_the_original_vm() -> None:
    matrix = _matrix()
    for action in successor.enumerate_legal_actions(matrix):
        expected = successor.apply_action(matrix, action)
        state = execute_program(
            matrix,
            successor.compile_action_to_primitives(matrix, action),
        )
        assert state.rows == expected


def test_existing_preparation_oracle_reaches_strict_canonical_verifier() -> None:
    _unlock_preparation()
    matrices = (
        ((0, 1, 2), (2, 0, 1)),
        ((0, 0, 3), (0, 2, 1), (1, 0, 4)),
        ((0, 0, 0), (4, 0, 2), (0, 5, 1)),
    )
    prepared = successor.build_preparation_states(
        matrices,
        maximum_steps=96,
    )
    assert prepared.oracle_calls >= len(matrices)
    assert len(prepared.oracle_source_sha256) == 64
    by_matrix = {state.rows: state.target_action for state in prepared.states}
    for source in matrices:
        matrix = successor.canonical_matrix(source)
        actions = []
        for _ in range(96):
            action = by_matrix[matrix]
            actions.append(action)
            if action.kind == successor.ACTION_HALT:
                break
            matrix = successor.apply_action(matrix, action)
        else:
            raise AssertionError("preparation trace did not halt")
        state = execute_program(
            source,
            successor.compile_trace_to_primitives(source, actions),
        )
        assert verify_reduction_program(source, state).passed


def test_preparation_oracle_lock_is_fail_closed() -> None:
    _unlock_preparation()
    successor.lock_preparation_oracle()
    with pytest.raises(successor.SuccessorValueError, match="locked"):
        successor.build_preparation_states((_matrix(),), maximum_steps=16)


def test_halt_is_unconditional_and_endpoint_is_not_rendered() -> None:
    terminal = ((1, 0), (0, 1))
    nonterminal = ((2, 1), (0, 1))
    for matrix in (terminal, nonterminal):
        actions = successor.enumerate_legal_actions(matrix)
        assert actions[-1] == successor.SuccessorAction(successor.ACTION_HALT)


def test_raw_zero_and_shuffled_counterfactual_controls_are_exact() -> None:
    matrix = _matrix()
    raw_resource = _resource()
    raw = successor.render_counterfactuals(
        matrix,
        mode=successor.MODE_RAW,
        binding_seed=91,
        resources=raw_resource,
    )
    assert raw.true_successors == raw.visible_successors
    assert raw_resource.successor_evaluations == len(raw.actions)
    assert raw_resource.successor_matrix_cells == (
        len(raw.actions) * len(matrix) * len(matrix[0])
    )

    zero_resource = _resource()
    zero = successor.render_counterfactuals(
        matrix,
        mode=successor.MODE_ZERO,
        binding_seed=91,
        resources=zero_resource,
    )
    assert zero.true_successors == raw.true_successors
    assert all(
        value == 0
        for candidate in zero.visible_successors
        for row in candidate
        for value in row
    )
    assert zero_resource.freeze() == raw_resource.freeze()

    shuffled_resource = _resource()
    shuffled = successor.render_counterfactuals(
        matrix,
        mode=successor.MODE_SHUFFLED,
        binding_seed=91,
        resources=shuffled_resource,
    )
    assert set(shuffled.visible_successors) == set(raw.true_successors)
    assert all(
        visible != true
        for visible, true in zip(
            shuffled.visible_successors,
            shuffled.true_successors,
            strict=True,
        )
    )
    assert shuffled_resource.freeze() == raw_resource.freeze()


def test_shuffled_binding_is_deterministic_and_action_order_independent() -> None:
    matrix = _matrix()
    actions = successor.enumerate_legal_actions(matrix)
    left = successor.render_counterfactuals(
        matrix,
        mode=successor.MODE_SHUFFLED,
        binding_seed=712,
        resources=_resource(),
        actions=actions,
    )
    right = successor.render_counterfactuals(
        matrix,
        mode=successor.MODE_SHUFFLED,
        binding_seed=712,
        resources=_resource(),
        actions=tuple(reversed(actions)),
    )
    assert left.binding_manifest_sha256 == right.binding_manifest_sha256
    left_by_action = dict(zip(left.actions, left.visible_successors, strict=True))
    right_by_action = dict(zip(right.actions, right.visible_successors, strict=True))
    assert left_by_action == right_by_action


def test_candidate_forward_has_only_raw_and_descriptor_inputs() -> None:
    controller = _tiny_controller()
    parameters = set(inspect.signature(controller.forward).parameters)
    assert parameters == {
        "current_values",
        "successor_values",
        "action_kind",
        "row_a",
        "row_b",
        "column",
        "row_codes",
        "column_codes",
        "planner_iterations",
    }
    forbidden = {
        "energy",
        "rank",
        "frontier",
        "reference",
        "schedule",
        "search",
        "verifier",
        "oracle",
    }
    names = tuple(name.lower() for name, _ in controller.named_parameters())
    assert not any(token in name for token in forbidden for name in names)
    assert isinstance(controller.shared_planner_cell, torch.nn.GRUCell)
    assert hasattr(controller, "raw_recall_projection")


def test_fixed_shared_recurrence_and_raw_recall_run_exact_depth() -> None:
    controller = _tiny_controller(planner_iterations=3)
    planner_calls = 0
    recall_calls = 0

    def count_planner(*_args: object) -> None:
        nonlocal planner_calls
        planner_calls += 1

    def count_recall(*_args: object) -> None:
        nonlocal recall_calls
        recall_calls += 1

    planner_hook = controller.shared_planner_cell.register_forward_hook(count_planner)
    recall_hook = controller.raw_recall_projection.register_forward_hook(count_recall)
    resources = _resource()
    scored = controller.score_actions(
        _matrix(),
        mode=successor.MODE_RAW,
        binding_seed=1,
        planner_iterations=5,
        resources=resources,
    )
    planner_hook.remove()
    recall_hook.remove()
    assert planner_calls == 5
    assert recall_calls == 5
    assert resources.planner_iterations == 5
    assert resources.recurrent_action_updates == 5 * len(scored.actions)
    assert scored.planner_iterations == 5


def test_progressive_depth_schedule_is_randomized_progressive_and_equal_budget() -> None:
    rng = random.Random(81)
    early = successor.progressive_paired_depths(
        update_index=0,
        total_updates=100,
        batch_size=8,
        fixed_depth=8,
        rng=rng,
    )
    late = successor.progressive_paired_depths(
        update_index=99,
        total_updates=100,
        batch_size=8,
        fixed_depth=8,
        rng=rng,
    )
    assert early == (8,) * 8
    assert sum(early) == sum(late) == 64
    assert min(late) >= 0
    assert max(late) <= 16
    assert len(set(late)) > 1


def test_parameter_count_is_geometry_general_equal_and_under_budget() -> None:
    models = [_tiny_controller() for _ in successor.ARM_SPECS]
    counts = {model.parameter_count for model in models}
    states = [
        {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
        for model in models
    ]
    assert len(counts) == 1
    assert all(state == states[0] for state in states)
    assert models[0].complete_system_parameter_count < successor.TOTAL_PARAMETER_BUDGET
    assert not hasattr(models[0], "row_embedding")
    assert not hasattr(models[0], "column_embedding")


def test_action_order_permutation_preserves_action_logits() -> None:
    controller = _tiny_controller().eval()
    matrix = _matrix()
    actions = successor.enumerate_legal_actions(matrix)
    with torch.no_grad():
        left = controller.score_actions(
            matrix,
            mode=successor.MODE_RAW,
            binding_seed=5,
            planner_iterations=2,
            resources=_resource(),
            actions=actions,
        )
        right = controller.score_actions(
            matrix,
            mode=successor.MODE_RAW,
            binding_seed=5,
            planner_iterations=2,
            resources=_resource(),
            actions=tuple(reversed(actions)),
        )
    left_by_action = {
        action: left.logits[index] for index, action in enumerate(left.actions)
    }
    right_by_action = {
        action: right.logits[index] for index, action in enumerate(right.actions)
    }
    for action in actions:
        torch.testing.assert_close(
            left_by_action[action],
            right_by_action[action],
            atol=1e-6,
            rtol=1e-6,
        )


def test_storage_row_column_permutation_with_coordinate_recoding_is_invariant() -> None:
    controller = _tiny_controller().eval()
    matrix = _matrix()
    actions = successor.enumerate_legal_actions(matrix)
    rendered = successor.render_counterfactuals(
        matrix,
        mode=successor.MODE_RAW,
        binding_seed=7,
        resources=_resource(),
        actions=actions,
    )
    row_order = [1, 0]
    column_order = [2, 0, 1]
    row_old_to_new = _inverse(row_order)
    column_old_to_new = _inverse(column_order)
    permuted_matrix = _permute_matrix(matrix, row_order, column_order)
    permuted_actions = tuple(
        _map_action(action, row_old_to_new, column_old_to_new)
        for action in actions
    )
    permuted_successors = tuple(
        _permute_matrix(candidate, row_order, column_order)
        for candidate in rendered.visible_successors
    )
    with torch.no_grad():
        original = _forward_inputs(
            controller,
            matrix,
            actions,
            rendered.visible_successors,
            row_codes=[0, 1],
            column_codes=[0, 1, 2],
            planner_iterations=2,
        )
        recoded = _forward_inputs(
            controller,
            permuted_matrix,
            permuted_actions,
            permuted_successors,
            row_codes=row_order,
            column_codes=column_order,
            planner_iterations=2,
        )
    torch.testing.assert_close(original, recoded, atol=1e-6, rtol=1e-6)


def test_finite_field_representative_recoding_is_exactly_invariant() -> None:
    controller = _tiny_controller().eval()
    matrix = _matrix()
    recoded = tuple(
        tuple(value + (row + column + 1) * successor.FIELD_MODULUS for column, value in enumerate(values))
        for row, values in enumerate(matrix)
    )
    with torch.no_grad():
        left = controller.score_actions(
            matrix,
            mode=successor.MODE_RAW,
            binding_seed=3,
            planner_iterations=2,
            resources=_resource(),
        )
        right = controller.score_actions(
            recoded,
            mode=successor.MODE_RAW,
            binding_seed=3,
            planner_iterations=2,
            resources=_resource(),
        )
    assert left.actions == right.actions
    torch.testing.assert_close(left.logits, right.logits, atol=0.0, rtol=0.0)


def test_candidate_rollout_does_not_call_verifier_or_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _tiny_controller().eval()
    successor.lock_preparation_oracle()

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("forbidden candidate-time call")

    monkeypatch.setattr(successor, "verify_reduction_program", forbidden)
    monkeypatch.setattr(successor, "build_preparation_states", forbidden)
    rollout = successor.candidate_successor_only_rollout(
        controller,
        _matrix(),
        input_mode=successor.MODE_RAW,
        planner_iterations=2,
        binding_seed=4,
        maximum_steps=2,
    )
    assert rollout.resources.oracle_calls == 0
    assert rollout.resources.search_calls == 0
    assert rollout.resources.verifier_calls == 0


def test_posthoc_assessor_is_the_only_strict_verifier_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = ((1, 0), (0, 1))
    rollout = successor.CandidateRollout(
        halted=True,
        invalid=False,
        overlong=False,
        actions=(successor.SuccessorAction(successor.ACTION_HALT),),
        output_rows=matrix,
        resources=_resource().freeze(),
    )
    calls = 0
    original = successor.verify_reduction_program

    def counted(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(successor, "verify_reduction_program", counted)
    assessed = successor.assess_rollout_posthoc(matrix, rollout)
    assert assessed.strict_canonical_certified
    assert assessed.posthoc_verifier_calls == 1
    assert calls == 1


def test_tiny_experiment_has_matched_budgets_and_two_depth_evaluations(
    tmp_path,
) -> None:
    _unlock_preparation()
    report = successor.run_bounded_experiment(
        successor.SuccessorExperimentConfig(
            seed=117,
            train_matrices=3,
            evaluation_matrices=2,
            train_maximum_rows=2,
            train_maximum_columns=3,
            evaluation_minimum_rows=3,
            evaluation_minimum_columns=4,
            evaluation_maximum_rows=3,
            evaluation_maximum_columns=4,
            maximum_preparation_steps=32,
            maximum_rollout_steps=3,
            optimizer_updates=1,
            batch_size=2,
            learning_rate=1e-3,
            amp_bfloat16=False,
            material_minimum_cases=2,
            device="cpu",
            controller=_tiny_config(planner_iterations=2),
        ),
        model_dir=tmp_path / "models",
    )
    assert tuple(arm.name for arm in report.arms) == successor.ARMS
    assert report.controls_equal_parameters
    assert report.controls_equal_optimizer_updates
    assert report.controls_equal_training_successor_evaluations
    assert report.fixed_and_progressive_equal_training_planner_iterations
    assert report.parameter_budget_passed
    assert report.preparation_oracle_locked_before_training_and_eval
    assert report.explicit_raw_input_recall_path
    for arm in report.arms:
        assert arm.optimizer_updates == 1
        assert arm.evaluation_total == 2
        assert arm.longer_evaluation_total == 2
        assert arm.model_file_sha256 is not None
        assert arm.evaluation_resources.oracle_calls == 0
        assert arm.evaluation_resources.search_calls == 0
        assert arm.evaluation_resources.verifier_calls == 0


def test_incomplete_or_noncanonical_trace_fails_original_assessor() -> None:
    matrix = _matrix()
    state = execute_program(
        matrix,
        (successor.AlgebraInstruction(successor.OP_HALT),),
    )
    with pytest.raises(AlgebraMachineError):
        verify_reduction_program(matrix, state)
