from __future__ import annotations

import inspect
from itertools import product

import pytest
import torch

from episode_functor_algebra_machine import execute_program
import ssqac_canonical_energy_controller as canonical


def _tiny_controller() -> canonical.CanonicalEnergyController:
    torch.manual_seed(17)
    return canonical.CanonicalEnergyController(
        canonical.CanonicalControllerConfig(
            width=12,
            message_layers=1,
            residual_hidden=20,
            field_harmonics=1,
            coordinate_harmonics=1,
            residual_bound=0.49,
        )
    )


def test_energy_zero_iff_strict_canonical_structure_on_named_cases() -> None:
    accepted = (
        ((1, 0, 3), (0, 1, 4), (0, 0, 0)),
        ((0, 1, 0), (0, 0, 1), (0, 0, 0)),
        ((1, 2, 0, 4), (0, 0, 1, 8)),
        ((0, 0), (0, 0)),
    )
    rejected = (
        ((0, 0, 0), (1, 0, 3), (0, 1, 4)),
        ((0, 1, 4), (1, 0, 3), (0, 0, 0)),
        ((2, 0, 3), (0, 1, 4), (0, 0, 0)),
        ((1, 1, 3), (0, 1, 4), (0, 0, 0)),
        ((1, 0, 3), (1, 1, 4), (0, 0, 0)),
    )
    for matrix in accepted:
        witness = canonical.canonical_energy_witness(matrix)
        assert witness.energy == 0
        assert canonical.is_canonical_rref_structure(matrix)
    for matrix in rejected:
        assert canonical.canonical_defect_energy(matrix) > 0
        assert not canonical.is_canonical_rref_structure(matrix)


def test_energy_zero_equivalence_is_exhaustive_for_small_matrices() -> None:
    for flat in product((0, 1, 2), repeat=4):
        matrix = (flat[:2], flat[2:])
        assert (canonical.canonical_defect_energy(matrix) == 0) == (
            canonical.is_canonical_rref_structure(matrix)
        )


def test_witness_is_ordered_and_columns_are_semantically_fixed() -> None:
    canonical_matrix = ((1, 0, 9), (0, 1, 7), (0, 0, 0))
    row_reversed = ((0, 1, 7), (1, 0, 9), (0, 0, 0))
    column_reversed = tuple(tuple(reversed(row)) for row in canonical_matrix)
    assert canonical.canonical_defect_energy(canonical_matrix) == 0
    assert canonical.canonical_defect_energy(row_reversed) > 0
    assert canonical.canonical_defect_energy(column_reversed) > 0
    witness = canonical.canonical_energy_witness(row_reversed)
    assert witness.rank == 2
    assert witness.assigned_rows != (0, 1)


def test_reference_schedule_compiles_to_unchanged_strict_verifier() -> None:
    matrices = (
        ((2, 1, 0), (1, 1, 1)),
        ((0, 0, 0), (1, 0, 3), (0, 1, 4)),
        ((1, 0, 160), (1, 0, 88), (0, 1, 138)),
        ((3, 2, 1, 0), (4, 0, 1, 2), (0, 5, 0, 1)),
    )
    for matrix in matrices:
        schedule = canonical.canonical_reference_schedule(matrix)
        actions = (*schedule, canonical.CanonicalAction(canonical.ACTION_HALT))
        assert canonical.strictly_verify_action_trace(matrix, actions)
        program = canonical.compile_action_trace_to_vm(matrix, actions)
        state = execute_program(matrix, program)
        assert canonical.canonical_defect_energy(state.rows) == 0
        assert canonical.is_canonical_rref_structure(state.rows)


def test_reference_frontier_actions_never_damage_explicit_energy() -> None:
    matrices = canonical.generate_matrices(
        seed=20260724,
        count=128,
        minimum_rows=2,
        maximum_rows=4,
        minimum_columns=2,
        maximum_columns=6,
    )
    for matrix in matrices:
        current = matrix
        for action in canonical.canonical_reference_schedule(matrix):
            before = canonical.canonical_defect_energy(current)
            current = canonical.apply_action(current, action)
            after = canonical.canonical_defect_energy(current)
            assert after <= before


def test_swap_normalize_and_eliminate_compile_exactly() -> None:
    matrix = ((0, 2, 1), (1, 3, 0))
    swap = canonical.CanonicalAction(
        canonical.ACTION_SWAP,
        row_a=0,
        row_b=1,
    )
    swapped = canonical.apply_action(matrix, swap)
    assert (
        execute_program(
            matrix,
            canonical.compile_action_to_vm(matrix, swap),
        ).rows
        == swapped
    )

    normalize = canonical.CanonicalAction(
        canonical.ACTION_NORMALIZE,
        row_a=0,
        column=1,
    )
    normalized = canonical.apply_action(matrix, normalize)
    assert (
        execute_program(
            matrix,
            canonical.compile_action_to_vm(matrix, normalize),
        ).rows
        == normalized
    )

    eliminate_matrix = ((1, 2, 0), (3, 1, 4))
    eliminate = canonical.CanonicalAction(
        canonical.ACTION_ELIMINATE,
        row_a=1,
        row_b=0,
        column=0,
    )
    eliminated = canonical.apply_action(eliminate_matrix, eliminate)
    assert (
        execute_program(
            eliminate_matrix,
            canonical.compile_action_to_vm(eliminate_matrix, eliminate),
        ).rows
        == eliminated
    )


def test_halt_and_illegal_actions_fail_closed() -> None:
    with pytest.raises(canonical.CanonicalEnergyError, match="zero energy"):
        canonical.apply_action(
            ((2, 0), (0, 1)),
            canonical.CanonicalAction(canonical.ACTION_HALT),
        )
    with pytest.raises(canonical.CanonicalEnergyError, match="row_a < row_b"):
        canonical.apply_action(
            ((1, 0), (0, 1)),
            canonical.CanonicalAction(
                canonical.ACTION_SWAP,
                row_a=1,
                row_b=0,
            ),
        )
    with pytest.raises(canonical.CanonicalEnergyError, match="nonunit"):
        canonical.apply_action(
            ((1, 0), (0, 1)),
            canonical.CanonicalAction(
                canonical.ACTION_NORMALIZE,
                row_a=0,
                column=0,
            ),
        )


def test_candidate_surface_has_no_forbidden_inputs_or_state() -> None:
    controller = _tiny_controller()
    forward = set(inspect.signature(controller.forward).parameters)
    forbidden = {
        "source",
        "query",
        "workspace",
        "target",
        "oracle",
        "hidden",
        "previous",
        "step",
        "recurrent",
    }
    assert not forward.intersection(forbidden)
    parameter_names = tuple(name.lower() for name, _ in controller.named_parameters())
    assert not any(
        token in name
        for name in parameter_names
        for token in (
            "source",
            "query",
            "workspace",
            "oracle",
            "recurrent",
            "step",
            "position_embedding",
        )
    )


def test_parameter_count_is_geometry_independent() -> None:
    controller = _tiny_controller()
    direct = sum(parameter.numel() for parameter in controller.parameters())
    assert direct == controller.parameter_count
    assert direct == controller.parameter_count_breakdown()["total"]
    controller.score_actions(((2, 1), (1, 3)))
    controller.score_actions(
        (
            (2, 1, 0, 4, 0, 3),
            (1, 3, 1, 0, 2, 0),
            (0, 1, 2, 1, 0, 5),
            (4, 0, 0, 1, 2, 1),
        )
    )
    assert controller.parameter_count == direct


def test_bounded_residual_cannot_reverse_integer_advantage() -> None:
    controller = _tiny_controller()
    scores = controller.score_actions(((2, 1, 0), (1, 3, 1)))
    assert torch.all(scores.learned_residual.abs() < 0.5)
    for left in range(len(scores.actions)):
        for right in range(len(scores.actions)):
            explicit_gap = (
                scores.explicit_reduction[left] - scores.explicit_reduction[right]
            )
            if explicit_gap >= 1:
                assert scores.total_score[left] > scores.total_score[right]


def test_frontier_features_are_matrix_only_and_geometry_relative() -> None:
    controller = _tiny_controller()
    matrix = (
        (1, 0, 202, 0, 0, 211),
        (0, 1, 161, 0, 0, 216),
        (0, 0, 120, 1, 0, 248),
        (0, 0, 113, 0, 1, 28),
    )
    witness = canonical.canonical_energy_witness(matrix)
    assert witness.settled_prefix == 2
    reference = canonical.canonical_reference_schedule(matrix)[0]
    features = controller._frontier_features(
        matrix,
        reference,
        witness,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert features.shape == (18,)
    assert features[-1].item() == 1.0
    unrelated = canonical.CanonicalAction(
        canonical.ACTION_NORMALIZE,
        row_a=0,
        column=2,
    )
    unrelated_features = controller._frontier_features(
        matrix,
        unrelated,
        witness,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert unrelated_features[-1].item() == 0.0


def test_frontier_feature_ablations_are_exact_and_fail_closed() -> None:
    controller = _tiny_controller()
    matrix = (
        (1, 0, 202, 0, 0, 211),
        (0, 1, 161, 0, 0, 216),
        (0, 0, 120, 1, 0, 248),
        (0, 0, 113, 0, 1, 28),
    )
    witness = canonical.canonical_energy_witness(matrix)
    action = canonical.canonical_reference_schedule(matrix)[0]
    full = controller._frontier_features(
        matrix,
        action,
        witness,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    zeroed = controller._ablate_frontier_features(
        full,
        frontier_ablation=canonical.FRONTIER_ZERO_ALL,
    )
    masked = controller._ablate_frontier_features(
        full,
        frontier_ablation=canonical.FRONTIER_MASK_ACTION_CORRECTNESS,
    )
    unchanged = controller._ablate_frontier_features(
        full,
        frontier_ablation=canonical.FRONTIER_FULL,
    )
    assert torch.equal(unchanged, full)
    assert torch.count_nonzero(zeroed).item() == 0
    assert torch.equal(
        masked[: canonical.FRONTIER_ACTION_CORRECTNESS_START],
        full[: canonical.FRONTIER_ACTION_CORRECTNESS_START],
    )
    assert (
        torch.count_nonzero(
            masked[canonical.FRONTIER_ACTION_CORRECTNESS_START :]
        ).item()
        == 0
    )
    with pytest.raises(canonical.CanonicalEnergyError, match="frontier ablation"):
        controller._ablate_frontier_features(
            full,
            frontier_ablation="unknown",
        )


def test_matched_inference_ablations_reuse_weights_and_successors() -> None:
    controller = _tiny_controller()
    matrix = (
        (1, 0, 202, 0, 0, 211),
        (0, 1, 161, 0, 0, 216),
        (0, 0, 120, 1, 0, 248),
        (0, 0, 113, 0, 1, 28),
    )
    before = canonical.model_sha256(controller)
    full = controller.score_actions(
        matrix,
        frontier_ablation=canonical.FRONTIER_FULL,
    )
    zeroed = controller.score_actions(
        matrix,
        frontier_ablation=canonical.FRONTIER_ZERO_ALL,
    )
    masked = controller.score_actions(
        matrix,
        frontier_ablation=canonical.FRONTIER_MASK_ACTION_CORRECTNESS,
    )
    assert canonical.model_sha256(controller) == before
    assert full.actions == zeroed.actions == masked.actions
    assert full.energy_before == zeroed.energy_before == masked.energy_before
    assert full.energy_after == zeroed.energy_after == masked.energy_after
    assert torch.equal(full.explicit_reduction, zeroed.explicit_reduction)
    assert torch.equal(full.explicit_reduction, masked.explicit_reduction)


def test_expert_and_random_controls_share_integer_plateau() -> None:
    counter = canonical.OracleCounter()
    states = canonical.build_expert_states(
        (
            ((2, 1, 0), (1, 3, 1)),
            ((1, 0, 160), (1, 0, 88), (0, 1, 138)),
        ),
        maximum_steps=32,
        counter=counter,
    )
    randomized = canonical.make_random_label_control(states, seed=41)
    assert states
    assert counter.calls > 0
    assert len(states) == len(randomized)
    for expert, control in zip(states, randomized, strict=True):
        transitions = canonical.evaluate_transitions(expert.rows)
        assert (
            transitions[expert.target_indices[0]].explicit_reduction
            == transitions[control.target_indices[0]].explicit_reduction
        )


def test_expert_selects_canonical_escape_on_frontier_plateau() -> None:
    matrix = (
        (1, 0, 202, 0, 0, 211),
        (0, 1, 161, 0, 0, 216),
        (0, 0, 120, 1, 0, 248),
        (0, 0, 113, 0, 1, 28),
    )
    transitions = canonical.evaluate_transitions(matrix)
    maximum = max(item.explicit_reduction for item in transitions)
    assert maximum == 0
    counter = canonical.OracleCounter()
    target = canonical.expert_action_indices(matrix, counter=counter)[0]
    assert transitions[target].explicit_reduction == maximum
    assert transitions[target].action == canonical.CanonicalAction(
        canonical.ACTION_NORMALIZE,
        row_a=2,
        column=2,
    )
    assert counter.calls == 1


def test_final_rollout_has_no_oracle_call_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _tiny_controller()

    def forbidden(*_args: object, **_kwargs: object) -> tuple[int, ...]:
        raise AssertionError("preparation oracle leaked into rollout")

    monkeypatch.setattr(canonical, "expert_action_indices", forbidden)
    assert (
        "expert_action_indices"
        not in canonical.final_oracle_free_rollout.__code__.co_names
    )
    terminal = ((1, 0, 3), (0, 1, 4), (0, 0, 0))
    result = canonical.final_oracle_free_rollout(
        controller,
        terminal,
        maximum_steps=2,
    )
    assert result.strict_canonical_certified
    assert result.oracle_calls == 0


def test_training_is_bounded_and_changes_only_model_weights() -> None:
    controller = _tiny_controller()
    counter = canonical.OracleCounter()
    states = canonical.build_expert_states(
        (((2, 1, 0), (1, 3, 1)),),
        maximum_steps=16,
        counter=counter,
    )
    before = canonical.model_sha256(controller)
    updates = canonical.train_controller(
        controller,
        states,
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        shuffle_seed=7,
        maximum_updates=2,
    )
    assert updates == 2
    assert canonical.model_sha256(controller) != before
    assert 0.0 <= canonical.label_accuracy(controller, states) <= 1.0


def test_geometry_contract_is_exact_and_fails_closed() -> None:
    with pytest.raises(canonical.CanonicalEnergyError, match="at most 3x4"):
        canonical.CanonicalExperimentConfig(train_maximum_rows=4)
    with pytest.raises(canonical.CanonicalEnergyError, match="exactly 4x5-6"):
        canonical.CanonicalExperimentConfig(evaluation_minimum_columns=4)
    with pytest.raises(canonical.CanonicalEnergyError, match="three distinct"):
        canonical.run_multiseed_experiment(
            canonical.CanonicalExperimentConfig(
                train_matrices=1,
                evaluation_matrices=1,
            ),
            seeds=(1, 2),
        )


def test_fixed_schedule_baseline_uses_unchanged_strict_verifier() -> None:
    evaluation = canonical.generate_matrices(
        seed=20260731,
        count=12,
        minimum_rows=4,
        maximum_rows=4,
        minimum_columns=5,
        maximum_columns=6,
    )
    summary = canonical.evaluate_fixed_schedule_strict(
        evaluation,
        maximum_steps=32,
    )
    assert summary.strict_canonical_certified == len(evaluation)
    assert summary.invalid == 0
    assert summary.overlong == 0
    assert summary.oracle_calls == 0


def test_resource_counts_separate_policy_search_from_fixed_schedule() -> None:
    controller = _tiny_controller()
    evaluation = canonical.generate_matrices(
        seed=20260801,
        count=2,
        minimum_rows=4,
        maximum_rows=4,
        minimum_columns=5,
        maximum_columns=6,
    )
    learned, learned_resources = canonical._measure_matrix_policy_arm(
        controller,
        evaluation,
        maximum_steps=32,
        learned_residual=True,
    )
    fixed, fixed_resources = canonical._measure_fixed_schedule_arm(
        evaluation,
        maximum_steps=32,
    )
    assert learned.total == len(evaluation)
    assert learned_resources.field_rank_calls > 0
    assert learned_resources.matching_cache_misses > 0
    assert learned_resources.matching_row_permutations > 0
    assert learned_resources.action_successor_batches > 0
    assert learned_resources.action_successor_evaluations > 0
    assert fixed.strict_canonical_certified == len(evaluation)
    assert fixed_resources.field_rank_calls > 0
    assert fixed_resources.matching_cache_misses == 0
    assert fixed_resources.matching_candidate_assignments == 0
    assert fixed_resources.reference_schedule_calls == len(evaluation)
    assert fixed_resources.reference_schedule_actions > 0
    assert fixed_resources.strict_verifier_calls == len(evaluation)
    assert fixed_resources.action_successor_batches == 0
    assert fixed_resources.action_successor_evaluations == 0


def test_three_seed_report_has_all_audited_arms_and_zero_final_oracle() -> None:
    config = canonical.CanonicalExperimentConfig(
        train_matrices=2,
        evaluation_matrices=1,
        maximum_expert_steps=16,
        maximum_rollout_steps=32,
        epochs=1,
        batch_size=16,
        learning_rate=1e-3,
        maximum_updates=1,
        controller=canonical.CanonicalControllerConfig(
            width=8,
            message_layers=1,
            residual_hidden=12,
            field_harmonics=1,
            coordinate_harmonics=1,
        ),
    )
    report = canonical.run_multiseed_experiment(
        config,
        seeds=(101, 102, 103),
    )
    repeated = canonical.run_multiseed_experiment(
        config,
        seeds=(101, 102, 103),
    )
    assert repeated.canonical_bytes() == report.canonical_bytes()
    assert report.seeds == (101, 102, 103)
    assert len(report.seed_reports) == 3
    assert report.evaluation_cases_per_arm == 3
    assert report.final_rollout_oracle_calls == 0
    assert report.fixed_schedule_strict_canonical_certified == 3
    assert report.fixed_schedule_reaches_ceiling
    assert report.learned_claim_downgraded
    assert report.status == canonical.HYBRID_INTERPRETATION
    assert report.ablation_collapse_rule == canonical.ABLATION_COLLAPSE_RULE
    assert report.learned_claim_downgrade_rule == canonical.LEARNED_CLAIM_DOWNGRADE_RULE
    assert report.fixed_schedule_ceiling_seed_count == 3
    assert report.fixed_schedule_resources.reference_schedule_calls == 3
    assert report.fixed_schedule_resources.strict_verifier_calls == 3
    assert report.expert_full_resources.action_successor_evaluations > 0
    for seed_report in report.seed_reports:
        assert seed_report.train_maximum_rows <= 3
        assert seed_report.train_maximum_columns <= 4
        assert seed_report.evaluation_rows == 4
        assert seed_report.evaluation_minimum_columns == 5
        assert seed_report.evaluation_maximum_columns == 6
        assert seed_report.final_rollout_oracle_calls == 0
        assert seed_report.fixed_schedule_reaches_ceiling
        assert seed_report.learned_claim_downgraded
        assert seed_report.status == canonical.HYBRID_INTERPRETATION
        assert seed_report.fixed_schedule_resources.reference_schedule_calls == 1
        assert seed_report.fixed_schedule_resources.strict_verifier_calls == 1
        for certified, invalid, overlong in (
            (
                seed_report.energy_only_strict_canonical_certified,
                seed_report.energy_only_invalid,
                seed_report.energy_only_overlong,
            ),
            (
                seed_report.expert_strict_canonical_certified,
                seed_report.expert_invalid,
                seed_report.expert_overlong,
            ),
            (
                seed_report.expert_zero_frontier_strict_canonical_certified,
                seed_report.expert_zero_frontier_invalid,
                seed_report.expert_zero_frontier_overlong,
            ),
            (
                seed_report.expert_masked_action_bits_strict_canonical_certified,
                seed_report.expert_masked_action_bits_invalid,
                seed_report.expert_masked_action_bits_overlong,
            ),
            (
                seed_report.random_strict_canonical_certified,
                seed_report.random_invalid,
                seed_report.random_overlong,
            ),
            (
                seed_report.fixed_schedule_strict_canonical_certified,
                seed_report.fixed_schedule_invalid,
                seed_report.fixed_schedule_overlong,
            ),
        ):
            assert certified + invalid + overlong == 1
    payload = report.canonical_bytes()
    assert b'"learned_claim_downgraded":true' in payload
    assert b'"action_successor_evaluations"' in payload
    assert b'"matching_candidate_assignments"' in payload


def test_generation_and_manifests_are_deterministic_and_disjoint() -> None:
    train = canonical.generate_matrices(
        seed=5,
        count=4,
        minimum_rows=2,
        maximum_rows=3,
        minimum_columns=2,
        maximum_columns=4,
    )
    evaluation = canonical.generate_matrices(
        seed=6,
        count=4,
        minimum_rows=4,
        maximum_rows=4,
        minimum_columns=5,
        maximum_columns=6,
        excluded=set(train),
    )
    assert set(train).isdisjoint(evaluation)
    assert canonical.matrix_manifest(train) == canonical.matrix_manifest(train)
    assert canonical.matrix_manifest(train) != canonical.matrix_manifest(evaluation)
