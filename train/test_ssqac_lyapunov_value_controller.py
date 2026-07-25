from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

import ssqac_lyapunov_value_controller as lane


SIMPLE = (
    (2, 0, 1),
    (0, 3, 1),
)
IDENTITY = (
    (1, 0, 0),
    (0, 1, 0),
)


def _boundary() -> lane.EvaluationBoundaryToken:
    return lane.EvaluationBoundaryToken(
        schema=lane.BOUNDARY_SCHEMA,
        dataset_manifest_sha256="a" * 64,
        preparation_source_sha256="b" * 64,
        preparation_source_destroyed=True,
        prepared_packet_sha256="c" * 64,
        prepared_packet_destroyed=True,
        training_labels_destroyed=True,
        training_tensors_destroyed=True,
        candidate_allowed_inputs=(
            "raw_current_field_matrix",
            "legal_action_descriptors",
            "raw_one_step_successor_matrices",
            "fixed_model_parameters",
        ),
        forbidden_candidate_inputs=("remaining_distance_label",),
    )


def _tiny_model() -> lane.LyapunovValueController:
    return lane.LyapunovValueController(
        lane.LyapunovConfig(
            field_width=8,
            width=16,
            cell_hidden=24,
            matrix_layers=1,
            state_hidden=24,
            coordinate_harmonics=1,
        )
    )


def test_treatment_is_not_action_imitation() -> None:
    treatment = next(
        arm for arm in lane.ARM_SPECS if arm.name == lane.ARM_TREATMENT
    )
    classifier = next(
        arm for arm in lane.ARM_SPECS if arm.name == lane.ARM_CLASSIFICATION
    )
    assert treatment.classification_weight == 0.0
    assert treatment.distance_weight > 0.0
    assert treatment.monotonic_weight > 0.0
    assert treatment.bellman_weight > 0.0
    assert classifier.classification_weight > 0.0
    assert classifier.distance_weight == 0.0
    assert classifier.monotonic_weight == 0.0
    assert classifier.bellman_weight == 0.0


def test_all_required_controls_are_present() -> None:
    assert set(lane.REPORTED_ARMS) == {
        lane.ARM_TREATMENT,
        lane.ARM_CLASSIFICATION,
        lane.ARM_SHUFFLED_DISTANCE,
        lane.ARM_ZERO_STRUCTURE,
        lane.ARM_RANDOM_LABELS,
        lane.ARM_SHUFFLED_BINDINGS,
    }


def test_distance_label_is_source_independent_and_reaches_terminal() -> None:
    oracle = lane.CanonicalDistanceOracle(32)
    first = oracle.label(SIMPLE)
    second = oracle.label(tuple(tuple(value + 257 for value in row) for row in SIMPLE))
    assert first == second
    assert first.remaining_distance > 0
    current = lane.canonical_matrix(SIMPLE)
    steps = 0
    while True:
        label = oracle.label(current)
        assert label.remaining_distance == first.remaining_distance - steps
        if label.terminal:
            break
        current = lane.apply_action(current, label.canonical_next_action)
        steps += 1
    assert steps == first.remaining_distance
    oracle.lock()
    with pytest.raises(lane.LyapunovValueError, match="locked"):
        oracle.label(SIMPLE)


def test_preparation_source_is_destroyed(tmp_path: Path) -> None:
    root = tmp_path / "transcript"
    dataset, receipt = lane.build_potential_dataset(
        (SIMPLE, IDENTITY),
        maximum_steps=32,
        preparation_root=root,
    )
    assert dataset.labels
    assert dataset.decisions
    assert receipt.serialized_source_destroyed
    assert not receipt.serialized_source_exists_after_destroy
    assert not root.exists()
    assert receipt.source_independence_conflicts == 0
    assert receipt.strict_preparation_verifier_calls == 2


def test_packet_round_trip_and_hash_binding(tmp_path: Path) -> None:
    packet_path = tmp_path / "packet.json"
    experiment = lane.ExperimentConfig(
        seed=7,
        train_matrices=2,
        evaluation_matrices=1,
        maximum_preparation_steps=32,
        maximum_rollout_steps=32,
    )
    receipt = lane.write_prepared_packet(
        experiment=experiment,
        output=packet_path,
    )
    packet = lane.load_prepared_packet(packet_path)
    assert packet.experiment_config == experiment
    assert packet.file_sha256 == receipt["file_sha256"]
    assert packet.dataset.manifest_sha256 == receipt["dataset_manifest_sha256"]
    envelope = json.loads(packet_path.read_text())
    envelope["payload"]["dataset"]["maximum_distance"] += 1
    packet_path.write_text(json.dumps(envelope))
    with pytest.raises(lane.LyapunovValueError, match="hash mismatch"):
        lane.load_prepared_packet(packet_path)


def test_evaluation_boundary_fails_open_packet() -> None:
    with pytest.raises(lane.LyapunovValueError, match="not sealed"):
        replace(_boundary(), prepared_packet_destroyed=False)


def test_model_is_under_complete_system_cap() -> None:
    model = lane.LyapunovValueController()
    assert model.parameter_count > 1_000_000
    assert model.complete_system_parameter_count < 200_000_000
    assert model.parameter_count_breakdown()["total"] == model.parameter_count


def test_geometry_general_encoder_accepts_mixed_sizes() -> None:
    model = _tiny_model()
    matrices = (
        SIMPLE,
        (
            (1, 2, 3, 4, 5),
            (0, 1, 0, 1, 0),
            (7, 0, 8, 0, 9),
            (0, 0, 0, 1, 1),
        ),
    )
    values, mask, rows, columns = lane.tensorize_matrices(
        matrices,
        device=torch.device("cpu"),
    )
    scores = model.encode_states(values, mask, rows, columns)
    assert scores.embedding.shape == (2, 24)
    assert scores.potential.shape == (2,)
    assert scores.terminal_logit.shape == (2,)
    assert torch.isfinite(scores.potential).all()


def test_action_features_are_fixed_width_and_geometry_relative() -> None:
    actions = lane.enumerate_legal_actions(SIMPLE)
    rendered = [
        lane.action_features(action, row_count=2, column_count=3)
        for action in actions
    ]
    assert all(len(features) == lane.ACTION_FEATURE_WIDTH for features in rendered)
    halt_index = next(
        index for index, action in enumerate(actions) if action.kind == lane.ACTION_HALT
    )
    assert rendered[halt_index][lane.ACTION_TO_INDEX[lane.ACTION_HALT]] == 1.0


def test_shuffled_binding_deranges_nonhalt_successors() -> None:
    matrix = lane.canonical_matrix(SIMPLE)
    actions = lane.enumerate_legal_actions(matrix)
    permutation = lane._binding_permutation(actions, matrix=matrix, seed=9)
    nonhalt = [
        index for index, action in enumerate(actions) if action.kind != lane.ACTION_HALT
    ]
    if len(nonhalt) > 1:
        assert all(permutation[index] != index for index in nonhalt)
    halt = next(
        index for index, action in enumerate(actions) if action.kind == lane.ACTION_HALT
    )
    assert permutation[halt] == halt


def test_arm_target_ablations_are_deterministic(tmp_path: Path) -> None:
    dataset, _ = lane.build_potential_dataset(
        (SIMPLE, IDENTITY),
        maximum_steps=32,
        preparation_root=tmp_path / "prep",
    )
    tensors = lane._build_training_tensors(dataset, device=torch.device("cpu"))
    shuffled_arm = next(
        arm for arm in lane.ARM_SPECS if arm.name == lane.ARM_SHUFFLED_DISTANCE
    )
    first = lane._arm_targets(
        tensors,
        arm=shuffled_arm,
        seed=12,
        maximum_distance=dataset.maximum_distance,
    )
    second = lane._arm_targets(
        tensors,
        arm=shuffled_arm,
        seed=12,
        maximum_distance=dataset.maximum_distance,
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], tensors.true_terminal)
    random_arm = next(
        arm for arm in lane.ARM_SPECS if arm.name == lane.ARM_RANDOM_LABELS
    )
    random_targets = lane._arm_targets(
        tensors,
        arm=random_arm,
        seed=99,
        maximum_distance=dataset.maximum_distance,
    )
    assert random_targets[0].shape == tensors.true_distances.shape
    assert random_targets[1].shape == tensors.true_terminal.shape


def test_tiny_training_has_zero_forbidden_calls(tmp_path: Path) -> None:
    dataset, _ = lane.build_potential_dataset(
        (SIMPLE, IDENTITY),
        maximum_steps=32,
        preparation_root=tmp_path / "prep",
    )
    tensors = lane._build_training_tensors(dataset, device=torch.device("cpu"))
    model = _tiny_model()
    treatment = next(
        arm for arm in lane.ARM_SPECS if arm.name == lane.ARM_TREATMENT
    )
    result = lane.train_arm(
        model,
        tensors,
        arm=treatment,
        config=lane.TrainingConfig(
            optimizer_updates=2,
            node_batch_size=4,
            decision_batch_size=2,
            learning_rate=1e-3,
            bf16=False,
        ),
        maximum_distance=dataset.maximum_distance,
        seed=4,
    )
    assert result.resources.optimizer_updates == 2
    assert result.resources.oracle_calls == 0
    assert result.resources.search_calls == 0
    assert result.resources.verifier_calls == 0
    assert result.final_loss >= 0.0


def test_candidate_receipt_has_zero_forbidden_calls() -> None:
    model = _tiny_model()
    rollout = lane.candidate_rollout(
        model,
        SIMPLE,
        inference_mode=lane.INFERENCE_POTENTIAL,
        binding_mode=lane.BINDING_RAW,
        binding_seed=3,
        maximum_steps=4,
        boundary=_boundary(),
    )
    assert rollout.resources.successor_evaluations > 0
    assert rollout.resources.oracle_calls == 0
    assert rollout.resources.search_calls == 0
    assert rollout.resources.verifier_calls == 0


def test_posthoc_assessor_is_strict() -> None:
    solved = lane.CandidateRollout(
        actions=(lane.SuccessorAction(lane.ACTION_HALT),),
        output_rows=IDENTITY,
        halted=True,
        cycled=False,
        overlong=False,
        potential_descent_steps=0,
        potential_nondescent_steps=0,
        binding_manifest_sha256="a" * 64,
        resources=lane.MutableCandidateResources().freeze(),
    )
    assert lane.assess_rollout_posthoc(IDENTITY, solved).strict_canonical_certified
    invalid = replace(solved, output_rows=lane.canonical_matrix(SIMPLE))
    assert not lane.assess_rollout_posthoc(SIMPLE, invalid).strict_canonical_certified


def test_cpu_packet_fit_smoke_destroys_packet(tmp_path: Path) -> None:
    packet = tmp_path / "packet.json"
    experiment = lane.ExperimentConfig(
        seed=101,
        train_matrices=2,
        evaluation_matrices=1,
        maximum_preparation_steps=32,
        maximum_rollout_steps=4,
    )
    lane.write_prepared_packet(experiment=experiment, output=packet)
    report = lane.run_bounded_experiment(
        prepared_packet_path=packet,
        model_config=lane.LyapunovConfig(
            field_width=4,
            width=8,
            cell_hidden=12,
            matrix_layers=1,
            state_hidden=12,
            coordinate_harmonics=1,
        ),
        training_config=lane.TrainingConfig(
            optimizer_updates=1,
            node_batch_size=2,
            decision_batch_size=1,
            learning_rate=1e-3,
            bf16=False,
        ),
        device=torch.device("cpu"),
        model_dir=tmp_path / "models",
    )
    assert not packet.exists()
    assert report.evaluation_boundary.prepared_packet_destroyed
    assert report.evaluation_boundary.training_labels_destroyed
    assert len(report.arms) == len(lane.REPORTED_ARMS)
    assert report.controls_equal_parameter_count
    assert report.controls_equal_optimizer_updates
    assert report.controls_equal_node_examples
    assert report.controls_equal_decision_examples
    assert report.controls_equal_transition_examples
    assert all(
        arm.candidate_resources.oracle_calls == 0
        and arm.candidate_resources.search_calls == 0
        and arm.candidate_resources.verifier_calls == 0
        for arm in report.arms
    )
