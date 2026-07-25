from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from episode_functor_algebra_machine import (
    OP_HALT,
    OP_INV,
    OP_LOAD,
    AlgebraInstruction,
    execute_program,
)
from episode_functor_neural_algebra_controller import (
    ControllerConfig,
    NeuralAlgebraController,
)
from pipeline import ssqac_reactive_dagger_pilot as pilot
from pipeline.ssqac_controller_trace_pilot import (
    generate_examples,
    next_reference_repair_instruction,
)


def _controller() -> NeuralAlgebraController:
    return NeuralAlgebraController(
        ControllerConfig(
            maximum_rows=4,
            maximum_columns=5,
            register_count=4,
            width=32,
            layers=1,
            heads=4,
            feedforward=64,
            maximum_steps=1,
        )
    )


def _tiny_args() -> Namespace:
    return Namespace(
        seed=77,
        expert_train_examples=2,
        expert_audit_examples=2,
        collection_examples=2,
        dagger_rounds=1,
        evaluation_examples=2,
        fit_maximum_rows=2,
        fit_maximum_columns=3,
        evaluation_minimum_rows=3,
        evaluation_minimum_columns=4,
        maximum_rows=4,
        maximum_columns=5,
        registers=4,
        width=24,
        layers=1,
        heads=4,
        feedforward=48,
        initial_epochs=1,
        dagger_epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        pre_error_weight=2.0,
        maximum_rollout_steps=4,
        device="cpu",
        output=None,
    )


def test_flattened_states_are_step_free_and_oracle_labeled() -> None:
    examples = generate_examples(
        seed=1,
        count=2,
        maximum_rows=2,
        maximum_columns=3,
        register_count=4,
    )
    states = pilot.flatten_expert_states(examples)
    assert states
    assert all(not hasattr(state, "step") for state in states)
    for state in states:
        snapshot = execute_program(
            state.rows,
            (),
            register_count=4,
        )
        snapshot = snapshot.__class__(
            schema=snapshot.schema,
            rows=state.rows,
            provenance=snapshot.provenance,
            registers=state.registers,
            halted=False,
            executed_instructions=0,
            opcode_counts=snapshot.opcode_counts,
            trace_sha256=snapshot.trace_sha256,
        )
        assert (
            next_reference_repair_instruction(
                snapshot,
                state.previous_instruction,
            )
            == state.target_instruction
        )


def test_deduplication_uses_observation_and_previous_instruction() -> None:
    base = pilot.ReactiveStateLabel(
        rows=((1, 0), (0, 1)),
        registers=(0, 0, 0, 0),
        previous_instruction=None,
        target_instruction=AlgebraInstruction(OP_HALT),
    )
    pre_error = pilot.ReactiveStateLabel(
        rows=base.rows,
        registers=base.registers,
        previous_instruction=None,
        target_instruction=base.target_instruction,
        pre_error=True,
    )
    different_previous = pilot.ReactiveStateLabel(
        rows=base.rows,
        registers=base.registers,
        previous_instruction=AlgebraInstruction(OP_LOAD, 0, 0, 0),
        target_instruction=base.target_instruction,
    )
    unique = pilot.deduplicate_states(
        (base, pre_error, different_previous)
    )
    assert len(unique) == 2
    assert sum(state.pre_error for state in unique) == 1


def test_conflicting_oracle_labels_fail_closed() -> None:
    left = pilot.ReactiveStateLabel(
        rows=((1, 0),),
        registers=(0, 0, 0, 0),
        previous_instruction=None,
        target_instruction=AlgebraInstruction(OP_HALT),
    )
    right = pilot.ReactiveStateLabel(
        rows=left.rows,
        registers=left.registers,
        previous_instruction=None,
        target_instruction=AlgebraInstruction(OP_LOAD, 0, 0, 0),
    )
    with pytest.raises(RuntimeError, match="labels conflict"):
        pilot.deduplicate_states((left, right))


def test_controller_reached_pre_error_state_is_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = generate_examples(
        seed=2,
        count=1,
        maximum_rows=2,
        maximum_columns=3,
        register_count=4,
    )
    monkeypatch.setattr(
        pilot,
        "_predict_instruction",
        lambda *_args, **_kwargs: AlgebraInstruction(OP_INV, 0, 1),
    )
    result = pilot.collect_on_policy_states(
        _controller(),
        pilot.delete_expert_traces(examples),
        device=torch.device("cpu"),
        maximum_rollout_steps=4,
    )
    assert result.reached_valid_states == 1
    assert result.pre_error_states == 1
    assert result.states[0].pre_error
    assert result.counts.invalid == 1
    assert result.states[0].target_instruction == (
        next_reference_repair_instruction(
            execute_program(examples[0].matrix, (), register_count=4),
            None,
        )
    )


def test_final_rollout_has_no_oracle_call_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = generate_examples(
        seed=3,
        count=2,
        maximum_rows=4,
        maximum_columns=5,
        register_count=4,
        minimum_rows=3,
        minimum_columns=4,
    )

    def forbidden(*_args: object, **_kwargs: object) -> AlgebraInstruction:
        raise AssertionError("preparation oracle leaked into final rollout")

    monkeypatch.setattr(
        pilot,
        "next_reference_repair_instruction",
        forbidden,
    )
    matrix_cases = pilot.delete_expert_traces(examples)
    assert all(not hasattr(case, "program") for case in matrix_cases)
    assert all(not hasattr(case, "snapshots") for case in matrix_cases)
    assert (
        "next_reference_repair_instruction"
        not in pilot.final_oracle_free_rollout.__code__.co_names
    )
    counts = pilot.final_oracle_free_rollout(
        _controller(),
        matrix_cases,
        device=torch.device("cpu"),
        maximum_rollout_steps=4,
    )
    assert counts.total == len(examples)
    assert counts.oracle_calls == 0


def test_reactive_training_update_is_finite() -> None:
    examples = generate_examples(
        seed=4,
        count=2,
        maximum_rows=2,
        maximum_columns=3,
        register_count=4,
    )
    states = pilot.flatten_expert_states(examples)
    controller = _controller()
    before = pilot._model_state_sha256(controller)
    updates = pilot.train_reactive_policy(
        controller,
        states,
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        pre_error_weight=2.0,
        device=torch.device("cpu"),
        shuffle_seed=9,
    )
    after = pilot._model_state_sha256(controller)
    accuracy = pilot.expert_instruction_accuracy(
        controller,
        states,
        batch_size=8,
        device=torch.device("cpu"),
    )
    assert updates > 0
    assert before != after
    assert 0.0 <= accuracy <= 1.0


def test_geometry_gate_requires_strictly_larger_final_split() -> None:
    args = _tiny_args()
    args.evaluation_minimum_rows = args.fit_maximum_rows
    with pytest.raises(ValueError, match="strictly larger"):
        pilot.run_pilot(args)


def test_tiny_pilot_reports_auditable_isolated_falsifier() -> None:
    report = pilot.run_pilot(_tiny_args())
    assert report.schema == pilot.PILOT_SCHEMA
    assert report.status == pilot.STATUS
    assert "not_reasoning" in report.status
    assert report.candidate_runtime == pilot.RUNTIME_BOUNDARY
    assert report.final_rollout_oracle_calls == 0
    assert report.fit_maximum_rows < report.evaluation_minimum_rows
    assert report.fit_maximum_columns < report.evaluation_minimum_columns
    assert report.final_closed_loop_total == report.evaluation_matrices
    assert len(report.rounds) == 1
    assert report.rounds[0].reached_valid_states > 0
    assert report.rounds[0].aggregate_states == report.final_aggregate_states
    assert report.total_optimizer_updates == (
        report.initial_optimizer_updates + report.dagger_optimizer_updates
    )
    assert report.initial_model_state_sha256
    assert report.final_model_state_sha256
    assert len(report.canonical_bytes()) > 100


def test_cli_writes_exact_canonical_report(
    tmp_path: pytest.TempPathFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "report.json"
    result = pilot.main(
        [
            "--seed",
            "88",
            "--expert-train-examples",
            "1",
            "--expert-audit-examples",
            "1",
            "--collection-examples",
            "1",
            "--dagger-rounds",
            "1",
            "--evaluation-examples",
            "1",
            "--fit-maximum-rows",
            "2",
            "--fit-maximum-columns",
            "3",
            "--evaluation-minimum-rows",
            "3",
            "--evaluation-minimum-columns",
            "4",
            "--maximum-rows",
            "3",
            "--maximum-columns",
            "4",
            "--width",
            "16",
            "--layers",
            "1",
            "--heads",
            "4",
            "--feedforward",
            "32",
            "--initial-epochs",
            "1",
            "--dagger-epochs",
            "1",
            "--batch-size",
            "8",
            "--maximum-rollout-steps",
            "4",
            "--output",
            str(output),
        ]
    )
    assert result == 0
    assert output.read_text() == capsys.readouterr().out
