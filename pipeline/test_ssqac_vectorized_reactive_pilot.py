from __future__ import annotations

from argparse import Namespace
import json

import pytest
import torch
from torch import nn

from episode_functor_algebra_machine import (
    FIELD_MODULUS,
    OP_AXPY,
    OP_LOAD,
    OP_SCALE,
    OP_SWAP,
    AlgebraInstruction,
    execute_program,
)
from pipeline import ssqac_controller_trace_pilot as trace_pilot
from pipeline import ssqac_vectorized_reactive_pilot as pilot


def _config(
    *,
    maximum_rows: int = 3,
    maximum_columns: int = 4,
) -> pilot.ReactivePolicyConfig:
    return pilot.ReactivePolicyConfig(
        maximum_rows=maximum_rows,
        maximum_columns=maximum_columns,
        register_count=4,
        width=16,
        blocks=1,
        feedforward=32,
        dropout=0.0,
    )


def _tiny_args() -> Namespace:
    return Namespace(
        seed=91,
        train_examples=2,
        audit_examples=1,
        evaluation_examples=2,
        fit_maximum_rows=2,
        fit_maximum_columns=3,
        evaluation_minimum_rows=3,
        evaluation_minimum_columns=4,
        maximum_rows=3,
        maximum_columns=4,
        registers=4,
        width=16,
        blocks=1,
        feedforward=32,
        dropout=0.0,
        epochs=1,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=0.01,
        maximum_rollout_steps=4,
        device="cpu",
        amp_bfloat16=False,
        compile=False,
        material_minimum_evaluation_cases=8,
        material_minimum_certification_rate=0.8,
        output=None,
        model_output=None,
    )


def _states(
    *,
    seed: int = 1,
    count: int = 2,
) -> tuple[pilot.FlattenedExpertState, ...]:
    examples = trace_pilot.generate_examples(
        seed=seed,
        count=count,
        maximum_rows=2,
        maximum_columns=3,
        register_count=4,
    )
    return pilot.flatten_expert_states(examples)


def _observation() -> pilot.ControllerObservation:
    return pilot.ControllerObservation(
        rows=((3, 0, 7), (0, 1, 0)),
        registers=(3, 0, 0, 0),
        previous_instruction=AlgebraInstruction(OP_LOAD, 0, 0, 0),
    )


def _inverse_order(order: list[int]) -> list[int]:
    inverse = [0] * len(order)
    for new_index, old_index in enumerate(order):
        inverse[old_index] = new_index
    return inverse


def _recode_instruction(
    instruction: AlgebraInstruction,
    row_old_to_new: list[int],
    column_old_to_new: list[int],
) -> AlgebraInstruction:
    if instruction.opcode == OP_LOAD:
        return AlgebraInstruction(
            OP_LOAD,
            row_old_to_new[instruction.a],
            column_old_to_new[instruction.b],
            instruction.c,
        )
    if instruction.opcode == OP_SCALE:
        return AlgebraInstruction(
            OP_SCALE,
            row_old_to_new[instruction.a],
            instruction.b,
        )
    if instruction.opcode == OP_AXPY:
        return AlgebraInstruction(
            OP_AXPY,
            row_old_to_new[instruction.a],
            row_old_to_new[instruction.b],
            instruction.c,
        )
    if instruction.opcode == OP_SWAP:
        return AlgebraInstruction(
            OP_SWAP,
            row_old_to_new[instruction.a],
            row_old_to_new[instruction.b],
        )
    return instruction


def _permuted_observation(
    observation: pilot.ControllerObservation,
    row_order: list[int],
    column_order: list[int],
) -> tuple[pilot.ControllerObservation, list[int], list[int]]:
    row_old_to_new = _inverse_order(row_order)
    column_old_to_new = _inverse_order(column_order)
    previous = observation.previous_instruction
    if previous is not None:
        previous = _recode_instruction(
            previous,
            row_old_to_new,
            column_old_to_new,
        )
    return (
        pilot.ControllerObservation(
            rows=tuple(
                tuple(
                    observation.rows[old_row][old_column] for old_column in column_order
                )
                for old_row in row_order
            ),
            registers=observation.registers,
            previous_instruction=previous,
        ),
        row_old_to_new,
        column_old_to_new,
    )


def _force_opcode(
    logits: pilot.PolicyLogits,
    opcode: str,
) -> pilot.PolicyLogits:
    opcode_logits = torch.full_like(logits.opcode, -10_000.0)
    opcode_logits[:, pilot._OPCODE_TO_INDEX[opcode]] = 10_000.0
    return pilot.PolicyLogits(
        opcode=opcode_logits,
        row_a=logits.row_a,
        row_b=logits.row_b,
        column=logits.column,
        register_a=logits.register_a,
        register_b=logits.register_b,
    )


def test_flattening_has_no_step_or_recurrent_state() -> None:
    states = _states()
    assert states
    assert all(not hasattr(state, "step") for state in states)
    assert all(not hasattr(state, "hidden") for state in states)
    assert all(
        isinstance(state.target_instruction, AlgebraInstruction) for state in states
    )


def test_tensorization_is_exact_masked_and_repeatable() -> None:
    states = _states()
    left = pilot.tensorize_states(states, _config())
    right = pilot.tensorize_states(states, _config())
    assert left.manifest_sha256 == right.manifest_sha256
    assert left.source_state_manifest_sha256 == right.source_state_manifest_sha256
    assert left.tensors["rows"].shape[1:] == (3, 4)
    assert left.tensors["row_mask"].dtype == torch.bool
    assert left.tensors["column_mask"].dtype == torch.bool
    visible = (
        left.tensors["row_mask"][:, :, None] & left.tensors["column_mask"][:, None, :]
    )
    assert torch.all(left.tensors["rows"][~visible] == 0)
    assert set(left.tensors) == {
        *pilot._INPUT_TENSOR_NAMES,
        *pilot._TARGET_TENSOR_NAMES,
    }


def test_load_target_uses_typed_row_column_and_register_labels() -> None:
    state = pilot.FlattenedExpertState(
        rows=((5, 0), (0, 1)),
        registers=(0, 0, 0, 0),
        previous_instruction=None,
        target_instruction=AlgebraInstruction(OP_LOAD, 0, 0, 3),
    )
    dataset = pilot.tensorize_states((state,), _config())
    assert dataset.tensors["target_opcode"].item() == (pilot._OPCODE_TO_INDEX[OP_LOAD])
    assert dataset.tensors["target_row_a"].item() == 0
    assert dataset.tensors["target_column"].item() == 0
    assert dataset.tensors["target_register_a"].item() == 3
    assert dataset.tensors["target_row_b"].item() == pilot.IGNORE_INDEX
    assert dataset.tensors["target_register_b"].item() == (pilot.IGNORE_INDEX)


def test_model_has_no_geometry_sized_embedding_or_parameter_table() -> None:
    small = pilot.GeometryEquivariantReactivePolicy(
        _config(maximum_rows=3, maximum_columns=4)
    )
    large = pilot.GeometryEquivariantReactivePolicy(
        _config(maximum_rows=6, maximum_columns=8)
    )
    assert {
        name: tuple(tensor.shape) for name, tensor in small.state_dict().items()
    } == {name: tuple(tensor.shape) for name, tensor in large.state_dict().items()}
    embeddings = [
        module for module in small.modules() if isinstance(module, nn.Embedding)
    ]
    assert {embedding.num_embeddings for embedding in embeddings} == {
        FIELD_MODULUS,
        len(pilot.OPCODES) + 1,
        small.config.register_count,
    }
    assert not hasattr(small, "row_embedding")
    assert not hasattr(small, "column_embedding")
    assert not hasattr(small, "row_positions")
    assert not hasattr(small, "column_positions")
    assert not hasattr(small, "row_coordinate_projection")
    assert not hasattr(small, "column_coordinate_projection")


def test_valid_logits_are_invariant_to_masked_padding_values() -> None:
    torch.manual_seed(3)
    model = pilot.GeometryEquivariantReactivePolicy(_config()).eval()
    inputs = pilot.tensorize_observations((_observation(),), _config())
    altered = {name: tensor.clone() for name, tensor in inputs.items()}
    visible = altered["row_mask"][:, :, None] & altered["column_mask"][:, None, :]
    altered["rows"][~visible] = 211
    with torch.no_grad():
        left = model(**inputs)
        right = model(**altered)
    for name, left_logits in left.as_mapping().items():
        right_logits = right.as_mapping()[name]
        finite = torch.isfinite(left_logits)
        assert torch.equal(finite, torch.isfinite(right_logits))
        assert torch.allclose(
            left_logits[finite],
            right_logits[finite],
            atol=1e-6,
            rtol=1e-6,
        ), name


def test_same_weights_extend_to_larger_padding_geometry() -> None:
    torch.manual_seed(4)
    small_config = _config(maximum_rows=3, maximum_columns=4)
    large_config = _config(maximum_rows=6, maximum_columns=8)
    small = pilot.GeometryEquivariantReactivePolicy(small_config).eval()
    large = pilot.GeometryEquivariantReactivePolicy(large_config).eval()
    large.load_state_dict(small.state_dict(), strict=True)
    small_inputs = pilot.tensorize_observations(
        (_observation(),),
        small_config,
    )
    large_inputs = pilot.tensorize_observations(
        (_observation(),),
        large_config,
    )
    with torch.no_grad():
        small_logits = small(**small_inputs)
        large_logits = large(**large_inputs)
    assert torch.allclose(
        small_logits.opcode,
        large_logits.opcode,
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        small_logits.row_a[:, :2],
        large_logits.row_a[:, :2],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        small_logits.column[:, :3],
        large_logits.column[:, :3],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.all(torch.isneginf(large_logits.row_a[:, 2:]))
    assert torch.all(torch.isneginf(large_logits.column[:, 3:]))


def test_random_row_column_recodings_permute_logits_and_actions() -> None:
    torch.manual_seed(8)
    config = _config(maximum_rows=4, maximum_columns=5)
    model = pilot.GeometryEquivariantReactivePolicy(config).eval()
    observation = pilot.ControllerObservation(
        rows=(
            (2, 5, 0, 11),
            (7, 0, 13, 3),
            (17, 19, 23, 0),
        ),
        registers=(29, 31, 37, 41),
        previous_instruction=AlgebraInstruction(OP_LOAD, 2, 1, 0),
    )
    original_inputs = pilot.tensorize_observations((observation,), config)
    with torch.no_grad():
        original = model(**original_inputs)

    for permutation_seed in range(12):
        generator = torch.Generator().manual_seed(100 + permutation_seed)
        row_order = torch.randperm(3, generator=generator).tolist()
        column_order = torch.randperm(4, generator=generator).tolist()
        permuted_observation, row_old_to_new, column_old_to_new = _permuted_observation(
            observation,
            row_order,
            column_order,
        )
        permuted_inputs = pilot.tensorize_observations(
            (permuted_observation,),
            config,
        )
        with torch.no_grad():
            permuted = model(**permuted_inputs)

        assert torch.allclose(
            permuted.opcode,
            original.opcode,
            atol=2e-5,
            rtol=2e-5,
        )
        assert torch.allclose(
            permuted.register_a,
            original.register_a,
            atol=2e-5,
            rtol=2e-5,
        )
        assert torch.allclose(
            permuted.register_b,
            original.register_b,
            atol=2e-5,
            rtol=2e-5,
        )
        for name in ("row_a", "row_b"):
            assert torch.allclose(
                permuted.as_mapping()[name][:, :3],
                original.as_mapping()[name][:, row_order],
                atol=2e-5,
                rtol=2e-5,
            ), (permutation_seed, name)
        assert torch.allclose(
            permuted.column[:, :4],
            original.column[:, column_order],
            atol=2e-5,
            rtol=2e-5,
        )
        for opcode in (OP_LOAD, OP_SCALE, OP_AXPY, OP_SWAP):
            original_action = pilot._harden_batch(_force_opcode(original, opcode))[0]
            permuted_action = pilot._harden_batch(_force_opcode(permuted, opcode))[0]
            assert permuted_action == _recode_instruction(
                original_action,
                row_old_to_new,
                column_old_to_new,
            )


def test_preparation_oracle_is_order_sensitive_diagnostic() -> None:
    original_matrix = ((2, 0), (0, 1))
    row_swapped_matrix = (original_matrix[1], original_matrix[0])
    original_target = trace_pilot.next_reference_repair_instruction(
        execute_program(original_matrix, (), register_count=4),
        None,
    )
    row_swapped_target = trace_pilot.next_reference_repair_instruction(
        execute_program(row_swapped_matrix, (), register_count=4),
        None,
    )
    recoded_original_target = _recode_instruction(
        original_target,
        [1, 0],
        [0, 1],
    )
    assert original_target == AlgebraInstruction(OP_LOAD, 0, 0, 0)
    assert row_swapped_target == AlgebraInstruction(OP_SWAP, 0, 1)
    assert row_swapped_target != recoded_original_target


def test_vectorized_training_updates_model_from_resident_tensors() -> None:
    torch.manual_seed(5)
    config = _config()
    dataset = pilot.tensorize_states(_states(seed=5), config)
    model = pilot.GeometryEquivariantReactivePolicy(config)
    before = pilot.model_state_sha256(model)
    metrics = pilot.train_vectorized_policy(
        model,
        dataset,
        epochs=1,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=0.01,
        device=torch.device("cpu"),
        amp_bfloat16=False,
        compile_model=False,
        shuffle_seed=6,
    )
    after = pilot.model_state_sha256(model)
    teacher = pilot.teacher_forced_metrics(
        model,
        dataset,
        batch_size=32,
        device=torch.device("cpu"),
        amp_bfloat16=False,
    )
    assert metrics.optimizer_updates > 0
    assert math_is_finite(metrics.mean_training_loss)
    assert math_is_finite(metrics.final_training_loss)
    assert before != after
    assert teacher.full_instruction_total == dataset.examples
    assert 0.0 <= teacher.full_instruction_accuracy <= 1.0


def math_is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def test_autonomous_matrix_only_rollout_has_no_oracle_call_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = trace_pilot.generate_examples(
        seed=7,
        count=2,
        maximum_rows=3,
        maximum_columns=4,
        register_count=4,
        minimum_rows=3,
        minimum_columns=4,
    )
    cases = pilot.delete_expert_artifacts(examples)
    assert all(not hasattr(case, "program") for case in cases)
    assert all(not hasattr(case, "snapshots") for case in cases)

    def forbidden(*_args: object, **_kwargs: object) -> AlgebraInstruction:
        raise AssertionError("preparation oracle leaked into final rollout")

    monkeypatch.setattr(
        trace_pilot,
        "next_reference_repair_instruction",
        forbidden,
    )
    assert (
        "next_reference_repair_instruction"
        not in pilot.autonomous_matrix_only_evaluate.__code__.co_names
    )
    model = pilot.GeometryEquivariantReactivePolicy(_config())
    metrics = pilot.autonomous_matrix_only_evaluate(
        model,
        cases,
        maximum_rollout_steps=4,
        device=torch.device("cpu"),
        amp_bfloat16=False,
    )
    assert metrics.total == len(cases)
    assert metrics.oracle_calls == 0
    assert metrics.model_batches <= 4


def test_material_gate_requires_enough_exact_certifications() -> None:
    weak = pilot.RolloutMetrics(
        certified=7,
        invalid=1,
        overlong=0,
        oracle_calls=0,
        model_batches=3,
        model_decisions=8,
    )
    assert not pilot._material_gate(
        weak,
        minimum_cases=64,
        minimum_rate=0.8,
    )
    strong = pilot.RolloutMetrics(
        certified=58,
        invalid=6,
        overlong=0,
        oracle_calls=0,
        model_batches=20,
        model_decisions=512,
    )
    assert pilot._material_gate(
        strong,
        minimum_cases=64,
        minimum_rate=0.8,
    )
    contaminated = pilot.RolloutMetrics(
        certified=64,
        invalid=0,
        overlong=0,
        oracle_calls=1,
        model_batches=20,
        model_decisions=512,
    )
    assert not pilot._material_gate(
        contaminated,
        minimum_cases=64,
        minimum_rate=0.8,
    )


def test_geometry_gate_requires_a_strict_larger_split() -> None:
    args = _tiny_args()
    args.evaluation_minimum_rows = args.fit_maximum_rows
    with pytest.raises(
        pilot.VectorizedReactivePilotError,
        match="strictly larger",
    ):
        pilot.run_pilot(args)


def test_bounded_cpu_pilot_reports_exact_non_reasoning_boundary() -> None:
    report = pilot.run_pilot(_tiny_args())
    assert report.schema == pilot.PILOT_SCHEMA
    assert report.status == pilot.STATUS_NOT_REASONING
    assert "not_reasoning" in report.status
    assert report.flattened_state_dataset
    assert report.dataset_resident_on_device
    assert not report.step_signal_exposed
    assert not report.recurrent_state
    assert not report.learned_absolute_row_table
    assert not report.learned_absolute_column_table
    assert not report.row_coordinate_features
    assert not report.column_coordinate_features
    assert report.exact_row_permutation_equivariance
    assert report.exact_column_permutation_equivariance
    assert report.preparation_oracle_order_sensitive
    assert report.shared_content_pointer_heads
    assert report.strict_geometry_disjoint
    assert report.fit_maximum_rows < report.evaluation_minimum_rows
    assert report.fit_maximum_columns < report.evaluation_minimum_columns
    assert report.final_rollout_oracle_calls == 0
    assert report.closed_loop_total == report.evaluation_matrices
    assert not report.material_certification_gate_passed
    assert report.optimizer_updates > 0
    assert len(report.model_state_sha256) == 64
    assert len(report.train_tensor_manifest_sha256) == 64
    assert len(report.canonical_bytes()) > 500


def test_cli_writes_canonical_report_and_hashed_model(
    tmp_path: pytest.TempPathFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "report.json"
    model_path = tmp_path / "model.pt"
    result = pilot.main(
        [
            "--seed",
            "92",
            "--train-examples",
            "1",
            "--audit-examples",
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
            "12",
            "--blocks",
            "1",
            "--feedforward",
            "24",
            "--epochs",
            "1",
            "--batch-size",
            "32",
            "--maximum-rollout-steps",
            "2",
            "--device",
            "cpu",
            "--no-amp-bfloat16",
            "--no-compile",
            "--material-minimum-evaluation-cases",
            "8",
            "--output",
            str(report_path),
            "--model-output",
            str(model_path),
        ]
    )
    stdout = capsys.readouterr().out
    assert result == 0
    assert report_path.read_text() == stdout
    payload = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )
    report_data = json.loads(stdout)
    assert payload["schema"] == pilot.MODEL_SCHEMA
    assert payload["model_state_sha256"] == report_data["model_state_sha256"]
