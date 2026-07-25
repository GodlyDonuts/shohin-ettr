from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import json
from pathlib import Path

import pytest
import torch

from episode_functor_algebra_machine import (
    OP_HALT,
    AlgebraInstruction,
)
import ssqac_search_distillation_pilot as pilot
from ssqac_controller_trace_pilot import compile_reference_program


@pytest.fixture(scope="module")
def prepared(
    tmp_path_factory: pytest.TempPathFactory,
) -> pilot.PreparedMatchedDatasets:
    matrices = pilot.generate_matrix_cases(
        seed=71,
        count=4,
        minimum_rows=2,
        maximum_rows=3,
        minimum_columns=2,
        maximum_columns=4,
    )
    scratch = tmp_path_factory.mktemp("search-distillation")
    return pilot.prepare_matched_datasets(
        matrices,
        seed=71,
        states_per_case=4,
        search_config=pilot.SearchPreparationConfig(
            max_nodes_expanded=512,
            max_edges_considered=10_000,
            max_depth=24,
            max_frontier=32,
            beam_width=32,
            max_program_instructions=96,
            policy_noise_scale=4.0,
        ),
        scratch_root=scratch,
    )


def test_strict_terminal_and_legal_macro_replay() -> None:
    source = ((2, 0), (0, 1))
    actions = pilot.enumerate_policy_actions(source)
    normalize = next(
        action
        for action in actions
        if action.kind == pilot.ACTION_NORMALIZE
        and action.row_a == 0
        and action.column == 0
    )
    primitives = pilot.compile_policy_action(source, normalize)
    assert [instruction.opcode for instruction in primitives] == [
        "LOAD",
        "INV",
        "SCALE",
    ]
    reduced = pilot.apply_policy_action(source, normalize)
    assert reduced == ((1, 0), (0, 1))
    assert pilot.strict_rref_terminal(reduced)
    assert pilot.enumerate_policy_actions(reduced) == (
        pilot.PolicyAction(pilot.ACTION_HALT),
    )


def test_search_macro_grammar_matches_reference_scheduler() -> None:
    source = ((0, 2, 1), (3, 0, 4))
    program = compile_reference_program(source)
    actions = pilot.decode_macro_program(source, program)
    replay = source
    rebuilt: list[AlgebraInstruction] = []
    for action in actions:
        rebuilt.extend(pilot.compile_policy_action(replay, action))
        replay = pilot.apply_policy_action(replay, action)
    assert tuple(rebuilt) == program
    assert actions[-1].kind == pilot.ACTION_HALT
    assert pilot.strict_rref_terminal(replay)


def test_autonomous_action_slots_depend_only_on_geometry() -> None:
    left = pilot.enumerate_candidate_action_universe(3, 4)
    right = pilot.enumerate_candidate_action_universe(3, 4)
    assert left == right
    assert len(left) == 1 + 3 * 4 + 3 * 2 * 4 + 3
    assert pilot.PolicyAction(pilot.ACTION_HALT) in left
    assert (
        pilot.PolicyAction(
            pilot.ACTION_ELIMINATE,
            row_a=2,
            row_b=0,
            column=3,
        )
        in left
    )


def test_decode_rejects_partial_or_ambiguous_program() -> None:
    source = ((2, 0), (0, 1))
    with pytest.raises(pilot.SearchDistillationError):
        pilot.decode_macro_program(
            source,
            (AlgebraInstruction("LOAD", 0, 0, 0),),
        )


def test_preparation_deletes_raw_search_traces_and_matches_states(
    prepared: pilot.PreparedMatchedDatasets,
) -> None:
    receipt = prepared.receipt
    assert receipt.completed_search_cases == receipt.requested_cases
    assert receipt.failed_search_cases == 0
    assert receipt.preparation_search_calls == receipt.requested_cases
    assert receipt.preparation_oracle_calls == receipt.retained_states
    assert receipt.trace_directory_deleted
    assert receipt.retained_search_trace_files == 0
    assert not receipt.raw_search_programs_retained
    assert receipt.arm_state_budgets_matched
    arms = [prepared.arm(name) for name in pilot.ARMS]
    assert len({len(states) for states in arms}) == 1
    assert (
        len({tuple(state.observation_sha256 for state in states) for states in arms})
        == 1
    )
    assert all(
        state.target in pilot.enumerate_policy_actions(state.rows)
        for states in arms
        for state in states
    )


def test_randomized_control_is_a_legal_alternative_when_available(
    prepared: pilot.PreparedMatchedDatasets,
) -> None:
    search = prepared.search_teacher
    randomized = prepared.randomized_label
    for treatment, control in zip(search, randomized, strict=True):
        legal = pilot.enumerate_policy_actions(treatment.rows)
        assert control.target in legal
        if len(legal) > 1:
            assert control.target != treatment.target


def test_preparation_manifests_are_exact_and_label_specific(
    prepared: pilot.PreparedMatchedDatasets,
) -> None:
    receipt = prepared.receipt
    assert receipt.observation_manifest_sha256 == pilot._observation_manifest(
        prepared.search_teacher
    )
    assert receipt.search_label_manifest_sha256 == pilot._label_manifest(
        prepared.search_teacher
    )
    assert receipt.ordinary_label_manifest_sha256 == pilot._label_manifest(
        prepared.ordinary_oracle
    )
    assert receipt.randomized_label_manifest_sha256 == pilot._label_manifest(
        prepared.randomized_label
    )
    assert (
        receipt.search_label_manifest_sha256 != receipt.randomized_label_manifest_sha256
    )


def test_tensorization_preserves_matched_observation_budget(
    prepared: pilot.PreparedMatchedDatasets,
) -> None:
    config = pilot.PolicyConfig(
        maximum_rows=6,
        maximum_columns=8,
        width=16,
        blocks=1,
        feedforward=32,
    )
    datasets = {
        name: pilot.tensorize_labeled_states(prepared.arm(name), config)
        for name in pilot.ARMS
    }
    assert len({dataset.examples for dataset in datasets.values()}) == 1
    assert (
        len({dataset.observation_manifest_sha256 for dataset in datasets.values()}) == 1
    )
    assert all(
        dataset.action_mask.gather(1, dataset.targets[:, None]).all()
        for dataset in datasets.values()
    )


def test_shared_policy_is_invariant_to_extra_masked_padding() -> None:
    state = pilot.LabeledPolicyState(
        rows=((2, 0, 1), (0, 1, 3)),
        target=pilot.PolicyAction(
            pilot.ACTION_NORMALIZE,
            row_a=0,
            column=0,
        ),
    )
    small_config = pilot.PolicyConfig(
        maximum_rows=3,
        maximum_columns=4,
        width=24,
        blocks=2,
        feedforward=48,
    )
    large_config = replace(
        small_config,
        maximum_rows=6,
        maximum_columns=8,
    )
    torch.manual_seed(9)
    small_model = pilot.EquivariantActionPolicy(small_config).eval()
    large_model = pilot.EquivariantActionPolicy(large_config).eval()
    large_model.load_state_dict(small_model.state_dict())
    small = pilot.tensorize_labeled_states((state,), small_config)
    large = pilot.tensorize_labeled_states((state,), large_config)
    with torch.no_grad():
        small_logits = small_model(**small.select(torch.tensor([0])))
        large_logits = large_model(**large.select(torch.tensor([0])))
    valid = int(small.action_mask[0].sum().item())
    assert torch.allclose(
        small_logits[0, :valid],
        large_logits[0, :valid],
        atol=1e-6,
        rtol=1e-6,
    )


def test_matched_models_receive_identical_parameter_and_update_budgets(
    prepared: pilot.PreparedMatchedDatasets,
) -> None:
    config = pilot.PolicyConfig(
        maximum_rows=6,
        maximum_columns=8,
        width=16,
        blocks=1,
        feedforward=32,
    )
    datasets = {
        name: pilot.tensorize_labeled_states(prepared.arm(name), config)
        for name in pilot.ARMS
    }
    torch.manual_seed(13)
    template = pilot.EquivariantActionPolicy(config)
    initial = template.state_dict()
    models = {name: pilot.EquivariantActionPolicy(config) for name in pilot.ARMS}
    for model in models.values():
        model.load_state_dict(initial)
    assert len({pilot.model_state_sha256(model) for model in models.values()}) == 1
    schedule = pilot.build_batch_schedule(
        examples=len(prepared.search_teacher),
        epochs=1,
        batch_size=8,
        seed=19,
    )
    training = pilot.TrainingConfig(
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=0.0,
        amp_bfloat16=False,
        torch_compile=False,
    )
    receipts = {
        name: pilot.train_policy(
            models[name],
            datasets[name],
            schedule=schedule,
            config=training,
            device=torch.device("cpu"),
        )
        for name in pilot.ARMS
    }
    assert len({model.parameter_count for model in models.values()}) == 1
    assert len({receipt.optimizer_updates for receipt in receipts.values()}) == 1
    assert len({receipt.batch_schedule_sha256 for receipt in receipts.values()}) == 1


def test_autonomous_runtime_has_no_preparation_or_assessor_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = pilot.PolicyConfig(
        maximum_rows=4,
        maximum_columns=5,
        width=16,
        blocks=1,
        feedforward=32,
    )

    class HaltOnlyPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config

        def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
            action_kind = inputs["action_kind"]
            halt_index = pilot.ACTION_TYPES.index(pilot.ACTION_HALT)
            return torch.where(
                action_kind == halt_index,
                torch.ones_like(action_kind, dtype=torch.float32),
                torch.full_like(action_kind, -1.0, dtype=torch.float32),
            )

    model = HaltOnlyPolicy()

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("forbidden dependency was called")

    monkeypatch.setattr(pilot, "prepare_matched_datasets", forbidden)
    monkeypatch.setattr(pilot, "assess_candidate_program", forbidden)
    monkeypatch.setattr(pilot, "verify_reduction_program", forbidden)
    monkeypatch.setattr(pilot, "enumerate_policy_actions", forbidden)
    monkeypatch.setattr(pilot, "compile_policy_action", forbidden)
    counter = pilot.RuntimeAccessCounter()
    candidate = pilot.greedy_model_vm_rollout(
        model,
        ((1, 0), (0, 1)),
        device=torch.device("cpu"),
        amp_bfloat16=False,
        maximum_macros=4,
        maximum_instructions=8,
        counter=counter,
    )
    assert candidate.termination == "halted"
    assert candidate.program == (AlgebraInstruction(OP_HALT),)
    assert counter.search_calls == 0
    assert counter.oracle_calls == 0
    assert counter.verifier_calls == 0
    tree = ast.parse(inspect.getsource(pilot.greedy_model_vm_rollout))
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    for forbidden_name in (
        "bounded_beam",
        "enumerate_policy_actions",
        "structural_potential",
        "oracle",
        "verify_reduction_program",
    ):
        assert all(forbidden_name not in name for name in called_names)


def test_separate_assessor_is_the_only_strict_certification_boundary() -> None:
    matrix = ((1, 0), (0, 1))
    valid = pilot.CandidateProgram(
        program=(AlgebraInstruction(OP_HALT),),
        termination="halted",
        model_decisions=1,
        vm_calls=1,
    )
    counter = pilot.RuntimeAccessCounter()
    assessment = pilot.assess_candidate_program(
        matrix,
        valid,
        counter=counter,
    )
    assert assessment.passed
    assert counter.verifier_calls == 1
    invalid = replace(valid, program=())
    rejected = pilot.assess_candidate_program(matrix, invalid)
    assert not rejected.passed


def test_geometry_holdout_must_be_strict_on_both_axes() -> None:
    args = pilot._parse_args(["--smoke"])
    args.evaluation_minimum_rows = args.fit_maximum_rows
    with pytest.raises(pilot.SearchDistillationError):
        pilot.run_pilot(args)


def test_h100_cli_exposes_non_smoke_capacity_controls() -> None:
    args = pilot._parse_args(
        [
            "--device",
            "cuda",
            "--train-cases",
            "512",
            "--evaluation-cases",
            "256",
            "--width",
            "384",
            "--blocks",
            "6",
            "--feedforward",
            "1536",
            "--torch-compile",
        ]
    )
    assert args.device == "cuda"
    assert args.train_cases == 512
    assert args.evaluation_cases == 256
    assert args.width == 384
    assert args.blocks == 6
    assert args.feedforward == 1536
    assert args.torch_compile


def test_bounded_cpu_smoke_cli_writes_auditable_report(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"
    assert pilot.main(["--smoke", "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="ascii"))
    assert report["schema"] == pilot.SCHEMA
    assert report["status"] == pilot.STATUS
    assert report["reasoning_claim_authorized"] is False
    assert report["strict_geometry_disjoint"] is True
    assert report["matched_parameter_budget"] is True
    assert report["matched_update_budget"] is True
    assert report["matched_data_budget"] is True
    assert report["identical_initial_weights"] is True
    assert report["final_search_calls"] == 0
    assert report["final_oracle_calls"] == 0
    assert report["preparation"]["trace_directory_deleted"] is True
    assert report["preparation"]["retained_search_trace_files"] == 0
    assert len(report["arms"]) == 3
    assert len({arm["optimizer_updates"] for arm in report["arms"]}) == 1
    assert all(arm["final_search_calls"] == 0 for arm in report["arms"])
    assert all(arm["final_oracle_calls"] == 0 for arm in report["arms"])
