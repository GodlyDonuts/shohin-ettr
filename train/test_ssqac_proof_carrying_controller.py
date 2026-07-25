"""Focused tests for the isolated proof-carrying SSQAC falsifier."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import inspect
from pathlib import Path

import pytest
import torch

import ssqac_proof_carrying_controller as proof


def _small_state() -> proof.ContractState:
    rows = proof.canonical_matrix(((2, 1), (0, 1)))
    actions = proof.enumerate_legal_actions(rows)
    expert = next(
        action for action in actions if action.kind == "NORMALIZE"
    )
    candidates = []
    for action in actions:
        successor = proof.apply_action(rows, action)
        candidates.append(
            proof.ContractCandidate(
                action=action,
                successor=successor,
                expert_preference=int(action == expert),
                current_remaining=2,
                successor_remaining=1 if action == expert else 2,
                successor_terminal=0,
                progress_consistent=int(action == expert),
            )
        )
    return proof.ContractState(rows, tuple(candidates))


def _tiny_config() -> proof.ControllerConfig:
    return proof.ControllerConfig(
        field_width=8,
        width=16,
        cell_hidden=24,
        matrix_layers=1,
        contract_hidden=20,
        coordinate_harmonics=1,
    )


def test_contract_state_requires_exact_legal_successors() -> None:
    state = _small_state()
    damaged = replace(
        state.candidates[0],
        successor=state.rows,
    )
    with pytest.raises(proof.ProofCarryingError):
        proof.ContractState(
            state.rows,
            (damaged, *state.candidates[1:]),
        )


def test_binding_control_is_a_deterministic_derangement() -> None:
    state = _small_state()
    actions = tuple(candidate.action for candidate in state.candidates)
    permutation = proof._binding_permutation(state.rows, actions, seed=17)
    assert sorted(permutation) == list(range(len(actions)))
    assert all(index != mapped for index, mapped in enumerate(permutation))
    assert permutation == proof._binding_permutation(state.rows, actions, seed=17)


def test_controller_has_distinct_heads_and_respects_budget() -> None:
    model = proof.ProofCarryingController(_tiny_config())
    breakdown = model.parameter_count_breakdown()
    assert breakdown["preference_head"] > 0
    assert breakdown["current_progress_head"] > 0
    assert breakdown["successor_progress_head"] > 0
    assert breakdown["terminal_head"] > 0
    assert breakdown["consistency_head"] > 0
    assert breakdown["contract_aggregation"] > 0
    assert breakdown["total"] == model.parameter_count
    assert model.complete_system_parameters < proof.TOTAL_PARAMETER_BUDGET


def test_forward_consumes_only_raw_triple_tensors() -> None:
    state = _small_state()
    model = proof.ProofCarryingController(_tiny_config())
    actions = tuple(candidate.action for candidate in state.candidates)
    successors = tuple(candidate.successor for candidate in state.candidates)
    resources = proof.MutableProofResources()
    outputs = proof._model_inputs(
        model,
        state.rows,
        actions,
        successors,
        None,
        resources,
    )
    assert outputs.preference_logits.shape == (len(actions),)
    assert outputs.current_progress_logits.shape == (
        len(actions),
        proof.MAX_PROGRESS_CLASS + 1,
    )
    assert outputs.successor_progress_logits.shape == (
        len(actions),
        proof.MAX_PROGRESS_CLASS + 1,
    )
    assert outputs.terminal_logits.shape == (len(actions),)
    assert outputs.consistency_logits.shape == (len(actions),)
    assert outputs.contract_logits.shape == (len(actions),)
    assert resources.candidate_oracle_calls == 0
    assert resources.candidate_search_calls == 0
    assert resources.candidate_verifier_calls == 0


def test_zeroed_proof_selection_is_exactly_classifier_only() -> None:
    state = _small_state()
    model = proof.ProofCarryingController(_tiny_config())
    actions = tuple(candidate.action for candidate in state.candidates)
    outputs = proof._model_inputs(
        model,
        state.rows,
        actions,
        tuple(candidate.successor for candidate in state.candidates),
        None,
        proof.MutableProofResources(),
    )
    classifier = proof._selection_logits(
        outputs,
        use_contract=False,
        zero_proof=False,
    )
    zeroed = proof._selection_logits(
        outputs,
        use_contract=True,
        zero_proof=True,
    )
    assert torch.equal(classifier, zeroed)


def test_control_specs_are_equal_parameter_architectures() -> None:
    counts = {
        proof.ProofCarryingController(_tiny_config()).parameter_count
        for _ in proof.ARM_SPECS
    }
    assert len(counts) == 1
    assert {spec.name for spec in proof.ARM_SPECS} == set(proof.ARMS)


def test_random_and_shuffled_targets_are_deterministic() -> None:
    state = _small_state()
    shuffled = next(
        spec
        for spec in proof.ARM_SPECS
        if spec.name == proof.ARM_SHUFFLED_PROGRESS
    )
    random_spec = next(
        spec for spec in proof.ARM_SPECS if spec.name == proof.ARM_RANDOM
    )
    shuffled_first = proof._proof_targets(state, spec=shuffled, seed=7)
    shuffled_second = proof._proof_targets(state, spec=shuffled, seed=7)
    assert shuffled_first[0] == shuffled_second[0]
    for left, right in zip(
        shuffled_first[1:],
        shuffled_second[1:],
        strict=True,
    ):
        assert torch.equal(left, right)
    first = proof._proof_targets(state, spec=random_spec, seed=7)
    second = proof._proof_targets(state, spec=random_spec, seed=7)
    assert first[0] == second[0]
    for left, right in zip(first[1:], second[1:], strict=True):
        assert torch.equal(left, right)


def test_preparation_round_trip_and_hashes(tmp_path: Path) -> None:
    artifact = proof.prepare_artifact(
        seed=11,
        train_matrices=2,
        evaluation_matrices=2,
        train_maximum_rows=2,
        train_maximum_columns=2,
        evaluation_minimum_rows=3,
        evaluation_minimum_columns=3,
        evaluation_maximum_rows=3,
        evaluation_maximum_columns=3,
        maximum_preparation_steps=32,
    )
    path = tmp_path / "prep.json"
    path.write_bytes(artifact.canonical_bytes())
    loaded = proof._load_preparation(path)
    assert loaded == artifact
    assert loaded.matched_legal_negative_triples == (
        loaded.legal_triples - len(loaded.states)
    )
    assert sha256(path.read_bytes()).hexdigest()


def test_cpu_training_smoke() -> None:
    torch.manual_seed(5)
    state = _small_state()
    model = proof.ProofCarryingController(_tiny_config())
    spec = proof.ARM_SPECS[0]
    result = proof.train_arm(
        model,
        (state,),
        spec=spec,
        optimizer_updates=2,
        batch_size=1,
        learning_rate=1e-3,
        seed=5,
        amp_bfloat16=False,
        binding_seed=9,
    )
    assert result.optimizer_updates == 2
    assert result.examples_seen == 2
    assert result.legal_triples_seen == 2 * len(state.candidates)


def test_candidate_rollout_source_has_no_oracle_search_or_verifier_call() -> None:
    source = inspect.getsource(proof.autonomous_rollout)
    assert "oracle" not in source
    assert "search" not in source
    assert "verifier" not in source
    assert "assess_rollout_posthoc" not in source


def test_module_does_not_name_protected_checkpoint() -> None:
    source = Path(proof.__file__).read_text(encoding="utf-8")
    assert "ckpt_0300000.pt" not in source
