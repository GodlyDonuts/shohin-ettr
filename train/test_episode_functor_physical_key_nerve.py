from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from episode_functor_conflict_compiler import DirectEvidenceProjector
from episode_functor_constrained_transport import PRIMARY_ANSWERS
from episode_functor_joint_assignment_semantics import (
    joint_semantic_compatibility,
)
from episode_functor_physical_key_nerve import (
    PhysicalKeyNerveError,
    _physical_paths,
    physical_key_nerve,
)
from episode_functor_witness_compiler import (
    ProofCarryingWitnessCompiler,
    RECORD_OBSERVATION,
    RECORD_TRANSITION,
    ROLE_ACTION,
    ROLE_OBSERVATION_STATE,
    ROLE_OBSERVER,
    ROLE_TRANSITION_DESTINATION,
    ROLE_TRANSITION_SOURCE,
    collate_witness_sources,
    scan_witness_source,
)
from pipeline.episode_functor_identifiable_board import (
    GrammarFactors,
    encode_source,
    generate_machine,
    hide_one_cell_per_relation,
)


def _fixture():
    machine = generate_machine(
        seed="physical-key-nerve-test-v1",
        split="mechanics",
        index=0,
        family="affine-f2-3",
    )
    evidence = hide_one_cell_per_relation(
        machine,
        seed="physical-key-nerve-test-v1",
        split="mechanics",
        index=0,
    )
    source = encode_source(evidence, GrammarFactors(0, 0, 0))
    batch = collate_witness_sources((scan_witness_source(source),))
    torch.manual_seed(20260724)
    witness = ProofCarryingWitnessCompiler(
        width=48,
        encoder_layers=1,
        decoder_layers=1,
        heads=3,
        feedforward=96,
        sinkhorn_iterations=16,
        projector=DirectEvidenceProjector(),
    )(batch)
    return batch, witness


def test_nerve_is_record_invariant_and_key_equivariant() -> None:
    batch, witness = _fixture()
    causal = physical_key_nerve(witness, batch.record_valid)
    record_order = torch.arange(
        witness.record_type_logits.shape[1]
    ).roll(7)
    reordered_evidence = replace(
        witness.relation_evidence,
        record_role_unique=(
            witness.relation_evidence.record_role_unique[
                :,
                record_order,
            ]
        ),
    )
    reordered = replace(
        witness,
        relation_evidence=reordered_evidence,
        record_type_logits=witness.record_type_logits[:, record_order],
        answer_logits=witness.answer_logits[:, record_order],
    )
    record_result = physical_key_nerve(
        reordered,
        batch.record_valid[:, record_order],
    )
    assert torch.allclose(
        causal.transition_relation,
        record_result.transition_relation,
        atol=1e-6,
        rtol=0.0,
    )
    generator = torch.Generator().manual_seed(81)
    key_order = torch.randperm(
        witness.unique_key_valid.shape[1],
        generator=generator,
    )
    key_evidence = replace(
        witness.relation_evidence,
        record_role_unique=(
            witness.relation_evidence.record_role_unique[
                ...,
                key_order,
            ]
        ),
    )
    key_recoded = replace(
        witness,
        relation_evidence=key_evidence,
        key_assignment_logits=witness.key_assignment_logits[
            ...,
            key_order,
        ],
        raw_key_assignment_logits=witness.raw_key_assignment_logits[
            ...,
            key_order,
        ],
        unique_key_bytes=witness.unique_key_bytes[:, key_order],
        unique_key_valid=witness.unique_key_valid[:, key_order],
    )
    key_result = physical_key_nerve(
        key_recoded,
        batch.record_valid,
    )
    assert torch.allclose(
        causal.state_signature[:, key_order],
        key_result.state_signature,
        atol=1e-5,
        rtol=0.0,
    )
    assert torch.allclose(
        causal.state_compatibility[:, :, key_order],
        key_result.state_compatibility,
        atol=1e-5,
        rtol=0.0,
    )
    for baseline, recoded in (
        (
            causal.action_left_compatibility,
            key_result.action_left_compatibility,
        ),
        (
            causal.action_right_compatibility,
            key_result.action_right_compatibility,
        ),
        (
            causal.action_observer_compatibility,
            key_result.action_observer_compatibility,
        ),
        (
            causal.action_commutator_compatibility,
            key_result.action_commutator_compatibility,
        ),
    ):
        assert torch.allclose(
            baseline[:, :, key_order],
            recoded,
            atol=1e-5,
            rtol=0.0,
        )


def test_real_nerve_exact_machine_signatures_identify_all_thirteen_keys() -> None:
    batch, witness = _fixture()
    transition_index = torch.tensor(
        (
            (1, 2, 3, 4, 5, 6, 7, 0),
            (0, 2, 1, 4, 3, 6, 5, 7),
            (3, 0, 5, 2, 7, 4, 1, 6),
        )
    )
    observer_index = torch.tensor(
        (
            (0, 1, 2, 3, 1, 2, 3, 0),
            (3, 1, 0, 2, 0, 3, 2, 1),
        )
    )
    role_unique = torch.zeros_like(
        witness.relation_evidence.record_role_unique
    )
    record_type = torch.full_like(
        witness.record_type_logits,
        -100.0,
    )
    answer = torch.full_like(witness.answer_logits, -100.0)
    record_valid = torch.zeros_like(batch.record_valid)
    for action in range(3):
        for state in range(8):
            record = action * 8 + state
            destination = int(transition_index[action, state])
            record_valid[0, record] = True
            record_type[0, record, RECORD_TRANSITION] = 100.0
            role_unique[
                0,
                record,
                ROLE_TRANSITION_SOURCE,
                state,
            ] = 1.0
            role_unique[
                0,
                record,
                ROLE_ACTION,
                8 + action,
            ] = 1.0
            role_unique[
                0,
                record,
                ROLE_TRANSITION_DESTINATION,
                destination,
            ] = 1.0
    for observed in range(2):
        for state in range(8):
            record = 24 + observed * 8 + state
            record_valid[0, record] = True
            record_type[0, record, RECORD_OBSERVATION] = 100.0
            answer[
                0,
                record,
                int(observer_index[observed, state]),
            ] = 100.0
            role_unique[
                0,
                record,
                ROLE_OBSERVATION_STATE,
                state,
            ] = 1.0
            role_unique[
                0,
                record,
                ROLE_OBSERVER,
                11 + observed,
            ] = 1.0

    key_valid = torch.zeros_like(witness.unique_key_valid)
    key_valid[:, :13] = True
    assignment = torch.full_like(
        witness.key_assignment_logits,
        -100.0,
    )
    active_slots = (
        tuple(range(8))
        + tuple(16 + index for index in range(3))
        + tuple(24 + index for index in range(2))
    )
    for key, slot in enumerate(active_slots):
        assignment[:, slot, key] = 100.0
    exact_witness = replace(
        witness,
        relation_evidence=replace(
            witness.relation_evidence,
            record_role_unique=role_unique,
        ),
        key_assignment_logits=assignment,
        raw_key_assignment_logits=assignment,
        record_type_logits=record_type,
        answer_logits=answer,
        unique_key_valid=key_valid,
    )
    nerve = physical_key_nerve(exact_witness, record_valid)
    transition = torch.nn.functional.one_hot(
        transition_index,
        8,
    ).float()[None]
    observer = torch.nn.functional.one_hot(
        observer_index,
        PRIMARY_ANSWERS,
    ).float()[None]
    semantics = joint_semantic_compatibility(
        nerve,
        transition,
        observer,
        key_valid,
    )
    expected = torch.arange(13)
    assert torch.equal(
        semantics.assignment_compatibility.argmax(-1)[0],
        expected,
    )
    diagonal = semantics.assignment_compatibility[
        0,
        expected,
        expected,
    ]
    assert torch.allclose(
        diagonal,
        torch.zeros_like(diagonal),
        atol=1e-7,
        rtol=0.0,
    )


def test_broken_glue_preserves_direct_relations_but_changes_paths() -> None:
    batch, witness = _fixture()
    causal = physical_key_nerve(witness, batch.record_valid)
    broken = physical_key_nerve(
        witness,
        batch.record_valid,
        mode="broken-glue",
    )
    assert torch.equal(
        causal.transition_relation,
        broken.transition_relation,
    )
    assert torch.equal(
        causal.observation_relation,
        broken.observation_relation,
    )
    assert not torch.equal(
        causal.state_signature,
        broken.state_signature,
    )
    assert not torch.equal(
        causal.action_left_compatibility,
        broken.action_left_compatibility,
    )
    assert not torch.equal(
        causal.action_commutator_compatibility,
        broken.action_commutator_compatibility,
    )
    assert torch.allclose(
        causal.path_mass,
        broken.path_mass,
        atol=1e-5,
        rtol=0.0,
    )
    assert bool(causal.path_mass.gt(0).all())

    generator = torch.Generator().manual_seed(113)
    order = torch.randperm(
        witness.unique_key_valid.shape[1],
        generator=generator,
    )
    recoded_evidence = replace(
        witness.relation_evidence,
        record_role_unique=(
            witness.relation_evidence.record_role_unique[..., order]
        ),
    )
    recoded = replace(
        witness,
        relation_evidence=recoded_evidence,
        key_assignment_logits=witness.key_assignment_logits[..., order],
        raw_key_assignment_logits=(
            witness.raw_key_assignment_logits[..., order]
        ),
        unique_key_bytes=witness.unique_key_bytes[:, order],
        unique_key_valid=witness.unique_key_valid[:, order],
    )
    recoded_broken = physical_key_nerve(
        recoded,
        batch.record_valid,
        mode="broken-glue",
    )
    assert torch.allclose(
        broken.action_left_compatibility[:, :, order],
        recoded_broken.action_left_compatibility,
        atol=1e-5,
        rtol=0.0,
    )
    assert torch.allclose(
        broken.path_mass,
        recoded_broken.path_mass,
        atol=1e-5,
        rtol=0.0,
    )
    causal_two_step, causal_to = _physical_paths(
        causal.transition_relation,
        causal.observation_relation,
        witness.unique_key_valid,
        broken_glue=False,
    )
    broken_two_step, broken_to = _physical_paths(
        causal.transition_relation,
        causal.observation_relation,
        witness.unique_key_valid,
        broken_glue=True,
    )
    for baseline, treatment, dimensions in (
        (causal_two_step, broken_two_step, (3, 4)),
        (causal_two_step, broken_two_step, (1, 2)),
        (causal_to, broken_to, (3, 4)),
        (causal_to, broken_to, (1, 2)),
    ):
        assert torch.allclose(
            baseline.sum(dimensions),
            treatment.sum(dimensions),
            atol=3e-5,
            rtol=3e-5,
        )


def test_broken_glue_rejects_singleton_physical_support() -> None:
    transition = torch.zeros(1, 3, 3, 3)
    observation = torch.zeros(1, 3, 3, PRIMARY_ANSWERS)
    transition[0, 0, 0, 0] = 1.0
    observation[0, 0, 0, 0] = 1.0
    key_valid = torch.tensor(((True, False, False),))
    with pytest.raises(
        PhysicalKeyNerveError,
        match="mass-preserving off-diagonal coupling",
    ):
        _physical_paths(
            transition,
            observation,
            key_valid,
            broken_glue=True,
        )


def test_zero_record_mass_has_bit_exact_zero_path_state() -> None:
    batch, witness = _fixture()
    empty = physical_key_nerve(
        witness,
        torch.zeros_like(batch.record_valid),
    )
    assert torch.equal(
        empty.transition_relation,
        torch.zeros_like(empty.transition_relation),
    )
    assert torch.equal(
        empty.observation_relation,
        torch.zeros_like(empty.observation_relation),
    )
    assert torch.equal(
        empty.state_signature,
        torch.zeros_like(empty.state_signature),
    )
    assert torch.equal(
        empty.path_mass,
        torch.zeros_like(empty.path_mass),
    )
    assert torch.equal(
        empty.action_left_compatibility,
        torch.zeros_like(empty.action_left_compatibility),
    )
    assert torch.equal(
        empty.action_commutator_compatibility,
        torch.zeros_like(empty.action_commutator_compatibility),
    )


def test_ordered_noncommuting_channels_do_not_collapse() -> None:
    batch, witness = _fixture()
    causal = physical_key_nerve(witness, batch.record_valid)
    assert not torch.equal(
        causal.action_left_compatibility,
        causal.action_right_compatibility,
    )
    assert bool(
        causal.action_commutator_compatibility.abs().sum().gt(0)
    )
    one_step = physical_key_nerve(
        witness,
        batch.record_valid,
        mode="one-step-only",
    )
    _, transition_observer = _physical_paths(
        causal.transition_relation,
        causal.observation_relation,
        witness.unique_key_valid,
        broken_glue=False,
    )
    assert torch.allclose(
        one_step.path_mass,
        transition_observer.sum((1, 2, 3, 4)),
        atol=1e-6,
        rtol=0.0,
    )
    assert bool(one_step.path_mass.lt(causal.path_mass).all())
    for value in (
        one_step.action_left_compatibility,
        one_step.action_right_compatibility,
        one_step.action_commutator_compatibility,
    ):
        assert torch.equal(value, torch.zeros_like(value))
    assert not torch.equal(
        one_step.action_observer_compatibility,
        torch.zeros_like(one_step.action_observer_compatibility),
    )


def test_explicit_assignment_transport_controls_path_signatures() -> None:
    batch, witness = _fixture()
    baseline = physical_key_nerve(witness, batch.record_valid)
    active = (
        tuple(range(8))
        + tuple(16 + index for index in range(3))
        + tuple(24 + index for index in range(2))
    )
    revised = witness.key_assignment_logits.clone()
    revised[:, active[0]], revised[:, active[1]] = (
        revised[:, active[1]].clone(),
        revised[:, active[0]].clone(),
    )
    explicit = physical_key_nerve(
        witness,
        batch.record_valid,
        key_assignment_logits=revised,
    )
    assert not torch.equal(
        baseline.state_signature,
        explicit.state_signature,
    )
    assert torch.equal(
        baseline.transition_relation,
        explicit.transition_relation,
    )
    with pytest.raises(
        PhysicalKeyNerveError,
        match="assignment transport",
    ):
        physical_key_nerve(
            witness,
            batch.record_valid,
            key_assignment_logits=revised[:, :-1],
        )
