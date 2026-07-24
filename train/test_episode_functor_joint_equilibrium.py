from __future__ import annotations

from dataclasses import replace

import torch

from episode_functor_conflict_compiler import (
    DirectEvidenceProjector,
    record_features_from_witness,
)
from episode_functor_joint_equilibrium import (
    ACTIVE_SLOTS,
    JointAssignmentSemanticsEquilibrium,
)
from episode_functor_constrained_transport import (
    PRIMARY_ACTIONS,
    PRIMARY_STATES,
)
from episode_functor_witness_compiler import (
    ProofCarryingWitnessCompiler,
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
        seed="joint-equilibrium-test-v1",
        split="mechanics",
        index=0,
        family="affine-f2-3",
    )
    evidence = hide_one_cell_per_relation(
        machine,
        seed="joint-equilibrium-test-v1",
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


def _small() -> JointAssignmentSemanticsEquilibrium:
    return JointAssignmentSemanticsEquilibrium(
        assignment_width=48,
        assignment_context_width=96,
        machine_width=64,
        machine_context_width=128,
        cycles=2,
        sinkhorn_iterations=16,
    )


def test_joint_equilibrium_forward_backward_and_invariants() -> None:
    batch, witness = _fixture()
    model = _small()
    result = model(
        batch,
        witness,
        record_features=record_features_from_witness(witness),
    )
    assert len(result.cycle_key_assignment_logits) == 2
    active = torch.tensor(ACTIVE_SLOTS)
    valid = witness.unique_key_valid[0]
    transport = result.key_assignment_logits[
        0,
    ].index_select(0, active)[:, valid].exp()
    assert torch.allclose(
        transport.sum(-1),
        torch.ones(13),
        atol=2e-4,
        rtol=0.0,
    )
    assert torch.allclose(
        transport.sum(0),
        torch.ones(13),
        atol=2e-4,
        rtol=0.0,
    )
    assert torch.allclose(
        result.transition_probabilities.sum(-1),
        torch.ones_like(result.transition_probabilities[..., 0]),
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        result.observer_probabilities.sum(-1),
        torch.ones_like(result.observer_probabilities[..., 0]),
        atol=1e-6,
        rtol=0.0,
    )
    loss = (
        result.transition_probabilities.square().sum()
        + result.observer_probabilities.square().sum()
        + result.key_assignment_logits[
            :,
            active,
        ].exp().square().sum()
    )
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)


def test_open_loop_executes_graph_connected_zero_updates() -> None:
    batch, witness = _fixture()
    model = _small()
    result = model(
        batch,
        witness,
        record_features=record_features_from_witness(witness),
        mode="open-loop",
    )
    for correction in result.cycle_assignment_correction:
        assert torch.equal(correction, torch.zeros_like(correction))
    for direction in result.cycle_machine_direction:
        assert torch.equal(direction, torch.zeros_like(direction))


def test_directed_cuts_are_distinct_and_finite() -> None:
    batch, witness = _fixture()
    model = _small()
    features = record_features_from_witness(witness)
    causal = model(batch, witness, record_features=features)
    m_cut = model(
        batch,
        witness,
        record_features=features,
        mode="machine-to-assignment-cut",
    )
    assert not torch.equal(
        causal.raw_key_assignment_logits,
        m_cut.raw_key_assignment_logits,
    )
    causal_loss = (
        causal.transition_probabilities.square().sum()
        + causal.observer_probabilities.square().sum()
    )
    causal_gradient = torch.autograd.grad(
        causal_loss,
        model.assignment_direction.weight,
        retain_graph=False,
    )[0]
    assert bool(causal_gradient.abs().sum().gt(0))

    model.zero_grad(set_to_none=True)
    a_cut = model(
        batch,
        witness,
        record_features=features,
        mode="assignment-to-machine-cut",
    )
    cut_loss = (
        a_cut.transition_probabilities.square().sum()
        + a_cut.observer_probabilities.square().sum()
    )
    cut_gradient = torch.autograd.grad(
        cut_loss,
        model.assignment_direction.weight,
        retain_graph=False,
    )[0]
    assert torch.equal(cut_gradient, torch.zeros_like(cut_gradient))


def test_complete_semantic_gauge_relabeling_commutes_through_all_cycles() -> None:
    batch, witness = _fixture()
    model = _small()
    baseline = model(
        batch,
        witness,
        record_features=record_features_from_witness(witness),
    )
    state_order = torch.tensor((3, 0, 7, 2, 5, 1, 6, 4))
    action_order = torch.tensor((2, 0, 1))
    observer_order = torch.tensor((1, 0))
    answer_order = torch.tensor((2, 0, 3, 1))
    role_order = torch.cat(
        (
            state_order,
            PRIMARY_STATES + action_order,
            PRIMARY_STATES
            + PRIMARY_ACTIONS
            + observer_order,
        )
    )
    active = torch.tensor(ACTIVE_SLOTS)
    full_order = torch.arange(
        witness.raw_key_assignment_logits.shape[1]
    )
    full_order[active] = active[role_order]

    transition = witness.projection.transition_transport[
        :,
        action_order,
    ][:, :, state_order][:, :, :, state_order]
    observer = witness.projection.observer_transport[
        :,
        observer_order,
    ][:, :, state_order][:, :, :, answer_order]
    projection = DirectEvidenceProjector()(
        transition.clamp_min(torch.finfo(transition.dtype).tiny).log(),
        observer.clamp_min(torch.finfo(observer.dtype).tiny).log(),
    )
    evidence = replace(
        witness.relation_evidence,
        transition_logits=witness.relation_evidence.transition_logits[
            :,
            action_order,
        ][:, :, state_order][:, :, :, state_order],
        observer_logits=witness.relation_evidence.observer_logits[
            :,
            observer_order,
        ][:, :, state_order][:, :, :, answer_order],
        record_role_slot=(
            witness.relation_evidence.record_role_slot[
                ...,
                full_order,
            ]
        ),
    )
    recoded_witness = replace(
        witness,
        projection=projection,
        relation_evidence=evidence,
        key_assignment_logits=witness.key_assignment_logits[
            :,
            full_order,
        ],
        raw_key_assignment_logits=witness.raw_key_assignment_logits[
            :,
            full_order,
        ],
        answer_logits=witness.answer_logits[..., answer_order],
    )
    recoded = model(
        batch,
        recoded_witness,
        record_features=record_features_from_witness(recoded_witness),
    )

    def recode_transition(value: torch.Tensor) -> torch.Tensor:
        return value[
            :,
            action_order,
        ][:, :, state_order][:, :, :, state_order]

    def recode_observer(value: torch.Tensor) -> torch.Tensor:
        return value[
            :,
            observer_order,
        ][:, :, state_order][:, :, :, answer_order]

    assert torch.allclose(
        recoded.key_assignment_logits,
        baseline.key_assignment_logits[:, full_order],
        atol=2e-5,
        rtol=0.0,
    )
    assert torch.allclose(
        recoded.transition_probabilities,
        recode_transition(baseline.transition_probabilities),
        atol=2e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        recoded.observer_probabilities,
        recode_observer(baseline.observer_probabilities),
        atol=2e-6,
        rtol=0.0,
    )
    for baseline_cycle, recoded_cycle in zip(
        baseline.cycle_transition_probabilities,
        recoded.cycle_transition_probabilities,
        strict=True,
    ):
        assert torch.allclose(
            recoded_cycle,
            recode_transition(baseline_cycle),
            atol=2e-6,
            rtol=0.0,
        )
    for baseline_cycle, recoded_cycle in zip(
        baseline.cycle_observer_probabilities,
        recoded.cycle_observer_probabilities,
        strict=True,
    ):
        assert torch.allclose(
            recoded_cycle,
            recode_observer(baseline_cycle),
            atol=2e-6,
            rtol=0.0,
        )


def test_default_controller_replaces_sequential_budget() -> None:
    model = JointAssignmentSemanticsEquilibrium()
    assert model.parameter_count() < 23_637_068
    assert model.parameter_count() > 15_000_000
