from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from episode_functor_conflict_compiler import (
    ConflictClaimAdapter,
    ConflictProofCarryingCompiler,
    ConflictSourceReentry,
    DirectEvidenceProjector,
    ConflictCompilerError,
    _hard_assign_keys_without_solver,
    record_features_from_witness,
)
from episode_functor_conflict_reentrant_revision import (
    ConflictGatedReentrantRevision,
    ConflictRevisionBatch,
    MACHINE_CATEGORIES,
    MACHINE_ROWS,
)
from episode_functor_constrained_transport import (
    PRIMARY_ACTIONS,
    PRIMARY_ANSWERS,
    PRIMARY_OBSERVERS,
    PRIMARY_STATES,
)
from episode_functor_witness_compiler import (
    MAX_RECORDS,
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


def _source() -> bytes:
    machine = generate_machine(
        seed="cgrfc-compiler-test-v1",
        split="mechanics",
        index=0,
        family="affine-f2-3",
    )
    evidence = hide_one_cell_per_relation(
        machine,
        seed="cgrfc-compiler-test-v1",
        split="mechanics",
        index=0,
    )
    return encode_source(evidence, GrammarFactors(0, 0, 0))


def test_direct_evidence_projector_has_no_hidden_completion_parameters() -> None:
    projector = DirectEvidenceProjector()
    generator = torch.Generator().manual_seed(20260724)
    transition = torch.randn(
        2,
        PRIMARY_ACTIONS,
        PRIMARY_STATES,
        PRIMARY_STATES,
        generator=generator,
    )
    observer = torch.randn(
        2,
        PRIMARY_OBSERVERS,
        PRIMARY_STATES,
        PRIMARY_ANSWERS,
        generator=generator,
    )
    projection = projector(transition, observer)
    assert projector.parameter_count() == 0
    assert torch.allclose(
        projection.transition_transport,
        transition.softmax(-1),
    )
    assert torch.allclose(
        projection.observer_transport,
        observer.softmax(-1),
    )


def test_raw_witness_states_form_invariant_direct_revision_claims() -> None:
    torch.manual_seed(20260724)
    batch = collate_witness_sources(
        (scan_witness_source(_source()),)
    )
    witness_compiler = ProofCarryingWitnessCompiler(
        width=48,
        encoder_layers=1,
        decoder_layers=1,
        heads=3,
        feedforward=96,
        sinkhorn_iterations=16,
        projector=DirectEvidenceProjector(),
    )
    witness = witness_compiler(batch)
    record_features = record_features_from_witness(witness)
    assert record_features.shape == (1, MAX_RECORDS, 32)
    adapter = ConflictClaimAdapter(record_width=32, hidden_width=128)
    (
        claim_logits,
        closure_claim_logits,
        claim_incidence,
        closure_incidence,
    ) = adapter(
        witness,
        record_features=record_features,
        record_valid=batch.record_valid,
    )
    assert claim_logits.shape == (
        1,
        MAX_RECORDS,
        MACHINE_ROWS,
        MACHINE_CATEGORIES,
    )
    assert closure_claim_logits.shape == claim_logits.shape
    assert claim_incidence.shape == (1, MAX_RECORDS, MACHINE_ROWS)
    assert closure_incidence.shape == claim_incidence.shape
    assert torch.isfinite(claim_logits).all()
    assert torch.isfinite(closure_claim_logits).all()
    assert (claim_incidence >= 0).all()
    assert (closure_incidence >= 0).all()
    transition_incidence = claim_incidence[
        :,
        :,
        : PRIMARY_ACTIONS * PRIMARY_STATES,
    ].reshape(1, MAX_RECORDS, PRIMARY_ACTIONS, PRIMARY_STATES)
    observer_incidence = claim_incidence[
        :,
        :,
        PRIMARY_ACTIONS * PRIMARY_STATES :,
    ].reshape(1, MAX_RECORDS, PRIMARY_OBSERVERS, PRIMARY_STATES)
    assert not torch.equal(
        transition_incidence[:, :, 0],
        transition_incidence[:, :, 1],
    )
    assert not torch.equal(
        transition_incidence[:, :, 1],
        transition_incidence[:, :, 2],
    )
    assert not torch.equal(
        observer_incidence[:, :, 0],
        observer_incidence[:, :, 1],
    )
    assert torch.equal(
        closure_incidence,
        torch.zeros_like(closure_incidence),
    )
    assert torch.equal(
        closure_claim_logits,
        torch.zeros_like(closure_claim_logits),
    )
    claim_probabilities = claim_logits.softmax(-1)
    claim_weights = claim_incidence[..., None]
    claim_mean = (
        claim_weights * claim_probabilities
    ).sum(1) / claim_weights.sum(1).clamp_min(
        torch.finfo(claim_weights.dtype).tiny
    )
    provisional = torch.cat(
        (
                witness.relation_evidence.transition_logits.float()
                .softmax(-1)
                .flatten(1, 2),
                torch.nn.functional.pad(
                    witness.relation_evidence.observer_logits.float()
                    .softmax(-1)
                    .flatten(1, 2),
                    (0, MACHINE_CATEGORIES - PRIMARY_ANSWERS),
                ),
        ),
        dim=1,
    )
    present = claim_incidence.sum(1).gt(0)[..., None]
    assert bool(
        (claim_mean - provisional).abs().masked_fill(
            ~present,
            0.0,
        ).amax().gt(1e-6)
    )
    identity_claims, _, identity_incidence, _ = adapter(
        witness,
        record_features=record_features,
        record_valid=batch.record_valid,
        mode="identity",
    )
    identity_weights = identity_incidence[..., None]
    identity_mean = (
        identity_weights * identity_claims.softmax(-1)
    ).sum(1) / identity_weights.sum(1).clamp_min(
        torch.finfo(identity_weights.dtype).tiny
    )
    assert torch.allclose(
        identity_mean.masked_fill(~present, 0.0),
        provisional.masked_fill(~present, 0.0),
        atol=1e-6,
        rtol=0.0,
    )
    revision_batch = ConflictRevisionBatch(
        transition_logits=witness.relation_evidence.transition_logits,
        observer_logits=witness.relation_evidence.observer_logits,
        claim_logits=claim_logits,
        closure_claim_logits=closure_claim_logits,
        claim_incidence=claim_incidence,
        closure_incidence=closure_incidence,
        record_features=record_features,
        record_valid=batch.record_valid,
    )
    revision = ConflictGatedReentrantRevision(
        record_width=32,
        controller_width=128,
        cycles=2,
    )
    result = revision(revision_batch)
    loss = (
        result.projection.transition_transport.square().mean()
        + result.projection.observer_transport.square().mean()
        + result.contradiction_energy.mean()
    )
    loss.backward()
    assert adapter.parameter_count() > 20_000
    assert all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        for parameter in adapter.parameters()
    )
    assert all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        for parameter in revision.parameters()
    )


def test_record_diagnostics_are_coordinate_recoding_invariant() -> None:
    torch.manual_seed(20260724)
    batch = collate_witness_sources(
        (scan_witness_source(_source()),)
    )
    witness = ProofCarryingWitnessCompiler(
        width=48,
        encoder_layers=1,
        decoder_layers=1,
        heads=3,
        feedforward=96,
        sinkhorn_iterations=16,
        projector=DirectEvidenceProjector(),
    )(batch)
    generator = torch.Generator().manual_seed(77)
    key_order = torch.randperm(
        witness.relation_evidence.record_role_unique.shape[-1],
        generator=generator,
    )
    state_order = torch.randperm(16, generator=generator)
    action_order = torch.randperm(8, generator=generator) + 16
    observer_order = torch.randperm(8, generator=generator) + 24
    slot_order = torch.cat((state_order, action_order, observer_order))
    answer_order = torch.randperm(4, generator=generator)
    evidence = replace(
        witness.relation_evidence,
        record_role_unique=(
            witness.relation_evidence.record_role_unique[
                ...,
                key_order,
            ]
        ),
        record_role_slot=(
            witness.relation_evidence.record_role_slot[
                ...,
                slot_order,
            ]
        ),
    )
    recoded = replace(
        witness,
        relation_evidence=evidence,
        answer_logits=witness.answer_logits[..., answer_order],
    )
    assert torch.allclose(
        record_features_from_witness(witness),
        record_features_from_witness(recoded),
        atol=1e-6,
        rtol=0.0,
    )


def test_record_local_claim_calibration_is_fully_recoding_equivariant() -> None:
    torch.manual_seed(20260724)
    batch = collate_witness_sources(
        (scan_witness_source(_source()),)
    )
    witness = ProofCarryingWitnessCompiler(
        width=48,
        encoder_layers=1,
        decoder_layers=1,
        heads=3,
        feedforward=96,
        sinkhorn_iterations=16,
        projector=DirectEvidenceProjector(),
    )(batch)
    adapter = ConflictClaimAdapter(record_width=32, hidden_width=128)
    features = record_features_from_witness(witness)
    claims, _, incidence, _ = adapter(
        witness,
        record_features=features,
        record_valid=batch.record_valid,
    )
    action_order = torch.tensor((2, 0, 1))
    state_order = torch.tensor((5, 2, 7, 0, 4, 1, 6, 3))
    observer_order = torch.tensor((1, 0))
    answer_order = torch.tensor((2, 0, 3, 1))
    slot_order = torch.cat(
        (
            state_order,
            torch.arange(PRIMARY_STATES, 16),
            16 + action_order,
            torch.arange(16 + PRIMARY_ACTIONS, 24),
            24 + observer_order,
            torch.arange(24 + PRIMARY_OBSERVERS, 32),
        )
    )
    recoded_evidence = replace(
        witness.relation_evidence,
        record_role_slot=(
            witness.relation_evidence.record_role_slot[..., slot_order]
        ),
    )
    recoded_witness = replace(
        witness,
        relation_evidence=recoded_evidence,
        answer_logits=witness.answer_logits[..., answer_order],
    )
    recoded_features = record_features_from_witness(recoded_witness)
    recoded_claims, _, recoded_incidence, _ = adapter(
        recoded_witness,
        record_features=recoded_features,
        record_valid=batch.record_valid,
    )
    transition_claims = claims[
        :,
        :,
        : PRIMARY_ACTIONS * PRIMARY_STATES,
    ].reshape(
        1,
        MAX_RECORDS,
        PRIMARY_ACTIONS,
        PRIMARY_STATES,
        MACHINE_CATEGORIES,
    )
    observer_claims = claims[
        :,
        :,
        PRIMARY_ACTIONS * PRIMARY_STATES :,
    ].reshape(
        1,
        MAX_RECORDS,
        PRIMARY_OBSERVERS,
        PRIMARY_STATES,
        MACHINE_CATEGORIES,
    )
    expected_transition_claims = transition_claims.index_select(
        2,
        action_order,
    ).index_select(3, state_order).index_select(4, state_order)
    expected_observer_claims = observer_claims.index_select(
        2,
        observer_order,
    ).index_select(3, state_order)
    expected_observer_claims = torch.cat(
        (
            expected_observer_claims[
                ...,
                :PRIMARY_ANSWERS,
            ].index_select(4, answer_order),
            expected_observer_claims[..., PRIMARY_ANSWERS:],
        ),
        dim=-1,
    )
    expected_claims = torch.cat(
        (
            expected_transition_claims.reshape(
                1,
                MAX_RECORDS,
                -1,
                MACHINE_CATEGORIES,
            ),
            expected_observer_claims.reshape(
                1,
                MAX_RECORDS,
                -1,
                MACHINE_CATEGORIES,
            ),
        ),
        dim=2,
    )
    transition_incidence = incidence[
        :,
        :,
        : PRIMARY_ACTIONS * PRIMARY_STATES,
    ].reshape(
        1,
        MAX_RECORDS,
        PRIMARY_ACTIONS,
        PRIMARY_STATES,
    )
    observer_incidence = incidence[
        :,
        :,
        PRIMARY_ACTIONS * PRIMARY_STATES :,
    ].reshape(
        1,
        MAX_RECORDS,
        PRIMARY_OBSERVERS,
        PRIMARY_STATES,
    )
    expected_incidence = torch.cat(
        (
            transition_incidence.index_select(
                2,
                action_order,
            ).index_select(3, state_order).reshape(
                1,
                MAX_RECORDS,
                -1,
            ),
            observer_incidence.index_select(
                2,
                observer_order,
            ).index_select(3, state_order).reshape(
                1,
                MAX_RECORDS,
                -1,
            ),
        ),
        dim=2,
    )
    assert torch.allclose(
        expected_claims,
        recoded_claims,
        atol=1e-5,
        rtol=0.0,
    )
    assert torch.allclose(
        expected_incidence,
        recoded_incidence,
        atol=1e-6,
        rtol=0.0,
    )


def test_solver_free_key_seal_requires_unique_untied_argmax() -> None:
    torch.manual_seed(20260724)
    batch = collate_witness_sources(
        (scan_witness_source(_source()),)
    )
    witness = ProofCarryingWitnessCompiler(
        width=48,
        encoder_layers=1,
        decoder_layers=1,
        heads=3,
        feedforward=96,
        sinkhorn_iterations=16,
        projector=DirectEvidenceProjector(),
    )(batch)
    logits = torch.full_like(witness.key_assignment_logits, -20.0)
    active_slots = (
        tuple(range(8))
        + tuple(16 + index for index in range(3))
        + tuple(24 + index for index in range(2))
    )
    unique = witness.unique_key_valid[0].nonzero().flatten()
    assert int(unique.numel()) == len(active_slots)
    for slot, key in zip(active_slots, unique.tolist(), strict=True):
        logits[0, slot, key] = 20.0
    sealed = _hard_assign_keys_without_solver(
        replace(witness, key_assignment_logits=logits)
    )
    assert sealed.state_keys.shape == (1, 16, 8)
    tied = logits.clone()
    tied[0, active_slots[0], unique[0]] = 0.0
    tied[0, active_slots[0], unique[1]] = 0.0
    with pytest.raises(
        ConflictCompilerError,
        match="categorical tie",
    ):
        _hard_assign_keys_without_solver(
            replace(witness, key_assignment_logits=tied)
        )
    nonfinite = logits.clone()
    nonfinite[0, active_slots[0], unique[0]] = float("nan")
    with pytest.raises(
        ConflictCompilerError,
        match="geometry differs",
    ):
        _hard_assign_keys_without_solver(
            replace(witness, key_assignment_logits=nonfinite)
        )


def test_conflict_reentry_routes_only_to_source_spans_and_has_controls() -> None:
    torch.manual_seed(20260724)
    batch = collate_witness_sources(
        (scan_witness_source(_source()),)
    )
    witness_compiler = ProofCarryingWitnessCompiler(
        width=48,
        encoder_layers=1,
        decoder_layers=1,
        heads=3,
        feedforward=96,
        sinkhorn_iterations=16,
        projector=DirectEvidenceProjector(),
    )
    witness = witness_compiler(batch)
    record_features = record_features_from_witness(witness)
    adapter = ConflictClaimAdapter(record_width=32, hidden_width=128)
    claims, closure, incidence, closure_incidence = adapter(
        witness,
        record_features=record_features,
        record_valid=batch.record_valid,
    )
    revision_batch = ConflictRevisionBatch(
        transition_logits=witness.relation_evidence.transition_logits,
        observer_logits=witness.relation_evidence.observer_logits,
        claim_logits=claims,
        closure_claim_logits=closure,
        claim_incidence=incidence,
        closure_incidence=closure_incidence,
        record_features=record_features,
        record_valid=batch.record_valid,
    )
    revision = ConflictGatedReentrantRevision(
        record_width=32,
        controller_width=128,
        cycles=2,
    )(revision_batch)
    reentry = ConflictSourceReentry(
        record_width=32,
        context_width=128,
        bottleneck_width=32,
        external_feature_width=96,
    )
    byte_count = int(batch.pointer.byte_ids.shape[1])
    causal = reentry(
        batch,
        revision_batch,
        revision,
        byte_count=byte_count,
        mode="causal",
    )
    deranged = reentry(
        batch,
        revision_batch,
        revision,
        byte_count=byte_count,
        mode="deranged",
    )
    open_loop = reentry(
        batch,
        revision_batch,
        revision,
        byte_count=byte_count,
        mode="open-loop",
    )
    no_correction = replace(
        revision,
        cycle_transition_logits=tuple(
            revision_batch.transition_logits
            for _ in revision.cycle_transition_logits
        ),
        cycle_observer_logits=tuple(
            revision_batch.observer_logits
            for _ in revision.cycle_observer_logits
        ),
    )
    zero_correction = reentry(
        batch,
        revision_batch,
        no_correction,
        byte_count=byte_count,
        mode="causal",
    )
    assert causal.shape == (1, byte_count, 96)
    assert torch.isfinite(causal).all()
    assert bool(causal.abs().sum().gt(0))
    assert not torch.equal(causal, deranged)
    assert torch.equal(open_loop, torch.zeros_like(open_loop))
    assert torch.equal(
        zero_correction,
        torch.zeros_like(zero_correction),
    )
    shifted_batch = replace(
        revision_batch,
        transition_logits=revision_batch.transition_logits + 11.0,
        observer_logits=revision_batch.observer_logits - 7.0,
    )
    shifted_no_correction = replace(
        revision,
        cycle_transition_logits=tuple(
            shifted_batch.transition_logits
            - shifted_batch.transition_logits.mean(-1, keepdim=True)
            for _ in revision.cycle_transition_logits
        ),
        cycle_observer_logits=tuple(
            shifted_batch.observer_logits
            - shifted_batch.observer_logits.mean(-1, keepdim=True)
            for _ in revision.cycle_observer_logits
        ),
    )
    gauge_zero = reentry(
        batch,
        shifted_batch,
        shifted_no_correction,
        byte_count=byte_count,
        mode="causal",
    )
    assert torch.equal(gauge_zero, torch.zeros_like(gauge_zero))
    covered = torch.zeros(byte_count, dtype=torch.bool)
    for record in range(int(batch.record_valid[0].sum())):
        start, end = (
            int(value)
            for value in batch.record_bounds[0, record].tolist()
        )
        covered[start:end] = True
    assert torch.equal(
        causal[0, ~covered],
        torch.zeros_like(causal[0, ~covered]),
    )
    singleton_incidence = torch.zeros_like(
        revision.cycle_claim_incidence[-1]
    )
    positive_record = int(
        revision.cycle_claim_incidence[-1]
        .sum(-1)
        .gt(0)
        .nonzero()[0, 1]
    )
    singleton_incidence[:, positive_record] = (
        revision.cycle_claim_incidence[-1][:, positive_record]
    )
    singleton_revision = replace(
        revision,
        cycle_claim_incidence=tuple(
            singleton_incidence
            for _ in revision.cycle_claim_incidence
        ),
        cycle_closure_incidence=tuple(
            torch.zeros_like(singleton_incidence)
            for _ in revision.cycle_closure_incidence
        ),
    )
    with pytest.raises(
        ConflictCompilerError,
        match="requires two positive records",
    ):
        reentry(
            batch,
            revision_batch,
            singleton_revision,
            byte_count=byte_count,
            mode="deranged",
        )
    causal.square().mean().backward(retain_graph=True)
    assert all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        for parameter in reentry.parameters()
    )
    destination_order = torch.tensor([5, 2, 7, 0, 4, 1, 6, 3])
    answer_order = torch.tensor([2, 0, 3, 1])
    recoded_batch = replace(
        revision_batch,
        transition_logits=revision_batch.transition_logits[
            ...,
            destination_order,
        ],
        observer_logits=revision_batch.observer_logits[..., answer_order],
    )
    recoded_revision = replace(
        revision,
        cycle_transition_logits=tuple(
            value[..., destination_order]
            for value in revision.cycle_transition_logits
        ),
        cycle_observer_logits=tuple(
            value[..., answer_order]
            for value in revision.cycle_observer_logits
        ),
    )
    for mode in ("causal", "deranged", "sign-scrambled"):
        original_feedback = reentry(
            batch,
            revision_batch,
            revision,
            byte_count=byte_count,
            mode=mode,
        )
        recoded_feedback = reentry(
            batch,
            recoded_batch,
            recoded_revision,
            byte_count=byte_count,
            mode=mode,
        )
        assert torch.allclose(
            original_feedback,
            recoded_feedback,
            atol=1e-6,
            rtol=0.0,
        )
    reentry.zero_grad(set_to_none=True)
    open_loop_connected = reentry(
        batch,
        revision_batch,
        revision,
        byte_count=byte_count,
        mode="open-loop",
    )
    open_loop_connected.square().mean().backward()
    assert all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        for parameter in reentry.parameters()
    )


def test_two_pass_reuses_witness_and_detaches_frozen_trunk_features() -> None:
    torch.manual_seed(20260724)
    batch = collate_witness_sources(
        (scan_witness_source(_source()),)
    )
    compiler = ConflictProofCarryingCompiler(
        external_feature_width=96,
        width=48,
        encoder_layers=1,
        decoder_layers=1,
        heads=3,
        feedforward=96,
        controller_width=128,
        cycles=2,
    )
    calls = []
    handle = compiler.witness.register_forward_hook(
        lambda *_: calls.append(1)
    )
    frozen = torch.randn(
        1,
        batch.pointer.byte_ids.shape[1],
        96,
        requires_grad=True,
    )
    try:
        with torch.autocast(
            device_type="cpu",
            dtype=torch.bfloat16,
        ):
            output = compiler(
                batch,
                frozen_byte_features=frozen,
                source_reentry_mode="causal",
            )
            loss = (
                output.revision.projection.transition_transport.square().mean()
                + output.revision.projection.observer_transport.square().mean()
                + output.source_feedback.square().mean()
            )
        loss.backward()
    finally:
        handle.remove()
    assert len(calls) == 2
    assert frozen.grad is None
    assert bool(output.source_feedback.abs().sum().gt(0))
    assert torch.isfinite(output.source_feedback).all()
    assert all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        for parameter in compiler.source_reentry.parameters()
    )
    open_loop = compiler(
        batch,
        frozen_byte_features=frozen.detach(),
        source_reentry_mode="open-loop",
    )
    assert torch.equal(
        open_loop.source_feedback,
        torch.zeros_like(open_loop.source_feedback),
    )
    assert torch.equal(
        open_loop.first_witness.record_type_logits,
        open_loop.witness.record_type_logits,
    )
    with pytest.raises(
        ConflictCompilerError,
        match="forbids solver-backed straight-through",
    ):
        compiler(
            batch,
            frozen_byte_features=frozen.detach(),
            straight_through=True,
        )


def test_maximum_integrated_parameter_receipt_stays_below_200m() -> None:
    compiler = ConflictProofCarryingCompiler()
    protected = 125_081_664
    query = 728_993
    complete = protected + compiler.parameter_count() + query
    assert compiler.witness.projector.parameter_count() == 0
    assert compiler.claim_adapter.parameter_count() == 5_386_721
    assert compiler.path_controller.parameter_count() == 3_978_602
    assert compiler.source_reentry.parameter_count() == 352_641
    assert compiler.revision.parameter_count() == 19_658_466
    assert compiler.parameter_count() == 74_067_262
    assert complete == 199_877_919
    assert 200_000_000 - complete == 122_081
