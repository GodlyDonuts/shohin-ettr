from __future__ import annotations

import pytest
import torch

from episode_functor_constrained_transport import (
    PRIMARY_ACTIONS,
    PRIMARY_ANSWERS,
    PRIMARY_OBSERVERS,
    PRIMARY_STATES,
)
from episode_functor_conflict_reentrant_revision import (
    ConflictGatedReentrantRevision,
    ConflictReentrantRevisionError,
    ConflictRevisionBatch,
    MACHINE_CATEGORIES,
    MACHINE_ROWS,
)
from episode_functor_machine import HardFunctorMachine


def _batch(
    *,
    batch_size: int = 2,
    records: int = 5,
    width: int = 64,
) -> ConflictRevisionBatch:
    generator = torch.Generator().manual_seed(20260724)
    transition = torch.randn(
        batch_size,
        PRIMARY_ACTIONS,
        PRIMARY_STATES,
        PRIMARY_STATES,
        generator=generator,
    )
    observer = torch.randn(
        batch_size,
        PRIMARY_OBSERVERS,
        PRIMARY_STATES,
        PRIMARY_ANSWERS,
        generator=generator,
    )
    claims = torch.randn(
        batch_size,
        records,
        MACHINE_ROWS,
        MACHINE_CATEGORIES,
        generator=generator,
    )
    closure = torch.randn(
        batch_size,
        records,
        MACHINE_ROWS,
        MACHINE_CATEGORIES,
        generator=generator,
    )
    incidence = torch.rand(
        batch_size,
        records,
        MACHINE_ROWS,
        generator=generator,
    )
    features = torch.randn(
        batch_size,
        records,
        width,
        generator=generator,
    )
    valid = torch.ones(batch_size, records, dtype=torch.bool)
    return ConflictRevisionBatch(
        transition_logits=transition,
        observer_logits=observer,
        claim_logits=claims,
        closure_claim_logits=closure,
        claim_incidence=incidence,
        closure_incidence=incidence.roll(1, dims=1),
        record_features=features,
        record_valid=valid,
    )


def test_revision_shapes_metrics_and_source_deleted_seal() -> None:
    batch = _batch()
    module = ConflictGatedReentrantRevision(
        record_width=batch.record_width,
        controller_width=128,
        cycles=3,
    )
    result = module(batch)
    assert result.contradiction_energy.shape == (batch.batch_size, 4)
    assert result.step_scale.shape == (
        batch.batch_size,
        3,
        MACHINE_ROWS,
    )
    assert len(result.cycle_transition_logits) == 3
    assert len(result.cycle_observer_logits) == 3
    assert torch.isfinite(result.contradiction_energy).all()
    assert (result.step_scale >= 0).all()
    sealed = module.seal(result)
    assert isinstance(sealed, HardFunctorMachine)
    assert not hasattr(sealed, "record_features")
    assert not hasattr(sealed, "claim_logits")


def test_tied_recurrence_does_not_add_parameters_with_cycles() -> None:
    one = ConflictGatedReentrantRevision(
        record_width=64,
        controller_width=128,
        cycles=1,
    )
    seven = ConflictGatedReentrantRevision(
        record_width=64,
        controller_width=128,
        cycles=7,
    )
    assert one.parameter_count() == seven.parameter_count()


def test_feedback_reinterprets_record_binding_without_new_cycle_parameters() -> None:
    batch = _batch(batch_size=1)
    module = ConflictGatedReentrantRevision(
        record_width=batch.record_width,
        controller_width=128,
        cycles=3,
    )
    causal = module(batch, routing_mode="causal")
    open_loop = module(batch, routing_mode="open-loop")
    assert not torch.equal(
        causal.cycle_claim_incidence[0],
        batch.claim_incidence,
    )
    assert not torch.equal(
        causal.cycle_claim_incidence[0],
        causal.cycle_claim_incidence[-1],
    )
    assert torch.equal(
        open_loop.cycle_claim_incidence[0],
        batch.claim_incidence,
    )
    assert torch.equal(
        open_loop.cycle_claim_incidence[-1],
        batch.claim_incidence,
    )


def test_euclidean_and_fisher_are_both_finite_and_distinct() -> None:
    batch = _batch(batch_size=1)
    module = ConflictGatedReentrantRevision(
        record_width=batch.record_width,
        controller_width=128,
        cycles=2,
    )
    euclidean = module(batch, update_metric="euclidean")
    fisher = module(batch, update_metric="quotient-fisher")
    assert torch.isfinite(euclidean.contradiction_energy).all()
    assert torch.isfinite(fisher.contradiction_energy).all()
    assert not torch.equal(
        euclidean.cycle_transition_logits[-1],
        fisher.cycle_transition_logits[-1],
    )


def test_controls_preserve_geometry_and_destroy_only_routing_signal() -> None:
    batch = _batch(batch_size=1)
    module = ConflictGatedReentrantRevision(
        record_width=batch.record_width,
        controller_width=128,
        cycles=2,
    )
    outputs = {
        mode: module(batch, routing_mode=mode)
        for mode in (
            "causal",
            "deranged",
            "open-loop",
            "sign-scrambled",
        )
    }
    assert {
        value.projection.machine.action_next.shape
        for value in outputs.values()
    } == {(1, 8, 16, 16)}
    assert torch.equal(
        outputs["open-loop"].contradiction_energy,
        torch.zeros_like(outputs["open-loop"].contradiction_energy),
    )
    assert not torch.equal(
        outputs["causal"].cycle_transition_logits[-1],
        outputs["deranged"].cycle_transition_logits[-1],
    )
    changed_source = ConflictRevisionBatch(
        transition_logits=batch.transition_logits,
        observer_logits=batch.observer_logits,
        claim_logits=batch.claim_logits.flip(1).mul(3.0),
        closure_claim_logits=batch.closure_claim_logits.flip(1).sub(2.0),
        claim_incidence=batch.claim_incidence.flip(1),
        closure_incidence=batch.closure_incidence.flip(1),
        record_features=batch.record_features.flip(1).mul(-4.0),
        record_valid=batch.record_valid,
    )
    changed_open_loop = module(
        changed_source,
        routing_mode="open-loop",
    )
    assert torch.equal(
        outputs["open-loop"].cycle_transition_logits[-1],
        changed_open_loop.cycle_transition_logits[-1],
    )
    assert torch.equal(
        outputs["open-loop"].cycle_observer_logits[-1],
        changed_open_loop.cycle_observer_logits[-1],
    )


def test_deranged_routing_is_mass_matched_and_record_equivariant() -> None:
    generator = torch.Generator().manual_seed(91)
    incidence = torch.rand(2, 7, MACHINE_ROWS, generator=generator)
    incidence[:, 5:] = 0.0
    deranged = ConflictGatedReentrantRevision._route_incidence(
        incidence,
        mode="deranged",
    )
    assert torch.allclose(
        incidence.sum(-1),
        deranged.sum(-1),
        atol=1e-5,
        rtol=0.0,
    )
    order = torch.tensor((4, 1, 6, 0, 3, 5, 2))
    reordered = ConflictGatedReentrantRevision._route_incidence(
        incidence[:, order],
        mode="deranged",
    )
    assert torch.allclose(
        deranged[:, order],
        reordered,
        atol=1e-6,
        rtol=0.0,
    )
    singleton = torch.zeros_like(incidence)
    singleton[:, 0] = incidence[:, 0]
    with pytest.raises(
        ConflictReentrantRevisionError,
        match="requires two positive records",
    ):
        ConflictGatedReentrantRevision._route_incidence(
            singleton,
            mode="deranged",
        )


def test_open_loop_keeps_every_parameter_in_the_backward_graph() -> None:
    batch = _batch(batch_size=1)
    module = ConflictGatedReentrantRevision(
        record_width=batch.record_width,
        controller_width=128,
        cycles=2,
    )
    result = module(batch, routing_mode="open-loop")
    loss = (
        result.cycle_transition_logits[-1].square().mean()
        + result.cycle_observer_logits[-1].square().mean()
    )
    loss.backward()
    parameters = tuple(module.named_parameters())
    assert parameters
    assert all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        for _, parameter in parameters
    )
    assert all(
        torch.equal(parameter.grad, torch.zeros_like(parameter.grad))
        for name, parameter in parameters
        if name.startswith("record_encoder.")
    )


def test_row_gauge_shift_leaves_revision_probabilities_invariant() -> None:
    batch = _batch(batch_size=1)
    shifted = ConflictRevisionBatch(
        transition_logits=batch.transition_logits + 7.0,
        observer_logits=batch.observer_logits - 3.0,
        claim_logits=batch.claim_logits + 11.0,
        closure_claim_logits=batch.closure_claim_logits - 5.0,
        claim_incidence=batch.claim_incidence,
        closure_incidence=batch.closure_incidence,
        record_features=batch.record_features,
        record_valid=batch.record_valid,
    )
    module = ConflictGatedReentrantRevision(
        record_width=batch.record_width,
        controller_width=128,
        cycles=2,
    )
    first = module(batch)
    second = module(shifted)
    assert torch.allclose(
        first.projection.transition_transport,
        second.projection.transition_transport,
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        first.projection.observer_transport,
        second.projection.observer_transport,
        atol=1e-6,
        rtol=0.0,
    )


def test_full_machine_recoding_is_exactly_equivariant() -> None:
    batch = _batch(batch_size=1)
    action_order = torch.tensor([2, 0, 1])
    state_order = torch.tensor([5, 2, 7, 0, 4, 1, 6, 3])
    observer_order = torch.tensor([1, 0])
    answer_order = torch.tensor([2, 0, 3, 1])
    action_inverse = action_order.argsort()
    state_inverse = state_order.argsort()
    observer_inverse = observer_order.argsort()
    answer_inverse = answer_order.argsort()

    transition_claim = batch.claim_logits[
        :,
        :,
        : PRIMARY_ACTIONS * PRIMARY_STATES,
    ].reshape(
        1,
        batch.record_count,
        PRIMARY_ACTIONS,
        PRIMARY_STATES,
        MACHINE_CATEGORIES,
    )
    observer_claim = batch.claim_logits[
        :,
        :,
        PRIMARY_ACTIONS * PRIMARY_STATES :,
    ].reshape(
        1,
        batch.record_count,
        PRIMARY_OBSERVERS,
        PRIMARY_STATES,
        MACHINE_CATEGORIES,
    )
    transition_closure = batch.closure_claim_logits[
        :,
        :,
        : PRIMARY_ACTIONS * PRIMARY_STATES,
    ].reshape_as(transition_claim)
    observer_closure = batch.closure_claim_logits[
        :,
        :,
        PRIMARY_ACTIONS * PRIMARY_STATES :,
    ].reshape_as(observer_claim)
    transition_incidence = batch.claim_incidence[
        :,
        :,
        : PRIMARY_ACTIONS * PRIMARY_STATES,
    ].reshape(
        1,
        batch.record_count,
        PRIMARY_ACTIONS,
        PRIMARY_STATES,
    )
    observer_incidence = batch.claim_incidence[
        :,
        :,
        PRIMARY_ACTIONS * PRIMARY_STATES :,
    ].reshape(
        1,
        batch.record_count,
        PRIMARY_OBSERVERS,
        PRIMARY_STATES,
    )
    transition_closure_incidence = batch.closure_incidence[
        :,
        :,
        : PRIMARY_ACTIONS * PRIMARY_STATES,
    ].reshape_as(transition_incidence)
    observer_closure_incidence = batch.closure_incidence[
        :,
        :,
        PRIMARY_ACTIONS * PRIMARY_STATES :,
    ].reshape_as(observer_incidence)

    def recode_claims(
        transition: torch.Tensor,
        observer: torch.Tensor,
    ) -> torch.Tensor:
        transition = transition.index_select(2, action_order)
        transition = transition.index_select(3, state_order)
        transition = transition.index_select(4, state_order)
        observer = observer.index_select(2, observer_order)
        observer = observer.index_select(3, state_order)
        observer_answer = observer[..., :PRIMARY_ANSWERS].index_select(
            4,
            answer_order,
        )
        observer = torch.cat(
            (observer_answer, observer[..., PRIMARY_ANSWERS:]),
            dim=-1,
        )
        return torch.cat(
            (
                transition.reshape(
                    1,
                    batch.record_count,
                    -1,
                    MACHINE_CATEGORIES,
                ),
                observer.reshape(
                    1,
                    batch.record_count,
                    -1,
                    MACHINE_CATEGORIES,
                ),
            ),
            dim=2,
        )

    recoded = ConflictRevisionBatch(
        transition_logits=batch.transition_logits.index_select(
            1,
            action_order,
        ).index_select(2, state_order).index_select(3, state_order),
        observer_logits=batch.observer_logits.index_select(
            1,
            observer_order,
        ).index_select(2, state_order).index_select(3, answer_order),
        claim_logits=recode_claims(
            transition_claim,
            observer_claim,
        ),
        closure_claim_logits=recode_claims(
            transition_closure,
            observer_closure,
        ),
        claim_incidence=torch.cat(
            (
                transition_incidence.index_select(
                    2,
                    action_order,
                ).index_select(3, state_order).reshape(
                    1,
                    batch.record_count,
                    -1,
                ),
                observer_incidence.index_select(
                    2,
                    observer_order,
                ).index_select(3, state_order).reshape(
                    1,
                    batch.record_count,
                    -1,
                ),
            ),
            dim=2,
        ),
        closure_incidence=torch.cat(
            (
                transition_closure_incidence.index_select(
                    2,
                    action_order,
                ).index_select(3, state_order).reshape(
                    1,
                    batch.record_count,
                    -1,
                ),
                observer_closure_incidence.index_select(
                    2,
                    observer_order,
                ).index_select(3, state_order).reshape(
                    1,
                    batch.record_count,
                    -1,
                ),
            ),
            dim=2,
        ),
        record_features=batch.record_features,
        record_valid=batch.record_valid,
    )
    module = ConflictGatedReentrantRevision(
        record_width=batch.record_width,
        controller_width=128,
        cycles=2,
    )
    original = module(batch)
    transformed = module(recoded)
    restored_transition = transformed.projection.transition_transport
    restored_transition = restored_transition.index_select(1, action_inverse)
    restored_transition = restored_transition.index_select(2, state_inverse)
    restored_transition = restored_transition.index_select(3, state_inverse)
    restored_observer = transformed.projection.observer_transport
    restored_observer = restored_observer.index_select(1, observer_inverse)
    restored_observer = restored_observer.index_select(2, state_inverse)
    restored_observer = restored_observer.index_select(3, answer_inverse)
    assert torch.allclose(
        original.projection.transition_transport,
        restored_transition,
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        original.projection.observer_transport,
        restored_observer,
        atol=1e-6,
        rtol=0.0,
    )


def test_invalid_values_and_modes_fail_closed() -> None:
    batch = _batch(batch_size=1)
    with pytest.raises(ConflictReentrantRevisionError):
        ConflictRevisionBatch(
            transition_logits=batch.transition_logits,
            observer_logits=batch.observer_logits,
            claim_logits=batch.claim_logits,
            closure_claim_logits=batch.closure_claim_logits,
            claim_incidence=batch.claim_incidence,
            closure_incidence=batch.closure_incidence,
            record_features=batch.record_features.fill_(float("nan")),
            record_valid=batch.record_valid,
        )
    clean = _batch(batch_size=1)
    module = ConflictGatedReentrantRevision(
        record_width=clean.record_width,
        controller_width=128,
    )
    with pytest.raises(ConflictReentrantRevisionError):
        module(clean, routing_mode="host-solver")
    with pytest.raises(ConflictReentrantRevisionError):
        module(clean, update_metric="natural-language")


def test_rows_without_evidence_do_not_receive_phantom_negative_claims() -> None:
    batch = _batch(batch_size=1)
    direct = batch.claim_incidence.clone()
    closure = batch.closure_incidence.clone()
    direct[:, :, 0] = 0.0
    closure[:, :, 0] = 0.0
    first = ConflictRevisionBatch(
        transition_logits=batch.transition_logits,
        observer_logits=batch.observer_logits,
        claim_logits=batch.claim_logits,
        closure_claim_logits=batch.closure_claim_logits,
        claim_incidence=direct,
        closure_incidence=closure,
        record_features=batch.record_features,
        record_valid=batch.record_valid,
    )
    mutated_claims = batch.claim_logits.clone()
    mutated_closure = batch.closure_claim_logits.clone()
    mutated_claims[:, :, 0] = mutated_claims[:, :, 0].mul(100.0)
    mutated_closure[:, :, 0] = mutated_closure[:, :, 0].sub(100.0)
    second = ConflictRevisionBatch(
        transition_logits=batch.transition_logits,
        observer_logits=batch.observer_logits,
        claim_logits=mutated_claims,
        closure_claim_logits=mutated_closure,
        claim_incidence=direct,
        closure_incidence=closure,
        record_features=batch.record_features,
        record_valid=batch.record_valid,
    )
    module = ConflictGatedReentrantRevision(
        record_width=batch.record_width,
        controller_width=128,
        cycles=2,
    )
    first_result = module(first)
    second_result = module(second)
    assert torch.equal(
        first_result.cycle_transition_logits[-1][:, 0, 0],
        second_result.cycle_transition_logits[-1][:, 0, 0],
    )


def test_large_lane_parameter_budget_is_below_full_system_cap() -> None:
    module = ConflictGatedReentrantRevision(
        record_width=512,
        controller_width=960,
        cycles=4,
    )
    protected_shohin = 125_081_664
    maximum_encoder_without_old_projector = 44_690_832
    query_parser = 6_003_489
    complete = (
        protected_shohin
        + maximum_encoder_without_old_projector
        + module.parameter_count()
        + query_parser
    )
    assert module.parameter_count() > 16_000_000
    assert complete < 200_000_000
