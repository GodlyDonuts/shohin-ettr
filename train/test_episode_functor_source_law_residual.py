from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
import torch

import episode_functor_source_law_residual as source_law_module
from episode_functor_source_law_residual import (
    CONTROL_SEED_SHA256,
    SourceLawResidualError,
    SourceLawResidualIssuer,
    _complete_from_visible_source_law,
    _source_law_residuals_from_visible,
    _transport_candidate_residuals,
    complete_from_source_law,
    source_law_residuals,
)
from episode_functor_witness_compiler import scan_witness_source
from pipeline.episode_functor_identifiable_board import (
    ACTION_COUNT,
    ANSWER_COUNT,
    OBSERVER_COUNT,
    STATE_COUNT,
    PartialEvidence,
    decode_source,
    generate_pilot_rows,
    solve_unique_completion,
)

def _issuer() -> SourceLawResidualIssuer:
    return SourceLawResidualIssuer()


def _visible(
    evidence: PartialEvidence,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    states = tuple(sorted(evidence.state_keys))
    actions = tuple(
        sorted({record[0] for record in evidence.transition_events})
    )
    observers = tuple(
        sorted({record[0] for record in evidence.observation_events})
    )
    state_index = {key: index for index, key in enumerate(states)}
    action_index = {key: index for index, key in enumerate(actions)}
    observer_index = {key: index for index, key in enumerate(observers)}
    transition = torch.zeros(ACTION_COUNT, STATE_COUNT, STATE_COUNT)
    transition_visible = torch.zeros(
        ACTION_COUNT,
        STATE_COUNT,
        dtype=torch.bool,
    )
    for action, source, destination in evidence.transition_events:
        action_slot = action_index[action]
        source_slot = state_index[source]
        transition[
            action_slot,
            source_slot,
            state_index[destination],
        ] = 1.0
        transition_visible[action_slot, source_slot] = True
    observer = torch.zeros(OBSERVER_COUNT, STATE_COUNT, ANSWER_COUNT)
    observer_visible = torch.zeros(
        OBSERVER_COUNT,
        STATE_COUNT,
        dtype=torch.bool,
    )
    for item, state, answer in evidence.observation_events:
        observer_slot = observer_index[item]
        state_slot = state_index[state]
        observer[observer_slot, state_slot, answer] = 1.0
        observer_visible[observer_slot, state_slot] = True
    return (
        transition,
        observer,
        transition_visible,
        observer_visible,
    )


def _fixture_sources() -> tuple[bytes, ...]:
    rows = generate_pilot_rows(
        seed="efc-source-law-residual-test-v1",
        counts={
            "train": 2,
            "mechanics": 2,
            "development": 2,
            "confirmation": 2,
        },
    )
    unique = {}
    for row in rows:
        unique.setdefault(row.world_id, row.source)
    return tuple(unique.values())


def _rename_source(
    source: bytes,
    *,
    state_order: tuple[int, ...],
    action_order: tuple[int, ...],
    observer_order: tuple[int, ...],
) -> tuple[bytes, dict[bytes, bytes]]:
    scanned = scan_witness_source(source)
    evidence = decode_source(source)
    states = tuple(sorted(evidence.state_keys))
    actions = tuple(
        sorted({record[0] for record in evidence.transition_events})
    )
    observers = tuple(
        sorted({record[0] for record in evidence.observation_events})
    )
    mapping = {}
    for prefix, keys, order in (
        (1, states, state_order),
        (2, actions, action_order),
        (3, observers, observer_order),
    ):
        for new_rank, old_index in enumerate(order):
            mapping[keys[old_index]] = (
                f"h{prefix:x}{new_rank:015x}".encode("ascii")
            )
    raw_by_value = {
        int.from_bytes(key, "little"): key
        for key in scanned.pointer.unique_keys
    }
    assert set(mapping) == set(raw_by_value)
    byte_mapping = {
        raw_by_value[key]: value for key, value in mapping.items()
    }
    output = bytearray()
    cursor = 0
    for (start, end), unique in zip(
        scanned.pointer.spans,
        scanned.pointer.occurrence_to_unique,
        strict=True,
    ):
        output.extend(source[cursor:start])
        replacement = byte_mapping[scanned.pointer.unique_keys[unique]]
        output.extend(replacement)
        cursor = end
    output.extend(source[cursor:])
    return bytes(output), byte_mapping


def _raw_key_recode() -> tuple[
    tuple[bytes, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    sources = _fixture_sources()
    state, action, observer, _ = _orders()
    state_order = tuple(state.tolist())
    action_order = tuple(action.tolist())
    observer_order = tuple(observer.tolist())
    recoded = []
    for source in sources:
        renamed, _ = _rename_source(
            source,
            state_order=state_order,
            action_order=action_order,
            observer_order=observer_order,
        )
        recoded.append(renamed)
    return (
        tuple(recoded),
        state_order,
        action_order,
        observer_order,
    )


def _fixture_batch() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    transitions = []
    observers = []
    transition_masks = []
    observer_masks = []
    expected_transitions = []
    expected_observers = []
    for source in _fixture_sources():
        evidence = decode_source(source)
        transition, observer, transition_mask, observer_mask = _visible(
            evidence
        )
        expected = solve_unique_completion(evidence)
        transitions.append(transition)
        observers.append(observer)
        transition_masks.append(transition_mask)
        observer_masks.append(observer_mask)
        expected_transitions.append(
            torch.tensor(expected.transitions, dtype=torch.float32)
        )
        expected_observers.append(
            torch.tensor(expected.observations, dtype=torch.long)
        )
    expected_transition = torch.nn.functional.one_hot(
        torch.stack(expected_transitions).to(torch.long),
        STATE_COUNT,
    ).to(torch.float32)
    expected_observer = torch.nn.functional.one_hot(
        torch.stack(expected_observers),
        ANSWER_COUNT,
    ).to(torch.float32)
    return (
        torch.stack(transitions),
        torch.stack(observers),
        torch.stack(transition_masks),
        torch.stack(observer_masks),
        expected_transition,
        expected_observer,
    )


def _orders() -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor((2, 5, 0, 7, 1, 6, 3, 4)),
        torch.tensor((2, 0, 1)),
        torch.tensor((1, 0)),
        torch.tensor((2, 0, 3, 1)),
    )


def _recode(
    transition: torch.Tensor,
    observer: torch.Tensor,
    transition_mask: torch.Tensor,
    observer_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state, action, item, answer = _orders()
    return (
        transition[
            :,
            action,
        ][:, :, state][:, :, :, state],
        observer[
            :,
            item,
        ][:, :, state][:, :, :, answer],
        transition_mask[:, action][:, :, state],
        observer_mask[:, item][:, :, state],
    )


def test_public_functions_have_no_target_supervisor_or_query_input() -> None:
    for function in (
        source_law_residuals,
        complete_from_source_law,
    ):
        parameters = tuple(inspect.signature(function).parameters)
        assert all(
            token not in parameter
            for parameter in parameters
            for token in ("label", "query", "supervisor", "target")
        )
        assert parameters == ("issuer", "capability")


def test_source_law_recovers_every_generated_hidden_cell() -> None:
    (
        transition,
        observer,
        transition_mask,
        observer_mask,
        expected_transition,
        expected_observer,
    ) = _fixture_batch()
    issuer = _issuer()
    capability = issuer.issue(_fixture_sources())
    residuals = source_law_residuals(
        issuer,
        capability,
    )
    completed = complete_from_source_law(
        issuer,
        capability,
    )
    private_residuals = _source_law_residuals_from_visible(
        transition,
        observer,
        transition_mask,
        observer_mask,
    )
    assert torch.equal(completed[0], expected_transition)
    assert torch.equal(completed[1], expected_observer)
    assert torch.equal(
        residuals.transition_residuals,
        private_residuals.transition_residuals,
    )
    assert bool(
        residuals.transition_residuals[
            residuals.transition_hidden_rows
        ].amin(-1).eq(0).all()
    )
    assert bool(
        residuals.observer_residuals[
            residuals.observer_hidden_rows
        ].amin(-1).eq(0).all()
    )


def test_treatment_is_exactly_gauge_equivariant() -> None:
    transition, observer, transition_mask, observer_mask, _, _ = (
        _fixture_batch()
    )
    baseline = _source_law_residuals_from_visible(
        transition,
        observer,
        transition_mask,
        observer_mask,
    )
    recoded_inputs = _recode(
        transition,
        observer,
        transition_mask,
        observer_mask,
    )
    recoded = _source_law_residuals_from_visible(*recoded_inputs)
    state, action, item, answer = _orders()
    assert torch.equal(
        recoded.transition_residuals,
        baseline.transition_residuals[
            :,
            action,
        ][:, :, state][:, :, :, state],
    )
    assert torch.equal(
        recoded.observer_residuals,
        baseline.observer_residuals[
            :,
            item,
        ][:, :, state][:, :, :, answer],
    )
    baseline_complete = _complete_from_visible_source_law(
        transition,
        observer,
        transition_mask,
        observer_mask,
    )
    recoded_complete = _complete_from_visible_source_law(*recoded_inputs)
    expected = _recode(
        baseline_complete[0],
        baseline_complete[1],
        transition_mask,
        observer_mask,
    )
    assert torch.equal(recoded_complete[0], expected[0])
    assert torch.equal(recoded_complete[1], expected[1])


def test_source_capability_cannot_be_forged_copied_or_cross_issued() -> None:
    issuer = _issuer()
    capability = issuer.issue(_fixture_sources())
    copied = replace(capability)
    with pytest.raises(SourceLawResidualError, match="provenance"):
        issuer.residuals(copied)
    with pytest.raises(SourceLawResidualError, match="provenance"):
        _issuer().residuals(capability)
    mutated = issuer.issue(_fixture_sources())
    object.__setattr__(
        mutated,
        "source_sha256",
        tuple("0" * 64 for _ in mutated.source_sha256),
    )
    with pytest.raises(SourceLawResidualError, match="provenance"):
        issuer.residuals(mutated)
    schema_mutated = issuer.issue(_fixture_sources())
    object.__setattr__(schema_mutated, "schema", "forged")
    with pytest.raises(SourceLawResidualError, match="provenance"):
        issuer.residuals(schema_mutated)
    source_mutated = issuer.issue(_fixture_sources())
    source_receipt = issuer._evidence[source_mutated.capability]
    object.__setattr__(
        source_receipt,
        "sources",
        (b"mutated",) + source_receipt.sources[1:],
    )
    with pytest.raises(SourceLawResidualError, match="provenance"):
        issuer.residuals(source_mutated)
    seed_probe = _issuer()
    seed_evidence = seed_probe.issue(_fixture_sources())
    source_law_module.CONTROL_SEED_SHA256 = "0" * 64
    try:
        assert (
            seed_probe.issue_control(seed_evidence).control_seed_sha256
            == CONTROL_SEED_SHA256
        )
    finally:
        source_law_module.CONTROL_SEED_SHA256 = CONTROL_SEED_SHA256
    issued = issuer._evidence[capability.capability]
    index = issued.transition.nonzero()[0].tolist()
    issued.transition[tuple(index)] = 0
    with pytest.raises(SourceLawResidualError, match="provenance"):
        issuer.residuals(capability)


def test_hard_derangement_preserves_multiset_and_conjugates() -> None:
    issuer = _issuer()
    evidence_capability = issuer.issue(_fixture_sources())
    control_capability = issuer.issue_control(evidence_capability)
    assert control_capability.control_seed_sha256 == CONTROL_SEED_SHA256
    with pytest.raises(SourceLawResidualError, match="already issued"):
        issuer.issue_control(evidence_capability)
    baseline = issuer.residuals(evidence_capability)
    controlled = issuer.controlled_residuals(
        evidence_capability,
        control_capability,
    )
    with pytest.raises(SourceLawResidualError, match="provenance"):
        issuer.controlled_residuals(
            evidence_capability,
            replace(control_capability),
        )
    assert torch.equal(
        controlled.transition_residuals.sort(-1).values,
        baseline.transition_residuals.sort(-1).values,
    )
    assert torch.equal(
        controlled.observer_residuals.sort(-1).values,
        baseline.observer_residuals.sort(-1).values,
    )
    issued_control = issuer._controls[control_capability.capability]
    for transport in (
        issued_control.transition_transport,
        issued_control.observer_transport,
    ):
        assert bool(transport.sum(-1).eq(1).all())
        assert bool(transport.sum(-2).eq(1).all())
        assert bool(
            transport.diagonal(dim1=-2, dim2=-1).eq(0).all()
        )

    (
        recoded_sources,
        state_order,
        action_order,
        observer_order,
    ) = _raw_key_recode()
    recoded_evidence, recoded_control = issuer.recode_control(
        evidence_capability,
        control_capability,
        recoded_sources,
    )
    recoded_controlled = issuer.controlled_residuals(
        recoded_evidence,
        recoded_control,
    )
    state = torch.tensor(state_order)
    action = torch.tensor(action_order)
    observer = torch.tensor(observer_order)
    assert torch.equal(
        recoded_controlled.transition_residuals,
        controlled.transition_residuals[
            :, action
        ][:, :, state][:, :, :, state],
    )
    assert torch.equal(
        recoded_controlled.observer_residuals,
        controlled.observer_residuals[
            :, observer
        ][:, :, state],
    )
    object.__setattr__(recoded_control, "schema", "forged")
    with pytest.raises(SourceLawResidualError, match="provenance"):
        issuer.controlled_residuals(
            recoded_evidence,
            recoded_control,
        )


def test_soft_transport_and_ambiguous_visibility_fail_closed() -> None:
    transition, observer, transition_mask, observer_mask, _, _ = (
        _fixture_batch()
    )
    residuals = _source_law_residuals_from_visible(
        transition,
        observer,
        transition_mask,
        observer_mask,
    )
    soft = torch.full(
        residuals.transition_residuals.shape + (STATE_COUNT,),
        1.0 / STATE_COUNT,
    )
    with pytest.raises(
        SourceLawResidualError,
        match="hard derangement",
    ):
        _transport_candidate_residuals(
            residuals.transition_residuals,
            soft,
        )
    identity = torch.eye(STATE_COUNT).expand(
        residuals.transition_residuals.shape + (STATE_COUNT,)
    )
    with pytest.raises(SourceLawResidualError, match="hard derangement"):
        _transport_candidate_residuals(
            residuals.transition_residuals,
            identity,
        )
    soft_visible = transition.clone()
    visible_row = transition_mask.nonzero()[0].tolist()
    batch, relation, row = visible_row
    category = int(soft_visible[batch, relation, row].argmax())
    other = (category + 1) % STATE_COUNT
    soft_visible[batch, relation, row, category] = 0.9
    soft_visible[batch, relation, row, other] = 0.1
    with pytest.raises(SourceLawResidualError, match="visibility"):
        _source_law_residuals_from_visible(
            soft_visible,
            observer,
            transition_mask,
            observer_mask,
        )
    ambiguous_mask = transition_mask.clone()
    first_hidden = (~ambiguous_mask[:, 0]).nonzero()[0]
    batch, hidden = first_hidden.tolist()
    extra = (hidden + 1) % STATE_COUNT
    ambiguous_mask[batch, 0, extra] = False
    ambiguous_transition = transition.clone()
    ambiguous_transition[batch, 0, extra] = 0
    with pytest.raises(
        SourceLawResidualError,
        match="exactly one row",
    ):
        _source_law_residuals_from_visible(
            ambiguous_transition,
            observer,
            ambiguous_mask,
            observer_mask,
        )


def test_law_residual_backward_is_finite_without_parameters() -> None:
    transition, observer, transition_mask, observer_mask, _, _ = (
        _fixture_batch()
    )
    transition.requires_grad_()
    observer.requires_grad_()
    result = _source_law_residuals_from_visible(
        transition,
        observer,
        transition_mask,
        observer_mask,
    )
    loss = (
        result.transition_residuals.square().sum()
        + result.observer_residuals.square().sum()
    )
    loss.backward()
    assert transition.grad is not None
    assert observer.grad is not None
    assert bool(torch.isfinite(transition.grad).all())
    assert bool(torch.isfinite(observer.grad).all())
    assert bool(transition.grad[transition_mask].abs().sum().gt(0))
    assert bool(observer.grad[observer_mask].abs().sum().gt(0))
    assert bool(transition.grad[~transition_mask].eq(0).all())
    assert bool(observer.grad[~observer_mask].eq(0).all())
