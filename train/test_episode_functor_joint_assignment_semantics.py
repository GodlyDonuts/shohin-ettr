from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from episode_functor_constrained_transport import (
    PRIMARY_ACTIONS,
    PRIMARY_ANSWERS,
    PRIMARY_OBSERVERS,
    PRIMARY_STATES,
)
from episode_functor_joint_assignment_semantics import (
    JointAssignmentSemanticsError,
    UNAVAILABLE_COMPATIBILITY,
    joint_semantic_compatibility,
    machine_behavior_signatures,
)
from episode_functor_physical_key_nerve import PhysicalKeyNerveResult


UNIQUE = PRIMARY_STATES + PRIMARY_ACTIONS + PRIMARY_OBSERVERS


def _machine() -> tuple[torch.Tensor, torch.Tensor]:
    transition_indices = torch.tensor(
        (
            (1, 2, 3, 4, 5, 6, 7, 0),
            (0, 2, 1, 4, 3, 6, 5, 7),
            (3, 0, 5, 2, 7, 4, 1, 6),
        )
    )
    observer_indices = torch.tensor(
        (
            (0, 1, 2, 3, 1, 2, 3, 0),
            (3, 1, 0, 2, 0, 3, 2, 1),
        )
    )
    return (
        F.one_hot(
            transition_indices,
            PRIMARY_STATES,
        ).float()[None],
        F.one_hot(
            observer_indices,
            PRIMARY_ANSWERS,
        ).float()[None],
    )


def _nerve_from_machine(
    transition: torch.Tensor,
    observer: torch.Tensor,
) -> PhysicalKeyNerveResult:
    transition_index = transition[0].argmax(-1)
    observer_index = observer[0].argmax(-1)
    physical_state = torch.zeros(1, UNIQUE, 104)
    physical_action = torch.zeros(1, UNIQUE, 512)
    physical_observer = torch.zeros(1, UNIQUE, 32)
    inverse_degree = 1.0 / float(PRIMARY_ACTIONS)
    for state in range(PRIMARY_STATES):
        state_values: list[float] = []
        for observed in range(PRIMARY_OBSERVERS):
            for answer in range(PRIMARY_ANSWERS):
                state_values.append(
                    float(observer_index[observed, state] == answer)
                )
        for action in range(PRIMARY_ACTIONS):
            destination = int(transition_index[action, state])
            for observed in range(PRIMARY_OBSERVERS):
                for answer in range(PRIMARY_ANSWERS):
                    state_values.append(
                        inverse_degree
                        * float(
                            observer_index[observed, destination]
                            == answer
                        )
                    )
        for left in range(PRIMARY_ACTIONS):
            middle = int(transition_index[left, state])
            for right in range(PRIMARY_ACTIONS):
                destination = int(transition_index[right, middle])
                for final_state in range(PRIMARY_STATES):
                    state_values.append(
                        inverse_degree
                        * float(destination == final_state)
                    )
        physical_state[0, state] = torch.tensor(state_values)

    action_start = PRIMARY_STATES
    for action in range(PRIMARY_ACTIONS):
        action_values: list[float] = []
        for state in range(PRIMARY_STATES):
            destination = int(transition_index[action, state])
            for final_state in range(PRIMARY_STATES):
                action_values.append(float(destination == final_state))
        for state in range(PRIMARY_STATES):
            destination = int(transition_index[action, state])
            for observed in range(PRIMARY_OBSERVERS):
                for answer in range(PRIMARY_ANSWERS):
                    action_values.append(
                        inverse_degree
                        * float(
                            observer_index[observed, destination]
                            == answer
                        )
                    )
        for right in range(PRIMARY_ACTIONS):
            for state in range(PRIMARY_STATES):
                middle = int(transition_index[action, state])
                destination = int(transition_index[right, middle])
                for final_state in range(PRIMARY_STATES):
                    action_values.append(
                        inverse_degree
                        * float(destination == final_state)
                    )
        for left in range(PRIMARY_ACTIONS):
            for state in range(PRIMARY_STATES):
                middle = int(transition_index[left, state])
                destination = int(transition_index[action, middle])
                for final_state in range(PRIMARY_STATES):
                    action_values.append(
                        inverse_degree
                        * float(destination == final_state)
                    )
        physical_action[0, action_start + action] = torch.tensor(
            action_values
        )

    observer_start = PRIMARY_STATES + PRIMARY_ACTIONS
    for observed in range(PRIMARY_OBSERVERS):
        observer_values: list[float] = []
        for state in range(PRIMARY_STATES):
            for answer in range(PRIMARY_ANSWERS):
                observer_values.append(
                    float(observer_index[observed, state] == answer)
                )
        physical_observer[0, observer_start + observed] = torch.tensor(
            observer_values
        )
    return PhysicalKeyNerveResult(
        transition_relation=torch.zeros(1, UNIQUE, UNIQUE, UNIQUE),
        observation_relation=torch.zeros(
            1,
            UNIQUE,
            UNIQUE,
            PRIMARY_ANSWERS,
        ),
        state_signature=physical_state,
        action_signature=physical_action,
        observer_signature=physical_observer,
        state_compatibility=torch.zeros(
            1,
            PRIMARY_STATES,
            UNIQUE,
        ),
        action_compatibility=torch.zeros(
            1,
            PRIMARY_ACTIONS,
            UNIQUE,
        ),
        observer_compatibility=torch.zeros(
            1,
            PRIMARY_OBSERVERS,
            UNIQUE,
        ),
        action_left_compatibility=torch.zeros(
            1,
            PRIMARY_ACTIONS,
            UNIQUE,
        ),
        action_right_compatibility=torch.zeros(
            1,
            PRIMARY_ACTIONS,
            UNIQUE,
        ),
        action_observer_compatibility=torch.zeros(
            1,
            PRIMARY_ACTIONS,
            UNIQUE,
        ),
        action_commutator_compatibility=torch.zeros(
            1,
            PRIMARY_ACTIONS,
            UNIQUE,
        ),
        path_mass=torch.ones(1),
        mode="causal",
    )


def _append_padding_key(
    nerve: PhysicalKeyNerveResult,
) -> PhysicalKeyNerveResult:
    unique = nerve.transition_relation.shape[1]
    transition = torch.zeros(1, unique + 1, unique + 1, unique + 1)
    transition[:, :unique, :unique, :unique] = (
        nerve.transition_relation
    )
    observation = torch.zeros(
        1,
        unique + 1,
        unique + 1,
        PRIMARY_ANSWERS,
    )
    observation[:, :unique, :unique] = nerve.observation_relation

    def pad_signature(value: torch.Tensor) -> torch.Tensor:
        return F.pad(value, (0, 0, 0, 1))

    def pad_compatibility(value: torch.Tensor) -> torch.Tensor:
        return F.pad(value, (0, 1))

    return replace(
        nerve,
        transition_relation=transition,
        observation_relation=observation,
        state_signature=pad_signature(nerve.state_signature),
        action_signature=pad_signature(nerve.action_signature),
        observer_signature=pad_signature(nerve.observer_signature),
        state_compatibility=pad_compatibility(
            nerve.state_compatibility
        ),
        action_compatibility=pad_compatibility(
            nerve.action_compatibility
        ),
        observer_compatibility=pad_compatibility(
            nerve.observer_compatibility
        ),
        action_left_compatibility=pad_compatibility(
            nerve.action_left_compatibility
        ),
        action_right_compatibility=pad_compatibility(
            nerve.action_right_compatibility
        ),
        action_observer_compatibility=pad_compatibility(
            nerve.action_observer_compatibility
        ),
        action_commutator_compatibility=pad_compatibility(
            nerve.action_commutator_compatibility
        ),
    )


def test_matching_machine_identifies_every_physical_key() -> None:
    transition, observer = _machine()
    result = joint_semantic_compatibility(
        _nerve_from_machine(transition, observer),
        transition,
        observer,
        torch.ones(1, UNIQUE, dtype=torch.bool),
    )
    expected = torch.arange(UNIQUE)
    assert torch.equal(
        result.assignment_compatibility.argmax(-1)[0],
        expected,
    )
    diagonal = result.assignment_compatibility[
        0,
        expected,
        expected,
    ]
    assert torch.equal(diagonal, torch.zeros_like(diagonal))
    assert bool(
        result.assignment_compatibility[
            0,
            :PRIMARY_STATES,
            :PRIMARY_STATES,
        ][~torch.eye(PRIMARY_STATES, dtype=torch.bool)].lt(0).all()
    )


def test_exact_physical_machine_match_has_zero_objective_gradient() -> None:
    transition, observer = _machine()
    transition.requires_grad_()
    observer.requires_grad_()
    result = joint_semantic_compatibility(
        _nerve_from_machine(transition.detach(), observer.detach()),
        transition,
        observer,
        torch.ones(1, UNIQUE, dtype=torch.bool),
    )
    expected = torch.arange(UNIQUE)
    diagonal = result.assignment_compatibility[
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
    (-diagonal.sum()).backward()
    assert transition.grad is not None
    assert observer.grad is not None
    assert torch.allclose(
        transition.grad,
        torch.zeros_like(transition.grad),
        atol=1e-7,
        rtol=0.0,
    )
    assert torch.allclose(
        observer.grad,
        torch.zeros_like(observer.grad),
        atol=1e-7,
        rtol=0.0,
    )


def test_physical_key_permutation_commutes_exactly() -> None:
    transition, observer = _machine()
    nerve = _nerve_from_machine(transition, observer)
    valid = torch.ones(1, UNIQUE, dtype=torch.bool)
    baseline = joint_semantic_compatibility(
        nerve,
        transition,
        observer,
        valid,
    )
    order = torch.randperm(
        UNIQUE,
        generator=torch.Generator().manual_seed(20260724),
    )
    recoded = replace(
        nerve,
        transition_relation=nerve.transition_relation[
            :,
            order,
        ][:, :, order][:, :, :, order],
        observation_relation=nerve.observation_relation[
            :,
            order,
        ][:, :, order],
        state_signature=nerve.state_signature[:, order],
        action_signature=nerve.action_signature[:, order],
        observer_signature=nerve.observer_signature[:, order],
    )
    permuted = joint_semantic_compatibility(
        recoded,
        transition,
        observer,
        valid[:, order],
    )
    assert torch.equal(
        baseline.assignment_compatibility[:, :, order],
        permuted.assignment_compatibility,
    )


def test_complete_semantic_gauge_relabeling_commutes_exactly() -> None:
    transition, observer = _machine()
    nerve = _nerve_from_machine(transition, observer)
    valid = torch.ones(1, UNIQUE, dtype=torch.bool)
    baseline = joint_semantic_compatibility(
        nerve,
        transition,
        observer,
        valid,
    )
    state_order = torch.tensor((3, 0, 7, 2, 5, 1, 6, 4))
    action_order = torch.tensor((2, 0, 1))
    observer_order = torch.tensor((1, 0))
    answer_order = torch.tensor((2, 0, 3, 1))
    recoded_transition = transition[
        :,
        action_order,
    ][:, :, state_order][:, :, :, state_order]
    recoded_observer = observer[
        :,
        observer_order,
    ][:, :, state_order][:, :, :, answer_order]
    recoded_nerve = _nerve_from_machine(
        recoded_transition,
        recoded_observer,
    )
    recoded = joint_semantic_compatibility(
        recoded_nerve,
        recoded_transition,
        recoded_observer,
        valid,
    )
    role_order = torch.cat(
        (
            state_order,
            PRIMARY_STATES + action_order,
            PRIMARY_STATES
            + PRIMARY_ACTIONS
            + observer_order,
        )
    )
    expected = baseline.assignment_compatibility[
        :,
        role_order,
    ][:, :, role_order]
    assert torch.allclose(
        recoded.assignment_compatibility,
        expected,
        atol=1e-7,
        rtol=0.0,
    )


def test_noncommuting_action_order_is_visible() -> None:
    transition, observer = _machine()
    _, action, _ = machine_behavior_signatures(transition, observer)
    left = action[..., 128:320]
    right = action[..., 320:512]
    assert not torch.equal(left, right)
    swapped = transition[:, (1, 0, 2)]
    _, swapped_action, _ = machine_behavior_signatures(swapped, observer)
    assert not torch.equal(action[:, 0], swapped_action[:, 0])


def test_machine_to_assignment_cut_is_graph_connected_zero() -> None:
    transition, observer = _machine()
    transition.requires_grad_()
    result = joint_semantic_compatibility(
        _nerve_from_machine(transition.detach(), observer),
        transition,
        observer,
        torch.ones(1, UNIQUE, dtype=torch.bool),
        mode="machine-to-assignment-cut",
    )
    available = result.assignment_compatibility.ne(
        UNAVAILABLE_COMPATIBILITY
    )
    assert torch.equal(
        result.assignment_compatibility[available],
        torch.zeros_like(result.assignment_compatibility[available]),
    )
    result.assignment_compatibility[available].sum().backward()
    assert transition.grad is not None
    assert torch.equal(transition.grad, torch.zeros_like(transition.grad))
    for signature in (
        result.machine_state_signature,
        result.machine_action_signature,
        result.machine_observer_signature,
    ):
        assert torch.equal(signature, torch.zeros_like(signature))

    changed_transition = transition.detach().roll(1, dims=-1)
    changed_observer = observer.roll(1, dims=-1)
    changed = joint_semantic_compatibility(
        _nerve_from_machine(transition.detach(), observer),
        changed_transition,
        changed_observer,
        torch.ones(1, UNIQUE, dtype=torch.bool),
        mode="machine-to-assignment-cut",
    )
    assert torch.equal(
        result.assignment_compatibility,
        changed.assignment_compatibility,
    )


def test_one_step_control_removes_all_ordered_path_channels() -> None:
    transition, observer = _machine()
    state, action, _ = machine_behavior_signatures(
        transition,
        observer,
        one_step_only=True,
    )
    assert torch.equal(state[..., 32:], torch.zeros_like(state[..., 32:]))
    assert torch.equal(action[..., 128:], torch.zeros_like(action[..., 128:]))
    result = joint_semantic_compatibility(
        _nerve_from_machine(transition, observer),
        transition,
        observer,
        torch.ones(1, UNIQUE, dtype=torch.bool),
        mode="one-step-only",
    )
    expected = torch.arange(UNIQUE)
    assert torch.allclose(
        result.assignment_compatibility[0, expected, expected],
        torch.zeros(UNIQUE),
        atol=1e-7,
        rtol=0.0,
    )


def test_invalid_probabilities_fail_closed() -> None:
    transition, observer = _machine()
    with pytest.raises(
        JointAssignmentSemanticsError,
        match="transition probabilities",
    ):
        machine_behavior_signatures(
            transition.clone().index_put_(
                (torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), torch.tensor([0])),
                torch.tensor([-1.0]),
            ),
            observer,
        )


def test_invalid_and_zero_evidence_keys_are_never_attractive() -> None:
    transition, observer = _machine()
    nerve = _append_padding_key(
        _nerve_from_machine(transition, observer)
    )
    valid = torch.ones(1, UNIQUE + 1, dtype=torch.bool)
    valid[:, -1] = False
    result = joint_semantic_compatibility(
        nerve,
        transition,
        observer,
        valid,
    )
    assert torch.equal(
        result.assignment_compatibility[..., -1],
        torch.full_like(
            result.assignment_compatibility[..., -1],
            UNAVAILABLE_COMPATIBILITY,
        ),
    )
    assert bool(
        result.assignment_compatibility.argmax(-1).ne(UNIQUE).all()
    )
    state_scores = result.state_compatibility
    assert torch.equal(
        state_scores[..., PRIMARY_STATES:],
        torch.full_like(
            state_scores[..., PRIMARY_STATES:],
            UNAVAILABLE_COMPATIBILITY,
        ),
    )


def test_negative_physical_signature_fails_closed() -> None:
    transition, observer = _machine()
    nerve = _nerve_from_machine(transition, observer)
    corrupted = replace(
        nerve,
        state_signature=nerve.state_signature.clone(),
    )
    corrupted.state_signature[0, 0, 0] = -1.0
    with pytest.raises(
        JointAssignmentSemanticsError,
        match="semantic and physical signatures",
    ):
        joint_semantic_compatibility(
            corrupted,
            transition,
            observer,
            torch.ones(1, UNIQUE, dtype=torch.bool),
        )


def test_wrong_valid_key_count_fails_closed() -> None:
    transition, observer = _machine()
    with pytest.raises(
        JointAssignmentSemanticsError,
        match="key geometry",
    ):
        joint_semantic_compatibility(
            _nerve_from_machine(transition, observer),
            transition,
            observer,
            torch.zeros(1, UNIQUE, dtype=torch.bool),
        )
