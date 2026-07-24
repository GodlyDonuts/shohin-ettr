from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

from episode_functor_counterfactual_repair import (
    COUNTERFACTUAL_REPAIR_MODES,
    COUNTERFACTUAL_REPAIR_ADMITTED,
    CounterfactualMachineRepair,
    CounterfactualRepairError,
    DEFAULT_PARAMETER_COUNT,
)
from episode_functor_runtime_constants import (
    PRIMARY_ACTIONS,
    PRIMARY_ANSWERS,
    PRIMARY_OBSERVERS,
    PRIMARY_STATES,
)


def _small(*, cycles: int = 1) -> CounterfactualMachineRepair:
    torch.manual_seed(197)
    return CounterfactualMachineRepair(
        width=16,
        memory_width=8,
        cycles=cycles,
    )


def _one_hot_fixture() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    transition = torch.zeros(
        1,
        PRIMARY_ACTIONS,
        PRIMARY_STATES,
        PRIMARY_STATES,
    )
    transition_evidence = torch.zeros_like(transition)
    for action in range(PRIMARY_ACTIONS):
        for state in range(PRIMARY_STATES):
            transition[0, action, state, (state + 2 * action + 2) % 8] = 1
            transition_evidence[
                0,
                action,
                state,
                (state + action + 1) % 8,
            ] = 1
    observer = torch.zeros(
        1,
        PRIMARY_OBSERVERS,
        PRIMARY_STATES,
        PRIMARY_ANSWERS,
    )
    observer_evidence = torch.zeros_like(observer)
    for item in range(PRIMARY_OBSERVERS):
        for state in range(PRIMARY_STATES):
            observer[0, item, state, (state + item + 1) % 4] = 1
            observer_evidence[
                0,
                item,
                state,
                (state + item) % 4,
            ] = 1
    return (
        transition,
        observer,
        transition_evidence,
        observer_evidence,
    )


def _soft_fixture() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    generator = torch.Generator().manual_seed(211)
    transition = torch.rand(
        1,
        PRIMARY_ACTIONS,
        PRIMARY_STATES,
        PRIMARY_STATES,
        generator=generator,
    )
    observer = torch.rand(
        1,
        PRIMARY_OBSERVERS,
        PRIMARY_STATES,
        PRIMARY_ANSWERS,
        generator=generator,
    )
    transition_evidence = torch.rand(
        transition.shape,
        generator=generator,
    )
    observer_evidence = torch.rand(
        observer.shape,
        generator=generator,
    )
    return (
        transition,
        observer,
        transition_evidence,
        observer_evidence,
    )


def _recode(
    transition: torch.Tensor,
    observer: torch.Tensor,
    state_order: torch.Tensor,
    action_order: torch.Tensor,
    observer_order: torch.Tensor,
    answer_order: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        transition[
            :,
            action_order,
        ][:, :, state_order][:, :, :, state_order],
        observer[
            :,
            observer_order,
        ][:, :, state_order][:, :, :, answer_order],
    )


def test_default_parameter_receipt_is_exact() -> None:
    assert COUNTERFACTUAL_REPAIR_ADMITTED is False
    model = CounterfactualMachineRepair()
    assert model.parameter_count() == DEFAULT_PARAMETER_COUNT
    assert model.parameter_count() == 9_618_567
    assert not any(
        isinstance(module, torch.nn.Embedding)
        for module in model.modules()
    )


def test_public_forward_has_no_oracle_feature_key_or_source_input() -> None:
    signature = inspect.signature(CounterfactualMachineRepair.forward)
    assert tuple(signature.parameters) == (
        "self",
        "transition_probabilities",
        "observer_probabilities",
        "transition_evidence",
        "observer_evidence",
        "mode",
    )
    forbidden = {
        "counterfactual",
        "feature",
        "key",
        "oracle",
        "query",
        "raw",
        "score",
        "source",
        "target",
    }
    public_names = set(signature.parameters) - {"self", "mode"}
    assert all(
        token not in name
        for name in public_names
        for token in forbidden
    )


def test_shapes_normalization_and_adaptive_outputs() -> None:
    model = _small(cycles=3)
    result = model(*_soft_fixture())
    assert result.transition_probabilities.shape == (
        1,
        PRIMARY_ACTIONS,
        PRIMARY_STATES,
        PRIMARY_STATES,
    )
    assert result.observer_probabilities.shape == (
        1,
        PRIMARY_OBSERVERS,
        PRIMARY_STATES,
        PRIMARY_ANSWERS,
    )
    assert len(result.cycle_transition_probabilities) == 3
    assert len(result.cycle_observer_probabilities) == 3
    assert result.cycle_halt_probabilities.shape == (1, 3)
    assert result.cycle_mixture_weights.shape == (1, 3)
    assert torch.allclose(
        result.transition_probabilities.sum(-1),
        torch.ones_like(result.transition_probabilities[..., 0]),
        atol=2e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        result.observer_probabilities.sum(-1),
        torch.ones_like(result.observer_probabilities[..., 0]),
        atol=2e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        result.cycle_mixture_weights.sum(-1),
        torch.ones(1),
        atol=2e-7,
        rtol=0.0,
    )
    assert bool(
        result.cycle_halt_probabilities[:, :-1].gt(0).all()
    )
    assert bool(
        result.cycle_halt_probabilities[:, :-1].lt(1).all()
    )


def test_fixed_cycle_executes_all_cycles_without_adaptive_exit() -> None:
    model = _small(cycles=3)
    result = model(*_soft_fixture(), mode="fixed-cycle")
    assert len(result.cycle_transition_probabilities) == 3
    assert torch.equal(
        result.cycle_halt_probabilities,
        torch.tensor(((0.0, 0.0, 1.0),)),
    )
    assert torch.equal(
        result.cycle_mixture_weights,
        torch.tensor(((0.0, 0.0, 1.0),)),
    )
    assert torch.allclose(
        result.transition_probabilities,
        result.cycle_transition_probabilities[-1],
        atol=2e-7,
        rtol=0.0,
    )
    assert torch.allclose(
        result.observer_probabilities,
        result.cycle_observer_probabilities[-1],
        atol=2e-7,
        rtol=0.0,
    )


def test_backward_is_finite_and_reaches_counterfactual_controller() -> None:
    model = _small(cycles=2)
    transition, observer, transition_evidence, observer_evidence = (
        _soft_fixture()
    )
    transition.requires_grad_()
    observer.requires_grad_()
    result = model(
        transition,
        observer,
        transition_evidence,
        observer_evidence,
    )
    loss = (
        result.transition_probabilities.square().sum()
        + result.observer_probabilities.square().sum()
        + result.cycle_halt_probabilities.sum()
    )
    loss.backward()
    assert transition.grad is not None
    assert observer.grad is not None
    assert bool(torch.isfinite(transition.grad).all())
    assert bool(torch.isfinite(observer.grad).all())
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert any(bool(gradient.abs().sum().gt(0)) for gradient in gradients)


def test_causal_differs_from_open_observational_twin() -> None:
    model = _small()
    inputs = _soft_fixture()
    causal = model(*inputs, mode="causal")
    open_twin = model(*inputs, mode="observational-twin")
    assert not torch.allclose(
        causal.transition_probabilities,
        open_twin.transition_probabilities,
    )
    assert not torch.allclose(
        causal.observer_probabilities,
        open_twin.observer_probabilities,
    )


def test_finite_machine_intervention_changes_repair_output() -> None:
    model = _small()
    transition, observer, transition_evidence, observer_evidence = (
        _one_hot_fixture()
    )
    baseline = model(
        transition,
        observer,
        transition_evidence,
        observer_evidence,
    )
    intervened = transition.clone()
    intervened[:, 1, 3] = F.one_hot(
        torch.tensor(6),
        PRIMARY_STATES,
    ).to(intervened.dtype)
    repaired = model(
        intervened,
        observer,
        transition_evidence,
        observer_evidence,
    )
    assert not torch.equal(
        baseline.transition_probabilities,
        repaired.transition_probabilities,
    )


@pytest.mark.parametrize("mode", sorted(COUNTERFACTUAL_REPAIR_MODES))
def test_complete_categorical_gauge_permutation_is_exact(mode: str) -> None:
    model = _small()
    inputs = _one_hot_fixture()
    baseline = model(*inputs, mode=mode)
    state_order = torch.tensor((3, 0, 7, 2, 5, 1, 6, 4))
    action_order = torch.tensor((2, 0, 1))
    observer_order = torch.tensor((1, 0))
    answer_order = torch.tensor((2, 0, 3, 1))
    transition, observer = _recode(
        inputs[0],
        inputs[1],
        state_order,
        action_order,
        observer_order,
        answer_order,
    )
    transition_evidence, observer_evidence = _recode(
        inputs[2],
        inputs[3],
        state_order,
        action_order,
        observer_order,
        answer_order,
    )
    recoded = model(
        transition,
        observer,
        transition_evidence,
        observer_evidence,
        mode=mode,
    )
    expected_transition, expected_observer = _recode(
        baseline.transition_probabilities,
        baseline.observer_probabilities,
        state_order,
        action_order,
        observer_order,
        answer_order,
    )
    assert torch.allclose(
        recoded.transition_probabilities,
        expected_transition,
        atol=2e-7,
        rtol=0.0,
    )
    assert torch.allclose(
        recoded.observer_probabilities,
        expected_observer,
        atol=2e-7,
        rtol=0.0,
    )
    expected_cycle_transition, expected_cycle_observer = _recode(
        baseline.cycle_transition_probabilities[0],
        baseline.cycle_observer_probabilities[0],
        state_order,
        action_order,
        observer_order,
        answer_order,
    )
    assert torch.allclose(
        recoded.cycle_transition_probabilities[0],
        expected_cycle_transition,
        atol=2e-7,
        rtol=0.0,
    )
    assert torch.allclose(
        recoded.cycle_observer_probabilities[0],
        expected_cycle_observer,
        atol=2e-7,
        rtol=0.0,
    )
    assert torch.allclose(
        recoded.cycle_halt_probabilities,
        baseline.cycle_halt_probabilities,
        atol=1e-7,
        rtol=0.0,
    )


def test_categorical_gauge_permutation_preserves_gradients() -> None:
    model = _small()
    state_order = torch.tensor((3, 0, 7, 2, 5, 1, 6, 4))
    action_order = torch.tensor((2, 0, 1))
    observer_order = torch.tensor((1, 0))
    answer_order = torch.tensor((2, 0, 3, 1))
    original_inputs = tuple(
        value.clone().requires_grad_() for value in _soft_fixture()
    )
    original = model(*original_inputs)
    original_loss = (
        original.transition_probabilities.square().sum()
        + original.observer_probabilities.square().sum()
        + original.cycle_halt_probabilities.square().sum()
    )
    parameters = tuple(model.parameters())
    original_gradients = torch.autograd.grad(
        original_loss,
        (*original_inputs, *parameters),
    )

    transition, observer = _recode(
        original_inputs[0].detach(),
        original_inputs[1].detach(),
        state_order,
        action_order,
        observer_order,
        answer_order,
    )
    transition_evidence, observer_evidence = _recode(
        original_inputs[2].detach(),
        original_inputs[3].detach(),
        state_order,
        action_order,
        observer_order,
        answer_order,
    )
    recoded_inputs = tuple(
        value.requires_grad_()
        for value in (
            transition,
            observer,
            transition_evidence,
            observer_evidence,
        )
    )
    recoded = model(*recoded_inputs)
    recoded_loss = (
        recoded.transition_probabilities.square().sum()
        + recoded.observer_probabilities.square().sum()
        + recoded.cycle_halt_probabilities.square().sum()
    )
    recoded_gradients = torch.autograd.grad(
        recoded_loss,
        (*recoded_inputs, *parameters),
    )
    expected_transition, expected_observer = _recode(
        original_gradients[0],
        original_gradients[1],
        state_order,
        action_order,
        observer_order,
        answer_order,
    )
    expected_transition_evidence, expected_observer_evidence = _recode(
        original_gradients[2],
        original_gradients[3],
        state_order,
        action_order,
        observer_order,
        answer_order,
    )
    for observed, expected in (
        (recoded_gradients[0], expected_transition),
        (recoded_gradients[1], expected_observer),
        (recoded_gradients[2], expected_transition_evidence),
        (recoded_gradients[3], expected_observer_evidence),
    ):
        assert torch.allclose(observed, expected, atol=2e-5, rtol=1e-5)
    assert all(
        torch.allclose(observed, expected, atol=2e-5, rtol=1e-5)
        for observed, expected in zip(
            recoded_gradients[4:],
            original_gradients[4:],
            strict=True,
        )
    )


def test_invalid_inputs_and_unknown_modes_fail_closed() -> None:
    model = _small()
    inputs = _soft_fixture()
    with pytest.raises(CounterfactualRepairError):
        model(*inputs, mode="oracle")
    invalid = inputs[0].clone()
    invalid[:, 0, 0, 0] = -1
    with pytest.raises(CounterfactualRepairError):
        model(invalid, inputs[1], inputs[2], inputs[3])
