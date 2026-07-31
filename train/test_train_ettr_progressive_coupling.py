from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from endogenous_typed_theory_reactor import TypedTheoryState
from ettr_objectives import ETTRCausalQueryPair
import train_ettr_progressive_coupling as progressive
from train_ettr_progressive_coupling import (
    ETTRProgressiveCouplingError,
    _distributed_environment,
    _distributed_mean,
    _distributed_parameter_sha256,
    _validate_args,
    deterministic_autonomous_choice,
    deterministic_exact_anchor_steps,
    factorial_delta_matching_loss,
    progressive_coupling_probability,
    reader_causal_binding_loss,
    select_state_source,
    select_trainable_architecture,
    truncate_state_credit,
)


class _CouplingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(3, 3)
        self.compiler = nn.Linear(3, 4)
        self.reactor = nn.Linear(4, 5)
        self.query_reader = nn.Linear(5, 6)


def _state(value: torch.Tensor, *, step: int) -> TypedTheoryState:
    batch = value.shape[0]
    return TypedTheoryState(
        value_probabilities=value,
        type_probabilities=value[..., :2],
        relations=value[:, None, :, :],
        active=value[..., 0],
        root=value[..., 1],
        committed=torch.zeros(batch, device=value.device),
        halted=torch.zeros(batch, device=value.device),
        step=step,
    )


def _arguments(tmp_path) -> SimpleNamespace:
    paths = {
        name: (tmp_path / name).resolve()
        for name in (
            "release",
            "data",
            "tokenizer",
            "protected",
            "checkpoint",
            "contract",
            "compiler",
            "reactor",
            "reader",
            "output",
        )
    }
    return SimpleNamespace(
        release_root=paths["release"],
        release_sha256="a" * 64,
        data_root=paths["data"],
        tokenizer=paths["tokenizer"],
        protected_checkpoint=paths["protected"],
        checkpoint=paths["checkpoint"],
        checkpoint_sha256="b" * 64,
        run_contract=paths["contract"],
        run_contract_sha256="c" * 64,
        initial_compiler=paths["compiler"],
        initial_compiler_sha256="d" * 64,
        initial_reactor=paths["reactor"],
        initial_reactor_sha256="e" * 64,
        initial_reader=paths["reader"],
        initial_reader_sha256="f" * 64,
        compiler_learning_rate=3e-4,
        reactor_learning_rate=1e-4,
        reader_learning_rate=1e-4,
        output=paths["output"],
        source_commit="1" * 40,
        architecture_seed=2,
        data_seed=3,
        coupling_seed=4,
        updates=1_000,
        start_position=20_000,
        warmup_updates=100,
        ramp_updates=700,
        counterfactual_delta_weight=2.0,
        exact_anchor_steps=4,
        credit_horizon=4,
        reader_causal_balance_mode="population",
        freeze_reader=False,
        profile_phase_timing=False,
        weight_decay=0.0,
        gradient_clip=1.0,
        eval_batches=4,
        log_every=10,
    )


def test_progressive_schedule_has_exact_ramp_and_autonomous_plateau() -> None:
    values = [
        progressive_coupling_probability(
            update,
            warmup_updates=100,
            ramp_updates=700,
        )
        for update in (1, 100, 101, 450, 800, 1_000)
    ]
    assert values[0] == values[1] == values[2] == 0.0
    assert values[3] == pytest.approx(349 / 699)
    assert values[4] == values[5] == 1.0


def test_factorial_delta_matching_rewards_effects_and_invariance() -> None:
    rows = torch.arange(8).reshape(2, 2, 2)
    target = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    exact = target.clone().requires_grad_(True)
    exact_loss, exact_support = factorial_delta_matching_loss(
        exact,
        target,
        rows,
    )
    assert float(exact_support) == 1.0
    assert float(exact_loss.detach()) == pytest.approx(0.0)

    collapsed = torch.zeros_like(target, requires_grad=True)
    collapsed_loss, collapsed_support = factorial_delta_matching_loss(
        collapsed,
        target,
        rows,
    )
    assert float(collapsed_support) == 1.0
    assert float(collapsed_loss.detach()) > 0.0
    collapsed_loss.backward()
    assert collapsed.grad is not None
    assert float(collapsed.grad.abs().sum()) > 0.0

    noninvariant = target.clone()
    noninvariant[1, 1] = 1.0
    noninvariant_loss, _support = factorial_delta_matching_loss(
        noninvariant,
        target,
        rows,
    )
    assert float(noninvariant_loss) > 0.0


def test_factorial_delta_matching_respects_row_support() -> None:
    rows = torch.arange(4).reshape(1, 2, 2)
    prediction = torch.zeros(4, 2)
    target = torch.zeros_like(prediction)
    loss, support = factorial_delta_matching_loss(
        prediction,
        target,
        rows,
        row_mask=torch.zeros(4, dtype=torch.bool),
    )
    assert float(loss) == 0.0
    assert float(support) == 0.0
    with pytest.raises(
        ETTRProgressiveCouplingError,
        match="delta geometry differs",
    ):
        factorial_delta_matching_loss(
            prediction,
            target,
            rows,
            row_mask=torch.ones(4),
        )


def test_reader_factor_balance_preserves_rare_causal_contrast() -> None:
    correct_logits = torch.zeros(8, 2, requires_grad=True)
    foil_logits = torch.zeros(8, 2, requires_grad=True)
    correct_target = torch.zeros(8, dtype=torch.long)
    foil_target = correct_target.clone()
    foil_target[0] = 1
    pair = ETTRCausalQueryPair(
        correct_logits=correct_logits,
        foil_logits=foil_logits,
        correct_target=correct_target,
        foil_target=foil_target,
    )
    population = reader_causal_binding_loss(
        pair,
        margin=1.0,
        balance_mode="population",
    )
    factor = reader_causal_binding_loss(
        pair,
        margin=1.0,
        balance_mode="factor",
    )
    assert float(factor.detach()) > float(population.detach())

    population.backward(retain_graph=True)
    population_effect_gradient = float(
        correct_logits.grad[0].abs().sum()
    )
    correct_logits.grad.zero_()
    foil_logits.grad.zero_()
    factor.backward()
    factor_effect_gradient = float(correct_logits.grad[0].abs().sum())
    assert factor_effect_gradient > population_effect_gradient


def test_reader_factor_balance_rejects_unknown_mode() -> None:
    pair = ETTRCausalQueryPair(
        correct_logits=torch.zeros(1, 2),
        foil_logits=torch.zeros(1, 2),
        correct_target=torch.zeros(1, dtype=torch.long),
        foil_target=torch.ones(1, dtype=torch.long),
    )
    with pytest.raises(
        ETTRProgressiveCouplingError,
        match="causal balance mode differs",
    ):
        reader_causal_binding_loss(
            pair,
            margin=1.0,
            balance_mode="unknown",
        )


def test_progressive_schedule_rejects_invalid_inputs() -> None:
    with pytest.raises(
        ETTRProgressiveCouplingError,
        match="schedule differs",
    ):
        progressive_coupling_probability(
            0,
            warmup_updates=0,
            ramp_updates=1,
        )


def test_coupling_choice_is_deterministic_and_has_hard_endpoints() -> None:
    assert not deterministic_autonomous_choice(
        0.0,
        coupling_seed=7,
        update=9,
        stage=2,
    )
    assert deterministic_autonomous_choice(
        1.0,
        coupling_seed=7,
        update=9,
        stage=2,
    )
    first = deterministic_autonomous_choice(
        0.37,
        coupling_seed=7,
        update=9,
        stage=2,
    )
    assert first is deterministic_autonomous_choice(
        0.37,
        coupling_seed=7,
        update=9,
        stage=2,
    )


def test_exact_anchor_steps_are_deterministic_spread_and_rotating() -> None:
    first = deterministic_exact_anchor_steps(
        tuple(range(64)),
        4,
        coupling_seed=7,
        update=9,
    )
    assert first == deterministic_exact_anchor_steps(
        tuple(range(64)),
        4,
        coupling_seed=7,
        update=9,
    )
    assert len(first) == 4
    cyclic_gaps = [
        (first[(index + 1) % len(first)] - first[index]) % 64
        for index in range(len(first))
    ]
    assert cyclic_gaps == [16, 16, 16, 16]
    covered = {
        step
        for update in range(1, 257)
        for step in deterministic_exact_anchor_steps(
            tuple(range(64)),
            4,
            coupling_seed=7,
            update=update,
        )
    }
    assert covered == set(range(64))


def test_exact_anchor_steps_reject_invalid_contract() -> None:
    with pytest.raises(
        ETTRProgressiveCouplingError,
        match="anchor schedule differs",
    ):
        deterministic_exact_anchor_steps(
            tuple(range(64)),
            65,
            coupling_seed=7,
            update=1,
        )


def test_exact_anchor_steps_use_only_supported_stages() -> None:
    eligible = (0, 1, 4, 9, 12, 18, 31)
    selected = deterministic_exact_anchor_steps(
        eligible,
        4,
        coupling_seed=13,
        update=5,
    )
    assert len(selected) == 4
    assert set(selected) <= set(eligible)


def test_credit_horizon_detaches_without_changing_state() -> None:
    values = torch.ones(2, 2, 2, requires_grad=True)
    state = _state(values, step=4)
    retained, truncated = truncate_state_credit(
        state,
        completed_steps=3,
        total_steps=8,
        credit_horizon=4,
        use_autonomous=True,
    )
    assert retained is state
    assert not truncated

    detached, truncated = truncate_state_credit(
        state,
        completed_steps=4,
        total_steps=8,
        credit_horizon=4,
        use_autonomous=True,
    )
    assert truncated
    assert detached is not state
    assert torch.equal(
        detached.value_probabilities,
        state.value_probabilities,
    )
    assert not detached.value_probabilities.requires_grad

    final, truncated = truncate_state_credit(
        state,
        completed_steps=8,
        total_steps=8,
        credit_horizon=4,
        use_autonomous=True,
    )
    assert final is state
    assert not truncated


def test_state_source_selects_complete_batch_and_preserves_gradient() -> None:
    exact_values = torch.zeros(2, 2, 2)
    autonomous_values = torch.ones(2, 2, 2, requires_grad=True)
    exact = _state(exact_values, step=3)
    autonomous = _state(autonomous_values, step=3)
    selected = select_state_source(
        exact,
        autonomous,
        use_autonomous=True,
    )
    assert selected is autonomous
    selected.value_probabilities.sum().backward()
    assert autonomous_values.grad is not None
    assert torch.equal(
        autonomous_values.grad,
        torch.ones_like(autonomous_values),
    )
    assert (
        select_state_source(
            exact,
            autonomous,
            use_autonomous=False,
        )
        is exact
    )


def test_state_source_rejects_step_or_geometry_mismatch() -> None:
    exact = _state(torch.zeros(2, 2, 2), step=2)
    wrong_step = _state(torch.ones(2, 2, 2), step=3)
    with pytest.raises(
        ETTRProgressiveCouplingError,
        match="state source differs",
    ):
        select_state_source(
            exact,
            wrong_step,
            use_autonomous=True,
        )
    wrong_shape = _state(torch.ones(3, 2, 2), step=2)
    with pytest.raises(
        ETTRProgressiveCouplingError,
        match="state geometry differs",
    ):
        select_state_source(
            exact,
            wrong_shape,
            use_autonomous=True,
        )


def test_progressive_ownership_freezes_only_base() -> None:
    model = _CouplingModel()
    receipt = select_trainable_architecture(model)
    assert all(
        not parameter.requires_grad
        for parameter in model.base.parameters()
    )
    expected = {
        id(parameter)
        for module in (
            model.compiler,
            model.reactor,
            model.query_reader,
        )
        for parameter in module.parameters()
    }
    observed = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    assert observed == expected
    assert receipt["frozen_base"] is True
    assert receipt["frozen_reader_anchor"] is False
    assert receipt["trainable_parameters"] == 16 + 25 + 36


def test_progressive_ownership_can_freeze_reader_as_semantic_anchor() -> None:
    model = _CouplingModel()
    receipt = select_trainable_architecture(
        model,
        freeze_reader=True,
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.query_reader.parameters()
    )
    expected = {
        id(parameter)
        for module in (model.compiler, model.reactor)
        for parameter in module.parameters()
    }
    observed = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    assert observed == expected
    assert receipt["frozen_reader_anchor"] is True
    assert receipt["trainable_parameters"] == 16 + 25
    assert receipt["trainable_component_parameters"]["reader"] == 0


def test_frozen_reader_anchor_still_routes_gradient_to_upstream_state() -> None:
    model = _CouplingModel()
    select_trainable_architecture(model, freeze_reader=True)
    value = model.compiler(torch.ones(2, 3))
    value = model.reactor(value)
    loss = model.query_reader(value).square().mean()
    loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in model.compiler.parameters()
    )
    assert all(
        parameter.grad is not None
        for parameter in model.reactor.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.query_reader.parameters()
    )


def test_progressive_arguments_bind_schedule_hashes_and_absolute_paths(
    tmp_path,
) -> None:
    arguments = _arguments(tmp_path)
    _validate_args(arguments)
    arguments.reader_causal_balance_mode = "unknown"
    with pytest.raises(
        ETTRProgressiveCouplingError,
        match="arguments differ",
    ):
        _validate_args(arguments)
    arguments.reader_causal_balance_mode = "population"
    arguments.ramp_updates = 901
    with pytest.raises(
        ETTRProgressiveCouplingError,
        match="arguments differ",
    ):
        _validate_args(arguments)


def test_distributed_environment_binds_rank_and_local_device(
    monkeypatch,
) -> None:
    values = {"RANK": "3", "WORLD_SIZE": "4", "LOCAL_RANK": "1"}
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    calls = []
    monkeypatch.setattr(progressive.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        progressive.torch.cuda,
        "set_device",
        lambda value: calls.append(("device", value)),
    )
    monkeypatch.setattr(
        progressive.dist,
        "init_process_group",
        lambda **kwargs: calls.append(("init", kwargs)),
    )
    monkeypatch.setattr(progressive.dist, "get_rank", lambda: 3)
    monkeypatch.setattr(progressive.dist, "get_world_size", lambda: 4)
    assert _distributed_environment() == (3, 4, 1)
    assert calls[0] == ("device", 1)
    assert calls[1][0] == "init"
    assert calls[1][1]["backend"] == "nccl"


def test_distributed_mean_uses_sum_then_world_average(monkeypatch) -> None:
    def reduce_sum(value: torch.Tensor, world_size: int) -> None:
        assert world_size == 4
        value.add_(torch.tensor(14.0))

    monkeypatch.setattr(progressive, "_all_reduce_sum", reduce_sum)
    assert _distributed_mean(
        torch.tensor(2.0),
        device=torch.device("cpu"),
        world_size=4,
    ) == pytest.approx(4.0)


def test_distributed_parameter_identity_rejects_rank_drift(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        progressive,
        "_parameter_sha256",
        lambda _model: "a" * 64,
    )

    def gather(values, _digest) -> None:
        values[:] = ["a" * 64, "b" * 64]

    monkeypatch.setattr(progressive.dist, "all_gather_object", gather)
    with pytest.raises(
        ETTRProgressiveCouplingError,
        match="parameter identity differs",
    ):
        _distributed_parameter_sha256(
            _CouplingModel(),
            rank=0,
            world_size=2,
        )
