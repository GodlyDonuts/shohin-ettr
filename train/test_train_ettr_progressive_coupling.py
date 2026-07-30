from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from endogenous_typed_theory_reactor import TypedTheoryState
from train_ettr_progressive_coupling import (
    ETTRProgressiveCouplingError,
    _validate_args,
    deterministic_autonomous_choice,
    progressive_coupling_probability,
    select_state_source,
    select_trainable_architecture,
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
    assert receipt["trainable_parameters"] == 16 + 25 + 36


def test_progressive_arguments_bind_schedule_hashes_and_absolute_paths(
    tmp_path,
) -> None:
    arguments = _arguments(tmp_path)
    _validate_args(arguments)
    arguments.ramp_updates = 901
    with pytest.raises(
        ETTRProgressiveCouplingError,
        match="arguments differ",
    ):
        _validate_args(arguments)
