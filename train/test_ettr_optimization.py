from __future__ import annotations

from dataclasses import replace

import pytest

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TheoryReactorConfig,
    TheoryReactorError,
)
from ettr_optimization import (
    ETTROptimizerBundle,
    ETTROptimizerConfig,
)
from model import GPT, GPTConfig


def _model() -> EndogenousTypedTheoryReactorGPT:
    base = GPT(
        GPTConfig(
            vocab_size=64,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            d_model=32,
            d_ff=64,
            seq_len=24,
        )
    )
    return EndogenousTypedTheoryReactorGPT(
        base,
        TheoryReactorConfig(
            d_model=32,
            state_width=32,
            num_slots=6,
            num_types=3,
            num_relations=3,
            num_heads=4,
            compiler_layers=1,
            reactor_layers=1,
            query_layers=1,
            ff_multiplier=2,
            max_steps=6,
            stage_after_block=1,
            parameter_cap=1_000_000,
        ),
    )


def _config(**changes: object) -> ETTROptimizerConfig:
    return replace(
        ETTROptimizerConfig(
            warmup_updates=10,
            total_updates=100,
        ),
        **changes,
    )


def test_optimizer_groups_are_disjoint_and_reconcile() -> None:
    model = _model()
    bundle = ETTROptimizerBundle(model, _config())
    parameter_ids = []
    if bundle.muon is not None:
        parameter_ids.extend(
            id(parameter)
            for group in bundle.muon.param_groups
            for parameter in group["params"]
        )
    parameter_ids.extend(
        id(parameter)
        for group in bundle.adam.param_groups
        for parameter in group["params"]
    )
    assert len(parameter_ids) == len(set(parameter_ids))
    assert bundle.receipt.unique_trainable_parameters == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert bundle.receipt.complete_system_parameters == (
        model.parameter_receipt().complete_system_parameters
    )


def test_frozen_base_optimizer_contains_only_architecture() -> None:
    model = _model()
    bundle = ETTROptimizerBundle(
        model,
        _config(train_base=False),
    )
    assert bundle.receipt.base_muon_parameters == 0
    assert bundle.receipt.base_adam_parameters == 0
    assert bundle.receipt.unique_trainable_parameters == (
        model.parameter_receipt().architecture_parameters
    )
    assert not any(parameter.requires_grad for parameter in model.base.parameters())


def test_schedule_and_state_resume_are_exact() -> None:
    model = _model()
    config = _config()
    first = ETTROptimizerBundle(model, config)
    assert first.apply_schedule(0) == 0
    assert first.apply_schedule(10) == 1
    first.next_update = 95
    expected_scale = first.apply_schedule()
    state = first.state_dict()

    second = ETTROptimizerBundle(_model(), config)
    second.load_state_dict(state)
    assert second.next_update == 95
    assert second.apply_schedule() == expected_scale
    assert [
        (group["ettr_group"], group["lr"]) for group in second.adam.param_groups
    ] == [(group["ettr_group"], group["lr"]) for group in first.adam.param_groups]


def test_optimizer_resume_rejects_contract_drift() -> None:
    first = ETTROptimizerBundle(_model(), _config())
    state = first.state_dict()
    second = ETTROptimizerBundle(
        _model(),
        _config(architecture_lr_adam=0.002),
    )
    with pytest.raises(TheoryReactorError, match="contract"):
        second.load_state_dict(state)
