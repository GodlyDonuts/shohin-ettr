from __future__ import annotations

import torch

from train_algebraic_state_semantic_pilot import (
    _set_active_owner,
    _set_training_ownership,
)


class _StateModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(3, 3)
        self.compiler = torch.nn.Linear(3, 3)
        self.reactor = torch.nn.Linear(3, 3)


def test_causal_owner_switches_exclude_cross_factor_parameters() -> None:
    model = _StateModel()
    reader = torch.nn.Linear(3, 2)
    trainable, count = _set_training_ownership(model, reader)
    assert count == sum(
        parameter.numel()
        for module in (model.compiler, model.reactor)
        for parameter in module.parameters()
    )
    assert set(trainable) == {
        parameter
        for module in (model.compiler, model.reactor)
        for parameter in module.parameters()
    }
    assert not any(parameter.requires_grad for parameter in model.base.parameters())
    assert not any(parameter.requires_grad for parameter in reader.parameters())

    _set_active_owner(model, "compiler")
    assert all(parameter.requires_grad for parameter in model.compiler.parameters())
    assert not any(parameter.requires_grad for parameter in model.reactor.parameters())

    _set_active_owner(model, "reactor")
    assert not any(parameter.requires_grad for parameter in model.compiler.parameters())
    assert all(parameter.requires_grad for parameter in model.reactor.parameters())

    _set_active_owner(model, None)
    assert all(parameter.requires_grad for parameter in model.compiler.parameters())
    assert all(parameter.requires_grad for parameter in model.reactor.parameters())
