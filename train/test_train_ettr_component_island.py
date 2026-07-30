from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from ettr_objectives import ETTRPacketTargets
from train_ettr_component_island import (
    ETTRComponentIslandError,
    _balanced_binary_nll,
    _masked_categorical_nll,
    compiler_packet_loss,
    _validate_args,
    select_trainable_component,
)


class _ComponentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(3, 3)
        self.compiler = nn.Linear(3, 3)
        self.reactor = nn.Linear(3, 3)
        self.query_reader = nn.Linear(3, 3)


@pytest.mark.parametrize(
    ("component", "attribute"),
    (
        ("compiler", "compiler"),
        ("reactor", "reactor"),
        ("reader", "query_reader"),
    ),
)
def test_component_selection_freezes_every_other_parameter(
    component: str,
    attribute: str,
) -> None:
    model = _ComponentModel()
    receipt = select_trainable_component(model, component)
    expected = {
        id(parameter)
        for parameter in getattr(model, attribute).parameters()
    }
    observed = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    assert observed == expected
    assert receipt["trainable_parameters"] == 12
    assert receipt["frozen_base"] is True


def test_component_selection_rejects_unknown_island() -> None:
    with pytest.raises(ETTRComponentIslandError, match="unknown"):
        select_trainable_component(_ComponentModel(), "joint")


def test_late_reader_injection_is_reader_only() -> None:
    arguments = type(
        "Args",
        (),
        {
            "release_sha256": "a" * 64,
            "checkpoint_sha256": "b" * 64,
            "run_contract_sha256": "c" * 64,
            "source_commit": "d" * 40,
            "architecture_seed": 1,
            "data_seed": 2,
            "updates": 1,
            "eval_batches": 2,
            "log_every": 1,
            "learning_rate": 3e-4,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "component": "compiler",
            "reader_injection": "late",
        },
    )()
    with pytest.raises(
        ETTRComponentIslandError,
        match="arguments differ",
    ):
        _validate_args(arguments)


def test_masked_categorical_nll_uses_only_supported_rows() -> None:
    probabilities = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
    targets = torch.tensor([0, 0])
    loss = _masked_categorical_nll(
        probabilities,
        targets,
        torch.tensor([True, False]),
    )
    assert loss is not None
    assert loss.item() == pytest.approx(-torch.log(torch.tensor(0.8)).item())


def test_balanced_binary_nll_does_not_let_negatives_swamp_positives() -> None:
    probabilities = torch.tensor([0.9, 0.9, 0.9, 0.9])
    targets = torch.tensor([True, False, False, False])
    loss = _balanced_binary_nll(
        probabilities,
        targets,
        torch.ones(4, dtype=torch.bool),
    )
    expected = 0.5 * (
        -torch.log(torch.tensor(0.9))
        - torch.log(torch.tensor(0.1))
    )
    assert loss is not None
    assert loss.item() == pytest.approx(expected.item())


def test_compiler_packet_loss_is_finite_and_backpropagates() -> None:
    active_logits = torch.tensor([[2.0, -2.0]], requires_grad=True)
    active = active_logits.sigmoid()
    value_logits = torch.tensor(
        [[[2.0, 0.0], [0.0, 2.0]]],
        requires_grad=True,
    )
    type_logits = torch.tensor(
        [[[2.0, 0.0], [0.0, 2.0]]],
        requires_grad=True,
    )
    relation_logits = torch.tensor(
        [[[[2.0, -2.0], [-2.0, -2.0]]]],
        requires_grad=True,
    )
    root_logits = torch.tensor([[2.0, -2.0]], requires_grad=True)
    prediction = type(
        "State",
        (),
        {
            "value_probabilities": value_logits.softmax(-1),
            "type_probabilities": type_logits.softmax(-1),
            "relations": relation_logits.sigmoid(),
            "active": active,
            "root": root_logits.softmax(-1),
            "committed": torch.tensor([0.1], requires_grad=True),
            "halted": torch.tensor([0.1], requires_grad=True),
        },
    )()
    targets = ETTRPacketTargets(
        value_code=torch.tensor([[0, 0]]),
        type_index=torch.tensor([[0, 0]]),
        relations=torch.tensor([[[[True, False], [False, False]]]]),
        active=torch.tensor([[True, False]]),
        root=torch.tensor([[True, False]]),
        committed=torch.tensor([False]),
        halted=torch.tensor([False]),
        slot_mask=torch.tensor([[True, True]]),
        relation_mask=torch.ones(1, 1, 2, 2, dtype=torch.bool),
    )
    loss, parts = compiler_packet_loss(prediction, targets)
    assert torch.isfinite(loss)
    assert set(parts) == {
        "active",
        "committed",
        "halted",
        "relations",
        "root",
        "type_index",
        "value_code",
    }
    loss.backward()
    assert active_logits.grad is not None
    assert relation_logits.grad is not None
