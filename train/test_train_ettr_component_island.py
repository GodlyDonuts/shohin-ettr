from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from endogenous_typed_theory_reactor import (
    GenericTransactionReactor,
    TheoryReactorConfig,
    TypedTheoryState,
)
from ettr_objectives import ETTRPacketTargets
from train_ettr_component_island import (
    ETTRComponentIslandError,
    _balanced_binary_nll,
    _masked_categorical_cross_entropy,
    _masked_class_balanced_cross_entropy,
    _masked_categorical_nll,
    _reactor_policy_logits,
    compiler_packet_loss,
    _validate_args,
    load_component_warm_start,
    select_trainable_component,
)
from safetensors.torch import save_file


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


@pytest.mark.parametrize(
    "reader_injection",
    ("late", "postnorm", "postnorm-scaled"),
)
def test_nonstage_reader_injection_is_reader_only(
    reader_injection: str,
) -> None:
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
            "reader_injection": reader_injection,
        },
    )()
    with pytest.raises(
        ETTRComponentIslandError,
        match="arguments differ",
    ):
        _validate_args(arguments)


@pytest.mark.parametrize(
    "reader_injection",
    ("stage", "late", "postnorm", "postnorm-scaled"),
)
def test_reader_injection_geometries_are_accepted(
    reader_injection: str,
) -> None:
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
            "component": "reader",
            "reader_injection": reader_injection,
        },
    )()
    _validate_args(arguments)


def test_component_warm_start_is_hash_bound(tmp_path) -> None:
    model = _ComponentModel()
    path = (tmp_path / "reader.safetensors").resolve()
    expected = {
        name: tensor.detach().clone()
        for name, tensor in model.query_reader.state_dict().items()
    }
    save_file(expected, path)
    import hashlib

    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    with torch.no_grad():
        for parameter in model.query_reader.parameters():
            parameter.zero_()
    observed = load_component_warm_start(
        model,
        "reader",
        path,
        expected_sha256=sha256,
    )
    assert observed == sha256
    for name, tensor in model.query_reader.state_dict().items():
        assert torch.equal(tensor, expected[name])


def test_component_warm_start_rejects_wrong_hash(tmp_path) -> None:
    model = _ComponentModel()
    path = (tmp_path / "reader.safetensors").resolve()
    save_file(model.query_reader.state_dict(), path)
    with pytest.raises(
        ETTRComponentIslandError,
        match="hash differs",
    ):
        load_component_warm_start(
            model,
            "reader",
            path,
            expected_sha256="0" * 64,
        )


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


def test_masked_categorical_cross_entropy_recovers_improbable_class() -> None:
    logits = torch.tensor(
        [[-100.0, 0.0], [0.0, -100.0]],
        requires_grad=True,
    )
    loss = _masked_categorical_cross_entropy(
        logits,
        torch.tensor([0, 1]),
        torch.tensor([True, False]),
    )
    assert loss is not None
    loss.backward()
    assert loss.item() == pytest.approx(100.0)
    assert logits.grad is not None
    assert logits.grad[0, 0].item() == pytest.approx(-1.0)
    assert logits.grad[0, 1].item() == pytest.approx(1.0)
    assert torch.equal(logits.grad[1], torch.zeros(2))


def test_masked_class_balanced_cross_entropy_equalizes_target_classes() -> None:
    logits = torch.tensor(
        [
            [4.0, 0.0],
            [4.0, 0.0],
            [4.0, 0.0],
            [4.0, 0.0],
        ],
        requires_grad=True,
    )
    targets = torch.tensor([0, 0, 0, 1])
    mask = torch.ones(4, dtype=torch.bool)
    loss = _masked_class_balanced_cross_entropy(logits, targets, mask)
    assert loss is not None
    row_losses = F.cross_entropy(logits, targets, reduction="none")
    expected = 0.5 * (row_losses[:3].mean() + row_losses[3])
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_reactor_policy_logit_capture_matches_frozen_policy() -> None:
    config = TheoryReactorConfig(
        d_model=8,
        state_width=8,
        num_slots=3,
        num_types=2,
        num_relations=2,
        num_value_codes=4,
        max_edges=18,
        num_heads=2,
        compiler_layers=1,
        reactor_layers=1,
        query_layers=1,
        ff_multiplier=2,
        max_steps=2,
        stage_after_block=0,
        parameter_cap=1_000_000,
    )
    reactor = GenericTransactionReactor(config)
    active = torch.ones(2, config.num_slots)
    root = torch.zeros_like(active)
    root[:, 0] = 1.0
    values = torch.zeros(2, config.num_slots, config.num_value_codes)
    values[..., 0] = 1.0
    types = torch.zeros(2, config.num_slots, config.num_types)
    types[..., 0] = 1.0
    state = TypedTheoryState(
        value_probabilities=values,
        type_probabilities=types,
        relations=torch.zeros(
            2,
            config.num_relations,
            config.num_slots,
            config.num_slots,
        ),
        active=active,
        root=root,
        committed=torch.zeros(2),
        halted=torch.zeros(2),
        step=0,
    )
    policy, logits = _reactor_policy_logits(
        reactor,
        state,
        command_hidden=torch.randn(2, 4, config.d_model),
        command_attention_mask=torch.ones(2, 4, dtype=torch.bool),
    )
    probabilities = {
        "opcode": policy.opcode_probabilities,
        "source": policy.source_probabilities,
        "target": policy.target_probabilities,
        "relation": policy.relation_probabilities,
        "type_index": policy.type_probabilities,
        "value_code": policy.value_probabilities,
    }
    for name, field_logits in logits.items():
        assert torch.equal(field_logits.softmax(-1), probabilities[name])
    loss = torch.stack(
        [
            _masked_categorical_cross_entropy(
                field_logits,
                torch.zeros(2, dtype=torch.long),
                torch.ones(2, dtype=torch.bool),
            )
            for field_logits in logits.values()
        ]
    ).mean()
    loss.backward()
    assert reactor.target_query.weight.grad is not None
    assert torch.isfinite(reactor.target_query.weight.grad).all()
    assert torch.count_nonzero(reactor.target_query.weight.grad)


def test_reactor_policy_logit_capture_can_apply_hard_policy() -> None:
    config = TheoryReactorConfig(
        d_model=8,
        state_width=8,
        num_slots=3,
        num_types=2,
        num_relations=2,
        num_value_codes=4,
        max_edges=18,
        num_heads=2,
        compiler_layers=1,
        reactor_layers=1,
        query_layers=1,
        ff_multiplier=2,
        max_steps=2,
        stage_after_block=0,
        parameter_cap=1_000_000,
    )
    reactor = GenericTransactionReactor(config)
    active = torch.ones(2, config.num_slots)
    root = torch.zeros_like(active)
    root[:, 0] = 1.0
    values = torch.zeros(2, config.num_slots, config.num_value_codes)
    values[..., 0] = 1.0
    types = torch.zeros(2, config.num_slots, config.num_types)
    types[..., 0] = 1.0
    state = TypedTheoryState(
        value_probabilities=values,
        type_probabilities=types,
        relations=torch.zeros(
            2,
            config.num_relations,
            config.num_slots,
            config.num_slots,
        ),
        active=active,
        root=root,
        committed=torch.zeros(2),
        halted=torch.zeros(2),
        step=0,
    )
    policy, logits = _reactor_policy_logits(
        reactor,
        state,
        command_hidden=torch.randn(2, 4, config.d_model),
        command_attention_mask=torch.ones(2, 4, dtype=torch.bool),
        hard=True,
    )
    for name in (
        "opcode",
        "source",
        "target",
        "relation",
        "type_index",
        "value_code",
    ):
        applied = getattr(policy, name)
        assert torch.equal(applied.sum(-1), torch.ones(2))
        assert torch.equal(
            applied.argmax(-1),
            logits[name].argmax(-1),
        )


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
