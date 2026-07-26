from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from endogenous_typed_theory_reactor import TheoryReactorError
from ettr_data_contract import ETTRContinuationBatch
from ettr_objectives import ETTRObjectiveConfig
from ettr_optimization import (
    ETTROptimizerBundle,
    ETTROptimizerConfig,
)
from ettr_train_step import ETTRTrainStep, ETTRTrainStepConfig
from test_ettr_data_contract import (
    _alignment,
    _packet,
    _transactions,
)
from test_ettr_episode import _batch, _runner


MANIFEST_SHA256 = "a" * 64
DATASET_SHA256 = "b" * 64


def _trainer(
    *,
    accumulation: int,
) -> tuple[ETTRTrainStep, ETTRContinuationBatch]:
    model = _runner().model
    optimizer = ETTROptimizerBundle(
        model,
        ETTROptimizerConfig(
            train_base=False,
            warmup_updates=1,
            total_updates=10,
        ),
    )
    objective = ETTRObjectiveConfig(
        vocab_size=64,
        num_slots=6,
        num_types=3,
        num_relations=3,
        num_value_codes=64,
        active_slot_budget=6,
        relation_edge_budget=96,
    )
    trainer = ETTRTrainStep(
        model,
        optimizer,
        objective,
        manifest_sha256=MANIFEST_SHA256,
        dataset_sha256=DATASET_SHA256,
        step_config=ETTRTrainStepConfig(
            gradient_accumulation_steps=accumulation,
        ),
    )
    batch = ETTRContinuationBatch(
        manifest_sha256=MANIFEST_SHA256,
        dataset_sha256=DATASET_SHA256,
        episodes=_batch(2),
        packet_targets=_packet(2),
        transaction_targets=_transactions(2),
        initial_committed=torch.zeros(2, dtype=torch.bool),
        initial_halted=torch.zeros(2, dtype=torch.bool),
        equivariance=_alignment(),
    )
    return trainer, batch


def test_update_runs_complete_native_objective_and_advances_cursor() -> None:
    trainer, batch = _trainer(accumulation=2)
    before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.compiler.named_parameters()
    }
    receipt = trainer.update((batch, batch))
    assert receipt.optimizer_step == 1
    assert receipt.learning_rate_scale == 0
    assert receipt.supervised_token_count > 0
    for name in (
        "total_loss",
        "token_lm_loss",
        "packet_loss",
        "transaction_loss",
        "equivariance_loss",
        "commit_halt_loss",
        "sparsity_loss",
        "anti_bypass_loss",
        "gradient_norm",
    ):
        value = getattr(receipt, name)
        assert value.shape == ()
        assert torch.isfinite(value)
    # Warmup update zero is intentional; the next scheduled update moves.
    second = trainer.update((batch, batch))
    assert second.optimizer_step == 2
    assert second.learning_rate_scale == 1
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in trainer.model.compiler.named_parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in trainer.model.base.parameters()
    )


def test_wrong_accumulation_window_fails_before_mutation() -> None:
    trainer, batch = _trainer(accumulation=2)
    before = {
        name: tensor.detach().clone()
        for name, tensor in trainer.model.state_dict().items()
    }
    with pytest.raises(TheoryReactorError, match="accumulation"):
        trainer.update((batch,))
    assert trainer.optimizer.next_update == 0
    for name, tensor in trainer.model.state_dict().items():
        assert torch.equal(before[name], tensor)


def test_invalid_batch_fails_before_optimizer_mutation() -> None:
    trainer, batch = _trainer(accumulation=1)
    before_lrs = tuple(
        group["lr"]
        for optimizer in (trainer.optimizer.muon, trainer.optimizer.adam)
        if optimizer is not None
        for group in optimizer.param_groups
    )
    invalid = replace(
        batch,
        initial_committed=torch.zeros(1, dtype=torch.bool),
    )
    with pytest.raises(TheoryReactorError, match="geometry"):
        trainer.update((invalid,))
    after_lrs = tuple(
        group["lr"]
        for optimizer in (trainer.optimizer.muon, trainer.optimizer.adam)
        if optimizer is not None
        for group in optimizer.param_groups
    )
    assert after_lrs == before_lrs
    assert trainer.optimizer.next_update == 0


def test_train_step_rejects_optimizer_from_equal_shape_model() -> None:
    first = _runner().model
    second = _runner().model
    optimizer = ETTROptimizerBundle(
        first,
        ETTROptimizerConfig(
            train_base=False,
            warmup_updates=1,
            total_updates=10,
        ),
    )
    objective = ETTRObjectiveConfig(
        vocab_size=64,
        num_slots=6,
        num_types=3,
        num_relations=3,
        num_value_codes=64,
        active_slot_budget=6,
        relation_edge_budget=96,
    )
    with pytest.raises(TheoryReactorError, match="not bound"):
        ETTRTrainStep(
            second,
            optimizer,
            objective,
            manifest_sha256=MANIFEST_SHA256,
            dataset_sha256=DATASET_SHA256,
        )


def test_train_step_rejects_batch_from_another_snapshot() -> None:
    trainer, batch = _trainer(accumulation=1)
    wrong = replace(batch, dataset_sha256="c" * 64)
    with pytest.raises(TheoryReactorError, match="snapshot differs"):
        trainer.update((wrong,))
