from __future__ import annotations

import pytest
import torch

from endogenous_typed_theory_reactor import TheoryReactorError
from ettr_joint_stream import (
    ETTRJointPositionScheduler,
    ETTRJointScheduleConfig,
    GeneralLanguageStepConfig,
    GeneralLanguageUpdateStep,
)
from test_ettr_train_step import _trainer


@pytest.mark.parametrize(
    ("general_weight", "ettr_weight"),
    ((19, 1), (17, 3)),
)
def test_position_scheduler_is_deterministic_and_tracks_target(
    general_weight: int,
    ettr_weight: int,
) -> None:
    config = ETTRJointScheduleConfig(
        general_position_weight=general_weight,
        ettr_position_weight=ettr_weight,
    )
    first = ETTRJointPositionScheduler(config)
    second = ETTRJointPositionScheduler(config)
    streams: list[str] = []
    for _ in range(10_000):
        stream = first.select(
            general_positions=32_768,
            ettr_positions=8_448,
        )
        streams.append(stream)
        first.record(
            stream=stream,
            positions=32_768 if stream == "general" else 8_448,
        )
        repeated = second.select(
            general_positions=32_768,
            ettr_positions=8_448,
        )
        assert repeated == stream
        second.record(
            stream=repeated,
            positions=32_768 if repeated == "general" else 8_448,
        )
    assert first.receipt == second.receipt
    observed = (
        first.receipt.ettr_positions / first.receipt.total_positions
    )
    target = ettr_weight / (general_weight + ettr_weight)
    assert abs(observed - target) < 0.0001
    assert set(streams) == {"general", "ettr"}
    assert first.state_dict()["receipt"] == {
        "general_positions": first.receipt.general_positions,
        "ettr_positions": first.receipt.ettr_positions,
        "general_updates": first.receipt.general_updates,
        "ettr_updates": first.receipt.ettr_updates,
    }


def test_position_scheduler_fails_closed_on_unrecorded_or_wrong_charge() -> None:
    scheduler = ETTRJointPositionScheduler(
        ETTRJointScheduleConfig(19, 1)
    )
    stream = scheduler.select(
        general_positions=32_768,
        ettr_positions=8_448,
    )
    with pytest.raises(TheoryReactorError, match="recorded"):
        scheduler.select(
            general_positions=32_768,
            ettr_positions=8_448,
        )
    with pytest.raises(TheoryReactorError, match="differs"):
        scheduler.record(stream=stream, positions=1)
    with pytest.raises(TheoryReactorError, match="during an update"):
        scheduler.state_dict()


@pytest.mark.parametrize(
    "config",
    (
        ETTRJointScheduleConfig(0, 1),
        ETTRJointScheduleConfig(1, 0),
        ETTRJointScheduleConfig(-1, 1),
        ETTRJointScheduleConfig(True, 1),
    ),
)
def test_position_scheduler_rejects_invalid_weights(
    config: ETTRJointScheduleConfig,
) -> None:
    with pytest.raises(TheoryReactorError, match="weights"):
        ETTRJointPositionScheduler(config)


def test_general_update_changes_only_base_then_ettr_uses_same_optimizer() -> None:
    trainer, ettr_batch = _trainer(
        accumulation=1,
        warmup_updates=0,
        train_base=True,
    )
    language = GeneralLanguageUpdateStep(
        trainer.model,
        trainer.optimizer,
        step_config=GeneralLanguageStepConfig(
            gradient_accumulation_steps=1,
        ),
    )
    base_before = {
        name: value.detach().clone()
        for name, value in trainer.model.base.named_parameters()
    }
    architecture_before = {
        name: value.detach().clone()
        for name, value in trainer.model.named_parameters()
        if not name.startswith("base.")
    }
    inputs = torch.randint(0, 64, (2, 8), dtype=torch.long)
    targets = torch.randint(0, 64, (2, 8), dtype=torch.long)
    language_receipt = language.update(((inputs, targets),))
    assert language_receipt.optimizer_step == 1
    assert language_receipt.supervised_token_count == 16
    assert torch.isfinite(language_receipt.loss)
    assert torch.isfinite(language_receipt.gradient_norm)
    assert any(
        not torch.equal(base_before[name], value)
        for name, value in trainer.model.base.named_parameters()
    )
    for name, value in trainer.model.named_parameters():
        if not name.startswith("base."):
            assert torch.equal(architecture_before[name], value)

    ettr_receipt = trainer.update((ettr_batch,))
    assert ettr_receipt.optimizer_step == 2
    assert trainer.optimizer.next_update == 2
    assert torch.isfinite(ettr_receipt.total_loss)
    assert any(
        not torch.equal(architecture_before[name], value)
        for name, value in trainer.model.named_parameters()
        if not name.startswith("base.")
    )


def test_general_update_rejects_frozen_base_and_invalid_batch() -> None:
    trainer, _ = _trainer(
        accumulation=1,
        warmup_updates=0,
        train_base=False,
    )
    with pytest.raises(TheoryReactorError, match="trainable base"):
        GeneralLanguageUpdateStep(trainer.model, trainer.optimizer)

    joint, _ = _trainer(
        accumulation=1,
        warmup_updates=0,
        train_base=True,
    )
    language = GeneralLanguageUpdateStep(joint.model, joint.optimizer)
    inputs = torch.randint(0, 64, (2, 8), dtype=torch.long)
    with pytest.raises(TheoryReactorError, match="batch differs"):
        language.update(((inputs, inputs[:, :-1]),))
