from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from train_ettr_disposition_motor import (
    DispositionMotor,
    ETTRDispositionMotorError,
    _gates,
    _query_classes,
    _validate_args,
)


def _arguments(tmp_path):
    paths = {
        name: (tmp_path / name).resolve()
        for name in (
            "release",
            "data",
            "tokenizer",
            "protected",
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
        run_contract=paths["contract"],
        run_contract_sha256="b" * 64,
        compiler=paths["compiler"],
        compiler_sha256="c" * 64,
        reactor=paths["reactor"],
        reactor_sha256="d" * 64,
        query_reader=paths["reader"],
        query_reader_sha256="e" * 64,
        output=paths["output"],
        source_commit="f" * 40,
        architecture_seed=1,
        data_seed=2,
        motor_seed=3,
        negative_token_id=28,
        positive_token_id=29,
        hidden=8,
        updates=10,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clip=1.0,
        eval_batches=4,
        log_every=2,
    )


def test_disposition_motor_is_small_and_emits_two_logits() -> None:
    motor = DispositionMotor(16, 8)
    assert motor(torch.randn(5, 16)).shape == (5, 2)
    assert motor(torch.randn(5, 16, dtype=torch.bfloat16)).dtype == torch.float32
    assert sum(parameter.numel() for parameter in motor.parameters()) == 186
    with pytest.raises(
        ETTRDispositionMotorError,
        match="features differ",
    ):
        motor(torch.randn(5, 2, 16))


def test_query_codebook_rejects_any_third_token() -> None:
    batch = SimpleNamespace(
        episodes=SimpleNamespace(
            query=SimpleNamespace(
                targets=torch.tensor(
                    [[-1, 28], [-1, 29], [-1, 30]],
                    dtype=torch.long,
                )
            ),
            query_read_index=torch.tensor([1, 1, 1], dtype=torch.long),
        )
    )
    with pytest.raises(
        ETTRDispositionMotorError,
        match="leaves the disposition codebook",
    ):
        _query_classes(
            batch,
            negative_token_id=28,
            positive_token_id=29,
        )
    batch.episodes.query.targets[-1, -1] = 28
    assert torch.equal(
        _query_classes(
            batch,
            negative_token_id=28,
            positive_token_id=29,
        ),
        torch.tensor([0, 1, 0]),
    )


def test_arguments_bind_hashes_paths_codebook_and_bounded_rates(
    tmp_path,
) -> None:
    arguments = _arguments(tmp_path)
    _validate_args(arguments)
    arguments.positive_token_id = arguments.negative_token_id
    with pytest.raises(
        ETTRDispositionMotorError,
        match="arguments differ",
    ):
        _validate_args(arguments)


def test_strict_gate_requires_state_advantage_and_autonomous_signal() -> None:
    def arm(factual, world, command):
        return {
            "factual_accuracy": factual,
            "query_binding": {
                "world": {"margin_rates": {"0.1": world}},
                "command": {"margin_rates": {"0.1": command}},
            },
        }

    report = {
        "treatment": {
            "exact_terminal": arm(0.9, 0.8, 0.7),
            "autonomous": arm(0.6, 0.1, 0.0),
        },
        "query_only": {
            "exact_terminal": arm(0.5, 0.0, 0.0),
            "autonomous": arm(0.5, 0.0, 0.0),
        },
    }
    assert all(_gates({"motors": report}).values())
    report["treatment"]["autonomous"] = arm(0.5, 0.1, 0.0)
    assert not _gates({"motors": report})[
        "autonomous_factual_above_chance"
    ]
