from __future__ import annotations

from types import SimpleNamespace

import pytest

from eval_ettr_v3 import (
    ETTRV3EvaluationError,
    _arm_summary,
    _paired_loss_summary,
    _validate_checkpoint_cursor,
)


LOSS_FIELDS = (
    "total",
    "token_lm",
    "packet",
    "world_intervention",
    "command_intervention",
    "world_query_binding",
    "command_query_binding",
    "transaction",
    "equivariance",
    "commit_halt",
    "sparsity",
    "anti_bypass",
)


def _loss(value: float) -> dict[str, float]:
    return {name: value for name in LOSS_FIELDS}


def _counts(satisfied: int) -> dict[str, int]:
    from eval_ettr_v3 import _COUNT_FIELDS

    result = {name: 0 for name in _COUNT_FIELDS}
    result.update(
        {
            "command_query_contrast_pairs": 4,
            "command_query_margin_satisfied": satisfied,
            "world_query_contrast_pairs": 4,
            "world_query_margin_satisfied": satisfied,
        }
    )
    return result


def test_arm_summary_aggregates_losses_and_margin_support() -> None:
    summary = _arm_summary(
        [_loss(4.0), _loss(2.0)],
        [_counts(1), _counts(3)],
    )
    assert summary["loss_means"]["total"] == 3.0
    assert summary["count_totals"]["world_query_contrast_pairs"] == 8
    assert summary["query_binding_margin_rates"] == {
        "command": 0.5,
        "world": 0.5,
    }


def test_paired_summary_requires_consistent_per_batch_gain() -> None:
    summary = _paired_loss_summary(
        [_loss(4.0), _loss(5.0), _loss(6.0)],
        [_loss(3.0), _loss(4.0), _loss(5.0)],
    )
    assert summary["total"]["checkpoint_minus_raw_mean"] == -1.0
    assert summary["total"]["confidence_95"] == [-1.0, -1.0]
    assert summary["total"]["improved_with_upper_95_below_zero"]
    assert summary["total"]["win_fraction"] == 1.0


def test_paired_summary_rejects_unpaired_populations() -> None:
    with pytest.raises(
        ETTRV3EvaluationError,
        match="paired development population differs",
    ):
        _paired_loss_summary([_loss(1.0)], [])


def test_checkpoint_cursor_reconciles_exact_training_contract() -> None:
    progress = SimpleNamespace(
        global_step=300_012,
        optimizer_step=12,
        gradient_accumulation_steps=2,
    )
    manifest = SimpleNamespace(
        dataset_sha256="b" * 64,
        sha256=lambda: "a" * 64,
    )
    stream = SimpleNamespace(manifest=manifest)
    contract = {
        "accumulation": 2,
        "compile_backend": "inductor",
        "compile_mode": "default",
        "data_seed": 17,
        "hard_transactions": False,
        "world_size": 4,
    }
    data_stream = SimpleNamespace(
        manifest_sha256="a" * 64,
        dataset_sha256="b" * 64,
        seed=17,
        sampler_state={
            "accumulation": 2,
            "compile_backend": "inductor",
            "compile_mode": "default",
            "hard_transactions": False,
            "consumed_stream_batches": 96,
            "release_file_sha256": "c" * 64,
            "schema": "shohin-ettr-il-v3-distributed-cursor-v1",
            "world_size": 4,
        },
    )
    _validate_checkpoint_cursor(
        progress,
        data_stream,
        run_contract=contract,
        stream=stream,
        release_sha256="c" * 64,
        protected_step=300_000,
    )
    data_stream.sampler_state["consumed_stream_batches"] = 95
    with pytest.raises(
        ETTRV3EvaluationError,
        match="checkpoint cursor differs",
    ):
        _validate_checkpoint_cursor(
            progress,
            data_stream,
            run_contract=contract,
            stream=stream,
            release_sha256="c" * 64,
            protected_step=300_000,
        )


def test_checkpoint_cursor_binds_nondefault_query_weight() -> None:
    progress = SimpleNamespace(
        global_step=300_012,
        optimizer_step=12,
        gradient_accumulation_steps=2,
    )
    manifest = SimpleNamespace(
        dataset_sha256="b" * 64,
        sha256=lambda: "a" * 64,
    )
    stream = SimpleNamespace(manifest=manifest)
    contract = {
        "accumulation": 2,
        "compile_backend": None,
        "compile_mode": None,
        "data_seed": 17,
        "hard_transactions": True,
        "query_binding_weight": 8.0,
        "world_size": 4,
    }
    data_stream = SimpleNamespace(
        manifest_sha256="a" * 64,
        dataset_sha256="b" * 64,
        seed=17,
        sampler_state={
            "accumulation": 2,
            "compile_backend": None,
            "compile_mode": None,
            "hard_transactions": True,
            "consumed_stream_batches": 96,
            "query_binding_weight": 8.0,
            "release_file_sha256": "c" * 64,
            "schema": "shohin-ettr-il-v3-distributed-cursor-v1",
            "world_size": 4,
        },
    )
    _validate_checkpoint_cursor(
        progress,
        data_stream,
        run_contract=contract,
        stream=stream,
        release_sha256="c" * 64,
        protected_step=300_000,
    )
    data_stream.sampler_state["query_binding_weight"] = 4.0
    with pytest.raises(
        ETTRV3EvaluationError,
        match="checkpoint cursor differs",
    ):
        _validate_checkpoint_cursor(
            progress,
            data_stream,
            run_contract=contract,
            stream=stream,
            release_sha256="c" * 64,
            protected_step=300_000,
        )
