from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from train_ettr_joint_instruction_canary import (
    ETTRTriCanaryError,
    _parse_weight,
    _validate_args,
)


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        release_root=tmp_path / "release",
        release_sha256="a" * 64,
        ettr_data_root=tmp_path / "data",
        tokenizer=tmp_path / "tokenizer.json",
        legacy_general_shard_dir=[tmp_path / "general"],
        legacy_general_weight=[1.0],
        parent_joint_model=tmp_path / "parent.pt",
        parent_joint_model_sha256="b" * 64,
        instruction_data=tmp_path / "instruction.jsonl",
        instruction_data_sha256="c" * 64,
        instruction_sample_weight=[("math", 0.8), ("code", 0.2)],
        output=tmp_path / "out",
        source_commit="d" * 40,
        updates=100,
        general_batch_size=16,
        instruction_batch_size=16,
        general_position_weight=15,
        instruction_position_weight=70,
        ettr_position_weight=15,
        data_seed=17,
        total_updates=100,
        warmup_updates=10,
        base_lr_muon=0.0015,
        base_lr_adam=0.00035,
        architecture_lr_muon=0.003,
        architecture_lr_adam=0.0006,
        nll_gradient_cap=4.0,
        query_binding_weight=1.0,
        query_binding_reduction="mixed-mean",
        query_binding_classification_weight=1.0,
        query_binding_effect_weight=1.0,
        query_binding_invariance_weight=1.0,
        query_binding_risk_temperature=1.0,
        packet_weight=1.0,
        intervention_weight=1.0,
        transaction_weight=1.0,
        commit_halt_weight=0.5,
        open_state_read_floor=0.0,
        soft_transaction_ettr_updates=0,
        execution_trace_read_scale=0.0,
        teacher_forced_transaction_weight=0.0,
        gradient_clip_mode="owner",
        log_every=1,
    )


def test_parse_instruction_weight() -> None:
    assert _parse_weight("code=0.2") == ("code", 0.2)
    with pytest.raises(argparse.ArgumentTypeError, match="NAME=WEIGHT"):
        _parse_weight("code")
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        _parse_weight("code=0")


def test_tri_canary_arguments_validate_and_reject_duplicate_groups(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    _validate_args(args)
    args.instruction_sample_weight = [("math", 0.5), ("math", 0.5)]
    with pytest.raises(ETTRTriCanaryError, match="arguments differ"):
        _validate_args(args)


def test_tri_canary_rejects_invalid_query_binding_risk(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.query_binding_risk_temperature = 0.0
    with pytest.raises(ETTRTriCanaryError, match="arguments differ"):
        _validate_args(args)
    args = _args(tmp_path)
    args.query_binding_effect_weight = -1.0
    with pytest.raises(ETTRTriCanaryError, match="arguments differ"):
        _validate_args(args)
    args = _args(tmp_path)
    args.intervention_weight = -1.0
    with pytest.raises(ETTRTriCanaryError, match="arguments differ"):
        _validate_args(args)
    args = _args(tmp_path)
    args.open_state_read_floor = 1.1
    with pytest.raises(ETTRTriCanaryError, match="arguments differ"):
        _validate_args(args)
    args = _args(tmp_path)
    args.soft_transaction_ettr_updates = args.updates + 1
    with pytest.raises(ETTRTriCanaryError, match="arguments differ"):
        _validate_args(args)
    args = _args(tmp_path)
    args.execution_trace_read_scale = 4.1
    with pytest.raises(ETTRTriCanaryError, match="arguments differ"):
        _validate_args(args)
