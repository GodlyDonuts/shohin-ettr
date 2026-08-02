from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from eval_parallel_terminal_state import (
    ParallelTerminalStateEvaluationError,
    _RUN_FILES,
    _run_receipt,
    _validate_args as validate_eval_args,
)
from train_ettr_component_island import _sha256_file
from train_parallel_terminal_state_pilot import (
    ParallelTerminalStatePilotError,
    _run_schemas,
    _validate_args as validate_train_args,
)


def _absolute(tmp_path: Path, name: str) -> Path:
    return tmp_path / name


def _train_args(tmp_path: Path):
    paths = {
        name: _absolute(tmp_path, name)
        for name in (
            "release_root",
            "data_root",
            "tokenizer",
            "protected_checkpoint",
            "joint_model",
            "joint_run_contract",
            "compiler",
            "compiler_contract",
        )
    }
    return SimpleNamespace(
        **paths,
        output=_absolute(tmp_path, "fresh-output"),
        release_sha256="a" * 64,
        joint_model_sha256="b" * 64,
        joint_run_contract_sha256="c" * 64,
        compiler_sha256="d" * 64,
        compiler_contract_sha256="e" * 64,
        source_commit="f" * 40,
        architecture_seed=31,
        data_seed=11,
        updates=500,
        start_position=0,
        eval_batches=32,
        log_every=10,
        learning_rate=3e-4,
        gradient_clip=1.0,
        causal_delta_weight=4.0,
        residual_edits=True,
        atomic_edits=False,
        lexical_command=False,
        token_native_command_mask=False,
        cover_verified_command_mask=False,
        token_native_occurrence_command=False,
        token_native_syntax_graph_command=False,
        token_native_declaration_binding_command=False,
        token_native_operation_recurrence_command=False,
        atomic_action_weight=1.0,
    )


def test_train_contract_accepts_only_fresh_absolute_output(tmp_path: Path) -> None:
    args = _train_args(tmp_path)
    validate_train_args(args)
    args.output.mkdir()
    with pytest.raises(ParallelTerminalStatePilotError, match="arguments differ"):
        validate_train_args(args)


def test_terminal_schema_tracks_residual_architecture() -> None:
    assert _run_schemas(True) == (
        "shohin-ettr-parallel-terminal-state-contract-v3",
        "shohin-ettr-parallel-terminal-state-report-v3",
        "shohin-ettr-parallel-terminal-state-metric-v3",
    )
    assert _run_schemas(False) == (
        "shohin-ettr-parallel-terminal-state-contract-v2",
        "shohin-ettr-parallel-terminal-state-report-v2",
        "shohin-ettr-parallel-terminal-state-metric-v2",
    )
    assert _run_schemas(False, True) == (
        "shohin-ettr-parallel-terminal-state-contract-v4",
        "shohin-ettr-parallel-terminal-state-report-v4",
        "shohin-ettr-parallel-terminal-state-metric-v4",
    )
    assert _run_schemas(False, True, True) == (
        "shohin-ettr-parallel-terminal-state-contract-v5",
        "shohin-ettr-parallel-terminal-state-report-v5",
        "shohin-ettr-parallel-terminal-state-metric-v5",
    )
    assert _run_schemas(False, True, True, True) == (
        "shohin-ettr-parallel-terminal-state-contract-v6",
        "shohin-ettr-parallel-terminal-state-report-v6",
        "shohin-ettr-parallel-terminal-state-metric-v6",
    )
    assert _run_schemas(False, True, True, True, True) == (
        "shohin-ettr-parallel-terminal-state-contract-v7",
        "shohin-ettr-parallel-terminal-state-report-v7",
        "shohin-ettr-parallel-terminal-state-metric-v7",
    )
    assert _run_schemas(False, True, True, True, False, True, True) == (
        "shohin-ettr-parallel-terminal-state-contract-v8",
        "shohin-ettr-parallel-terminal-state-report-v8",
        "shohin-ettr-parallel-terminal-state-metric-v8",
    )
    assert _run_schemas(False, True, True, True, False, True, True, True) == (
        "shohin-ettr-parallel-terminal-state-contract-v9",
        "shohin-ettr-parallel-terminal-state-report-v9",
        "shohin-ettr-parallel-terminal-state-metric-v9",
    )
    effect_set = (
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        False,
        True,
    )
    assert _run_schemas(*effect_set) == (
        "shohin-ettr-parallel-terminal-state-contract-v12",
        "shohin-ettr-parallel-terminal-state-report-v12",
        "shohin-ettr-parallel-terminal-state-metric-v12",
    )
    assert _run_schemas(*effect_set, True) == (
        "shohin-ettr-parallel-terminal-state-contract-v13",
        "shohin-ettr-parallel-terminal-state-report-v13",
        "shohin-ettr-parallel-terminal-state-metric-v13",
    )
    assert _run_schemas(*effect_set, True, True) == (
        "shohin-ettr-parallel-terminal-state-contract-v14",
        "shohin-ettr-parallel-terminal-state-report-v14",
        "shohin-ettr-parallel-terminal-state-metric-v14",
    )
    assert _run_schemas(*effect_set[:-1], False, True, False, True) == (
        "shohin-ettr-parallel-terminal-state-contract-v15",
        "shohin-ettr-parallel-terminal-state-report-v15",
        "shohin-ettr-parallel-terminal-state-metric-v15",
    )


def test_write_link_rail_args_are_exclusive_and_operation_bound(tmp_path: Path) -> None:
    args = _train_args(tmp_path)
    args.residual_edits = False
    args.atomic_edits = True
    args.lexical_command = True
    args.token_native_command_mask = True
    args.cover_verified_command_mask = True
    args.token_native_syntax_graph_command = True
    args.token_native_declaration_binding_command = True
    args.token_native_operation_recurrence_command = True
    args.token_native_operation_state_command = True
    args.factorized_operation_effect_command = False
    args.operation_effect_set_command = False
    args.operation_effect_role_anchors = True
    args.operation_effect_cardinality_gate = False
    args.operation_effect_write_link_rails = True
    args.training_initial_state = "oracle"
    validate_train_args(args)
    args.operation_effect_set_command = True
    with pytest.raises(ParallelTerminalStatePilotError, match="arguments differ"):
        validate_train_args(args)


def test_terminal_run_receipt_rejects_mutation(tmp_path: Path) -> None:
    for index, name in enumerate(_RUN_FILES):
        (tmp_path / name).write_bytes(f"artifact-{index}".encode())
    lines = "".join(f"{_sha256_file(tmp_path / name)}  {name}\n" for name in _RUN_FILES)
    (tmp_path / "SHA256SUMS").write_text(lines, encoding="ascii")
    digest = _sha256_file(tmp_path / "SHA256SUMS")
    receipt = _run_receipt(tmp_path, digest)
    assert tuple(sorted(receipt)) == tuple(sorted(_RUN_FILES))
    (tmp_path / _RUN_FILES[0]).write_bytes(b"mutated")
    with pytest.raises(
        ParallelTerminalStateEvaluationError,
        match="run file differs",
    ):
        _run_receipt(tmp_path, digest)


def test_eval_contract_rejects_non_hash_receipt(tmp_path: Path) -> None:
    train = _train_args(tmp_path)
    args = SimpleNamespace(
        release_root=train.release_root,
        data_root=train.data_root,
        tokenizer=train.tokenizer,
        protected_checkpoint=train.protected_checkpoint,
        joint_model=train.joint_model,
        joint_run_contract=train.joint_run_contract,
        compiler=train.compiler,
        compiler_contract=train.compiler_contract,
        terminal_run_dir=_absolute(tmp_path, "terminal-run"),
        output=_absolute(tmp_path, "fresh-eval"),
        release_sha256=train.release_sha256,
        joint_model_sha256=train.joint_model_sha256,
        joint_run_contract_sha256=train.joint_run_contract_sha256,
        compiler_sha256=train.compiler_sha256,
        compiler_contract_sha256=train.compiler_contract_sha256,
        terminal_run_sha256s_sha256="not-a-hash",
        source_commit=train.source_commit,
        data_seed=train.data_seed,
        max_batches=32,
    )
    with pytest.raises(
        ParallelTerminalStateEvaluationError,
        match="arguments differ",
    ):
        validate_eval_args(args)
