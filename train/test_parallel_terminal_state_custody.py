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
    )


def test_train_contract_accepts_only_fresh_absolute_output(tmp_path: Path) -> None:
    args = _train_args(tmp_path)
    validate_train_args(args)
    args.output.mkdir()
    with pytest.raises(ParallelTerminalStatePilotError, match="arguments differ"):
        validate_train_args(args)


def test_terminal_run_receipt_rejects_mutation(tmp_path: Path) -> None:
    for index, name in enumerate(_RUN_FILES):
        (tmp_path / name).write_bytes(f"artifact-{index}".encode())
    lines = "".join(
        f"{_sha256_file(tmp_path / name)}  {name}\n" for name in _RUN_FILES
    )
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
