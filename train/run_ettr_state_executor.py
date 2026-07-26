#!/usr/bin/env python3
"""Fresh-process ETTR executor with a genuine post-seal COMMAND surface."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import stat

from safetensors.torch import load_file
import torch

from endogenous_typed_theory_reactor import (
    GenericTransactionReactor,
    TheoryReactorConfig,
)
from ettr_factorial_custody import (
    ETTRFactorialExecutionManifest,
    ETTRStageExecutionReceipt,
    STAGE_RECEIPT_SCHEMA,
    sha256_file,
    write_json_once,
)
from ettr_state_io import read_state, typed_state_sha256, write_state_once
from model import GPT, GPTConfig


class ETTRExecutorError(ValueError):
    """A fresh-process executor input or output contract failed."""


def _immutable_regular(path: Path) -> None:
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o222
    ):
        raise ETTRExecutorError(f"executor input is not immutable regular file: {path}")


def _load_config(path: Path) -> TheoryReactorConfig:
    _immutable_regular(path)
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ETTRExecutorError("executor configuration is malformed") from exc
    canonical = (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    if payload != canonical:
        raise ETTRExecutorError("executor configuration is not canonical")
    try:
        config = TheoryReactorConfig(**value)
    except TypeError as exc:
        raise ETTRExecutorError("executor configuration keys differ") from exc
    if asdict(config) != value:
        raise ETTRExecutorError("executor configuration values differ")
    config.validate()
    return config


def _read_canonical_json(path: Path) -> tuple[object, bytes]:
    _immutable_regular(path)
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ETTRExecutorError(f"executor JSON is malformed: {path}") from exc
    canonical = (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    if payload != canonical:
        raise ETTRExecutorError(f"executor JSON is not canonical: {path}")
    return value, payload


def execute(
    *,
    config_path: Path,
    state_path: Path,
    reactor_path: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    expected_step: int,
    command_path: Path,
    execution_manifest_path: Path,
    execution_manifest_sha256: str,
    compiler_receipt_path: Path,
    compiler_receipt_sha256: str,
    output_path: Path,
    receipt_output_path: Path,
    steps: int,
    hard: bool,
) -> None:
    for path in (
        config_path,
        state_path,
        reactor_path,
        checkpoint_path,
        command_path,
        execution_manifest_path,
        compiler_receipt_path,
    ):
        _immutable_regular(path)
    execution_manifest = ETTRFactorialExecutionManifest.from_path(
        execution_manifest_path
    )
    execution_manifest.validate_hash(execution_manifest_sha256)
    compiler_receipt = ETTRStageExecutionReceipt.from_path(compiler_receipt_path)
    compiler_receipt.validate(
        execution_manifest,
        expected_receipt_sha256=compiler_receipt_sha256,
    )
    if compiler_receipt.stage != "world":
        raise ETTRExecutorError("executor compiler receipt stage differs")
    config = _load_config(config_path)
    state = read_state(state_path, config)
    input_state_sha256 = sha256_file(state_path)
    if (
        sha256_file(config_path) != execution_manifest.config_sha256
        or sha256_file(reactor_path) != execution_manifest.reactor_sha256
        or sha256_file(checkpoint_path) != checkpoint_sha256
        or checkpoint_sha256 != execution_manifest.checkpoint_sha256
        or expected_step != execution_manifest.checkpoint_step
        or sha256_file(command_path) != execution_manifest.command_tokens_sha256
        or compiler_receipt.output_state_file_sha256 != input_state_sha256
        or compiler_receipt.output_state_tensor_sha256 != typed_state_sha256(state)
        or sha256_file(Path(__file__).resolve())
        != execution_manifest.executor_runner_sha256
        or hard != execution_manifest.executor_hard
        or steps != execution_manifest.executor_steps
    ):
        raise ETTRExecutorError("executor execution manifest differs")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("step") != expected_step
        or not isinstance(checkpoint.get("cfg"), dict)
        or not isinstance(checkpoint.get("model"), dict)
    ):
        raise ETTRExecutorError("executor checkpoint contract differs")
    base = GPT(GPTConfig(**checkpoint["cfg"])).eval()
    try:
        incompatibility = base.load_state_dict(
            checkpoint["model"],
            strict=True,
        )
    except RuntimeError as exc:
        raise ETTRExecutorError("executor base weights differ") from exc
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ETTRExecutorError("executor base strict load differs")
    if base.cfg.d_model != config.d_model:
        raise ETTRExecutorError("executor base and reactor widths differ")
    reactor = GenericTransactionReactor(config).eval()
    try:
        incompatibility = reactor.load_state_dict(
            load_file(reactor_path),
            strict=True,
        )
    except RuntimeError as exc:
        raise ETTRExecutorError("reactor weights differ") from exc
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ETTRExecutorError("reactor strict load differs")
    command_payload, command_bytes = _read_canonical_json(command_path)
    if not isinstance(command_payload, dict) or set(command_payload) != {
        "attention_mask",
        "token_ids",
    }:
        raise ETTRExecutorError("executor command token schema differs")
    token_ids = torch.tensor(
        command_payload["token_ids"],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        command_payload["attention_mask"],
        dtype=torch.bool,
    )
    if (
        token_ids.ndim != 2
        or attention_mask.shape != token_ids.shape
        or token_ids.shape[0] != state.value_probabilities.shape[0]
        or token_ids.shape[1] > base.cfg.seq_len
    ):
        raise ETTRExecutorError("executor command token geometry differs")
    with torch.no_grad():
        command_hidden = base.tok(token_ids)
        cos = base.cos[: token_ids.shape[1]]
        sin = base.sin[: token_ids.shape[1]]
        for block in base.blocks[: config.stage_after_block + 1]:
            command_hidden, _ = block(
                command_hidden,
                cos,
                sin,
            )
        terminal, _ = reactor(
            state,
            steps=steps,
            hard=hard,
            command_hidden=command_hidden,
            command_attention_mask=attention_mask,
        )
    state_receipt = write_state_once(
        output_path,
        terminal,
        config,
        forbidden_source=command_bytes,
    )
    receipt = ETTRStageExecutionReceipt(
        schema=STAGE_RECEIPT_SCHEMA,
        stage="command",
        manifest_sha256=execution_manifest_sha256,
        parent_receipt_sha256=compiler_receipt_sha256,
        input_state_file_sha256=input_state_sha256,
        input_state_tensor_sha256=typed_state_sha256(state),
        token_input_sha256=execution_manifest.command_tokens_sha256,
        component_sha256=execution_manifest.reactor_sha256,
        checkpoint_sha256=execution_manifest.checkpoint_sha256,
        output_state_file_sha256=state_receipt.sha256,
        output_state_tensor_sha256=typed_state_sha256(terminal),
        row_count=token_ids.shape[0],
    )
    if receipt.row_count != execution_manifest.row_count:
        raise ETTRExecutorError("executor qualification row count differs")
    write_json_once(receipt_output_path, asdict(receipt))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--reactor", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-sha256",
        required=True,
    )
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--command", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--execution-manifest-sha256", required=True)
    parser.add_argument("--compiler-receipt", type=Path, required=True)
    parser.add_argument("--compiler-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--hard", action="store_true")
    arguments = parser.parse_args()
    execute(
        config_path=arguments.config,
        state_path=arguments.state,
        reactor_path=arguments.reactor,
        checkpoint_path=arguments.checkpoint,
        checkpoint_sha256=arguments.checkpoint_sha256,
        expected_step=arguments.expected_step,
        command_path=arguments.command,
        execution_manifest_path=arguments.execution_manifest,
        execution_manifest_sha256=arguments.execution_manifest_sha256,
        compiler_receipt_path=arguments.compiler_receipt,
        compiler_receipt_sha256=arguments.compiler_receipt_sha256,
        output_path=arguments.output,
        receipt_output_path=arguments.receipt_output,
        steps=arguments.steps,
        hard=arguments.hard,
    )


if __name__ == "__main__":
    main()
