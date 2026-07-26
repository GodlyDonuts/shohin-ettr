#!/usr/bin/env python3
"""Fresh-process late query against an immutable ETTR terminal state."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import stat

from safetensors.torch import load_file
import torch

from endogenous_typed_theory_reactor import (
    SourceDeletedQueryReader,
    TheoryReactorConfig,
)
from ettr_factorial_custody import (
    ETTRFactorialExecutionManifest,
    ETTRLateQueryExecutionReceipt,
    ETTRStageExecutionReceipt,
    QUERY_RECEIPT_SCHEMA,
    sha256_file,
    token_tensor_sha256,
    write_json_once,
)
from ettr_state_io import read_state, typed_state_sha256
from model import GPT, GPTConfig


ANSWER_SCHEMA = "shohin-ettr-late-query-answer-v1"


class ETTRLateQueryError(ValueError):
    """A detached query-process contract failed."""


def _immutable_regular(path: Path) -> None:
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o222
    ):
        raise ETTRLateQueryError(f"query input is not immutable regular file: {path}")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _read_canonical_json(path: Path) -> object:
    _immutable_regular(path)
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ETTRLateQueryError(f"malformed query JSON: {path}") from exc
    if payload != _canonical_json_bytes(value):
        raise ETTRLateQueryError(f"noncanonical query JSON: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def answer(
    *,
    config_path: Path,
    state_path: Path,
    reader_path: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    expected_step: int,
    query_path: Path,
    execution_manifest_path: Path,
    execution_manifest_sha256: str,
    executor_receipt_path: Path,
    executor_receipt_sha256: str,
    tokenization_receipt_sha256: str,
    model_assembly_receipt_sha256: str,
    output_path: Path,
    receipt_output_path: Path,
) -> None:
    for path in (
        config_path,
        state_path,
        reader_path,
        checkpoint_path,
        query_path,
        execution_manifest_path,
        executor_receipt_path,
    ):
        _immutable_regular(path)
    execution_manifest = ETTRFactorialExecutionManifest.from_path(
        execution_manifest_path
    )
    execution_manifest.validate_hash(execution_manifest_sha256)
    executor_receipt = ETTRStageExecutionReceipt.from_path(executor_receipt_path)
    executor_receipt.validate(
        execution_manifest,
        expected_receipt_sha256=executor_receipt_sha256,
    )
    if executor_receipt.stage != "command":
        raise ETTRLateQueryError("late query executor receipt stage differs")
    config_payload = _read_canonical_json(config_path)
    if not isinstance(config_payload, dict):
        raise ETTRLateQueryError("query configuration differs")
    try:
        config = TheoryReactorConfig(**config_payload)
    except TypeError as exc:
        raise ETTRLateQueryError("query configuration keys differ") from exc
    config.validate()
    state = read_state(state_path, config)
    if (
        _sha256_file(checkpoint_path) != checkpoint_sha256
        or checkpoint_sha256 != execution_manifest.checkpoint_sha256
        or expected_step != execution_manifest.checkpoint_step
        or sha256_file(config_path) != execution_manifest.config_sha256
        or sha256_file(state_path) != executor_receipt.output_state_file_sha256
        or typed_state_sha256(state)
        != executor_receipt.output_state_tensor_sha256
        or sha256_file(reader_path) != execution_manifest.reader_sha256
        or sha256_file(query_path) != execution_manifest.query_tokens_sha256
        or tokenization_receipt_sha256
        != execution_manifest.tokenization_receipt_sha256
        or model_assembly_receipt_sha256
        != execution_manifest.model_assembly_receipt_sha256
        or sha256_file(Path(__file__).resolve())
        != execution_manifest.query_runner_sha256
    ):
        raise ETTRLateQueryError("late-query execution manifest differs")
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
        raise ETTRLateQueryError("query checkpoint contract differs")
    base = GPT(GPTConfig(**checkpoint["cfg"])).eval()
    try:
        incompatibility = base.load_state_dict(
            checkpoint["model"],
            strict=True,
        )
    except RuntimeError as exc:
        raise ETTRLateQueryError("query base weights differ") from exc
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ETTRLateQueryError("query base strict load differs")
    if base.cfg.d_model != config.d_model:
        raise ETTRLateQueryError("query base and reactor widths differ")
    reader = SourceDeletedQueryReader(config).eval()
    try:
        incompatibility = reader.load_state_dict(
            load_file(reader_path),
            strict=True,
        )
    except RuntimeError as exc:
        raise ETTRLateQueryError("query reader weights differ") from exc
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ETTRLateQueryError("query reader strict load differs")
    query_payload = _read_canonical_json(query_path)
    if not isinstance(query_payload, dict) or set(query_payload) != {
        "attention_mask",
        "token_ids",
    }:
        raise ETTRLateQueryError("late query schema differs")
    token_ids = torch.tensor(
        query_payload["token_ids"],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        query_payload["attention_mask"],
        dtype=torch.bool,
    )
    if (
        token_ids.ndim != 2
        or attention_mask.shape != token_ids.shape
        or token_ids.shape[0] != state.value_probabilities.shape[0]
        or token_ids.shape[1] > base.cfg.seq_len
    ):
        raise ETTRLateQueryError("late query tensor geometry differs")
    with torch.no_grad():
        hidden = base.tok(token_ids)
        cos = base.cos[: token_ids.shape[1]]
        sin = base.sin[: token_ids.shape[1]]
        for block in base.blocks[: config.stage_after_block + 1]:
            hidden, _ = block(hidden, cos, sin)
        hidden = hidden + reader(
            hidden,
            state,
            attention_mask=attention_mask,
        )
        for block in base.blocks[config.stage_after_block + 1 :]:
            hidden, _ = block(hidden, cos, sin)
        logits = base.head(base.norm(hidden))
        answers = logits.argmax(-1)
    answer_payload = {
        "schema": ANSWER_SCHEMA,
        "token_ids": answers.tolist(),
    }
    answer_file_sha256 = write_json_once(
        output_path,
        answer_payload,
    )
    receipt = ETTRLateQueryExecutionReceipt(
        schema=QUERY_RECEIPT_SCHEMA,
        execution_manifest_sha256=execution_manifest_sha256,
        tokenization_receipt_sha256=tokenization_receipt_sha256,
        model_assembly_receipt_sha256=model_assembly_receipt_sha256,
        executor_receipt_sha256=executor_receipt_sha256,
        terminal_state_file_sha256=executor_receipt.output_state_file_sha256,
        terminal_state_tensor_sha256=executor_receipt.output_state_tensor_sha256,
        query_tokens_sha256=sha256_file(query_path),
        reader_sha256=sha256_file(reader_path),
        checkpoint_sha256=checkpoint_sha256,
        answer_file_sha256=answer_file_sha256,
        answer_token_tensor_sha256=token_tensor_sha256(answers),
        row_count=answers.shape[0],
    )
    write_json_once(receipt_output_path, asdict(receipt))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-sha256",
        required=True,
    )
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--execution-manifest-sha256", required=True)
    parser.add_argument("--executor-receipt", type=Path, required=True)
    parser.add_argument("--executor-receipt-sha256", required=True)
    parser.add_argument("--tokenization-receipt-sha256", required=True)
    parser.add_argument("--model-assembly-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    arguments = parser.parse_args()
    answer(
        config_path=arguments.config,
        state_path=arguments.state,
        reader_path=arguments.reader,
        checkpoint_path=arguments.checkpoint,
        checkpoint_sha256=arguments.checkpoint_sha256,
        expected_step=arguments.expected_step,
        query_path=arguments.query,
        execution_manifest_path=arguments.execution_manifest,
        execution_manifest_sha256=arguments.execution_manifest_sha256,
        executor_receipt_path=arguments.executor_receipt,
        executor_receipt_sha256=arguments.executor_receipt_sha256,
        tokenization_receipt_sha256=arguments.tokenization_receipt_sha256,
        model_assembly_receipt_sha256=arguments.model_assembly_receipt_sha256,
        output_path=arguments.output,
        receipt_output_path=arguments.receipt_output,
    )


if __name__ == "__main__":
    main()
