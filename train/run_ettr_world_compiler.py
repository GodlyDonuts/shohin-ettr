#!/usr/bin/env python3
"""Fresh-process raw-token compiler for an immutable ETTR state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat

from safetensors.torch import load_file
import torch

from endogenous_typed_theory_reactor import (
    EndogenousTheoryCompiler,
    TheoryReactorConfig,
)
from ettr_state_io import write_state_once
from model import GPT, GPTConfig


class ETTRCompilerError(ValueError):
    """A compiler-process input or output contract failed."""


def _immutable_regular(path: Path) -> None:
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o222
    ):
        raise ETTRCompilerError(
            f"compiler input is not immutable regular file: {path}"
        )


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


def _read_canonical_json(path: Path) -> tuple[object, bytes]:
    _immutable_regular(path)
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ETTRCompilerError(
            f"malformed compiler JSON: {path}"
        ) from exc
    if payload != _canonical_json_bytes(value):
        raise ETTRCompilerError(
            f"noncanonical compiler JSON: {path}"
        )
    return value, payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def compile_world(
    *,
    config_path: Path,
    compiler_path: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    expected_step: int,
    world_path: Path,
    output_path: Path,
    hard: bool,
) -> None:
    for path in (
        config_path,
        compiler_path,
        checkpoint_path,
        world_path,
    ):
        _immutable_regular(path)
    config_payload, _ = _read_canonical_json(config_path)
    if not isinstance(config_payload, dict):
        raise ETTRCompilerError(
            "compiler configuration differs"
        )
    try:
        config = TheoryReactorConfig(**config_payload)
    except TypeError as exc:
        raise ETTRCompilerError(
            "compiler configuration keys differ"
        ) from exc
    config.validate()
    if _sha256_file(checkpoint_path) != checkpoint_sha256:
        raise ETTRCompilerError(
            "compiler checkpoint hash differs"
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("step") != expected_step
        or not isinstance(checkpoint.get("cfg"), dict)
        or not isinstance(checkpoint.get("model"), dict)
    ):
        raise ETTRCompilerError(
            "compiler checkpoint contract differs"
        )
    base = GPT(GPTConfig(**checkpoint["cfg"])).eval()
    try:
        incompatibility = base.load_state_dict(
            checkpoint["model"],
            strict=True,
        )
    except RuntimeError as exc:
        raise ETTRCompilerError(
            "compiler base weights differ"
        ) from exc
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ETTRCompilerError(
            "compiler base strict load differs"
        )
    if base.cfg.d_model != config.d_model:
        raise ETTRCompilerError(
            "compiler base and reactor widths differ"
        )
    compiler = EndogenousTheoryCompiler(config).eval()
    try:
        incompatibility = compiler.load_state_dict(
            load_file(compiler_path),
            strict=True,
        )
    except RuntimeError as exc:
        raise ETTRCompilerError(
            "compiler weights differ"
        ) from exc
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ETTRCompilerError(
            "compiler strict load differs"
        )
    world_payload, world_bytes = _read_canonical_json(world_path)
    if (
        not isinstance(world_payload, dict)
        or set(world_payload) != {"attention_mask", "token_ids"}
    ):
        raise ETTRCompilerError(
            "world token schema differs"
        )
    token_ids = torch.tensor(
        world_payload["token_ids"],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        world_payload["attention_mask"],
        dtype=torch.bool,
    )
    if (
        token_ids.ndim != 2
        or attention_mask.shape != token_ids.shape
        or token_ids.shape[1] > base.cfg.seq_len
    ):
        raise ETTRCompilerError(
            "world token geometry differs"
        )
    with torch.no_grad():
        hidden = base.tok(token_ids)
        cos = base.cos[: token_ids.shape[1]]
        sin = base.sin[: token_ids.shape[1]]
        for block in base.blocks[: config.stage_after_block + 1]:
            hidden, _ = block(hidden, cos, sin)
        state = compiler(
            hidden,
            attention_mask=attention_mask,
            hard=hard,
        )
    write_state_once(
        output_path,
        state,
        config,
        forbidden_source=world_bytes,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-sha256",
        required=True,
    )
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hard", action="store_true")
    arguments = parser.parse_args()
    compile_world(
        config_path=arguments.config,
        compiler_path=arguments.compiler,
        checkpoint_path=arguments.checkpoint,
        checkpoint_sha256=arguments.checkpoint_sha256,
        expected_step=arguments.expected_step,
        world_path=arguments.world,
        output_path=arguments.output,
        hard=arguments.hard,
    )


if __name__ == "__main__":
    main()
