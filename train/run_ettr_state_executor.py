#!/usr/bin/env python3
"""Fresh-process ETTR executor with a closed structural input surface."""

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
from ettr_state_io import read_state, write_state_once


class ETTRExecutorError(ValueError):
    """A fresh-process executor input or output contract failed."""


def _immutable_regular(path: Path) -> None:
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o222
    ):
        raise ETTRExecutorError(
            f"executor input is not immutable regular file: {path}"
        )


def _load_config(path: Path) -> TheoryReactorConfig:
    _immutable_regular(path)
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ETTRExecutorError(
            "executor configuration is malformed"
        ) from exc
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
        raise ETTRExecutorError(
            "executor configuration is not canonical"
        )
    try:
        config = TheoryReactorConfig(**value)
    except TypeError as exc:
        raise ETTRExecutorError(
            "executor configuration keys differ"
        ) from exc
    if asdict(config) != value:
        raise ETTRExecutorError(
            "executor configuration values differ"
        )
    config.validate()
    return config


def execute(
    *,
    config_path: Path,
    state_path: Path,
    reactor_path: Path,
    output_path: Path,
    steps: int,
    hard: bool,
) -> None:
    for path in (config_path, state_path, reactor_path):
        _immutable_regular(path)
    config = _load_config(config_path)
    state = read_state(state_path, config)
    reactor = GenericTransactionReactor(config).eval()
    try:
        incompatibility = reactor.load_state_dict(
            load_file(reactor_path),
            strict=True,
        )
    except RuntimeError as exc:
        raise ETTRExecutorError(
            "reactor weights differ"
        ) from exc
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ETTRExecutorError(
            "reactor strict load differs"
        )
    with torch.no_grad():
        terminal, _ = reactor(
            state,
            steps=steps,
            hard=hard,
        )
    write_state_once(
        output_path,
        terminal,
        config,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--reactor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--hard", action="store_true")
    arguments = parser.parse_args()
    execute(
        config_path=arguments.config,
        state_path=arguments.state,
        reactor_path=arguments.reactor,
        output_path=arguments.output,
        steps=arguments.steps,
        hard=arguments.hard,
    )


if __name__ == "__main__":
    main()
