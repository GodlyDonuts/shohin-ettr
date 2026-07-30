#!/usr/bin/env python3
"""Diagnose one exact ETTR-v3 production batch without publishing a model."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Sequence

import torch

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TheoryReactorConfig,
)
from ettr_checkpoint import load_protected_base_model
from ettr_objectives import ETTRObjectiveConfig
from ettr_optimization import ETTROptimizerBundle, ETTROptimizerConfig
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_train_step import ETTRTrainStep, ETTRTrainStepConfig
from ettr_v3_streaming import ETTRV3StreamingRelease


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ETTRV3DiagnosticError(RuntimeError):
    """The exact first-batch diagnostic contract failed."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        + b"\n"
    )


def _write_no_replace(path: Path, payload: bytes) -> str:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--protected-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture-seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument(
        "--transaction-mode",
        choices=("hard", "soft"),
        required=True,
    )
    parser.add_argument("--nll-gradient-cap", type=float)
    return parser.parse_args(argv)


def _gradient_receipt(
    model: EndogenousTypedTheoryReactorGPT,
) -> dict[str, dict[str, int]]:
    modules = {
        "base": model.base,
        "compiler": model.compiler,
        "reactor": model.reactor,
        "query_reader": model.query_reader,
    }
    receipt = {}
    for name, module in modules.items():
        gradients = tuple(
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad and parameter.grad is not None
        )
        receipt[name] = {
            "finite_elements": sum(
                int(torch.isfinite(value).sum().item())
                for value in gradients
            ),
            "gradient_tensors": len(gradients),
            "nonfinite_elements": sum(
                int((~torch.isfinite(value)).sum().item())
                for value in gradients
            ),
        }
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if (
        _HEX64.fullmatch(args.release_sha256) is None
        or args.architecture_seed < 0
        or args.data_seed < 0
        or args.output.exists()
        or args.output.is_symlink()
        or not torch.cuda.is_available()
        or (
            args.nll_gradient_cap is not None
            and (
                not math.isfinite(args.nll_gradient_cap)
                or args.nll_gradient_cap <= 0.0
                or args.transaction_mode != "hard"
            )
        )
    ):
        raise ETTRV3DiagnosticError("ETTR v3 diagnostic arguments differ")

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    stream = ETTRV3StreamingRelease(
        args.release_root,
        expected_release_sha256=args.release_sha256,
        data_root=args.data_root,
        tokenizer_path=args.tokenizer,
    )
    base, protected = load_protected_base_model(
        args.protected_checkpoint,
    )
    if (
        protected.checkpoint_sha256
        != stream.manifest.protected_checkpoint_sha256
    ):
        raise ETTRV3DiagnosticError(
            "protected checkpoint differs from the release"
        )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.architecture_seed)
        model = EndogenousTypedTheoryReactorGPT(
            base,
            TheoryReactorConfig(),
        )
    model.to(device=device, dtype=torch.bfloat16)
    optimizer = ETTROptimizerBundle(
        model,
        ETTROptimizerConfig(
            train_base=False,
            warmup_updates=2_000,
            total_updates=300_000,
        ),
    )
    step = ETTRTrainStep(
        model,
        optimizer,
        ETTRObjectiveConfig(
            vocab_size=model.base.cfg.vocab_size,
            nll_gradient_cap=args.nll_gradient_cap,
        ),
        manifest=stream.manifest,
        packet_sufficiency=ETTRDiskPacketSufficiencyIndex(
            stream.packet_index_root
        ),
        manifest_sha256=stream.manifest.sha256(),
        step_config=ETTRTrainStepConfig(
            hard_transactions=args.transaction_mode == "hard",
        ),
    )
    iterator = stream.iter_positioned_batches(
        "train",
        rank=0,
        world_size=1,
        epoch=0,
        seed=args.data_seed,
        device=device,
    )
    position, batch = next(iterator)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    receipt = step.update((batch,))
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    loss_names = (
        "total_loss",
        "token_lm_loss",
        "packet_loss",
        "world_intervention_loss",
        "command_intervention_loss",
        "world_query_binding_loss",
        "command_query_binding_loss",
        "transaction_loss",
        "equivariance_loss",
        "commit_halt_loss",
        "sparsity_loss",
        "anti_bypass_loss",
        "gradient_norm",
    )
    metrics = {
        name: float(getattr(receipt, name).detach().float().cpu())
        for name in loss_names
    }
    payload = {
        "architecture_seed": args.architecture_seed,
        "data_seed": args.data_seed,
        "elapsed_seconds": elapsed,
        "gradient_components": _gradient_receipt(model),
        "gates": {
            "finite_gradient_norm": torch.isfinite(
                receipt.gradient_norm
            ).item(),
            "finite_losses": all(
                torch.isfinite(getattr(receipt, name)).item()
                for name in loss_names
            ),
        },
        "metrics": metrics,
        "nll_gradient_cap": args.nll_gradient_cap,
        "optimizer_config": asdict(optimizer.config),
        "position": position,
        "protected_checkpoint_sha256": protected.checkpoint_sha256,
        "release_file_sha256": args.release_sha256,
        "schema": "shohin-ettr-v3-first-production-batch-diagnostic-v1",
        "status": "pass",
        "transaction_mode": args.transaction_mode,
        "world_size": 1,
    }
    _write_no_replace(args.output, _canonical_bytes(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
