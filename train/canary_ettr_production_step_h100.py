#!/usr/bin/env python3
"""Run two isolated production ETTR updates on one H100.

The canary uses deterministic synthetic causal rectangles and an in-memory
packet-sufficiency index. It never reads training shards or writes a model
checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Mapping, Sequence

import torch

from ettr_data_contract import (
    ETTR_CONTINUATION_SCHEMA,
    ETTRContinuationManifest,
    ETTRPacketSufficiencyIndex,
)
from ettr_optimization import ETTROptimizerBundle, ETTROptimizerConfig
from ettr_train_step import ETTRTrainStep, ETTRTrainStepConfig
from profile_ettr_h100 import (
    ProfileSettings,
    _device_batches,
    _model_from_checkpoint,
    _objective_config,
    _parameter_sha256,
    load_checkpoint_read_only,
    require_h100,
    sha256_file,
    synthetic_batches,
)


SCHEMA = "shohin-ettr-production-step-h100-canary-v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_LOSS_FIELDS = (
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


class ETTRProductionCanaryError(RuntimeError):
    """The production-step H100 canary failed closed."""


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


def _write_no_replace(path: Path, payload: bytes) -> None:
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


def _manifest(
    index: ETTRPacketSufficiencyIndex,
    *,
    checkpoint_sha256: str,
) -> ETTRContinuationManifest:
    receipt = index.receipt
    manifest = ETTRContinuationManifest(
        schema=ETTR_CONTINUATION_SCHEMA,
        protected_checkpoint_sha256=checkpoint_sha256,
        tokenizer_sha256="1" * 64,
        qualification_payload_sha256="2" * 64,
        hybrid_payload_sha256="3" * 64,
        train_rows=index.train_rows,
        validation_rows=index.validation_rows,
        train_payload_sha256=index.train_payload_sha256,
        validation_payload_sha256=index.validation_payload_sha256,
        dataset_sha256=ETTRContinuationManifest.combined_dataset_sha256(
            index.train_payload_sha256,
            index.validation_payload_sha256,
        ),
        packet_sufficiency_train_batches=index.train_batches,
        packet_sufficiency_validation_batches=index.validation_batches,
        packet_sufficiency_rows=receipt.rows,
        packet_sufficiency_unique_contexts=receipt.unique_contexts,
        packet_sufficiency_train_contexts=index.train_contexts,
        packet_sufficiency_validation_contexts=index.validation_contexts,
        packet_sufficiency_context_sha256=receipt.context_sha256,
        packet_sufficiency_target_bound_sha256=receipt.target_bound_sha256,
        source_deleted=True,
        immutable_snapshot=True,
        live_writer_input=False,
        family_label_fields=(),
    )
    manifest.validate()
    return manifest


def _receipt_values(receipt: object) -> dict[str, float]:
    values = {
        name: float(getattr(receipt, name).detach().float().cpu())
        for name in _LOSS_FIELDS
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise ETTRProductionCanaryError("canary emitted a nonfinite metric")
    return values


def run(
    *,
    output: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
    expected_step: int,
    source_commit: str,
    compile_mode: str,
) -> Mapping[str, object]:
    if (
        _HEX64.fullmatch(checkpoint_sha256) is None
        or _HEX40.fullmatch(source_commit) is None
        or expected_step != 300_000
    ):
        raise ETTRProductionCanaryError("canary identity argument differs")
    output = output.resolve()
    try:
        output.mkdir(mode=0o700, parents=True)
    except FileExistsError as exc:
        raise ETTRProductionCanaryError(
            f"refusing existing canary output: {output}"
        ) from exc

    device = torch.device("cuda", 0)
    device_receipt = require_h100(device)
    checkpoint_before = sha256_file(checkpoint)
    payload, _ = load_checkpoint_read_only(
        checkpoint,
        expected_sha256=checkpoint_sha256,
        expected_step=expected_step,
    )
    model = _model_from_checkpoint(payload, seed=2026072901)
    objective_config = _objective_config(model)
    settings = ProfileSettings(
        mode="h100",
        batch_size=16,
        microsteps=1,
        warmup_updates=1,
        measured_updates=1,
        world_tokens=256,
        command_tokens=64,
        query_tokens=128,
        reactor_steps=4,
        learning_rate=1e-4,
        seed=2026072902,
        train_scope="all",
        compile_mode=compile_mode,
    )
    settings.validate()
    train_batches, train_batch_sha256 = synthetic_batches(
        settings,
        reactor_config=model.config,
        objective_config=objective_config,
    )
    validation_batches, validation_batch_sha256 = synthetic_batches(
        replace(settings, seed=settings.seed + 1),
        reactor_config=model.config,
        objective_config=objective_config,
    )
    index = ETTRPacketSufficiencyIndex.from_splits(
        train_batches,
        validation_batches,
    )
    manifest = _manifest(
        index,
        checkpoint_sha256=checkpoint_sha256,
    )
    train_batch = replace(
        train_batches[0],
        manifest_sha256=manifest.sha256(),
        dataset_sha256=manifest.dataset_sha256,
    )
    validation_batch = replace(
        validation_batches[0],
        manifest_sha256=manifest.sha256(),
        dataset_sha256=manifest.dataset_sha256,
    )
    index.verify_train((train_batch,))
    index.verify_validation((validation_batch,))

    model.to(device=device, dtype=torch.bfloat16)
    model.train()
    optimizer = ETTROptimizerBundle(
        model,
        ETTROptimizerConfig(
            train_base=True,
            warmup_updates=2_000,
            total_updates=300_000,
        ),
    )
    step = ETTRTrainStep(
        model,
        optimizer,
        objective_config,
        manifest=manifest,
        packet_sufficiency=index,
        manifest_sha256=manifest.sha256(),
        step_config=ETTRTrainStepConfig(
            gradient_accumulation_steps=1,
            compile_backend="inductor",
            compile_mode=compile_mode,
        ),
    )
    device_batch = _device_batches((train_batch,), device)[0]
    initial_parameter_sha256 = _parameter_sha256(model)
    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    receipts = []
    for _ in range(2):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        receipt = step.update((device_batch,))
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - started)
        receipts.append(_receipt_values(receipt))
    final_parameter_sha256 = _parameter_sha256(model)
    checkpoint_after = sha256_file(checkpoint)
    encoded_tokens = (
        settings.batch_size
        * (
            settings.world_tokens
            + 2 * settings.command_tokens
            + 3 * settings.query_tokens
        )
    )
    if (
        optimizer.next_update != 2
        or initial_parameter_sha256 == final_parameter_sha256
        or checkpoint_before != checkpoint_after
        or checkpoint_after != checkpoint_sha256
    ):
        raise ETTRProductionCanaryError(
            "canary update or protected-checkpoint custody differs"
        )
    report = {
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256_after": checkpoint_after,
            "sha256_before": checkpoint_before,
            "step": expected_step,
            "unchanged": checkpoint_before == checkpoint_after,
        },
        "compile_backend": "inductor",
        "compile_mode": compile_mode,
        "device": device_receipt,
        "encoded_tokens_per_update": encoded_tokens,
        "final_parameter_sha256": final_parameter_sha256,
        "initial_parameter_sha256": initial_parameter_sha256,
        "manifest_sha256": manifest.sha256(),
        "optimizer": {
            "config": asdict(optimizer.config),
            "next_update": optimizer.next_update,
            "receipt": asdict(optimizer.receipt),
        },
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "receipts": receipts,
        "schema": SCHEMA,
        "source": {
            "commit": source_commit,
            "ettr_train_step_sha256": sha256_file(
                Path(__file__).with_name("ettr_train_step.py")
            ),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "status": "pass",
        "timings_seconds": timings,
        "train_batch_sha256": train_batch_sha256,
        "validation_batch_sha256": validation_batch_sha256,
        "warm_steady_encoded_tokens_per_second": (
            encoded_tokens / timings[-1]
        ),
    }
    _write_no_replace(output / "report.json", _canonical_bytes(report))
    output.chmod(0o500)
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
        ),
        default="default",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    report = run(
        output=arguments.output,
        checkpoint=arguments.checkpoint,
        checkpoint_sha256=arguments.checkpoint_sha256,
        expected_step=arguments.expected_step,
        source_commit=arguments.source_commit,
        compile_mode=arguments.compile_mode,
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
