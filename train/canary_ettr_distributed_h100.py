#!/usr/bin/env python3
"""Exercise two synthetic ETTR updates across multiple H100 ranks."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Sequence

import torch
import torch.distributed as dist

from canary_ettr_production_step_h100 import (
    _canonical_bytes,
    _manifest,
    _receipt_values,
    _write_no_replace,
)
from ettr_data_contract import ETTRPacketSufficiencyIndex
from ettr_distributed import ETTRDistributedGradientAverager
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


SCHEMA = "shohin-ettr-distributed-production-step-h100-canary-v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ETTRDistributedCanaryError(RuntimeError):
    """The distributed ETTR canary cannot preserve its contract."""


def _environment(expected_world_size: int) -> tuple[int, int, int]:
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    except (KeyError, ValueError) as exc:
        raise ETTRDistributedCanaryError(
            "distributed environment differs"
        ) from exc
    if (
        world_size != expected_world_size
        or not 2 <= world_size <= 20
        or not 0 <= rank < world_size
        or local_rank < 0
        or not torch.cuda.is_available()
    ):
        raise ETTRDistributedCanaryError(
            "distributed rank geometry differs"
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=15),
    )
    if dist.get_rank() != rank or dist.get_world_size() != world_size:
        raise ETTRDistributedCanaryError(
            "distributed process group differs"
        )
    return rank, world_size, local_rank


def _all_gather(value: object, world_size: int) -> list[object]:
    values: list[object] = [None] * world_size
    dist.all_gather_object(values, value)
    return values


def _settings(seed: int, compile_mode: str) -> ProfileSettings:
    value = ProfileSettings(
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
        seed=seed,
        train_scope="architecture",
        compile_mode=compile_mode,
    )
    value.validate()
    return value


def run(
    *,
    output: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
    expected_step: int,
    source_commit: str,
    expected_world_size: int,
    compile_mode: str,
) -> dict[str, object] | None:
    if (
        _HEX64.fullmatch(checkpoint_sha256) is None
        or _HEX40.fullmatch(source_commit) is None
        or expected_step != 300_000
        or not 2 <= expected_world_size <= 20
    ):
        raise ETTRDistributedCanaryError("canary identity argument differs")
    rank, world_size, local_rank = _environment(expected_world_size)
    device = torch.device("cuda", local_rank)
    try:
        device_receipt = require_h100(device)
        checkpoint_before = sha256_file(checkpoint)
        payload, _ = load_checkpoint_read_only(
            checkpoint,
            expected_sha256=checkpoint_sha256,
            expected_step=expected_step,
        )
        model = _model_from_checkpoint(payload, seed=2026072901)
        objective_config = _objective_config(model)
        settings = _settings(2026072902, compile_mode)
        train_batches = []
        train_hashes = []
        for index in range(world_size):
            batches, digest = synthetic_batches(
                replace(settings, seed=settings.seed + index),
                reactor_config=model.config,
                objective_config=objective_config,
            )
            train_batches.append(batches[0])
            train_hashes.append(digest)
        validation_batches, validation_hash = synthetic_batches(
            replace(settings, seed=settings.seed + 10_000),
            reactor_config=model.config,
            objective_config=objective_config,
        )
        index = ETTRPacketSufficiencyIndex.from_splits(
            tuple(train_batches),
            validation_batches,
        )
        manifest = _manifest(
            index,
            checkpoint_sha256=checkpoint_sha256,
        )
        train_batches = [
            replace(
                batch,
                manifest_sha256=manifest.sha256(),
                dataset_sha256=manifest.dataset_sha256,
            )
            for batch in train_batches
        ]
        validation_batch = replace(
            validation_batches[0],
            manifest_sha256=manifest.sha256(),
            dataset_sha256=manifest.dataset_sha256,
        )
        index.verify_train(tuple(train_batches))
        index.verify_validation((validation_batch,))

        model.to(device=device, dtype=torch.bfloat16)
        model.train()
        optimizer = ETTROptimizerBundle(
            model,
            ETTROptimizerConfig(
                train_base=False,
                warmup_updates=2_000,
                total_updates=300_000,
            ),
        )
        averager = ETTRDistributedGradientAverager(
            world_size=world_size,
            all_reduce_sum=lambda value: dist.all_reduce(
                value,
                op=dist.ReduceOp.SUM,
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
            gradient_synchronizer=averager,
        )
        device_batch = _device_batches((train_batches[rank],), device)[0]
        initial_parameter_sha256 = _parameter_sha256(model)
        torch.cuda.reset_peak_memory_stats(device)
        timings = []
        receipts = []
        for _ in range(2):
            dist.barrier()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            receipt = step.update((device_batch,))
            torch.cuda.synchronize(device)
            elapsed = torch.tensor(
                time.perf_counter() - started,
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            timings.append(float(elapsed.cpu()))
            receipts.append(_receipt_values(receipt))
        final_parameter_sha256 = _parameter_sha256(model)
        checkpoint_after = sha256_file(checkpoint)
        rank_result = {
            "checkpoint_sha256_after": checkpoint_after,
            "device": device_receipt,
            "final_parameter_sha256": final_parameter_sha256,
            "initial_parameter_sha256": initial_parameter_sha256,
            "local_rank": local_rank,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "rank": rank,
            "receipts": receipts,
            "train_batch_sha256": train_hashes[rank],
        }
        gathered = _all_gather(rank_result, world_size)
        if (
            optimizer.next_update != 2
            or checkpoint_before != checkpoint_sha256
            or checkpoint_after != checkpoint_sha256
            or initial_parameter_sha256 == final_parameter_sha256
            or len(
                {
                    str(value["initial_parameter_sha256"])
                    for value in gathered
                }
            )
            != 1
            or len(
                {
                    str(value["final_parameter_sha256"])
                    for value in gathered
                }
            )
            != 1
        ):
            raise ETTRDistributedCanaryError(
                "distributed update or rank consistency differs"
            )
        if rank != 0:
            return None
        output = output.resolve()
        try:
            output.mkdir(mode=0o700, parents=True)
        except FileExistsError as exc:
            raise ETTRDistributedCanaryError(
                f"refusing existing canary output: {output}"
            ) from exc
        report = {
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256_after": checkpoint_after,
                "sha256_before": checkpoint_before,
                "step": expected_step,
                "unchanged": True,
            },
            "compile_backend": "inductor",
            "compile_mode": compile_mode,
            "manifest_sha256": manifest.sha256(),
            "optimizer": {
                "config": asdict(optimizer.config),
                "next_update": optimizer.next_update,
                "receipt": asdict(optimizer.receipt),
            },
            "rank_results": gathered,
            "schema": SCHEMA,
            "source": {
                "commit": source_commit,
                "ettr_distributed_sha256": sha256_file(
                    Path(__file__).with_name("ettr_distributed.py")
                ),
                "ettr_train_step_sha256": sha256_file(
                    Path(__file__).with_name("ettr_train_step.py")
                ),
                "script_sha256": sha256_file(Path(__file__)),
            },
            "status": "pass",
            "timings_max_rank_seconds": timings,
            "validation_batch_sha256": validation_hash,
            "world_size": world_size,
        }
        report_bytes = _canonical_bytes(report)
        _write_no_replace(output / "report.json", report_bytes)
        report_sha256 = hashlib.sha256(report_bytes).hexdigest()
        _write_no_replace(
            output / "SHA256SUMS",
            f"{report_sha256}  report.json\n".encode("ascii"),
        )
        (output / "report.json").chmod(0o400)
        (output / "SHA256SUMS").chmod(0o400)
        output.chmod(0o500)
        return report
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--expected-world-size", type=int, required=True)
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
        expected_world_size=arguments.expected_world_size,
        compile_mode=arguments.compile_mode,
    )
    if report is not None:
        print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
