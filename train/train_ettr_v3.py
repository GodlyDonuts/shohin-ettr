#!/usr/bin/env python3
"""Fail-closed distributed trainer for one admitted ETTR-IL-v3 release."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
from typing import Sequence

import numpy as np
import torch
import torch.distributed as dist

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TheoryReactorConfig,
)
from ettr_checkpoint import (
    BaseProvenance,
    DataStreamState,
    EpisodeLifecycleState,
    TrainingProgress,
    load_ettr_checkpoint,
    load_protected_base_model,
    save_ettr_checkpoint,
)
from ettr_distributed import (
    ETTRDistributedCursor,
    ETTRDistributedGradientAverager,
)
from ettr_objectives import ETTRObjectiveConfig, ETTRObjectiveWeights
from ettr_optimization import ETTROptimizerBundle, ETTROptimizerConfig
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_train_step import ETTRTrainStep, ETTRTrainStepConfig
from ettr_v3_streaming import ETTRV3StreamingRelease


_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE = _ROOT / "pipeline"
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from ettr_il_v3_protocol import CHARGED_POSITIONS_PER_ROW  # noqa: E402


RUN_SCHEMA = "shohin-ettr-il-v3-distributed-run-v1"
CURSOR_SCHEMA = "shohin-ettr-il-v3-distributed-cursor-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


class ETTRV3TrainerError(RuntimeError):
    """The distributed trainer cannot preserve its frozen run contract."""


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


def _write_no_replace(path: Path, payload: bytes, mode: int = 0o400) -> str:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, mode)
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


def _distributed_environment() -> tuple[int, int, int]:
    values = tuple(
        int(os.environ.get(name, default))
        for name, default in (
            ("RANK", "0"),
            ("WORLD_SIZE", "1"),
            ("LOCAL_RANK", "0"),
        )
    )
    rank, world_size, local_rank = values
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ETTRV3TrainerError("distributed environment differs")
    if not torch.cuda.is_available():
        raise ETTRV3TrainerError("ETTR v3 training requires CUDA")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=15),
        )
        if dist.get_rank() != rank or dist.get_world_size() != world_size:
            raise ETTRV3TrainerError("distributed process group differs")
    return rank, world_size, local_rank


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _all_reduce_sum(value: torch.Tensor, world_size: int) -> None:
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)


def _broadcast_rank_zero_error(
    error: str | None,
    *,
    rank: int,
    world_size: int,
) -> None:
    if world_size == 1:
        if error is not None:
            raise ETTRV3TrainerError(error)
        return
    values = [error if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    if values[0] is not None:
        raise ETTRV3TrainerError(str(values[0]))


def _seed_update(
    *,
    data_seed: int,
    epoch: int,
    position: int,
    rank: int,
) -> int:
    digest = hashlib.sha256(
        f"{data_seed}\x1f{epoch}\x1f{position}\x1f{rank}".encode("ascii")
    ).digest()
    seed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return seed


def _data_state(
    *,
    stream: ETTRV3StreamingRelease,
    release_sha256: str,
    cursor: ETTRDistributedCursor,
    data_seed: int,
    world_size: int,
    accumulation: int,
    optimizer_step: int,
    compile_backend: str | None,
    compile_mode: str | None,
    hard_transactions: bool,
    nll_gradient_cap: float | None = None,
    query_binding_weight: float = 1.0,
) -> DataStreamState:
    sampler_state = {
        "accumulation": accumulation,
        "compile_backend": compile_backend,
        "compile_mode": compile_mode,
        "hard_transactions": hard_transactions,
        "consumed_stream_batches": optimizer_step
        * world_size
        * accumulation,
        "release_file_sha256": release_sha256,
        "schema": CURSOR_SCHEMA,
        "world_size": world_size,
    }
    if nll_gradient_cap is not None:
        sampler_state["nll_gradient_cap"] = nll_gradient_cap
    if query_binding_weight != 1.0:
        sampler_state["query_binding_weight"] = query_binding_weight
    return DataStreamState(
        manifest_sha256=stream.manifest.sha256(),
        dataset_sha256=stream.manifest.dataset_sha256,
        generation=0,
        seed=data_seed,
        epoch=cursor.epoch,
        shard_index=0,
        sample_index=cursor.position,
        token_offset=0,
        sampler_state=sampler_state,
    )


def _validate_resume_cursor(
    state: DataStreamState,
    *,
    stream: ETTRV3StreamingRelease,
    release_sha256: str,
    data_seed: int,
    world_size: int,
    accumulation: int,
    optimizer_step: int,
    compile_backend: str | None,
    compile_mode: str | None,
    hard_transactions: bool,
    nll_gradient_cap: float | None = None,
    query_binding_weight: float = 1.0,
) -> ETTRDistributedCursor:
    expected_sampler = {
        "accumulation": accumulation,
        "compile_backend": compile_backend,
        "compile_mode": compile_mode,
        "hard_transactions": hard_transactions,
        "consumed_stream_batches": optimizer_step
        * world_size
        * accumulation,
        "release_file_sha256": release_sha256,
        "schema": CURSOR_SCHEMA,
        "world_size": world_size,
    }
    if nll_gradient_cap is not None:
        expected_sampler["nll_gradient_cap"] = nll_gradient_cap
    if query_binding_weight != 1.0:
        expected_sampler["query_binding_weight"] = query_binding_weight
    if (
        state.manifest_sha256 != stream.manifest.sha256()
        or state.dataset_sha256 != stream.manifest.dataset_sha256
        or state.generation != 0
        or state.seed != data_seed
        or state.shard_index != 0
        or state.token_offset != 0
        or dict(state.sampler_state) != expected_sampler
    ):
        raise ETTRV3TrainerError(
            "resume data stream differs from this admitted run"
        )
    cursor = ETTRDistributedCursor(state.epoch, state.sample_index)
    cursor.validate(
        core_batches=len(stream.records["train"]),
        world_size=world_size,
        accumulation=accumulation,
    )
    return cursor


def _metrics(
    receipt: object,
    *,
    device: torch.device,
    world_size: int,
) -> dict[str, float]:
    names = (
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
    values = torch.stack(
        [getattr(receipt, name).detach().float() for name in names]
    ).to(device)
    _all_reduce_sum(values, world_size)
    values.div_(world_size)
    return {
        name: float(value)
        for name, value in zip(names, values.cpu().tolist(), strict=True)
    }


def _checkpoint(
    *,
    output: Path,
    model: EndogenousTypedTheoryReactorGPT,
    protected_base: BaseProvenance,
    optimizer: ETTROptimizerBundle,
    stream: ETTRV3StreamingRelease,
    release_sha256: str,
    cursor: ETTRDistributedCursor,
    data_seed: int,
    world_size: int,
    accumulation: int,
    compile_backend: str | None,
    compile_mode: str | None,
    hard_transactions: bool,
    nll_gradient_cap: float | None,
    query_binding_weight: float,
    rank: int,
) -> None:
    if rank != 0:
        return
    update = optimizer.next_update
    charged_per_batch = (
        int(stream.release["training_rows_per_batch"])
        * CHARGED_POSITIONS_PER_ROW
    )
    progress = TrainingProgress(
        global_step=protected_base.step + update,
        optimizer_step=update,
        micro_step=0,
        gradient_accumulation_steps=accumulation,
        tokens_seen=update * world_size * accumulation * charged_per_batch,
    )
    lifecycle = EpisodeLifecycleState(
        episode_index=update * world_size * accumulation,
        phase="between_episodes",
        episode_sha256=None,
        token_offset=0,
        reactor_step=0,
        source_deleted=False,
        committed=False,
        halted=False,
    )
    path = output / f"checkpoint-update-{update:07d}.pt"
    digest = save_ettr_checkpoint(
        path,
        model=model,
        protected_base=protected_base,
        optimizer=optimizer,
        scheduler=None,
        progress=progress,
        data_stream=_data_state(
            stream=stream,
            release_sha256=release_sha256,
            cursor=cursor,
            data_seed=data_seed,
            world_size=world_size,
            accumulation=accumulation,
            optimizer_step=update,
            compile_backend=compile_backend,
            compile_mode=compile_mode,
            hard_transactions=hard_transactions,
            nll_gradient_cap=nll_gradient_cap,
            query_binding_weight=query_binding_weight,
        ),
        episode_lifecycle=lifecycle,
    )
    _write_no_replace(
        path.with_suffix(".json"),
        _canonical_bytes(
            {
                "checkpoint": path.name,
                "checkpoint_sha256": digest,
                "optimizer_step": update,
                "schema": "shohin-ettr-checkpoint-sidecar-v1",
            }
        ),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--protected-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--architecture-seed", type=int, default=2026072801)
    parser.add_argument("--data-seed", type=int, default=2026072802)
    parser.add_argument("--total-updates", type=int, default=300_000)
    parser.add_argument("--warmup-updates", type=int, default=2_000)
    parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
    )
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--soft-transactions", action="store_true")
    parser.add_argument("--nll-gradient-cap", type=float)
    parser.add_argument("--query-binding-weight", type=float, default=1.0)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if (
        _HEX64.fullmatch(args.release_sha256) is None
        or _HEX40.fullmatch(args.source_commit) is None
        or args.updates < 1
        or args.accumulation < 1
        or args.checkpoint_every < 1
        or args.log_every < 1
        or args.total_updates < 1
        or args.warmup_updates < 0
        or (
            args.nll_gradient_cap is not None
            and (
                not math.isfinite(args.nll_gradient_cap)
                or args.nll_gradient_cap <= 0.0
                or args.soft_transactions
            )
        )
        or not math.isfinite(args.query_binding_weight)
        or not 0.0 < args.query_binding_weight <= 1_000.0
        or (args.resume_checkpoint is None) != (args.resume_sha256 is None)
        or (
            args.resume_sha256 is not None
            and _HEX64.fullmatch(args.resume_sha256) is None
        )
    ):
        raise ETTRV3TrainerError("ETTR v3 trainer arguments differ")

    rank, world_size, local_rank = _distributed_environment()
    device = torch.device("cuda", local_rank)
    try:
        stream = ETTRV3StreamingRelease(
            args.release_root,
            expected_release_sha256=args.release_sha256,
            data_root=args.data_root,
            tokenizer_path=args.tokenizer,
        )
        verify_error = None
        if rank == 0:
            try:
                stream.verify_source_shards()
            except BaseException as exc:
                verify_error = (
                    f"rank-zero source verification failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        _broadcast_rank_zero_error(
            verify_error,
            rank=rank,
            world_size=world_size,
        )
        if (
            stream.manifest.protected_checkpoint_sha256
            != stream.release["protected_checkpoint_sha256"]
        ):
            raise ETTRV3TrainerError(
                "release protected-checkpoint identity differs"
            )

        base, protected_base = load_protected_base_model(
            args.protected_checkpoint
        )
        if (
            protected_base.checkpoint_sha256
            != stream.manifest.protected_checkpoint_sha256
        ):
            raise ETTRV3TrainerError(
                "protected checkpoint differs from the ETTR release"
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
                train_base=not args.freeze_base,
                warmup_updates=args.warmup_updates,
                total_updates=args.total_updates,
            ),
        )
        packet_index = ETTRDiskPacketSufficiencyIndex(
            stream.packet_index_root
        )
        averager = ETTRDistributedGradientAverager(
            world_size=world_size,
            all_reduce_sum=lambda value: _all_reduce_sum(
                value,
                world_size,
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
            packet_sufficiency=packet_index,
            manifest_sha256=stream.manifest.sha256(),
            objective_weights=ETTRObjectiveWeights(
                world_query_binding=args.query_binding_weight,
                command_query_binding=args.query_binding_weight,
            ),
            step_config=ETTRTrainStepConfig(
                gradient_accumulation_steps=args.accumulation,
                hard_transactions=not args.soft_transactions,
                compile_backend=(
                    None if args.compile_mode is None else "inductor"
                ),
                compile_mode=args.compile_mode,
            ),
            gradient_synchronizer=averager,
        )

        cursor = ETTRDistributedCursor(epoch=0, position=0)
        if args.resume_checkpoint is not None:
            resumed = load_ettr_checkpoint(
                args.resume_checkpoint,
                expected_sha256=args.resume_sha256,
                model=model,
                protected_base=protected_base,
                optimizer=optimizer,
                scheduler=None,
            )
            if (
                resumed.progress.global_step
                != protected_base.step + optimizer.next_update
                or resumed.progress.gradient_accumulation_steps
                != args.accumulation
            ):
                raise ETTRV3TrainerError(
                    "resume training progress differs"
                )
            cursor = _validate_resume_cursor(
                resumed.data_stream,
                stream=stream,
                release_sha256=args.release_sha256,
                data_seed=args.data_seed,
                world_size=world_size,
                accumulation=args.accumulation,
                optimizer_step=optimizer.next_update,
                compile_backend=(
                    None if args.compile_mode is None else "inductor"
                ),
                compile_mode=args.compile_mode,
                hard_transactions=not args.soft_transactions,
                nll_gradient_cap=args.nll_gradient_cap,
                query_binding_weight=args.query_binding_weight,
            )
        cursor.validate(
            core_batches=len(stream.records["train"]),
            world_size=world_size,
            accumulation=args.accumulation,
        )
        target_update = optimizer.next_update + args.updates
        if target_update > args.total_updates:
            raise ETTRV3TrainerError(
                "requested run exceeds the frozen optimizer horizon"
            )

        if rank == 0:
            try:
                args.output.mkdir(mode=0o700, parents=True)
            except FileExistsError as exc:
                raise ETTRV3TrainerError(
                    "refusing an existing ETTR run output"
                ) from exc
            contract = {
                "accumulation": args.accumulation,
                "architecture_seed": args.architecture_seed,
                "data_seed": args.data_seed,
                "freeze_base": args.freeze_base,
                "hard_transactions": not args.soft_transactions,
                "nll_gradient_cap": args.nll_gradient_cap,
                "query_binding_weight": args.query_binding_weight,
                "compile_backend": (
                    None if args.compile_mode is None else "inductor"
                ),
                "compile_mode": args.compile_mode,
                "model_config": asdict(model.config),
                "optimizer_config": asdict(optimizer.config),
                "parameter_receipt": asdict(model.parameter_receipt()),
                "release_file_sha256": args.release_sha256,
                "release_source_commit": stream.release[
                    "source_commit"
                ],
                "resume_checkpoint_sha256": args.resume_sha256,
                "schema": RUN_SCHEMA,
                "source_commit": args.source_commit,
                "start_optimizer_step": optimizer.next_update,
                "target_optimizer_step": target_update,
                "world_size": world_size,
            }
            _write_no_replace(
                args.output / "run-contract.json",
                _canonical_bytes(contract),
            )
        _barrier(world_size)

        model.train()
        log_path = args.output / "train.jsonl"
        if rank == 0:
            _write_no_replace(log_path, b"", mode=0o600)
        _barrier(world_size)
        log_handle = log_path.open("ab", buffering=0) if rank == 0 else None
        try:
            while optimizer.next_update < target_update:
                active_epoch = cursor.epoch
                usable = cursor.validate(
                    core_batches=len(stream.records["train"]),
                    world_size=world_size,
                    accumulation=args.accumulation,
                )
                if usable == 0:
                    raise ETTRV3TrainerError(
                        "training split has no complete distributed update"
                    )
                iterator = stream.iter_positioned_batches(
                    "train",
                    rank=rank,
                    world_size=world_size,
                    epoch=cursor.epoch,
                    seed=args.data_seed,
                    start_position=cursor.position,
                    device=device,
                )
                while (
                    cursor.epoch == active_epoch
                    and cursor.position < usable
                    and optimizer.next_update < target_update
                ):
                    batches = []
                    for microstep in range(args.accumulation):
                        position, batch = next(iterator)
                        expected = (
                            cursor.position
                            + rank
                            + microstep * world_size
                        )
                        if position != expected:
                            raise ETTRV3TrainerError(
                                "rank-local stream position differs"
                            )
                        batches.append(batch)
                    update_seed = _seed_update(
                        data_seed=args.data_seed,
                        epoch=cursor.epoch,
                        position=cursor.position,
                        rank=rank,
                    )
                    receipt = step.update(tuple(batches))
                    cursor = cursor.advance(
                        core_batches=len(stream.records["train"]),
                        world_size=world_size,
                        accumulation=args.accumulation,
                    )
                    if (
                        receipt.optimizer_step % args.log_every == 0
                        or receipt.optimizer_step == target_update
                    ):
                        values = _metrics(
                            receipt,
                            device=device,
                            world_size=world_size,
                        )
                        if rank == 0 and log_handle is not None:
                            log_handle.write(
                                _canonical_bytes(
                                    {
                                        **values,
                                        "epoch": cursor.epoch,
                                        "learning_rate_scale": (
                                            receipt.learning_rate_scale
                                        ),
                                        "next_position": cursor.position,
                                        "optimizer_step": (
                                            receipt.optimizer_step
                                        ),
                                        "schema": (
                                            "shohin-ettr-train-metric-v1"
                                        ),
                                        "update_seed_rank_zero": update_seed,
                                    }
                                )
                            )
                    if (
                        receipt.optimizer_step % args.checkpoint_every == 0
                        or receipt.optimizer_step == target_update
                    ):
                        _barrier(world_size)
                        _checkpoint(
                            output=args.output,
                            model=model,
                            protected_base=protected_base,
                            optimizer=optimizer,
                            stream=stream,
                            release_sha256=args.release_sha256,
                            cursor=cursor,
                            data_seed=args.data_seed,
                            world_size=world_size,
                            accumulation=args.accumulation,
                            compile_backend=(
                                None
                                if args.compile_mode is None
                                else "inductor"
                            ),
                            compile_mode=args.compile_mode,
                            hard_transactions=not args.soft_transactions,
                            nll_gradient_cap=args.nll_gradient_cap,
                            query_binding_weight=args.query_binding_weight,
                            rank=rank,
                        )
                        _barrier(world_size)
        finally:
            if log_handle is not None:
                log_handle.close()
            packet_index.close()
        return 0
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
