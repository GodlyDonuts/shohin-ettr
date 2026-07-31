#!/usr/bin/env python3
"""Bounded one-H100 joint language/ETTR scientific canary.

This trainer deliberately has no resume path.  It tests whether native
co-adaptation is promising before a production exact-resume stream contract
is implemented.  Every input is hash-bound, every output is no-replace, and
the run report labels warm-start and random initialization distinctly.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import sys
import tempfile
from typing import Sequence

import numpy as np
import torch

from data import ShardLoader
from data_contract import resolve_training_data_contract
from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TheoryReactorConfig,
)
from ettr_checkpoint import load_protected_base_model
from ettr_distributed import ETTRDistributedCursor
from ettr_joint_stream import (
    ETTRJointPositionScheduler,
    ETTRJointScheduleConfig,
    GeneralLanguageStepConfig,
    GeneralLanguageUpdateStep,
)
from ettr_objectives import ETTRObjectiveConfig, ETTRObjectiveWeights
from ettr_optimization import ETTROptimizerBundle, ETTROptimizerConfig
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_train_step import ETTRTrainStep, ETTRTrainStepConfig
from ettr_v3_streaming import ETTRV3StreamingRelease
from model import GPT, GPTConfig
from workspace_checkpoint import file_sha256


_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE = _ROOT / "pipeline"
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from ettr_il_v3_protocol import CHARGED_POSITIONS_PER_ROW  # noqa: E402


RUN_SCHEMA = "shohin-ettr-joint-stream-canary-v1"
MODEL_SCHEMA = "shohin-ettr-joint-model-canary-v1"
REPORT_SCHEMA = "shohin-ettr-joint-stream-report-v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ETTRJointCanaryError(RuntimeError):
    """The bounded joint-stream experiment cannot preserve its contract."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _dataclass_contract(value: object) -> dict[str, object]:
    """Return a JSON-safe, exact receipt for a frozen dataclass config."""

    payload = asdict(value)
    return {
        key: str(item) if isinstance(item, torch.dtype) else item
        for key, item in payload.items()
    }


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


def _torch_save_no_replace(path: Path, payload: object) -> str:
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise ETTRJointCanaryError(
            "joint-model output destination differs"
        )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary_path, 0o400)
        os.link(temporary_path, path)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return file_sha256(path)


def _seed_update(
    *,
    seed: int,
    optimizer_step: int,
    stream: str,
) -> int:
    digest = hashlib.sha256(
        f"{seed}\x1f{optimizer_step}\x1f{stream}".encode("ascii")
    ).digest()
    update_seed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    random.seed(update_seed)
    np.random.seed(update_seed % (2**32))
    torch.manual_seed(update_seed)
    torch.cuda.manual_seed(update_seed)
    return update_seed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--ettr-data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--general-contract", type=Path)
    parser.add_argument("--general-contract-sha256")
    parser.add_argument(
        "--legacy-general-shard-dir",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--legacy-general-weight",
        type=float,
        action="append",
        default=[],
    )
    parser.add_argument("--protected-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--initialization",
        choices=("warm", "random"),
        required=True,
    )
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--general-batch-size", type=int, default=16)
    parser.add_argument("--general-position-weight", type=int, required=True)
    parser.add_argument("--ettr-position-weight", type=int, required=True)
    parser.add_argument("--architecture-seed", type=int, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--total-updates", type=int, default=300_000)
    parser.add_argument("--warmup-updates", type=int, default=2_000)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--nll-gradient-cap", type=float)
    parser.add_argument("--query-binding-weight", type=float, default=1.0)
    parser.add_argument(
        "--gradient-clip-mode",
        choices=("global", "owner"),
        default="global",
    )
    parser.add_argument("--deep-verify-general", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    paths = (
        args.release_root,
        args.ettr_data_root,
        args.tokenizer,
        args.protected_checkpoint,
        args.output,
    )
    if args.general_contract is not None:
        paths += (args.general_contract,)
    contract_mode = (
        args.general_contract is not None
        or args.general_contract_sha256 is not None
    )
    legacy_mode = bool(
        args.legacy_general_shard_dir
        or args.legacy_general_weight
    )
    if (
        _HEX64.fullmatch(args.release_sha256) is None
        or _HEX40.fullmatch(args.source_commit) is None
        or any(not path.is_absolute() for path in paths)
        or contract_mode == legacy_mode
        or (
            contract_mode
            and (
                args.general_contract is None
                or args.general_contract_sha256 is None
                or _HEX64.fullmatch(args.general_contract_sha256) is None
            )
        )
        or (
            legacy_mode
            and (
                args.deep_verify_general
                or
                not args.legacy_general_shard_dir
                or len(args.legacy_general_shard_dir)
                != len(args.legacy_general_weight)
                or any(
                    not path.is_absolute()
                    for path in args.legacy_general_shard_dir
                )
                or any(
                    not math.isfinite(weight) or weight <= 0
                    for weight in args.legacy_general_weight
                )
            )
        )
        or args.updates < 2
        or args.general_batch_size < 1
        or args.general_position_weight < 1
        or args.ettr_position_weight < 1
        or args.total_updates < args.updates
        or args.warmup_updates < 0
        or args.warmup_updates >= args.total_updates
        or args.log_every < 1
        or not 0 <= args.architecture_seed < 2**63
        or not 0 <= args.base_seed < 2**63
        or not 0 <= args.data_seed < 2**63
        or (
            args.nll_gradient_cap is not None
            and (
                not math.isfinite(args.nll_gradient_cap)
                or args.nll_gradient_cap <= 0
            )
        )
        or not math.isfinite(args.query_binding_weight)
        or not 0 < args.query_binding_weight <= 1_000
    ):
        raise ETTRJointCanaryError(
            "joint-stream canary arguments differ"
        )


def _legacy_general_resolution(
    shard_dirs: Sequence[Path],
    weights: Sequence[float],
    *,
    tokenizer_sha256: str,
) -> dict[str, object]:
    corpora: list[dict[str, object]] = []
    seen_paths: set[Path] = set()
    for index, (shard_dir, weight) in enumerate(
        zip(shard_dirs, weights, strict=True)
    ):
        try:
            directory_metadata = shard_dir.lstat()
        except OSError as exc:
            raise ETTRJointCanaryError(
                f"legacy general shard directory {index} is unreadable"
            ) from exc
        if (
            stat.S_ISLNK(directory_metadata.st_mode)
            or not stat.S_ISDIR(directory_metadata.st_mode)
            or shard_dir in seen_paths
        ):
            raise ETTRJointCanaryError(
                f"legacy general shard directory {index} differs"
            )
        shards = sorted(shard_dir.glob("*.u16.zst"))
        if not shards:
            raise ETTRJointCanaryError(
                f"legacy general shard directory {index} is empty"
            )
        inventory = []
        for path in shards:
            metadata = path.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size <= 0
            ):
                raise ETTRJointCanaryError(
                    f"legacy general shard {index} differs"
                )
            inventory.append(
                {
                    "device": metadata.st_dev,
                    "inode": metadata.st_ino,
                    "mtime_ns": metadata.st_mtime_ns,
                    "name": path.name,
                    "size": metadata.st_size,
                }
            )
        manifest_path = shard_dir / "manifest.json"
        manifest_sha256 = (
            file_sha256(manifest_path)
            if manifest_path.is_file() and not manifest_path.is_symlink()
            else None
        )
        corpora.append(
            {
                "inventory": inventory,
                "manifest_sha256": manifest_sha256,
                "path": str(shard_dir),
                "weight": float(weight),
            }
        )
        seen_paths.add(shard_dir)
    total_weight = sum(float(value) for value in weights)
    identity = {
        "corpora": corpora,
        "schema": "shohin-legacy-general-canary-inventory-v1",
        "tokenizer_sha256": tokenizer_sha256,
    }
    inventory_sha256 = hashlib.sha256(
        _canonical_bytes(identity)
    ).hexdigest()
    return {
        "contract": None,
        "contract_payload_sha256": None,
        "corpora": corpora,
        "domain_weights": [
            float(value) / total_weight for value in weights
        ],
        "inventory_sha256": inventory_sha256,
        "legacy_scientific_control": True,
        "shard_dirs": [str(path) for path in shard_dirs],
        "tokenizer_sha256": tokenizer_sha256,
        "tokenizer_vocab_size": None,
    }


def _build_model(
    args: argparse.Namespace,
) -> tuple[EndogenousTypedTheoryReactorGPT, dict[str, object]]:
    protected, provenance = load_protected_base_model(
        args.protected_checkpoint
    )
    if args.initialization == "warm":
        base = protected
        base_identity = {
            "initialization": "protected-step-300k-weights",
            "protected_base_state_sha256": provenance.base_state_sha256,
            "base_seed": None,
        }
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.base_seed)
            base = GPT(GPTConfig(**provenance.base_config))
        base_identity = {
            "initialization": "deterministic-random-weights",
            "protected_base_state_sha256": None,
            "base_seed": args.base_seed,
        }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.architecture_seed)
        model = EndogenousTypedTheoryReactorGPT(
            base,
            TheoryReactorConfig(),
        )
    return model, {
        **base_identity,
        "protected_checkpoint_sha256": provenance.checkpoint_sha256,
        "protected_config_sha256": provenance.config_sha256,
        "protected_step": provenance.step,
    }


def _ettr_metric_payload(receipt: object) -> dict[str, float]:
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
        "base_gradient_norm",
        "architecture_gradient_norm",
    )
    return {
        name: float(getattr(receipt, name).detach().float().cpu())
        for name in names
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ETTRJointCanaryError(
            "bounded joint-stream canary requires exactly one H100"
        )
    if not torch.cuda.is_available():
        raise ETTRJointCanaryError(
            "joint-stream canary requires CUDA"
        )
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    if "H100" not in torch.cuda.get_device_name(device).upper():
        raise ETTRJointCanaryError(
            "joint-stream canary requires an H100"
        )
    if args.output.exists() or args.output.is_symlink():
        raise ETTRJointCanaryError(
            "refusing an existing joint-stream output"
        )

    stream = ETTRV3StreamingRelease(
        args.release_root,
        expected_release_sha256=args.release_sha256,
        data_root=args.ettr_data_root,
        tokenizer_path=args.tokenizer,
    )
    source_verification = stream.verify_source_shards()
    if args.general_contract is not None:
        general = resolve_training_data_contract(
            args.general_contract,
            expected_sha256=args.general_contract_sha256,
            deep_verify=args.deep_verify_general,
        )
    else:
        general = _legacy_general_resolution(
            args.legacy_general_shard_dir,
            args.legacy_general_weight,
            tokenizer_sha256=stream.manifest.tokenizer_sha256,
        )
    if (
        general["tokenizer_sha256"]
        != stream.manifest.tokenizer_sha256
    ):
        raise ETTRJointCanaryError(
            "general and ETTR tokenizer identities differ"
        )
    active_domains = sum(
        weight > 0 for weight in general["domain_weights"]
    )
    if args.general_batch_size < active_domains:
        raise ETTRJointCanaryError(
            "general batch does not cover every admitted domain"
        )

    model, initialization = _build_model(args)
    if (
        general["tokenizer_vocab_size"] is not None
        and general["tokenizer_vocab_size"] != model.base.cfg.vocab_size
    ):
        raise ETTRJointCanaryError(
            "general tokenizer vocabulary differs from the model"
        )
    model.to(device=device, dtype=torch.bfloat16)
    optimizer = ETTROptimizerBundle(
        model,
        ETTROptimizerConfig(
            train_base=True,
            warmup_updates=args.warmup_updates,
            total_updates=args.total_updates,
        ),
    )
    general_step_config = GeneralLanguageStepConfig(
        gradient_accumulation_steps=1,
    )
    language_step = GeneralLanguageUpdateStep(
        model,
        optimizer,
        step_config=general_step_config,
    )
    packet_index = ETTRDiskPacketSufficiencyIndex(
        stream.packet_index_root
    )
    objective_config = ETTRObjectiveConfig(
        vocab_size=model.base.cfg.vocab_size,
        nll_gradient_cap=args.nll_gradient_cap,
    )
    objective_weights = ETTRObjectiveWeights(
        world_query_binding=args.query_binding_weight,
        command_query_binding=args.query_binding_weight,
    )
    ettr_step_config = ETTRTrainStepConfig(
        gradient_accumulation_steps=1,
        gradient_clip_mode=args.gradient_clip_mode,
        hard_transactions=True,
    )
    ettr_step = ETTRTrainStep(
        model,
        optimizer,
        objective_config,
        manifest=stream.manifest,
        packet_sufficiency=packet_index,
        manifest_sha256=stream.manifest.sha256(),
        objective_weights=objective_weights,
        step_config=ettr_step_config,
    )
    schedule = ETTRJointPositionScheduler(
        ETTRJointScheduleConfig(
            args.general_position_weight,
            args.ettr_position_weight,
        )
    )
    language_loader = ShardLoader(
        general["shard_dirs"],
        seq_len=model.base.cfg.seq_len,
        batch_size=args.general_batch_size,
        rank=0,
        world=1,
        seed=args.data_seed,
        domain_weights=general["domain_weights"],
    )
    general_positions = args.general_batch_size * model.base.cfg.seq_len
    ettr_positions = (
        int(stream.release["training_rows_per_batch"])
        * CHARGED_POSITIONS_PER_ROW
    )

    args.output.mkdir(mode=0o700, parents=True)
    run_contract = {
        "architecture_seed": args.architecture_seed,
        "base_seed": args.base_seed,
        "data_seed": args.data_seed,
        "ettr_manifest_sha256": stream.manifest.sha256(),
        "ettr_positions_per_update": ettr_positions,
        "ettr_release_sha256": args.release_sha256,
        "general_contract": general["contract"],
        "general_contract_payload_sha256": general[
            "contract_payload_sha256"
        ],
        "general_corpora": general["corpora"],
        "general_inventory_sha256": general.get("inventory_sha256"),
        "general_legacy_scientific_control": general.get(
            "legacy_scientific_control",
            False,
        ),
        "general_positions_per_update": general_positions,
        "general_step_config": _dataclass_contract(
            general_step_config
        ),
        "initialization": initialization,
        "model_config": asdict(model.config),
        "objective_config": asdict(objective_config),
        "objective_weights": asdict(objective_weights),
        "optimizer_config": asdict(optimizer.config),
        "ettr_step_config": _dataclass_contract(ettr_step_config),
        "parameter_receipt": asdict(model.parameter_receipt()),
        "schedule_config": asdict(schedule.config),
        "schema": RUN_SCHEMA,
        "scientific_canary_non_resumable": True,
        "source_commit": args.source_commit,
        "source_verification": source_verification,
        "target_updates": args.updates,
        "tokenizer_sha256": stream.manifest.tokenizer_sha256,
    }
    run_contract_sha256 = _write_no_replace(
        args.output / "run-contract.json",
        _canonical_bytes(run_contract),
    )
    log_path = args.output / "train.jsonl"
    _write_no_replace(log_path, b"", mode=0o600)
    log_handle = log_path.open("ab", buffering=0)

    cursor = ETTRDistributedCursor(epoch=0, position=0)
    final_general_loss: float | None = None
    final_ettr_loss: float | None = None
    model.train()
    try:
        while optimizer.next_update < args.updates:
            selected = schedule.select(
                general_positions=general_positions,
                ettr_positions=ettr_positions,
            )
            update_seed = _seed_update(
                seed=args.data_seed,
                optimizer_step=optimizer.next_update,
                stream=selected,
            )
            if selected == "general":
                batch = language_loader.next_batch(device)
                receipt = language_step.update((batch,))
                charged = int(receipt.supervised_token_count)
                if charged != general_positions:
                    raise ETTRJointCanaryError(
                        "general charged positions differ"
                    )
                final_general_loss = float(
                    receipt.loss.detach().float().cpu()
                )
                metrics = {
                    "gradient_norm": float(
                        receipt.gradient_norm.detach().float().cpu()
                    ),
                    "language_loss": final_general_loss,
                }
            else:
                usable = cursor.validate(
                    core_batches=len(stream.records["train"]),
                    world_size=1,
                    accumulation=1,
                )
                if usable == 0:
                    raise ETTRJointCanaryError(
                        "ETTR stream has no usable training batch"
                    )
                iterator = stream.iter_positioned_batches(
                    "train",
                    rank=0,
                    world_size=1,
                    epoch=cursor.epoch,
                    seed=args.data_seed,
                    start_position=cursor.position,
                    device=device,
                )
                position, batch = next(iterator)
                if position != cursor.position:
                    raise ETTRJointCanaryError(
                        "ETTR stream position differs"
                    )
                receipt = ettr_step.update((batch,))
                cursor = cursor.advance(
                    core_batches=len(stream.records["train"]),
                    world_size=1,
                    accumulation=1,
                )
                charged = ettr_positions
                final_ettr_loss = float(
                    receipt.total_loss.detach().float().cpu()
                )
                metrics = _ettr_metric_payload(receipt)
            position_receipt = schedule.record(
                stream=selected,
                positions=charged,
            )
            if (
                receipt.optimizer_step % args.log_every == 0
                or receipt.optimizer_step == args.updates
            ):
                log_handle.write(
                    _canonical_bytes(
                        {
                            **metrics,
                            **asdict(position_receipt),
                            "learning_rate_scale": (
                                receipt.learning_rate_scale
                            ),
                            "optimizer_step": receipt.optimizer_step,
                            "schema": (
                                "shohin-ettr-joint-stream-metric-v1"
                            ),
                            "stream": selected,
                            "update_seed": update_seed,
                        }
                    )
                )
    finally:
        log_handle.close()
        packet_index.close()

    if general.get("legacy_scientific_control") is True:
        final_general = _legacy_general_resolution(
            args.legacy_general_shard_dir,
            args.legacy_general_weight,
            tokenizer_sha256=stream.manifest.tokenizer_sha256,
        )
        if (
            final_general["inventory_sha256"]
            != general["inventory_sha256"]
        ):
            raise ETTRJointCanaryError(
                "legacy general shard inventory changed during training"
            )

    model.eval()
    schedule_state = schedule.state_dict()
    full_model_path = args.output / "joint-model-final.pt"
    full_model_sha256 = _torch_save_no_replace(
        full_model_path,
        {
            "base_config": asdict(model.base.cfg),
            "ettr_config": asdict(model.config),
            "initialization": initialization,
            "model": model.state_dict(),
            "optimizer_step": optimizer.next_update,
            "run_contract_sha256": run_contract_sha256,
            "schedule": schedule_state,
            "schema": MODEL_SCHEMA,
            "source_commit": args.source_commit,
        },
    )
    base_path = args.output / "base-eval-final.pt"
    base_sha256 = _torch_save_no_replace(
        base_path,
        {
            "cfg": asdict(model.base.cfg),
            "data_seed": args.data_seed,
            "data_stream_generation": 0,
            "data_stream_seed": args.data_seed,
            "initialization": initialization,
            "model": model.base.state_dict(),
            "schema": "shohin-joint-canary-base-eval-v1",
            "step": optimizer.next_update,
        },
    )
    final_receipt = schedule.receipt
    report = {
        "base_eval_checkpoint": base_path.name,
        "base_eval_checkpoint_sha256": base_sha256,
        "ettr_fraction_observed": (
            final_receipt.ettr_positions
            / final_receipt.total_positions
        ),
        "final_ettr_loss": final_ettr_loss,
        "final_general_loss": final_general_loss,
        "full_model_checkpoint": full_model_path.name,
        "full_model_checkpoint_sha256": full_model_sha256,
        "optimizer_step": optimizer.next_update,
        "run_contract_sha256": run_contract_sha256,
        "schedule_receipt": asdict(final_receipt),
        "schema": REPORT_SCHEMA,
    }
    _write_no_replace(
        args.output / "final-report.json",
        _canonical_bytes(report),
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
