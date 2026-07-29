#!/usr/bin/env python3
"""Evaluate ETTR learning on one immutable development stream.

The evaluator always scores the deterministic raw architecture initialization.
When a trained checkpoint is supplied, it reconstructs the exact training
contract, restores the checkpoint through the production continuation loader,
and scores both arms on the same development batches.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import statistics
from typing import Mapping, Sequence

import torch

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TheoryReactorConfig,
)
from ettr_checkpoint import (
    BaseProvenance,
    load_ettr_checkpoint,
    load_protected_base_model,
)
from ettr_data_contract import continuation_batch_payload_sha256
from ettr_objectives import (
    ETTRCompositeLoss,
    ETTRObjectiveConfig,
    ETTRObjectiveReceipt,
)
from ettr_optimization import ETTROptimizerBundle, ETTROptimizerConfig
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_train_step import ETTRCompositeTrainingSubject
from ettr_v3_streaming import (
    ETTRV3StreamingRelease,
    move_continuation_batch,
)


REPORT_SCHEMA = "shohin-ettr-il-v3-paired-development-evaluation-v1"
RUN_SCHEMA = "shohin-ettr-il-v3-distributed-run-v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_LOSS_FIELDS = (
    "total",
    "token_lm",
    "packet",
    "world_intervention",
    "command_intervention",
    "world_query_binding",
    "command_query_binding",
    "transaction",
    "equivariance",
    "commit_halt",
    "sparsity",
    "anti_bypass",
)
_COUNT_FIELDS = tuple(
    field.name
    for field in fields(ETTRObjectiveReceipt)
    if field.name
    not in {
        "schema",
        "batch_size",
        "sequence_tokens",
        "equivariance_pairs",
        "causal_lm_shift",
        "weights",
    }
)
_RUN_KEYS = {
    "accumulation",
    "architecture_seed",
    "compile_backend",
    "compile_mode",
    "data_seed",
    "freeze_base",
    "model_config",
    "optimizer_config",
    "parameter_receipt",
    "release_file_sha256",
    "release_source_commit",
    "resume_checkpoint_sha256",
    "schema",
    "source_commit",
    "start_optimizer_step",
    "target_optimizer_step",
    "world_size",
}


class ETTRV3EvaluationError(RuntimeError):
    """The development evaluation cannot preserve its frozen contract."""


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _physical_file(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ETTRV3EvaluationError(f"{label} cannot be inspected") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise ETTRV3EvaluationError(
            f"{label} is not a physical single-link regular file"
        )
    return metadata


def _read_hash_bound_json(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> Mapping[str, object]:
    if _HEX64.fullmatch(expected_sha256) is None:
        raise ETTRV3EvaluationError(f"{label} SHA-256 differs")
    before = _physical_file(path, label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size != before.st_size
        ):
            raise ETTRV3EvaluationError(f"{label} identity changed")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            payload = source.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        after.st_dev != opened.st_dev
        or after.st_ino != opened.st_ino
        or after.st_size != opened.st_size
        or after.st_mtime_ns != opened.st_mtime_ns
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise ETTRV3EvaluationError(f"{label} changed or hash differs")
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ETTRV3EvaluationError(f"{label} is malformed") from exc
    if not isinstance(value, Mapping) or payload != _canonical_bytes(value):
        raise ETTRV3EvaluationError(f"{label} is not canonical")
    return value


def _write_no_replace(path: Path, payload: bytes) -> str:
    if not path.is_absolute():
        raise ETTRV3EvaluationError("evaluation output must be absolute")
    if not path.parent.is_dir() or path.exists() or path.is_symlink():
        raise ETTRV3EvaluationError("evaluation output destination differs")
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


def _optimizer_config(value: object) -> ETTROptimizerConfig:
    if not isinstance(value, Mapping):
        raise ETTRV3EvaluationError("run optimizer configuration differs")
    expected = {field.name for field in fields(ETTROptimizerConfig)}
    if set(value) != expected:
        raise ETTRV3EvaluationError("run optimizer fields differ")
    payload = dict(value)
    betas = payload.get("adam_betas")
    if (
        not isinstance(betas, list)
        or len(betas) != 2
        or any(not isinstance(item, float) for item in betas)
    ):
        raise ETTRV3EvaluationError("run Adam beta configuration differs")
    payload["adam_betas"] = tuple(betas)
    try:
        config = ETTROptimizerConfig(**payload)
        config.validate()
    except (TypeError, ValueError) as exc:
        raise ETTRV3EvaluationError(
            "run optimizer configuration is invalid"
        ) from exc
    return config


def _validate_run_contract(
    value: Mapping[str, object],
    *,
    release_sha256: str,
    release_source_commit: str,
    architecture_seed: int,
) -> tuple[TheoryReactorConfig, ETTROptimizerConfig]:
    if set(value) != _RUN_KEYS or value.get("schema") != RUN_SCHEMA:
        raise ETTRV3EvaluationError("ETTR run contract schema differs")
    if (
        value.get("release_file_sha256") != release_sha256
        or value.get("release_source_commit") != release_source_commit
        or value.get("architecture_seed") != architecture_seed
        or _HEX40.fullmatch(str(value.get("source_commit"))) is None
        or _HEX40.fullmatch(str(value.get("release_source_commit"))) is None
        or type(value.get("freeze_base")) is not bool
        or type(value.get("accumulation")) is not int
        or value["accumulation"] < 1
        or type(value.get("data_seed")) is not int
        or not 0 <= value["data_seed"] < 2**63
        or type(value.get("world_size")) is not int
        or value["world_size"] < 1
        or value.get("compile_backend") not in {None, "inductor"}
        or (
            value.get("compile_backend") is None
            and value.get("compile_mode") is not None
        )
        or (
            value.get("compile_mode")
            not in {
                None,
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            }
        )
        or (
            value.get("resume_checkpoint_sha256") is not None
            and _HEX64.fullmatch(
                str(value.get("resume_checkpoint_sha256"))
            )
            is None
        )
        or not isinstance(value.get("parameter_receipt"), Mapping)
        or type(value.get("start_optimizer_step")) is not int
        or type(value.get("target_optimizer_step")) is not int
        or value["start_optimizer_step"] < 0
        or value["target_optimizer_step"] <= value["start_optimizer_step"]
    ):
        raise ETTRV3EvaluationError("ETTR run contract identity differs")
    model_value = value.get("model_config")
    if not isinstance(model_value, Mapping):
        raise ETTRV3EvaluationError("ETTR run model configuration differs")
    try:
        model_config = TheoryReactorConfig(**model_value)
        model_config.validate()
    except (TypeError, ValueError) as exc:
        raise ETTRV3EvaluationError(
            "ETTR run model configuration is invalid"
        ) from exc
    optimizer_config = _optimizer_config(value.get("optimizer_config"))
    if optimizer_config.train_base == value["freeze_base"]:
        raise ETTRV3EvaluationError("ETTR run base-training contract differs")
    return model_config, optimizer_config


def _validate_checkpoint_cursor(
    progress: object,
    data_stream: object,
    *,
    run_contract: Mapping[str, object],
    stream: ETTRV3StreamingRelease,
    release_sha256: str,
    protected_step: int,
) -> None:
    sampler = getattr(data_stream, "sampler_state", None)
    if not isinstance(sampler, Mapping):
        raise ETTRV3EvaluationError("checkpoint sampler state differs")
    expected_sampler = {
        "accumulation": run_contract["accumulation"],
        "compile_backend": run_contract["compile_backend"],
        "compile_mode": run_contract["compile_mode"],
        "consumed_stream_batches": (
            progress.optimizer_step
            * run_contract["world_size"]
            * run_contract["accumulation"]
        ),
        "release_file_sha256": release_sha256,
        "schema": "shohin-ettr-il-v3-distributed-cursor-v1",
        "world_size": run_contract["world_size"],
    }
    if (
        progress.global_step != protected_step + progress.optimizer_step
        or progress.gradient_accumulation_steps
        != run_contract["accumulation"]
        or data_stream.manifest_sha256 != stream.manifest.sha256()
        or data_stream.dataset_sha256 != stream.manifest.dataset_sha256
        or data_stream.seed != run_contract["data_seed"]
        or dict(sampler) != expected_sampler
    ):
        raise ETTRV3EvaluationError(
            "checkpoint cursor differs from its exact run contract"
        )


def _build_model(
    protected_checkpoint: Path,
    *,
    architecture_seed: int,
    model_config: TheoryReactorConfig,
    device: torch.device,
) -> tuple[EndogenousTypedTheoryReactorGPT, BaseProvenance]:
    base, provenance = load_protected_base_model(protected_checkpoint)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(architecture_seed)
        model = EndogenousTypedTheoryReactorGPT(base, model_config)
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model, provenance


def _parameter_sha256(model: EndogenousTypedTheoryReactorGPT) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().contiguous()
        descriptor = {
            "dtype": str(value.dtype),
            "name": name,
            "shape": list(value.shape),
        }
        digest.update(_canonical_bytes(descriptor))
        digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def _loss_values(loss: ETTRCompositeLoss) -> dict[str, float]:
    values = {
        name: float(getattr(loss, name).detach().float().cpu())
        for name in _LOSS_FIELDS
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise ETTRV3EvaluationError("development loss is non-finite")
    return values


def _count_values(receipt: ETTRObjectiveReceipt) -> dict[str, int]:
    values = {
        name: int(getattr(receipt, name).detach().cpu())
        for name in _COUNT_FIELDS
    }
    if any(value < 0 for value in values.values()):
        raise ETTRV3EvaluationError("development receipt count is negative")
    return values


def _arm_summary(
    losses: Sequence[Mapping[str, float]],
    counts: Sequence[Mapping[str, int]],
) -> dict[str, object]:
    if not losses or len(losses) != len(counts):
        raise ETTRV3EvaluationError("development arm population differs")
    loss_means = {
        name: statistics.fmean(value[name] for value in losses)
        for name in _LOSS_FIELDS
    }
    count_totals = {
        name: sum(value[name] for value in counts)
        for name in _COUNT_FIELDS
    }

    def rate(numerator: str, denominator: str) -> float | None:
        support = count_totals[denominator]
        return None if support == 0 else count_totals[numerator] / support

    return {
        "batches": len(losses),
        "count_totals": count_totals,
        "loss_means": loss_means,
        "query_binding_margin_rates": {
            "command": rate(
                "command_query_margin_satisfied",
                "command_query_contrast_pairs",
            ),
            "world": rate(
                "world_query_margin_satisfied",
                "world_query_contrast_pairs",
            ),
        },
    }


def _paired_loss_summary(
    baseline: Sequence[Mapping[str, float]],
    checkpoint: Sequence[Mapping[str, float]],
) -> dict[str, object]:
    if not baseline or len(baseline) != len(checkpoint):
        raise ETTRV3EvaluationError("paired development population differs")
    result: dict[str, object] = {}
    for name in _LOSS_FIELDS:
        deltas = [
            candidate[name] - raw[name]
            for raw, candidate in zip(baseline, checkpoint, strict=True)
        ]
        mean = statistics.fmean(deltas)
        standard_error = (
            statistics.stdev(deltas) / math.sqrt(len(deltas))
            if len(deltas) > 1
            else 0.0
        )
        half_width = 1.96 * standard_error
        result[name] = {
            "checkpoint_minus_raw_mean": mean,
            "confidence_95": [mean - half_width, mean + half_width],
            "improved_with_upper_95_below_zero": mean + half_width < 0.0,
            "standard_error": standard_error,
            "win_fraction": sum(delta < 0.0 for delta in deltas) / len(deltas),
        }
    return result


def _evaluate(
    subject: ETTRCompositeTrainingSubject,
    batch: object,
) -> tuple[dict[str, float], dict[str, int]]:
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        loss = subject.objective_loss(batch)
    return _loss_values(loss), _count_values(loss.receipt)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--protected-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--architecture-seed", type=int, default=2026072801)
    parser.add_argument("--data-seed", type=int, default=2026072802)
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--run-contract", type=Path)
    parser.add_argument("--run-contract-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    checkpoint_values = (
        args.checkpoint,
        args.checkpoint_sha256,
        args.run_contract,
        args.run_contract_sha256,
    )
    if (
        _HEX64.fullmatch(args.release_sha256) is None
        or _HEX40.fullmatch(args.source_commit) is None
        or not 0 <= args.architecture_seed < 2**63
        or not 0 <= args.data_seed < 2**63
        or args.max_batches < 2
        or (any(value is not None for value in checkpoint_values)
            and not all(value is not None for value in checkpoint_values))
    ):
        raise ETTRV3EvaluationError("ETTR development arguments differ")
    if not torch.cuda.is_available():
        raise ETTRV3EvaluationError("ETTR development evaluation requires CUDA")
    device = torch.device("cuda", 0)
    if "H100" not in torch.cuda.get_device_name(device).upper():
        raise ETTRV3EvaluationError("ETTR development evaluation requires H100")

    stream = ETTRV3StreamingRelease(
        args.release_root,
        expected_release_sha256=args.release_sha256,
        data_root=args.data_root,
        tokenizer_path=args.tokenizer,
    )
    source_verification = stream.verify_source_shards()
    model_config = TheoryReactorConfig()
    optimizer_config = None
    run_contract = None
    if args.run_contract is not None:
        run_contract = _read_hash_bound_json(
            args.run_contract,
            expected_sha256=args.run_contract_sha256,
            label="ETTR run contract",
        )
        model_config, optimizer_config = _validate_run_contract(
            run_contract,
            release_sha256=args.release_sha256,
            release_source_commit=stream.release["source_commit"],
            architecture_seed=args.architecture_seed,
        )

    raw_model, raw_provenance = _build_model(
        args.protected_checkpoint,
        architecture_seed=args.architecture_seed,
        model_config=model_config,
        device=device,
    )
    if (
        raw_provenance.checkpoint_sha256
        != stream.manifest.protected_checkpoint_sha256
    ):
        raise ETTRV3EvaluationError(
            "protected checkpoint differs from the ETTR release"
        )
    objective_config = ETTRObjectiveConfig(
        vocab_size=raw_model.base.cfg.vocab_size
    )
    raw_subject = ETTRCompositeTrainingSubject(
        raw_model,
        objective_config,
        None,
        hard_transactions=True,
    )
    if (
        run_contract is not None
        and run_contract["parameter_receipt"]
        != asdict(raw_model.parameter_receipt())
    ):
        raise ETTRV3EvaluationError(
            "run parameter receipt differs from the reconstructed model"
        )
    raw_parameter_sha256 = _parameter_sha256(raw_model)

    checkpoint_model = None
    checkpoint_subject = None
    checkpoint_progress = None
    checkpoint_parameter_sha256 = None
    if args.checkpoint is not None:
        assert optimizer_config is not None and run_contract is not None
        checkpoint_model, checkpoint_provenance = _build_model(
            args.protected_checkpoint,
            architecture_seed=args.architecture_seed,
            model_config=model_config,
            device=device,
        )
        optimizer = ETTROptimizerBundle(checkpoint_model, optimizer_config)
        resumed = load_ettr_checkpoint(
            args.checkpoint,
            expected_sha256=args.checkpoint_sha256,
            model=checkpoint_model,
            protected_base=checkpoint_provenance,
            optimizer=optimizer,
            scheduler=None,
        )
        if not (
            run_contract["start_optimizer_step"]
            <= resumed.progress.optimizer_step
            <= run_contract["target_optimizer_step"]
        ):
            raise ETTRV3EvaluationError(
                "checkpoint lies outside its exact run contract"
            )
        _validate_checkpoint_cursor(
            resumed.progress,
            resumed.data_stream,
            run_contract=run_contract,
            stream=stream,
            release_sha256=args.release_sha256,
            protected_step=checkpoint_provenance.step,
        )
        checkpoint_model.eval()
        checkpoint_subject = ETTRCompositeTrainingSubject(
            checkpoint_model,
            objective_config,
            None,
            hard_transactions=True,
        )
        checkpoint_progress = asdict(resumed.progress)
        checkpoint_parameter_sha256 = _parameter_sha256(checkpoint_model)
        del optimizer
        torch.cuda.empty_cache()

    raw_losses: list[dict[str, float]] = []
    raw_counts: list[dict[str, int]] = []
    checkpoint_losses: list[dict[str, float]] = []
    checkpoint_counts: list[dict[str, int]] = []
    batch_reports = []
    packet_index = ETTRDiskPacketSufficiencyIndex(
        stream.packet_index_root
    )
    try:
        iterator = stream.iter_positioned_batches(
            "development",
            rank=0,
            world_size=1,
            epoch=0,
            seed=args.data_seed,
        )
        for position, cpu_batch in iterator:
            if len(raw_losses) >= args.max_batches:
                break
            packet_index.verify_validation((cpu_batch,))
            batch_sha256 = continuation_batch_payload_sha256(cpu_batch)
            batch = move_continuation_batch(cpu_batch, device)
            batch.validate(raw_model.config, objective_config)
            raw_loss, raw_count = _evaluate(raw_subject, batch)
            raw_losses.append(raw_loss)
            raw_counts.append(raw_count)
            arm_values: dict[str, object] = {"raw": raw_loss}
            if checkpoint_subject is not None:
                checkpoint_loss, checkpoint_count = _evaluate(
                    checkpoint_subject,
                    batch,
                )
                checkpoint_losses.append(checkpoint_loss)
                checkpoint_counts.append(checkpoint_count)
                arm_values["checkpoint"] = checkpoint_loss
            batch_reports.append(
                {
                    "batch_payload_sha256": batch_sha256,
                    "losses": arm_values,
                    "position": position,
                }
            )
            del batch
    finally:
        packet_index.close()
    if len(raw_losses) != args.max_batches:
        raise ETTRV3EvaluationError(
            "development split contains fewer batches than requested"
        )

    arms: dict[str, object] = {
        "raw": {
            **_arm_summary(raw_losses, raw_counts),
            "parameter_sha256": raw_parameter_sha256,
        }
    }
    paired = None
    gates = None
    if checkpoint_subject is not None:
        checkpoint_summary = _arm_summary(
            checkpoint_losses,
            checkpoint_counts,
        )
        arms["checkpoint"] = {
            **checkpoint_summary,
            "checkpoint_progress": checkpoint_progress,
            "checkpoint_sha256": args.checkpoint_sha256,
            "parameter_sha256": checkpoint_parameter_sha256,
            "run_contract_sha256": args.run_contract_sha256,
        }
        paired = _paired_loss_summary(raw_losses, checkpoint_losses)
        raw_rates = arms["raw"]["query_binding_margin_rates"]
        checkpoint_rates = arms["checkpoint"]["query_binding_margin_rates"]

        def rate_gain(name: str) -> bool:
            raw_rate = raw_rates[name]
            trained_rate = checkpoint_rates[name]
            return (
                raw_rate is not None
                and trained_rate is not None
                and trained_rate > raw_rate
            )

        gates = {
            "all_metrics_finite": True,
            "checkpoint_parameters_changed": (
                checkpoint_parameter_sha256 != raw_parameter_sha256
            ),
            "command_query_margin_rate_increased": rate_gain("command"),
            "paired_total_loss_upper_95_below_zero": paired["total"][
                "improved_with_upper_95_below_zero"
            ],
            "world_query_margin_rate_increased": rate_gain("world"),
        }
        gates["strict_learning_signal"] = all(gates.values())

    report = {
        "architecture_seed": args.architecture_seed,
        "arms": arms,
        "batches": batch_reports,
        "data_seed": args.data_seed,
        "device": {
            "bf16": torch.cuda.is_bf16_supported(),
            "name": torch.cuda.get_device_name(device),
        },
        "gates": gates,
        "paired_checkpoint_minus_raw": paired,
        "protected_checkpoint_sha256": raw_provenance.checkpoint_sha256,
        "release_file_sha256": args.release_sha256,
        "release_manifest_sha256": stream.manifest.sha256(),
        "schema": REPORT_SCHEMA,
        "source_commit": args.source_commit,
        "source_verification": source_verification,
        "split": "development",
    }
    payload = _canonical_bytes(report)
    digest = _write_no_replace(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_sha256": digest,
                "strict_learning_signal": (
                    None if gates is None else gates["strict_learning_signal"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
