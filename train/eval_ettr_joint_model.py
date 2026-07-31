#!/usr/bin/env python3
"""Paired source-deleted ETTR evaluation for a joint-stream model artifact."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import torch

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TheoryReactorConfig,
)
from ettr_checkpoint import load_protected_base_model
from ettr_data_contract import continuation_batch_payload_sha256
from ettr_objectives import ETTRObjectiveConfig
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_train_step import ETTRCompositeTrainingSubject
from ettr_v3_streaming import (
    ETTRV3StreamingRelease,
    move_continuation_batch,
)
from eval_ettr_v3 import (
    _arm_summary,
    _canonical_bytes,
    _evaluate,
    _paired_loss_summary,
    _parameter_sha256,
    _read_hash_bound_json,
    _write_no_replace,
)
from model import GPT, GPTConfig
from workspace_checkpoint import file_sha256


REPORT_SCHEMA = "shohin-ettr-joint-paired-development-evaluation-v1"
RUN_SCHEMA = "shohin-ettr-joint-stream-canary-v1"
MODEL_SCHEMA = "shohin-ettr-joint-model-canary-v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ETTRJointEvaluationError(RuntimeError):
    """A joint-stream model cannot be evaluated under its bound contract."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--protected-checkpoint", type=Path, required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--run-contract-sha256", required=True)
    parser.add_argument("--joint-model", type=Path, required=True)
    parser.add_argument("--joint-model-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--max-batches", type=int, default=64)
    return parser.parse_args(argv)


def _load_joint_payload(
    path: Path,
    *,
    expected_sha256: str,
) -> Mapping[str, object]:
    if (
        _HEX64.fullmatch(expected_sha256) is None
        or file_sha256(path) != expected_sha256
    ):
        raise ETTRJointEvaluationError(
            "joint-model checkpoint hash differs"
        )
    try:
        value = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise ETTRJointEvaluationError(
            "joint-model checkpoint is unreadable"
        ) from exc
    expected_keys = {
        "base_config",
        "ettr_config",
        "initialization",
        "model",
        "optimizer_step",
        "run_contract_sha256",
        "schedule",
        "schema",
        "source_commit",
    }
    optional_keys = {"query_readout_geometry"}
    if (
        not isinstance(value, Mapping)
        or not expected_keys <= set(value) <= expected_keys | optional_keys
        or value.get("schema") != MODEL_SCHEMA
        or value.get("run_contract_sha256") is None
        or not isinstance(value.get("model"), Mapping)
        or value.get("query_readout_geometry", "stage")
        not in {"stage", "late", "postnorm", "postnorm-scaled"}
    ):
        raise ETTRJointEvaluationError(
            "joint-model checkpoint contract differs"
        )
    return value


def _build_initial_model(
    protected_checkpoint: Path,
    *,
    run_contract: Mapping[str, object],
    device: torch.device,
) -> tuple[EndogenousTypedTheoryReactorGPT, object]:
    protected, provenance = load_protected_base_model(
        protected_checkpoint
    )
    initialization = run_contract.get("initialization")
    if not isinstance(initialization, Mapping):
        raise ETTRJointEvaluationError(
            "joint initialization receipt differs"
        )
    mode = initialization.get("initialization")
    if mode == "protected-step-300k-weights":
        base = protected
    elif mode == "deterministic-random-weights":
        base_seed = initialization.get("base_seed")
        if (
            not isinstance(base_seed, int)
            or isinstance(base_seed, bool)
            or not 0 <= base_seed < 2**63
        ):
            raise ETTRJointEvaluationError(
                "joint random base seed differs"
            )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(base_seed)
            base = GPT(GPTConfig(**provenance.base_config))
    else:
        raise ETTRJointEvaluationError(
            "joint initialization mode differs"
        )
    architecture_seed = run_contract.get("architecture_seed")
    model_config = run_contract.get("model_config")
    if (
        not isinstance(architecture_seed, int)
        or isinstance(architecture_seed, bool)
        or not isinstance(model_config, Mapping)
    ):
        raise ETTRJointEvaluationError(
            "joint architecture contract differs"
        )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(architecture_seed)
        model = EndogenousTypedTheoryReactorGPT(
            base,
            TheoryReactorConfig(**model_config),
        )
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model, provenance


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = (
        args.release_root,
        args.data_root,
        args.tokenizer,
        args.protected_checkpoint,
        args.run_contract,
        args.joint_model,
        args.output,
    )
    if (
        _HEX64.fullmatch(args.release_sha256) is None
        or _HEX64.fullmatch(args.run_contract_sha256) is None
        or _HEX64.fullmatch(args.joint_model_sha256) is None
        or _HEX40.fullmatch(args.source_commit) is None
        or any(not path.is_absolute() for path in paths)
        or args.max_batches < 2
    ):
        raise ETTRJointEvaluationError(
            "joint evaluation arguments differ"
        )
    if not torch.cuda.is_available():
        raise ETTRJointEvaluationError(
            "joint evaluation requires CUDA"
        )
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    if "H100" not in torch.cuda.get_device_name(device).upper():
        raise ETTRJointEvaluationError(
            "joint evaluation requires an H100"
        )
    run_contract = _read_hash_bound_json(
        args.run_contract,
        expected_sha256=args.run_contract_sha256,
        label="joint run contract",
    )
    if (
        run_contract.get("schema") != RUN_SCHEMA
        or run_contract.get("source_commit") is None
        or run_contract.get("ettr_release_sha256")
        != args.release_sha256
    ):
        raise ETTRJointEvaluationError(
            "joint run contract differs"
        )
    payload = _load_joint_payload(
        args.joint_model,
        expected_sha256=args.joint_model_sha256,
    )
    if (
        payload["run_contract_sha256"] != args.run_contract_sha256
        or payload["source_commit"] != run_contract["source_commit"]
        or payload["base_config"]
        != load_protected_base_model(
            args.protected_checkpoint
        )[1].base_config
        or payload["ettr_config"] != run_contract["model_config"]
    ):
        raise ETTRJointEvaluationError(
            "joint model and run contract differ"
        )

    stream = ETTRV3StreamingRelease(
        args.release_root,
        expected_release_sha256=args.release_sha256,
        data_root=args.data_root,
        tokenizer_path=args.tokenizer,
    )
    source_verification = stream.verify_source_shards()
    raw_model, provenance = _build_initial_model(
        args.protected_checkpoint,
        run_contract=run_contract,
        device=device,
    )
    candidate_model, candidate_provenance = _build_initial_model(
        args.protected_checkpoint,
        run_contract=run_contract,
        device=device,
    )
    if (
        provenance != candidate_provenance
        or provenance.checkpoint_sha256
        != stream.manifest.protected_checkpoint_sha256
        or run_contract.get("parameter_receipt")
        != asdict(raw_model.parameter_receipt())
    ):
        raise ETTRJointEvaluationError(
            "joint protected model receipt differs"
        )
    try:
        incompatibility = candidate_model.load_state_dict(
            payload["model"],
            strict=True,
        )
    except (RuntimeError, TypeError) as exc:
        raise ETTRJointEvaluationError(
            "joint model strict load differs"
        ) from exc
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ETTRJointEvaluationError(
            "joint model strict load differs"
        )
    candidate_model.eval()

    objective_config = ETTRObjectiveConfig(
        vocab_size=raw_model.base.cfg.vocab_size
    )
    raw_subject = ETTRCompositeTrainingSubject(
        raw_model,
        objective_config,
        None,
        hard_transactions=True,
    )
    candidate_subject = ETTRCompositeTrainingSubject(
        candidate_model,
        objective_config,
        None,
        hard_transactions=True,
    )
    raw_parameter_sha256 = _parameter_sha256(raw_model)
    candidate_parameter_sha256 = _parameter_sha256(candidate_model)
    raw_losses: list[dict[str, float]] = []
    raw_counts: list[dict[str, int]] = []
    candidate_losses: list[dict[str, float]] = []
    candidate_counts: list[dict[str, int]] = []
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
            seed=run_contract["data_seed"],
        )
        for position, cpu_batch in iterator:
            if len(raw_losses) >= args.max_batches:
                break
            packet_index.verify_validation((cpu_batch,))
            batch_sha256 = continuation_batch_payload_sha256(cpu_batch)
            batch = move_continuation_batch(cpu_batch, device)
            batch.validate(raw_model.config, objective_config)
            raw_loss, raw_count = _evaluate(raw_subject, batch)
            candidate_loss, candidate_count = _evaluate(
                candidate_subject,
                batch,
            )
            raw_losses.append(raw_loss)
            raw_counts.append(raw_count)
            candidate_losses.append(candidate_loss)
            candidate_counts.append(candidate_count)
            batch_reports.append(
                {
                    "batch_payload_sha256": batch_sha256,
                    "candidate_loss": candidate_loss,
                    "position": position,
                    "raw_loss": raw_loss,
                }
            )
            del batch
    finally:
        packet_index.close()
    if len(raw_losses) != args.max_batches:
        raise ETTRJointEvaluationError(
            "joint development population is incomplete"
        )

    raw_summary = _arm_summary(raw_losses, raw_counts)
    candidate_summary = _arm_summary(
        candidate_losses,
        candidate_counts,
    )
    paired = _paired_loss_summary(raw_losses, candidate_losses)
    raw_rates = raw_summary["query_binding_margin_rates"]
    candidate_rates = candidate_summary[
        "query_binding_margin_rates"
    ]

    def rate_gain(name: str) -> bool:
        raw_rate = raw_rates[name]
        candidate_rate = candidate_rates[name]
        return (
            raw_rate is not None
            and candidate_rate is not None
            and candidate_rate > raw_rate
        )

    gates = {
        "all_metrics_finite": True,
        "candidate_parameters_changed": (
            candidate_parameter_sha256 != raw_parameter_sha256
        ),
        "command_query_margin_rate_increased": rate_gain("command"),
        "paired_total_loss_upper_95_below_zero": paired["total"][
            "improved_with_upper_95_below_zero"
        ],
        "world_query_margin_rate_increased": rate_gain("world"),
    }
    gates["strict_learning_signal"] = all(gates.values())
    report = {
        "arms": {
            "candidate": {
                **candidate_summary,
                "parameter_sha256": candidate_parameter_sha256,
            },
            "raw": {
                **raw_summary,
                "parameter_sha256": raw_parameter_sha256,
            },
        },
        "batches": batch_reports,
        "device": torch.cuda.get_device_name(device),
        "gates": gates,
        "joint_model_sha256": args.joint_model_sha256,
        "optimizer_step": payload["optimizer_step"],
        "paired_candidate_minus_raw": paired,
        "protected_checkpoint_sha256": provenance.checkpoint_sha256,
        "release_file_sha256": args.release_sha256,
        "run_contract_sha256": args.run_contract_sha256,
        "schema": REPORT_SCHEMA,
        "source_commit": args.source_commit,
        "source_verification": source_verification,
        "training_source_commit": run_contract["source_commit"],
    }
    digest = _write_no_replace(
        args.output,
        _canonical_bytes(report),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_sha256": digest,
                "strict_learning_signal": gates[
                    "strict_learning_signal"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
