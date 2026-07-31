#!/usr/bin/env python3
"""Paired source-deleted evaluation of a three-stream ETTR candidate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import torch

from ettr_data_contract import continuation_batch_payload_sha256
from ettr_objectives import ETTRObjectiveConfig
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_train_step import ETTRCompositeTrainingSubject
from ettr_v3_streaming import (
    ETTRV3StreamingRelease,
    move_continuation_batch,
)
from eval_ettr_joint_model import (
    MODEL_SCHEMA,
    RUN_SCHEMA as PARENT_RUN_SCHEMA,
    _build_initial_model,
    _load_joint_payload,
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
from probe_ettr_causal_queries import (
    _objective_pairs_and_traces,
    _pair_rows,
    _state_summary,
    _summary,
    _trace_summary,
)


REPORT_SCHEMA = "shohin-ettr-tri-paired-development-evaluation-v2"
RUN_SCHEMA = "shohin-ettr-tri-stream-canary-v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ETTRTriEvaluationError(RuntimeError):
    """A tri-stream candidate cannot be evaluated under its parent contract."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--protected-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-run-contract", type=Path, required=True)
    parser.add_argument("--parent-run-contract-sha256", required=True)
    parser.add_argument("--parent-joint-model", type=Path, required=True)
    parser.add_argument("--parent-joint-model-sha256", required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--run-contract-sha256", required=True)
    parser.add_argument("--joint-model", type=Path, required=True)
    parser.add_argument("--joint-model-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--max-batches", type=int, default=128)
    return parser.parse_args(argv)


def _load_model_state(
    model: torch.nn.Module,
    payload: Mapping[str, object],
    *,
    label: str,
) -> None:
    try:
        incompatibility = model.load_state_dict(
            payload["model"],
            strict=True,
        )
    except (RuntimeError, TypeError) as exc:
        raise ETTRTriEvaluationError(
            f"{label} strict load differs"
        ) from exc
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ETTRTriEvaluationError(f"{label} strict load differs")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = (
        args.release_root,
        args.data_root,
        args.tokenizer,
        args.protected_checkpoint,
        args.parent_run_contract,
        args.parent_joint_model,
        args.run_contract,
        args.joint_model,
        args.output,
    )
    hashes = (
        args.release_sha256,
        args.parent_run_contract_sha256,
        args.parent_joint_model_sha256,
        args.run_contract_sha256,
        args.joint_model_sha256,
    )
    if (
        any(_HEX64.fullmatch(value) is None for value in hashes)
        or _HEX40.fullmatch(args.source_commit) is None
        or any(not path.is_absolute() for path in paths)
        or args.max_batches < 2
    ):
        raise ETTRTriEvaluationError(
            "tri-stream evaluation arguments differ"
        )
    if not torch.cuda.is_available():
        raise ETTRTriEvaluationError(
            "tri-stream evaluation requires CUDA"
        )
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    if "H100" not in torch.cuda.get_device_name(device).upper():
        raise ETTRTriEvaluationError(
            "tri-stream evaluation requires an H100"
        )

    parent_contract = _read_hash_bound_json(
        args.parent_run_contract,
        expected_sha256=args.parent_run_contract_sha256,
        label="parent run contract",
    )
    run_contract = _read_hash_bound_json(
        args.run_contract,
        expected_sha256=args.run_contract_sha256,
        label="tri run contract",
    )
    if (
        parent_contract.get("schema") != PARENT_RUN_SCHEMA
        or run_contract.get("schema") != RUN_SCHEMA
        or run_contract.get("ettr_release_sha256")
        != args.release_sha256
        or run_contract.get("parent_run_contract_sha256")
        != args.parent_run_contract_sha256
        or run_contract.get("parent_joint_model_sha256")
        != args.parent_joint_model_sha256
    ):
        raise ETTRTriEvaluationError(
            "tri-stream run lineage differs"
        )
    parent_payload = _load_joint_payload(
        args.parent_joint_model,
        expected_sha256=args.parent_joint_model_sha256,
    )
    candidate_payload = _load_joint_payload(
        args.joint_model,
        expected_sha256=args.joint_model_sha256,
    )
    parent_ettr_config = dict(parent_payload["ettr_config"])
    candidate_ettr_config = dict(candidate_payload["ettr_config"])
    parent_static_config = {
        name: value
        for name, value in parent_ettr_config.items()
        if name != "open_state_read_floor"
    }
    candidate_static_config = {
        name: value
        for name, value in candidate_ettr_config.items()
        if name != "open_state_read_floor"
    }
    if (
        parent_payload.get("schema") != MODEL_SCHEMA
        or parent_payload["run_contract_sha256"]
        != args.parent_run_contract_sha256
        or candidate_payload["run_contract_sha256"]
        != args.run_contract_sha256
        or candidate_payload["base_config"]
        != parent_payload["base_config"]
        or candidate_static_config != parent_static_config
        or candidate_ettr_config != run_contract["model_config"]
    ):
        raise ETTRTriEvaluationError(
            "tri-stream model lineage differs"
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
        run_contract=parent_contract,
        device=device,
    )
    parent_model, parent_provenance = _build_initial_model(
        args.protected_checkpoint,
        run_contract=parent_contract,
        device=device,
    )
    candidate_model, candidate_provenance = _build_initial_model(
        args.protected_checkpoint,
        run_contract=parent_contract,
        device=device,
    )
    candidate_model.set_open_state_read_floor(
        float(
            candidate_ettr_config.get(
                "open_state_read_floor",
                0.0,
            )
        )
    )
    if (
        provenance != parent_provenance
        or provenance != candidate_provenance
        or provenance.checkpoint_sha256
        != stream.manifest.protected_checkpoint_sha256
        or parent_contract.get("parameter_receipt")
        != asdict(raw_model.parameter_receipt())
        or run_contract.get("parameter_receipt")
        != asdict(raw_model.parameter_receipt())
    ):
        raise ETTRTriEvaluationError(
            "tri-stream protected model receipt differs"
        )
    _load_model_state(parent_model, parent_payload, label="parent model")
    _load_model_state(
        candidate_model,
        candidate_payload,
        label="candidate model",
    )
    raw_model.eval()
    parent_model.eval()
    candidate_model.eval()

    objective_config = ETTRObjectiveConfig(
        vocab_size=raw_model.base.cfg.vocab_size
    )
    subjects = {
        "raw": ETTRCompositeTrainingSubject(
            raw_model,
            objective_config,
            None,
            hard_transactions=True,
        ),
        "parent": ETTRCompositeTrainingSubject(
            parent_model,
            objective_config,
            None,
            hard_transactions=True,
        ),
        "candidate": ETTRCompositeTrainingSubject(
            candidate_model,
            objective_config,
            None,
            hard_transactions=True,
        ),
    }
    models = {
        "raw": raw_model,
        "parent": parent_model,
        "candidate": candidate_model,
    }
    losses: dict[str, list[dict[str, float]]] = {
        name: [] for name in subjects
    }
    counts: dict[str, list[dict[str, int]]] = {
        name: [] for name in subjects
    }
    causal_rows: dict[
        str,
        dict[str, list[dict[str, object]]],
    ] = {
        name: {"command": [], "world": []}
        for name in subjects
    }
    causal_states: dict[
        str,
        dict[str, list[dict[str, object]]],
    ] = {
        name: {"command": [], "world": []}
        for name in subjects
    }
    causal_traces: dict[
        str,
        dict[str, list[dict[str, object]]],
    ] = {
        name: {"command": [], "world": []}
        for name in subjects
    }
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
            if len(losses["candidate"]) >= args.max_batches:
                break
            packet_index.verify_validation((cpu_batch,))
            batch_sha256 = continuation_batch_payload_sha256(cpu_batch)
            batch = move_continuation_batch(cpu_batch, device)
            batch.validate(raw_model.config, objective_config)
            observed = {}
            for name, subject in subjects.items():
                arm_loss, arm_count = _evaluate(subject, batch)
                losses[name].append(arm_loss)
                counts[name].append(arm_count)
                observed[name] = arm_loss
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    pairs, state_pairs, trace_pairs = (
                        _objective_pairs_and_traces(
                            models[name],
                            batch,
                        )
                    )
                for kind in ("command", "world"):
                    causal_rows[name][kind].extend(
                        row | {"depth_bucket": "all"}
                        for row in _pair_rows(pairs[kind])
                    )
                    causal_states[name][kind].extend(
                        state_pairs[kind]
                    )
                    causal_traces[name][kind].extend(
                        trace_pairs[kind]
                    )
            batch_reports.append(
                {
                    "batch_payload_sha256": batch_sha256,
                    "losses": observed,
                    "position": position,
                }
            )
            del batch
    finally:
        packet_index.close()
    if len(losses["candidate"]) != args.max_batches:
        raise ETTRTriEvaluationError(
            "tri-stream development population is incomplete"
        )

    summaries = {
        name: {
            **_arm_summary(losses[name], counts[name]),
            "causal_geometry": {
                kind: {
                    "query": _summary(causal_rows[name][kind]),
                    "state": _state_summary(
                        causal_states[name][kind]
                    ),
                    "trace": _trace_summary(
                        causal_traces[name][kind]
                    ),
                }
                for kind in ("command", "world")
            },
            "parameter_sha256": _parameter_sha256(models[name]),
        }
        for name in subjects
    }
    parent_delta = _paired_loss_summary(
        losses["parent"],
        losses["candidate"],
    )
    raw_delta = _paired_loss_summary(
        losses["raw"],
        losses["candidate"],
    )
    parent_rates = summaries["parent"][
        "query_binding_margin_rates"
    ]
    candidate_rates = summaries["candidate"][
        "query_binding_margin_rates"
    ]

    def rate_gain(name: str) -> bool:
        parent_rate = parent_rates[name]
        candidate_rate = candidate_rates[name]
        return (
            parent_rate is not None
            and candidate_rate is not None
            and candidate_rate > parent_rate
        )

    gates = {
        "all_metrics_finite": True,
        "candidate_parameters_changed_from_parent": (
            summaries["candidate"]["parameter_sha256"]
            != summaries["parent"]["parameter_sha256"]
        ),
        "command_query_margin_rate_increased_from_parent": rate_gain(
            "command"
        ),
        "paired_total_loss_upper_95_below_parent": parent_delta["total"][
            "improved_with_upper_95_below_zero"
        ],
        "world_query_margin_rate_increased_from_parent": rate_gain(
            "world"
        ),
    }
    gates["strict_parent_improvement"] = all(gates.values())
    report = {
        "arms": summaries,
        "batches": batch_reports,
        "candidate_minus_parent": parent_delta,
        "candidate_minus_raw": raw_delta,
        "device": torch.cuda.get_device_name(device),
        "gates": gates,
        "joint_model_sha256": args.joint_model_sha256,
        "optimizer_step": candidate_payload["optimizer_step"],
        "parent_joint_model_sha256": args.parent_joint_model_sha256,
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
                "strict_parent_improvement": gates[
                    "strict_parent_improvement"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
