#!/usr/bin/env python3
"""Measure continuous causal-query geometry in a joint ETTR checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
from tokenizers import Tokenizer

from ettr_checkpoint import load_protected_base_model
from ettr_objectives import ETTRObjectiveConfig
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_v3_streaming import (
    ETTRV3StreamingRelease,
    move_continuation_batch,
)
from eval_ettr_joint_model import (
    ETTRJointEvaluationError,
    MODEL_SCHEMA,
    RUN_SCHEMA,
    _build_initial_model,
    _load_joint_payload,
)
from eval_ettr_v3 import (
    _HEX40,
    _HEX64,
    _parameter_sha256,
    _read_hash_bound_json,
    _sha256_file,
    _write_no_replace,
)
from probe_ettr_causal_queries import (
    _attach_decoded_predictions,
    _batch_metadata,
    _objective_pairs,
    _pair_rows,
    _retain_examples,
    _state_summary,
    _summary,
)


REPORT_SCHEMA = "shohin-ettr-joint-causal-query-probe-v1"


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
    parser.add_argument("--max-batches", type=int, default=128)
    parser.add_argument("--retain-per-depth", type=int, default=1)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
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
        or not 1 <= args.retain_per_depth <= 8
    ):
        raise ETTRJointEvaluationError(
            "joint causal-query probe arguments differ"
        )


def _arm_report(
    rows: Mapping[str, list[dict[str, object]]],
    arm: str,
) -> dict[str, object]:
    return {
        kind: {
            "query": _summary(
                [
                    row[arm]
                    | {"depth_bucket": row["depth_bucket"]}
                    for row in values
                ]
            ),
            "state": _state_summary(
                [row[f"{arm}_state"] for row in values]
            ),
        }
        for kind, values in rows.items()
    }


def _causal_shift(
    raw: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for kind in ("command", "world"):
        raw_query = raw[kind]["query"]
        candidate_query = candidate[kind]["query"]
        raw_did = raw_query["difference_in_differences"]
        candidate_did = candidate_query["difference_in_differences"]
        result[kind] = {
            "difference_in_differences_delta": {
                name: float(candidate_did[name]) - float(raw_did[name])
                for name in (
                    "maximum",
                    "mean",
                    "minimum",
                    "p05",
                    "p25",
                    "p50",
                    "p75",
                    "p95",
                )
            },
            "joint_top1_rate_delta": (
                float(candidate_query["joint_top1_rate"])
                - float(raw_query["joint_top1_rate"])
            ),
            "margin_rate_delta": {
                threshold: (
                    float(candidate_query["margin_rates"][threshold])
                    - float(raw_query["margin_rates"][threshold])
                )
                for threshold in raw_query["margin_rates"]
            },
            "paired_order_joint_rate_delta": (
                float(candidate_query["paired_order_joint_rate"])
                - float(raw_query["paired_order_joint_rate"])
            ),
        }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    if not torch.cuda.is_available():
        raise ETTRJointEvaluationError(
            "joint causal-query probe requires CUDA"
        )
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    if "H100" not in torch.cuda.get_device_name(device).upper():
        raise ETTRJointEvaluationError(
            "joint causal-query probe requires an H100"
        )

    run_contract = _read_hash_bound_json(
        args.run_contract,
        expected_sha256=args.run_contract_sha256,
        label="joint run contract",
    )
    if (
        run_contract.get("schema") != RUN_SCHEMA
        or run_contract.get("ettr_release_sha256")
        != args.release_sha256
    ):
        raise ETTRJointEvaluationError(
            "joint causal-query run contract differs"
        )
    payload = _load_joint_payload(
        args.joint_model,
        expected_sha256=args.joint_model_sha256,
    )
    if (
        payload.get("schema") != MODEL_SCHEMA
        or payload["run_contract_sha256"] != args.run_contract_sha256
        or payload["source_commit"] != run_contract.get("source_commit")
        or payload["ettr_config"] != run_contract.get("model_config")
    ):
        raise ETTRJointEvaluationError(
            "joint causal-query model contract differs"
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
        or payload["base_config"]
        != load_protected_base_model(
            args.protected_checkpoint
        )[1].base_config
    ):
        raise ETTRJointEvaluationError(
            "joint causal-query provenance differs"
        )
    try:
        incompatibility = candidate_model.load_state_dict(
            payload["model"],
            strict=True,
        )
    except (RuntimeError, TypeError) as exc:
        raise ETTRJointEvaluationError(
            "joint causal-query model strict load differs"
        ) from exc
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ETTRJointEvaluationError(
            "joint causal-query model strict load differs"
        )
    raw_model.eval()
    candidate_model.eval()

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rows: dict[str, list[dict[str, object]]] = {
        "command": [],
        "world": [],
    }
    packet_index = ETTRDiskPacketSufficiencyIndex(
        stream.packet_index_root
    )
    batches = 0
    objective_config = ETTRObjectiveConfig(
        vocab_size=raw_model.base.cfg.vocab_size
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
            if batches >= args.max_batches:
                break
            packet_index.verify_validation((cpu_batch,))
            batch = move_continuation_batch(cpu_batch, device)
            batch.validate(raw_model.config, objective_config)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                raw_pairs, raw_states = _objective_pairs(raw_model, batch)
                candidate_pairs, candidate_states = _objective_pairs(
                    candidate_model,
                    batch,
                )
            for kind in ("command", "world"):
                metadata = _batch_metadata(tokenizer, cpu_batch, kind)
                raw_values = _pair_rows(raw_pairs[kind])
                candidate_values = _pair_rows(candidate_pairs[kind])
                if not (
                    len(metadata)
                    == len(raw_values)
                    == len(candidate_values)
                    == len(raw_states[kind])
                    == len(candidate_states[kind])
                ):
                    raise ETTRJointEvaluationError(
                        "joint causal-query row population differs"
                    )
                for (
                    values,
                    raw,
                    candidate,
                    raw_state,
                    candidate_state,
                ) in zip(
                    metadata,
                    raw_values,
                    candidate_values,
                    raw_states[kind],
                    candidate_states[kind],
                    strict=True,
                ):
                    values["development_position"] = position
                    values["raw"] = raw
                    values["candidate"] = candidate
                    values["raw_state"] = raw_state
                    values["candidate_state"] = candidate_state
                    rows[kind].append(values)
            batches += 1
            del (
                batch,
                raw_pairs,
                raw_states,
                candidate_pairs,
                candidate_states,
            )
    finally:
        packet_index.close()
    if batches != args.max_batches:
        raise ETTRJointEvaluationError(
            "joint causal-query development split is too short"
        )

    retained = {
        kind: _retain_examples(
            [
                {
                    **row,
                    "checkpoint": row["candidate"],
                }
                for row in values
            ],
            retain_per_depth=args.retain_per_depth,
        )
        for kind, values in rows.items()
    }
    for values in retained.values():
        for row in values:
            row["checkpoint"] = row.pop("candidate")
            row["checkpoint_state"] = row.pop("candidate_state")
            _attach_decoded_predictions(tokenizer, row)
            row["candidate"] = row.pop("checkpoint")
            row["candidate_state"] = row.pop("checkpoint_state")

    raw_report = _arm_report(rows, "raw")
    candidate_report = _arm_report(rows, "candidate")
    report = {
        "arms": {
            "candidate": {
                **candidate_report,
                "parameter_sha256": _parameter_sha256(candidate_model),
            },
            "raw": {
                **raw_report,
                "parameter_sha256": _parameter_sha256(raw_model),
            },
        },
        "batches": batches,
        "causal_shift": _causal_shift(raw_report, candidate_report),
        "data_seed": run_contract["data_seed"],
        "device": {
            "bf16": torch.cuda.is_bf16_supported(),
            "name": torch.cuda.get_device_name(device),
        },
        "joint_model_sha256": args.joint_model_sha256,
        "optimizer_step": payload["optimizer_step"],
        "protected_checkpoint_sha256": provenance.checkpoint_sha256,
        "release_file_sha256": args.release_sha256,
        "release_manifest_sha256": stream.manifest.sha256(),
        "retained_examples": retained,
        "run_contract_sha256": args.run_contract_sha256,
        "schema": REPORT_SCHEMA,
        "source_commit": args.source_commit,
        "source_verification": source_verification,
        "split": "development",
        "tokenizer_sha256": _sha256_file(args.tokenizer),
        "training_source_commit": run_contract["source_commit"],
    }
    payload_bytes = (
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    digest = _write_no_replace(args.output, payload_bytes)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_sha256": digest,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
