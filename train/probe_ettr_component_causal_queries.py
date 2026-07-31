#!/usr/bin/env python3
"""Measure continuous causal-query margins for an ETTR component assembly."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

import torch

from ettr_data_contract import (
    ETTRContinuationBatch,
    continuation_batch_payload_sha256,
)
from ettr_objectives import ETTRObjectiveConfig
from ettr_packet_index import ETTRDiskPacketSufficiencyIndex
from ettr_v3_streaming import ETTRV3StreamingRelease, move_continuation_batch
from eval_ettr_component_assembly import (
    ETTRComponentAssemblyError,
    _validate_args as _validate_assembly_args,
    load_hash_bound_component,
)
from eval_ettr_v3 import (
    _build_model,
    _canonical_bytes,
    _parameter_sha256,
    _read_hash_bound_json,
    _sha256_file,
    _validate_run_contract,
    _write_no_replace,
)
from probe_ettr_causal_queries import (
    _depth_bucket,
    _objective_pairs,
    _pair_rows,
    _state_summary,
    _summary,
)


REPORT_SCHEMA = "shohin-ettr-component-causal-query-probe-v1"
_COMPONENTS = ("compiler", "reactor", "query_reader")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--protected-checkpoint", type=Path, required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--run-contract-sha256", required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--compiler-sha256", required=True)
    parser.add_argument("--reactor", type=Path, required=True)
    parser.add_argument("--reactor-sha256", required=True)
    parser.add_argument("--query-reader", type=Path, required=True)
    parser.add_argument("--query-reader-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--architecture-seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--max-batches", type=int, default=32)
    return parser.parse_args(argv)


def _depths(
    batch: ETTRContinuationBatch,
) -> dict[str, list[str]]:
    (
        _world_packet,
        _world_command,
        world_target,
        _command_packet,
        _command_command,
        command_target,
    ) = batch.causal_rectangles.intervention_indices()
    transaction_depths = batch.transaction_targets.step_mask.sum(dim=1)
    return {
        "command": [
            _depth_bucket(int(transaction_depths[int(index)]))
            for index in command_target
        ],
        "world": [
            _depth_bucket(int(transaction_depths[int(index)]))
            for index in world_target
        ],
    }


def _component_hashes(args: argparse.Namespace) -> dict[str, str]:
    return {
        "compiler": args.compiler_sha256,
        "query_reader": args.query_reader_sha256,
        "reactor": args.reactor_sha256,
    }


def _component_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "compiler": args.compiler,
        "query_reader": args.query_reader,
        "reactor": args.reactor,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_assembly_args(args)
    if not torch.cuda.is_available():
        raise ETTRComponentAssemblyError(
            "component causal-query probe requires CUDA"
        )
    device = torch.device("cuda", 0)
    if "H100" not in torch.cuda.get_device_name(device).upper():
        raise ETTRComponentAssemblyError(
            "component causal-query probe requires H100"
        )
    if args.output.exists() or args.output.is_symlink():
        raise ETTRComponentAssemblyError(
            "component causal-query probe output already exists"
        )

    stream = ETTRV3StreamingRelease(
        args.release_root,
        expected_release_sha256=args.release_sha256,
        data_root=args.data_root,
        tokenizer_path=args.tokenizer,
    )
    source_verification = stream.verify_source_shards()
    run_contract = _read_hash_bound_json(
        args.run_contract,
        expected_sha256=args.run_contract_sha256,
        label="ETTR run contract",
    )
    model_config, _optimizer_config = _validate_run_contract(
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
    candidate_model, candidate_provenance = _build_model(
        args.protected_checkpoint,
        architecture_seed=args.architecture_seed,
        model_config=model_config,
        device=device,
    )
    if (
        raw_provenance != candidate_provenance
        or raw_provenance.checkpoint_sha256
        != stream.manifest.protected_checkpoint_sha256
        or run_contract["parameter_receipt"]
        != asdict(raw_model.parameter_receipt())
        or run_contract["parameter_receipt"]
        != asdict(candidate_model.parameter_receipt())
    ):
        raise ETTRComponentAssemblyError(
            "component causal-query provenance differs"
        )

    component_paths = _component_paths(args)
    component_hashes = _component_hashes(args)
    for name in _COMPONENTS:
        load_hash_bound_component(
            getattr(candidate_model, name),
            component_paths[name],
            expected_sha256=component_hashes[name],
            label=name.replace("_", " "),
        )
    raw_model.eval()
    candidate_model.eval()
    raw_parameter_sha256 = _parameter_sha256(raw_model)
    candidate_parameter_sha256 = _parameter_sha256(candidate_model)
    if candidate_parameter_sha256 == raw_parameter_sha256:
        raise ETTRComponentAssemblyError(
            "component causal-query parameters did not change"
        )

    objective_config = ETTRObjectiveConfig(
        vocab_size=raw_model.base.cfg.vocab_size
    )
    pair_rows: dict[str, dict[str, list[dict[str, object]]]] = {
        arm: {"command": [], "world": []}
        for arm in ("raw", "component_assembly")
    }
    state_rows: dict[str, dict[str, list[dict[str, object]]]] = {
        arm: {"command": [], "world": []}
        for arm in ("raw", "component_assembly")
    }
    batch_payload_sha256: list[str] = []
    packet_index = ETTRDiskPacketSufficiencyIndex(
        stream.packet_index_root
    )
    batches = 0
    try:
        iterator = stream.iter_positioned_batches(
            "development",
            rank=0,
            world_size=1,
            epoch=0,
            seed=args.data_seed,
        )
        for _position, cpu_batch in iterator:
            if batches >= args.max_batches:
                break
            packet_index.verify_validation((cpu_batch,))
            batch_payload_sha256.append(
                continuation_batch_payload_sha256(cpu_batch)
            )
            depth_buckets = _depths(cpu_batch)
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
                arm_values = {
                    "raw": _pair_rows(raw_pairs[kind]),
                    "component_assembly": _pair_rows(
                        candidate_pairs[kind]
                    ),
                }
                for arm, values in arm_values.items():
                    if len(values) != len(depth_buckets[kind]):
                        raise ETTRComponentAssemblyError(
                            "component causal-query row population differs"
                        )
                    pair_rows[arm][kind].extend(
                        value | {"depth_bucket": depth}
                        for value, depth in zip(
                            values,
                            depth_buckets[kind],
                            strict=True,
                        )
                    )
                state_rows["raw"][kind].extend(raw_states[kind])
                state_rows["component_assembly"][kind].extend(
                    candidate_states[kind]
                )
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
        raise ETTRComponentAssemblyError(
            "component causal-query development split is too short"
        )

    report = {
        "architecture_seed": args.architecture_seed,
        "arms": {
            arm: {
                kind: {
                    "query": _summary(pair_rows[arm][kind]),
                    "state": _state_summary(state_rows[arm][kind]),
                }
                for kind in ("command", "world")
            }
            for arm in ("raw", "component_assembly")
        },
        "batch_payload_sha256": batch_payload_sha256,
        "batches": batches,
        "component_sha256": component_hashes,
        "data_seed": args.data_seed,
        "device": {
            "bf16": torch.cuda.is_bf16_supported(),
            "name": torch.cuda.get_device_name(device),
        },
        "parameter_sha256": {
            "component_assembly": candidate_parameter_sha256,
            "raw": raw_parameter_sha256,
        },
        "probe_source_sha256": _sha256_file(Path(__file__).resolve()),
        "protected_checkpoint_sha256": raw_provenance.checkpoint_sha256,
        "release_file_sha256": args.release_sha256,
        "release_manifest_sha256": stream.manifest.sha256(),
        "run_contract_sha256": args.run_contract_sha256,
        "schema": REPORT_SCHEMA,
        "source_commit": args.source_commit,
        "source_verification": source_verification,
        "split": "development",
        "tokenizer_sha256": _sha256_file(args.tokenizer),
    }
    digest = _write_no_replace(args.output, _canonical_bytes(report))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
