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
from ettr_token_transcode import (
    TokenNativeETTRTranscoder,
    receipt_value as transcode_receipt_value,
)
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
    _objective_geometry,
    _packet_geometry_summary,
    _pair_rows,
    _state_summary,
    _summary,
    _trace_summary,
    _transaction_geometry_summary,
)
from workspace_checkpoint import file_sha256


REPORT_SCHEMA = "shohin-ettr-tri-paired-development-evaluation-v3"
RUN_SCHEMA = "shohin-ettr-tri-stream-canary-v1"
COMPOSITION_KIND = "hash-bound-component-transplant"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_READOUT_GEOMETRIES = {
    "stage",
    "late",
    "postnorm",
    "postnorm-scaled",
}
_ISLAND_CONTRACT_SCHEMA = "shohin-ettr-joint-component-island-contract-v1"
_ISLAND_REPORT_SCHEMA = "shohin-ettr-joint-component-island-report-v1"


class ETTRTriEvaluationError(RuntimeError):
    """A tri-stream candidate cannot be evaluated under its parent contract."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--target-tokenizer", type=Path)
    parser.add_argument("--protected-checkpoint", type=Path, required=True)
    parser.add_argument("--external-base-root", type=Path)
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


def _validate_run_lineage(
    parent_contract: Mapping[str, object],
    run_contract: Mapping[str, object],
    *,
    release_sha256: str,
    parent_run_contract_sha256: str,
    parent_joint_model_sha256: str,
) -> Mapping[str, object] | None:
    if (
        run_contract.get("schema") != RUN_SCHEMA
        or run_contract.get("ettr_release_sha256") != release_sha256
    ):
        raise ETTRTriEvaluationError("tri-stream run lineage differs")
    composition = run_contract.get("component_composition")
    if composition is None:
        if (
            parent_contract.get("schema") != PARENT_RUN_SCHEMA
            or run_contract.get("parent_run_contract_sha256")
            != parent_run_contract_sha256
            or run_contract.get("parent_joint_model_sha256")
            != parent_joint_model_sha256
        ):
            raise ETTRTriEvaluationError("tri-stream run lineage differs")
        return None
    if not isinstance(composition, Mapping):
        raise ETTRTriEvaluationError("component composition lineage differs")
    geometry = composition.get("query_readout_geometry", "stage")
    if geometry not in _READOUT_GEOMETRIES:
        raise ETTRTriEvaluationError("component readout geometry differs")
    parent_static = dict(parent_contract)
    candidate_static = dict(run_contract)
    parent_static.pop("source_commit", None)
    candidate_static.pop("source_commit", None)
    candidate_static.pop("component_composition", None)
    parent_geometry = parent_static.pop(
        "query_readout_geometry",
        "stage",
    )
    candidate_geometry = candidate_static.pop(
        "query_readout_geometry",
        "stage",
    )
    components = composition.get("components")
    if (
        parent_contract.get("schema") != RUN_SCHEMA
        or parent_static != candidate_static
        or composition.get("kind") != COMPOSITION_KIND
        or composition.get("optimizer_updates") != 0
        or composition.get("parent_run_contract_sha256")
        != parent_run_contract_sha256
        or composition.get("parent_joint_model_sha256")
        != parent_joint_model_sha256
        or composition.get("source_commit")
        != run_contract.get("source_commit")
        or parent_geometry != "stage"
        or candidate_geometry != geometry
        or not isinstance(components, Mapping)
        or set(components) != {"compiler", "reactor", "reader"}
    ):
        raise ETTRTriEvaluationError("component composition lineage differs")
    for receipt in components.values():
        if (
            not isinstance(receipt, Mapping)
            or set(receipt) != {"path", "sha256"}
            or not isinstance(receipt["path"], str)
            or not Path(receipt["path"]).is_absolute()
            or not isinstance(receipt["sha256"], str)
            or _HEX64.fullmatch(receipt["sha256"]) is None
        ):
            raise ETTRTriEvaluationError(
                "component composition receipt differs"
            )
    reader_training = composition.get("reader_training")
    if geometry == "stage":
        if reader_training is not None:
            raise ETTRTriEvaluationError(
                "stage readout carries a training receipt"
            )
    else:
        if (
            not isinstance(reader_training, Mapping)
            or set(reader_training) != {"contract", "report"}
        ):
            raise ETTRTriEvaluationError(
                "reader training receipt differs"
            )
        receipts = {}
        for name in ("contract", "report"):
            receipt = reader_training[name]
            if (
                not isinstance(receipt, Mapping)
                or set(receipt) != {"path", "sha256"}
                or not isinstance(receipt["path"], str)
                or not Path(receipt["path"]).is_absolute()
                or not isinstance(receipt["sha256"], str)
                or _HEX64.fullmatch(receipt["sha256"]) is None
            ):
                raise ETTRTriEvaluationError(
                    "reader training receipt differs"
                )
            receipts[name] = _read_hash_bound_json(
                Path(receipt["path"]),
                expected_sha256=receipt["sha256"],
                label=f"reader island {name}",
            )
        if (
            receipts["contract"].get("schema")
            != _ISLAND_CONTRACT_SCHEMA
            or receipts["contract"].get("component") != "reader"
            or receipts["contract"].get("reader_injection") != geometry
            or receipts["contract"].get("parent_joint_model_sha256")
            != parent_joint_model_sha256
            or receipts["report"].get("schema")
            != _ISLAND_REPORT_SCHEMA
            or receipts["report"].get("component") != "reader"
            or receipts["report"].get("reader_injection") != geometry
            or receipts["report"].get("contract_sha256")
            != reader_training["contract"]["sha256"]
            or receipts["report"].get("final_component_sha256")
            != components["reader"]["sha256"]
            or receipts["report"].get("parent_joint_model_sha256")
            != parent_joint_model_sha256
        ):
            raise ETTRTriEvaluationError(
                "reader training lineage differs"
            )
    return composition


def _validate_model_lineage(
    parent_payload: Mapping[str, object],
    candidate_payload: Mapping[str, object],
    *,
    parent_run_contract_sha256: str,
    run_contract_sha256: str,
    parent_contract: Mapping[str, object],
    run_contract: Mapping[str, object],
    composition: Mapping[str, object] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    parent_ettr_config = dict(parent_payload["ettr_config"])
    candidate_ettr_config = dict(candidate_payload["ettr_config"])
    parent_geometry = parent_payload.get(
        "query_readout_geometry",
        "stage",
    )
    candidate_geometry = candidate_payload.get(
        "query_readout_geometry",
        "stage",
    )
    expected_geometry = (
        "stage"
        if composition is None
        else composition.get("query_readout_geometry", "stage")
    )
    parent_static_config = {
        name: value
        for name, value in parent_ettr_config.items()
        if name
        not in {
            "execution_trace_read_scale",
            "open_state_read_floor",
            "valid_pointer_masks",
        }
    }
    candidate_static_config = {
        name: value
        for name, value in candidate_ettr_config.items()
        if name
        not in {
            "execution_trace_read_scale",
            "open_state_read_floor",
            "valid_pointer_masks",
        }
    }
    if (
        parent_payload.get("schema") != MODEL_SCHEMA
        or candidate_payload.get("schema") != MODEL_SCHEMA
        or parent_payload.get("run_contract_sha256")
        != parent_run_contract_sha256
        or candidate_payload.get("run_contract_sha256")
        != run_contract_sha256
        or candidate_payload.get("base_config")
        != parent_payload.get("base_config")
        or candidate_payload.get("base_import")
        != parent_payload.get("base_import")
        or candidate_payload.get("base_rms_norm_eps")
        != parent_payload.get("base_rms_norm_eps")
        or candidate_static_config != parent_static_config
        or candidate_ettr_config != run_contract.get("model_config")
        or parent_geometry != "stage"
        or candidate_geometry != expected_geometry
    ):
        raise ETTRTriEvaluationError("tri-stream model lineage differs")
    if composition is not None:
        parent_state = parent_payload.get("model")
        candidate_state = candidate_payload.get("model")
        parent_initialization = parent_payload.get("initialization")
        direct_external_parent = (
            isinstance(parent_initialization, Mapping)
            and parent_initialization.get("initialization")
            == "external-smollm2-135m-control"
            and parent_initialization
            == parent_contract.get("initialization")
            and parent_payload.get("base_import")
            == parent_initialization.get("base_import")
        )
        inherited_parent = (
            isinstance(parent_initialization, Mapping)
            and parent_initialization.get("initialization")
            == "parent-joint-model"
            and parent_initialization.get("parent_joint_model_sha256")
            == parent_contract.get("parent_joint_model_sha256")
        )
        if (
            not (direct_external_parent or inherited_parent)
            or candidate_payload.get("initialization") != composition
            or candidate_payload.get("optimizer_step")
            != parent_payload.get("optimizer_step")
            or candidate_payload.get("schedule")
            != parent_payload.get("schedule")
            or not isinstance(parent_state, Mapping)
            or not isinstance(candidate_state, Mapping)
        ):
            raise ETTRTriEvaluationError(
                "component composition model lineage differs"
            )
        parent_base = {
            name: value
            for name, value in parent_state.items()
            if name.startswith("base.")
        }
        candidate_base = {
            name: value
            for name, value in candidate_state.items()
            if name.startswith("base.")
        }
        if (
            not parent_base
            or set(parent_base) != set(candidate_base)
            or any(
                not torch.equal(parent_base[name], candidate_base[name])
                for name in parent_base
            )
        ):
            raise ETTRTriEvaluationError(
                "component composition changed base weights"
            )
    return parent_ettr_config, candidate_ettr_config


def _load_initialization_contract(
    parent_contract: Mapping[str, object],
    *,
    composition: Mapping[str, object] | None,
) -> Mapping[str, object]:
    if composition is None:
        return parent_contract
    initialization = parent_contract.get("initialization")
    if (
        isinstance(initialization, Mapping)
        and initialization.get("initialization")
        == "external-smollm2-135m-control"
    ):
        return parent_contract
    parent_joint_model = parent_contract.get("parent_joint_model")
    expected_sha256 = parent_contract.get("parent_run_contract_sha256")
    if (
        not isinstance(parent_joint_model, str)
        or not Path(parent_joint_model).is_absolute()
        or not isinstance(expected_sha256, str)
        or _HEX64.fullmatch(expected_sha256) is None
    ):
        raise ETTRTriEvaluationError(
            "composition ancestor lineage differs"
        )
    ancestor_path = Path(parent_joint_model).with_name("run-contract.json")
    ancestor = _read_hash_bound_json(
        ancestor_path,
        expected_sha256=expected_sha256,
        label="composition ancestor run contract",
    )
    if ancestor.get("schema") != PARENT_RUN_SCHEMA:
        raise ETTRTriEvaluationError(
            "composition ancestor lineage differs"
        )
    return ancestor


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = [
        args.release_root,
        args.data_root,
        args.tokenizer,
        args.protected_checkpoint,
        args.parent_run_contract,
        args.parent_joint_model,
        args.run_contract,
        args.joint_model,
        args.output,
    ]
    paths.extend(
        path
        for path in (
            args.target_tokenizer,
            args.external_base_root,
        )
        if path is not None
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
    composition = _validate_run_lineage(
        parent_contract,
        run_contract,
        release_sha256=args.release_sha256,
        parent_run_contract_sha256=args.parent_run_contract_sha256,
        parent_joint_model_sha256=args.parent_joint_model_sha256,
    )
    parent_payload = _load_joint_payload(
        args.parent_joint_model,
        expected_sha256=args.parent_joint_model_sha256,
    )
    candidate_payload = _load_joint_payload(
        args.joint_model,
        expected_sha256=args.joint_model_sha256,
    )
    parent_ettr_config, candidate_ettr_config = _validate_model_lineage(
        parent_payload,
        candidate_payload,
        parent_run_contract_sha256=args.parent_run_contract_sha256,
        run_contract_sha256=args.run_contract_sha256,
        parent_contract=parent_contract,
        run_contract=run_contract,
        composition=composition,
    )
    initialization_contract = _load_initialization_contract(
        parent_contract,
        composition=composition,
    )

    stream = ETTRV3StreamingRelease(
        args.release_root,
        expected_release_sha256=args.release_sha256,
        data_root=args.data_root,
        tokenizer_path=args.tokenizer,
    )
    source_verification = stream.verify_source_shards()
    transcoder = (
        None
        if args.target_tokenizer is None
        else TokenNativeETTRTranscoder(
            args.tokenizer,
            args.target_tokenizer,
        )
    )
    transcode_receipt = (
        None
        if transcoder is None
        else transcode_receipt_value(transcoder.receipt)
    )
    external_initialization = initialization_contract.get(
        "initialization"
    )
    external_mode = (
        isinstance(external_initialization, Mapping)
        and external_initialization.get("initialization")
        == "external-smollm2-135m-control"
    )
    if external_mode:
        if (
            args.external_base_root is None
            or transcoder is None
            or transcode_receipt is None
            or initialization_contract.get("token_transcode")
            != transcode_receipt
            or external_initialization.get("token_transcode")
            != transcode_receipt
            or parent_contract.get("token_transcode")
            != transcode_receipt
            or run_contract.get("token_transcode")
            != transcode_receipt
        ):
            raise ETTRTriEvaluationError(
                "external token transcode lineage differs"
            )
    elif (
        args.external_base_root is not None
        or transcoder is not None
    ):
        raise ETTRTriEvaluationError(
            "external evaluation arguments differ"
        )
    external_tokenizer_sha256 = (
        None
        if transcoder is None
        else transcoder.receipt.target_tokenizer_sha256
    )
    raw_model, provenance = _build_initial_model(
        args.protected_checkpoint,
        run_contract=initialization_contract,
        device=device,
        external_base_root=args.external_base_root,
        external_tokenizer_sha256=external_tokenizer_sha256,
    )
    parent_model, parent_provenance = _build_initial_model(
        args.protected_checkpoint,
        run_contract=initialization_contract,
        device=device,
        external_base_root=args.external_base_root,
        external_tokenizer_sha256=external_tokenizer_sha256,
    )
    candidate_model, candidate_provenance = _build_initial_model(
        args.protected_checkpoint,
        run_contract=initialization_contract,
        device=device,
        external_base_root=args.external_base_root,
        external_tokenizer_sha256=external_tokenizer_sha256,
    )
    candidate_model.set_open_state_read_floor(
        float(
            candidate_ettr_config.get(
                "open_state_read_floor",
                0.0,
            )
        )
    )
    candidate_model.set_execution_trace_read_scale(
        float(
            candidate_ettr_config.get(
                "execution_trace_read_scale",
                0.0,
            )
        )
    )
    candidate_model.set_valid_pointer_masks(
        bool(candidate_ettr_config.get("valid_pointer_masks", False))
    )
    candidate_model.set_query_readout_geometry(
        str(
            candidate_payload.get(
                "query_readout_geometry",
                "stage",
            )
        )
    )
    protected_checkpoint_sha256 = file_sha256(
        args.protected_checkpoint
    )
    protected_provenance_sha256 = getattr(
        provenance,
        "checkpoint_sha256",
        None,
    )
    parameter_receipt = asdict(
        raw_model.parameter_receipt(enforce_cap=not external_mode)
    )
    parent_parameter_receipt = parent_contract.get(
        "parameter_receipt"
    )
    run_parameter_receipt = run_contract.get("parameter_receipt")
    if (
        provenance != parent_provenance
        or provenance != candidate_provenance
        or protected_checkpoint_sha256
        != stream.manifest.protected_checkpoint_sha256
        or (
            protected_provenance_sha256 is not None
            and protected_provenance_sha256
            != protected_checkpoint_sha256
        )
        or (
            external_mode
            and parent_parameter_receipt
            not in (None, parameter_receipt)
        )
        or (
            external_mode
            and run_parameter_receipt
            not in (None, parameter_receipt)
        )
        or (
            not external_mode
            and parent_parameter_receipt != parameter_receipt
        )
        or (
            not external_mode
            and run_parameter_receipt != parameter_receipt
        )
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
    if (
        external_mode
        and _parameter_sha256(raw_model)
        != _parameter_sha256(parent_model)
    ):
        raise ETTRTriEvaluationError(
            "external raw parent initialization differs"
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
    transaction_rows: dict[
        str,
        dict[str, list[dict[str, object]]],
    ] = {
        name: {
            "factual": [],
            "world_intervention": [],
            "command_intervention": [],
        }
        for name in subjects
    }
    packet_rows: dict[
        str,
        dict[str, list[dict[str, object]]],
    ] = {
        name: {
            "initial": [],
            "factual_terminal": [],
            "world_intervention_terminal": [],
            "command_intervention_terminal": [],
        }
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
            source_batch_sha256 = continuation_batch_payload_sha256(
                cpu_batch
            )
            if transcoder is not None:
                cpu_batch = transcoder.transcode_batch(cpu_batch)
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
                    (
                        pairs,
                        state_pairs,
                        trace_pairs,
                        transaction_geometry,
                        packet_geometry,
                    ) = _objective_geometry(
                        models[name],
                        batch,
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
                for kind, row in transaction_geometry.items():
                    transaction_rows[name][kind].append(row)
                for kind, row in packet_geometry.items():
                    packet_rows[name][kind].append(row)
            batch_report = {
                "batch_payload_sha256": batch_sha256,
                "losses": observed,
                "position": position,
            }
            if transcoder is not None:
                batch_report["source_batch_payload_sha256"] = (
                    source_batch_sha256
                )
            batch_reports.append(batch_report)
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
            "transaction_geometry": {
                kind: _transaction_geometry_summary(
                    transaction_rows[name][kind]
                )
                for kind in (
                    "factual",
                    "world_intervention",
                    "command_intervention",
                )
            },
            "packet_geometry": {
                kind: _packet_geometry_summary(
                    packet_rows[name][kind]
                )
                for kind in (
                    "initial",
                    "factual_terminal",
                    "world_intervention_terminal",
                    "command_intervention_terminal",
                )
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
        "protected_checkpoint_sha256": protected_checkpoint_sha256,
        "release_file_sha256": args.release_sha256,
        "run_contract_sha256": args.run_contract_sha256,
        "schema": REPORT_SCHEMA,
        "source_commit": args.source_commit,
        "source_verification": source_verification,
        "training_source_commit": run_contract["source_commit"],
    }
    if external_mode:
        report["base_import"] = asdict(provenance)
        report["token_transcode"] = transcode_receipt
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
